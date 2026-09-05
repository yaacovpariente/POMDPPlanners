# SPDX-License-Identifier: MIT

"""Pinned QA inputs for the Battleship POMDP.

An environment nobody has solved is indistinguishable from a broken one, so the
Battleship QA pass measures a planner against a baseline. Both sides of that
comparison are *test inputs*, not measurements, and neither is reproducible
without the other: a completion rate quoted without the planner config and
without the baseline it beat is a number nobody can check.

So this module holds three things and no results:

* :func:`battleship_pft_dpw_config` — the PFT-DPW settings the QA pass used.
* :class:`BattleshipRandomProber` — the baseline it was measured against.
* :data:`BATTLESHIP_QA_NUM_STEPS` — the probe horizon, which decides how hard
  the baseline is and therefore what "beating it" is worth.
* :class:`BattleshipFixedBoardWorld` — the pinned-board world environment that
  makes the two arms of the comparison run on the *same* hidden layouts.

Measured values deliberately live nowhere in the repository. A committed
completion rate goes stale silently and starts lying; the tests recompute it.
"""

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.battleship_pomdp import BattleshipPOMDP


#: Probes allowed per episode.
#:
#: Fixed at 18 for a 5x5 board with seven ship cells, and fixed *before* any
#: measurement. It is the number that decides whether the baseline is a real
#: opponent. A no-replacement random prober needs the last of the seven ship
#: cells to appear within its first 18 draws of 25, which happens with
#: probability C(18,7)/C(25,7) = 31824/480700 ~= 6.6%. Raise it towards 25 and
#: random clears the board almost always, so beating it means nothing; drop it
#: much below 14 and neither side can finish, so the comparison collapses onto
#: return alone.
BATTLESHIP_QA_NUM_STEPS = 18

#: Particles in the QA belief.
#:
#: BattleshipBelief redraws its particles from the exact posterior on every
#: update, so this sets sampling noise inside the search rather than how long
#: the belief survives — it cannot deplete. 200 matches the VOPP QA starting
#: config's belief size.
BATTLESHIP_QA_BELIEF_PARTICLES = 200


def battleship_pft_dpw_config(time_out_in_seconds: float = 0.75) -> Dict[str, Any]:
    """PFT-DPW settings used for the Battleship QA pass.

    Battleship has no torch vectorized model, so QA uses PFT-DPW on the scalar
    Environment API rather than VOPP.

    The widening constants are deliberately set to *disable* widening on both
    axes. Progressive widening exists to keep a large or continuous branching
    factor manageable, and Battleship has neither: 25 probe actions and exactly
    two possible readings per probe. With ``alpha = 0`` the caps are constant,
    ``k_a = 25`` admits every cell and ``k_o = 2`` admits both readings, so the
    tree is a plain full-width search and no cell is excluded by an accident of
    the sampler.

    Args:
        time_out_in_seconds: Wall-clock budget per decision. PFT-DPW takes a
            timeout directly, so it hits the budget exactly instead of being
            calibrated towards it. Defaults to 0.75.

    Returns:
        Keyword arguments for :class:`PFT_DPW`, minus ``environment``,
        ``discount_factor``, ``name`` and ``action_sampler``, which the caller
        supplies.
    """
    return {
        "depth": 10,
        "k_a": 25.0,
        "alpha_a": 0.0,
        "k_o": 2.0,
        "alpha_o": 0.0,
        "exploration_constant": 2.0,
        "time_out_in_seconds": time_out_in_seconds,
    }


class BattleshipRandomProber(Policy):
    """Uniform random probing of cells not yet probed.

    The baseline has to be the strongest *uninformed* policy, not the weakest
    one available. A policy drawing uniformly from all 25 cells would waste most
    of its turns re-probing, and beating it would prove nothing about the
    planner: repeat probes are strictly dominated and avoiding them needs no
    belief at all. Drawing without replacement removes that free win, so
    whatever margin the planner shows is a margin from reasoning about *where*
    the ships are.

    It reads which cells are unprobed from the belief's own particles, which all
    share one probe half. That is information the agent genuinely has — it is
    its own action history. The hidden occupancy half is never touched.

    Attributes:
        environment: The Battleship environment.
    """

    def __init__(
        self,
        environment: Any,
        discount_factor: float,
        name: str = "RandomProber",
        log_path: Optional[Any] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the baseline.

        Args:
            environment: The Battleship environment.
            discount_factor: Discount factor, matched to the environment's.
            name: Policy name. Defaults to ``"RandomProber"``.
            log_path: Optional log directory.
            debug: Enable debug logging.
            use_queue_logger: Whether to use queue-based logging.
        """
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            name=name,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Discrete actions, discrete observations."""
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        """No per-decision diagnostics; the baseline has nothing to report."""
        return []

    def action(self, belief: Any) -> Tuple[List[Any], PolicyRunData]:
        """Pick one unprobed cell uniformly at random.

        Args:
            belief: The current belief. Only its probe half is read.

        Returns:
            A one-action list and empty run data. When every cell has been
            probed — which the horizon makes unreachable, but which a longer run
            could hit — it falls back to a uniform draw over all cells rather
            than failing.
        """
        num_cells = int(self.environment.num_cells)  # type: ignore[attr-defined]
        particle = np.asarray(belief.particles[0], dtype=np.float64)
        unprobed = np.flatnonzero(particle[num_cells:] <= 0.5)
        if unprobed.size == 0:
            return [random.randrange(num_cells)], PolicyRunData(info_variables=[])
        return [int(random.choice(unprobed.tolist()))], PolicyRunData(info_variables=[])


#: Board identifiers used by the paired QA pass.
#:
#: Twenty boards, fixed here rather than in the driver so the board set cannot
#: drift between runs of the comparison.
BATTLESHIP_QA_BOARD_IDS = tuple(range(20))


def battleship_qa_layout_index(board_id: int, num_layouts: int) -> int:
    """Pick the hidden layout for one QA board.

    Deterministic in ``board_id`` alone, and that is the whole point: the
    simulator seeds each episode from ``md5(env_name, policy_name, episode_id)``,
    so anything that consults the global RNG hands a different board to each
    policy and the comparison stops being paired.

    Args:
        board_id: The board identifier, ``0..19`` for the QA set.
        num_layouts: Number of enumerated legal layouts to choose from.

    Returns:
        A row index into ``FleetLayoutTable.masks``.
    """
    return int(np.random.default_rng(int(board_id)).integers(0, int(num_layouts)))


class _PointMassInitialState(Distribution):
    """A degenerate initial-state distribution supported on exactly one board."""

    def __init__(self, state: np.ndarray):
        """Store the single supported state.

        Args:
            state: The pinned initial state array.
        """
        self.state = np.asarray(state, dtype=np.float64)

    def sample(self, n_samples: int = 1) -> List[Any]:
        """Return ``n_samples`` copies of the pinned state.

        Args:
            n_samples: How many copies to return. Defaults to 1.

        Returns:
            A list of independent copies, so a caller mutating one draw cannot
            corrupt the distribution or any later episode.
        """
        return [self.state.copy() for _ in range(int(n_samples))]

    def probability(self, values: List[Any]) -> np.ndarray:
        """One for the pinned state, zero for anything else.

        Args:
            values: Candidate states.

        Returns:
            ``float64`` array of probabilities.
        """
        return np.array(
            [
                1.0 if np.array_equal(np.asarray(v, dtype=np.float64), self.state) else 0.0
                for v in values
            ],
            dtype=np.float64,
        )


class BattleshipFixedBoardWorld(BattleshipPOMDP):
    """Battleship with the hidden fleet pinned to one layout.

    Used only as the *world* side of a paired evaluation. Dynamics, reward,
    observation model, terminality and metrics are all inherited unchanged; the
    single difference is that :meth:`initial_state_dist` is a point mass rather
    than the uniform prior.

    Why this and not a seeded belief: the episode runner takes the true start
    from ``initial_belief.sample()`` only when the world and the planner's model
    are the same object, and PFT-DPW also calls ``belief.sample()`` inside its
    own search. Pinning the belief would therefore hand the planner the answer.
    Giving the runner a *different* world object routes the true start through
    ``environment.initial_state_dist()``, which the planner never touches, so
    the fleet stays hidden while both arms get the same board.

    ``board_id`` and ``layout_index`` are public attributes, so they reach
    ``config_id`` and every board gets its own cache entry instead of colliding
    with its neighbours.

    Attributes:
        board_id: The QA board identifier this world was built for.
        layout_index: Row of the layout table the fleet occupies.
    """

    def __init__(
        self,
        board_id: int,
        board_size: int = 5,
        ship_lengths: Sequence[int] = (3, 2, 2),
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Build the world for one pinned board.

        Args:
            board_id: Board identifier; selects the layout deterministically.
            board_size: Side length of the square board. Defaults to 5.
            ship_lengths: Ship lengths. Defaults to ``(3, 2, 2)``.
            name: Environment name. Defaults to ``"Battleship_board<NN>"``. Each
                board needs its own name because the simulation API refuses two
                runs that share an environment name.
            **kwargs: Forwarded to :class:`BattleshipPOMDP`.
        """
        super().__init__(
            board_size=board_size,
            ship_lengths=ship_lengths,
            name=name if name is not None else f"Battleship_board{int(board_id):02d}",
            **kwargs,
        )
        self.board_id = int(board_id)
        self.layout_index = battleship_qa_layout_index(self.board_id, self.layouts.num_layouts)

    @property
    def hidden_occupancy(self) -> np.ndarray:
        """The pinned fleet as a ``uint8`` occupancy row."""
        return np.asarray(self.layouts.masks[self.layout_index], dtype=np.uint8)

    def initial_state_dist(self) -> Distribution:
        """A point mass on the pinned board, nothing probed."""
        state = np.zeros(2 * self.num_cells, dtype=np.float64)
        state[: self.num_cells] = self.hidden_occupancy
        return _PointMassInitialState(state)
