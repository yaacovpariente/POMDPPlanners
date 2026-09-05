# SPDX-License-Identifier: MIT

"""Module for the Battleship POMDP environment.

Battleship as an *exploration* problem. A fleet of straight ships is placed at
random on a square board and never moves. The agent cannot see the board; it
probes one cell per step and is told, exactly, whether that cell holds a ship.
The task is to hit every occupied cell — to sink the whole fleet — in as few
probes as possible.

The state carries two halves, so that "what is on the board" and "what has
already been probed" are both part of the world rather than bookkeeping the
runner keeps on the side:

* the occupancy half, hidden and constant for the whole episode;
* the probe half, fully determined by the agent's own action history.

All the uncertainty therefore lives in the occupancy half, and it is structured:
the occupied cells always decompose into the configured fleet. Keeping a belief
that respects that structure is the whole difficulty of the problem, and it is
why this environment ships its own belief
(:class:`~POMDPPlanners.environments.battleship_pomdp.battleship_belief.BattleshipBelief`)
rather than relying on a generic particle filter — observations here are
deterministic, so a generic filter kills every particle that disagrees and is
left with an empty, silently meaningless belief within a handful of probes.

Differences from the original issue's sketch, and why:

* The state is not occupancy alone. With an occupancy-only state, probing the
  same occupied cell forever pays ``+1`` forever: the optimal policy is to find
  one ship cell and re-probe it, which is not Battleship and not a search
  problem. Carrying the probe history makes the ``+1`` payable once per cell.
* ``+1`` is paid only for a cell that was occupied *and* unprobed. A repeat
  probe scores the miss penalty, exactly like probing water: it buys no
  information and costs a turn.
* The episode terminates when every occupied cell has been probed. Without a
  terminal condition, "solved the board" is not a measurable event and the
  completion metric has nothing to key on.

Classes:
    BattleshipStepChannel: Per-step measurement channels.
    BattleshipPOMDPMetrics: Metric names.
    BattleshipPOMDP: The environment.
"""

from enum import Enum
from pathlib import Path
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.distributions import DiscreteDistribution, Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.core.simulation.step_info_metrics import (
    EpisodeReduction,
    StepInfoMetric,
)
from POMDPPlanners.environments.battleship_pomdp.battleship_layouts import (
    BattleshipInitialStateDistribution,
    FleetLayoutTable,
    get_layout_table,
)


#: Observation emitted after probing an occupied cell.
HIT = 1
#: Observation emitted after probing an empty cell.
MISS = 0


class BattleshipStepChannel(Enum):
    """Per-step measurement channels reported by :meth:`BattleshipPOMDP.step_info`."""

    NEW_SHIP_CELL_HIT = "new_ship_cell_hit"
    WATER_PROBE = "water_probe"
    REPEAT_PROBE = "repeat_probe"
    FLEET_SUNK = "fleet_sunk"
    FLEET_NOT_SUNK = "fleet_not_sunk"
    EPISODE_FAILURE = "episode_failure"
    FLEET_HIT_FRACTION = "fleet_hit_fraction"
    RECORDED_STEP = "recorded_step"


class BattleshipPOMDPMetrics(Enum):
    """Metric names for the Battleship POMDP environment."""

    TASK_COMPLETION_RATE = "task_completion_rate"
    ENDED_BY_GOAL = "ended_by_goal"
    ENDED_BY_FAILURE = "ended_by_failure"
    ENDED_BY_TIMEOUT = "ended_by_timeout"
    AVERAGE_EPISODE_LENGTH = "average_episode_length"
    AVERAGE_UNIQUE_SHIP_CELL_HITS = "average_unique_ship_cell_hits"
    AVERAGE_WATER_PROBES = "average_water_probes"
    AVERAGE_REPEAT_PROBES = "average_repeat_probes"
    MAX_FLEET_HIT_FRACTION = "max_fleet_hit_fraction"


# Type alias for a Battleship state.
BattleshipState = np.ndarray


def create_battleship_state(occupancy: Sequence[int], probed: Optional[Sequence[int]] = None) -> BattleshipState:
    """Build a Battleship state array.

    Args:
        occupancy: Per-cell occupancy in row-major order, 1 for ship.
        probed: Per-cell probe flags in row-major order. Defaults to all-unprobed.

    Returns:
        ``float64`` array ``[occupancy | probed]`` of length ``2 * num_cells``.
    """
    occupancy_arr = np.asarray(occupancy, dtype=np.float64).ravel()
    if probed is None:
        probed_arr = np.zeros_like(occupancy_arr)
    else:
        probed_arr = np.asarray(probed, dtype=np.float64).ravel()
    return np.concatenate([occupancy_arr, probed_arr])


class BattleshipPOMDP(DiscreteActionsEnvironment):  # pylint: disable=too-many-public-methods
    """Battleship as a POMDP: probe a hidden fleet until every ship cell is hit.

    Dynamics:
        Deterministic. Action ``a`` is the flat index of the cell to probe;
        probing sets that cell's probe flag and changes nothing else. The
        occupancy half never changes, which is what makes the fleet "hidden and
        fixed" rather than something the agent can disturb.

    Observation model:
        Deterministic and noiseless: :data:`HIT` if the probed cell is occupied,
        :data:`MISS` otherwise. The agent already knows which cells it probed,
        so the observation adds exactly one bit per step, and only for a cell
        it has not seen before.

    Reward:
        ``hit_reward`` for probing an occupied cell that had not been probed;
        ``-miss_penalty`` for anything else — water, or a repeat of any cell.
        Exactly one of the two fires per step, so the two never stack and the
        declared range is the tighter pair, not their sum.

    Terminal:
        Every occupied cell has been probed. There is no failure terminal: an
        episode that does not finish inside the runner's step budget is a
        timeout, which the metrics report separately from completion.

    Attributes:
        board_size: Side length of the square board.
        ship_lengths: Ship lengths making up the fleet.
        allow_adjacent_ships: Whether ships may touch, including diagonally.
        hit_reward: Reward for a newly hit ship cell.
        miss_penalty: Magnitude of the penalty for a miss or a repeat probe.
        num_cells: ``board_size ** 2``.
        num_ship_cells: Total occupied cells, ``sum(ship_lengths)``.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = BattleshipPOMDP()
        >>> state = env.initial_state_dist().sample()[0]
        >>> next_state, observation, reward = env.sample_next_step(state, 0)
        >>> observation in (0, 1)
        True
        >>> env.is_terminal(state)
        False
    """

    def __init__(
        self,
        board_size: int = 5,
        ship_lengths: Sequence[int] = (3, 2, 2),
        allow_adjacent_ships: bool = True,
        hit_reward: float = 1.0,
        miss_penalty: float = 0.1,
        max_layouts: int = 2_000_000,
        discount_factor: float = 0.99,
        name: str = "Battleship",
        output_dir: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the Battleship POMDP.

        Args:
            board_size: Side length of the square board. Defaults to 5.
            ship_lengths: Ship lengths. Defaults to ``(3, 2, 2)`` — seven of
                twenty-five cells occupied, which leaves a random searcher a
                low but non-zero chance of clearing the board inside a
                sub-board-size horizon and so keeps the baseline meaningful.
            allow_adjacent_ships: Whether ships may touch, including diagonally.
                Defaults to ``True`` (the permissive rule), which leaves the
                layout space larger and the belief correspondingly less
                informative.
            hit_reward: Reward for probing a not-yet-probed occupied cell.
                Defaults to ``1.0``.
            miss_penalty: Magnitude of the penalty charged for a probe that is
                not a new hit. Must be non-negative; passed as a magnitude so
                the sign convention cannot be inverted by accident. Defaults to
                ``0.1``.
            max_layouts: Cap on the number of legal placement configurations the
                exact belief will enumerate. Defaults to ``2_000_000``.
            discount_factor: Discount factor. Defaults to ``0.99``.
            name: Environment name. Defaults to ``"Battleship"``.
            output_dir: Output directory for logging. Defaults to ``None``.
            debug: Enable debug logging. Defaults to ``False``.
            use_queue_logger: Whether to use queue-based logging.

        Raises:
            ValueError: If the geometry is invalid, the fleet cannot be placed,
                or ``miss_penalty`` is negative.
        """
        if board_size < 1:
            raise ValueError(f"board_size must be at least 1, got {board_size}")
        if not tuple(ship_lengths):
            raise ValueError("ship_lengths must contain at least one ship")
        if miss_penalty < 0.0:
            raise ValueError(
                f"miss_penalty is a magnitude and must be non-negative, got {miss_penalty}"
            )

        # Exactly one of the two branches fires per step: a probe is either a
        # new hit or it is not. Nothing stacks, so the range is the tighter pair
        # of the two outcomes rather than their sum. It holds for every board,
        # every fleet and every step, including the repeat-probe branch, which
        # scores the same as water by construction.
        min_reward = min(float(hit_reward), -float(miss_penalty))
        max_reward = max(float(hit_reward), -float(miss_penalty))

        super().__init__(
            discount_factor=discount_factor,
            name=name,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
            ),
            reward_range=(min_reward, max_reward),
            output_dir=output_dir,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )

        self.board_size = int(board_size)
        self.ship_lengths: Tuple[int, ...] = tuple(int(length) for length in ship_lengths)
        self.allow_adjacent_ships = bool(allow_adjacent_ships)
        self.hit_reward = float(hit_reward)
        self.miss_penalty = float(miss_penalty)
        self.max_layouts = int(max_layouts)

        self.num_cells = self.board_size * self.board_size
        self.num_ship_cells = int(sum(self.ship_lengths))
        if self.num_ship_cells > self.num_cells:
            raise ValueError(
                f"fleet {self.ship_lengths} needs {self.num_ship_cells} cells but the "
                f"{self.board_size}x{self.board_size} board has only {self.num_cells}"
            )

        # Validates the geometry eagerly (a fleet that cannot be placed must
        # fail at construction, not at the first episode) and warms the shared
        # table. Stored under a leading underscore so it stays out of config_id
        # and out of __eq__: it is a derived read-only artifact of the three
        # geometry parameters above, and putting a 12k-row table into the cache
        # key would be both slow and redundant.
        self._layouts: Optional[FleetLayoutTable] = get_layout_table(
            board_size=self.board_size,
            ship_lengths=self.ship_lengths,
            allow_adjacent_ships=self.allow_adjacent_ships,
            max_layouts=self.max_layouts,
        )

    # ── geometry helpers ────────────────────────────────────────────

    @property
    def layouts(self) -> FleetLayoutTable:
        """The enumerated legal fleet layouts for this board geometry."""
        if self._layouts is None:
            self._layouts = get_layout_table(
                board_size=self.board_size,
                ship_lengths=self.ship_lengths,
                allow_adjacent_ships=self.allow_adjacent_ships,
                max_layouts=self.max_layouts,
            )
        return self._layouts

    def occupancy(self, state: BattleshipState) -> np.ndarray:
        """Return the hidden occupancy half of ``state`` as a ``bool`` array."""
        return np.asarray(state, dtype=np.float64)[: self.num_cells] > 0.5

    def probed(self, state: BattleshipState) -> np.ndarray:
        """Return the observed probe half of ``state`` as a ``bool`` array."""
        return np.asarray(state, dtype=np.float64)[self.num_cells :] > 0.5

    def get_actions(self) -> List[int]:
        """Return every probe action: one per board cell, in row-major order."""
        return list(range(self.num_cells))

    def action_to_cell(self, action: int) -> Tuple[int, int]:
        """Return the ``(row, col)`` the probe action ``action`` targets."""
        return divmod(int(action), self.board_size)

    # ── dynamics ────────────────────────────────────────────────────

    def sample_next_state(
        self, state: BattleshipState, action: int, n_samples: int = 1
    ) -> Any:
        """Apply the probe. Deterministic, so every sample is the same state.

        Args:
            state: Current state.
            action: Flat index of the probed cell.
            n_samples: How many successor samples to return. Defaults to 1.

        Returns:
            A single ``float64`` state when ``n_samples == 1``, else an
            ``(n_samples, 2 * num_cells)`` ``float64`` array.
        """
        next_state = np.array(state, dtype=np.float64, copy=True)
        next_state[self.num_cells + int(action)] = 1.0
        if n_samples == 1:
            return next_state
        return np.tile(next_state, (int(n_samples), 1))

    def sample_next_state_batch(self, states: Any, action: int) -> np.ndarray:
        """Apply one probe to every input state.

        Args:
            states: ``(N, 2 * num_cells)`` array-like of particles.
            action: The probed cell index, shared by every particle.

        Returns:
            ``(N, 2 * num_cells)`` ``float64`` array. The dtype matches
            :meth:`sample_next_state` so a particle filter mixing the two paths
            cannot silently change particle precision.
        """
        next_states = np.array(states, dtype=np.float64, copy=True)
        if next_states.ndim == 1:
            next_states = next_states.reshape(1, -1)
        next_states[:, self.num_cells + int(action)] = 1.0
        return next_states

    def transition_log_probability(
        self, state: BattleshipState, action: int, next_states: Any
    ) -> np.ndarray:
        """Log-probability of each candidate successor. Deterministic.

        Args:
            state: Current state.
            action: The probed cell index.
            next_states: Candidate successors.

        Returns:
            ``0.0`` for the one realisable successor, ``-inf`` otherwise.
        """
        expected = self.sample_next_state(state=state, action=action)
        candidates = np.asarray(next_states, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        matches = np.all(np.abs(candidates - expected) < 0.5, axis=1)
        return np.where(matches, 0.0, -np.inf)

    def sample_observation(
        self, next_state: BattleshipState, action: int, n_samples: int = 1
    ) -> Any:
        """Report whether the probed cell holds a ship. Deterministic.

        Args:
            next_state: The post-probe state.
            action: The probed cell index.
            n_samples: How many observation samples to return. Defaults to 1.

        Returns:
            :data:`HIT` or :data:`MISS` when ``n_samples == 1``, else a list of
            ``n_samples`` copies of it.
        """
        observation = HIT if bool(self.occupancy(next_state)[int(action)]) else MISS
        if n_samples == 1:
            return observation
        return [observation] * int(n_samples)

    def observation_log_probability(
        self, next_state: BattleshipState, action: int, observations: Any
    ) -> np.ndarray:
        """Log-likelihood of each candidate observation. Deterministic.

        Args:
            next_state: The post-probe state.
            action: The probed cell index.
            observations: Candidate observations.

        Returns:
            ``0.0`` for the observation the sensor would emit, ``-inf``
            otherwise. ``-inf`` is deliberate rather than a floored value: an
            impossible reading really has zero likelihood here, and the base
            :class:`WeightedParticleBelief` path already floors it with ``eps``
            before normalising. This environment does not override
            ``observation_log_probability_per_state``, precisely so that flooring
            stays in play for any caller that does use a generic filter.
        """
        truth = HIT if bool(self.occupancy(next_state)[int(action)]) else MISS
        candidates = np.asarray(observations).ravel()
        return np.where(candidates == truth, 0.0, -np.inf).astype(np.float64)

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        """Check whether two hit/miss observations are the same."""
        return int(observation1) == int(observation2)

    def hash_observation(self, observation: Any) -> Hashable:
        """Return a hashable key agreeing with :meth:`is_equal_observation`."""
        return int(observation)

    def hash_action(self, action: Any) -> Hashable:
        """Return a hashable key for a probe action (already an int)."""
        return int(action)

    # ── reward ──────────────────────────────────────────────────────

    def reward(self, state: BattleshipState, action: int, next_state: Any = None) -> float:
        """Score one probe.

        The reward is a pure function of ``(state, action)``: whether the probe
        lands on a new ship cell is decided by the state it is taken from, so
        ``next_state`` is accepted and ignored, and
        :attr:`reward_requires_next_state` stays ``False``.

        Args:
            state: The state the probe is taken from.
            action: The probed cell index.
            next_state: Unused.

        Returns:
            ``hit_reward`` for a newly hit ship cell, ``-miss_penalty`` otherwise.
        """
        del next_state
        state_arr = np.asarray(state, dtype=np.float64)
        cell = int(action)
        holds_ship = bool(state_arr[cell] > 0.5)
        already_probed = bool(state_arr[self.num_cells + cell] > 0.5)
        return self.hit_reward if holds_ship and not already_probed else -self.miss_penalty

    def reward_batch(
        self,
        states: Union[np.ndarray, Sequence[Any]],
        action: int,
        next_states: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Score one probe against a batch of states.

        Args:
            states: ``(N, 2 * num_cells)`` array-like of states.
            action: The probed cell index.
            next_states: Unused; see :meth:`reward`.

        Returns:
            ``(N,)`` ``float64`` array agreeing element-wise with :meth:`reward`.
        """
        del next_states
        states_arr = np.asarray(states, dtype=np.float64)
        if states_arr.ndim == 1:
            states_arr = states_arr.reshape(1, -1)
        cell = int(action)
        is_new_hit = (states_arr[:, cell] > 0.5) & (states_arr[:, self.num_cells + cell] <= 0.5)
        return np.where(is_new_hit, self.hit_reward, -self.miss_penalty).astype(np.float64)

    # ── terminal / initial ──────────────────────────────────────────

    def is_terminal(self, state: BattleshipState) -> bool:
        """Whether every occupied cell has been probed."""
        state_arr = np.asarray(state, dtype=np.float64)
        return not bool(
            np.any((state_arr[: self.num_cells] > 0.5) & (state_arr[self.num_cells :] <= 0.5))
        )

    def initial_state_dist(self) -> Distribution:
        """Uniform distribution over legal fleet placements, nothing probed."""
        return BattleshipInitialStateDistribution(self.layouts)

    def initial_observation_dist(self) -> DiscreteDistribution:
        """The pre-probe observation, which carries no information."""
        return DiscreteDistribution(values=[MISS], probs=np.array([1.0]))

    # ── metrics ─────────────────────────────────────────────────────

    def step_info(self, state: Any, action: Any, next_state: Any) -> Dict[str, float]:
        """Report the per-step channels this environment's metrics are built on.

        Draws no randomness: every channel is read straight off ``state`` and
        ``action``.

        Args:
            state: The state the probe was taken from, or the final state on the
                terminal bookkeeping step.
            action: The probed cell, or ``None`` on the terminal step.
            next_state: The realised successor, or ``None`` on the terminal step.
                The two probe-outcome channels are scored against ``state``,
                because whether a probe is a new hit is decided by the board it
                was taken from — the same convention :meth:`reward` uses. The
                board-progress channels are scored against ``next_state``
                instead, and that is deliberate: the episode runner checks its
                step budget *before* it checks terminality, so an episode whose
                final allowed probe sinks the fleet never records a terminal
                bookkeeping step. Reading progress from ``state`` alone would
                score that episode as an unfinished timeout with the last hit
                missing, which is wrong in the one place it matters most.

        Returns:
            The channels named by :class:`BattleshipStepChannel`. The three
            probe-outcome channels report ``0.0`` on the terminal step, where no
            probe was taken; the board-progress channels are still reported
            there, because a completed episode's final board is recorded only on
            that step.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        occupancy = state_arr[: self.num_cells] > 0.5
        probed = state_arr[self.num_cells :] > 0.5

        new_hit = 0.0
        water = 0.0
        repeat = 0.0
        if action is not None:
            cell = int(action)
            if probed[cell]:
                repeat = 1.0
            elif occupancy[cell]:
                new_hit = 1.0
            else:
                water = 1.0

        board = state_arr if next_state is None else np.asarray(next_state, dtype=np.float64)
        board_hits = np.count_nonzero(
            (board[: self.num_cells] > 0.5) & (board[self.num_cells :] > 0.5)
        )
        sunk = float(self.is_terminal(board))
        hit_fraction = (
            float(board_hits) / float(self.num_ship_cells) if self.num_ship_cells else 1.0
        )
        return {
            BattleshipStepChannel.NEW_SHIP_CELL_HIT.value: new_hit,
            BattleshipStepChannel.WATER_PROBE.value: water,
            BattleshipStepChannel.REPEAT_PROBE.value: repeat,
            BattleshipStepChannel.FLEET_SUNK.value: sunk,
            BattleshipStepChannel.FLEET_NOT_SUNK.value: 1.0 - sunk,
            # Battleship has no way to fail: an unfinished episode has simply
            # run out of steps. The channel is still emitted, as a constant,
            # because a declared-but-unreported channel is silently dropped and
            # because a reader comparing the three end-reason rates needs all
            # three to sum to one.
            BattleshipStepChannel.EPISODE_FAILURE.value: 0.0,
            BattleshipStepChannel.FLEET_HIT_FRACTION.value: hit_fraction,
            BattleshipStepChannel.RECORDED_STEP.value: 1.0,
        }

    def get_metric_specs(self) -> List[StepInfoMetric]:
        """Declare the Battleship metrics.

        Completion reduces with ``ANY`` rather than ``ALL``: sinking the fleet
        is a thing that happens once and stays true, not a condition every step
        has to hold. ``ended_by_*`` reduce with ``LAST`` — a completed episode's
        last recorded step is the terminal bookkeeping step, where the board is
        sunk, and a timed-out episode's last step is an ordinary probe from a
        board that is not.

        There is no danger metric: nothing in Battleship can be damaged, entered
        or violated. The nearest thing is the repeat probe — an action that
        cannot buy information and only burns a turn — which is reported as a
        count.

        Returns:
            One spec per metric named in :class:`BattleshipPOMDPMetrics`.
        """
        return [
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.TASK_COMPLETION_RATE.value,
                channel=BattleshipStepChannel.FLEET_SUNK.value,
                per_episode=EpisodeReduction.ANY,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.ENDED_BY_GOAL.value,
                channel=BattleshipStepChannel.FLEET_SUNK.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.ENDED_BY_FAILURE.value,
                channel=BattleshipStepChannel.EPISODE_FAILURE.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.ENDED_BY_TIMEOUT.value,
                channel=BattleshipStepChannel.FLEET_NOT_SUNK.value,
                per_episode=EpisodeReduction.LAST,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.AVERAGE_EPISODE_LENGTH.value,
                channel=BattleshipStepChannel.RECORDED_STEP.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.AVERAGE_UNIQUE_SHIP_CELL_HITS.value,
                channel=BattleshipStepChannel.NEW_SHIP_CELL_HIT.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.AVERAGE_WATER_PROBES.value,
                channel=BattleshipStepChannel.WATER_PROBE.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.AVERAGE_REPEAT_PROBES.value,
                channel=BattleshipStepChannel.REPEAT_PROBE.value,
                per_episode=EpisodeReduction.SUM,
            ),
            StepInfoMetric(
                name=BattleshipPOMDPMetrics.MAX_FLEET_HIT_FRACTION.value,
                channel=BattleshipStepChannel.FLEET_HIT_FRACTION.value,
                per_episode=EpisodeReduction.MAX,
            ),
        ]

    # ── pickling ────────────────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the layout table before pickling.

        The table is a 12k-row derived artifact shared process-wide. Shipping a
        copy to every parallel worker would cost far more than the ~20 ms it
        takes each worker to rebuild it once, and two workers holding private
        copies of the same read-only table is pure waste.
        """
        state = self.__dict__.copy()
        state["_layouts"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore, leaving the layout table to be rebuilt on first use."""
        vars(self).update(state)
        self._layouts = None

    # ── visualization ───────────────────────────────────────────────

    def cache_visualization(
        self, history: List[StepData], output_dir: Path, episode_index: int
    ) -> None:
        """Write the episode's animated GIF into ``output_dir``.

        Args:
            history: Episode history.
            output_dir: Directory to write into.
            episode_index: Zero-based episode index, used to name the file.
        """
        # Imported lazily: matplotlib is heavy and every parallel worker imports
        # this module, while almost none of them render anything.
        from POMDPPlanners.environments.battleship_pomdp.battleship_visualizer import (  # pylint: disable=import-outside-toplevel
            BattleshipVisualizer,
        )

        cache_path = output_dir / f"battleship_board_{episode_index}.gif"
        BattleshipVisualizer(self).create_visualization(history, cache_path)
