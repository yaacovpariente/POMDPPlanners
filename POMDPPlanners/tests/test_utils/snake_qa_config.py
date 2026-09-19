# SPDX-License-Identifier: MIT

"""Pinned QA inputs for the Snake POMDP.

An environment nobody has solved is indistinguishable from a broken one, so the
Snake QA pass measures a planner against a baseline. Both sides of that
comparison are *test inputs*, not measurements, and neither is reproducible
without the other: a completion rate quoted without the planner config and
without the baseline it beat is a number nobody can check.

So this module holds three things and no results:

* :func:`snake_pft_dpw_config` -- the PFT-DPW settings the QA pass used.
* :class:`SnakeRandomSurvivor` -- the baseline it was measured against.
* :data:`SNAKE_QA_NUM_STEPS` and :data:`SNAKE_QA_ENV_KWARGS` -- the horizon and
  the world, which together decide how hard the task is and therefore what
  "beating the baseline" is worth.

Measured values deliberately live nowhere in the repository. A committed
completion rate goes stale silently and starts lying; the tests recompute it.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.environments.snake_pomdp.snake_pomdp import SnakeAction


#: The world the QA pass runs on.
#:
#: Smaller than the environment's own defaults, and deliberately so. On the
#: default 12x12 grid with a target length of 10 the snake has to find seven
#: separate food items, each of which starts as a near-uniform posterior over
#: 140 cells; a single episode runs to several hundred steps and a tree search
#: with a per-decision budget has no chance of finishing one inside a QA run.
#: The 8x8 grid with a target of 5 keeps every rule of the environment intact --
#: the same sensors, the same three ways to die -- while making completion
#: reachable, which is what the QA gate needs to be able to observe.
SNAKE_QA_ENV_KWARGS: Dict[str, Any] = {
    "grid_size": 8,
    "target_length": 5,
    "window_radius": 2,
    "detection_probability": 0.9,
    "scent_accuracy": 0.7,
    "starvation_limit": 60,
}

#: Steps allowed per episode.
#:
#: Fixed at 60 and fixed *before* any measurement. Two food items have to be
#: found on the QA world, and a snake that walks straight at the belief's mode
#: crosses the 8x8 grid in about seven steps, so 60 leaves room for a search
#: that goes the wrong way once or twice without making the horizon itself the
#: thing being measured. It matches ``starvation_limit``, so an episode that
#: runs out of steps was also about to starve.
SNAKE_QA_NUM_STEPS = 60

#: Particles in the QA belief.
#:
#: SnakeBelief redraws its particles from the exact posterior on every update,
#: so this sets sampling noise inside the search rather than how long the belief
#: survives -- it cannot deplete. 200 matches the Battleship QA belief size.
SNAKE_QA_BELIEF_PARTICLES = 200


def snake_pft_dpw_config(time_out_in_seconds: float = 0.75) -> Dict[str, Any]:
    """PFT-DPW settings used for the Snake QA pass.

    Snake has no torch vectorized model, so QA uses PFT-DPW on the scalar
    Environment API rather than VOPP.

    Action widening is disabled on purpose: there are three actions, so
    ``alpha_a = 0`` with ``k_a = 3`` admits all of them at every node and no
    action is excluded by an accident of the sampler. Observation widening is
    left on, because the observation space is *not* small -- a reading carries
    the body, a possible sighting and one of four quadrants -- so a node would
    otherwise grow a fresh child for almost every rollout and never revisit one.

    The depth is 12 rather than the horizon: a plan that reaches the food is
    about seven steps on the QA world, and searching deeper than that spends the
    budget on branches past the next meal, which the respawn makes unpredictable
    anyway.

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
        "depth": 12,
        "k_a": 3.0,
        "alpha_a": 0.0,
        "k_o": 3.0,
        "alpha_o": 0.3,
        "exploration_constant": 2.0,
        "time_out_in_seconds": time_out_in_seconds,
    }


class SnakeRandomSurvivor(Policy):
    """Uniform random turning, skipping moves that are certainly fatal.

    The baseline has to be the strongest *uninformed* policy, not the weakest
    one available. A policy drawing uniformly from the three actions walks into
    a wall within a few steps on an 8x8 grid, and beating it would prove nothing
    about the planner: avoiding a wall needs no belief at all, because the body
    and the grid are both fully observed. Filtering those moves out removes that
    free win, so whatever margin the planner shows is a margin from reasoning
    about *where the food is*.

    It reads the body from the belief's own particles, which all share one body.
    That is information the agent genuinely has -- it is its own action history.
    The hidden food is never touched.

    Attributes:
        environment: The Snake environment.
    """

    def __init__(
        self,
        environment: Any,
        discount_factor: float,
        name: str = "RandomSurvivor",
        log_path: Optional[Any] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the baseline.

        Args:
            environment: The Snake environment.
            discount_factor: Discount factor, matched to the environment's.
            name: Policy name. Defaults to ``"RandomSurvivor"``.
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
        """Pick uniformly among the turns that do not certainly kill.

        Args:
            belief: The current belief. Only the shared body is read.

        Returns:
            A one-action list and empty run data. When every turn is fatal --
            the snake has boxed itself in -- it falls back to a uniform draw
            rather than failing.
        """
        environment = self.environment
        particle = np.asarray(belief.particles[0], dtype=np.float64)
        body = environment.body(particle)  # type: ignore[attr-defined]
        heading = environment.heading(particle)  # type: ignore[attr-defined]
        survivable = []
        for action in environment.get_actions():  # type: ignore[attr-defined]
            turned = environment.turn(heading, action)  # type: ignore[attr-defined]
            cell = (body[0][0] + turned[0], body[0][1] + turned[1])
            # ``body[:-1]`` rather than ``body``: the tail cell is released on
            # any step that does not eat, so entering it is usually legal. The
            # baseline cannot know whether this step eats, so it takes the
            # optimistic reading -- the same one a human player takes.
            if environment.in_grid(cell) and cell not in body[:-1]:  # type: ignore[attr-defined]
                survivable.append(int(action))
        if not survivable:
            return [int(random.choice([int(a) for a in SnakeAction]))], PolicyRunData(
                info_variables=[]
            )
        return [int(random.choice(survivable))], PolicyRunData(info_variables=[])
