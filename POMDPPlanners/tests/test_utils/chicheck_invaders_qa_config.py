# SPDX-License-Identifier: MIT

"""The uninformed baseline the Chicheck Invaders QA pass measures against.

A completion rate on its own proves nothing: an environment whose flock clears
itself, or whose terminal condition fires immediately, is "solved" by anything
at all. The gate is therefore a *comparison*, and both sides of it are test
inputs rather than measurements -- neither is reproducible without the other.

The PFT-DPW settings live in :mod:`env_qa_planner_configs` beside the other
environments'; this module holds the baseline, because it is a class rather than
a dict and because it has to be importable from a module a worker process can
find. A policy defined in a run script cannot be pickled, so the parallel task
manager refuses it.

Measured values deliberately live nowhere in the repository. A committed
completion rate goes stale silently and starts lying; the runs recompute it.

Classes:
    UniformRandomGunner: The uninformed baseline.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo


class UniformRandomGunner(Policy):
    """Draws one of the four actions uniformly at random, ignoring the belief.

    Uniform over all four actions rather than something cleverer, and that is
    the right baseline here rather than a weak one. The obvious "stronger"
    uninformed policy -- fire whenever the cooldown allows -- already encodes
    the one insight the task turns on, that shots are what clear the flock, so
    beating it would say nothing about reasoning under uncertainty. It is also
    not obviously weaker: with four chickens on an eight-column grid, firing
    constantly hits something eventually. Uniform random is the policy that
    knows nothing at all, which is what the gate asks the planner to beat.

    Attributes:
        actions: The environment's discrete action set.
    """

    def __init__(
        self,
        environment: Any,
        discount_factor: float,
        name: str = "UniformRandomGunner",
        log_path: Optional[Any] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        """Initialize the baseline.

        Args:
            environment: The environment being played.
            discount_factor: Discount factor, matched to the environment's.
            name: Policy name. Defaults to ``"UniformRandomGunner"``.
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
        self.actions = list(environment.get_actions())

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        """Discrete actions and discrete observations."""
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.DISCRETE
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        """No per-decision diagnostics; the baseline has nothing to report."""
        return []

    def action(self, belief: Any) -> Tuple[List[Any], PolicyRunData]:
        """Pick one action uniformly at random.

        Draws from the global ``np.random`` stream, which the simulator seeds
        per episode, so the baseline reproduces with the rest of a run.

        Args:
            belief: The current belief. Deliberately unused.

        Returns:
            A one-action list and empty run data.
        """
        del belief
        return [int(np.random.choice(self.actions))], PolicyRunData(info_variables=[])
