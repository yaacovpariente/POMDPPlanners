# SPDX-License-Identifier: MIT

"""Contracts for the two pieces a planner-side generative model injects.

A forward-only world -- a live simulator such as IsaacLab or CARLA -- can step
itself but cannot resample from an arbitrary state, and cannot say what reward a
hypothetical in-tree transition would earn. A planner searching such a world
therefore needs a *separate* generative model, and the two swappable pieces of
that model are the dynamics and the objective.

Both live here rather than beside any one environment because model *learning*
is generic: a routine that fits a transition from rollouts and hands it back
should not have to import an environment package to name what it produced. The
environments that first defined these interfaces re-export them, so existing
import paths are unchanged.

The transition is deliberately two methods, not one. A sampler alone is enough
for a planner that only rolls forward, but a particle filter needs the matching
density to weight, and a model *fitted by likelihood* needs it to state its own
training objective. Anything implementing both can be planned with and can be
learned.

Classes:
    TransitionModel: Interface for a state-transition model (sample + log-density).
    RewardModel: Interface for a reward model over a transition.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class TransitionModel(ABC):
    """Interface for a state-transition model over the shared state/observation space.

    A concrete transition supplies a generative next-state sampler and the matching
    log-density, both conditioned on ``(state, action)``. Implementations may be
    analytic Gaussians or models fitted from rollouts; the planner cannot tell the
    difference, which is the point.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        """Sample ``n_samples`` next states for ``(state, action)``.

        Args:
            state: The current state, a length-``dim`` vector.
            action: The action applied at ``state``.
            n_samples: Number of next states to draw.

        Returns:
            A single ``(dim,)`` next state when ``n_samples == 1``, else a
            ``(n_samples, dim)`` array.
        """

    @abstractmethod
    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        """Log-density of each of ``next_states`` under the transition for ``(state, action)``.

        Args:
            state: The current state, a length-``dim`` vector.
            action: The action applied at ``state``.
            next_states: A single ``(dim,)`` next state or a ``(n, dim)`` batch.

        Returns:
            A ``(n,)`` array of log-densities.
        """

    @property
    def fingerprint(self) -> Optional[str]:
        """Hash of the fitted parameters, or ``None`` when there are none to hash.

        An environment holds its transition privately, and ``config_id`` skips
        private attributes, so nothing about a *fitted* transition reaches the
        simulation cache key on its own: two rounds of a fitting loop look like
        the same experiment and the second is served the first's episodes. A
        transition with fitted parameters overrides this to report them.

        ``None`` -- the default -- says the transition is analytic and its
        configuration is already covered by the environment holding it.
        """
        return None


class RewardModel(ABC):
    """Interface for a reward model over the shared state/observation space.

    A concrete reward model scores a ``(state, action, next_state)`` transition.
    The planner-side model needs one because the forward-only world cannot be
    queried for the reward of a hypothetical in-tree transition.

    Note:
        This is an abstract base class and cannot be instantiated directly.
    """

    @abstractmethod
    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Return the scalar reward for a ``(state, action, next_state)`` transition."""
