"""Vectorized particle belief updaters for the Discrete Light-Dark POMDP.

This module implements concrete
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
subclasses that perform batched state transitions and observation
log-likelihood evaluations for the Discrete Light-Dark environment,
replacing per-particle Python loops with NumPy array operations.

Three updaters correspond to the three
:class:`~POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp.ObservationModelType`
variants:

Classes:
    DiscreteLightDarkVectorizedUpdater: ``NORMAL`` — discrete
        observations with binary near/far error factor.
    DiscreteLightDarkNoObsInDarkVectorizedUpdater:
        ``NO_OBS_IN_DARK`` — returns ``"None"`` when far from beacons.
    DiscreteLightDarkDistanceBasedVectorizedUpdater:
        ``DISTANCE_BASED`` — per-particle linear error scaling by
        distance to nearest beacon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        DiscreteLightDarkPOMDP,
    )


class DiscreteLightDarkVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for the Discrete Light-Dark POMDP.

    Performs all-particle transitions and observation log-likelihood
    evaluations using vectorized NumPy operations.  The transition is
    stochastic: the intended action executes with probability
    ``1 - transition_error_prob`` and a uniformly random other action
    executes otherwise.

    Attributes:
        transition_error_prob: Probability of executing a random action.
        observation_error_prob: Base observation error probability.
        beacons: Beacon positions as a (2, n_beacons) array.
        beacon_radius: Distance threshold for beacon proximity.
        grid_size: Grid boundary for the environment.
        actions: List of action names.
        action_to_vector: Mapping from action names to direction vectors.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        ...     DiscreteLightDarkPOMDP,
        ... )
        >>> env = DiscreteLightDarkPOMDP(discount_factor=0.95)
        >>> updater = DiscreteLightDarkVectorizedUpdater.from_environment(env)
        >>> particles = np.array([[5, 5], [3, 3], [7, 7]], dtype=float)
        >>> next_p = updater.batch_transition(particles, "right")
        >>> next_p.shape
        (3, 2)
        >>> obs = np.array([6, 5])
        >>> ll = updater.batch_observation_log_likelihood(next_p, "right", obs)
        >>> ll.shape
        (3,)
    """

    def __init__(
        self,
        transition_error_prob: float,
        observation_error_prob: float,
        beacons: np.ndarray,
        beacon_radius: float,
        grid_size: int,
        actions: List[str],
        action_to_vector: Dict[str, np.ndarray],
    ):
        """Initialize the vectorized updater.

        Args:
            transition_error_prob: Probability of executing a random action.
            observation_error_prob: Base observation error probability.
            beacons: Beacon positions as a (2, n_beacons) array.
            beacon_radius: Distance threshold for near-beacon classification.
            grid_size: Grid boundary used by the environment.
            actions: Ordered list of action names.
            action_to_vector: Mapping from action name to 2-D direction vector.
        """
        self.transition_error_prob = transition_error_prob
        self.observation_error_prob = observation_error_prob
        self.beacons = np.asarray(beacons, dtype=float)
        self.beacon_radius = beacon_radius
        self.beacon_radius_sq = beacon_radius * beacon_radius
        self.grid_size = grid_size
        self.actions = actions
        self.action_to_vector = action_to_vector

        self._precompute_tables()

    @classmethod
    def from_environment(
        cls, env: "DiscreteLightDarkPOMDP"
    ) -> "DiscreteLightDarkVectorizedUpdater":
        """Construct an updater from a DiscreteLightDarkPOMDP instance.

        Args:
            env: Environment to extract parameters from.

        Returns:
            A new updater instance of the called class.
        """
        return cls(
            transition_error_prob=env.transition_error_prob,
            observation_error_prob=env.observation_error_prob,
            beacons=env.beacons,
            beacon_radius=env.beacon_radius,
            grid_size=env.grid_size,
            actions=list(env.actions),
            action_to_vector={k: v.copy() for k, v in env.action_to_vector.items()},
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_str = action if isinstance(action, str) else str(action)
        intended_idx = self.actions.index(action_str)
        n = particles.shape[0]

        error_mask = np.random.random(n) < self.transition_error_prob
        chosen = np.full(n, intended_idx, dtype=int)

        error_count = int(error_mask.sum())
        if error_count > 0:
            other = [i for i in range(self._n_actions) if i != intended_idx]
            chosen[error_mask] = np.random.choice(other, size=error_count)

        return particles + self._action_vectors_matrix[chosen]

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # noqa: ARG002
        observation: Union[np.ndarray, str],
    ) -> np.ndarray:
        px = next_particles[:, 0]
        py = next_particles[:, 1]
        near_mask = self._batch_near_beacon(px, py)

        obs = np.asarray(observation, dtype=float).ravel()
        matched_idx = self._match_observation_offset(next_particles, obs)

        log_probs = np.full(next_particles.shape[0], -np.inf)
        for j in range(self._n_obs):
            mask_j = matched_idx == j
            if not np.any(mask_j):
                continue
            log_probs[mask_j & near_mask] = self._obs_log_probs_near[j]
            log_probs[mask_j & ~near_mask] = self._obs_log_probs_far[j]

        return log_probs

    @property
    def config_id(self) -> str:
        return config_to_id(self._build_config_dict("DiscreteLightDarkVectorizedUpdater"))

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_tables(self):
        self._n_actions = len(self.actions)
        self._n_obs = self._n_actions + 1

        # Action vectors as matrix for fast indexing
        self._action_vectors_matrix = np.array(
            [self.action_to_vector[a] for a in self.actions], dtype=float
        )

        # Observation offsets: 4 directional + identity (exact match)
        self._obs_offsets = np.vstack([self._action_vectors_matrix, np.zeros((1, 2))])

        # Near-beacon observation log probabilities (error_factor = 0.2)
        near_error = self.observation_error_prob * 0.2
        near_probs = np.ones(self._n_obs) * (near_error / (self._n_obs - 1))
        near_probs[-1] = 1.0 - near_error
        self._obs_log_probs_near = np.log(near_probs)

        # Far-from-beacon observation log probabilities (error_factor = 1.0)
        far_error = self.observation_error_prob * 1.0
        far_probs = np.ones(self._n_obs) * (far_error / (self._n_obs - 1))
        far_probs[-1] = 1.0 - far_error
        self._obs_log_probs_far = np.log(far_probs)

        # Beacon arrays for vectorized distance computation
        self._beacon_x = np.ascontiguousarray(self.beacons[0])
        self._beacon_y = np.ascontiguousarray(self.beacons[1])

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def _batch_near_beacon(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        bx = px[:, np.newaxis] - self._beacon_x
        by = py[:, np.newaxis] - self._beacon_y
        sq_distances = bx**2 + by**2
        return sq_distances.min(axis=1) < self.beacon_radius_sq

    def _batch_min_distance_to_beacon(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        bx = px[:, np.newaxis] - self._beacon_x
        by = py[:, np.newaxis] - self._beacon_y
        sq_distances = bx**2 + by**2
        return np.sqrt(sq_distances.min(axis=1))

    def _match_observation_offset(self, next_particles: np.ndarray, obs: np.ndarray) -> np.ndarray:
        diff = obs - next_particles  # (N, 2)
        # Compare against each of the n_obs offsets
        # offsets: (n_obs, 2), diff: (N, 2)
        matches = np.all(
            np.abs(diff[:, np.newaxis, :] - self._obs_offsets[np.newaxis, :, :]) < 1e-9,
            axis=2,
        )  # (N, n_obs)
        # For each particle, find which offset matched (or -1 if none)
        any_match = matches.any(axis=1)
        result = np.full(next_particles.shape[0], -1, dtype=int)
        result[any_match] = matches[any_match].argmax(axis=1)
        return result

    def _build_config_dict(self, class_name: str) -> dict:
        return {
            "class": class_name,
            "transition_error_prob": self.transition_error_prob,
            "observation_error_prob": self.observation_error_prob,
            "beacons": self.beacons.tolist(),
            "beacon_radius": self.beacon_radius,
            "grid_size": self.grid_size,
            "actions": self.actions,
            "action_to_vector": {k: v.tolist() for k, v in self.action_to_vector.items()},
        }


class DiscreteLightDarkNoObsInDarkVectorizedUpdater(
    DiscreteLightDarkVectorizedUpdater,
):
    """Vectorized updater for the ``NO_OBS_IN_DARK`` discrete observation model.

    Particles far from all beacons receive ``"None"`` observations
    (log-likelihood 0 for ``"None"``, ``-inf`` for any array observation).
    Particles near a beacon use the near-beacon discrete distribution.

    Inherits transition logic and precomputation from
    :class:`DiscreteLightDarkVectorizedUpdater`.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        ...     DiscreteLightDarkPOMDP, ObservationModelType,
        ... )
        >>> env = DiscreteLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     observation_model_type=ObservationModelType.NO_OBS_IN_DARK,
        ... )
        >>> updater = DiscreteLightDarkNoObsInDarkVectorizedUpdater.from_environment(env)
        >>> particles = np.array([[5, 5], [3, 3]], dtype=float)
        >>> next_p = updater.batch_transition(particles, "right")
        >>> ll = updater.batch_observation_log_likelihood(next_p, "right", "None")
        >>> ll.shape
        (2,)
    """

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # noqa: ARG002
        observation: Union[np.ndarray, str],
    ) -> np.ndarray:
        px = next_particles[:, 0]
        py = next_particles[:, 1]
        near_mask = self._batch_near_beacon(px, py)

        if isinstance(observation, str):
            return np.where(near_mask, -np.inf, 0.0)

        return self._log_lik_near_only(next_particles, near_mask, observation)

    @property
    def config_id(self) -> str:
        return config_to_id(
            self._build_config_dict("DiscreteLightDarkNoObsInDarkVectorizedUpdater")
        )

    def _log_lik_near_only(
        self,
        next_particles: np.ndarray,
        near_mask: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).ravel()
        matched_idx = self._match_observation_offset(next_particles, obs)

        log_probs = np.full(next_particles.shape[0], -np.inf)
        for j in range(self._n_obs):
            mask_j = matched_idx == j
            if not np.any(mask_j):
                continue
            log_probs[mask_j & near_mask] = self._obs_log_probs_near[j]
            # Far particles stay at -inf (actual observations impossible)

        return log_probs


class DiscreteLightDarkDistanceBasedVectorizedUpdater(
    DiscreteLightDarkVectorizedUpdater,
):
    """Vectorized updater for the ``DISTANCE_BASED`` discrete observation model.

    Error probability scales linearly per particle based on distance to
    the nearest beacon.  Particles beyond ``beacon_radius`` receive
    ``"None"`` observations.

    The scaling formula is:
        ``error_factor = 0.0001 + 0.9999 * (distance / beacon_radius)``

    Inherits transition logic from
    :class:`DiscreteLightDarkVectorizedUpdater`.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
        ...     DiscreteLightDarkPOMDP, ObservationModelType,
        ... )
        >>> env = DiscreteLightDarkPOMDP(
        ...     discount_factor=0.95,
        ...     observation_model_type=ObservationModelType.DISTANCE_BASED,
        ... )
        >>> updater = DiscreteLightDarkDistanceBasedVectorizedUpdater.from_environment(env)
        >>> particles = np.array([[5, 5], [3, 3]], dtype=float)
        >>> next_p = updater.batch_transition(particles, "right")
        >>> ll = updater.batch_observation_log_likelihood(next_p, "right", "None")
        >>> ll.shape
        (2,)
    """

    _MIN_ERROR_FACTOR = 0.0001

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # noqa: ARG002
        observation: Union[np.ndarray, str],
    ) -> np.ndarray:
        px = next_particles[:, 0]
        py = next_particles[:, 1]
        min_dist = self._batch_min_distance_to_beacon(px, py)
        within_mask = min_dist <= self.beacon_radius

        if isinstance(observation, str):
            return np.where(within_mask, -np.inf, 0.0)

        return self._distance_scaled_log_lik(next_particles, min_dist, within_mask, observation)

    @property
    def config_id(self) -> str:
        return config_to_id(
            self._build_config_dict("DiscreteLightDarkDistanceBasedVectorizedUpdater")
        )

    def _distance_scaled_log_lik(
        self,
        next_particles: np.ndarray,
        min_dist: np.ndarray,
        within_mask: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=float).ravel()
        matched_idx = self._match_observation_offset(next_particles, obs)
        n = next_particles.shape[0]

        # Compute per-particle error factor (only meaningful within beacon_radius)
        distance_ratio = np.zeros(n)
        distance_ratio[within_mask] = min_dist[within_mask] / self.beacon_radius
        error_factor = self._MIN_ERROR_FACTOR + (1.0 - self._MIN_ERROR_FACTOR) * distance_ratio
        obs_error = self.observation_error_prob * error_factor

        # Per-particle log probabilities
        is_exact = matched_idx == (self._n_obs - 1)
        is_directional = (matched_idx >= 0) & (matched_idx < self._n_obs - 1)

        log_probs = np.full(n, -np.inf)

        # Exact match: log(1 - obs_error)
        exact_within = is_exact & within_mask
        if np.any(exact_within):
            log_probs[exact_within] = np.log(1.0 - obs_error[exact_within])

        # Directional match: log(obs_error / (n_obs - 1))
        dir_within = is_directional & within_mask
        if np.any(dir_within):
            log_probs[dir_within] = np.log(obs_error[dir_within] / (self._n_obs - 1))

        return log_probs
