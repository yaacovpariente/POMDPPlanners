# SPDX-License-Identifier: MIT

"""Vectorized whole-map filter for the occupancy-grid mapping POMDP.

This is the batched twin of
:class:`~POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_belief.OccupancyGridMappingBelief`.
It keeps that filter's semantics exactly -- condition every particle on the
observed scan, weight by the predictive density times the motion probability,
no epsilon floor, no routine resampling, bounded prior replay on zero support
-- and replaces its per-particle loops with the batched kernels of
:class:`~.occupancy_grid_mapping_vectorized_updater.OccupancyGridMappingVectorizedUpdater`.

The shared :class:`VectorizedWeightedParticleBelief` update runs predict then
reweight. That cannot work here: a freshly drawn scan equals the observed
one with probability zero, so every weight would be ``-inf``. ``update`` is
therefore overridden to run the conditional cycle instead.

Classes:
    OccupancyGridMappingVectorizedBelief: Batched conditional whole-map filter.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_beliefs.occupancy_grid_mapping_vectorized_updater import (
    OccupancyGridMappingVectorizedUpdater,
)
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_pomdp import (
        OccupancyGridMappingPOMDP,
    )

#: Fresh prior maps tested per replay attempt, and the number of attempts.
#: Together they give the 4096-proposal bound the scalar filter documents.
_REPLAY_BATCH = 512
_REPLAY_ATTEMPTS = 8


class OccupancyGridMappingVectorizedBelief(VectorizedWeightedParticleBelief):
    """Weighted whole maps, conditioned on each observed scan in one batched call.

    Same contract as the scalar ``OccupancyGridMappingBelief``: identical
    particles, weights, history and restart count for the same seed. Routine
    resampling is refused because it would drop the low-weight motion
    hypotheses the filter exists to keep.

    Attributes:
        history: Every ``(action, observation)`` conditioned on so far, kept
            for prior replay.
        support_restarts: How many times support collapsed and was rebuilt.
        pre_resample_ess: Effective sample size of the weights after the update.
        update_seconds: Wall-clock time of the last update.
        replay_proposals: Prior maps tested by the last update's replay, or 0.
        updater: The batched kernels for the environment.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
        ...     OccupancyGridMappingPOMDP,
        ... )
        >>> env = OccupancyGridMappingPOMDP()
        >>> belief = OccupancyGridMappingVectorizedBelief.initial(env, n_particles=8)
        >>> _, observation, _ = env.sample_next_step(belief.sample(), 0)
        >>> updated = belief.update(0, observation, env)
        >>> updated.particles.shape
        (8, 228)
        >>> len(updated.history)
        1
    """

    updater: OccupancyGridMappingVectorizedUpdater

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        particles: Any,
        log_weights: Any,
        updater: OccupancyGridMappingVectorizedUpdater,
        resampling: bool = False,
        ess_factor: float = 0.5,
        history: Any = (),
        support_restarts: int = 0,
    ):
        """Initialize the filter.

        Args:
            particles: State array or list of shape ``(N, state_size)``.
            log_weights: Log-weights of shape ``(N,)``. ``-inf`` is allowed;
                at least one entry must be finite.
            updater: The batched kernels for the environment.
            resampling: Must be ``False``.
            ess_factor: Kept for interface compatibility; resampling is off.
            history: ``(action, observation)`` pairs already conditioned on.
            support_restarts: Replay count carried over from earlier updates.

        Raises:
            ValueError: If resampling is requested or the weights have no
                finite support.
        """
        if resampling:
            raise ValueError(
                "occupancy filter preserves support; routine resampling is unsupported"
            )
        weights = np.asarray(log_weights, dtype=np.float64)
        if (
            np.any(np.isnan(weights))
            or np.any(np.isposinf(weights))
            or not np.any(np.isfinite(weights))
        ):
            raise ValueError("weights need finite support and no NaN or positive infinity")
        super().__init__(
            np.asarray(particles, dtype=np.float64),
            weights.copy(),
            updater,
            False,
            ess_factor,
        )
        self.history = tuple((int(a), np.asarray(o, dtype=np.float64)) for a, o in history)
        self.support_restarts = int(support_restarts)
        self.pre_resample_ess = float(1.0 / np.sum(self.normalized_weights**2))
        self.update_seconds = 0.0
        self.replay_proposals = 0

    @classmethod
    def initial(
        cls, environment: "OccupancyGridMappingPOMDP", n_particles: int = 30
    ) -> "OccupancyGridMappingVectorizedBelief":
        """Sample the whole-map prior without reading the true state.

        Args:
            environment: The environment whose prior and sensor to use.
            n_particles: Number of map hypotheses. Defaults to 30.

        Returns:
            A uniformly weighted filter over fresh prior maps.
        """
        return cls(
            np.asarray(environment.initial_state_dist().sample(int(n_particles))),
            np.full(int(n_particles), -np.log(int(n_particles))),
            OccupancyGridMappingVectorizedUpdater.from_environment(environment),
        )

    def to_dict(self) -> dict:
        """Serialize everything but the updater, which is rebuilt from the environment."""
        return {
            "particles": self.particles.tolist(),
            "log_weights": self.log_weights.tolist(),
            "resampling": self.resampling,
            "ess_factor": self.ess_factor,
            "history": [(a, o.tolist()) for a, o in self.history],
            "support_restarts": self.support_restarts,
        }

    @property
    def config_id(self) -> str:
        """Include the updater and replay history in cache identity."""
        return config_to_id(
            {
                "type": type(self).__name__,
                "version": 2,
                "base": super().config_id,
                "history": [(a, o.tolist()) for a, o in self.history],
            }
        )

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional[Environment] = None,
        state: Optional[Any] = None,
    ) -> "OccupancyGridMappingVectorizedBelief":
        """Condition every particle on the observed scan; the true state is ignored.

        Args:
            action: The action taken.
            observation: ``[row, col, heading, ranges...]``.
            pomdp: The environment. Needed only when support collapses and
                fresh prior maps must be drawn.
            state: Ignored.

        Returns:
            The conditioned filter.

        Raises:
            RuntimeError: If support collapses and ``pomdp`` is not given, or
                the bounded replay finds no map consistent with the history.
        """
        del state
        started = time.perf_counter()
        observation = np.asarray(observation, dtype=np.float64)
        weights = self.log_weights + self.updater.batch_predictive_log_likelihood(
            self.particles, action, observation
        )
        history = self.history + ((int(action), observation.copy()),)
        restarts = self.support_restarts
        replay_ess = None
        proposals = 0
        if not np.any(np.isfinite(weights)):
            if pomdp is None:
                raise RuntimeError(
                    "occupancy whole-map particle support exhausted and no environment "
                    "was given to draw fresh prior maps from"
                )
            particles, weights, replay_ess, proposals = self._replay_prior(
                cast("OccupancyGridMappingPOMDP", pomdp), history
            )
            restarts += 1
        else:
            particles = self.updater.batch_state_from_observation(self.particles, observation)
        result = type(self)(
            particles,
            weights,
            self.updater,
            False,
            self.ess_factor,
            history,
            restarts,
        )
        if replay_ess is not None:
            result.pre_resample_ess = replay_ess
        result.update_seconds = time.perf_counter() - started
        result.replay_proposals = proposals
        return result

    def _replay_prior(self, pomdp: "OccupancyGridMappingPOMDP", history):
        """Restore support from the original prior; never change a hidden map to fit.

        Draws the same candidates from the same global stream as the scalar
        filter, so a seeded run gives the same replayed particles.
        """
        for attempt in range(_REPLAY_ATTEMPTS):
            candidates = np.asarray(pomdp.initial_state_dist().sample(_REPLAY_BATCH))
            weights = np.zeros(len(candidates))
            for action, observation in history:
                alive = np.flatnonzero(np.isfinite(weights))
                if alive.size == 0:
                    break
                scores = self.updater.batch_predictive_log_likelihood(
                    candidates[alive], action, observation
                )
                weights[alive] += scores
                survivors = alive[np.isfinite(scores)]
                if survivors.size:
                    candidates[survivors] = self.updater.batch_state_from_observation(
                        candidates[survivors], observation
                    )
            finite = np.isfinite(weights)
            if np.any(finite):
                probabilities = np.exp(weights - np.max(weights))
                probabilities /= probabilities.sum()
                indices = np.random.choice(len(candidates), len(self.particles), p=probabilities)
                return (
                    candidates[indices],
                    np.full(len(indices), -np.log(len(indices))),
                    float(1.0 / np.sum(probabilities**2)),
                    (attempt + 1) * _REPLAY_BATCH,
                )
        raise RuntimeError(
            "occupancy whole-map particle support exhausted after "
            f"{_REPLAY_ATTEMPTS * _REPLAY_BATCH} prior proposals"
        )
