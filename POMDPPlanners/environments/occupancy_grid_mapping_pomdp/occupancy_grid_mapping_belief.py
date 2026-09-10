# SPDX-License-Identifier: MIT
"""Conditional filtering for the occupancy environment's stored continuous scan.

The proposal installs the observed scan rather than drawing an independent
scan that would match it with probability zero. Importance weights are the
predictive Gaussian density times the exact motion probability. No epsilon
floor admits impossible poses. Whole-map support can still collapse; a bounded
prior replay then reweights fresh map hypotheses against the complete history.
"""

import time
from typing import cast

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.utils.config_to_id import config_to_id


class OccupancyGridMappingBelief(WeightedParticleBelief):
    """Weighted whole maps sharing one observation-derived inverse map.

    No routine resampling is used, to preserve low-weight motion hypotheses.
    On zero support, up to 4096 fresh prior maps are tested against all past
    observations. This is importance sampling from the original prior, not an
    exact posterior sampler. Failure to find support raises explicitly.
    """

    def __init__(
        self,
        particles,
        log_weights,
        resampling=False,
        ess_factor=0.5,
        history=(),
        support_restarts=0,
    ):
        if resampling:
            raise ValueError(
                "occupancy filter preserves support; routine resampling is unsupported"
            )
        weights = np.asarray(log_weights, dtype=float)
        if (
            np.any(np.isnan(weights))
            or np.any(np.isposinf(weights))
            or not np.any(np.isfinite(weights))
        ):
            raise ValueError("weights need finite support and no NaN or positive infinity")
        # Base public validation requires finite weights; retain exact zeros after it.
        super().__init__(
            np.asarray(particles),
            np.where(np.isneginf(weights), -1e300, weights - np.max(weights) - 1),
            False,
            ess_factor,
        )
        self.log_weights = weights.copy()
        self.history = tuple((int(a), np.asarray(o, dtype=float)) for a, o in history)
        self.support_restarts = int(support_restarts)
        self.pre_resample_ess = float(1 / np.sum(self.normalized_weights**2))
        self.update_seconds = 0.0
        self.replay_proposals = 0

    def to_dict(self):
        """Preserve history needed by prior replay after a configuration round trip."""
        result = super().to_dict()
        result.update(
            history=[(a, o.tolist()) for a, o in self.history],
            support_restarts=self.support_restarts,
        )
        return result

    @classmethod
    def initial(cls, environment, n_particles=30):
        """Sample the original whole-map prior without reading the true state."""
        return cls(
            environment.initial_state_dist().sample(n_particles),
            np.full(n_particles, -np.log(n_particles)),
        )

    @property
    def config_id(self):
        """Include the updater and replay history in cache identity."""
        return config_to_id(
            {
                "type": type(self).__name__,
                "version": 2,
                "base": super().config_id,
                "history": [(a, o.tolist()) for a, o in self.history],
            }
        )

    def update(self, action, observation, pomdp, state=None):
        """Condition on observations only; the runner's true state is ignored."""
        del state
        started = time.perf_counter()
        replay_ess = None
        proposals = 0
        observation = np.asarray(observation, dtype=float)
        likelihoods = np.array(
            [
                pomdp.predictive_observation_log_probability(p, action, observation)
                for p in self.particles
            ]
        )
        weights = self.log_weights + likelihoods
        history = self.history + ((int(action), observation.copy()),)
        restarts = self.support_restarts
        particles = self.particles
        if not np.any(np.isfinite(weights)):
            particles, weights, replay_ess, proposals = self._replay_prior(pomdp, history)
            restarts += 1
            # Replay already includes the final observed map update.
            next_particles = particles
        else:
            next_particles = np.asarray(
                [pomdp.state_from_observation(p, observation) for p in particles]
            )
        result = cast(
            OccupancyGridMappingBelief,
            type(self)._from_validated_arrays(next_particles, weights, False, self.ess_factor),
        )
        result.history = history
        result.support_restarts = restarts
        result.pre_resample_ess = (
            float(1 / np.sum(result.normalized_weights**2)) if replay_ess is None else replay_ess
        )
        result.update_seconds = time.perf_counter() - started
        result.replay_proposals = proposals
        return result

    def _replay_prior(self, pomdp, history):
        """Restore support from original prior; never change a hidden map to fit."""
        for attempt in range(8):
            candidates = np.asarray(pomdp.initial_state_dist().sample(512))
            weights = np.zeros(len(candidates))
            for action, observation in history:
                alive = np.flatnonzero(np.isfinite(weights))
                if not len(alive):
                    break
                for index in alive:
                    score = pomdp.predictive_observation_log_probability(
                        candidates[index], action, observation
                    )
                    weights[index] += score
                    if np.isfinite(score):
                        candidates[index] = pomdp.state_from_observation(
                            candidates[index], observation
                        )
            finite = np.isfinite(weights)
            if np.any(finite):
                probabilities = np.exp(weights - np.max(weights))
                probabilities /= probabilities.sum()
                indices = np.random.choice(len(candidates), len(self.particles), p=probabilities)
                return (
                    candidates[indices],
                    np.full(len(indices), -np.log(len(indices))),
                    float(1 / np.sum(probabilities**2)),
                    (attempt + 1) * 512,
                )
        raise RuntimeError(
            "occupancy whole-map particle support exhausted after 4096 prior proposals"
        )
