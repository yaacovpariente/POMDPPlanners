# SPDX-License-Identifier: MIT

from typing import Any, List

import numpy as np

from POMDPPlanners.core.belief import (
    Belief,
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.config_types import BeliefConfig
from POMDPPlanners.core.environment import Environment


def create_belief(environment: Environment, belief_config: BeliefConfig) -> Belief:
    """Create a belief instance from a belief config.

    Args:
        environment: The POMDP environment
        belief_config: BeliefConfig object for the belief

    Returns:
        An instance of the specified belief class
    """
    belief_params = belief_config.params.copy()
    n_particles = belief_params.pop("n_particles")
    particles: List[Any] = environment.initial_state_dist().sample(n_samples=n_particles)
    log_weights: np.ndarray = np.log(np.ones(n_particles) / n_particles)
    # Inject particles and log_weights into params
    belief_params["particles"] = particles
    belief_params["log_weights"] = log_weights
    # Create a new config with updated params
    updated_config = BeliefConfig(class_name=belief_config.class_name, params=belief_params)
    return Belief.from_config(updated_config)


def get_initial_belief(
    environment: Environment, n_particles: int, resampling: bool = True
) -> Belief:
    """Create initial belief from environment's initial state distribution."""
    particles: List[Any] = environment.initial_state_dist().sample(n_samples=n_particles)
    log_weights: np.ndarray = np.log(np.ones(n_particles) / n_particles)

    return WeightedParticleBelief(
        particles=particles, log_weights=log_weights, resampling=resampling
    )


class WeightedParticleBeliefDiscreteLightDark(WeightedParticleBeliefReinvigoration):
    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        resampling: bool = False,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.2,
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )

        self.reinvigoration_fraction = reinvigoration_fraction
        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

        self.actions = ["up", "down", "right", "left"]

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBeliefReinvigoration",
    ) -> "WeightedParticleBeliefReinvigoration":
        effective_sample_size = 1 / np.sum(np.square(self.normalized_weights))

        if effective_sample_size < self.ess_threshold:
            states = [observation + self.action_to_vector[action] for action in self.actions]
            states.append(observation)

            n_reinvigorate = int(self.reinvigoration_fraction * len(belief.particles))
            reinvigorate_indices = np.random.choice(len(states), size=n_reinvigorate, replace=True)
            reinvigorated_states = [states[i] for i in reinvigorate_indices]

            replace_indices = np.random.choice(
                len(belief.particles), size=n_reinvigorate, replace=True
            )
            belief.particles[: len(replace_indices)] = reinvigorated_states

        return belief


class WeightedParticleBeliefDiscreteLightDarkFullCoverage(WeightedParticleBeliefReinvigoration):
    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.05,
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=False,  # resampling is done in only the reinvigorate method
            ess_factor=ess_factor,
        )

        self.reinvigoration_particles_weights_sum = reinvigoration_fraction

        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

        self.actions = ["up", "down", "right", "left"]

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBeliefReinvigoration",
    ) -> "WeightedParticleBeliefReinvigoration":
        self.particles, self.log_weights = self._resample(
            particles=self.particles, log_weights=self.log_weights
        )

        # Recalculate normalized weights after resampling
        self.normalized_weights = np.exp(self.log_weights - np.max(self.log_weights))
        self.normalized_weights = self.normalized_weights / np.sum(self.normalized_weights)

        states = [observation + self.action_to_vector[action] for action in self.actions]
        states.append(observation)

        n_states = len(states)
        self.particles[-n_states:] = states

        return self


class WeightedParticleBeliefContinuousLightDarkFullCoverage(WeightedParticleBeliefReinvigoration):
    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.05,
        reinvigoration_cov_matrix: np.ndarray = np.eye(2),
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=False,  # resampling is done in only the reinvigorate method
            ess_factor=ess_factor,
        )

        self.reinvigoration_particles_weights_sum = reinvigoration_fraction
        self.reinvigoration_cov_matrix = reinvigoration_cov_matrix

        self.action_to_vector = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "right": np.array([1, 0]),
            "left": np.array([-1, 0]),
        }

        self.actions = ["up", "down", "right", "left"]

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBeliefReinvigoration",
    ) -> "WeightedParticleBeliefReinvigoration":
        self.particles, self.log_weights = self._resample(
            particles=self.particles, log_weights=self.log_weights
        )

        # Recalculate normalized weights after resampling
        self.normalized_weights = np.exp(self.log_weights - np.max(self.log_weights))
        self.normalized_weights = self.normalized_weights / np.sum(self.normalized_weights)

        states = [observation + self.action_to_vector[action] for action in self.actions]
        states.append(observation)

        n_reinvigorate = int(self.reinvigoration_particles_weights_sum * len(self.particles))

        if n_reinvigorate > 0:  # Only proceed if we need to reinvigorate
            # Sample centers for each particle to reinvigorate
            center_indices = np.random.randint(0, len(states), size=n_reinvigorate)
            centers = np.array([states[i] for i in center_indices])

            # Sample all particles at once
            reinvigorated_states = (
                np.random.multivariate_normal(
                    mean=np.zeros(2),  # We'll add the centers after sampling
                    cov=self.reinvigoration_cov_matrix,
                    size=n_reinvigorate,
                )
                + centers
            )

            # Clip to ensure within grid bounds
            reinvigorated_states = np.clip(reinvigorated_states, 0, pomdp.grid_size)  # type: ignore[attr-defined]

            self.particles[-n_reinvigorate:] = list(reinvigorated_states)

        return self


class WeightedParticleBeliefSanityPOMDP(WeightedParticleBeliefReinvigoration):
    def __init__(
        self,
        particles: List[Any],
        log_weights: np.ndarray,
        resampling: bool = False,
        ess_factor: float = 0.5,
        reinvigoration_fraction: float = 0.2,
    ):
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )

        self.reinvigoration_fraction = reinvigoration_fraction

    def reinvigorate(
        self, action: Any, observation: Any, pomdp: Environment, belief: Belief
    ) -> Belief:
        """Reinvigorate particles by sampling from initial state distribution."""
        n_reinvigorate = int(self.reinvigoration_fraction * len(self.particles))
        reinvigorated_states = pomdp.initial_state_dist().sample(n_samples=n_reinvigorate)

        # Create new belief with reinvigorated particles
        new_particles = self.particles + reinvigorated_states
        new_log_weights = np.concatenate(
            [self.log_weights, np.log(np.ones(n_reinvigorate) / n_reinvigorate)]
        )

        return WeightedParticleBelief(
            particles=new_particles,
            log_weights=new_log_weights,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )
