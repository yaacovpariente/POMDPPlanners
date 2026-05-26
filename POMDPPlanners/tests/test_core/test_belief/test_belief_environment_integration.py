# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Smoke-tests for belief classes against POMDP environments.

This module verifies that the concrete belief classes work correctly with
every POMDP environment in the package. Each test constructs a belief,
performs an update, and samples from the result.

Particle beliefs (WeightedParticleBelief, WeightedParticleBeliefStateUpdate,
UnweightedParticleBeliefStateUpdate) are tested against all 11 environments.

Gaussian beliefs (GaussianBelief with linear_kalman_filter_updater,
extended_kalman_filter_updater, and unscented_kalman_filter_updater)
are tested against ContinuousLightDarkPOMDPDiscreteActions, which has
a linear-Gaussian state-space model matching the Kalman filter assumptions.

GaussianMixtureBelief is smoke-tested with a single-component mixture
on ContinuousLightDarkPOMDPDiscreteActions.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief import (
    GaussianBelief,
    GaussianMixtureBelief,
    GaussianMixtureBeliefUpdater,
    LinearKalmanFilterUpdater,
    ExtendedKalmanFilterUpdater,
    UnscentedKalmanFilterUpdater,
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
)
from POMDPPlanners.environments import (
    CartPolePOMDP,
    ContinuousLightDarkPOMDPDiscreteActions,
    DiscreteLightDarkPOMDP,
    LaserTagPOMDP,
    MountainCarPOMDP,
    PushPOMDP,
    SafeAntVelocityPOMDP,
    SanityPOMDP,
    TigerPOMDP,
)
from POMDPPlanners.environments.pacman_pomdp import create_simple_maze_pacman
from POMDPPlanners.environments.rock_sample_pomdp import create_random_rock_sample
from POMDPPlanners.tests.test_utils.metric_invariants_utils import (
    verify_belief_invariants,
)

NUM_PARTICLES = 10
NUM_STATE_UPDATE_PARTICLES = 3


def _make_tiger():
    return TigerPOMDP(discount_factor=0.95)


def _make_sanity():
    return SanityPOMDP(discount_factor=0.95)


def _make_cartpole():
    return CartPolePOMDP(discount_factor=0.95, noise_cov=np.eye(4) * 0.1)


def _make_mountain_car():
    return MountainCarPOMDP(discount_factor=0.95)


def _make_continuous_light_dark():
    return ContinuousLightDarkPOMDPDiscreteActions(discount_factor=0.95)


def _make_discrete_light_dark():
    return DiscreteLightDarkPOMDP(discount_factor=0.95)


def _make_push():
    return PushPOMDP(discount_factor=0.95)


def _make_laser_tag():
    return LaserTagPOMDP(discount_factor=0.95)


def _make_pacman():
    return create_simple_maze_pacman(seed=42)


def _make_rock_sample():
    return create_random_rock_sample(num_rocks=3, seed=42)


def _make_safe_ant_velocity():
    return SafeAntVelocityPOMDP(discount_factor=0.95)


ENV_FACTORIES = [
    ("TigerPOMDP", _make_tiger),
    ("SanityPOMDP", _make_sanity),
    ("CartPolePOMDP", _make_cartpole),
    ("MountainCarPOMDP", _make_mountain_car),
    ("ContinuousLightDarkDiscreteActions", _make_continuous_light_dark),
    ("DiscreteLightDarkPOMDP", _make_discrete_light_dark),
    ("PushPOMDP", _make_push),
    ("LaserTagPOMDP", _make_laser_tag),
    ("PacManPOMDP", _make_pacman),
    ("RockSamplePOMDP", _make_rock_sample),
    ("SafeAntVelocityPOMDP", _make_safe_ant_velocity),
]


def _simulate_one_step(env):
    """Sample particles, pick an action, and simulate one environment step.

    Returns:
        Tuple of (particles, action, observation, next_states) where
        particles and next_states are lists of length NUM_PARTICLES.
    """
    particles = env.initial_state_dist().sample(n_samples=NUM_PARTICLES)
    action = env.get_actions()[0]
    state = particles[0]
    next_state = env.sample_next_state(state, action)
    observation = env.sample_observation(next_state, action)
    next_states = [env.sample_next_state(p, action) for p in particles]
    return particles, action, observation, next_states


class TestWeightedParticleBeliefAllEnvironments:
    """Smoke-test WeightedParticleBelief.update against every environment."""

    @pytest.mark.parametrize(
        "_env_name,env_factory", ENV_FACTORIES, ids=[e[0] for e in ENV_FACTORIES]
    )
    def test_smoke(self, _env_name, env_factory):
        """Smoke-test WeightedParticleBelief construct-update-sample cycle.

        Purpose: Validates that WeightedParticleBelief can be constructed,
            updated, and sampled for every environment without errors.

        Given: A POMDP environment with sampled initial particles and
            uniform log-weights.
        When: A WeightedParticleBelief is created and updated with an
            action-observation pair from a simulated step.
        Then: The updated belief is a WeightedParticleBelief, sampling
            succeeds, and the particle count is preserved.

        Test type: integration
        """
        np.random.seed(42)
        env = env_factory()
        particles, action, observation, _ = _simulate_one_step(env)

        log_weights = np.full(len(particles), -np.log(len(particles)))
        belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
        verify_belief_invariants(belief, expected_n_particles=len(particles))

        updated = belief.update(action=action, observation=observation, pomdp=env)

        assert isinstance(updated, WeightedParticleBelief)
        assert len(updated.particles) == len(particles)
        sample = updated.sample()
        assert sample is not None
        verify_belief_invariants(updated, expected_n_particles=len(particles))


class TestWeightedParticleBeliefStateUpdateAllEnvironments:
    """Smoke-test WeightedParticleBeliefStateUpdate.update against every environment."""

    @pytest.mark.parametrize(
        "_env_name,env_factory", ENV_FACTORIES, ids=[e[0] for e in ENV_FACTORIES]
    )
    def test_smoke(self, _env_name, env_factory):
        """Smoke-test WeightedParticleBeliefStateUpdate incremental update cycle.

        Purpose: Validates that WeightedParticleBeliefStateUpdate can be
            incrementally built by adding state particles for every
            environment without errors.

        Given: A POMDP environment with simulated next-states from an
            action-observation step.
        When: An empty WeightedParticleBeliefStateUpdate is created and
            updated by adding state particles one at a time.
        Then: The final belief has the correct number of particles,
            positive weights_sum, correct type, and can be sampled.

        Test type: integration
        """
        np.random.seed(42)
        env = env_factory()
        _, action, observation, next_states = _simulate_one_step(env)

        belief = WeightedParticleBeliefStateUpdate()
        for state in next_states[:NUM_STATE_UPDATE_PARTICLES]:
            belief = belief.update(
                action=action,
                observation=observation,
                pomdp=env,
                state=state,
            )

        assert isinstance(belief, WeightedParticleBeliefStateUpdate)
        assert len(belief.particles) == NUM_STATE_UPDATE_PARTICLES
        assert belief.weights_sum > 0
        sample = belief.sample()
        assert sample is not None


class TestUnweightedParticleBeliefStateUpdateAllEnvironments:
    """Smoke-test UnweightedParticleBeliefStateUpdate.update against every environment."""

    @pytest.mark.parametrize(
        "_env_name,env_factory", ENV_FACTORIES, ids=[e[0] for e in ENV_FACTORIES]
    )
    def test_smoke(self, _env_name, env_factory):
        """Smoke-test UnweightedParticleBeliefStateUpdate incremental update cycle.

        Purpose: Validates that UnweightedParticleBeliefStateUpdate can be
            incrementally built by adding state particles for every
            environment without errors.

        Given: A POMDP environment with simulated next-states from an
            action-observation step.
        When: An empty UnweightedParticleBeliefStateUpdate is created and
            updated by adding state particles one at a time.
        Then: The final belief has the correct number of particles,
            weights_sum equal to particle count, correct type, and can
            be sampled.

        Test type: integration
        """
        np.random.seed(42)
        env = env_factory()
        _, action, observation, next_states = _simulate_one_step(env)

        belief = UnweightedParticleBeliefStateUpdate()
        for state in next_states[:NUM_STATE_UPDATE_PARTICLES]:
            belief = belief.update(
                action=action,
                observation=observation,
                pomdp=env,
                state=state,
            )

        assert isinstance(belief, UnweightedParticleBeliefStateUpdate)
        assert len(belief.particles) == NUM_STATE_UPDATE_PARTICLES
        assert belief.weights_sum == NUM_STATE_UPDATE_PARTICLES
        sample = belief.sample()
        assert sample is not None
        verify_belief_invariants(belief, expected_n_particles=NUM_STATE_UPDATE_PARTICLES)


# ---------------------------------------------------------------------------
# Gaussian belief tests against ContinuousLightDarkPOMDPDiscreteActions
# ---------------------------------------------------------------------------

ACTION_TO_VECTOR = {
    "up": np.array([0.0, 1.0]),
    "down": np.array([0.0, -1.0]),
    "right": np.array([1.0, 0.0]),
    "left": np.array([-1.0, 0.0]),
}

NUM_GAUSSIAN_UPDATE_STEPS = 5


class _SingleComponentKFUpdater(GaussianMixtureBeliefUpdater):
    """GMM updater that applies a single KF updater to the first component."""

    def __init__(self, kf_updater):
        self._kf = kf_updater

    def update(self, means, covariances, weights, action, observation):
        new_mean, new_cov = self._kf.update(
            means[0], covariances[0], ACTION_TO_VECTOR[action], observation
        )
        return [new_mean], [new_cov], weights

    @property
    def config_id(self) -> str:
        return f"single_component_kf_{self._kf.config_id}"


def _make_continuous_light_dark_with_cov():
    Q = 0.01 * np.eye(2)
    R = 0.1 * np.eye(2)
    env = ContinuousLightDarkPOMDPDiscreteActions(
        discount_factor=0.95,
        state_transition_cov_matrix=Q,
        observation_cov_matrix=R,
    )
    return env, Q, R


def _simulate_trajectory(env, n_steps):
    state = env.initial_state_dist().sample()[0]
    action = env.get_actions()[0]
    actions = []
    observations = []
    for _ in range(n_steps):
        next_state = env.sample_next_state(state, action)
        observation = env.sample_observation(next_state, action)
        actions.append(action)
        observations.append(observation)
        state = next_state
    return actions, observations


class TestGaussianBeliefLinearKalmanFilterContinuousLightDark:
    """Smoke-test GaussianBelief with linear Kalman filter on ContinuousLightDark."""

    def test_smoke(self):
        """Smoke-test GaussianBelief with linear KF over multiple update steps.

        Purpose: Validates that GaussianBelief with a linear Kalman filter
            updater can be constructed, updated over multiple steps, and
            sampled using the ContinuousLightDarkPOMDPDiscreteActions
            environment.

        Given: A ContinuousLightDarkPOMDPDiscreteActions environment with
            known process and observation noise, and a GaussianBelief
            initialized at the start state with a linear Kalman filter
            updater.
        When: The belief is updated over multiple action-observation steps
            from a simulated trajectory.
        Then: The updated belief is a GaussianBelief, has 2D mean and
            2x2 covariance, covariance is symmetric, and sampling succeeds.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        updater = LinearKalmanFilterUpdater(A=np.eye(2), B=np.eye(2), H=np.eye(2), Q=Q, R=R)
        belief = GaussianBelief(
            mean=env.start_state.copy(),
            covariance=np.eye(2),
            updater=updater,
        )

        for action, obs in zip(actions, observations):
            belief = belief.update(action=ACTION_TO_VECTOR[action], observation=obs, pomdp=env)

        assert isinstance(belief, GaussianBelief)
        assert belief.mean.shape == (2,)
        assert belief.covariance.shape == (2, 2)
        np.testing.assert_allclose(belief.covariance, belief.covariance.T, atol=1e-10)
        sample = belief.sample()
        assert sample.shape == (2,)

    def test_covariance_shrinks(self):
        """Test that Kalman filter covariance decreases with observations.

        Purpose: Validates that the linear Kalman filter reduces uncertainty
            as observations accumulate.

        Given: A GaussianBelief with identity covariance and a linear
            Kalman filter updater on ContinuousLightDarkPOMDPDiscreteActions.
        When: The belief is updated over multiple steps.
        Then: The trace of the posterior covariance is smaller than the
            trace of the prior covariance.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        updater = LinearKalmanFilterUpdater(A=np.eye(2), B=np.eye(2), H=np.eye(2), Q=Q, R=R)
        initial_cov = np.eye(2)
        belief = GaussianBelief(
            mean=env.start_state.copy(),
            covariance=initial_cov,
            updater=updater,
        )

        for action, obs in zip(actions, observations):
            belief = belief.update(action=ACTION_TO_VECTOR[action], observation=obs, pomdp=env)

        assert np.trace(belief.covariance) < np.trace(initial_cov)


class TestGaussianBeliefExtendedKalmanFilterContinuousLightDark:
    """Smoke-test GaussianBelief with extended Kalman filter on ContinuousLightDark."""

    def test_smoke(self):
        """Smoke-test GaussianBelief with EKF over multiple update steps.

        Purpose: Validates that GaussianBelief with an extended Kalman
            filter updater can be constructed, updated over multiple steps,
            and sampled using ContinuousLightDarkPOMDPDiscreteActions.

        Given: A ContinuousLightDarkPOMDPDiscreteActions environment with
            known process and observation noise, and a GaussianBelief
            initialized at the start state with an EKF updater using
            identity transition and observation functions.
        When: The belief is updated over multiple action-observation steps
            from a simulated trajectory.
        Then: The updated belief is a GaussianBelief, has 2D mean and
            2x2 covariance, covariance is symmetric, and sampling succeeds.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x + u,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=Q,
            R=R,
        )
        belief = GaussianBelief(
            mean=env.start_state.copy(),
            covariance=np.eye(2),
            updater=updater,
        )

        for action, obs in zip(actions, observations):
            belief = belief.update(action=ACTION_TO_VECTOR[action], observation=obs, pomdp=env)

        assert isinstance(belief, GaussianBelief)
        assert belief.mean.shape == (2,)
        assert belief.covariance.shape == (2, 2)
        np.testing.assert_allclose(belief.covariance, belief.covariance.T, atol=1e-10)
        sample = belief.sample()
        assert sample.shape == (2,)

    def test_ekf_matches_linear_kf(self):
        """Test that EKF with linear functions matches linear Kalman filter.

        Purpose: Validates that the EKF produces identical results to the
            linear KF when the system is linear (identity functions).

        Given: Both a linear KF and an EKF configured with identity
            transition/observation functions on the same environment.
        When: Both are updated with the same action-observation trajectory.
        Then: The resulting means and covariances are numerically equal.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        lkf_updater = LinearKalmanFilterUpdater(A=np.eye(2), B=np.eye(2), H=np.eye(2), Q=Q, R=R)
        ekf_updater = ExtendedKalmanFilterUpdater(
            transition_fn=lambda x, u: x + u,
            observation_fn=lambda x: x,
            transition_jacobian=lambda x, u: np.eye(len(x)),
            observation_jacobian=lambda x: np.eye(len(x)),
            Q=Q,
            R=R,
        )

        start_mean = env.start_state.copy()
        start_cov = np.eye(2)
        lkf_belief = GaussianBelief(mean=start_mean, covariance=start_cov, updater=lkf_updater)
        ekf_belief = GaussianBelief(
            mean=start_mean.copy(), covariance=start_cov.copy(), updater=ekf_updater
        )

        for action, obs in zip(actions, observations):
            action_vec = ACTION_TO_VECTOR[action]
            lkf_belief = lkf_belief.update(action=action_vec, observation=obs, pomdp=env)
            ekf_belief = ekf_belief.update(action=action_vec, observation=obs, pomdp=env)

        np.testing.assert_allclose(lkf_belief.mean, ekf_belief.mean, atol=1e-10)
        np.testing.assert_allclose(lkf_belief.covariance, ekf_belief.covariance, atol=1e-10)


# ---------------------------------------------------------------------------
# UKF belief tests against ContinuousLightDarkPOMDPDiscreteActions
# ---------------------------------------------------------------------------


class TestGaussianBeliefUnscentedKalmanFilterContinuousLightDark:
    """Smoke-test GaussianBelief with unscented Kalman filter on ContinuousLightDark."""

    def test_smoke(self):
        """Smoke-test GaussianBelief with UKF over multiple update steps.

        Purpose: Validates that GaussianBelief with an unscented Kalman
            filter updater can be constructed, updated over multiple steps,
            and sampled using ContinuousLightDarkPOMDPDiscreteActions.

        Given: A ContinuousLightDarkPOMDPDiscreteActions environment with
            known process and observation noise, and a GaussianBelief
            initialized at the start state with a UKF updater using
            identity transition and observation functions.
        When: The belief is updated over multiple action-observation steps.
        Then: The updated belief is a GaussianBelief, has 2D mean and
            2x2 covariance, covariance is symmetric, and sampling succeeds.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x + u,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
        )
        belief = GaussianBelief(
            mean=env.start_state.copy(),
            covariance=np.eye(2),
            updater=updater,
        )

        for action, obs in zip(actions, observations):
            belief = belief.update(action=ACTION_TO_VECTOR[action], observation=obs, pomdp=env)

        assert isinstance(belief, GaussianBelief)
        assert belief.mean.shape == (2,)
        assert belief.covariance.shape == (2, 2)
        np.testing.assert_allclose(belief.covariance, belief.covariance.T, atol=1e-10)
        sample = belief.sample()
        assert sample.shape == (2,)

    def test_ukf_matches_linear_kf(self):
        """Test that UKF with linear functions matches linear Kalman filter.

        Purpose: Validates that the UKF produces identical results to the
            linear KF when the system is linear.

        Given: Both a linear KF and a UKF configured with linear
            transition/observation functions on the same environment.
        When: Both are updated with the same action-observation trajectory.
        Then: The resulting means and covariances are numerically close.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        lkf_updater = LinearKalmanFilterUpdater(A=np.eye(2), B=np.eye(2), H=np.eye(2), Q=Q, R=R)
        ukf_updater = UnscentedKalmanFilterUpdater(
            transition_fn=lambda x, u: x + u,
            observation_fn=lambda x: x,
            Q=Q,
            R=R,
        )

        start_mean = env.start_state.copy()
        start_cov = np.eye(2)
        lkf_belief = GaussianBelief(mean=start_mean, covariance=start_cov, updater=lkf_updater)
        ukf_belief = GaussianBelief(
            mean=start_mean.copy(), covariance=start_cov.copy(), updater=ukf_updater
        )

        for action, obs in zip(actions, observations):
            action_vec = ACTION_TO_VECTOR[action]
            lkf_belief = lkf_belief.update(action=action_vec, observation=obs, pomdp=env)
            ukf_belief = ukf_belief.update(action=action_vec, observation=obs, pomdp=env)

        np.testing.assert_allclose(lkf_belief.mean, ukf_belief.mean, atol=1e-6)
        np.testing.assert_allclose(lkf_belief.covariance, ukf_belief.covariance, atol=1e-6)


# ---------------------------------------------------------------------------
# Gaussian Mixture Belief tests against ContinuousLightDarkPOMDPDiscreteActions
# ---------------------------------------------------------------------------


class TestGaussianMixtureBeliefContinuousLightDark:
    """Smoke-test GaussianMixtureBelief on ContinuousLightDark."""

    def test_smoke_single_component(self):
        """Smoke-test single-component GaussianMixtureBelief on ContinuousLightDark.

        Purpose: Validates that a single-component GaussianMixtureBelief can
            be constructed, updated, and sampled using a real POMDP environment.

        Given: A ContinuousLightDarkPOMDPDiscreteActions environment and a
            single-component GMM using a KF-based updater wrapped for GMM.
        When: The belief is updated over multiple steps.
        Then: The updated belief is a GaussianMixtureBelief, has correct
            dimensions, and sampling succeeds.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        kf_updater = LinearKalmanFilterUpdater(A=np.eye(2), B=np.eye(2), H=np.eye(2), Q=Q, R=R)
        gmm_updater = _SingleComponentKFUpdater(kf_updater)

        belief = GaussianMixtureBelief(
            means=[env.start_state.copy()],
            covariances=[np.eye(2)],
            weights=np.array([1.0]),
            updater=gmm_updater,
        )

        for action, obs in zip(actions, observations):
            belief = belief.update(action=action, observation=obs, pomdp=env)

        assert isinstance(belief, GaussianMixtureBelief)
        assert belief.n_components == 1
        assert belief.dim == 2
        assert belief.means[0].shape == (2,)
        assert belief.covariances[0].shape == (2, 2)
        sample = belief.sample()
        assert sample.shape == (2,)

    def test_single_component_matches_gaussian_belief(self):
        """Test single-component GMM matches GaussianBelief on the same trajectory.

        Purpose: Validates that a single-component GMM produces the same
            posterior as GaussianBelief using the same update logic.

        Given: Both a GaussianBelief and a single-component GMM initialized
            identically on ContinuousLightDarkPOMDPDiscreteActions.
        When: Both are updated with the same trajectory.
        Then: The posterior means and covariances match.

        Test type: integration
        """
        np.random.seed(42)
        env, Q, R = _make_continuous_light_dark_with_cov()
        actions, observations = _simulate_trajectory(env, NUM_GAUSSIAN_UPDATE_STEPS)

        kf_updater = LinearKalmanFilterUpdater(A=np.eye(2), B=np.eye(2), H=np.eye(2), Q=Q, R=R)

        gaussian_belief = GaussianBelief(
            mean=env.start_state.copy(),
            covariance=np.eye(2),
            updater=kf_updater,
        )

        gmm_updater = _SingleComponentKFUpdater(kf_updater)

        gmm_belief = GaussianMixtureBelief(
            means=[env.start_state.copy()],
            covariances=[np.eye(2)],
            weights=np.array([1.0]),
            updater=gmm_updater,
        )

        for action, obs in zip(actions, observations):
            gaussian_belief = gaussian_belief.update(
                action=ACTION_TO_VECTOR[action], observation=obs, pomdp=env
            )
            gmm_belief = gmm_belief.update(action=action, observation=obs, pomdp=env)

        np.testing.assert_allclose(gmm_belief.means[0], gaussian_belief.mean, atol=1e-10)
        np.testing.assert_allclose(
            gmm_belief.covariances[0], gaussian_belief.covariance, atol=1e-10
        )
