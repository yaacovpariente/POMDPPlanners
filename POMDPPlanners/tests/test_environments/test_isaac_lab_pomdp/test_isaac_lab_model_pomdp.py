# SPDX-License-Identifier: MIT

"""Unit tests for the IsaacLab planner-side generative model.

These tests exercise :class:`GaussianObservationModel` and
:class:`IsaacLabModelPOMDP` as pure-numpy components — they never launch Isaac
Sim, so they run in the plain project venv and in CI. The additive-Normal
observation model is checked for numerical agreement against a closed-form
diagonal-Gaussian reference; the generative model is checked for its full
abstract-method surface plus a POMCPOW + particle-belief integration round-trip.
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.environments.isaac_lab_pomdp import (
    GaussianObservationModel,
    GaussianRandomWalkTransition,
    IsaacLabModelPOMDP,
    IsaacLabSimulatorTransition,
    LinearGaussianTransition,
    LinearRewardModel,
)
from POMDPPlanners.environments.isaac_lab_pomdp import isaac_lab_model_pomdp
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler


def _reference_diagonal_gaussian_logpdf(
    values: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Closed-form log-density of a diagonal Gaussian, independent of the model impl."""
    rows = np.atleast_2d(np.asarray(values, dtype=float))
    variance = np.asarray(std, dtype=float) ** 2
    quadratic = np.sum((rows - mean) ** 2 / variance, axis=1)
    normalizer = np.sum(np.log(2.0 * np.pi * variance))
    return -0.5 * (quadratic + normalizer)


def _build_model(observation_dim: int = 5, action_dim: int = 3) -> IsaacLabModelPOMDP:
    """Construct a small model with a zero action plus random preset vectors."""
    rng = np.random.default_rng(0)
    presets = [np.zeros(action_dim)] + [rng.standard_normal(action_dim) for _ in range(4)]
    return IsaacLabModelPOMDP(
        observation_dim=observation_dim,
        action_presets=presets,
        discount_factor=0.99,
        observation_noise_std=0.1,
        process_noise_std=0.05,
    )


def test_observation_log_probability_matches_scipy_single_and_batch():
    """Additive-Normal log-density agrees with scipy for one and many observations.

    Purpose: Validates the observation model is a correct isotropic-per-channel Gaussian.

    Given: A GaussianObservationModel with per-channel std and a fixed state.
    When: log_probability is evaluated on one observation and on a batch.
    Then: Values equal scipy.multivariate_normal.logpdf with the same diagonal covariance.

    Test type: unit
    """
    rng = np.random.default_rng(1)
    dim = 5
    std = np.array([0.1, 0.2, 0.05, 0.3, 0.15])
    model = GaussianObservationModel(dim, std)
    state = rng.standard_normal(dim)

    obs = rng.standard_normal(dim)
    single = model.log_probability(state, obs)
    assert single.shape == (1,)
    assert np.allclose(single[0], _reference_diagonal_gaussian_logpdf(obs, state, std)[0])

    batch = rng.standard_normal((4, dim))
    assert np.allclose(
        model.log_probability(state, batch),
        _reference_diagonal_gaussian_logpdf(batch, state, std),
    )


def test_observation_sampling_shapes_and_mean_centered_on_state():
    """Sampled observations have the right shape and are centered on the state.

    Purpose: Validates the observation model draws obs = state + zero-mean noise.

    Given: A GaussianObservationModel and a fixed state.
    When: A single sample and a large batch are drawn.
    Then: Single sample is (dim,), batch is (n, dim), and the empirical mean ~ state.

    Test type: unit
    """
    dim = 5
    model = GaussianObservationModel(dim, 0.1)
    state = np.arange(dim, dtype=float)
    assert model.sample(state).shape == (dim,)
    samples = model.sample(state, n_samples=20000)
    assert samples.shape == (20000, dim)
    assert np.allclose(samples.mean(axis=0), state, atol=0.02)


def test_observation_model_rejects_nonpositive_std():
    """A non-positive noise std is rejected at construction.

    Purpose: Validates covariance construction guards against degenerate noise.

    Given: An observation dimension and a zero (or negative) standard deviation.
    When: GaussianObservationModel is constructed.
    Then: A ValueError is raised.

    Test type: unit
    """
    with pytest.raises(ValueError):
        GaussianObservationModel(3, 0.0)
    with pytest.raises(ValueError):
        GaussianObservationModel(3, np.array([0.1, -0.2, 0.3]))


def test_model_transition_is_gaussian_random_walk_centered_on_state():
    """The placeholder transition samples a Gaussian random walk about the state.

    Purpose: Validates sample_next_state / transition_log_probability shapes and centering.

    Given: An IsaacLabModelPOMDP and a fixed state.
    When: Next states are sampled (single and batch) and scored.
    Then: Shapes are correct and the empirical next-state mean ~ the input state.

    Test type: unit
    """
    model = _build_model()
    state = np.arange(model.observation_dim, dtype=float)
    action = model.get_actions()[1]
    assert model.sample_next_state(state, action).shape == (model.observation_dim,)
    batch = model.sample_next_state(state, action, n_samples=5000)
    assert batch.shape == (5000, model.observation_dim)
    assert np.allclose(batch.mean(axis=0), state, atol=0.02)
    log_probs = model.transition_log_probability(state, action, batch[:4])
    assert log_probs.shape == (4,)


def test_model_observation_surface_shapes():
    """The model's observation methods return the belief-filter-expected shapes.

    Purpose: Validates sample_observation and the scalar/per-state density helpers.

    Given: An IsaacLabModelPOMDP, a state, and an observation.
    When: The observation and its densities (single, batch, per-state) are computed.
    Then: All shapes/types match what the particle belief and POMCPOW expect.

    Test type: unit
    """
    model = _build_model()
    state = np.arange(model.observation_dim, dtype=float)
    action = model.get_actions()[1]
    obs = model.sample_observation(state, action)
    assert obs.shape == (model.observation_dim,)
    assert model.observation_log_probability(state, action, [obs, obs]).shape == (2,)
    assert isinstance(model.observation_log_probability_single(state, action, obs), float)
    per_state = model.observation_log_probability_per_state([state, state, state], action, obs)
    assert per_state.shape == (3,)
    assert len(model.sample_next_state_batch([state, state], action)) == 2


def test_linear_reward_recovers_known_reward_and_drives_the_model():
    """A fitted linear reward recovers the true reward and is used by the model.

    Purpose: Validates LinearRewardModel.fit and its wiring into IsaacLabModelPOMDP.reward.

    Given: Transitions labeled by a known reward = w_s.state + w_a.action + w_n.next + b.
    When: LinearRewardModel.fit is applied and injected into a model.
    Then: The model's reward reproduces the true reward for held-out transitions.

    Test type: integration
    """
    rng = np.random.default_rng(6)
    dim, action_dim, n = 4, 2, 300
    w_s = rng.standard_normal(dim)
    w_a = rng.standard_normal(action_dim)
    w_n = rng.standard_normal(dim)
    bias = 0.7
    states = rng.standard_normal((n, dim))
    actions = rng.standard_normal((n, action_dim))
    next_states = rng.standard_normal((n, dim))
    rewards = states @ w_s + actions @ w_a + next_states @ w_n + bias

    reward_model = LinearRewardModel.fit(states, actions, next_states, rewards, regularization=0.0)
    model = IsaacLabModelPOMDP(
        observation_dim=dim,
        action_presets=[np.zeros(action_dim), np.ones(action_dim)],
        discount_factor=0.99,
        reward_model=reward_model,
    )
    test_s, test_a, test_n = states[0], actions[0], next_states[0]
    expected = float(test_s @ w_s + test_a @ w_a + test_n @ w_n + bias)
    assert np.isclose(model.reward(test_s, test_a, test_n), expected, atol=1e-6)


def test_model_reward_is_flat_zero_without_a_reward_model():
    """Without a reward model the model reward stays a flat zero (undirected planning).

    Purpose: Validates the documented default when no reward model is supplied.

    Given: An IsaacLabModelPOMDP built without a reward_model.
    When: reward is evaluated on an arbitrary transition.
    Then: It returns 0.0.

    Test type: unit
    """
    model = _build_model()
    state = np.ones(model.observation_dim)
    assert model.reward(state, model.get_actions()[1], state) == 0.0


def test_model_static_semantics():
    """Reward, terminality, actions, hashing, and equality behave as specified.

    Purpose: Validates the remaining concrete Environment surface.

    Given: An IsaacLabModelPOMDP with five action presets.
    When: reward, is_terminal, get_actions, hash_action, and is_equal_observation are called.
    Then: Reward is the flat placeholder 0.0, states are never terminal, and the
        five presets are returned with hashable actions and array-equality on observations.

    Test type: unit
    """
    model = _build_model()
    state = np.zeros(model.observation_dim)
    action = model.get_actions()[2]
    assert model.reward(state, action, state) == 0.0
    assert model.is_terminal(state) is False
    assert len(model.get_actions()) == 5
    assert isinstance(model.hash_action(action), bytes)
    assert model.is_equal_observation(state, state.copy())
    assert not model.is_equal_observation(state, state + 1.0)


def test_model_has_no_initial_priors():
    """The model exposes no initial-state/observation prior (belief seeded externally).

    Purpose: Validates the documented contract that the belief is seeded from the world.

    Given: An IsaacLabModelPOMDP.
    When: initial_state_dist / initial_observation_dist are called.
    Then: Both raise NotImplementedError.

    Test type: unit
    """
    model = _build_model()
    with pytest.raises(NotImplementedError):
        model.initial_state_dist()
    with pytest.raises(NotImplementedError):
        model.initial_observation_dist()


def test_random_walk_transition_is_action_ignoring_and_centered():
    """The random-walk transition centers on the state and ignores the action.

    Purpose: Validates GaussianRandomWalkTransition sampling and action-invariance.

    Given: A GaussianRandomWalkTransition and a fixed state.
    When: Next states are sampled under two different actions.
    Then: Both empirical means equal the state (the action has no effect).

    Test type: unit
    """
    transition = GaussianRandomWalkTransition(dim=4, process_noise_std=0.05)
    state = np.array([1.0, -2.0, 3.0, 0.5])
    mean_a = transition.sample_next_state(state, action=np.zeros(3), n_samples=8000).mean(axis=0)
    mean_b = transition.sample_next_state(state, action=np.ones(3), n_samples=8000).mean(axis=0)
    assert np.allclose(mean_a, state, atol=0.02)
    assert np.allclose(mean_b, state, atol=0.02)


def test_linear_transition_recovers_known_dynamics_from_rollouts():
    """Fitting recovers the true A, B, b of a noise-free linear system.

    Purpose: Validates LinearGaussianTransition.fit performs correct system identification.

    Given: Rollouts generated by a known next = A@state + B@action + b (no noise).
    When: LinearGaussianTransition.fit is applied to the rollouts.
    Then: The fitted mean reproduces the true next state for held-out (state, action) pairs.

    Test type: unit
    """
    rng = np.random.default_rng(3)
    dim, action_dim, n = 4, 2, 400
    true_a = rng.standard_normal((dim, dim)) * 0.3
    true_b = rng.standard_normal((dim, action_dim)) * 0.5
    true_bias = rng.standard_normal(dim) * 0.1
    states = rng.standard_normal((n, dim))
    actions = rng.standard_normal((n, action_dim))
    next_states = states @ true_a.T + actions @ true_b.T + true_bias

    transition = LinearGaussianTransition.fit(states, actions, next_states, regularization=0.0)
    test_state = rng.standard_normal(dim)
    test_action = rng.standard_normal(action_dim)
    expected = true_a @ test_state + true_b @ test_action + true_bias
    got = transition.sample_next_state(test_state, test_action, n_samples=6000).mean(axis=0)
    assert np.allclose(got, expected, atol=0.01)


def test_linear_transition_is_action_conditioned():
    """Different actions yield different predicted next states under the linear model.

    Purpose: Validates the learned transition makes actions move the state (unlike the walk).

    Given: A LinearGaussianTransition fit on action-dependent rollouts.
    When: The transition log-density scores a next state under two different actions.
    Then: The two log-densities differ, i.e. the action changes the predicted distribution.

    Test type: unit
    """
    rng = np.random.default_rng(4)
    dim, action_dim, n = 3, 2, 300
    true_b = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    states = rng.standard_normal((n, dim))
    actions = rng.standard_normal((n, action_dim))
    next_states = states + actions @ true_b.T
    transition = LinearGaussianTransition.fit(states, actions, next_states)

    state = np.zeros(dim)
    candidate = np.array([1.0, 0.0, 0.5])
    log_p_a = transition.log_probability(state, np.array([1.0, 0.0]), candidate)
    log_p_b = transition.log_probability(state, np.array([-1.0, 0.0]), candidate)
    assert not np.isclose(log_p_a[0], log_p_b[0])


def test_linear_transition_fit_rejects_too_few_transitions():
    """Fitting with fewer than two transitions raises.

    Purpose: Validates the fit guard against under-determined system identification.

    Given: A single transition sample.
    When: LinearGaussianTransition.fit is called.
    Then: A ValueError is raised.

    Test type: unit
    """
    with pytest.raises(ValueError):
        LinearGaussianTransition.fit(np.zeros((1, 3)), np.zeros((1, 2)), np.zeros((1, 3)))


def test_model_accepts_injected_transition():
    """An injected transition overrides the default random walk in the model.

    Purpose: Validates the TransitionModel seam is wired through IsaacLabModelPOMDP.

    Given: An IsaacLabModelPOMDP constructed with an explicit LinearGaussianTransition.
    When: sample_next_state is called for two different actions from the same state.
    Then: The predicted next-state means differ, proving the injected model is used.

    Test type: integration
    """
    rng = np.random.default_rng(5)
    dim, action_dim, n = 3, 3, 200
    states = rng.standard_normal((n, dim))
    actions = rng.standard_normal((n, action_dim))
    next_states = states + actions  # action directly displaces the state
    transition = LinearGaussianTransition.fit(states, actions, next_states)
    presets = [np.zeros(action_dim), np.ones(action_dim)]
    model = IsaacLabModelPOMDP(
        observation_dim=dim, action_presets=presets, discount_factor=0.99, transition=transition
    )
    state = np.zeros(dim)
    mean_zero = model.sample_next_state(state, presets[0], n_samples=6000).mean(axis=0)
    mean_one = model.sample_next_state(state, presets[1], n_samples=6000).mean(axis=0)
    assert np.allclose(mean_zero, np.zeros(dim), atol=0.05)
    assert np.allclose(mean_one, np.ones(dim), atol=0.05)


def test_pomcpow_plans_and_belief_updates_on_the_model():
    """POMCPOW selects a preset action and the particle belief updates on the model.

    Purpose: Validates the model is a drop-in generative model for POMCPOW + belief.

    Given: An IsaacLabModelPOMDP, a jittered particle belief, and a POMCPOW planner.
    When: The planner selects an action and the belief is updated with an observation.
    Then: The action is one of the model's presets and the update returns a new belief.

    Test type: integration
    """
    np.random.seed(0)
    model = _build_model()
    dim = model.observation_dim
    particles = [np.random.normal(0.0, 0.1, size=dim) for _ in range(50)]
    log_weights = np.log(np.ones(50) / 50)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=True)
    actions = model.get_actions()
    planner = POMCPOW(
        environment=model,
        discount_factor=0.99,
        depth=8,
        exploration_constant=10.0,
        k_o=4.0,
        alpha_o=0.1,
        k_a=float(len(actions)),
        alpha_a=0.0,
        name="POMCPOW-test",
        action_sampler=DiscreteActionSampler(actions),
        time_out_in_seconds=1,
    )
    selected, _ = planner.action(belief)
    action = np.asarray(selected[0], dtype=float).reshape(-1)
    assert any(np.array_equal(action, preset) for preset in actions)

    updated = belief.update(
        action=action,
        observation=np.random.normal(0.0, 1.0, size=dim),
        pomdp=model,
        state=np.zeros(dim),
    )
    assert updated is not belief


class _FakeSimEnv:
    """Minimal fake IsaacLab env: remembers the written state and last action."""

    def __init__(self, dynamics):
        self._dynamics = dynamics
        self.written = None
        self.last_action = None
        self.steps = 0

    def step(self, action):
        self.last_action = action
        self.steps += 1
        return None, 0.0, False, False, {}


def _build_simulator_transition(monkeypatch, dynamics, dim, **kwargs):
    """Wire an IsaacLabSimulatorTransition to a fake env with injected read/write."""
    fake = _FakeSimEnv(dynamics)
    monkeypatch.setattr(isaac_lab_model_pomdp, "_build_isaac_env", lambda *a, **k: fake)

    def _writer(env, states):
        env.written = np.asarray(states, dtype=float).copy()

    def _reader(env):
        return dynamics(env.written, env.last_action)

    transition = IsaacLabSimulatorTransition(
        task_id="Fake-Isaac-v0",
        dim=dim,
        state_writer=_writer,
        state_reader=_reader,
        device="cpu",
        **kwargs,
    )
    return transition, fake


def test_simulator_transition_uses_the_physics_step_as_the_mean(monkeypatch):
    """The transition writes the state, steps once, and centers on the sim result.

    Purpose: Validates IsaacLabSimulatorTransition samples f_sim(state, action) + noise.

    Given: A fake env whose "physics" doubles the written state.
    When: sample_next_state draws a large batch from a fixed state.
    Then: The written state equals the input, the sim was stepped, and the empirical
        mean equals the sim result (twice the state).

    Test type: unit
    """
    dim = 4
    transition, fake = _build_simulator_transition(
        monkeypatch, lambda states, action: states * 2.0, dim, process_noise_std=0.01
    )
    state = np.arange(dim, dtype=float)
    samples = transition.sample_next_state(state, np.ones(2), n_samples=6000)
    assert samples.shape == (6000, dim)
    assert np.allclose(np.asarray(fake.written)[0], state)
    assert fake.steps >= 1
    assert np.allclose(samples.mean(axis=0), state * 2.0, atol=0.03)


def test_simulator_transition_log_probability_peaks_at_the_sim_step(monkeypatch):
    """The transition log-density is a Gaussian centered on the physics step.

    Purpose: Validates log_probability scores next states around f_sim(state, action).

    Given: A fake env whose "physics" adds one to the written state.
    When: log_probability scores the sim result and a point offset from it.
    Then: The density at the sim result exceeds the density at the offset point.

    Test type: unit
    """
    dim = 3
    transition, _ = _build_simulator_transition(
        monkeypatch, lambda states, action: states + 1.0, dim, process_noise_std=0.1
    )
    state = np.zeros(dim)
    mean = state + 1.0
    at_mean = transition.log_probability(state, np.ones(2), mean)
    off_mean = transition.log_probability(state, np.ones(2), mean + 0.5)
    assert at_mean.shape == (1,)
    assert at_mean[0] > off_mean[0]


def test_simulator_transition_drives_the_model_and_is_action_agnostic_mean(monkeypatch):
    """An injected simulator transition overrides the default walk inside the model.

    Purpose: Validates the TransitionModel seam accepts IsaacLabSimulatorTransition.

    Given: A model built with a simulator transition whose physics negates the state.
    When: sample_next_state is called through the model.
    Then: The model's next-state mean equals the negated state.

    Test type: integration
    """
    dim = 3
    transition, _ = _build_simulator_transition(
        monkeypatch, lambda states, action: -states, dim, process_noise_std=0.01
    )
    model = IsaacLabModelPOMDP(
        observation_dim=dim,
        action_presets=[np.zeros(2), np.ones(2)],
        discount_factor=0.99,
        transition=transition,
    )
    state = np.array([1.0, -2.0, 3.0])
    mean = model.sample_next_state(state, model.get_actions()[1], n_samples=6000).mean(axis=0)
    assert np.allclose(mean, -state, atol=0.03)


def test_simulator_transition_default_reader_concatenates_scene_buffers(monkeypatch):
    """The default reader stacks root pose/velocity and joint buffers per env.

    Purpose: Validates _default_read_states mirrors the world's default extractor.

    Given: A fake env exposing scene articulation buffers for a single env.
    When: A next state is sampled with no explicit state_reader.
    Then: The sampled mean matches the concatenated scene buffers.

    Test type: unit
    """

    class _Data:
        root_pos_w = np.array([[1.0, 2.0, 3.0]])
        root_quat_w = np.array([[0.0, 0.0, 0.0, 1.0]])
        root_lin_vel_w = np.array([[0.1, 0.2, 0.3]])
        root_ang_vel_w = np.array([[0.0, 0.0, 0.0]])
        joint_pos = np.array([[0.5, -0.5]])
        joint_vel = np.array([[0.0, 0.0]])

    class _Asset:
        data = _Data()

    class _Scene:
        def __getitem__(self, key):
            del key
            return _Asset()

    class _Unwrapped:
        scene = _Scene()

    class _SceneEnv:
        unwrapped = _Unwrapped()

        def step(self, action):
            del action
            return None, 0.0, False, False, {}

    fake = _SceneEnv()
    monkeypatch.setattr(isaac_lab_model_pomdp, "_build_isaac_env", lambda *a, **k: fake)
    expected = np.concatenate(
        [
            _Data.root_pos_w[0],
            _Data.root_quat_w[0],
            _Data.root_lin_vel_w[0],
            _Data.root_ang_vel_w[0],
            _Data.joint_pos[0],
            _Data.joint_vel[0],
        ]
    )
    transition = IsaacLabSimulatorTransition(
        task_id="Fake-Isaac-v0",
        dim=expected.shape[0],
        state_writer=lambda env, states: None,
        device="cpu",
        process_noise_std=0.001,
    )
    mean = transition.sample_next_state(
        np.zeros(expected.shape[0]), np.ones(2), n_samples=4000
    ).mean(axis=0)
    assert np.allclose(mean, expected, atol=0.02)
