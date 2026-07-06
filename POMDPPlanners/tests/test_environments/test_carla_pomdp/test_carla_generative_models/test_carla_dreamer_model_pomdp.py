# SPDX-License-Identifier: MIT

"""Tests for the Dreamer-backed CARLA generative model.

Covers :class:`DreamerCarlaModelPOMDP` wiring onto the
:class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_model_pomdp.CarlaModelPOMDP`
interface: the RSSM imagination transition, decoder observation model, decoder-log-density
reweighting path, learned reward head, continue-head termination, and posterior belief
seeding. A lightweight fake :class:`DreamerWorldModel` stands in for a trained RSSM so the
tests run with no CARLA server and no deep-learning framework.
"""

from typing import Dict, Mapping

import numpy as np
import pytest

from POMDPPlanners.environments.carla_pomdp.carla_generative_models import (
    DreamerCarlaModelPOMDP,
)

_LATENT_DIM = 3


class _FakeDreamerWorldModel:
    """Deterministic stand-in RSSM with analytically checkable mappings.

    ``imagine`` adds the throttle to the latent (so control routing is observable),
    ``decode`` broadcasts the latent's first component into the ``gnss`` head, ``reward``
    sums the latent, and ``continue_prob`` is a fixed value so termination is controllable.
    """

    latent_dim = _LATENT_DIM

    def __init__(self, continue_value: float = 1.0) -> None:
        self.continue_value = continue_value
        self.last_controls: np.ndarray = np.empty((0, 3))

    def encode(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.full(self.latent_dim, float(np.sum(observation["gnss"])))

    def imagine(self, latents: np.ndarray, controls: np.ndarray) -> np.ndarray:
        self.last_controls = np.asarray(controls, dtype=float)
        throttle = np.asarray(controls, dtype=float)[:, 0:1]
        return np.asarray(latents, dtype=float) + throttle

    def decode(self, latents: np.ndarray) -> Dict[str, np.ndarray]:
        batch = np.asarray(latents).shape[0]
        first = np.asarray(latents, dtype=float)[:, 0:1]
        return {"gnss": np.repeat(first, 3, axis=1), "agents": np.zeros((batch, 25))}

    def decode_log_prob(
        self, latents: np.ndarray, observation: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        target = float(np.sum(observation["gnss"]))
        return -np.sum((np.asarray(latents, dtype=float) - target) ** 2, axis=1)

    def reward(self, latents: np.ndarray) -> np.ndarray:
        return np.sum(np.asarray(latents, dtype=float), axis=1)

    def continue_prob(self, latents: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(latents).shape[0], self.continue_value)


def _make_observation() -> Dict[str, np.ndarray]:
    return {"gnss": np.array([1.0, 2.0, 3.0]), "agents": np.zeros(25)}


def _make_env(
    continue_value: float = 1.0,
    continue_threshold: float = 0.5,
    seed_observation: bool = True,
) -> DreamerCarlaModelPOMDP:
    return DreamerCarlaModelPOMDP(
        _FakeDreamerWorldModel(continue_value=continue_value),
        discount_factor=0.95,
        continue_threshold=continue_threshold,
        initial_observation=_make_observation() if seed_observation else None,
    )


def test_sample_next_state_single_applies_imagination_step():
    """A single next-state draw is the world model's imagination of the latent.

    Purpose: Validates sample_next_state routes one latent through the RSSM imagination.

    Given: A Dreamer-backed model and a latent state of ones
    When: sample_next_state is called with the first action and n_samples=1
    Then: A 1-D latent is returned equal to the latent plus that action's throttle

    Test type: unit
    """
    env = _make_env()
    state = np.ones(_LATENT_DIM)

    next_state = env.sample_next_state(state, action=0)

    throttle = env.action_presets[0][0]
    assert next_state.shape == (_LATENT_DIM,)
    np.testing.assert_allclose(next_state, state + throttle)


def test_sample_next_state_multiple_returns_stacked_latents():
    """Requesting several next states returns a stacked latent array.

    Purpose: Validates the n_samples>1 batching contract of sample_next_state.

    Given: A Dreamer-backed model and a zero latent state
    When: sample_next_state is called with n_samples=4
    Then: A (4, latent_dim) array is returned, each row imagined under the action

    Test type: unit
    """
    env = _make_env()
    state = np.zeros(_LATENT_DIM)

    next_states = env.sample_next_state(state, action=0, n_samples=4)

    assert next_states.shape == (4, _LATENT_DIM)
    np.testing.assert_allclose(next_states, np.full((4, _LATENT_DIM), env.action_presets[0][0]))


def test_sample_next_state_batch_imagines_one_successor_per_particle():
    """The batched transition imagines one successor per input particle.

    Purpose: Validates sample_next_state_batch vectorizes the imagination over particles.

    Given: A Dreamer-backed model and three distinct latent particles
    When: sample_next_state_batch is called under a single action
    Then: A (3, latent_dim) array is returned, each row advanced by the action throttle

    Test type: unit
    """
    env = _make_env()
    particles = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    successors = env.sample_next_state_batch(particles, action=0)

    assert successors.shape == (3, _LATENT_DIM)
    np.testing.assert_allclose(successors, particles + env.action_presets[0][0])


def test_transition_routes_selected_action_control_triple():
    """The action index is resolved to its control triple before imagination.

    Purpose: Validates the discrete action is mapped to its (throttle, steer, brake) preset.

    Given: A Dreamer-backed model whose fake records the controls it receives
    When: sample_next_state is called with a specific non-zero action index
    Then: The world model is handed exactly that action's preset triple

    Test type: unit
    """
    world_model = _FakeDreamerWorldModel()
    env = DreamerCarlaModelPOMDP(
        world_model, discount_factor=0.95, initial_observation=_make_observation()
    )
    action = len(env.get_actions()) - 1

    env.sample_next_state(np.zeros(_LATENT_DIM), action=action)

    expected = np.asarray(env.action_presets[action], dtype=float)
    np.testing.assert_allclose(world_model.last_controls[0], expected)


def test_transition_log_probability_is_not_implemented():
    """The deterministic recurrence exposes no tractable transition density.

    Purpose: Validates transition_log_probability raises rather than fabricating a density.

    Given: A Dreamer-backed model
    When: transition_log_probability is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = _make_env()

    with pytest.raises(NotImplementedError):
        env.transition_log_probability(np.zeros(_LATENT_DIM), 0, np.zeros((1, _LATENT_DIM)))


def test_sample_observation_single_returns_decoded_dict():
    """A single observation draw decodes the latent into the CARLA schema.

    Purpose: Validates sample_observation returns a {gnss, agents} dict of correct shapes.

    Given: A Dreamer-backed model and a latent next-state
    When: sample_observation is called with n_samples=1
    Then: A dict with 'gnss' (3,) and 'agents' (25,) arrays is returned

    Test type: unit
    """
    env = _make_env()

    observation = env.sample_observation(np.ones(_LATENT_DIM), action=0)

    assert sorted(observation) == ["agents", "gnss"]
    assert observation["gnss"].shape == (3,)
    assert observation["agents"].shape == (25,)


def test_sample_observation_multiple_returns_independent_copies():
    """Multiple observation draws are independent dicts, not shared references.

    Purpose: Validates the n_samples>1 path returns a list of independent observations.

    Given: A Dreamer-backed model and a latent next-state
    When: sample_observation is called with n_samples=2 and one result is mutated
    Then: A list of two dicts is returned and mutating one leaves the other unchanged

    Test type: unit
    """
    env = _make_env()

    observations = env.sample_observation(np.ones(_LATENT_DIM), action=0, n_samples=2)

    assert isinstance(observations, list) and len(observations) == 2
    observations[0]["gnss"][0] = 999.0
    assert observations[1]["gnss"][0] != 999.0


def test_observation_log_probability_scores_each_candidate():
    """Each candidate observation is scored by the decoder log-density.

    Purpose: Validates observation_log_probability returns one decoder log-prob per obs.

    Given: A Dreamer-backed model, a latent, and two candidate observations
    When: observation_log_probability is called with the list of observations
    Then: A length-2 array of the world model's decoder log-densities is returned

    Test type: unit
    """
    env = _make_env()
    latent = np.array([1.0, 1.0, 1.0])
    obs_a = _make_observation()
    obs_b = {"gnss": np.zeros(3), "agents": np.zeros(25)}

    log_probs = env.observation_log_probability(latent, action=0, observations=[obs_a, obs_b])

    expected = env.world_model.decode_log_prob(
        latent.reshape(1, -1), obs_a
    ), env.world_model.decode_log_prob(latent.reshape(1, -1), obs_b)
    assert log_probs.shape == (2,)
    np.testing.assert_allclose(log_probs, [expected[0][0], expected[1][0]])


def test_observation_log_probability_accepts_single_observation():
    """A bare (non-list) observation is scored as a one-element result.

    Purpose: Validates the singleton observation input path of observation_log_probability.

    Given: A Dreamer-backed model, a latent, and a single observation (not in a list)
    When: observation_log_probability is called with that observation
    Then: A length-1 array is returned

    Test type: unit
    """
    env = _make_env()

    log_probs = env.observation_log_probability(
        np.ones(_LATENT_DIM), action=0, observations=_make_observation()
    )

    assert log_probs.shape == (1,)


def test_observation_log_probability_per_state_scores_all_particles():
    """One observation is scored against every candidate next-state particle.

    Purpose: Validates the vectorized reweighting path observation_log_probability_per_state.

    Given: A Dreamer-backed model, three latent particles, and one observation
    When: observation_log_probability_per_state is called
    Then: A length-3 array of per-particle decoder log-densities is returned

    Test type: unit
    """
    env = _make_env()
    particles = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    observation = _make_observation()

    log_probs = env.observation_log_probability_per_state(particles, 0, observation)

    expected = env.world_model.decode_log_prob(particles, observation)
    assert log_probs.shape == (3,)
    np.testing.assert_allclose(log_probs, expected)


def test_reward_uses_resulting_state_when_provided():
    """The reward head scores the resulting state when next_state is passed.

    Purpose: Validates reward evaluates the learned head on next_state over the prior state.

    Given: A Dreamer-backed model, a prior latent, and a distinct resulting latent
    When: reward is called with both state and next_state
    Then: The scalar equals the world model's reward head on the resulting latent

    Test type: unit
    """
    env = _make_env()
    state = np.zeros(_LATENT_DIM)
    next_state = np.array([1.0, 2.0, 3.0])

    reward = env.reward(state, action=0, next_state=next_state)

    assert reward == pytest.approx(float(np.sum(next_state)))


def test_reward_falls_back_to_state_without_next_state():
    """Without a resulting state the reward head scores the given state.

    Purpose: Validates reward's fallback to state when next_state is omitted.

    Given: A Dreamer-backed model and a latent state
    When: reward is called with no next_state
    Then: The scalar equals the world model's reward head on that state

    Test type: unit
    """
    env = _make_env()
    state = np.array([2.0, 2.0, 2.0])

    reward = env.reward(state, action=0)

    assert reward == pytest.approx(float(np.sum(state)))


def test_is_terminal_true_when_continue_probability_below_threshold():
    """Termination fires when the continue head drops below the threshold.

    Purpose: Validates is_terminal maps a low continue probability to a terminal state.

    Given: A Dreamer-backed model whose continue head returns 0.3 (threshold 0.5)
    When: is_terminal is queried
    Then: True is returned

    Test type: unit
    """
    env = _make_env(continue_value=0.3, continue_threshold=0.5)

    assert env.is_terminal(np.zeros(_LATENT_DIM)) is True


def test_is_terminal_false_when_continue_probability_at_or_above_threshold():
    """A continue probability at the threshold is treated as non-terminal.

    Purpose: Validates the strict-less-than threshold semantics of is_terminal.

    Given: A Dreamer-backed model whose continue head returns exactly the threshold 0.5
    When: is_terminal is queried
    Then: False is returned

    Test type: unit
    """
    env = _make_env(continue_value=0.5, continue_threshold=0.5)

    assert env.is_terminal(np.zeros(_LATENT_DIM)) is False


def test_initial_state_dist_encodes_seed_observation():
    """The initial belief latent is the posterior encoding of the seed observation.

    Purpose: Validates initial_state_dist seeds the belief via the world model's encoder.

    Given: A Dreamer-backed model seeded with an initial observation
    When: initial_state_dist().sample(2) is drawn
    Then: Two independent latents equal to encode(initial_observation) are returned

    Test type: unit
    """
    env = _make_env()

    samples = env.initial_state_dist().sample(2)

    expected = env.world_model.encode(_make_observation())
    assert len(samples) == 2
    np.testing.assert_allclose(samples[0], expected)
    samples[0][0] = 123.0
    assert samples[1][0] != 123.0


def test_initial_observation_dist_returns_independent_seed_copies():
    """The initial observation distribution yields independent seed copies.

    Purpose: Validates initial_observation_dist returns copies of the seed observation.

    Given: A Dreamer-backed model seeded with an initial observation
    When: initial_observation_dist().sample(2) is drawn and one copy is mutated
    Then: Both equal the seed observation and mutating one leaves the other unchanged

    Test type: unit
    """
    env = _make_env()

    samples = env.initial_observation_dist().sample(2)

    seed = _make_observation()
    np.testing.assert_allclose(samples[0]["gnss"], seed["gnss"])
    samples[0]["gnss"][0] = 42.0
    assert samples[1]["gnss"][0] != 42.0


def test_initial_hooks_raise_without_seed_observation():
    """Unseeded initial-distribution hooks raise a guiding error.

    Purpose: Validates the initial hooks refuse to run without a seed observation.

    Given: A Dreamer-backed model constructed with no initial_observation
    When: initial_state_dist or initial_observation_dist is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = _make_env(seed_observation=False)

    with pytest.raises(NotImplementedError):
        env.initial_state_dist()
    with pytest.raises(NotImplementedError):
        env.initial_observation_dist()


def test_get_actions_indexes_the_control_presets():
    """The discrete action set indexes the control presets.

    Purpose: Validates get_actions returns contiguous indices into action_presets.

    Given: A Dreamer-backed model
    When: get_actions is called
    Then: The returned indices match range(len(action_presets))

    Test type: unit
    """
    env = _make_env()

    assert env.get_actions() == list(range(len(env.action_presets)))
