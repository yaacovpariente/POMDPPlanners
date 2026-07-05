# SPDX-License-Identifier: MIT

"""Tests for the CARLA generative-model interface and its factored reference impl.

Covers the abstract :class:`CarlaModelPOMDP` interface contract, the concrete
:class:`FactoredCarlaModelPOMDP` factored observation model / identity transition /
shared reward, and the intended learned-dynamics subclassing pattern. All tests run on
hand-built state arrays with no CARLA server.
"""

from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.environments.carla_pomdp.carla_model_pomdp import CarlaModelPOMDP
from POMDPPlanners.environments.carla_pomdp.carla_factored_model_pomdp import (
    FactoredCarlaModelPOMDP,
)
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
    driving_quality_reward,
)


def _make_state(
    agent_rows: List[np.ndarray],
    ego: Any = None,
    max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
) -> np.ndarray:
    """Assemble a CARLA state vector from an ego part and filled agent slots."""
    ego_part = np.zeros(EGO_STATE_WIDTH) if ego is None else np.asarray(ego, dtype=float)
    rows = np.zeros((max_tracked_agents, AGENT_SLOT_WIDTH))
    for index, row in enumerate(agent_rows):
        rows[index] = row
    return np.concatenate([ego_part, rows.reshape(-1)])


def test_base_interface_cannot_be_instantiated():
    """Test that the abstract CarlaModelPOMDP interface cannot be instantiated.

    Purpose: Validates CarlaModelPOMDP is an abstract interface, not a usable class

    Given: The abstract CarlaModelPOMDP base with unimplemented dynamics methods
    When: Direct instantiation is attempted
    Then: A TypeError is raised for the remaining abstract methods

    Test type: unit
    """
    with pytest.raises(TypeError):
        CarlaModelPOMDP(discount_factor=0.95)  # type: ignore[abstract]


def test_get_actions_indexes_presets():
    """Test that the discrete action set is the indices into the control presets.

    Purpose: Validates the schema-level discrete action enumeration

    Given: A factored model built with three control presets
    When: get_actions is called
    Then: It returns [0, 1, 2], one index per preset

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(
        discount_factor=0.95,
        action_presets=[(0.5, 0.0, 0.0), (0.3, -0.5, 0.0), (0.0, 0.0, 1.0)],
    )
    assert env.get_actions() == [0, 1, 2]


def test_sample_next_state_identity_single_and_batch():
    """Test the identity-transition placeholder honours the n_samples convention.

    Purpose: Validates the placeholder returns a single state for n_samples=1 and a
        stacked batch otherwise, matching the repo convention

    Given: A factored model and a hand-built state vector
    When: sample_next_state is called with n_samples 1 and 3
    Then: n_samples=1 returns the single unchanged state (1-D); n_samples=3 returns a
        (3, D) stack of copies

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    state = _make_state([np.array([1.0, 3.0, 0.0, 0.0, 4.0])])

    single = env.sample_next_state(state, action=0)
    assert single.shape == state.shape
    np.testing.assert_array_equal(single, state)

    batch = env.sample_next_state(state, action=0, n_samples=3)
    assert batch.shape == (3, state.shape[0])
    for row in batch:
        np.testing.assert_array_equal(row, state)


def test_transition_log_probability_raises():
    """Test that the placeholder transition density raises NotImplementedError.

    Purpose: Validates the transition density is an explicit, documented stub

    Given: A factored model with a placeholder identity transition
    When: transition_log_probability is called
    Then: NotImplementedError is raised

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    state = _make_state([])
    with pytest.raises(NotImplementedError):
        env.transition_log_probability(state, action=0, next_states=state)


def test_is_terminal_always_false():
    """Test that the model reports no terminal states.

    Purpose: Validates the model's is_terminal is a constant False

    Given: A factored model and an arbitrary state
    When: is_terminal is called
    Then: It returns False

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    assert env.is_terminal(_make_state([])) is False


def test_initial_distributions_raise():
    """Test that the initial-distribution hooks direct seeding to the world.

    Purpose: Validates the model refuses to synthesise an initial belief itself

    Given: A factored model paired with a forward-only world
    When: initial_state_dist or initial_observation_dist is called
    Then: NotImplementedError is raised in both cases

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    with pytest.raises(NotImplementedError):
        env.initial_state_dist()
    with pytest.raises(NotImplementedError):
        env.initial_observation_dist()


def test_sample_observation_keys_and_hides_unseen():
    """Test that sampled observations expose the schema keys and hide unseen agents.

    Purpose: Validates the factored observation zeroes out-of-range/occluded agents

    Given: A state with one in-range agent and one agent beyond perception range
    When: sample_observation is called (no pose noise via zero pose_std is not used;
        instead we inspect the present flags which noise does not touch)
    Then: The observation has 'gnss' and 'agents' keys; the near agent's slot is present
        and the far agent's slot is zeroed

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95, perception_range=50.0)
    near = np.array([1.0, 10.0, 0.0, 0.0, 0.0])
    far = np.array([1.0, 100.0, 0.0, 0.0, 0.0])
    state = _make_state([near, far])

    observation = env.sample_observation(state, action=0)
    assert sorted(observation) == ["agents", "gnss"]
    obs_rows = observation["agents"].reshape(DEFAULT_MAX_TRACKED_AGENTS, AGENT_SLOT_WIDTH)
    assert obs_rows[0, 0] == 1.0  # near agent detected
    assert obs_rows[1, 0] == 0.0  # far agent hidden


def test_observation_log_probability_prefers_matching_observation():
    """Test that a matching observation scores higher than a mismatched one.

    Purpose: Validates the factored observation likelihood ranks the truthful reading
        above a corrupted one

    Given: A state with a single detected agent and its clean rendered observation
    When: observation_log_probability is scored for the clean observation vs a
        pose-shifted one
    Then: The clean observation has the higher log-probability

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    state = _make_state([np.array([1.0, 10.0, 1.0, 0.2, 3.0])])
    clean = env._render_observation(state, noisy=False)  # pylint: disable=protected-access

    mismatched = {
        "gnss": clean["gnss"].copy(),
        "agents": clean["agents"].copy(),
    }
    mismatched["agents"][1] += 5.0  # shift the detected agent's rel_x far from truth

    clean_lp = env.observation_log_probability(state, action=0, observations=clean)[0]
    mismatched_lp = env.observation_log_probability(state, action=0, observations=mismatched)[0]
    assert clean_lp > mismatched_lp


def test_observation_log_probability_penalizes_seeing_an_occluded_agent():
    """Test that observing an occluded agent is scored as near-impossible.

    Purpose: Validates the occlusion gating in the observation likelihood

    Given: A target agent occluded by a blocker on the ego->target sight line
    When: observation_log_probability scores an observation that (wrongly) reports the
        occluded target as detected vs one that correctly omits it
    Then: The correct (omitted) observation scores far higher than the impossible one

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95, occlusion_radius=1.5)
    target = np.array([1.0, 10.0, 0.0, 0.0, 0.0])
    blocker = np.array([1.0, 5.0, 0.0, 0.0, 0.0])
    state = _make_state([target, blocker])

    rows = np.zeros((DEFAULT_MAX_TRACKED_AGENTS, AGENT_SLOT_WIDTH))
    rows[1] = blocker  # only the visible blocker is reported
    omitted = {"gnss": state[:2].copy(), "agents": rows.reshape(-1)}

    wrongly_seen = {"gnss": state[:2].copy(), "agents": rows.reshape(-1).copy()}
    wrongly_seen["agents"][:AGENT_SLOT_WIDTH] = target  # report the occluded target

    omitted_lp = env.observation_log_probability(state, action=0, observations=omitted)[0]
    seen_lp = env.observation_log_probability(state, action=0, observations=wrongly_seen)[0]
    assert omitted_lp > seen_lp + 40.0  # the impossible reading is floored at _LOG_EPS


def test_reward_matches_shared_driving_quality_reward():
    """Test that the model reward equals the shared world reward for the same state.

    Purpose: Validates world and model score a transition identically by construction

    Given: A resulting ego state with longitudinal speed and a steering action preset
    When: The model reward is computed for that next_state
    Then: It equals driving_quality_reward called with the preset's steer and the model's
        reward parameters

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(
        discount_factor=0.95,
        action_presets=[(0.5, 0.0, 0.0), (0.3, -0.5, 0.0)],
        desired_speed=8.0,
        out_lane_thresh=2.0,
        collision_penalty=100.0,
    )
    next_state = _make_state([], ego=[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])

    reward = env.reward(state=_make_state([]), action=1, next_state=next_state)
    expected = driving_quality_reward(next_state, -0.5, False, 8.0, 2.0, 100.0)
    assert reward == pytest.approx(expected)


def test_reward_falls_back_to_state_when_next_state_is_none():
    """Test that reward scores the current state when no next_state is supplied.

    Purpose: Validates the belief-expectation reward path (reward_batch passes only
        state, action) is supported

    Given: A model and a state carrying longitudinal speed
    When: reward is called with next_state=None
    Then: It equals the reward computed from that state as the resulting state

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(
        discount_factor=0.95,
        action_presets=[(0.5, 0.0, 0.0), (0.3, -0.5, 0.0)],
    )
    state = _make_state([], ego=[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])

    reward = env.reward(state=state, action=1)
    expected = driving_quality_reward(state, -0.5, False, 8.0, 2.0, 100.0)
    assert reward == pytest.approx(expected)


def test_hash_and_equal_observation():
    """Test observation hashing and equality over the CARLA observation dict.

    Purpose: Validates the schema-level equality/hash agree for identical observations
        and differ for distinct ones

    Given: Two identical observations and one with a different agents payload
    When: is_equal_observation and hash_observation are applied
    Then: Identical observations compare equal and hash equal; the distinct one differs

    Test type: unit
    """
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    obs_a = {"gnss": np.array([1.0, 2.0]), "agents": np.zeros(AGENT_SLOT_WIDTH)}
    obs_b = {"gnss": np.array([1.0, 2.0]), "agents": np.zeros(AGENT_SLOT_WIDTH)}
    obs_c = {"gnss": np.array([1.0, 2.0]), "agents": np.ones(AGENT_SLOT_WIDTH)}

    assert env.is_equal_observation(obs_a, obs_b)
    assert env.hash_observation(obs_a) == env.hash_observation(obs_b)
    assert not env.is_equal_observation(obs_a, obs_c)
    assert env.hash_observation(obs_a) != env.hash_observation(obs_c)


def test_config_id_stable_and_sensitive_to_params():
    """Test the deterministic config id is stable and parameter-sensitive.

    Purpose: Validates config_id round-trips schema/model attributes for caching

    Given: Two identically configured models and one with a different detect_prob
    When: config_id is read from each
    Then: The identical models share a config_id; the differing model has a distinct one

    Test type: configuration
    """
    env_a = FactoredCarlaModelPOMDP(discount_factor=0.95, detect_prob=0.95)
    env_b = FactoredCarlaModelPOMDP(discount_factor=0.95, detect_prob=0.95)
    env_c = FactoredCarlaModelPOMDP(discount_factor=0.95, detect_prob=0.5)

    assert env_a.config_id == env_b.config_id
    assert env_a.config_id != env_c.config_id


def test_learned_dynamics_subclass_plugs_into_sample_next_step():
    """Test the intended pattern: override transition/reward, keep the factored obs model.

    Purpose: Validates the interface supports a study-specific (e.g. learned) model that
        replaces dynamics while reusing the factored observation and CARLA schema

    Given: A subclass overriding sample_next_state (a fixed drift) and reward (constant)
    When: sample_next_step is driven from a hand-built state
    Then: The next_state reflects the overridden transition, the observation carries the
        schema keys, and the reward is the overridden constant

    Test type: integration
    """

    class DriftModel(FactoredCarlaModelPOMDP):
        def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
            del action
            nxt = np.asarray(state, dtype=float).copy()
            nxt[0] += 1.0  # deterministic forward drift in ego x
            if n_samples == 1:
                return nxt
            return np.stack([nxt for _ in range(n_samples)])

        def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
            del state, action, next_state
            return 2.0

    env = DriftModel(discount_factor=0.95)
    state = _make_state([np.array([1.0, 10.0, 0.0, 0.0, 0.0])])

    next_state, observation, reward = env.sample_next_step(state, action=0)
    assert next_state[0] == state[0] + 1.0
    assert sorted(observation) == ["agents", "gnss"]
    assert reward == 2.0


def test_factored_model_docstring_example():
    """Test the usage example from the FactoredCarlaModelPOMDP docstring.

    Purpose: Validates the class docstring example executes and behaves as documented

    Given: The exact setup from the docstring (seeded rng, zero state, first action)
    When: sample_next_step is called and terminal state is checked
    Then: The observation has the documented keys and is_terminal is False

    Test type: example
    """
    np.random.seed(42)
    env = FactoredCarlaModelPOMDP(discount_factor=0.95)
    width = EGO_STATE_WIDTH + env.max_tracked_agents * AGENT_SLOT_WIDTH
    state = np.zeros(width)
    action = env.get_actions()[0]

    _, observation, _ = env.sample_next_step(state, action)
    assert sorted(observation) == ["agents", "gnss"]
    assert env.is_terminal(state) is False
