# SPDX-License-Identifier: MIT

"""Tests for the RockSample ZERO_MEAN_HAZARD_SHOCK and DISTANCE_DECAYED_HAZARD_PENALTY
reward-model variants.

The CONSTANT_HAZARD_PENALTY model is exercised by the env-level parity tests in
``test_rock_sample_pomdp.py`` / ``test_rock_sample_kernel_cache.py`` and
is not re-tested here. These tests target only the new variants and
their integration with :class:`RockSamplePOMDP` via the
``reward_model_type`` enum.
"""

# pylint: disable=protected-access  # Tests reach into internal hooks for parity checks.

import numpy as np
import pytest

from POMDPPlanners.environments.rock_sample_pomdp import (
    RewardModelType,
    RockSamplePOMDP,
    create_rock_sample_state,
)
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_utils.rock_sample_reward_models import (
    RockSampleDistanceDecayedHazardPenaltyRewardModel,
    RockSampleZeroMeanHazardShockRewardModel,
    RockSampleRewardModel,
)


# ---------------------------------------------------------------------------
# Env-level dispatch
# ---------------------------------------------------------------------------


def test_env_builds_standard_reward_model_by_default():
    """Default ``reward_model_type`` produces a CONSTANT_HAZARD_PENALTY ``RockSampleRewardModel``.

    Purpose: Validates the default dispatch path through ``_build_reward_model``.

    Given: A ``RockSamplePOMDP`` constructed with no explicit reward_model_type.
    When: The env is built.
    Then: ``env.reward_model`` is exactly a ``RockSampleRewardModel`` and not
        one of the two new subclasses.

    Test type: unit
    """
    env = RockSamplePOMDP(dangerous_areas=[(2, 2)])
    assert type(env.reward_model) is RockSampleRewardModel  # pylint: disable=unidiomatic-typecheck


def test_env_builds_high_variance_reward_model_on_request():
    """ZERO_MEAN_HAZARD_SHOCK dispatch path constructs the right subclass.

    Purpose: Validates that the env's reward-model factory picks the
        high-variance subclass when asked.

    Given: A ``RockSamplePOMDP`` with ``reward_model_type=ZERO_MEAN_HAZARD_SHOCK``.
    When: The env is built.
    Then: ``env.reward_model`` is a ``RockSampleZeroMeanHazardShockRewardModel``
        and the env-level reward_range spans both signs of the penalty.

    Test type: unit
    """
    env = RockSamplePOMDP(
        dangerous_areas=[(2, 2)],
        dangerous_area_penalty=-5.0,
        reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
    )
    assert isinstance(env.reward_model, RockSampleZeroMeanHazardShockRewardModel)
    # HIGH_VARIANCE flips the penalty sign with probability 0.5, so the
    # reward_range must include the positive sign too.
    assert env.reward_range is not None
    min_r, max_r = env.reward_range
    assert min_r <= -5.0
    assert max_r >= 5.0


def test_env_builds_decaying_reward_model_on_request():
    """DISTANCE_DECAYED_HAZARD_PENALTY dispatch path constructs the right subclass.

    Purpose: Validates the env's reward-model factory picks the
        decaying-probability subclass and threads ``penalty_decay`` through.

    Given: A ``RockSamplePOMDP`` with reward_model_type=DISTANCE_DECAYED_HAZARD_PENALTY
        and penalty_decay=2.5.
    When: The env is built.
    Then: ``env.reward_model`` is a ``RockSampleDistanceDecayedHazardPenaltyRewardModel``
        with ``penalty_decay`` 2.5.

    Test type: unit
    """
    env = RockSamplePOMDP(
        dangerous_areas=[(2, 2)],
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
        penalty_decay=2.5,
    )
    assert isinstance(env.reward_model, RockSampleDistanceDecayedHazardPenaltyRewardModel)
    assert env.reward_model.penalty_decay == 2.5


def test_decaying_model_rejects_non_positive_penalty_decay():
    """Validation: ``penalty_decay`` must be > 0 for the decaying model.

    Purpose: Validates the decaying-model constructor refuses zero / negative decay.

    Given: A direct ``RockSampleDistanceDecayedHazardPenaltyRewardModel`` build with
        ``penalty_decay=0.0`` (and again with a negative value).
    When: The constructor is called.
    Then: ``ValueError`` is raised in both cases.

    Test type: unit
    """
    with pytest.raises(ValueError, match="penalty_decay must be positive"):
        RockSampleDistanceDecayedHazardPenaltyRewardModel(
            map_size=(5, 5),
            rock_positions=[(0, 0)],
            step_penalty=0.0,
            bad_rock_penalty=-1.0,
            good_rock_reward=10.0,
            sensor_use_penalty=0.0,
            exit_reward=10.0,
            dangerous_areas=[(2, 2)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=-5.0,
            dangerous_area_hit_probability=1.0,
            penalty_decay=0.0,
        )
    with pytest.raises(ValueError, match="penalty_decay must be positive"):
        RockSampleDistanceDecayedHazardPenaltyRewardModel(
            map_size=(5, 5),
            rock_positions=[(0, 0)],
            step_penalty=0.0,
            bad_rock_penalty=-1.0,
            good_rock_reward=10.0,
            sensor_use_penalty=0.0,
            exit_reward=10.0,
            dangerous_areas=[(2, 2)],
            dangerous_area_radius=1.0,
            dangerous_area_penalty=-5.0,
            dangerous_area_hit_probability=1.0,
            penalty_decay=-1.0,
        )


# ---------------------------------------------------------------------------
# ZERO_MEAN_HAZARD_SHOCK semantics
# ---------------------------------------------------------------------------


def _make_high_variance_env() -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=(7, 7),
        rock_positions=[(0, 0), (2, 2), (3, 3), (5, 5)],
        dangerous_areas=[(2, 2), (4, 4)],
        dangerous_area_penalty=-5.0,
        reward_model_type=RewardModelType.ZERO_MEAN_HAZARD_SHOCK,
    )


def test_high_variance_scalar_returns_zero_outside_zone():
    """High-variance contribution is exactly zero outside any dangerous zone.

    Purpose: Validates the high-variance model never adds noise when the
        realised position is outside all dangerous areas.

    Given: A high-variance env with dangerous areas at (2,2) and (4,4),
        radius 1.0. A scalar reward call with a next_state at (0,0)
        (which lies outside both zones).
    When: ``compute_reward`` is invoked many times under a fixed seed.
    Then: Every call returns the deterministic base reward (no penalty
        sign flip) — i.e. the same value across all repetitions.

    Test type: unit
    """
    env = _make_high_variance_env()
    state = create_rock_sample_state((0, 0), (False, False, False, False))
    next_state = create_rock_sample_state((0, 0), (False, False, False, False))

    rewards = []
    np.random.seed(42)
    for _ in range(50):
        rewards.append(env.reward_model.compute_reward(state, action=1, next_state=next_state))

    assert all(r == rewards[0] for r in rewards)


def test_high_variance_scalar_has_zero_expected_contribution_in_zone():
    """In-zone high-variance contribution averages to ~0 over many trials.

    Purpose: Validates the ±penalty 50/50 split has zero expected value.

    Given: A high-variance env. A state and next_state placing the robot
        inside a dangerous zone. Action = 1 (north, no rock/exit term).
    When: ``compute_reward`` is invoked 2000 times under a fixed seed.
    Then: The mean reward is within 0.5 of the (deterministic) base
        reward, confirming the ±penalty noise averages out. The std is
        close to ``|dangerous_area_penalty|``.

    Test type: unit
    """
    env = _make_high_variance_env()
    state = create_rock_sample_state((2, 2), (False, False, False, False))
    next_state = create_rock_sample_state((2, 2), (False, False, False, False))

    np.random.seed(0)
    rewards = np.array(
        [
            env.reward_model.compute_reward(state, action=1, next_state=next_state)
            for _ in range(2000)
        ]
    )

    base_reward = env.step_penalty  # action=1 contributes only step_penalty
    assert abs(rewards.mean() - base_reward) < 0.5
    assert abs(rewards.std() - abs(env.dangerous_area_penalty)) < 0.5


def test_high_variance_batch_seeded_parity_with_scalar_loop():
    """Batch high-variance path is bit-identical to a row-by-row scalar replay.

    Purpose: Validates that the vectorised batch implementation produces
        exactly the same per-row rewards as repeatedly calling the
        scalar ``compute_reward`` under the same seed.

    Given: A high-variance env, 64 identical states all inside a
        dangerous zone, ``next_states`` matching, action = 1.
    When: ``compute_reward_batch`` is called once with seed=99, then the
        same env is reseeded and each row is computed via scalar
        ``compute_reward`` in ascending index order.
    Then: The two reward arrays are exactly equal — confirming
        ``np.random.random(n_in_zone)`` consumes the same uniform
        sequence the scalar loop would.

    Test type: unit
    """
    env = _make_high_variance_env()
    state = create_rock_sample_state((2, 2), (False, False, False, False))
    n = 64
    states = np.tile(state, (n, 1)).astype(np.float64)
    next_states = np.tile(state, (n, 1)).astype(np.float64)

    np.random.seed(99)
    batch_rewards = env.reward_model.compute_reward_batch(states, action=1, next_states=next_states)

    np.random.seed(99)
    scalar_rewards = np.array(
        [
            env.reward_model.compute_reward(states[i], action=1, next_state=next_states[i])
            for i in range(n)
        ]
    )

    np.testing.assert_array_equal(batch_rewards, scalar_rewards)


# ---------------------------------------------------------------------------
# DISTANCE_DECAYED_HAZARD_PENALTY semantics
# ---------------------------------------------------------------------------


def _make_decaying_env(penalty_decay: float = 1.0) -> RockSamplePOMDP:
    return RockSamplePOMDP(
        map_size=(7, 7),
        rock_positions=[(0, 0), (2, 2), (3, 3), (5, 5)],
        dangerous_areas=[(3, 3)],
        dangerous_area_penalty=-5.0,
        reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
        penalty_decay=penalty_decay,
    )


def test_decaying_scalar_hit_rate_decreases_with_distance():
    """Decaying-model hit rate falls off with distance to the closest centre.

    Purpose: Validates the ``exp(-dist / penalty_decay)`` gating: states
        near a dangerous-area centre get the penalty often; far states
        get it rarely.

    Given: A decaying env with one centre at (3, 3) and penalty_decay=1.0.
        Two next-states: one at (3, 3) (distance 0, prob 1.0) and one
        at (6, 6) (distance ~4.24, prob ~exp(-4.24) ≈ 0.014).
    When: 2000 scalar reward draws are taken at each next-state under a
        fixed seed.
    Then: Near-state hit rate > 0.9; far-state hit rate < 0.1.

    Test type: unit
    """
    env = _make_decaying_env(penalty_decay=1.0)
    state = create_rock_sample_state((3, 3), (False, False, False, False))
    near = create_rock_sample_state((3, 3), (False, False, False, False))
    far = create_rock_sample_state((6, 6), (False, False, False, False))

    base_reward = env.step_penalty  # action=1 contributes only step_penalty
    np.random.seed(7)
    near_hits = sum(
        env.reward_model.compute_reward(state, action=1, next_state=near) != base_reward
        for _ in range(2000)
    )
    np.random.seed(7)
    far_hits = sum(
        env.reward_model.compute_reward(state, action=1, next_state=far) != base_reward
        for _ in range(2000)
    )

    assert near_hits / 2000 > 0.9
    assert far_hits / 2000 < 0.1


def test_decaying_batch_seeded_parity_with_scalar_loop():
    """Batch decaying-prob path matches a row-by-row scalar replay bit-identically.

    Purpose: Validates that the vectorised njit-backed batch implementation
        produces exactly the same per-row rewards as repeatedly calling the
        scalar ``compute_reward`` under the same seed. Confirms
        ``np.random.random(n)`` in the batch path consumes the same
        uniform sequence the scalar loop would.

    Given: A decaying env, 32 next-states at a mix of distances from the
        centre, action = 1.
    When: ``compute_reward_batch`` runs once with seed=11, then the same
        env is reseeded and each row is computed via scalar
        ``compute_reward`` in ascending index order.
    Then: The two reward arrays are exactly equal.

    Test type: unit
    """
    env = _make_decaying_env(penalty_decay=1.5)
    rng_states = []
    for r in range(7):
        for c in range(7):
            rng_states.append(create_rock_sample_state((r, c), (False, False, False, False)))
    next_states = np.asarray(rng_states[:32], dtype=np.float64)
    states = next_states.copy()

    np.random.seed(11)
    batch_rewards = env.reward_model.compute_reward_batch(states, action=1, next_states=next_states)

    np.random.seed(11)
    scalar_rewards = np.array(
        [
            env.reward_model.compute_reward(states[i], action=1, next_state=next_states[i])
            for i in range(states.shape[0])
        ]
    )

    np.testing.assert_array_equal(batch_rewards, scalar_rewards)


def test_decaying_at_centre_always_applies_penalty():
    """At distance zero, the decaying probability is 1.0 — penalty is certain.

    Purpose: Validates the boundary case ``min_dist == 0`` of the
        ``exp(-dist / penalty_decay)`` gating.

    Given: A decaying env with one centre at (3, 3) and any positive
        penalty_decay. A next-state exactly at (3, 3).
    When: ``compute_reward`` is invoked 50 times under a fixed seed.
    Then: Every call yields ``base_reward + dangerous_area_penalty``.

    Test type: unit
    """
    env = _make_decaying_env(penalty_decay=1.0)
    state = create_rock_sample_state((3, 3), (False, False, False, False))
    next_state = create_rock_sample_state((3, 3), (False, False, False, False))

    np.random.seed(3)
    expected = env.step_penalty + env.dangerous_area_penalty
    for _ in range(50):
        r = env.reward_model.compute_reward(state, action=1, next_state=next_state)
        assert r == expected
