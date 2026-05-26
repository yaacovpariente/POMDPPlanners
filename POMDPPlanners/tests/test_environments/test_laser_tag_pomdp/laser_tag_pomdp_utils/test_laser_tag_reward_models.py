# SPDX-License-Identifier: MIT

"""Tests for the LaserTag POMDP reward-model variants.

Covers the high-variance and decaying-hit-probability variants added
alongside the standard model and exposed via
:class:`LaserTagPOMDP`'s ``reward_model_type`` parameter. The existing
standard-model behaviour is already exercised by
``test_laser_tag_pomdp.py`` and ``test_laser_tag_reward_native.py``.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import (
    LaserTagPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_utils.laser_tag_reward_models import (
    BaseLaserTagRewardModel,
    LaserTagDistanceDecayedHazardPenaltyRewardModel,
    LaserTagZeroMeanHazardShockRewardModel,
    LaserTagRewardModel,
)


# Default ``LaserTagPOMDP`` walls / danger areas / radii — referenced by the
# scenario-style tests so the expected wall and zone cells stay in sync with
# the env defaults.
_DEFAULT_WALLS = {
    (1, 2),
    (3, 0),
    (3, 4),
    (5, 0),
    (6, 4),
    (9, 1),
    (9, 4),
    (10, 6),
}
_DEFAULT_DANGER_AREAS = [(5, 3), (7, 1), (2, 5)]
_DEFAULT_DANGER_RADIUS = 1.0
_DEFAULT_DANGER_PENALTY = 5.0
_DEFAULT_STEP_COST = 1.0
_DEFAULT_TAG_REWARD = 10.0
_DEFAULT_TAG_PENALTY = 10.0
_DEFAULT_FLOOR = (11, 7)
_DEFAULT_ACTION_DIRS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, 1),
    3: (0, -1),
    4: (0, 0),
}


def _make_hv_model() -> LaserTagZeroMeanHazardShockRewardModel:
    return LaserTagZeroMeanHazardShockRewardModel(
        floor_shape=_DEFAULT_FLOOR,
        walls=_DEFAULT_WALLS,
        dangerous_areas=_DEFAULT_DANGER_AREAS,
        dangerous_area_radius=_DEFAULT_DANGER_RADIUS,
        dangerous_area_penalty=_DEFAULT_DANGER_PENALTY,
        tag_reward=_DEFAULT_TAG_REWARD,
        tag_penalty=_DEFAULT_TAG_PENALTY,
        step_cost=_DEFAULT_STEP_COST,
        action_directions=_DEFAULT_ACTION_DIRS,
    )


def _make_decay_model(
    penalty_decay: float = 1.0,
) -> LaserTagDistanceDecayedHazardPenaltyRewardModel:
    return LaserTagDistanceDecayedHazardPenaltyRewardModel(
        floor_shape=_DEFAULT_FLOOR,
        walls=_DEFAULT_WALLS,
        dangerous_areas=_DEFAULT_DANGER_AREAS,
        dangerous_area_radius=_DEFAULT_DANGER_RADIUS,
        dangerous_area_penalty=_DEFAULT_DANGER_PENALTY,
        tag_reward=_DEFAULT_TAG_REWARD,
        tag_penalty=_DEFAULT_TAG_PENALTY,
        step_cost=_DEFAULT_STEP_COST,
        action_directions=_DEFAULT_ACTION_DIRS,
        penalty_decay=penalty_decay,
    )


class TestLaserTagZeroMeanHazardShockRewardModel:
    """Behavioural tests for the HV variant."""

    def test_subclass_inherits_base(self):
        """HV model is both a ``BaseLaserTagRewardModel`` and a ``LaserTagRewardModel``.

        Purpose: Validates the class hierarchy so callers that expect either
            base type can interchangeably use the HV variant.

        Given: A constructed ``LaserTagZeroMeanHazardShockRewardModel``.
        When: ``isinstance`` is checked against the abstract and concrete bases.
        Then: Both checks return True.

        Test type: unit
        """
        model = _make_hv_model()
        assert isinstance(model, LaserTagRewardModel)
        assert isinstance(model, BaseLaserTagRewardModel)

    def test_wall_penalty_remains_deterministic(self):
        """Wall hits in the HV model always subtract exactly ``dangerous_area_penalty``.

        Purpose: Validates that walls in the HV variant are still scored with
            the deterministic penalty even though dangerous-area contributions
            are stochastic.

        Given: A HV reward model and a state whose realised next position is
            on a wall cell but outside every dangerous area.
        When: ``compute_reward`` is called many times with action 0 (North).
        Then: Every call returns ``-step_cost - dangerous_area_penalty`` —
            no random ±penalty drift.

        Test type: unit
        """
        model = _make_hv_model()
        # Wall (3, 4); robot starts at (4, 4) so action 1 (South) is irrelevant —
        # we thread the next_state explicitly so the assertion does not depend
        # on the env's transition model.
        state = np.array([4.0, 4.0, 0.0, 0.0, 0.0])
        next_state = np.array([3.0, 4.0, 0.0, 0.0, 0.0])  # wall (3, 4)
        rewards = [model.compute_reward(state, action=0, next_state=next_state) for _ in range(50)]
        assert all(
            r == pytest.approx(-_DEFAULT_STEP_COST - _DEFAULT_DANGER_PENALTY) for r in rewards
        )

    def test_danger_zone_emits_symmetric_penalty(self):
        """HV dangerous-area contribution is ``±dangerous_area_penalty`` with mean ≈ 0.

        Purpose: Validates the variant's signature property — danger zones
            contribute zero expected reward with high variance — by averaging
            over a large sample of seeded draws.

        Given: A HV reward model and a state whose realised next position is
            inside a dangerous zone but not on a wall.
        When: ``compute_reward`` is called over 2000 trials.
        Then: Each observed reward is exactly
            ``-step_cost ± dangerous_area_penalty`` and the sample mean of
            the danger contribution is within 0.5 of zero.

        Test type: unit
        """
        np.random.seed(42)
        model = _make_hv_model()
        # (5, 3) is a dangerous-area centre; not a wall.
        state = np.array([4.0, 3.0, 0.0, 0.0, 0.0])
        next_state = np.array([5.0, 3.0, 0.0, 0.0, 0.0])
        rewards = np.array(
            [model.compute_reward(state, action=1, next_state=next_state) for _ in range(2000)]
        )
        # Contributions are (-1 - 5) = -6 or (-1 + 5) = 4.
        contributions = rewards + _DEFAULT_STEP_COST  # subtract -(-1) i.e. add step_cost
        assert set(np.unique(contributions).tolist()) == {
            -_DEFAULT_DANGER_PENALTY,
            _DEFAULT_DANGER_PENALTY,
        }
        assert abs(float(np.mean(contributions))) < 0.5

    def test_batch_danger_zone_matches_scalar_distribution(self):
        """``compute_reward_batch`` HV draws keep the same ±penalty distribution.

        Purpose: Validates the batch path applies the same per-row ±penalty
            distribution as the scalar path so vectorised callers see the
            same expected reward and variance as per-particle callers.

        Given: A HV reward model and a (N, 5) state batch all at the same
            dangerous-area centre.
        When: ``compute_reward_batch`` is called once with N=4000.
        Then: Observed contributions are exactly ``±dangerous_area_penalty``
            and the sample mean is within 0.5 of zero.

        Test type: unit
        """
        np.random.seed(123)
        model = _make_hv_model()
        n = 4000
        states = np.tile(np.array([4.0, 3.0, 0.0, 0.0, 0.0]), (n, 1))
        next_states = np.tile(np.array([5.0, 3.0, 0.0, 0.0, 0.0]), (n, 1))
        rewards = model.compute_reward_batch(states, action=1, next_states=next_states)
        contributions = rewards + _DEFAULT_STEP_COST
        assert set(np.unique(contributions).tolist()) == {
            -_DEFAULT_DANGER_PENALTY,
            _DEFAULT_DANGER_PENALTY,
        }
        assert abs(float(np.mean(contributions))) < 0.5


class TestLaserTagDistanceDecayedHazardPenaltyRewardModel:
    """Behavioural tests for the Decaying-Hit-Probability variant."""

    def test_rejects_non_positive_decay(self):
        """``penalty_decay`` must be strictly positive.

        Purpose: Validates the constructor rejects nonsensical decay lengths.

        Given: A reward-model factory invoked with ``penalty_decay <= 0``.
        When: The constructor runs.
        Then: ``ValueError`` is raised.

        Test type: unit
        """
        with pytest.raises(ValueError, match="penalty_decay"):
            _make_decay_model(penalty_decay=0.0)
        with pytest.raises(ValueError, match="penalty_decay"):
            _make_decay_model(penalty_decay=-1.0)

    def test_wall_penalty_remains_deterministic(self):
        """Wall hits in the Decay model always subtract exactly ``dangerous_area_penalty``.

        Purpose: Validates that walls retain deterministic scoring while the
            dangerous-area contribution is the only stochastic piece.

        Given: A Decay reward model with a small ``penalty_decay`` so the
            stochastic danger contribution is essentially zero far from
            any centre, and a state whose realised next position is on a
            wall far from every dangerous-area centre.
        When: ``compute_reward`` is called many times with action 0.
        Then: Every call returns at most ``-step_cost - dangerous_area_penalty``
            (the wall penalty); the stochastic danger contribution is
            negligible at this distance.

        Test type: unit
        """
        np.random.seed(7)
        model = _make_decay_model(penalty_decay=0.1)
        # Wall (10, 6); nearest danger centre (7, 1) sits at distance ≈ 6.7 →
        # exp(-67) is vanishingly small, so the wall penalty dominates.
        state = np.array([9.0, 6.0, 0.0, 0.0, 0.0])
        next_state = np.array([10.0, 6.0, 0.0, 0.0, 0.0])
        rewards = [model.compute_reward(state, action=1, next_state=next_state) for _ in range(50)]
        assert all(
            r == pytest.approx(-_DEFAULT_STEP_COST - _DEFAULT_DANGER_PENALTY) for r in rewards
        )

    def test_at_centre_always_triggers_penalty(self):
        """When the realised position equals a dangerous-area centre, penalty fires every time.

        Purpose: Validates the boundary case where ``min_dist == 0`` so
            ``hit_prob == 1.0`` — every trial must emit the full penalty.

        Given: A Decay reward model and a state whose realised next position
            is exactly on a dangerous-area centre (not on a wall).
        When: ``compute_reward`` is called over 50 trials.
        Then: Every call returns ``-step_cost - dangerous_area_penalty``.

        Test type: unit
        """
        np.random.seed(7)
        model = _make_decay_model(penalty_decay=1.0)
        state = np.array([4.0, 3.0, 0.0, 0.0, 0.0])
        next_state = np.array([5.0, 3.0, 0.0, 0.0, 0.0])  # centre (5, 3)
        rewards = [model.compute_reward(state, action=1, next_state=next_state) for _ in range(50)]
        assert all(
            r == pytest.approx(-_DEFAULT_STEP_COST - _DEFAULT_DANGER_PENALTY) for r in rewards
        )

    def test_far_position_penalty_rate_decays(self):
        """Empirical hit rate ≈ ``exp(-dist / penalty_decay)``.

        Purpose: Validates that the batch path's hit rate follows the
            distance-decaying formula by sampling a large batch of identical
            far-away positions and comparing the observed rate to the
            analytic target.

        Given: A Decay reward model with ``penalty_decay = 2.0`` and a batch
            of 6000 states whose realised positions sit at distance 2.0 from
            the nearest dangerous-area centre.
        When: ``compute_reward_batch`` is called once.
        Then: The fraction of rows that received the danger penalty is
            within 0.05 of ``exp(-2.0 / 2.0) ≈ 0.368``.

        Test type: unit
        """
        np.random.seed(99)
        decay = 2.0
        model = _make_decay_model(penalty_decay=decay)
        n = 6000
        # Position (3, 3): nearest centre (5, 3) at distance 2.0; not a wall,
        # not within radius of any centre under the standard model.
        states = np.tile(np.array([3.0, 3.0, 0.0, 0.0, 0.0]), (n, 1))
        next_states = states.copy()
        rewards = model.compute_reward_batch(states, action=1, next_states=next_states)
        # Each row is either -step_cost (no penalty) or -step_cost - penalty.
        contributions = rewards + _DEFAULT_STEP_COST
        # Expected: 0 (no hit) or -dangerous_area_penalty.
        assert set(np.unique(contributions).tolist()) == {0.0, -_DEFAULT_DANGER_PENALTY}
        observed_rate = float(np.mean(contributions == -_DEFAULT_DANGER_PENALTY))
        expected_rate = float(np.exp(-2.0 / decay))
        assert abs(observed_rate - expected_rate) < 0.05


class TestLaserTagPOMDPRewardModelTypeWiring:
    """Sanity tests for the ``reward_model_type`` constructor parameter on the env."""

    @pytest.mark.parametrize(
        "rmt, expected_cls",
        [
            (RewardModelType.CONSTANT_HAZARD_PENALTY, LaserTagRewardModel),
            (RewardModelType.ZERO_MEAN_HAZARD_SHOCK, LaserTagZeroMeanHazardShockRewardModel),
            (
                RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
                LaserTagDistanceDecayedHazardPenaltyRewardModel,
            ),
        ],
    )
    def test_env_instantiates_correct_reward_model_subclass(self, rmt, expected_cls):
        """``LaserTagPOMDP`` builds the reward model class selected by ``reward_model_type``.

        Purpose: Validates the env's branching logic from ``reward_model_type``
            to the concrete reward-model class.

        Given: A ``LaserTagPOMDP`` constructed with a specific ``reward_model_type``.
        When: ``env.reward_model`` is inspected.
        Then: It is an instance of the expected concrete subclass.

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95, reward_model_type=rmt, penalty_decay=2.0)
        assert isinstance(env.reward_model, expected_cls)

    def test_env_default_reward_model_is_standard(self):
        """The default ``reward_model_type`` preserves the historical standard model.

        Purpose: Regression — ensures that omitting ``reward_model_type``
            yields the original deterministic-OR reward semantics so existing
            configs and tests keep their pre-refactor behaviour.

        Given: A ``LaserTagPOMDP`` constructed without specifying
            ``reward_model_type``.
        When: ``env.reward_model`` is inspected.
        Then: It is exactly a ``LaserTagRewardModel`` (not a subclass).

        Test type: unit
        """
        env = LaserTagPOMDP(discount_factor=0.95)
        assert (
            type(env.reward_model) is LaserTagRewardModel
        )  # pylint: disable=unidiomatic-typecheck
