# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

"""Tests for the PacMan POMDP reward-model variants.

Covers the high-variance and decaying-hit-probability variants added
alongside the standard model and exposed via
:class:`PacManPOMDP`'s ``reward_model_type`` parameter. The existing
standard-model behaviour is already exercised by
``test_pacman_pomdp.py``.
"""

from typing import Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import (
    PacManPOMDP,
    RewardModelType,
)
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_utils.pacman_reward_models import (
    BasePacManRewardModel,
    PacManDistanceDecayedHazardPenaltyRewardModel,
    PacManZeroMeanHazardShockRewardModel,
    PacManRewardModel,
)


# Default maze / hazard configuration shared by the scenario-style tests.
# A 7x7 maze with one zone centred at (3, 3) radius 1.0 mirrors the
# ``test_pacman_pomdp.py`` scenarios for the standard model.
_MAZE_SIZE: Tuple[int, int] = (7, 7)
_DANGER_CENTRE: Tuple[int, int] = (3, 3)
_DANGER_RADIUS = 1.0
_DANGER_PENALTY = 5.0
_STEP_PENALTY = -1.0


def _make_env(
    reward_model_type: RewardModelType,
    penalty_decay: float = 1.0,
) -> PacManPOMDP:
    return PacManPOMDP(
        maze_size=_MAZE_SIZE,
        walls=set(),
        initial_pacman_pos=(0, 0),
        num_ghosts=1,
        initial_ghost_positions=[(6, 6)],
        initial_pellets=[(1, 1)],
        dangerous_areas={_DANGER_CENTRE},
        dangerous_area_radius=_DANGER_RADIUS,
        dangerous_area_penalty=_DANGER_PENALTY,
        step_penalty=_STEP_PENALTY,
        ghost_collision_penalty=-100.0,
        pellet_reward=10.0,
        win_reward=100.0,
        reward_model_type=reward_model_type,
        penalty_decay=penalty_decay,
    )


def _state_with_pac(env: PacManPOMDP, pacman_pos: Tuple[int, int]) -> np.ndarray:
    return env.make_state(
        pacman_pos=pacman_pos,
        ghost_positions=((6, 6),),
        pellets=((1, 1),),
        score=0.0,
        terminal=False,
    )


class TestPacManZeroMeanHazardShockRewardModel:
    """Behavioural tests for the HV variant."""

    def test_subclass_inherits_base(self):
        """HV model is both a ``BasePacManRewardModel`` and a ``PacManRewardModel``.

        Purpose: Validates the class hierarchy so callers that expect either
            base type can interchangeably use the HV variant.

        Given: A ``PacManPOMDP`` constructed with the HV reward variant.
        When: ``isinstance`` is checked against the abstract and concrete bases.
        Then: Both checks return True.

        Test type: unit
        """
        env = _make_env(RewardModelType.ZERO_MEAN_HAZARD_SHOCK)
        assert isinstance(env.reward_model, PacManRewardModel)
        assert isinstance(env.reward_model, BasePacManRewardModel)
        assert isinstance(env.reward_model, PacManZeroMeanHazardShockRewardModel)

    def test_outside_zone_matches_step_penalty(self):
        """Positions outside every zone score exactly the step penalty.

        Purpose: Validates the HV variant leaves non-dangerous-area transitions
            untouched — only the danger contribution changes versus the standard
            model.

        Given: An HV env and a transition that ends outside the single zone
            (distance > radius).
        When: ``reward(state, action, next_state)`` is called.
        Then: Reward equals ``step_penalty`` exactly.

        Test type: unit
        """
        env = _make_env(RewardModelType.ZERO_MEAN_HAZARD_SHOCK)
        state = _state_with_pac(env, (0, 0))
        next_state = _state_with_pac(env, (0, 1))
        reward = env.reward_model.compute_reward(state, action=1, next_state=next_state)
        assert reward == pytest.approx(_STEP_PENALTY)

    def test_danger_zone_emits_symmetric_penalty(self):
        """HV dangerous-area contribution is ``±dangerous_area_penalty`` with mean ≈ 0.

        Purpose: Validates the variant's signature property — danger zones
            contribute zero expected reward with high variance — by averaging
            over a large sample of seeded draws.

        Given: An HV reward model and a transition whose realised next position
            is on a zone centre.
        When: ``compute_reward`` is called over 2000 trials.
        Then: Each observed reward is exactly
            ``step_penalty ± dangerous_area_penalty`` and the sample mean of
            the danger contribution is within 0.5 of zero.

        Test type: unit
        """
        np.random.seed(42)
        env = _make_env(RewardModelType.ZERO_MEAN_HAZARD_SHOCK)
        state = _state_with_pac(env, (3, 2))
        next_state = _state_with_pac(env, _DANGER_CENTRE)
        rewards = np.array(
            [
                env.reward_model.compute_reward(state, action=1, next_state=next_state)
                for _ in range(2000)
            ]
        )
        # Contributions are (step_penalty - penalty) or (step_penalty + penalty).
        contributions = rewards - _STEP_PENALTY
        assert set(np.unique(contributions).tolist()) == {-_DANGER_PENALTY, _DANGER_PENALTY}
        assert abs(float(np.mean(contributions))) < 0.5

    def test_batch_danger_zone_matches_scalar_distribution(self):
        """``compute_reward_batch`` HV draws keep the same ±penalty distribution.

        Purpose: Validates the batch path applies the same per-row ±penalty
            distribution as the scalar path so vectorised callers see the same
            expected reward and variance as per-particle callers.

        Given: An HV reward model and an ``(N, state_dim)`` state batch all
            entering the same zone centre.
        When: ``compute_reward_batch`` is called once with ``N=4000``.
        Then: Observed contributions are exactly ``±dangerous_area_penalty``
            and the sample mean is within 0.5 of zero.

        Test type: unit
        """
        np.random.seed(123)
        env = _make_env(RewardModelType.ZERO_MEAN_HAZARD_SHOCK)
        n = 4000
        state = _state_with_pac(env, (3, 2))
        next_state = _state_with_pac(env, _DANGER_CENTRE)
        states = np.tile(state, (n, 1))
        next_states = np.tile(next_state, (n, 1))
        rewards = env.reward_model.compute_reward_batch(states, action=1, next_states=next_states)
        contributions = rewards - _STEP_PENALTY
        assert set(np.unique(contributions).tolist()) == {-_DANGER_PENALTY, _DANGER_PENALTY}
        assert abs(float(np.mean(contributions))) < 0.5

    def test_terminal_state_returns_zero(self):
        """Terminal states never accrue reward, even in a zone.

        Purpose: Pins the terminal-zero contract on the HV variant so future
            kernel changes cannot accidentally start scoring danger
            contributions for terminal rows.

        Given: An HV reward model and a terminal state.
        When: ``compute_reward`` is called with a transition into the zone.
        Then: Reward is exactly 0.0 on every trial.

        Test type: unit
        """
        np.random.seed(7)
        env = _make_env(RewardModelType.ZERO_MEAN_HAZARD_SHOCK)
        state = env.make_state(
            pacman_pos=(3, 3),
            ghost_positions=((6, 6),),
            pellets=((1, 1),),
            score=0.0,
            terminal=True,
        )
        next_state = _state_with_pac(env, _DANGER_CENTRE)
        for _ in range(30):
            assert env.reward_model.compute_reward(state, action=1, next_state=next_state) == 0.0


class TestPacManDistanceDecayedHazardPenaltyRewardModel:
    """Behavioural tests for the Decaying-Hit-Probability variant."""

    def test_rejects_non_positive_decay(self):
        """``penalty_decay`` must be strictly positive.

        Purpose: Validates the constructor rejects nonsensical decay lengths.

        Given: An env factory invoked with ``penalty_decay <= 0``.
        When: The constructor runs.
        Then: ``ValueError`` is raised with a message mentioning ``penalty_decay``.

        Test type: unit
        """
        with pytest.raises(ValueError, match="penalty_decay"):
            _make_env(RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, penalty_decay=0.0)
        with pytest.raises(ValueError, match="penalty_decay"):
            _make_env(RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, penalty_decay=-1.0)

    def test_at_centre_always_triggers_penalty(self):
        """When the realised position equals a centre, penalty fires every time.

        Purpose: Validates the boundary case where ``min_dist == 0`` so
            ``hit_prob == 1.0`` — every trial must emit the full penalty.

        Given: A Decay reward model and a transition whose realised next
            position is exactly on a zone centre.
        When: ``compute_reward`` is called over 50 trials.
        Then: Every call returns ``step_penalty - dangerous_area_penalty``.

        Test type: unit
        """
        np.random.seed(7)
        env = _make_env(RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, penalty_decay=1.0)
        state = _state_with_pac(env, (3, 2))
        next_state = _state_with_pac(env, _DANGER_CENTRE)
        for _ in range(50):
            reward = env.reward_model.compute_reward(state, action=1, next_state=next_state)
            assert reward == pytest.approx(_STEP_PENALTY - _DANGER_PENALTY)

    def test_far_position_penalty_rate_decays(self):
        """Empirical hit rate ≈ ``exp(-dist / penalty_decay)``.

        Purpose: Validates the batch path's hit rate follows the
            distance-decaying formula by sampling a large batch of identical
            far-away positions and comparing the observed rate to the analytic
            target.

        Given: A Decay reward model with ``penalty_decay = 2.0`` and a batch
            of 6000 states whose realised positions sit at distance 2.0 from
            the nearest zone centre.
        When: ``compute_reward_batch`` is called once.
        Then: The fraction of rows that received the danger penalty is within
            0.05 of ``exp(-2.0 / 2.0) ≈ 0.368``.

        Test type: unit
        """
        np.random.seed(99)
        decay = 2.0
        env = _make_env(RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, penalty_decay=decay)
        n = 6000
        # Position (3, 1): nearest centre (3, 3) at distance 2.0. Use action 4
        # (stay) so the batch path's neighbor-table lookup keeps the realised
        # next position at (3, 1) instead of shifting it by the action vector.
        state = _state_with_pac(env, (3, 1))
        states = np.tile(state, (n, 1))
        next_states = states.copy()
        rewards = env.reward_model.compute_reward_batch(states, action=4, next_states=next_states)
        contributions = rewards - _STEP_PENALTY
        assert set(np.unique(contributions).tolist()) == {0.0, -_DANGER_PENALTY}
        observed_rate = float(np.mean(contributions == -_DANGER_PENALTY))
        expected_rate = float(np.exp(-2.0 / decay))
        assert abs(observed_rate - expected_rate) < 0.05

    def test_no_zones_returns_step_penalty_only(self):
        """With no zones configured the decay contribution is identically zero.

        Purpose: Guards the no-op path — when ``dangerous_areas`` is empty the
            decay kernel must not be called and the reward must equal
            ``step_penalty``.

        Given: A Decay-variant env constructed without dangerous areas.
        When: ``compute_reward`` is called on an arbitrary transition.
        Then: Reward equals exactly ``step_penalty``.

        Test type: unit
        """
        env = PacManPOMDP(
            maze_size=_MAZE_SIZE,
            walls=set(),
            initial_pacman_pos=(0, 0),
            num_ghosts=1,
            initial_ghost_positions=[(6, 6)],
            initial_pellets=[(1, 1)],
            step_penalty=_STEP_PENALTY,
            ghost_collision_penalty=-100.0,
            pellet_reward=10.0,
            win_reward=100.0,
            reward_model_type=RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
            penalty_decay=1.0,
        )
        state = _state_with_pac(env, (0, 0))
        next_state = _state_with_pac(env, (0, 1))
        for _ in range(20):
            reward = env.reward_model.compute_reward(state, action=1, next_state=next_state)
            assert reward == pytest.approx(_STEP_PENALTY)

    def test_terminal_state_returns_zero(self):
        """Terminal states never accrue reward, even when the realised next position is at a centre.

        Purpose: Pins the terminal-zero contract on the Decay variant.

        Given: A Decay env and a terminal state with a realised next position at a zone centre.
        When: ``compute_reward`` is called over 30 trials.
        Then: Reward is exactly 0.0 every trial.

        Test type: unit
        """
        np.random.seed(7)
        env = _make_env(RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY, penalty_decay=1.0)
        state = env.make_state(
            pacman_pos=(3, 3),
            ghost_positions=((6, 6),),
            pellets=((1, 1),),
            score=0.0,
            terminal=True,
        )
        next_state = _state_with_pac(env, _DANGER_CENTRE)
        for _ in range(30):
            assert env.reward_model.compute_reward(state, action=1, next_state=next_state) == 0.0


class TestPacManPOMDPRewardModelTypeWiring:
    """Sanity tests for the ``reward_model_type`` constructor parameter on the env."""

    @pytest.mark.parametrize(
        "rmt, expected_cls",
        [
            (RewardModelType.CONSTANT_HAZARD_PENALTY, PacManRewardModel),
            (RewardModelType.ZERO_MEAN_HAZARD_SHOCK, PacManZeroMeanHazardShockRewardModel),
            (
                RewardModelType.DISTANCE_DECAYED_HAZARD_PENALTY,
                PacManDistanceDecayedHazardPenaltyRewardModel,
            ),
        ],
    )
    def test_env_instantiates_correct_reward_model_subclass(self, rmt, expected_cls):
        """``PacManPOMDP`` builds the reward model class selected by ``reward_model_type``.

        Purpose: Validates the env's branching logic from ``reward_model_type``
            to the concrete reward-model class.

        Given: A ``PacManPOMDP`` constructed with a specific ``reward_model_type``.
        When: ``env.reward_model`` is inspected.
        Then: It is an instance of the expected concrete subclass.

        Test type: unit
        """
        env = _make_env(rmt, penalty_decay=2.0)
        assert isinstance(env.reward_model, expected_cls)

    def test_env_default_reward_model_is_standard(self):
        """The default ``reward_model_type`` preserves the historical standard model.

        Purpose: Regression — ensures that omitting ``reward_model_type``
            yields the original deterministic reward semantics so existing
            configs and tests keep their pre-refactor behaviour.

        Given: A ``PacManPOMDP`` constructed without specifying ``reward_model_type``.
        When: ``env.reward_model`` is inspected.
        Then: It is exactly a ``PacManRewardModel`` (not a subclass).

        Test type: unit
        """
        env = PacManPOMDP(maze_size=_MAZE_SIZE)
        # pylint: disable-next=unidiomatic-typecheck
        assert type(env.reward_model) is PacManRewardModel


# Helper builders for swap-collision tests. The standard ``_state_with_pac``
# uses a fixed ghost cell at (6, 6) which is incompatible with a swap test —
# the swap test needs the ghost adjacent to pacman in the pre-state.
def _make_swap_state(
    env: PacManPOMDP, pacman_pos: Tuple[int, int], ghost_pos: Tuple[int, int]
) -> np.ndarray:
    return env.make_state(
        pacman_pos=pacman_pos,
        ghost_positions=(ghost_pos,),
        pellets=((1, 1),),
        score=0.0,
        terminal=False,
    )


def _make_terminal_after_swap(
    env: PacManPOMDP,
    pacman_pos: Tuple[int, int],
    ghost_pos: Tuple[int, int],
) -> np.ndarray:
    return env.make_state(
        pacman_pos=pacman_pos,
        ghost_positions=(ghost_pos,),
        pellets=((1, 1),),
        score=0.0,
        terminal=True,
    )


class TestPacManGhostSwapCollisionDetection:
    """Regression coverage for the ghost-swap arm of the collision rule.

    The C++ transition (``apply_transition``) and C++ rollout reward both
    treat pacman and a ghost trading cells in a single step as a collision
    that terminates and accrues ``ghost_collision_penalty``. The Python
    reward kernels previously only detected same-cell collisions and so
    silently lost the penalty on every swap, breaking C++/Python reward
    parity at the per-step level.
    """

    def test_scalar_swap_arm_credits_collision_penalty(self):
        """Scalar kernel charges ghost_collision_penalty on a pacman-ghost swap.

        Purpose: Validates that ``compute_reward`` detects the swap arm
            of the canonical Pacman collision rule (pacman and a ghost
            trade cells) and credits ``ghost_collision_penalty`` exactly
            like the C++ rollout reward kernel.

        Given: A standard reward model with pacman at (0, 0) and a ghost
            at (0, 1) in the pre-state, a next-state where pacman moved
            east to (0, 1) and the ghost is now at (0, 0) with terminal
            flag set (mirroring what the C++ transition writes on a swap
            collision). Positions sit well outside the danger zone so the
            expected value isolates the step + collision contributions.
        When: ``compute_reward(state, action=1, next_state=next_state)``
            is called on the env's reward model.
        Then: Reward equals ``step_penalty + ghost_collision_penalty``.

        Test type: unit
        """
        env = _make_env(RewardModelType.CONSTANT_HAZARD_PENALTY)
        # Use cells well clear of the danger zone (centred at (3, 3), radius 1)
        # so the expected reward depends only on step + collision penalties.
        # Action 1 (east) moves pacman (0, 0) -> (0, 1); ghost (0, 1) -> (0, 0)
        # completes the swap.
        state = _make_swap_state(env, pacman_pos=(0, 0), ghost_pos=(0, 1))
        next_state = _make_terminal_after_swap(env, pacman_pos=(0, 1), ghost_pos=(0, 0))
        reward = env.reward_model.compute_reward(state, action=1, next_state=next_state)
        assert reward == pytest.approx(_STEP_PENALTY + env.ghost_collision_penalty)

    def test_batch_swap_arm_credits_collision_penalty(self):
        """Batch kernel charges ghost_collision_penalty on a pacman-ghost swap.

        Purpose: Validates that ``compute_reward_batch`` detects the same
            swap arm as the scalar path so vectorised callers (e.g. PFT
            planners) do not silently drop the collision penalty.

        Given: A standard reward model, a batch with one row whose
            pre-state has pacman at (0, 0) and a ghost at (0, 1), and a
            matching next-state batch where pacman moved east to (0, 1)
            and the ghost is at (0, 0) (a swap, with terminal set).
            Positions are clear of the danger zone.
        When: ``compute_reward_batch(states, action=1, next_states=...)``
            is called.
        Then: The single returned reward equals
            ``step_penalty + ghost_collision_penalty``.

        Test type: unit
        """
        env = _make_env(RewardModelType.CONSTANT_HAZARD_PENALTY)
        # Use cells well clear of the danger zone (centred at (3, 3), radius 1)
        # so the expected reward depends only on step + collision penalties.
        state = _make_swap_state(env, pacman_pos=(0, 0), ghost_pos=(0, 1))
        next_state = _make_terminal_after_swap(env, pacman_pos=(0, 1), ghost_pos=(0, 0))
        states = state.reshape(1, -1)
        next_states = next_state.reshape(1, -1)
        rewards = env.reward_model.compute_reward_batch(states, action=1, next_states=next_states)
        assert rewards.shape == (1,)
        assert float(rewards[0]) == pytest.approx(_STEP_PENALTY + env.ghost_collision_penalty)

    def test_scalar_and_batch_agree_on_swap(self):
        """Scalar and batch reward kernels agree on a swap-collision row.

        Purpose: Pins scalar/batch parity for the swap arm so future
            kernel edits cannot diverge the two code paths.

        Given: The same swap-collision (state, action, next_state) used
            by the dedicated scalar and batch tests above.
        When: Both ``compute_reward`` and ``compute_reward_batch`` are
            invoked on the standard reward model.
        Then: The two outputs are exactly equal.

        Test type: unit
        """
        env = _make_env(RewardModelType.CONSTANT_HAZARD_PENALTY)
        # Use cells well clear of the danger zone (centred at (3, 3), radius 1)
        # so the expected reward depends only on step + collision penalties.
        state = _make_swap_state(env, pacman_pos=(0, 0), ghost_pos=(0, 1))
        next_state = _make_terminal_after_swap(env, pacman_pos=(0, 1), ghost_pos=(0, 0))
        scalar = env.reward_model.compute_reward(state, action=1, next_state=next_state)
        batch = env.reward_model.compute_reward_batch(
            state.reshape(1, -1), action=1, next_states=next_state.reshape(1, -1)
        )
        assert float(batch[0]) == pytest.approx(scalar)
