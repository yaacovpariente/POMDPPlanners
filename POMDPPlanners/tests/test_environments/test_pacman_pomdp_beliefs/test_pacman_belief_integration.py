"""Integration tests for PacMan vectorized belief."""

# pylint: disable=protected-access

import numpy as np
import pytest

from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.cost import belief_expectation_reward
from POMDPPlanners.environments.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp_beliefs.pacman_belief_factory import (
    create_pacman_belief,
)
from POMDPPlanners.utils.belief_factory import BeliefType


@pytest.fixture()
def env():
    return PacManPOMDP(
        maze_size=(5, 5),
        walls={(2, 2)},
        initial_pellets=[(1, 1), (3, 3)],
        initial_pacman_pos=(0, 0),
        num_ghosts=1,
        initial_ghost_positions=[(4, 4)],
        discount_factor=0.95,
    )


class TestBeliefUpdateCycle:
    def test_update_preserves_shapes(self, env):
        """Test that belief update preserves particle array shapes.

        Purpose: Validates belief update cycle with vectorized belief.

        Given: A vectorized belief with 50 particles.
        When: update is called with an action and observation.
        Then: Updated belief has same particle count and state dim.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_pacman_belief(
            env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=50
        )
        action = 2  # South
        obs = np.array([4.0, 4.0])  # Observed ghost at (4,4)
        updated = belief.update(action=action, observation=obs, pomdp=env)
        assert isinstance(updated, VectorizedWeightedParticleBelief)
        assert updated.particles.shape == (50, env._state_dim)
        assert updated.log_weights.shape == (50,)


class TestBeliefSampleValidState:
    def test_sample_converts_to_valid_state(self, env):
        """Test that sampled array state converts to valid PacManState.

        Purpose: Validates end-to-end sample -> state conversion.

        Given: A vectorized belief created from the environment.
        When: A state is sampled and converted to PacManState.
        Then: The PacManState has valid fields.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_pacman_belief(env, n_particles=50)
        arr = belief.sample()
        state = env.array_to_state(arr)
        assert isinstance(state.pacman_pos, tuple)
        assert len(state.pacman_pos) == 2
        assert isinstance(state.ghost_positions, tuple)
        assert len(state.ghost_positions) == env.num_ghosts


class TestBeliefExpectationReward:
    def test_returns_float(self, env):
        """Test that belief_expectation_reward returns a float.

        Purpose: Validates reward expectation computation with vectorized belief.

        Given: A vectorized belief and a PacManPOMDP environment.
        When: belief_expectation_reward is called.
        Then: A finite float is returned.

        Test type: integration
        """
        np.random.seed(42)
        belief = create_pacman_belief(env, n_particles=50)
        reward = belief_expectation_reward(belief=belief, action=2, env=env)
        assert isinstance(reward, float)
        assert np.isfinite(reward)
