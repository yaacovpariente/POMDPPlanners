# SPDX-License-Identifier: MIT

"""Tests for the maze vectorized particle belief updaters.

The updaters reimplement the environment's event rule in NumPy, so every test
here is a parity test: the batched path must agree with the environment's own
scalar path on the same states, for every action and every observation label.
A drifting copy is the failure mode these guard, since nothing else would
notice a belief that walks through a wall the world refuses.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.maze_pomdp.maze_pomdp import (
    ACTIONS,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    OBSERVATIONS,
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    create_maze_state,
)
from POMDPPlanners.environments.maze_pomdp.maze_pomdp_beliefs import (
    ContinuousMazeVectorizedUpdater,
    DiscreteMazeVectorizedUpdater,
    MazeVectorizedWeightedParticleBelief,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief

# The environment's own cell tolerance. The continuous updater stops a step at
# the tolerance-widened cell boundary rather than the exact one, so positions
# agree to within it and not to the last bit.
_POSITION_ATOL = 1e-7


@pytest.fixture
def discrete_env():
    return DiscreteMazePOMDP(discount_factor=0.95, maze_width=11, maze_height=13, maze_seed=3)


@pytest.fixture
def continuous_env():
    return ContinuousMazePOMDP(
        discount_factor=0.95,
        maze_width=11,
        maze_height=13,
        maze_seed=3,
        max_step_size=1.0,
    )


def _spread_states(env, count, rng, jitter=0.0):
    """States covering every walkable cell, both goal sides and every cue phase."""
    cells = sorted(env.walkable_cells)
    picked = [cells[index] for index in rng.integers(0, len(cells), size=count)]
    goals = rng.integers(0, 2, size=count).astype(float)
    phases = rng.integers(0, 3, size=count).astype(float)
    states = np.array(
        [[cell[0], cell[1], goal, phase] for cell, goal, phase in zip(picked, goals, phases)],
        dtype=np.float64,
    )
    if jitter:
        states[:, :2] += rng.uniform(-jitter, jitter, size=(count, 2))
    return states


class TestDiscreteMazeVectorizedUpdater:
    def test_batch_transition_matches_environment(self, discrete_env):
        """Test that the batched step equals the environment's own step.

        Purpose: The updater duplicates the environment's lookup rule; this is
        the test that the duplicate has not drifted.

        Given: Particles spread over every walkable cell, goal side and cue phase.
        When: Each action is applied through both paths.
        Then: The successor arrays are identical.

        Test type: unit
        """
        updater = DiscreteMazeVectorizedUpdater.from_environment(discrete_env)
        states = _spread_states(discrete_env, 400, np.random.default_rng(0))

        for action in ACTIONS:
            batched = updater.batch_transition(states, action)
            scalar = discrete_env.sample_next_state_batch(states, action)
            np.testing.assert_array_equal(batched, scalar, err_msg=f"action {action}")

    def test_wall_refuses_the_move_and_still_consumes_an_armed_cue(self, discrete_env):
        """Test that a refused move leaves the position and spends the cue.

        Purpose: The cue is single-use; a wall collision must not turn standing
        on the cue cell into a way to read it twice.

        Given: A particle on the cue cell with the cue emitting.
        When: Every action is applied.
        Then: The cue phase is consumed whether or not the move was refused.

        Test type: unit
        """
        updater = DiscreteMazeVectorizedUpdater.from_environment(discrete_env)
        state = create_maze_state(discrete_env.cue_cell, GOAL_LEFT, CUE_EMITTING)

        for action in ACTIONS:
            batched = updater.batch_transition(state[np.newaxis, :], action)
            scalar = discrete_env.sample_next_state(state, action)
            np.testing.assert_array_equal(batched[0], scalar, err_msg=f"action {action}")

    def test_goal_states_are_absorbing(self, discrete_env):
        """Test that a particle in a goal cell never moves again.

        Purpose: An over-long rollout must not walk a particle back out of a
        terminal state and collect its payout twice.

        Given: Particles standing in each goal cell.
        When: Every action is applied.
        Then: The particles come back unchanged.

        Test type: unit
        """
        updater = DiscreteMazeVectorizedUpdater.from_environment(discrete_env)
        states = np.stack(
            [
                create_maze_state(discrete_env.left_goal_cell, GOAL_LEFT, CUE_UNSEEN),
                create_maze_state(discrete_env.right_goal_cell, GOAL_RIGHT, CUE_UNSEEN),
            ]
        )

        for action in ACTIONS:
            np.testing.assert_array_equal(updater.batch_transition(states, action), states)

    @pytest.mark.parametrize("observation", OBSERVATIONS)
    def test_observation_log_likelihood_matches_environment(self, discrete_env, observation):
        """Test that the batched likelihood equals the environment's.

        Purpose: The reweighting step decides what the belief believes; a
        likelihood that disagrees with the world's is the quiet failure.

        Given: Particles in every cue phase and on both goal sides.
        When: Each observation label is scored through both paths.
        Then: The log-likelihood arrays are identical, infinities included.

        Test type: unit
        """
        updater = DiscreteMazeVectorizedUpdater.from_environment(discrete_env)
        states = _spread_states(discrete_env, 200, np.random.default_rng(1))
        code = float(OBSERVATIONS.index(observation))

        batched = updater.batch_observation_log_likelihood(states, ACTIONS[0], np.array(code))
        scalar = discrete_env.observation_log_probability_per_state(states, ACTIONS[0], observation)
        np.testing.assert_array_equal(batched, scalar)

    def test_config_id_is_stable_and_separates_maps(self, discrete_env):
        """Test that the identity is deterministic and map-specific.

        Purpose: ``config_id`` is a cache key. A key that misses the map would
        serve one maze's cached beliefs for another.

        Given: Two updaters on one environment, and one on a different seed.
        When: Their identities are compared.
        Then: The pair agrees and the different map does not.

        Test type: unit
        """
        first = DiscreteMazeVectorizedUpdater.from_environment(discrete_env)
        second = DiscreteMazeVectorizedUpdater.from_environment(discrete_env)
        other_map = DiscreteMazeVectorizedUpdater.from_environment(
            DiscreteMazePOMDP(discount_factor=0.95, maze_width=11, maze_height=13, maze_seed=4)
        )

        assert first.config_id == second.config_id
        assert first.config_id != other_map.config_id


class TestContinuousMazeVectorizedUpdater:
    def test_batch_transition_matches_environment(self, continuous_env):
        """Test that the batched sweep equals the environment's swept path.

        Purpose: The continuous rule is the hard one to duplicate -- walls
        refuse the whole move, goals stop it partway, and ties are judged walls
        first.

        Given: Off-centre particles and displacements in every direction,
            including some longer than ``max_step_size``.
        When: Each displacement is applied through both paths.
        Then: The successors agree to within the environment's cell tolerance.

        Test type: unit
        """
        updater = ContinuousMazeVectorizedUpdater.from_environment(continuous_env)
        rng = np.random.default_rng(2)
        states = _spread_states(continuous_env, 300, rng, jitter=0.45)

        for action in rng.uniform(-1.2, 1.2, size=(40, 2)):
            batched = updater.batch_transition(states, action)
            scalar = continuous_env.sample_next_state_batch(states, action)
            np.testing.assert_allclose(
                batched, scalar, atol=_POSITION_ATOL, err_msg=f"action {action}"
            )

    def test_zero_displacement_keeps_the_position(self, continuous_env):
        """Test that a zero action is legal and moves nothing.

        Purpose: A zero vector is a legal action here, and treating it as a
        degenerate segment must not divide by its own length.

        Given: Off-centre particles.
        When: The zero displacement is applied.
        Then: Positions are unchanged and match the environment.

        Test type: unit
        """
        updater = ContinuousMazeVectorizedUpdater.from_environment(continuous_env)
        states = _spread_states(continuous_env, 50, np.random.default_rng(3), jitter=0.3)

        batched = updater.batch_transition(states, np.zeros(2))
        scalar = continuous_env.sample_next_state_batch(states, np.zeros(2))

        np.testing.assert_allclose(batched[:, :2], states[:, :2], atol=_POSITION_ATOL)
        np.testing.assert_allclose(batched, scalar, atol=_POSITION_ATOL)

    def test_long_action_is_scaled_not_rejected(self, continuous_env):
        """Test that an over-long displacement is capped the way the world caps it.

        Purpose: A planner sampling from the wrong disc still produces legal
        moves; the belief has to cap them identically or it tracks a faster
        agent than the one being simulated.

        Given: A displacement twice the maximum step size.
        When: It is applied through both paths.
        Then: The successors agree.

        Test type: unit
        """
        updater = ContinuousMazeVectorizedUpdater.from_environment(continuous_env)
        states = _spread_states(continuous_env, 40, np.random.default_rng(4), jitter=0.2)
        action = np.array([2.0, 0.0])

        np.testing.assert_allclose(
            updater.batch_transition(states, action),
            continuous_env.sample_next_state_batch(states, action),
            atol=_POSITION_ATOL,
        )


class TestMazeBeliefFactory:
    def test_default_belief_is_vectorized(self, discrete_env):
        """Test that the factory hands back the vectorized belief by default.

        Purpose: Registration is what makes the updater reachable; an
        unregistered environment silently falls back to the generic filter.

        Given: A discrete maze environment.
        When: The top-level factory is asked for its belief.
        Then: A maze vectorized particle belief comes back.

        Test type: unit
        """
        belief = create_environment_belief(discrete_env, n_particles=32)
        assert isinstance(belief, MazeVectorizedWeightedParticleBelief)

    def test_particle_type_still_available(self, continuous_env):
        """Test that the scalar particle belief is still selectable.

        Purpose: The vectorized belief is the default, not the only option;
        comparisons against the generic filter have to stay possible.

        Given: A continuous maze environment.
        When: ``BeliefType.PARTICLE`` is requested.
        Then: A belief comes back that is not the vectorized one.

        Test type: unit
        """
        belief = create_environment_belief(
            continuous_env, belief_type=BeliefType.PARTICLE, n_particles=32
        )
        assert not isinstance(belief, MazeVectorizedWeightedParticleBelief)

    def test_cue_reading_concentrates_the_belief(self, discrete_env):
        """Test that a cue reading moves the posterior onto the named goal side.

        Purpose: The maze hides exactly one bit, and reading the cue is the only
        way to learn it. A belief that does not shift on the reading is not
        tracking the task.

        Given: The prior, with half its mass on each goal side.
        When: The agent steps onto the cue cell and reads ``"left_cue"``.
        Then: More than half the weight sits on the left goal side.

        Test type: integration
        """
        np.random.seed(0)
        belief = create_environment_belief(discrete_env, n_particles=200)

        belief = belief.update("up", OBSERVATION_LEFT_CUE, discrete_env)

        left_weight = belief.normalized_weights[belief.particles[:, 2] == GOAL_LEFT].sum()
        assert left_weight > 0.5

    def test_unknown_observation_label_is_impossible_not_an_error(self, discrete_env):
        """Test that an unrecognised label scores ``-inf`` rather than raising.

        Purpose: A label the environment cannot emit is impossible under every
        particle. Raising from inside a tree search would take the run down
        instead of ruling the branch out.

        Given: The prior belief.
        When: It is updated with a label that is not in the alphabet.
        Then: The update returns and every log-weight is ``-inf``.

        Test type: unit
        """
        np.random.seed(0)
        belief = create_environment_belief(discrete_env, n_particles=16)

        updated = belief.update("up", "not_an_observation", discrete_env)

        assert np.all(np.isneginf(updated.log_weights))

    @pytest.mark.parametrize(
        "observation", [OBSERVATION_LEFT_CUE, OBSERVATION_RIGHT_CUE, OBSERVATION_EMPTY]
    )
    def test_update_keeps_the_particle_count(self, continuous_env, observation):
        """Test that an update returns the same number of particles.

        Purpose: A filter that quietly sheds particles degrades over an episode
        in a way no single update reveals.

        Given: A continuous maze belief.
        When: It is updated with each observation label.
        Then: The particle count is unchanged.

        Test type: unit
        """
        np.random.seed(0)
        belief = create_environment_belief(continuous_env, n_particles=48)

        updated = belief.update(np.array([0.0, 0.5]), observation, continuous_env)

        assert updated.particles.shape == (48, 4)
