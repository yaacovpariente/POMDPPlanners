# SPDX-License-Identifier: MIT

"""Tests for LaserTagVectorizedUpdater.

This module tests the vectorized batch transition and observation
log-likelihood methods for the LaserTag POMDP, including equivalence tests
that verify the vectorized results match the per-particle loop.

Note on stochastic equivalence: The opponent movement is stochastic and
the vectorized path samples from a different RNG sequence than the
per-particle loop (one bulk ``np.random.random(K)`` vs sequential
``np.random.choice`` calls). Therefore robot positions are tested for
exact match while opponent positions are tested statistically.
"""

from collections import Counter

import numpy as np
import pytest

from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.environments.laser_tag_pomdp import LaserTagPOMDP, OpponentPolicy
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp_beliefs import (
    LaserTagVectorizedUpdater,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_obs_log_likelihood_matches_loop,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    laser_tag_pinned_kwargs as _lt_pinned_kwargs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return LaserTagPOMDP(
        discount_factor=0.95,
        **_lt_pinned_kwargs(
            floor_shape=(11, 7),
            walls={(1, 2), (3, 0), (3, 4), (5, 0), (6, 4), (9, 1), (9, 4), (10, 6)},
            measurement_noise=1.0,
        ),
    )


@pytest.fixture
def env_no_walls():
    return LaserTagPOMDP(
        discount_factor=0.95,
        **_lt_pinned_kwargs(
            floor_shape=(5, 5),
            walls=set(),
            measurement_noise=1.0,
        ),
    )


@pytest.fixture
def updater(env):
    return LaserTagVectorizedUpdater.from_environment(env)


@pytest.fixture
def updater_no_walls(env_no_walls):
    return LaserTagVectorizedUpdater.from_environment(env_no_walls)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFromEnvironment:
    def test_from_environment_creates_updater(self, env):
        """Test that from_environment constructs a valid updater.

        Purpose: Validates the factory classmethod.

        Given: A LaserTagPOMDP instance.
        When: from_environment is called.
        Then: A LaserTagVectorizedUpdater is returned with matching parameters.

        Test type: unit
        """
        updater = LaserTagVectorizedUpdater.from_environment(env)
        assert isinstance(updater, LaserTagVectorizedUpdater)
        assert updater.floor_shape == env.floor_shape
        assert updater.measurement_noise == env.measurement_noise
        assert updater.transition_error_prob == env.transition_error_prob

    def test_valid_cell_shape(self, updater, env):
        """Test that valid_cell has correct shape.

        Purpose: Validates valid_cell array shape matches floor_shape.

        Given: A LaserTagPOMDP with floor_shape (11, 7).
        When: from_environment is called.
        Then: valid_cell has shape (11, 7).

        Test type: unit
        """
        assert updater.valid_cell.shape == env.floor_shape

    def test_valid_cell_walls_marked(self, updater, env):
        """Test that walls are marked False in valid_cell.

        Purpose: Validates that wall positions are correctly marked.

        Given: A LaserTagPOMDP with known wall positions.
        When: from_environment is called.
        Then: valid_cell is False at each wall position.

        Test type: unit
        """
        for wr, wc in env.walls:
            assert not updater.valid_cell[wr, wc]

    def test_wall_dist_table_shape(self, updater, env):
        """Test that wall_dist_table has correct shape.

        Purpose: Validates wall distance lookup table shape.

        Given: A LaserTagPOMDP with floor_shape (11, 7).
        When: from_environment is called.
        Then: wall_dist_table has shape (11, 7, 8).

        Test type: unit
        """
        r, c = env.floor_shape
        assert updater.wall_dist_table.shape == (r, c, 8)

    def test_wall_dist_spot_check(self, updater_no_walls):
        """Test wall distance values for a simple grid with no walls.

        Purpose: Validates that wall distances are correctly computed.

        Given: A 5x5 grid with no walls.
        When: Wall distances are checked for cell (2, 2).
        Then: Each direction has the expected distance to the grid boundary.

        Test type: unit
        """
        # Cell (2, 2) in 5x5 grid: centre cell
        # N (-1,0): 2 clear cells before boundary
        # NE (-1,1): 2 clear cells (hits boundary at both axes)
        # E (0,1): 2 clear cells
        # SE (1,1): 2 clear cells
        # S (1,0): 2 clear cells
        # SW (1,-1): 2 clear cells
        # W (0,-1): 2 clear cells
        # NW (-1,-1): 2 clear cells
        expected = [2, 2, 2, 2, 2, 2, 2, 2]
        np.testing.assert_array_equal(updater_no_walls.wall_dist_table[2, 2, :], expected)


# ---------------------------------------------------------------------------
# batch_transition tests
# ---------------------------------------------------------------------------


class TestBatchTransition:
    def test_output_shape(self, updater_no_walls):
        """Test that batch_transition returns correct shape.

        Purpose: Validates output shape of batch_transition.

        Given: 30 particles of dimension 5.
        When: batch_transition is called with action 0 (North).
        Then: Result has shape (30, 5).

        Test type: unit
        """
        particles = np.tile(np.array([2.0, 2.0, 3.0, 3.0, 0.0]), (30, 1))
        result = updater_no_walls.batch_transition(particles, action=0)
        assert result.shape == (30, 5)

    def test_robot_moves_north(self, updater_no_walls):
        """Test that action 0 (North) moves robot row by -1.

        Purpose: Validates robot movement direction for North action.

        Given: A particle with robot at (2, 2).
        When: batch_transition is called with action 0 (North).
        Then: Robot row decreases by 1.

        Test type: unit
        """
        np.random.seed(0)
        particle = np.array([[2.0, 2.0, 4.0, 4.0, 0.0]])
        result = updater_no_walls.batch_transition(particle, action=0)
        assert result[0, 0] == 1.0  # row decreased
        assert result[0, 1] == 2.0  # col unchanged

    def test_wall_blocks_robot(self, updater):
        """Test that walls block robot movement.

        Purpose: Validates collision detection for robot movement.

        Given: Robot at (2, 2) with wall at (1, 2), moving North.
        When: batch_transition is called.
        Then: Robot stays at (2, 2).

        Test type: unit
        """
        np.random.seed(0)
        # Wall at (1, 2): robot at (2, 2) moving North → blocked
        particle = np.array([[2.0, 2.0, 8.0, 5.0, 0.0]])
        result = updater.batch_transition(particle, action=0)
        assert result[0, 0] == 2.0  # robot stays
        assert result[0, 1] == 2.0

    def test_tag_terminal(self, updater_no_walls):
        """Test that tag action at opponent position creates terminal state.

        Purpose: Validates successful tagging behavior.

        Given: Robot and opponent at the same position.
        When: Tag action (4) is executed.
        Then: Terminal flag is set to 1.0.

        Test type: unit
        """
        particle = np.array([[3.0, 3.0, 3.0, 3.0, 0.0]])
        result = updater_no_walls.batch_transition(particle, action=4)
        assert result[0, 4] == 1.0

    def test_tag_no_terminal(self, updater_no_walls):
        """Test that tag action at different position does not create terminal state.

        Purpose: Validates failed tagging behavior.

        Given: Robot and opponent at different positions.
        When: Tag action (4) is executed.
        Then: Terminal flag remains 0.0.

        Test type: unit
        """
        np.random.seed(0)
        particle = np.array([[3.0, 3.0, 4.0, 4.0, 0.0]])
        result = updater_no_walls.batch_transition(particle, action=4)
        assert result[0, 4] == 0.0

    def test_terminal_unchanged(self, updater_no_walls):
        """Test that terminal particles are not modified.

        Purpose: Validates that terminal state is preserved.

        Given: A terminal particle.
        When: batch_transition is called.
        Then: The particle is unchanged.

        Test type: unit
        """
        particle = np.array([[3.0, 3.0, 4.0, 4.0, 1.0]])
        result = updater_no_walls.batch_transition(particle, action=0)
        np.testing.assert_array_equal(result[0], particle[0])

    def test_opponent_stochastic(self, updater_no_walls):
        """Test that opponent movement is stochastic.

        Purpose: Validates that opponent position varies across samples.

        Given: Multiple independent samples with same starting state.
        When: batch_transition is called many times.
        Then: Opponent positions are not always the same.

        Test type: unit
        """
        np.random.seed(42)
        particle = np.array([[1.0, 1.0, 3.0, 3.0, 0.0]])
        seen_positions = set()
        for _ in range(50):
            result = updater_no_walls.batch_transition(particle, action=0)
            opp = (result[0, 2], result[0, 3])
            seen_positions.add(opp)
        # Should see at least 2 different opponent positions
        assert len(seen_positions) >= 2

    @pytest.mark.parametrize(
        "opponent_policy",
        [OpponentPolicy.EVADE, OpponentPolicy.PURSUE, OpponentPolicy.EVADE_WHEN_SPOTTED],
    )
    @pytest.mark.parametrize(
        "walls",
        [set(), {(5, 6), (5, 4)}, {(3, 5)}],
        ids=["open", "wall_adjacent_slack", "occluded_ray"],
    )
    def test_batch_transition_opponent_distribution_matches_env(self, opponent_policy, walls):
        """Native belief batch transition matches the env opponent distribution per policy.

        Purpose: Validates that the C++ belief kernel (belief_sample_opponent_move via
            belief_batch_transition_discrete) reproduces the env's Python opponent-move
            distribution under ALL policies, including the wall-blocked stay-slack path and
            the occluded-ray case (which makes EVADE_WHEN_SPOTTED fall back to random) —
            guarding against a missed direction flip, pre-move reference, or spotted-predicate
            divergence in the belief path.

        Given: A LaserTagPOMDP (open grid, walls blocking both opponent x-moves, or a wall on
            the robot-opponent ray), robot above the opponent, action South, evaluated for
            EVADE, PURSUE, and EVADE_WHEN_SPOTTED.
        When: 4000 particles are pushed through updater.batch_transition and 4000 samples
            through the env's Python sample_next_state.
        Then: The per-cell opponent next-position probabilities agree within 0.05.

        Test type: integration
        """
        env = LaserTagPOMDP(
            discount_factor=0.95,
            **_lt_pinned_kwargs(
                floor_shape=(7, 11),
                walls=walls,
                dangerous_areas=set(),
                opponent_policy=opponent_policy,
            ),
        )
        updater = LaserTagVectorizedUpdater.from_environment(env)
        state = np.array([2.0, 5.0, 5.0, 5.0, 0.0])
        action = 1  # robot moves South -> robot_next (3, 5), still above the opponent
        n = 4000

        np.random.seed(0)
        out = updater.batch_transition(np.tile(state, (n, 1)), action=np.array(action))
        belief_counts = Counter((int(r[2]), int(r[3])) for r in out)

        np.random.seed(0)
        env_samples = env.sample_next_state(state, action, n_samples=n)
        env_counts = Counter((int(s[2]), int(s[3])) for s in env_samples)

        for cell in set(belief_counts) | set(env_counts):
            p_belief = belief_counts.get(cell, 0) / n
            p_env = env_counts.get(cell, 0) / n
            assert abs(p_belief - p_env) < 0.05, (
                f"policy={opponent_policy} cell={cell}: belief {p_belief:.3f} vs "
                f"env {p_env:.3f} — C++ belief and Python env kernels diverge"
            )


# ---------------------------------------------------------------------------
# batch_observation_log_likelihood tests
# ---------------------------------------------------------------------------


class TestBatchObservationLogLikelihood:
    def test_output_shape(self, updater_no_walls):
        """Test that batch_observation_log_likelihood returns correct shape.

        Purpose: Validates output shape.

        Given: 20 particles.
        When: batch_observation_log_likelihood is called.
        Then: Result has shape (20,).

        Test type: unit
        """
        particles = np.tile(np.array([2.0, 2.0, 3.0, 3.0, 0.0]), (20, 1))
        obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = updater_no_walls.batch_observation_log_likelihood(
            particles, action=0, observation=obs
        )
        assert result.shape == (20,)

    def test_finite_non_terminal(self, updater_no_walls):
        """Test that log-likelihoods are finite for non-terminal particles.

        Purpose: Validates that non-terminal particles produce finite values.

        Given: Non-terminal particles and non-terminal observation.
        When: batch_observation_log_likelihood is called.
        Then: All values are finite.

        Test type: unit
        """
        particles = np.tile(np.array([2.0, 2.0, 3.0, 3.0, 0.0]), (10, 1))
        obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = updater_no_walls.batch_observation_log_likelihood(
            particles, action=0, observation=obs
        )
        assert np.all(np.isfinite(result))

    def test_terminal_obs_terminal_particle_zero(self, updater_no_walls):
        """Test that terminal obs + terminal particle gives log-likelihood 0.0.

        Purpose: Validates terminal observation handling.

        Given: Terminal particles and terminal observation (all -1).
        When: batch_observation_log_likelihood is called.
        Then: Log-likelihood is 0.0 (probability 1.0).

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 3.0, 3.0, 1.0]])
        terminal_obs = np.array([-1.0] * 8)
        result = updater_no_walls.batch_observation_log_likelihood(
            particles, action=0, observation=terminal_obs
        )
        np.testing.assert_allclose(result[0], 0.0, atol=1e-12)

    def test_terminal_obs_non_terminal_particle_neginf(self, updater_no_walls):
        """Test that terminal obs + non-terminal particle gives -inf.

        Purpose: Validates cross-terminal/non-terminal handling.

        Given: Non-terminal particle and terminal observation.
        When: batch_observation_log_likelihood is called.
        Then: Log-likelihood is -inf.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 3.0, 3.0, 0.0]])
        terminal_obs = np.array([-1.0] * 8)
        result = updater_no_walls.batch_observation_log_likelihood(
            particles, action=0, observation=terminal_obs
        )
        assert result[0] == -np.inf

    def test_non_terminal_obs_terminal_particle_neginf(self, updater_no_walls):
        """Test that non-terminal obs + terminal particle gives -inf.

        Purpose: Validates cross-terminal/non-terminal handling.

        Given: Terminal particle and non-terminal observation.
        When: batch_observation_log_likelihood is called.
        Then: Log-likelihood is -inf.

        Test type: unit
        """
        particles = np.array([[2.0, 2.0, 3.0, 3.0, 1.0]])
        obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result = updater_no_walls.batch_observation_log_likelihood(
            particles, action=0, observation=obs
        )
        assert result[0] == -np.inf

    def test_closer_measurement_higher_likelihood(self, updater_no_walls):
        """Test that particles matching observation better have higher likelihood.

        Purpose: Validates that the Gaussian observation model assigns higher
                 likelihood to particles whose laser measurements are closer
                 to the observation.

        Given: Two particles producing different laser measurements, one
               closer to the observation than the other.
        When: batch_observation_log_likelihood is called.
        Then: The particle with measurements closer to the observation has
              higher log-likelihood.

        Test type: unit
        """
        # Robot at (2, 2) in 5x5 grid: all wall distances = 2
        # Observation matches wall-distance-only measurements
        obs = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        # Particle A: opponent at (4, 4), far from robot → doesn't block any rays
        # True measurements = all 2 → matches obs well
        particle_a = np.array([[2.0, 2.0, 4.0, 4.0, 0.0]])

        # Particle B: opponent at (2, 3) → blocks East ray at distance 0
        # True measurements differ from obs significantly in East direction
        particle_b = np.array([[2.0, 2.0, 2.0, 3.0, 0.0]])

        ll_a = updater_no_walls.batch_observation_log_likelihood(
            particle_a, action=0, observation=obs
        )
        ll_b = updater_no_walls.batch_observation_log_likelihood(
            particle_b, action=0, observation=obs
        )
        assert ll_a[0] > ll_b[0]


# ---------------------------------------------------------------------------
# config_id tests
# ---------------------------------------------------------------------------


class TestConfigId:
    def test_config_id_deterministic(self, updater):
        """Test that config_id is deterministic.

        Purpose: Validates reproducibility of config_id.

        Given: An updater.
        When: config_id is called twice.
        Then: The same ID is returned.

        Test type: unit
        """
        assert updater.config_id == updater.config_id

    def test_config_id_differs_for_different_params(self, env):
        """Test that config_id changes when parameters differ.

        Purpose: Validates that different configurations produce different IDs.

        Given: Two updaters with different measurement_noise.
        When: config_id is computed for both.
        Then: The IDs differ.

        Test type: unit
        """
        u1 = LaserTagVectorizedUpdater.from_environment(env)

        env2 = LaserTagPOMDP(
            discount_factor=0.95,
            **_lt_pinned_kwargs(
                floor_shape=env.floor_shape,
                walls=env.walls,
                measurement_noise=env.measurement_noise * 2,
            ),
        )
        u2 = LaserTagVectorizedUpdater.from_environment(env2)
        assert u1.config_id != u2.config_id


# ---------------------------------------------------------------------------
# Equivalence test: vectorized vs per-particle loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithPerParticleLoop:
    def test_robot_position_matches_per_particle_loop(self, env_no_walls, updater_no_walls):
        """Test that vectorized robot positions match per-particle env.sample_next_state.

        Purpose: Verifies that robot movement (deterministic part) in
                 batch_transition matches the per-particle loop exactly.
                 Only robot positions (columns 0:2) are compared because
                 opponent movement uses different RNG primitives between
                 the vectorized and per-particle paths.

        Given: A set of non-terminal particles with transition_error_prob=0.
        When: batch_transition is called and the per-particle loop is run via
            env.sample_next_state.
        Then: Robot positions match exactly for all 5 actions.

        Test type: integration
        """
        np.random.seed(123)
        n = 50
        particles = np.zeros((n, 5))
        valid_positions = [(r, c) for r in range(5) for c in range(5)]
        for i in range(n):
            rp = valid_positions[np.random.randint(len(valid_positions))]
            op = valid_positions[np.random.randint(len(valid_positions))]
            particles[i] = [rp[0], rp[1], op[0], op[1], 0.0]

        for action_idx in range(5):
            np.random.seed(999)
            vectorized = updater_no_walls.batch_transition(particles, action=action_idx)

            np.random.seed(999)
            per_particle = np.empty_like(particles)
            for i in range(n):
                next_state = env_no_walls.sample_next_state(state=particles[i], action=action_idx)
                per_particle[i] = next_state

            np.testing.assert_array_equal(
                vectorized[:, :2],
                per_particle[:, :2],
                err_msg=f"Robot position mismatch for action {action_idx}",
            )

    def test_opponent_movement_distribution(self, env_no_walls, updater_no_walls):
        """Test that vectorized opponent movement has correct distribution.

        Purpose: Verifies that the opponent movement probabilities in the
                 vectorized path match the environment's model statistically.

        Given: A fixed starting state with robot and opponent at known positions.
        When: Many samples are collected from both the vectorized and
              per-particle paths.
        Then: The distribution of opponent positions is consistent across
              both paths (chi-squared or frequency comparison).

        Test type: integration
        """
        np.random.seed(42)
        n_samples = 2000
        # Robot at (1, 1), opponent at (3, 3) → opponent evades away from robot
        particle = np.array([[1.0, 1.0, 3.0, 3.0, 0.0]])
        particles = np.tile(particle, (n_samples, 1))

        result = updater_no_walls.batch_transition(particles, action=0)
        opp_positions = result[:, 2:4]

        # Count unique positions
        unique, counts = np.unique(opp_positions, axis=0, return_counts=True)
        freq = dict(zip([tuple(u) for u in unique], counts / n_samples))

        # Expected: h_target=(3,4) prob=0.4, v_target=(4,3) prob=0.4, stay=(3,3) prob=0.2
        assert abs(freq.get((3.0, 4.0), 0) - 0.4) < 0.05
        assert abs(freq.get((4.0, 3.0), 0) - 0.4) < 0.05
        assert abs(freq.get((3.0, 3.0), 0) - 0.2) < 0.05

    def test_observation_log_likelihood_matches_per_particle_loop(
        self, env_no_walls, updater_no_walls
    ):
        """Test vectorized log-likelihood matches per-particle env.observation_log_probability.

        Purpose: Verifies that batch_observation_log_likelihood matches the
                 per-particle observation log-probability from the environment.

        Given: A set of non-terminal particles and a non-terminal observation.
        When: batch_observation_log_likelihood is called, and per-particle
              env.observation_log_probability is computed.
        Then: Results match within floating-point tolerance.

        Test type: integration
        """
        np.random.seed(42)
        n = 30
        particles = np.zeros((n, 5))
        valid_positions = [(r, c) for r in range(5) for c in range(5)]
        for i in range(n):
            rp = valid_positions[np.random.randint(len(valid_positions))]
            op = valid_positions[np.random.randint(len(valid_positions))]
            while op == rp:
                op = valid_positions[np.random.randint(len(valid_positions))]
            particles[i] = [rp[0], rp[1], op[0], op[1], 0.0]

        obs_tuple = (1.5, 2.0, 1.0, 2.5, 1.5, 1.0, 2.0, 1.5)
        obs_array = np.array(obs_tuple, dtype=float)

        def per_particle_ll_fn(particle, action, _observation):
            log_prob = env_no_walls.observation_log_probability(
                next_state=particle, action=action, observations=[obs_tuple]
            )[0]
            return float(log_prob)

        assert_batch_obs_log_likelihood_matches_loop(
            updater=updater_no_walls,
            particles=particles,
            action=0,
            observation=obs_array,
            per_particle_ll_fn=per_particle_ll_fn,
        )

    def test_terminal_observation_equivalence(self, env_no_walls, updater_no_walls):
        """Test terminal observation handling matches per-particle loop.

        Purpose: Verifies that terminal observation probabilities are
                 correctly handled in both paths.

        Given: A mix of terminal and non-terminal particles.
        When: Terminal observation is evaluated.
        Then: Terminal particles get log-likelihood 0, non-terminal get -inf.

        Test type: integration
        """
        particles = np.array(
            [
                [2.0, 2.0, 3.0, 3.0, 0.0],  # non-terminal
                [1.0, 1.0, 4.0, 4.0, 1.0],  # terminal
                [3.0, 3.0, 0.0, 0.0, 0.0],  # non-terminal
                [0.0, 0.0, 2.0, 2.0, 1.0],  # terminal
            ]
        )
        terminal_obs = np.array([-1.0] * 8)

        result = updater_no_walls.batch_observation_log_likelihood(
            particles, action=0, observation=terminal_obs
        )

        assert result[0] == -np.inf  # non-terminal
        assert result[1] == 0.0  # terminal
        assert result[2] == -np.inf  # non-terminal
        assert result[3] == 0.0  # terminal


# ---------------------------------------------------------------------------
# Belief filtering: ESS-gated resampling drops off-sight particles
# ---------------------------------------------------------------------------

# Open 7x15 grid scenario shared by the resampling tests.
_ROBOT = (3, 5)
_TRUE_OPP = (3, 8)  # 3 cells East of the robot, on the East laser ray (in sight)
_FAR_OFF_RAY = (0, 12)  # far from the robot and off all 8 laser rays (never spotted)
_NEAR_THRESHOLD = 2.0  # opponent within this many cells of _TRUE_OPP counts as "relevant"

# Noiseless laser reading for robot (3, 5), opponent (3, 8) on an open 7x15 grid:
# the opponent blocks the East beam (reading distance-1 = 2); West runs 5 cells to
# the edge; every other beam runs 3 cells to the edge. The no-opponent reading would
# show East = 9, so this observation is informative about the opponent's location.
_IN_SIGHT_OBS = np.array([3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 5.0, 3.0])


def _make_open_env() -> LaserTagPOMDP:
    """Build a wall-free 7x15 LaserTag env with unit measurement noise."""
    return LaserTagPOMDP(
        discount_factor=0.95,
        **_lt_pinned_kwargs(
            floor_shape=(7, 15),
            walls=set(),
            dangerous_areas=set(),
            measurement_noise=1.0,
        ),
    )


def _mixed_belief(
    n_relevant: int, n_irrelevant: int, ess_factor: float
) -> VectorizedWeightedParticleBelief:
    """Build a uniform-weight belief mixing near-true and far off-ray particles."""
    relevant = np.tile(np.array([*_ROBOT, *_TRUE_OPP, 0.0], dtype=float), (n_relevant, 1))
    irrelevant = np.tile(np.array([*_ROBOT, *_FAR_OFF_RAY, 0.0], dtype=float), (n_irrelevant, 1))
    particles = np.vstack([relevant, irrelevant])
    log_weights = np.full(len(particles), -np.log(len(particles)))
    updater = LaserTagVectorizedUpdater.from_environment(_make_open_env())
    return VectorizedWeightedParticleBelief(
        particles, log_weights, updater, resampling=True, ess_factor=ess_factor
    )


def _opp_distance_from_true(belief: VectorizedWeightedParticleBelief) -> np.ndarray:
    """Euclidean distance of each particle's opponent cell from _TRUE_OPP."""
    opp = belief.particles[:, 2:4]
    return np.sqrt(((opp - np.array(_TRUE_OPP, dtype=float)) ** 2).sum(axis=1))


class TestLaserTagBeliefResamplingFiltering:
    """Test ESS-gated resampling filters off-sight particles in the LaserTag belief.

    Purpose: Validates that the dedicated LaserTag belief, driven by
        LaserTagVectorizedUpdater, removes particles whose opponent is far and out of
        the laser's line of sight once an in-sight observation collapses their weights

    Given: A VectorizedWeightedParticleBelief mixing a minority of near-true particles
        with a majority of far off-ray particles
    When: The belief is updated with an in-sight laser observation
    Then: Resampling fires (or is gated off) according to the ESS threshold

    Test type: integration
    """

    def test_resampling_filters_off_sight_particles(self):
        """Test an in-sight observation filters out far off-ray particles via resampling.

        Purpose: Validates that the ESS-gated resample step removes irrelevant particles
            (opponent far and off the laser line of sight) when the observation strongly
            favours the minority near the true opponent location

        Given: A 200-particle belief (40 near the true opponent at (3, 8), 160 far
            off-ray at (0, 12)) with resampling enabled and ess_factor=0.5, so the
            threshold is 100 effective particles
        When: The belief is updated with the in-sight observation (East beam blocked at
            reading 2); the post-reweight ESS (~20) falls below the threshold
        Then: Resampling fires (weights reset to uniform) and every surviving particle
            places the opponent within 2 cells of the true location — all 160 far
            off-ray particles are filtered out

        Test type: integration
        """
        np.random.seed(0)
        belief = _mixed_belief(n_relevant=40, n_irrelevant=160, ess_factor=0.5)

        assert belief.n_particles == 200
        assert int((_opp_distance_from_true(belief) > _NEAR_THRESHOLD).sum()) == 160

        updated = belief.update(action=4, observation=_IN_SIGHT_OBS)

        assert updated.n_particles == 200
        assert np.allclose(
            updated.log_weights, updated.log_weights[0]
        ), "Resampling should have fired and reset weights to uniform"
        assert (
            int((_opp_distance_from_true(updated) > _NEAR_THRESHOLD).sum()) == 0
        ), "All far off-sight particles should have been filtered out"

    def test_resampling_gated_off_keeps_irrelevant_particles(self):
        """Test irrelevant particles survive when ESS stays above the resample threshold.

        Purpose: Locks in that the filtering is ESS-gated, not automatic: with the same
            degenerate weights but a threshold below the ESS, the resample step does not
            fire and the far off-sight particles remain in the belief

        Given: The identical 200-particle mix (40 near-true, 160 far off-ray) and the
            same in-sight observation, but ess_factor=0.05 so the threshold is only 10
            effective particles — below the post-reweight ESS (~20)
        When: The belief is updated
        Then: Resampling does not fire (weights stay non-uniform), all 160 far off-ray
            particles survive, yet they carry essentially zero normalized weight — the
            reweight captured the information, only the gated resample step removes them

        Test type: integration
        """
        np.random.seed(0)
        belief = _mixed_belief(n_relevant=40, n_irrelevant=160, ess_factor=0.05)

        updated = belief.update(action=4, observation=_IN_SIGHT_OBS)

        assert not np.allclose(
            updated.log_weights, updated.log_weights[0]
        ), "Resampling should not have fired, so weights should remain non-uniform"

        far_mask = _opp_distance_from_true(updated) > _NEAR_THRESHOLD
        assert int(far_mask.sum()) == 160, "Far off-sight particles should not be filtered out"
        assert (
            updated.normalized_weights[far_mask].sum() < 1e-6
        ), "Surviving far particles should carry essentially zero weight"
