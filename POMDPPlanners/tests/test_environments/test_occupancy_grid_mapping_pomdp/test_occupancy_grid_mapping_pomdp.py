# SPDX-License-Identifier: MIT

"""Tests for the occupancy-grid mapping POMDP's own dynamics, reward and metrics.

The cross-environment conformance suite already covers the shared
``Environment`` contracts for this environment. What is specific to it, and
therefore tested here, is the part that decides what the environment *means*:
that the reward really is entropy reduction and not a proxy for it, that the
robot cannot walk through walls, that occluded cells stay unknown, that the two
ways an episode can end are told apart, and that the hidden true map is in fact
hidden.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridAction,
    OccupancyGridMappingMetrics,
    OccupancyGridMappingPOMDP,
    OccupancyGridStepChannel,
    RangeNoiseModel,
    create_occupancy_grid_state,
    grid_entropy_bits,
)


@pytest.fixture(name="env")
def _env() -> OccupancyGridMappingPOMDP:
    """The environment at its defaults."""
    return OccupancyGridMappingPOMDP()


@pytest.fixture(name="open_room_env")
def _open_room_env() -> OccupancyGridMappingPOMDP:
    """A small walled room with no interior obstacles and a short-range sensor.

    Every test that needs a *known* world uses this, because the default prior
    draws obstacles at random and a test written against a random map is a test
    written against whatever seed happened to be set.
    """
    return OccupancyGridMappingPOMDP(
        num_rows=7,
        num_cols=7,
        max_range_cells=2.5,
        num_obstacles=0,
        max_steps=12,
    )


def _empty_room(env: OccupancyGridMappingPOMDP) -> np.ndarray:
    """A walled, otherwise empty map of ``env``'s size."""
    occupancy = np.zeros((env.num_rows, env.num_cols))
    occupancy[0, :] = occupancy[-1, :] = 1.0
    occupancy[:, 0] = occupancy[:, -1] = 1.0
    return occupancy


# -- dynamics ------------------------------------------------------------


def test_forward_moves_one_cell_along_the_heading(open_room_env):
    """A forward move into free space advances exactly one cell.

    Purpose: The action set is the robot's whole control authority, and a move
        of the wrong size or direction would silently change every reachability
        argument made about this environment.

    Given: A robot at the centre of an empty room facing north
    When: It moves forward
    Then: Its row decreases by one and its column and heading are unchanged

    Test type: unit
    """
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    next_state = open_room_env.sample_next_state(state, int(OccupancyGridAction.FORWARD))
    assert open_room_env.pose(next_state) == (2, 3, 0)


@pytest.mark.parametrize(
    ("action", "expected_heading"),
    [(OccupancyGridAction.TURN_LEFT, 3), (OccupancyGridAction.TURN_RIGHT, 1)],
)
def test_turning_rotates_without_moving(open_room_env, action, expected_heading):
    """Turning changes the heading and nothing else about where the robot is.

    Purpose: A turn that also moved would make the collision count and the
        new-cell count meaningless, and both are reported metrics.

    Given: A robot at the centre of an empty room facing north
    When: It turns
    Then: Its cell is unchanged and its heading has rotated one quarter turn

    Test type: unit
    """
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    next_state = open_room_env.sample_next_state(state, int(action))
    assert open_room_env.pose(next_state) == (3, 3, expected_heading)


def test_a_wall_blocks_the_robot_and_is_counted_as_a_collision(open_room_env):
    """Driving into a wall wastes the step; it does not pass through it.

    Purpose: This is the only way the hidden map constrains the robot's motion,
        and it is what makes a wrong map hypothesis observable at all. It is
        also the environment's one danger metric.

    Given: A robot one cell inside the north wall, facing it
    When: It moves forward
    Then: Its pose is unchanged and ``step_info`` reports a collision

    Test type: unit
    """
    state = create_occupancy_grid_state(
        open_room_env, _empty_room(open_room_env), row=1, col=3, heading=0
    )
    next_state = open_room_env.sample_next_state(state, int(OccupancyGridAction.FORWARD))
    assert open_room_env.pose(next_state) == (1, 3, 0)
    info = open_room_env.step_info(state, int(OccupancyGridAction.FORWARD), next_state)
    assert info[OccupancyGridStepChannel.OBSTACLE_COLLISION.value] == 1.0
    assert info[OccupancyGridStepChannel.SUCCESSFUL_TRANSLATION.value] == 0.0


def test_the_hidden_map_is_never_altered_by_the_robot(open_room_env):
    """The world does not change; only what the robot knows about it does.

    Purpose: This is the property that makes the reward purely informational.
        A transition that touched the true map would turn exploration into
        modification and every entropy reduction would be partly self-inflicted.

    Given: A robot in an empty room
    When: It takes each of the three actions in turn
    Then: The true-map block of the state is byte-identical throughout

    Test type: unit
    """
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    original = np.array(open_room_env.true_map(state), copy=True)
    for action in OccupancyGridAction:
        state = open_room_env.sample_next_state(state, int(action))
        np.testing.assert_array_equal(open_room_env.true_map(state), original)




# -- observation model ---------------------------------------------------






def test_a_reading_from_the_wrong_pose_has_zero_likelihood(open_room_env):
    """A candidate whose pose disagrees is impossible, not merely unlikely.

    Purpose: This is the factor that lets a filter reject a map under which the
        robot's move would have been blocked. Softening it would make a
        hypothesis that contradicts an *exactly reported* quantity survive.

    Given: A reading drawn from a state, and the same reading with its row moved
    When: Both are scored
    Then: The first is finite and the second is minus infinity

    Test type: unit
    """
    np.random.seed(0)
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    scanned = open_room_env.sample_next_state(state, int(OccupancyGridAction.FORWARD))
    observation = open_room_env.sample_observation(scanned, 0)
    displaced = observation.copy()
    displaced[0] += 1.0
    scores = open_room_env.observation_log_probability(scanned, 0, [observation, displaced])
    assert np.isfinite(scores[0])
    assert scores[1] == -np.inf




# -- reward --------------------------------------------------------------






def test_the_first_scan_of_an_unknown_room_pays(open_room_env):
    """Information gain is positive where there is information to gain.

    Purpose: Guards the sign convention. A reward defined the other way round
        would still be zero on a turn and would still fall inside the declared
        range, so the turn test alone cannot catch it.

    Given: An all-unknown grid
    When: Any action is taken
    Then: The reward is strictly positive

    Test type: unit
    """
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    assert open_room_env.reward(state, int(OccupancyGridAction.TURN_LEFT)) > 0.0


def test_reward_range_bounds_a_long_random_rollout(env):
    """Nothing a rollout produces falls outside the declared bound.

    Purpose: The declared reward range is a hard bound for downstream CVaR and
        confidence-interval code, and getting one wrong is the single
        most-repeated bug in this repository.

    Given: Twenty seeded random episodes at the defaults
    When: Every reward is collected
    Then: All of them lie inside ``reward_range``

    Test type: integration
    """
    minimum, maximum = env.reward_range
    for seed in range(20):
        np.random.seed(seed)
        state = env.initial_state_dist().sample()[0]
        while not env.is_terminal(state):
            action = int(np.random.randint(3))
            reward = env.reward(state, action)
            assert minimum <= reward <= maximum, f"seed {seed}: reward {reward}"
            state = env.sample_next_state(state, action)


# -- terminal conditions and metrics -------------------------------------


def test_the_step_budget_ends_the_episode(open_room_env):
    """An episode that never resolves the map still stops.

    Purpose: Without this the runner's own budget would be the only limit, and
        a planner's rollouts would search past a horizon the environment says
        does not exist.

    Given: A state whose step counter has reached the budget
    When: Terminality is checked
    Then: It is terminal, while one step short of the budget it is not

    Test type: unit
    """
    room = _empty_room(open_room_env)
    assert not open_room_env.is_terminal(
        create_occupancy_grid_state(open_room_env, room, step=open_room_env.max_steps - 1)
    )
    assert open_room_env.is_terminal(
        create_occupancy_grid_state(open_room_env, room, step=open_room_env.max_steps)
    )


def test_a_resolved_map_ends_the_episode_and_is_reported_as_a_goal(open_room_env):
    """Crossing the entropy threshold is completion, not a timeout.

    Purpose: A completion rate is meaningless if the two ways an episode can
        stop are not told apart -- the metric skill's whole reason for the three
        end-reason channels.

    Given: A state whose grid is clamped to certainty everywhere
    When: Terminality and ``step_info`` are read
    Then: It is terminal, and it is reported as resolved rather than unresolved

    Test type: unit
    """
    resolved = create_occupancy_grid_state(
        open_room_env,
        _empty_room(open_room_env),
        log_odds=np.full(open_room_env.num_cells, open_room_env.log_odds_clamp),
    )
    assert open_room_env.is_terminal(resolved)
    info = open_room_env.step_info(resolved, None, None)
    assert info[OccupancyGridStepChannel.MAP_RESOLVED.value] == 1.0
    assert info[OccupancyGridStepChannel.MAP_UNRESOLVED.value] == 0.0
    assert info[OccupancyGridStepChannel.EPISODE_FAILURE.value] == 0.0


def test_an_unknown_grid_is_not_terminal_and_reports_full_entropy(open_room_env):
    """The starting grid is one bit per cell and the episode has not ended.

    Purpose: A terminal condition that fired on the initial state would make
        every episode a zero-step success, which is exactly the degeneracy the
        env-QA gate's random baseline exists to catch.

    Given: A freshly built all-unknown state
    When: Terminality and ``step_info`` are read
    Then: It is not terminal and the residual entropy is the cell count

    Test type: unit
    """
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    assert not open_room_env.is_terminal(state)
    info = open_room_env.step_info(state, None, None)
    assert info[OccupancyGridStepChannel.RESIDUAL_ENTROPY_BITS.value] == pytest.approx(
        float(open_room_env.num_cells)
    )
    assert info[OccupancyGridStepChannel.RESOLVED_CELL_FRACTION.value] == 0.0


def test_step_info_reports_progress_from_the_realised_successor(open_room_env):
    """Progress is scored against the state the step produced.

    Purpose: The episode runner checks its step budget before terminality, so an
        episode whose last allowed step resolves the map records no terminal
        bookkeeping step. Reading progress from ``state`` alone would score that
        episode as an unresolved timeout with its final gain missing.

    Given: A step from an all-unknown grid into a scanned one
    When: ``step_info`` is called with both states
    Then: The residual entropy it reports is the successor's, not the source's

    Test type: unit
    """
    state = create_occupancy_grid_state(open_room_env, _empty_room(open_room_env))
    next_state = open_room_env.sample_next_state(state, int(OccupancyGridAction.FORWARD))
    info = open_room_env.step_info(state, int(OccupancyGridAction.FORWARD), next_state)
    assert info[OccupancyGridStepChannel.RESIDUAL_ENTROPY_BITS.value] == pytest.approx(
        open_room_env.entropy_bits(next_state)
    )
    assert info[
        OccupancyGridStepChannel.RESIDUAL_ENTROPY_BITS.value
    ] < open_room_env.entropy_bits(state)


def test_declared_metric_names_match_the_metric_enum(env):
    """The enum a reader looks things up in is the list the env produces.

    Purpose: A metric named in one place and produced under another name is
        silently dropped by every consumer that looks it up by name.

    Given: The environment's declared metric specs
    When: Their names are compared with the metric enum
    Then: The two sets are equal

    Test type: unit
    """
    assert set(env.get_metric_names()) == {
        metric.value for metric in OccupancyGridMappingMetrics
    }


# -- configuration -------------------------------------------------------


def test_the_start_cell_and_its_neighbours_are_always_free(env):
    """An episode never begins inside a wall or boxed in.

    Purpose: The map prior places obstacles at random, so without this a run
        could open on a state where every action is a collision, and the
        resulting zero completion rate would look like a planner failure.

    Given: Fifty draws from the map prior
    When: The start cell and its four neighbours are read
    Then: All of them are free in every draw

    Test type: integration
    """
    np.random.seed(0)
    distribution = env.initial_state_dist()
    for state in distribution.sample(50):
        occupancy = env.true_map(state)
        assert occupancy[env.start_row, env.start_col] == 0.0
        assert occupancy[env.start_row - 1, env.start_col] == 0.0
        assert occupancy[env.start_row + 1, env.start_col] == 0.0
        assert occupancy[env.start_row, env.start_col - 1] == 0.0
        assert occupancy[env.start_row, env.start_col + 1] == 0.0


def test_the_map_prior_actually_varies(env):
    """Different episodes get different worlds.

    Purpose: If the prior collapsed to one map the true state would be known,
        the belief would be a point mass, and this would be an MDP wearing a
        POMDP's interface -- with every contract test still passing.

    Given: Twenty draws from the map prior
    When: Their true maps are compared
    Then: More than one distinct map appears

    Test type: integration
    """
    np.random.seed(0)
    maps = {
        np.asarray(env.true_map(state)).tobytes()
        for state in env.initial_state_dist().sample(20)
    }
    assert len(maps) > 1


def test_a_start_cell_on_the_wall_is_rejected():
    """A start pose inside the boundary wall fails at construction.

    Purpose: Failing here rather than at the first episode is the difference
        between a clear error and a run whose every episode is a collision.

    Given: A walled grid and a start cell on its edge
    When: The environment is constructed
    Then: ``ValueError``

    Test type: unit
    """
    with pytest.raises(ValueError, match="boundary wall"):
        OccupancyGridMappingPOMDP(num_rows=7, num_cols=7, start_row=0, start_col=3)


def test_the_ray_templates_survive_pickling(env):
    """A worker that unpickles the environment still has a sensor.

    Purpose: The templates are dropped from the pickle on purpose, so the
        rebuild on the other side is the only thing standing between a parallel
        run and an environment with no rays at all.

    Given: The environment pickled and restored
    When: A scan is cast in the restored copy
    Then: It matches the original's, beam for beam

    Test type: unit
    """
    import pickle  # pylint: disable=import-outside-toplevel

    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    restored = pickle.loads(pickle.dumps(env))
    np.testing.assert_allclose(restored.nominal_scan(state), env.nominal_scan(state))


def test_config_id_is_unchanged_by_using_the_environment(env):
    """Using the environment does not move its cache key.

    Purpose: ``config_id`` keys the result cache. Anything memoized onto the
        instance that leaked into it would give a used environment a different
        identity from a fresh one, and every cached result would miss.

    Given: A fresh environment's config id
    When: An episode's worth of steps is taken and the id is read again
    Then: It is unchanged, and equal to a freshly built environment's

    Test type: integration
    """
    before = env.config_id
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    for _ in range(5):
        state, _, _ = env.sample_next_step(state, int(np.random.randint(3)))
    assert env.config_id == before
    assert env.config_id == OccupancyGridMappingPOMDP().config_id


# -- range noise model configuration ---------------------------------------


def test_range_noise_model_accepts_its_string_name_and_rejects_unknown_ones():
    """A YAML config can name the law; a typo fails at construction.

    Test type: unit
    """
    env = OccupancyGridMappingPOMDP(range_noise_model="truncated_normal")
    assert env.range_noise_model is RangeNoiseModel.TRUNCATED_NORMAL
    assert OccupancyGridMappingPOMDP().range_noise_model is RangeNoiseModel.GAUSSIAN
    with pytest.raises(ValueError, match="range_noise_model"):
        OccupancyGridMappingPOMDP(range_noise_model="clipped")
    with pytest.raises(ValueError, match="range_noise_std_cells"):
        OccupancyGridMappingPOMDP(range_noise_model="truncated_normal", range_noise_std_cells=0.0)


def test_the_two_range_laws_have_distinct_identities_that_survive_serialization():
    """A truncated-normal run must never read a Gaussian cache, or vice versa.

    Purpose: ``config_id`` keys the result cache and the belief identities
        hang off the updater's. If the mode were not in them, the two laws
        would silently share cached episodes.

    Given: Two otherwise identical environments, one per law
    When: Their ids are compared, and the truncated one is round-tripped
        through ``to_dict`` / ``from_dict`` and through pickle
    Then: The ids differ, the rebuilt environments keep the enum member and
        the id, and the sensor contract version is 3

    Test type: unit
    """
    import pickle  # pylint: disable=import-outside-toplevel

    gaussian = OccupancyGridMappingPOMDP()
    truncated = OccupancyGridMappingPOMDP(range_noise_model=RangeNoiseModel.TRUNCATED_NORMAL)
    assert gaussian.config_id != truncated.config_id
    assert gaussian != truncated
    assert gaussian.sensor_contract_version == truncated.sensor_contract_version == 3
    assert truncated.to_dict()["params"]["range_noise_model"]["value"] == "truncated_normal"
    for rebuilt in (
        OccupancyGridMappingPOMDP.from_dict(truncated.to_dict()),
        pickle.loads(pickle.dumps(truncated)),
    ):
        assert isinstance(rebuilt, OccupancyGridMappingPOMDP)
        assert rebuilt.range_noise_model is RangeNoiseModel.TRUNCATED_NORMAL
        assert rebuilt.config_id == truncated.config_id
        assert rebuilt == truncated
    updaters = {
        env.range_noise_model: env._batched_kernels  # pylint: disable=protected-access
        for env in (gaussian, truncated)
    }
    assert (
        updaters[RangeNoiseModel.GAUSSIAN].config_id
        != updaters[RangeNoiseModel.TRUNCATED_NORMAL].config_id
    )


def test_gaussian_mode_reproduces_the_original_seeded_scan(open_room_env):
    """The default law draws exactly what the environment drew before the option.

    Purpose: This is the regression guard for "no silent behaviour change":
        the pre-option transition was ``nominal + normal(0, sigma)`` from the
        global stream, and a seeded run must still produce those numbers.

    Test type: unit
    """
    env = open_room_env
    state = create_occupancy_grid_state(env, _empty_room(env))
    moved = env.sample_next_state(state, int(OccupancyGridAction.TURN_LEFT))
    np.random.seed(3)
    successor = env.sample_next_state(state, int(OccupancyGridAction.TURN_LEFT))
    np.random.seed(3)
    expected = env.nominal_scan(moved) + np.random.normal(
        0.0, env.range_noise_std_cells, env.num_beams
    )
    np.testing.assert_array_equal(successor[env.scan_offset :], expected)
    observation = env.sample_observation(successor, int(OccupancyGridAction.TURN_LEFT))
    residual = (observation[3:] - env.nominal_scan(moved)) / env.range_noise_std_cells
    pooled = -0.5 * np.sum(residual**2) - env.num_beams * np.log(
        env.range_noise_std_cells * np.sqrt(2 * np.pi)
    )
    assert env.predictive_observation_log_probability(
        state, int(OccupancyGridAction.TURN_LEFT), observation
    ) == pytest.approx(pooled)


def test_truncated_mode_never_stores_or_reveals_a_negative_range():
    """A long noisy rollout keeps every stored scan and observation non-negative.

    Purpose: The map update and the returned observation read the same stored
        scan, so one check on the state covers both; the noise is set high
        enough that the Gaussian law would go negative many times.

    Test type: integration
    """
    env = OccupancyGridMappingPOMDP(
        num_rows=7,
        num_cols=7,
        max_range_cells=2.5,
        num_obstacles=2,
        range_noise_std_cells=1.5,
        range_noise_model="truncated_normal",
        max_steps=200,
    )
    np.random.seed(5)
    state = env.initial_state_dist().sample()[0]
    for _ in range(120):
        action = int(np.random.randint(3))
        state, observation, _ = env.sample_next_step(state, action)
        assert observation[3:].min() >= 0.0
        np.testing.assert_array_equal(observation[3:], state[env.scan_offset :])
