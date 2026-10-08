# SPDX-License-Identifier: MIT

"""The Chicheck Invaders observation model: what it draws and what it scores.

The two halves of an observation model can be wrong independently -- a sampler
that draws from one law and a likelihood that scores another look fine in
isolation and corrupt every belief update together. Most of this file therefore
checks the two against each other: the enumerated likelihood has to sum to one
over the readings the sampler can produce, and the sampler's frequencies have to
match the likelihood's masses.
"""

import itertools

import numpy as np
import pytest  # noqa: F401

from POMDPPlanners.environments.chicheck_invaders_pomdp import (
    MODE_DIVE,
    MODE_PATROL,
    OBSERVATION_SHIP_WIDTH,
    ChicheckInvadersAction,
    ChicheckInvadersPOMDP,
    ObservationMode,
    create_chicheck_invaders_state,
    noiseless_preset,
    rounded_normal_pmf,
)

# Readings are not truncated at the grid edge, so an exact sum needs a window
# several standard deviations wider than the grid on either side. Twelve cells
# each way at sigma = 1 leaves under 1e-30 of mass outside.
_READING_WINDOW = range(-12, 13)


def build_env(**overrides):
    """Build a one-chicken world small enough to enumerate observations over."""
    settings = {
        "num_columns": 3,
        "num_rows": 3,
        "num_chickens": 1,
        "dive_probability": 0.0,
        "ship_column_noise_std": 0.0,
        "discount_factor": 0.95,
    }
    settings.update(overrides)
    return ChicheckInvadersPOMDP(**settings)


def enumerate_observations(env):
    """Every reading the sampler can produce for a one-chicken world.

    Yields the silent slot and, for each reported combination, every integer
    reading in ``_READING_WINDOW``. The ship's own reading is pinned by
    ``ship_column_noise_std = 0`` so the enumeration stays small enough to sum
    exactly rather than by Monte Carlo.
    """
    assert env.num_chickens == 1
    assert env.ship_column_noise_std == 0.0
    base = OBSERVATION_SHIP_WIDTH
    camera_options = [(0.0, 0.0)] + [(1.0, float(k)) for k in _READING_WINDOW]
    radar_options = [(0.0, 0.0, 0.0)] + [
        (1.0, float(k), float(drop)) for k in _READING_WINDOW for drop in (0.0, -1.0)
    ]
    for (cam_flag, cam_value), (rad_flag, rad_rows, rad_drop) in itertools.product(
        camera_options, radar_options
    ):
        observation = np.zeros(env.observation_size, dtype=np.float64)
        observation[0] = 0.0  # filled in by the caller
        observation[base + 0] = cam_flag
        observation[base + 1] = cam_value
        observation[base + 2] = rad_flag
        observation[base + 3] = rad_rows
        observation[base + 4] = rad_drop
        yield observation


@pytest.mark.parametrize(
    "chicken",
    [
        pytest.param([1, 2, 1, MODE_PATROL, 1], id="patrolling_in_both_sensors"),
        pytest.param([1, 1, 1, MODE_DIVE, 1], id="diving_in_both_sensors"),
        pytest.param([0, 2, 1, MODE_PATROL, 1], id="offset_from_the_ship"),
        pytest.param([1, 2, 1, MODE_PATROL, 0], id="dead_and_therefore_silent"),
    ],
)
def test_observation_probabilities_sum_to_one(chicken):
    """The likelihood is a probability distribution over the readings.

    Purpose: This is the single check that catches most observation-model bugs
        at once. A missed-detection branch, a mis-signed offset or a noise law
        that is renormalised in one place and not another all show up as a sum
        that is not one, and every importance weight in the belief is a ratio of
        these numbers.

    Given: A one-chicken world with the ship's own reading pinned exact.
    When: Every reading the sampler can produce is enumerated and its
        likelihood exponentiated.
    Then: They sum to one.

    Test type: unit
    """
    env = build_env()
    state = create_chicheck_invaders_state(env, chickens=[chicken])
    total = 0.0
    for observation in enumerate_observations(env):
        observation[0] = float(env.ship_column(state))
        total += float(np.exp(env.observation_log_probability_single(state, None, observation)))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_sampled_observation_frequencies_match_the_likelihood():
    """What the sampler draws is what the likelihood says it draws.

    Purpose: The sampler and the likelihood are written separately, and a
        divergence between them is invisible in either one alone while silently
        biasing every particle weight.

    Given: A one-chicken world and twenty thousand draws from one successor.
    When: The empirical frequency of each distinct reading is compared with its
        likelihood.
    Then: Every reading seen more than fifty times matches its predicted mass to
        within four standard errors.

    Test type: unit
    """
    env = build_env()
    state = create_chicheck_invaders_state(env, chickens=[[1, 2, 1, MODE_PATROL, 1]])
    draws = 20_000
    np.random.seed(17)
    counts: dict = {}
    for _ in range(draws):
        observation = env.sample_observation(state, None)
        key = env.hash_observation(observation)
        counts[key] = counts.get(key, (observation, 0))[0], counts.get(key, (None, 0))[1] + 1

    for observation, count in counts.values():
        if count <= 50:
            continue
        predicted = float(np.exp(env.observation_log_probability_single(state, None, observation)))
        empirical = count / draws
        error = 4.0 * np.sqrt(max(predicted * (1.0 - predicted), 1e-12) / draws)
        assert abs(empirical - predicted) <= error, (
            f"reading {observation.tolist()} drawn {empirical:.4f} of the time but "
            f"scored {predicted:.4f}"
        )


def test_a_reading_from_outside_a_sensors_reach_is_impossible():
    """A chicken the camera cannot see cannot have been reported by it.

    Purpose: This is the branch that lets the belief *delete* hypotheses rather
        than merely shrink them, and the only one that returns zero mass.

    Given: A chicken far outside the camera cone.
    When: A reading that claims the camera reported it is scored.
    Then: The log-likelihood is the impossible floor.

    Test type: unit
    """
    env = build_env(num_columns=9, num_rows=9, camera_slope=0.25, radar_radius=2.0)
    state = create_chicheck_invaders_state(env, chickens=[[8, 1, 1, MODE_PATROL, 1]], ship_column=0)
    camera, radar = env.sensor_reach(state)
    assert not camera[0] and not radar[0]

    observation = np.zeros(env.observation_size, dtype=np.float64)
    observation[0] = 0.0
    observation[OBSERVATION_SHIP_WIDTH + 0] = 1.0
    observation[OBSERVATION_SHIP_WIDTH + 1] = 8.0
    assert env.observation_log_probability_single(state, None, observation) < -1e17


def test_silence_inside_both_sensors_costs_the_two_miss_chances():
    """A chicken that should have been seen and was not is evidence against.

    Purpose: This is what makes the observation informative about a chicken no
        sensor reported, and the reason the likelihood carries a factor for
        every slot rather than only the reported ones.

    Given: One world where the chicken sits inside both sensors and one where it
        is outside both.
    When: The all-silent reading is scored against each.
    Then: The in-reach world scores exactly the product of the two miss chances,
        and the out-of-reach world scores one.

    Test type: unit
    """
    env = build_env(num_columns=9, num_rows=9, camera_slope=0.25, radar_radius=2.0)
    seen = create_chicheck_invaders_state(env, chickens=[[0, 1, 1, MODE_PATROL, 1]], ship_column=0)
    unseen = create_chicheck_invaders_state(
        env, chickens=[[8, 1, 1, MODE_PATROL, 1]], ship_column=0
    )
    assert all(env.sensor_reach(seen))
    assert not any(env.sensor_reach(unseen))

    silent = np.zeros(env.observation_size, dtype=np.float64)
    assert float(np.exp(env.observation_log_probability_single(seen, None, silent))) == (
        pytest.approx(
            (1.0 - env.camera_detection_probability) * (1.0 - env.radar_detection_probability)
        )
    )
    assert float(np.exp(env.observation_log_probability_single(unseen, None, silent))) == (
        pytest.approx(1.0)
    )


def test_the_noiseless_preset_makes_the_reading_a_function_of_the_state():
    """Under the noiseless preset one successor always yields one reading.

    Purpose: The preset is what the deterministic tests and the fully
        observable comparison rest on. If any draw survived in it, those tests
        would be flaky for reasons unrelated to what they check.

    Given: A world built from ``noiseless_preset`` and one successor.
    When: Fifty readings are drawn from it under different seeds.
    Then: They are all the same reading, and it scores a likelihood of one.

    Test type: unit
    """
    env = ChicheckInvadersPOMDP(
        num_columns=5, num_rows=4, num_chickens=2, discount_factor=0.95, **noiseless_preset()
    )
    state = create_chicheck_invaders_state(
        env, chickens=[[1, 2, 1, MODE_PATROL, 1], [3, 1, -1, MODE_DIVE, 1]]
    )
    readings = set()
    for seed in range(50):
        np.random.seed(seed)
        readings.add(env.hash_observation(env.sample_observation(state, None)))
    assert len(readings) == 1
    observation = env.sample_observation(state, None)
    assert env.observation_log_probability_single(state, None, observation) == pytest.approx(0.0)


def test_camera_reports_the_column_offset_and_radar_the_row_and_drop():
    """Each sensor reports its own coordinate and nothing else.

    Purpose: The split is the whole point of the observation model. A camera
        that leaked the row, or a radar that leaked the column, would make one
        sensor sufficient and the belief trivial.

    Given: The noiseless preset and a diving chicken two columns right and two
        rows up.
    When: The reading is drawn.
    Then: The camera slot holds ``+2``, the radar slot holds ``2`` rows with the
        dropping flag set, and flipping the chicken's mode changes only the
        flag.

    Test type: unit
    """
    env = ChicheckInvadersPOMDP(
        num_columns=5, num_rows=5, num_chickens=1, discount_factor=0.95, **noiseless_preset()
    )
    base = OBSERVATION_SHIP_WIDTH
    diving = create_chicheck_invaders_state(env, chickens=[[4, 2, 1, MODE_DIVE, 1]], ship_column=2)
    reading = env.sample_observation(diving, None)
    assert reading[base + 0] == 1.0 and reading[base + 1] == 2.0
    assert reading[base + 2] == 1.0 and reading[base + 3] == 2.0
    assert reading[base + 4] == -1.0

    patrolling = create_chicheck_invaders_state(
        env, chickens=[[4, 2, 1, MODE_PATROL, 1]], ship_column=2
    )
    patrol_reading = env.sample_observation(patrolling, None)
    assert patrol_reading[base + 4] == 0.0
    assert np.array_equal(patrol_reading[: base + 4], reading[: base + 4])


def test_fully_observable_mode_returns_the_state_itself():
    """In ``FULL`` mode the reading is the state, scored as a point mass.

    Test type: unit
    """
    env = build_env(num_chickens=2, observation_mode=ObservationMode.FULL)
    state = create_chicheck_invaders_state(
        env, chickens=[[1, 2, 1, MODE_PATROL, 1], [0, 1, -1, MODE_DIVE, 1]]
    )
    reading = env.sample_observation(state, int(ChicheckInvadersAction.STAY))
    assert np.array_equal(reading, state)
    assert env.observation_log_probability_single(state, None, reading) == pytest.approx(0.0)
    assert env.observation_log_probability_single(state, None, reading + 1.0) < -1e17


def test_rounded_normal_masses_sum_to_one_and_collapse_at_zero_noise():
    """The integer noise law is a distribution, and a point mass when exact.

    Test type: unit
    """
    masses = rounded_normal_pmf(np.arange(-40, 41), 2.5, 1.0)
    assert float(masses.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(rounded_normal_pmf(3.0, 3.0, 0.0)) == 1.0
    assert float(rounded_normal_pmf(2.0, 3.0, 0.0)) == 0.0


def test_masked_fields_are_zero_so_equal_readings_hash_alike():
    """Two readings that report the same things compare and hash equal.

    Purpose: Tree-search planners index belief children by
        ``hash_observation``. A masked field left holding whatever the sampler
        last computed would give two identical reports two different keys, and
        the planner would split one belief node into many.

    Test type: unit
    """
    env = build_env(num_chickens=2)
    state = create_chicheck_invaders_state(
        env, chickens=[[1, 2, 1, MODE_PATROL, 1], [0, 1, -1, MODE_DIVE, 1]]
    )
    np.random.seed(3)
    readings = [env.sample_observation(state, None) for _ in range(200)]
    for first, second in itertools.combinations(readings[:20], 2):
        assert env.is_equal_observation(first, second) == (
            env.hash_observation(first) == env.hash_observation(second)
        )
    silent = np.zeros(env.observation_size, dtype=np.float64)
    silent[0] = float(env.ship_column(state))
    assert env.hash_observation(silent) == env.hash_observation(np.array(silent, copy=True))


def test_batched_and_per_state_likelihood_paths_agree():
    """Every likelihood entry point returns the same number.

    Purpose: Particle filters take the per-state path, tree expansion the
        scalar one and reweighting the batched one. A divergence changes the
        observation model depending on which planner is running.

    Test type: unit
    """
    env = build_env(num_chickens=2)
    states = [
        create_chicheck_invaders_state(
            env, chickens=[[1, 2, 1, MODE_PATROL, 1], [0, 1, -1, MODE_DIVE, 1]]
        ),
        create_chicheck_invaders_state(
            env, chickens=[[0, 2, -1, MODE_DIVE, 1], [2, 1, 1, MODE_PATROL, 1]]
        ),
    ]
    np.random.seed(9)
    observation = env.sample_observation(states[0], None)

    per_state = env.observation_log_probability_per_state(states, None, observation)
    scalar = [env.observation_log_probability_single(s, None, observation) for s in states]
    batched = [env.observation_log_probability(s, None, [observation])[0] for s in states]
    assert np.allclose(per_state, scalar)
    assert np.allclose(per_state, batched)
