# SPDX-License-Identifier: MIT

"""Tests for the pluggable occupancy-grid map update rule.

Two things have to hold at once. The default rule must reproduce the update the
environment performed before the rule became an object -- every cached result
and the pinned golden visualization were produced under it, and a golden hash
separates maps that differ by 1e-13, so "close" is not good enough. And a rule
supplied by a user must reach every path: the scalar transition, the batched
particle kernels and both rewards, with an identity that cannot collide with
the default's in a cache.

The legacy arithmetic is written out here rather than imported, so that these
tests still fail if someone rewrites the shipped rule into something that is
only algebraically the same.
"""

import pickle

import numpy as np
import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    NearestCellLogOddsUpdateRule,
    OccupancyGridMappingPOMDP,
    OccupancyUpdateRule,
    ProbabilityOccupancyUpdateRule,
    create_occupancy_grid_state,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_beliefs import (
    OccupancyGridMappingVectorizedUpdater,
)
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_sensor import (
    observed_scan_evidence_counts,
)
from POMDPPlanners.tests.test_environments.test_occupancy_grid_mapping_pomdp.test_occupancy_grid_mapping_beliefs.test_occupancy_grid_mapping_vectorized_updater import (  # noqa: E501
    diverse_particles,
    make_env,
)

ACTIONS = (0, 1, 2)


# ----------------------------------------------------------------------
# The update as it was written before the rule became an object.
# ----------------------------------------------------------------------


def legacy_scalar_delta(env, row, col, heading, ranges):
    """The pre-refactor scalar increment, transcribed from ``develop`` 6adbd45b."""
    offsets, distances = env._ray_templates[heading]  # pylint: disable=protected-access
    rows = offsets[:, :, 0] + row
    cols = offsets[:, :, 1] + col
    valid = (
        (rows >= 0)
        & (rows < env.num_rows)
        & (cols >= 0)
        & (cols < env.num_cols)
        & np.isfinite(distances)
    )
    valid = np.logical_and.accumulate(valid, axis=1)
    has_cell = valid.any(axis=1)
    nearest = np.argmin(np.where(valid, np.abs(distances - ranges[:, None]), np.inf), axis=1)
    hit = (ranges < env.max_range_cells) & has_cell
    slots = np.arange(distances.shape[1])[None, :]
    free = valid & ((slots < nearest[:, None]) | ~hit[:, None])
    occupied = valid & (slots == nearest[:, None]) & hit[:, None]
    flat = rows * env.num_cols + cols
    delta = (
        np.bincount(flat[free], minlength=env.num_cells) * env.free_log_odds
        + np.bincount(flat[occupied], minlength=env.num_cells) * env.occupied_log_odds
    )
    delta[row * env.num_cols + col] += env.free_log_odds
    return delta


def legacy_state_from_observation(env, state, observation):
    """The pre-refactor ``state_from_observation``."""
    row, col, heading = (int(value) for value in observation[:3])
    next_state = np.array(state, dtype=float, copy=True)
    next_state[0] += 1
    next_state[1:4] = observation[:3]
    next_state[env.scan_offset :] = observation[3:]
    delta = legacy_scalar_delta(env, row, col, heading, observation[3:])
    end = env.log_odds_offset + env.num_cells
    next_state[env.log_odds_offset : end] = np.clip(
        next_state[env.log_odds_offset : end] + delta, -env.log_odds_clamp, env.log_odds_clamp
    )
    return next_state


def legacy_batch_deltas(env, ranges, rows, cols, headings):
    """The pre-refactor ``_scan_deltas``, transcribed from ``develop`` 6adbd45b."""
    count = ranges.shape[0]
    deltas = np.zeros((count, env.num_cells), dtype=np.float64)
    slot_offsets = np.arange(count) * env.num_cells
    for heading in np.unique(headings):
        index = np.flatnonzero(headings == heading)
        offsets, distances = env._ray_templates[int(heading)]  # pylint: disable=protected-access
        length = distances.shape[1]
        ray_rows = offsets[None, :, :, 0] + rows[index, None, None]
        ray_cols = offsets[None, :, :, 1] + cols[index, None, None]
        valid = (
            (ray_rows >= 0)
            & (ray_rows < env.num_rows)
            & (ray_cols >= 0)
            & (ray_cols < env.num_cols)
            & np.isfinite(distances)[None]
        )
        valid = np.logical_and.accumulate(valid, axis=2)
        has_cell = valid.any(axis=2)
        gaps = np.abs(distances[None] - ranges[index][:, :, None])
        nearest = np.argmin(np.where(valid, gaps, np.inf), axis=2)
        hit = (ranges[index] < env.max_range_cells) & has_cell
        slots = np.arange(length)[None, None, :]
        free = valid & ((slots < nearest[:, :, None]) | ~hit[:, :, None])
        occupied = valid & (slots == nearest[:, :, None]) & hit[:, :, None]
        flat = ray_rows * env.num_cols + ray_cols + slot_offsets[index][:, None, None]
        deltas += (
            np.bincount(flat[free], minlength=count * env.num_cells).reshape(count, env.num_cells)
            * env.free_log_odds
        )
        deltas += (
            np.bincount(flat[occupied], minlength=count * env.num_cells).reshape(
                count, env.num_cells
            )
            * env.occupied_log_odds
        )
    deltas[np.arange(count), rows * env.num_cols + cols] += env.free_log_odds
    return deltas


# ----------------------------------------------------------------------
# A test-only rule, written on probabilities.
# ----------------------------------------------------------------------


class OddsProductProbabilityRule(ProbabilityOccupancyUpdateRule):
    """The same inverse sensor model, stated on ``p`` instead of on ``L``.

    Defined at module level rather than inside a test because it has to survive
    pickling to a worker and a rebuild from a serialized config, and neither
    works for a class defined in a function body.

    Attributes:
        hit_probability: Probability a cell is occupied given one occupied sighting.
        miss_probability: The same for one free sighting.
    """

    def __init__(self, hit_probability=0.85, miss_probability=0.15, log_odds_clamp=6.0):
        """Initialize the rule.

        Args:
            hit_probability: Inverse sensor probability for an occupied sighting.
            miss_probability: Inverse sensor probability for a free sighting.
            log_odds_clamp: Symmetric bound on the resulting log-odds.
        """
        super().__init__(log_odds_clamp=log_odds_clamp)
        self.hit_probability = float(hit_probability)
        self.miss_probability = float(miss_probability)

    def update_probabilities(self, probabilities, free_counts, occupied_counts):
        """Multiply the prior odds by one likelihood ratio per sighting."""
        ratio = (self.miss_probability / (1.0 - self.miss_probability)) ** free_counts * (
            self.hit_probability / (1.0 - self.hit_probability)
        ) ** occupied_counts
        scaled = probabilities * ratio
        return scaled / (1.0 - probabilities + scaled)

    def parameters(self):
        """Both probabilities and the clamp."""
        return {
            "hit_probability": self.hit_probability,
            "miss_probability": self.miss_probability,
            **super().parameters(),
        }


class ForgetfulRule(OccupancyUpdateRule):
    """A rule that throws the scan away, so a wired-up path is unmistakable."""

    def update_log_odds(
        self,
        log_odds,
        row,
        col,
        heading,
        observed_ranges,
        template,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """Return the prior map untouched."""
        return np.array(log_odds, dtype=np.float64, copy=True)

    def parameters(self):
        """No parameters."""
        return {}


# ----------------------------------------------------------------------
# The default rule reproduces the previous update exactly.
# ----------------------------------------------------------------------


def random_observations(env, count=8, seed=3):
    """Readings spanning misses, in-range hits and negative draws."""
    rng = np.random.RandomState(seed)
    observations = []
    for _ in range(count):
        ranges = rng.uniform(-0.5, env.max_range_cells + 1.0, env.num_beams)
        observations.append(
            np.r_[
                float(rng.randint(1, env.num_rows - 1)),
                float(rng.randint(1, env.num_cols - 1)),
                float(rng.randint(0, 4)),
                ranges,
            ]
        )
    return observations


def random_log_odds(env, count=8, seed=5):
    """Partly resolved maps, including cells already at the clamp."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0, 3.0, size=(count, env.num_cells))


def test_default_rule_reproduces_the_previous_scalar_update_exactly():
    """Purpose: the shipped default must be the pre-refactor update, bit for bit.

    Given: Partly resolved maps and scans covering misses, hits and negative
        readings.
    When: ``state_from_observation`` and the transcribed pre-refactor update are
        both applied.
    Then: Every successor is equal element by element, not merely close --
        a pinned golden visualization hash separates maps differing by 1e-13.

    Test type: unit
    """
    env = make_env()
    maps = random_log_odds(env)
    for observation in random_observations(env):
        for log_odds in maps:
            state = create_occupancy_grid_state(
                env, np.zeros(env.num_cells), log_odds=log_odds, step=2
            )
            assert np.array_equal(
                env.state_from_observation(state, observation),
                legacy_state_from_observation(env, state, observation),
            )


def test_default_rule_reproduces_the_previous_batched_update_exactly():
    """Purpose: the batched kernel must also be the pre-refactor arithmetic.

    Given: One scan per particle at mixed poses and headings.
    When: The rule's batched form and the transcribed pre-refactor
        ``_scan_deltas`` plus clamp are both applied.
    Then: The resulting maps are equal element by element.

    Test type: unit
    """
    env = make_env()
    rng = np.random.RandomState(17)
    count = 16
    rows = rng.randint(1, env.num_rows - 1, count)
    cols = rng.randint(1, env.num_cols - 1, count)
    headings = rng.randint(0, 4, count)
    ranges = rng.uniform(-0.5, env.max_range_cells + 1.0, (count, env.num_beams))
    log_odds = rng.normal(0.0, 3.0, (count, env.num_cells))

    updated = env.update_rule.batch_update_log_odds(
        log_odds,
        rows,
        cols,
        headings,
        ranges,
        env._ray_templates,  # pylint: disable=protected-access
        env.num_rows,
        env.num_cols,
        env.max_range_cells,
    )
    expected = np.clip(
        log_odds + legacy_batch_deltas(env, ranges, rows, cols, headings),
        -env.log_odds_clamp,
        env.log_odds_clamp,
    )
    assert np.array_equal(updated, expected)


def test_default_rule_batched_transition_matches_the_previous_update_exactly():
    """Purpose: the belief's generative transition must be unchanged too.

    Given: Diverse particles and one observation.
    When: ``batch_state_from_observation`` and the scalar
        ``state_from_observation`` are both applied.
    Then: Every particle's successor is equal element by element.

    Test type: unit
    """
    env = make_env()
    updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
    particles = diverse_particles(env)
    for observation in random_observations(env, count=3):
        batched = updater.batch_state_from_observation(particles, observation)
        scalar = np.asarray(
            [env.state_from_observation(particle, observation) for particle in particles]
        )
        assert np.array_equal(batched, scalar)
        assert np.array_equal(
            batched,
            np.asarray(
                [
                    legacy_state_from_observation(env, particle, observation)
                    for particle in particles
                ]
            ),
        )


# ----------------------------------------------------------------------
# A custom rule reaches every path.
# ----------------------------------------------------------------------


def test_probability_rule_is_accepted_and_reaches_every_path():
    """Purpose: a rule written on ``p`` must drive all four entry points alike.

    Given: An environment built with a probability-space rule.
    When: ``state_from_observation``, ``batch_state_from_observation``,
        ``reward`` and ``reward_batch`` are evaluated on the same particles.
    Then: The two map paths agree exactly and the two reward paths agree to
        floating-point precision, so no path kept the built-in rule.

    Test type: unit
    """
    env = make_env(update_rule=OddsProductProbabilityRule())
    updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
    particles = diverse_particles(env)

    for observation in random_observations(env, count=3):
        scalar = np.asarray(
            [env.state_from_observation(particle, observation) for particle in particles]
        )
        assert np.array_equal(updater.batch_state_from_observation(particles, observation), scalar)

    for action in ACTIONS:
        np.testing.assert_allclose(
            env.reward_batch(particles, action),
            [env.reward(particle, action) for particle in particles],
            rtol=1e-12,
            atol=1e-9,
        )


def test_probability_rule_restating_the_default_agrees_with_it():
    """Purpose: ``L`` and ``p`` must be two ways of writing one update.

    Given: The default rule's constants restated as an odds product on ``p``.
    When: Both rules map the same partly resolved maps and scans.
    Then: The resulting maps agree to floating-point precision, which is what
        makes the probability base class usable for porting a published rule.

    Test type: unit
    """
    default_env = make_env()
    probability_env = make_env(update_rule=OddsProductProbabilityRule())
    maps = random_log_odds(default_env, count=4)
    for observation in random_observations(default_env, count=4):
        for log_odds in maps:
            state = create_occupancy_grid_state(
                default_env, np.zeros(default_env.num_cells), log_odds=log_odds
            )
            np.testing.assert_allclose(
                probability_env.state_from_observation(state, observation),
                default_env.state_from_observation(state, observation),
                rtol=0,
                atol=1e-9,
            )


def test_a_rule_that_ignores_the_scan_leaves_every_path_unchanged():
    """Purpose: no path may quietly fall back to the built-in update.

    Given: A rule that returns the prior map untouched.
    When: The scalar transition and the batched kernel run.
    Then: The maps are unchanged and the expected reward is exactly the step
        cost, because no scan can reduce entropy.

    Test type: unit
    """
    env = make_env(update_rule=ForgetfulRule(), step_cost=0.25)
    updater = OccupancyGridMappingVectorizedUpdater.from_environment(env)
    particles = diverse_particles(env)
    observation = random_observations(env, count=1)[0]
    end = env.log_odds_offset + env.num_cells

    successors = updater.batch_state_from_observation(particles, observation)
    assert np.array_equal(
        successors[:, env.log_odds_offset : end], particles[:, env.log_odds_offset : end]
    )
    scalar = env.state_from_observation(particles[0], observation)
    assert np.array_equal(
        scalar[env.log_odds_offset : end], particles[0][env.log_odds_offset : end]
    )
    np.testing.assert_allclose(env.reward_batch(particles, 0), -0.25, atol=1e-12)


def test_batched_kernels_follow_the_rule_rather_than_a_cached_default():
    """Purpose: the lazily built kernels must be keyed on the rule.

    Given: Two environments identical but for their update rule.
    When: Their batched kernels are built.
    Then: The kernels carry the environment's own rule and different identifiers.

    Test type: unit
    """
    default_env = make_env()
    custom_env = make_env(update_rule=OddsProductProbabilityRule())
    default_kernels = default_env._batched_kernels  # pylint: disable=protected-access
    custom_kernels = custom_env._batched_kernels  # pylint: disable=protected-access
    assert isinstance(default_kernels.update_rule, NearestCellLogOddsUpdateRule)
    assert isinstance(custom_kernels.update_rule, OddsProductProbabilityRule)
    assert default_kernels.config_id != custom_kernels.config_id


# ----------------------------------------------------------------------
# Identity, pickling and the config round trip.
# ----------------------------------------------------------------------


def test_the_default_rule_leaves_the_environment_identity_alone():
    """Purpose: adding this knob must not invalidate results already cached.

    Given: The default environment, and the same environment handed the default
        rule explicitly.
    When: Their ``config_id`` values are compared.
    Then: They are equal, because the default rule is nothing but the
        environment's own hit/miss probabilities and clamp restated.

    Test type: unit
    """
    env = make_env()
    explicit = make_env(
        update_rule=NearestCellLogOddsUpdateRule(
            free_log_odds=env.free_log_odds,
            occupied_log_odds=env.occupied_log_odds,
            log_odds_clamp=env.log_odds_clamp,
        )
    )
    assert explicit.config_id == env.config_id
    assert explicit == env


@pytest.mark.parametrize(
    "rule",
    [
        OddsProductProbabilityRule(),
        OddsProductProbabilityRule(hit_probability=0.9),
        ForgetfulRule(),
        NearestCellLogOddsUpdateRule(free_log_odds=-0.5, occupied_log_odds=0.5, log_odds_clamp=6.0),
    ],
)
def test_a_non_default_rule_changes_the_environment_identity(rule):
    """Purpose: two rules must never share a cache entry.

    Given: A rule that is not the environment's default.
    When: The environment's ``config_id`` is taken.
    Then: It differs from the default environment's, and the two environments
        are not equal.

    Test type: unit
    """
    env = make_env()
    custom = make_env(update_rule=rule)
    assert custom.config_id != env.config_id
    assert custom != env


def test_rules_that_differ_only_in_a_parameter_have_different_identities():
    """Purpose: a parameter left out of ``parameters`` would be a silent collision.

    Test type: unit
    """
    first = OddsProductProbabilityRule(hit_probability=0.85)
    second = OddsProductProbabilityRule(hit_probability=0.80)
    assert first.config_id != second.config_id
    assert first != second
    assert first == OddsProductProbabilityRule(hit_probability=0.85)


def test_a_rule_survives_pickling():
    """Purpose: the environment is pickled to every parallel worker.

    Given: An environment with a custom rule.
    When: It is pickled and restored.
    Then: The rule, the identity and the update all come back unchanged.

    Test type: unit
    """
    env = make_env(update_rule=OddsProductProbabilityRule(hit_probability=0.8))
    restored = pickle.loads(pickle.dumps(env))
    assert restored.config_id == env.config_id
    assert restored.update_rule == env.update_rule
    particle = diverse_particles(env)[0]
    observation = random_observations(env, count=1)[0]
    assert np.array_equal(
        restored.state_from_observation(particle, observation),
        env.state_from_observation(particle, observation),
    )


def test_a_rule_survives_the_config_round_trip():
    """Purpose: a study's saved config must rebuild the environment it ran.

    Given: An environment with a custom rule.
    When: It is serialized with ``to_dict`` and rebuilt with ``from_dict``.
    Then: The rebuilt environment carries the same rule and the same identity.

    Test type: unit
    """
    env = make_env(update_rule=OddsProductProbabilityRule(miss_probability=0.2))
    rebuilt = Environment.from_dict(env.to_dict())
    assert isinstance(rebuilt, OccupancyGridMappingPOMDP)
    assert isinstance(rebuilt.update_rule, OddsProductProbabilityRule)
    assert rebuilt.update_rule == env.update_rule
    assert rebuilt.config_id == env.config_id


def test_the_default_environment_round_trips_to_the_same_identity():
    """Purpose: serializing the default must not promote the rule into the identity.

    Test type: unit
    """
    env = make_env()
    rebuilt = Environment.from_dict(env.to_dict())
    assert rebuilt.config_id == env.config_id


def test_a_non_rule_is_rejected_at_construction():
    """Purpose: a typo in a config should fail loudly, not map with the default.

    Test type: unit
    """
    with pytest.raises(TypeError, match="update_rule"):
        make_env(update_rule="nearest-cell")


def test_a_probability_rule_returning_an_impossible_value_is_rejected():
    """Purpose: a probability outside [0, 1] is an author error the clamp hides.

    Test type: unit
    """

    class OutOfRangeRule(ProbabilityOccupancyUpdateRule):
        """Returns a probability above one."""

        def update_probabilities(self, probabilities, free_counts, occupied_counts):
            """Return an impossible probability."""
            return probabilities + 2.0

    # Driven with the default rule: the broken one raises on the first
    # transition, which would fail the test before it reached its assertion.
    default_env = make_env()
    particle = diverse_particles(default_env)[0]
    observation = random_observations(default_env, count=1)[0]
    env = make_env(update_rule=OutOfRangeRule())
    with pytest.raises(ValueError, match="outside"):
        env.state_from_observation(particle, observation)


# ----------------------------------------------------------------------
# The base class's own batched fallback.
# ----------------------------------------------------------------------


class ScalarOnlyDecayRule(OccupancyUpdateRule):
    """A rule with no batched override, so the base class's loop is exercised.

    It halves the prior before folding the scan in, which makes the result
    depend on both the previous map and the scan -- a fallback that silently
    dropped either would not survive the comparison below.
    """

    def update_log_odds(
        self,
        log_odds,
        row,
        col,
        heading,
        observed_ranges,
        template,
        num_rows,
        num_cols,
        max_range_cells,
    ):
        """Halve the prior, add one unit of evidence per sighting, clamp at 6."""
        free_counts, occupied_counts = observed_scan_evidence_counts(
            observed_ranges, template, row, col, num_rows, num_cols, max_range_cells
        )
        delta = occupied_counts - free_counts
        return np.clip(0.5 * np.asarray(log_odds, dtype=np.float64) + delta, -6.0, 6.0)

    def parameters(self):
        """No parameters."""
        return {}


def test_the_default_batched_form_equals_looping_the_scalar_one():
    """Purpose: a rule that implements only the scalar form must still be usable.

    Given: A rule with no batched override, and one scan per particle.
    When: ``batch_update_log_odds`` and an explicit loop over
        ``update_log_odds`` are both applied.
    Then: The results are equal element by element, so the fallback is not a
        second, subtly different implementation of the rule.

    Test type: unit
    """
    env = make_env(update_rule=ScalarOnlyDecayRule())
    rng = np.random.RandomState(29)
    count = 10
    rows = rng.randint(1, env.num_rows - 1, count)
    cols = rng.randint(1, env.num_cols - 1, count)
    headings = rng.randint(0, 4, count)
    ranges = rng.uniform(-0.5, env.max_range_cells + 1.0, (count, env.num_beams))
    log_odds = rng.normal(0.0, 3.0, (count, env.num_cells))
    templates = env._ray_templates  # pylint: disable=protected-access

    batched = env.update_rule.batch_update_log_odds(
        log_odds,
        rows,
        cols,
        headings,
        ranges,
        templates,
        env.num_rows,
        env.num_cols,
        env.max_range_cells,
    )
    expected = np.stack(
        [
            env.update_rule.update_log_odds(
                log_odds[index],
                int(rows[index]),
                int(cols[index]),
                int(headings[index]),
                ranges[index],
                templates[int(headings[index])],
                env.num_rows,
                env.num_cols,
                env.max_range_cells,
            )
            for index in range(count)
        ]
    )
    assert np.array_equal(batched, expected)


def test_the_default_batched_form_accepts_an_empty_particle_array():
    """Purpose: an empty particle set is a real state, not an error.

    Given: No particles at all, which a filter reaches when every motion
        outcome of a branch was pruned.
    When: The base class's batched form and the default rule's own are called.
    Then: Both return an empty map array of the right shape rather than
        raising, which ``np.stack`` of an empty list would.

    Test type: unit
    """
    env = make_env()
    empty_maps = np.zeros((0, env.num_cells))
    empty_int = np.zeros(0, dtype=np.int64)
    empty_ranges = np.zeros((0, env.num_beams))
    templates = env._ray_templates  # pylint: disable=protected-access

    for rule in (ScalarOnlyDecayRule(), env.update_rule):
        result = rule.batch_update_log_odds(
            empty_maps,
            empty_int,
            empty_int,
            empty_int,
            empty_ranges,
            templates,
            env.num_rows,
            env.num_cols,
            env.max_range_cells,
        )
        assert result.shape == (0, env.num_cells)


def test_an_attached_rule_cannot_be_mutated_behind_the_identity():
    """Purpose: a rule's parameters must not change after they were hashed.

    Given: An environment built with a custom rule, whose ``config_id`` already
        carries that rule's parameters.
    When: One of those parameters is assigned on the rule object afterwards.
    Then: The assignment is refused, because the identifier is a string taken
        once rather than a live view, so the change would map under a different
        law while the cache still answered to the old identity.

    Test type: unit
    """
    rule = OddsProductProbabilityRule()
    env = make_env(update_rule=rule)
    assert env.update_rule is rule
    with pytest.raises(AttributeError, match="read-only"):
        rule.hit_probability = 0.5
    # Through the environment's own handle too: ``setattr`` rather than a dotted
    # assignment because the property is typed as the abstract base, which has
    # no such attribute.
    with pytest.raises(AttributeError, match="read-only"):
        setattr(env.update_rule, "hit_probability", 0.5)
    assert rule.hit_probability == 0.85


def test_a_rule_is_mutable_until_an_environment_adopts_it():
    """Purpose: freezing must not make a rule awkward to build.

    Given: A freshly constructed rule.
    When: Its parameters are set, as a subclass constructor does.
    Then: The assignments succeed, and the environment built from it is
        identified by the values in force at that moment.

    Test type: unit
    """
    rule = OddsProductProbabilityRule()
    rule.hit_probability = 0.9
    assert (
        make_env(update_rule=rule).config_id
        == make_env(update_rule=OddsProductProbabilityRule(hit_probability=0.9)).config_id
    )


def test_a_frozen_rule_stays_frozen_through_pickling():
    """Purpose: the environment reaches a worker by pickle, rules included.

    Test type: unit
    """
    env = pickle.loads(pickle.dumps(make_env(update_rule=OddsProductProbabilityRule())))
    with pytest.raises(AttributeError, match="read-only"):
        setattr(env.update_rule, "hit_probability", 0.5)
