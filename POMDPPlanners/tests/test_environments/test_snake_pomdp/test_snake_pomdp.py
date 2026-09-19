# SPDX-License-Identifier: MIT

"""Dynamics, observation model and reward of the Snake POMDP.

One test per rule the formal model states, because the rules interact: the tail
is released unless the snake eats, and *that* is what decides whether the cell
behind it is safe to enter. A test that only checked "moving forward works"
would pass against several wrong implementations of that pair.
"""

import numpy as np
import pytest

from POMDPPlanners.environments.snake_pomdp.snake_pomdp import (
    DIRECTIONS,
    OBSERVATION_LIVE,
    TERMINAL_OBSERVATION,
    SnakeAction,
    SnakePOMDP,
    SnakeQuadrant,
    SnakeTermination,
    create_snake_state,
    quadrants_for_offset,
)


def build_env(**overrides):
    """Return a small Snake environment, with overrides applied.

    Small on purpose: a 7x7 grid with a target length of 6 makes every rule
    reachable in a handful of hand-written steps, and keeps the starvation limit
    short enough to test directly.
    """
    kwargs = {
        "grid_size": 7,
        "target_length": 6,
        "window_radius": 2,
        "detection_probability": 0.9,
        "scent_accuracy": 0.7,
        "starvation_limit": 8,
        "discount_factor": 0.98,
    }
    kwargs.update(overrides)
    return SnakePOMDP(**kwargs)


def state_for(env, body, food, steps_since_food=0):
    """Build a running state for ``env``."""
    return create_snake_state(
        body=body,
        food=food,
        steps_since_food=steps_since_food,
        target_length=env.target_length,
    )


# ---------------------------------------------------------------------------
# Headings and turning
# ---------------------------------------------------------------------------


def test_heading_is_the_step_from_the_neck_to_the_head():
    """The heading is derived, not stored, so it has to be read off the body.

    Purpose: Every transition starts by turning the heading, so a heading read
        the wrong way round turns the whole action set inside out.
    Given: A snake whose head is one cell east of its second segment.
    When: The heading is read.
    Then: It is east.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (0, 0))
    assert env.heading(state) == (0, 1)


@pytest.mark.parametrize(
    "heading, action, expected",
    [
        ((0, 1), SnakeAction.TURN_LEFT, (-1, 0)),
        ((0, 1), SnakeAction.GO_STRAIGHT, (0, 1)),
        ((0, 1), SnakeAction.TURN_RIGHT, (1, 0)),
        ((-1, 0), SnakeAction.TURN_LEFT, (0, -1)),
        ((1, 0), SnakeAction.TURN_RIGHT, (0, -1)),
        ((0, -1), SnakeAction.TURN_RIGHT, (-1, 0)),
    ],
)
def test_turning_rotates_the_heading_ninety_degrees(heading, action, expected):
    """Left is counter-clockwise, right is clockwise, straight keeps the heading.

    Purpose: Row 0 is the top of the grid, so "counter-clockwise" in grid
        coordinates is the opposite sign to the one a screen-space intuition
        suggests. Pinning all four headings is what catches a flipped rotation.
    Given: Each heading and each action.
    When: The turn is applied.
    Then: The new heading is the expected 90-degree rotation.

    Test type: unit
    """
    env = build_env()
    assert env.turn(heading, int(action)) == expected


def test_there_is_no_reversing_action():
    """No action turns the heading through 180 degrees.

    Purpose: Reversing would walk the head straight into the neck on every
        step, which is why the action set is relative rather than absolute.
    Given: Every heading and every action.
    When: The turns are applied.
    Then: No result is the opposite of the heading it came from.

    Test type: unit
    """
    env = build_env()
    for heading in DIRECTIONS:
        for action in SnakeAction:
            turned = env.turn(heading, int(action))
            assert turned != (-heading[0], -heading[1])


# ---------------------------------------------------------------------------
# Tail release and the blocked tail
# ---------------------------------------------------------------------------


def test_a_step_that_does_not_eat_releases_the_tail():
    """The body keeps its length and the tail cell is vacated.

    Purpose: The tail release is what keeps the snake the same length between
        meals, and it is the precondition for the "chase your own tail" rule
        below.
    Given: A four-cell snake that moves onto an empty cell.
    When: The transition is taken.
    Then: The body has the same length, the head is the new cell, and the old
        tail cell is no longer occupied.

    Test type: unit
    """
    env = build_env()
    body = [(3, 3), (3, 2), (3, 1), (3, 0)]
    state = state_for(env, body, (0, 6))
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.body(successor) == ((3, 4), (3, 3), (3, 2), (3, 1))
    assert (3, 0) not in env.body(successor)


def test_stepping_into_the_cell_the_tail_just_left_is_legal():
    """Chasing your own tail is not a self hit.

    Purpose: This is the single rule most often implemented wrongly -- checking
        the collision against the *old* body makes a legal move fatal, and the
        snake can then never turn inside its own loop.
    Given: A snake curled so that turning brings the head onto its own tail cell,
        with the food elsewhere so the step does not eat.
    When: The transition is taken.
    Then: The successor is still running.

    Test type: unit
    """
    env = build_env()
    # Head at (2, 3) heading east; the tail sits at (3, 3), which a right turn
    # steps into. It is released on the same step, so the cell is free.
    body = [(2, 3), (2, 2), (3, 2), (3, 3)]
    state = state_for(env, body, (0, 0))
    successor = env.sample_next_state(state, int(SnakeAction.TURN_RIGHT))
    assert env.termination(successor) is SnakeTermination.RUNNING
    assert env.body(successor)[0] == (3, 3)


def test_the_tail_is_blocked_on_the_step_the_snake_eats():
    """Eating keeps the tail in place, so the tail cell is a self hit.

    Purpose: The mirror of the rule above, and the reason it cannot simply be
        "the tail is always safe". The two differ only in whether the step ate.
    Given: The same curled snake, but with the food placed on the tail cell the
        head is about to enter.
    When: The transition is taken.
    Then: The episode ends as a self hit.

    Test type: unit
    """
    env = build_env()
    body = [(2, 3), (2, 2), (3, 2), (3, 3)]
    state = state_for(env, body, (3, 3))
    successor = env.sample_next_state(state, int(SnakeAction.TURN_RIGHT))
    assert env.termination(successor) is SnakeTermination.SELF


def test_eating_grows_the_body_by_one_cell():
    """The old tail is kept, so the length goes up by one.

    Given: A three-cell snake with the food directly ahead.
    When: The transition is taken.
    Then: The body is four cells and the counter is reset.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4), steps_since_food=5)
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.body(successor) == ((3, 4), (3, 3), (3, 2), (3, 1))
    assert env.steps_since_food(successor) == 0


# ---------------------------------------------------------------------------
# Food respawn
# ---------------------------------------------------------------------------


def test_food_respawns_only_on_cells_the_new_body_leaves_free():
    """Every respawn lands off the body, and every free cell is reachable.

    Purpose: A respawn onto a body cell would make the food unreachable, and a
        respawn drawn from the *old* body's free cells could land under the new
        head. Both are silent: the episode just becomes unwinnable.
    Given: A snake that eats, and 4000 draws of the successor.
    When: The food cells of those successors are collected.
    Then: None is on the new body, and every free cell appears.

    Test type: unit
    """
    env = build_env(grid_size=4, target_length=6)
    state = state_for(env, [(1, 1), (1, 0), (2, 0)], (1, 2))
    np.random.seed(0)
    drawn = {
        env.food(env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))) for _ in range(4000)
    }
    new_body = {(1, 2), (1, 1), (1, 0), (2, 0)}
    free = {(row, col) for row in range(4) for col in range(4) if (row, col) not in new_body}
    assert drawn == free


def test_food_does_not_move_when_the_snake_does_not_eat():
    """A step that eats nothing leaves the food exactly where it was.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (0, 6))
    for _ in range(20):
        successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
        assert env.food(successor) == (0, 6)


def test_the_transition_is_deterministic_when_nothing_is_eaten():
    """The body update carries no randomness of its own.

    Purpose: The formal model says the only random part of a transition is the
        respawn. If anything else drew, seeded trajectories would not reproduce.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (0, 6))
    successors = env.sample_next_state(state, int(SnakeAction.TURN_LEFT), n_samples=25)
    assert np.all(successors == successors[0])


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def test_leaving_the_grid_is_a_wall_hit():
    """The walls sit outside the playable cells, so a head off the grid has hit one.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5))
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.termination(successor) is SnakeTermination.WALL


def test_entering_the_body_is_a_self_hit():
    """A head that lands on a cell the body still occupies ends the episode.

    Test type: unit
    """
    env = build_env()
    body = [(2, 3), (2, 2), (3, 2), (3, 3), (4, 3)]
    state = state_for(env, body, (0, 0))
    successor = env.sample_next_state(state, int(SnakeAction.TURN_RIGHT))
    assert env.termination(successor) is SnakeTermination.SELF


def test_reaching_the_target_length_is_a_win():
    """Growing to ``target_length`` ends the episode as a win, with no respawn.

    Test type: unit
    """
    env = build_env(target_length=4)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4))
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.termination(successor) is SnakeTermination.WIN
    assert env.snake_length(successor) == 4
    assert env.food(successor) is None


def test_the_counter_reaching_the_limit_is_starvation():
    """``starvation_limit`` steps with no food ends the episode.

    Test type: unit
    """
    env = build_env(starvation_limit=4)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (0, 6), steps_since_food=3)
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.termination(successor) is SnakeTermination.STARVATION
    assert env.steps_since_food(successor) == 4


def test_a_wall_hit_on_the_starving_step_is_reported_as_the_wall():
    """Wall and self are checked before the clock, as the formal model orders them.

    Purpose: The ordering decides which failure the metrics attribute the
        episode to, and both conditions can fire on one step.
    Given: A snake one step from both the wall and the starvation limit.
    When: It steps into the wall.
    Then: The termination reason is the wall.

    Test type: unit
    """
    env = build_env(starvation_limit=4)
    state = state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5), steps_since_food=3)
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.termination(successor) is SnakeTermination.WALL


def test_a_win_is_checked_before_starvation():
    """Eating resets the counter, so the winning step can never starve.

    Test type: unit
    """
    env = build_env(target_length=4, starvation_limit=4)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4), steps_since_food=3)
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.termination(successor) is SnakeTermination.WIN


def test_the_terminal_state_absorbs():
    """Every action from a terminal state returns it unchanged, for zero reward.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (5, 5))
    dead = env.sample_next_state(
        state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5)), int(SnakeAction.GO_STRAIGHT)
    )
    assert env.is_terminal(dead)
    for action in env.get_actions():
        assert np.array_equal(env.sample_next_state(dead, action), dead)
        assert env.reward(dead, action) == 0.0
    assert not env.is_terminal(state)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, food, counter, target, limit, expected",
    [
        # Eating pays +1.
        ([(3, 3), (3, 2), (3, 1)], (3, 4), 0, 6, 8, 1.0),
        # Winning happens by eating, so it pays the same +1 and nothing more.
        ([(3, 3), (3, 2), (3, 1)], (3, 4), 0, 4, 8, 1.0),
        # A wall hit pays -1.
        ([(0, 3), (1, 3), (2, 3)], (5, 5), 0, 6, 8, -1.0),
        # Starving pays -1.
        ([(3, 3), (3, 2), (3, 1)], (0, 6), 3, 6, 4, -1.0),
        # An ordinary step pays nothing.
        ([(3, 3), (3, 2), (3, 1)], (0, 6), 0, 6, 8, 0.0),
    ],
)
def test_reward_pays_only_for_eating_and_dying(body, food, counter, target, limit, expected):
    """The reward is +1 for food, -1 for death and 0 for everything else.

    Test type: unit
    """
    env = build_env(target_length=target, starvation_limit=limit)
    state = state_for(env, body, food, steps_since_food=counter)
    assert env.reward(state, int(SnakeAction.GO_STRAIGHT)) == expected


def test_the_declared_reward_range_is_exactly_the_two_outcomes():
    """Nothing stacks, so the range is the tighter pair rather than a sum.

    Test type: unit
    """
    assert build_env().reward_range == (-1.0, 1.0)


def test_reward_does_not_depend_on_the_realised_successor():
    """The flag stays ``False`` and the reward ignores ``next_state``.

    Purpose: Simulation drivers reorder their RNG draws on this flag. Claiming
        a next-state dependency the reward does not have buys the reordering for
        nothing.

    Test type: unit
    """
    env = build_env()
    assert env.reward_requires_next_state is False
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4))
    successor = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT))
    assert env.reward(state, int(SnakeAction.GO_STRAIGHT)) == env.reward(
        state, int(SnakeAction.GO_STRAIGHT), successor
    )


# ---------------------------------------------------------------------------
# Scent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta, expected",
    [
        ((-1, 1), (int(SnakeQuadrant.NORTH_EAST),)),
        ((-3, -2), (int(SnakeQuadrant.NORTH_WEST),)),
        ((2, 4), (int(SnakeQuadrant.SOUTH_EAST),)),
        ((5, -1), (int(SnakeQuadrant.SOUTH_WEST),)),
        # Same row: both eastern quadrants are compatible.
        ((0, 3), (int(SnakeQuadrant.NORTH_EAST), int(SnakeQuadrant.SOUTH_EAST))),
        ((0, -3), (int(SnakeQuadrant.NORTH_WEST), int(SnakeQuadrant.SOUTH_WEST))),
        # Same column: both northern or both southern quadrants.
        ((-4, 0), (int(SnakeQuadrant.NORTH_EAST), int(SnakeQuadrant.NORTH_WEST))),
        ((4, 0), (int(SnakeQuadrant.SOUTH_EAST), int(SnakeQuadrant.SOUTH_WEST))),
    ],
)
def test_compatible_quadrants_include_the_row_and_column_ties(delta, expected):
    """One quadrant off the axes, two on them.

    Purpose: The tie case is the part of the scent model a reader is most
        likely to simplify away, and getting it wrong redistributes probability
        without changing the shape of anything visible.

    Test type: unit
    """
    assert quadrants_for_offset(*delta) == tuple(sorted(expected))


def test_food_on_the_head_has_no_quadrant():
    """A zero offset cannot happen, and is refused rather than guessed at.

    Test type: unit
    """
    with pytest.raises(ValueError, match="never coincide"):
        quadrants_for_offset(0, 0)


def test_scent_probabilities_split_accuracy_across_one_quadrant():
    """Off the axes, the single compatible quadrant carries the whole accuracy.

    Test type: unit
    """
    env = build_env(scent_accuracy=0.7)
    probabilities = env.scent_probabilities((3, 3), (1, 5))
    assert probabilities[int(SnakeQuadrant.NORTH_EAST)] == pytest.approx(0.7)
    for quadrant in (
        SnakeQuadrant.NORTH_WEST,
        SnakeQuadrant.SOUTH_EAST,
        SnakeQuadrant.SOUTH_WEST,
    ):
        assert probabilities[int(quadrant)] == pytest.approx(0.1)
    assert probabilities.sum() == pytest.approx(1.0)


def test_scent_probabilities_split_accuracy_across_a_tie():
    """On the head's row, the two eastern quadrants share the accuracy.

    Purpose: This is the case the formal model spells out and the one an
        implementation is most likely to drop. Both the compatible pair and the
        incompatible pair change, so checking only one side would miss a model
        that normalised the wrong way.

    Test type: unit
    """
    env = build_env(scent_accuracy=0.7)
    probabilities = env.scent_probabilities((3, 3), (3, 6))
    assert probabilities[int(SnakeQuadrant.NORTH_EAST)] == pytest.approx(0.35)
    assert probabilities[int(SnakeQuadrant.SOUTH_EAST)] == pytest.approx(0.35)
    assert probabilities[int(SnakeQuadrant.NORTH_WEST)] == pytest.approx(0.15)
    assert probabilities[int(SnakeQuadrant.SOUTH_WEST)] == pytest.approx(0.15)
    assert probabilities.sum() == pytest.approx(1.0)


def test_sampled_scents_follow_the_declared_probabilities():
    """The sampler and the likelihood agree, including on a tie.

    Purpose: The likelihood is what the belief conditions on and the sampler is
        what the world emits. A mismatch between them is a wrong posterior that
        no single-sided test can see.
    Given: A state whose food shares the head's row, and 20000 readings.
    When: The observed quadrant frequencies are compared with
        ``scent_probabilities``.
    Then: They agree to within Monte Carlo error.

    Test type: integration
    """
    env = build_env(scent_accuracy=0.7, detection_probability=0.0)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 6))
    np.random.seed(0)
    readings = env.sample_observation(state, int(SnakeAction.GO_STRAIGHT), n_samples=20000)
    counts = np.zeros(len(SnakeQuadrant))
    for reading in readings:
        counts[env.decode_observation(reading)[2]] += 1
    assert np.allclose(counts / len(readings), env.scent_probabilities((3, 3), (3, 6)), atol=0.02)


# ---------------------------------------------------------------------------
# Vision window
# ---------------------------------------------------------------------------


def test_the_window_is_the_in_grid_part_of_the_square_around_the_head():
    """The window is clipped to the grid, so a head in a corner sees fewer cells.

    Test type: unit
    """
    env = build_env(window_radius=2)
    assert len(env.window_cells((3, 3))) == 25
    assert len(env.window_cells((0, 0))) == 9
    assert (0, -1) not in env.window_cells((0, 0))


def test_food_outside_the_window_is_never_reported():
    """The window has no reach beyond its radius.

    Test type: unit
    """
    env = build_env(window_radius=1, detection_probability=1.0)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (6, 6))
    np.random.seed(0)
    for reading in env.sample_observation(state, int(SnakeAction.GO_STRAIGHT), n_samples=200):
        assert env.decode_observation(reading)[1] is None


def test_food_inside_the_window_is_reported_at_the_detection_rate():
    """A sighting names the exact cell, and only at the declared rate.

    Test type: integration
    """
    env = build_env(window_radius=2, detection_probability=0.9)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (4, 4))
    np.random.seed(1)
    readings = env.sample_observation(state, int(SnakeAction.GO_STRAIGHT), n_samples=20000)
    seen = [env.decode_observation(reading)[1] for reading in readings]
    assert set(cell for cell in seen if cell is not None) == {(4, 4)}
    assert np.mean([cell is not None for cell in seen]) == pytest.approx(0.9, abs=0.02)


def test_there_are_no_false_positives():
    """A sighting of any cell but the food has zero likelihood.

    Purpose: This is what makes a sighting conclusive, and it is the property
        the exact belief relies on to collapse onto one cell.

    Test type: unit
    """
    env = build_env(window_radius=2)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (4, 4))
    body = env.body(state)
    impossible = env.encode_observation_tuple(body, (2, 2), int(SnakeQuadrant.SOUTH_EAST))
    scores = env.observation_log_probability(state, int(SnakeAction.GO_STRAIGHT), [impossible])
    assert scores[0] == -np.inf


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def test_the_observation_reports_the_body_exactly():
    """The agent is told its own body, head first, with no noise.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (4, 4))
    np.random.seed(0)
    reading = env.sample_observation(state, int(SnakeAction.GO_STRAIGHT))
    assert env.decode_observation(reading)[0] == env.body(state)


def test_every_terminal_state_emits_the_same_reading():
    """Wins and deaths are indistinguishable through the sensor.

    Purpose: Distinct terminal readings would let a planner read the outcome
        off the observation rather than off the reward, which is not the model.

    Test type: unit
    """
    env = build_env(target_length=4, starvation_limit=4)
    wall = env.sample_next_state(
        state_for(env, [(0, 3), (1, 3), (2, 3)], (5, 5)), int(SnakeAction.GO_STRAIGHT)
    )
    win = env.sample_next_state(
        state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 4)), int(SnakeAction.GO_STRAIGHT)
    )
    for state in (wall, win):
        assert env.sample_observation(state, 0) == TERMINAL_OBSERVATION
        assert env.observation_log_probability(state, 0, [TERMINAL_OBSERVATION])[0] == 0.0


def test_observation_log_probability_sums_to_one_over_the_reachable_readings():
    """The three parts of a reading are a proper joint distribution.

    Purpose: The parts are scored independently and added in log space. A
        normalisation error in either part is invisible in a single likelihood
        and fatal in a belief update.
    Given: A state whose food sits inside the window and on the head's row, so
        both the sighting and the scent tie are exercised at once.
    When: Every reachable reading is scored and the likelihoods are summed.
    Then: They sum to one.

    Test type: unit
    """
    env = build_env(window_radius=2, detection_probability=0.9, scent_accuracy=0.7)
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (3, 5))
    body = env.body(state)
    readings = [
        env.encode_observation_tuple(body, seen, int(quadrant))
        for seen in (None, (3, 5))
        for quadrant in SnakeQuadrant
    ]
    scores = env.observation_log_probability(state, int(SnakeAction.GO_STRAIGHT), readings)
    assert float(np.exp(scores).sum()) == pytest.approx(1.0)


def test_observations_are_hashable_and_compare_by_value():
    """``hash_observation`` agrees with ``is_equal_observation``.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (4, 4))
    np.random.seed(0)
    reading = env.sample_observation(state, 0)
    copy = tuple(int(value) for value in reading)
    assert env.is_equal_observation(reading, copy)
    assert env.hash_observation(reading) == env.hash_observation(copy)
    assert reading[0] == OBSERVATION_LIVE


# ---------------------------------------------------------------------------
# Transition likelihood
# ---------------------------------------------------------------------------


def test_transition_log_probability_is_uniform_over_the_respawn_cells():
    """Every consistent successor of an eating step shares the respawn mass.

    Test type: unit
    """
    env = build_env(grid_size=4, target_length=6)
    state = state_for(env, [(1, 1), (1, 0), (2, 0)], (1, 2))
    np.random.seed(0)
    successors = env.sample_next_state(state, int(SnakeAction.GO_STRAIGHT), n_samples=200)
    unique = np.unique(successors, axis=0)
    scores = env.transition_log_probability(state, int(SnakeAction.GO_STRAIGHT), unique)
    assert np.allclose(scores, -np.log(12))
    assert float(np.exp(scores).sum()) == pytest.approx(1.0)


def test_transition_log_probability_rejects_an_inconsistent_successor():
    """A successor the body update cannot produce scores ``-inf``.

    Test type: unit
    """
    env = build_env()
    state = state_for(env, [(3, 3), (3, 2), (3, 1)], (0, 6))
    wrong = state_for(env, [(3, 5), (3, 4), (3, 3)], (0, 6))
    scores = env.transition_log_probability(state, int(SnakeAction.GO_STRAIGHT), [wrong])
    assert scores[0] == -np.inf


# ---------------------------------------------------------------------------
# Batch paths and serialization
# ---------------------------------------------------------------------------


def test_the_batch_paths_agree_with_the_single_ones():
    """A particle filter and a tree expansion must score the same transition alike.

    Test type: integration
    """
    env = build_env()
    np.random.seed(0)
    states = env.initial_state_dist().sample(6)
    action = int(SnakeAction.GO_STRAIGHT)
    batched = env.reward_batch(states, action)
    looped = np.array([env.reward(state, action) for state in states])
    assert np.allclose(batched, looped)

    np.random.seed(2)
    batch_states = env.sample_next_state_batch(states, action)
    assert batch_states.shape == (6, env.state_size)
    assert batch_states.dtype == np.float64


def test_serialization_round_trips_and_config_id_is_stable():
    """An environment rebuilt from its own dict is the same environment.

    Test type: integration
    """
    env = build_env()
    rebuilt = SnakePOMDP.from_dict(env.to_dict())
    assert rebuilt == env
    assert rebuilt.config_id == env.config_id

    before = env.config_id
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    env.sample_next_step(state, int(SnakeAction.GO_STRAIGHT))
    assert env.config_id == before


def test_the_initial_state_is_the_documented_starting_snake():
    """Head at the centre, three cells extending west, heading east, counter zero.

    Test type: unit
    """
    env = build_env(grid_size=12)
    np.random.seed(0)
    state = env.initial_state_dist().sample()[0]
    assert env.body(state) == ((6, 6), (6, 5), (6, 4))
    assert env.heading(state) == (0, 1)
    assert env.steps_since_food(state) == 0
    assert env.food(state) not in env.body(state)


def test_the_initial_food_is_uniform_over_the_free_cells():
    """The prior really is uniform, which is what the belief starts from.

    Test type: integration
    """
    env = build_env(grid_size=4, target_length=6)
    distribution = env.initial_state_dist()
    np.random.seed(0)
    states = distribution.sample(8000)
    counts = np.zeros(16)
    for state in states:
        food = env.food(state)
        counts[food[0] * 4 + food[1]] += 1
    free = distribution.free_cells()
    assert set(np.flatnonzero(counts)) == set(int(cell) for cell in free)
    assert np.allclose(counts[free] / len(states), 1.0 / free.size, atol=0.02)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"grid_size": 2}, "grid_size"),
        ({"target_length": 3}, "target_length"),
        ({"detection_probability": 1.5}, "detection_probability"),
        ({"scent_accuracy": -0.1}, "scent_accuracy"),
        ({"window_radius": -1}, "window_radius"),
        ({"starvation_limit": 0}, "starvation_limit"),
    ],
)
def test_invalid_configurations_are_refused_at_construction(kwargs, message):
    """A bad setting fails where it is written, not on the first episode.

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        build_env(**kwargs)
