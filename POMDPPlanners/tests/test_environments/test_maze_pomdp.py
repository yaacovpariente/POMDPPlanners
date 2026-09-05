# SPDX-License-Identifier: MIT

"""Behavior tests for the generated discrete and continuous Maze POMDPs."""

from collections import deque

import numpy as np
import pytest

from POMDPPlanners.environments import get_environment
from POMDPPlanners.environments.maze_pomdp import (
    ContinuousMazePOMDP,
    DiscreteMazePOMDP,
    MazeGeometry,
)
from POMDPPlanners.environments.t_maze_pomdp.maze_pomdp import (
    ACTION_OFFSETS,
    CUE_CONSUMED,
    CUE_EMITTING,
    CUE_UNSEEN,
    GOAL_LEFT,
    GOAL_RIGHT,
    OBSERVATION_EMPTY,
    OBSERVATION_LEFT_CUE,
    OBSERVATION_RIGHT_CUE,
    create_maze_state,
)
from POMDPPlanners.environments.t_maze_pomdp.t_maze_pomdp import TMazePOMDP


def _path(geometry, target):
    """Return one shortest cell path from start to ``target``."""
    parents = {geometry.start_cell: None}
    queue = deque([geometry.start_cell])
    while queue:
        cell = queue.popleft()
        if cell == target:
            break
        for neighbour in geometry.neighbours(cell):
            if neighbour not in parents:
                parents[neighbour] = cell
                queue.append(neighbour)
    cells = []
    cell = target
    while cell is not None:
        cells.append(cell)
        cell = parents[cell]
    return list(reversed(cells))


def _action_between(first, second):
    delta = (second[0] - first[0], second[1] - first[1])
    return next(action for action, offset in ACTION_OFFSETS.items() if offset == delta)


class TestGeneratedGeometry:
    @pytest.mark.parametrize("width,height", [(7, 9), (11, 13), (17, 19)])
    def test_documented_sizes_are_connected_and_landmarks_are_distinct(self, width, height):
        geometry = MazeGeometry(width=width, height=height, seed=7)
        distances = geometry.shortest_path_lengths()
        assert len(distances) == len(geometry.walkable)
        assert len(
            {
                geometry.start_cell,
                geometry.cue_cell,
                geometry.left_goal_cell,
                geometry.right_goal_cell,
            }
        ) == 4
        assert geometry.left_goal_cell in distances
        assert geometry.right_goal_cell in distances

    def test_medium_layout_has_branches_dead_ends_and_loops(self):
        geometry = MazeGeometry(width=11, height=13, seed=7, loop_fraction=0.15)
        assert geometry.branch_cells()
        assert geometry.dead_end_cells()
        assert geometry.loop_count() >= 1

    def test_seed_is_repeatable_and_can_change_the_layout(self):
        first = MazeGeometry(width=11, height=13, seed=3)
        again = MazeGeometry(width=11, height=13, seed=3)
        other = MazeGeometry(width=11, height=13, seed=4)
        assert first.walkable == again.walkable
        assert first.walkable != other.walkable

    def test_geometry_does_not_advance_global_numpy_rng(self):
        np.random.seed(42)
        expected = np.random.random(4)
        np.random.seed(42)
        MazeGeometry(width=11, height=13, seed=99)
        assert np.array_equal(np.random.random(4), expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 6},
            {"width": 8},
            {"height": 8},
            {"height": 10},
            {"width": 7.5},
            {"loop_fraction": -0.1},
            {"loop_fraction": 1.1},
        ],
    )
    def test_invalid_dimensions_and_loop_fraction_are_rejected(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            MazeGeometry(**kwargs)


class TestSharedModel:
    def test_same_seed_and_dimensions_share_exact_geometry(self):
        discrete = DiscreteMazePOMDP(maze_width=11, maze_height=13, maze_seed=8)
        continuous = ContinuousMazePOMDP(maze_width=11, maze_height=13, maze_seed=8)
        assert discrete.geometry.walkable == continuous.geometry.walkable
        assert discrete.start_cell == continuous.start_cell
        assert discrete.cue_cell == continuous.cue_cell
        assert discrete.geometry.goal_cells == continuous.geometry.goal_cells

    @pytest.mark.parametrize("env_type", [DiscreteMazePOMDP, ContinuousMazePOMDP])
    def test_serialization_and_config_identity_round_trip(self, env_type):
        env = env_type(maze_width=11, maze_height=13, maze_seed=12, cue_accuracy=1.0)
        before = env.config_id
        rebuilt = env_type.from_dict(env.to_dict())
        assert rebuilt == env
        assert rebuilt.config_id == before
        action = (
            env.get_actions()[0]
            if isinstance(env, DiscreteMazePOMDP)
            else np.array([0.0, 0.5])
        )
        env.sample_next_state(env.initial_state_dist().sample()[0], action)
        assert env.config_id == before

    def test_public_registry_uses_maze_names_and_keeps_legacy_alias(self):
        assert isinstance(get_environment("DiscreteMazePOMDP"), DiscreteMazePOMDP)
        assert isinstance(get_environment("ContinuousMazePOMDP"), ContinuousMazePOMDP)
        assert isinstance(get_environment("TMazePOMDP"), TMazePOMDP)


class TestCueAndRewards:
    @pytest.mark.parametrize(
        "goal,expected",
        [(GOAL_LEFT, OBSERVATION_LEFT_CUE), (GOAL_RIGHT, OBSERVATION_RIGHT_CUE)],
    )
    def test_discrete_cue_is_emitted_once_even_after_bump_and_revisit(self, goal, expected):
        env = DiscreteMazePOMDP(cue_accuracy=1.0)
        state = create_maze_state(env.start_cell, goal, CUE_UNSEEN)
        state = env.sample_next_state(state, "up")
        assert tuple(state[:2]) == env.cue_cell
        assert state[3] == CUE_EMITTING
        assert env.sample_observation(state, "up") == expected

        # The bump has to be a real bump, or "even after a bump" is untested: the
        # cue cell's only neighbours are the start below and the maze above.
        bumped = env.sample_next_state(state, "left")
        assert tuple(bumped[:2]) == env.cue_cell
        assert env.step_info(state, "left", bumped)["wall_collision"] == 1.0
        assert bumped[3] == CUE_CONSUMED
        assert env.sample_observation(bumped, "left") == OBSERVATION_EMPTY

        # And the revisit has to leave and come back, not stand still.
        left = env.sample_next_state(bumped, "down")
        assert tuple(left[:2]) == env.start_cell
        state = env.sample_next_state(left, "up")
        assert tuple(state[:2]) == env.cue_cell
        assert state[3] == CUE_CONSUMED
        assert env.sample_observation(state, "up") == OBSERVATION_EMPTY

    def test_continuous_crossing_arms_cue_and_zero_motion_consumes_it(self):
        env = ContinuousMazePOMDP(max_step_size=2.0, cue_accuracy=1.0)
        state = create_maze_state(env.start_cell, GOAL_LEFT, CUE_UNSEEN)
        state = env.sample_next_state(state, np.array([0.0, 1.8]))
        assert state[1] == pytest.approx(2.8)
        assert state[3] == CUE_EMITTING
        assert env.sample_observation(state, np.zeros(2)) == OBSERVATION_LEFT_CUE
        next_state = env.sample_next_state(state, np.zeros(2))
        assert np.array_equal(next_state[:2], state[:2])
        assert next_state[3] == CUE_CONSUMED
        assert env.step_info(state, np.zeros(2), next_state)["wall_collision"] == 0.0

    @pytest.mark.parametrize("target_side", [GOAL_LEFT, GOAL_RIGHT])
    def test_correct_and_wrong_goal_are_terminal_and_paid_once(self, target_side):
        env = DiscreteMazePOMDP()
        target = env.goal_cell(target_side)
        path = _path(env.geometry, target)
        predecessor = path[-2]
        action = _action_between(predecessor, target)

        correct = create_maze_state(predecessor, target_side, CUE_CONSUMED)
        correct_next = env.sample_next_state(correct, action)
        assert env.is_terminal(correct_next)
        assert env.reward(correct, action, correct_next) == 10.0
        assert env.reward(correct_next, action, env.sample_next_state(correct_next, action)) == 0.0

        wrong_side = GOAL_RIGHT if target_side == GOAL_LEFT else GOAL_LEFT
        wrong = create_maze_state(predecessor, wrong_side, CUE_CONSUMED)
        wrong_next = env.sample_next_state(wrong, action)
        assert env.is_terminal(wrong_next)
        assert env.reward(wrong, action, wrong_next) == -10.0

    def test_observation_scalar_and_per_state_likelihoods_agree(self):
        env = DiscreteMazePOMDP(cue_accuracy=0.8)
        states = np.stack(
            [
                create_maze_state(env.cue_cell, GOAL_LEFT, CUE_EMITTING),
                create_maze_state(env.cue_cell, GOAL_RIGHT, CUE_EMITTING),
                create_maze_state(env.start_cell, GOAL_LEFT, CUE_UNSEEN),
            ]
        )
        for observation in (OBSERVATION_LEFT_CUE, OBSERVATION_RIGHT_CUE, OBSERVATION_EMPTY):
            expected = np.array(
                [
                    env.observation_log_probability(state, "up", [observation])[0]
                    for state in states
                ]
            )
            actual = env.observation_log_probability_per_state(states, "up", observation)
            assert np.array_equal(actual, expected)

    def test_step_info_separates_correct_wrong_and_timeout_endings(self):
        env = DiscreteMazePOMDP()
        correct = create_maze_state(env.left_goal_cell, GOAL_LEFT, CUE_CONSUMED)
        wrong = create_maze_state(env.right_goal_cell, GOAL_LEFT, CUE_CONSUMED)
        active = create_maze_state(env.start_cell, GOAL_LEFT, CUE_UNSEEN)
        correct_info = env.step_info(correct, None, None)
        wrong_info = env.step_info(wrong, None, None)
        timeout_info = env.step_info(active, None, None)
        keys = ("ended_by_goal", "ended_by_failure", "ended_by_timeout")
        assert tuple(correct_info[key] for key in keys) == (1.0, 0.0, 0.0)
        assert tuple(wrong_info[key] for key in keys) == (0.0, 1.0, 0.0)
        assert tuple(timeout_info[key] for key in keys) == (0.0, 0.0, 1.0)


class TestContinuousMotion:
    def test_real_position_is_not_quantized(self):
        env = ContinuousMazePOMDP()
        state = create_maze_state(env.start_cell, GOAL_LEFT)
        next_state = env.sample_next_state(state, np.array([0.0, 0.4]))
        assert next_state[1] == pytest.approx(env.start_cell[1] + 0.4)

    def test_action_is_scaled_to_the_documented_bound(self):
        env = ContinuousMazePOMDP(max_step_size=1.0)
        assert np.allclose(env.clip_action(np.array([3.0, 4.0])), np.array([0.6, 0.8]))
        for invalid in (np.array([1.0]), np.array([1.0, 2.0, 3.0]), np.array([np.nan, 0.0])):
            with pytest.raises(ValueError):
                env.clip_action(invalid)

    def test_swept_segment_cannot_tunnel_through_a_wall(self):
        env = ContinuousMazePOMDP(maze_width=11, maze_height=13, maze_seed=7, max_step_size=2.0)
        wall_case = None
        for x in range(1, env.maze_width - 1):
            for y in range(1, env.maze_height - 1):
                if (x, y) in env.walkable_cells:
                    continue
                if (x - 1, y) in env.walkable_cells and (x + 1, y) in env.walkable_cells:
                    wall_case = ((x - 1, y), np.array([2.0, 0.0]))
                    break
                if (x, y - 1) in env.walkable_cells and (x, y + 1) in env.walkable_cells:
                    wall_case = ((x, y - 1), np.array([0.0, 2.0]))
                    break
            if wall_case is not None:
                break
        assert wall_case is not None
        start, action = wall_case
        state = create_maze_state(start, GOAL_LEFT, CUE_CONSUMED)
        next_state = env.sample_next_state(state, action)
        assert np.array_equal(next_state[:2], state[:2])
        assert env.step_info(state, action, next_state)["wall_collision"] == 1.0

    def test_diagonal_between_two_walkable_cells_cannot_cut_a_wall_corner(self):
        """Refusal must come from the corner, not from the destination.

        Both endpoints here are walkable, so the only thing that can stop the move
        is the wall the swept path touches at the corner point between them.
        """
        env = ContinuousMazePOMDP(max_step_size=2.0)
        pairs = [
            (cell, (cell[0] + dx, cell[1] + dy))
            for cell in sorted(env.walkable_cells)
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            if (cell[0] + dx, cell[1] + dy) in env.walkable_cells
            and (cell[0] + dx, cell[1]) not in env.walkable_cells
            and cell not in env.geometry.goal_cells
        ]
        assert pairs, "default layout must contain at least one such diagonal"
        for origin, destination in pairs:
            state = create_maze_state(origin, GOAL_LEFT, CUE_CONSUMED)
            action = np.array(
                [destination[0] - origin[0], destination[1] - origin[1]], dtype=float
            )
            next_state = env.sample_next_state(state, action)
            assert np.array_equal(next_state[:2], state[:2]), (origin, destination)
            assert env.step_info(state, action, next_state)["wall_collision"] == 1.0

    def test_diagonal_cannot_cut_a_wall_corner_into_a_goal(self):
        """A goal touched at the same instant as a wall must not win the tie.

        Regression: the swept path was judged one cell at a time in dictionary
        order, so a goal cell sharing a corner with a wall ended the move before
        the wall was ever checked. On the default map that let ``[-1, -1]`` from
        (2, 7) finish inside the left goal, paid, having squeezed past wall (2, 6).
        """
        env = ContinuousMazePOMDP(max_step_size=2.0)
        goal = env.left_goal_cell
        approach = (goal[0] + 1, goal[1])
        flank = (goal[0] + 1, goal[1] - 1)
        diagonal = (goal[0], goal[1] - 1)
        assert approach in env.walkable_cells
        assert diagonal in env.walkable_cells
        assert flank not in env.walkable_cells

        state = create_maze_state(approach, GOAL_LEFT, CUE_CONSUMED)
        action = np.array([diagonal[0] - approach[0], diagonal[1] - approach[1]], dtype=float)
        next_state = env.sample_next_state(state, action)
        assert np.array_equal(next_state[:2], state[:2])
        assert not env.is_terminal(next_state)
        assert env.reward(state, action, next_state) == -1.0
        assert env.step_info(state, action, next_state)["wall_collision"] == 1.0

    @pytest.mark.parametrize("side", [GOAL_LEFT, GOAL_RIGHT])
    def test_goal_crossing_stops_and_terminates_before_endpoint(self, side):
        env = ContinuousMazePOMDP(max_step_size=2.0)
        goal = env.goal_cell(side)
        predecessor = _path(env.geometry, goal)[-2]
        direction = np.array([goal[0] - predecessor[0], goal[1] - predecessor[1]], dtype=float)
        state = create_maze_state(predecessor, side, CUE_CONSUMED)
        next_state = env.sample_next_state(state, direction * 1.8)
        assert env.is_terminal(next_state)
        assert env.reward(state, direction * 1.8, next_state) == 10.0
        assert not np.allclose(next_state[:2], np.array(predecessor) + direction * 1.8)
