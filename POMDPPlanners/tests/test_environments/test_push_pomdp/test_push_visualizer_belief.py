# SPDX-License-Identifier: MIT

"""Tests for belief rendering in the Push POMDP visualizers.

Both Push visualizers draw the belief as two particle clouds -- one over the
robot position, one over the object position -- following the convention used
by the LaserTag and Pacman visualizers. These tests cover:
- Particle extraction from a belief over Push states
- The scatter artists actually receiving those particle positions
- Byte-for-byte determinism of repeated renders, which the golden-file
  snapshot tests depend on
"""

import hashlib
from pathlib import Path
from typing import Any, List

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import (
    ContinuousPushPOMDP,
)
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp_visualizer import (
    ContinuousPushPOMDPVisualizer,
    extract_belief_positions,
)
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp_visualizer import (
    PushPOMDPVisualizer,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    continuous_push_pinned_kwargs,
    push_pinned_kwargs,
)


def _belief_from(states: List[np.ndarray]) -> WeightedParticleBelief:
    """Build a uniform-weight belief over the given Push states."""
    return WeightedParticleBelief(
        particles=[np.asarray(s, dtype=float) for s in states],
        log_weights=np.full(len(states), 1.0),
    )


def _spread_belief(center: np.ndarray, n: int = 25, spread: float = 0.4) -> WeightedParticleBelief:
    """Build a belief whose robot/object positions are scattered around ``center``."""
    rng = np.random.default_rng(0)
    particles = []
    for _ in range(n):
        particle = np.asarray(center, dtype=float).copy()
        particle[:4] += rng.normal(0.0, spread, size=4)
        particles.append(particle)
    return _belief_from(particles)


def _history(states: List[np.ndarray], actions: List[Any], beliefs: List[Any]) -> List[StepData]:
    """Build an episode history with one StepData per state."""
    history = []
    for i, state in enumerate(states):
        is_last = i == len(states) - 1
        history.append(
            StepData(
                state=state,
                action=None if is_last else actions[i],
                next_state=states[i + 1] if not is_last else state,
                observation=None,
                reward=0.0,
                belief=beliefs[i],
            )
        )
    return history


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestExtractBeliefPositions:
    """Test the shared belief-to-particle-positions helper.

    Purpose: Validates that robot and object positions are read out of a
        belief over Push states correctly and deterministically

    Given: Beliefs over 6-dimensional Push states
    When: extract_belief_positions is called
    Then: Robot and object position arrays are returned

    Test type: unit
    """

    def test_extracts_robot_and_object_positions(self):
        """Purpose: Validates the state layout is split into the two clouds.

        Given: A belief over two Push states
        When: extract_belief_positions is called
        Then: Columns 0-1 form the robot cloud and columns 2-3 the object cloud

        Test type: unit
        """
        belief = _belief_from(
            [
                np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([1.5, 2.5, 3.5, 4.5, 5.0, 6.0]),
            ]
        )

        robot_points, object_points = extract_belief_positions(belief)

        np.testing.assert_allclose(robot_points, [[1.0, 2.0], [1.5, 2.5]])
        np.testing.assert_allclose(object_points, [[3.0, 4.0], [3.5, 4.5]])

    def test_none_belief_yields_empty_clouds(self):
        """Purpose: Validates histories without beliefs still render.

        Given: No belief
        When: extract_belief_positions is called
        Then: Two empty (0, 2) arrays are returned

        Test type: unit
        """
        robot_points, object_points = extract_belief_positions(None)

        assert robot_points.shape == (0, 2)
        assert object_points.shape == (0, 2)

    def test_non_belief_object_yields_empty_clouds(self):
        """Purpose: Validates an unusable belief degrades instead of raising.

        Given: An object without to_unique_support_distribution
        When: extract_belief_positions is called
        Then: Two empty (0, 2) arrays are returned

        Test type: unit
        """
        robot_points, object_points = extract_belief_positions(object())

        assert robot_points.shape == (0, 2)
        assert object_points.shape == (0, 2)

    def test_duplicate_particles_are_collapsed(self):
        """Purpose: Validates identical particles are drawn once, not stacked.

        Given: A belief holding the same state three times
        When: extract_belief_positions is called
        Then: A single position is returned per cloud

        Test type: unit
        """
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        belief = _belief_from([state, state.copy(), state.copy()])

        robot_points, object_points = extract_belief_positions(belief)

        assert robot_points.shape == (1, 2)
        assert object_points.shape == (1, 2)

    def test_repeated_calls_return_identical_ordering(self):
        """Purpose: Validates the particle ordering the GIF hash depends on.

        Given: One belief over many particles
        When: extract_belief_positions is called twice
        Then: Both calls return element-wise identical arrays

        Test type: unit
        """
        belief = _spread_belief(np.array([4.0, 4.0, 5.0, 5.0, 8.0, 8.0]), n=50)

        first_robot, first_object = extract_belief_positions(belief)
        second_robot, second_object = extract_belief_positions(belief)

        np.testing.assert_array_equal(first_robot, second_robot)
        np.testing.assert_array_equal(first_object, second_object)


class TestPushVisualizerBeliefScatters:
    """Test that the Push visualizers push belief particles into their artists.

    Purpose: Validates the belief actually reaches the rendered scatters

    Given: A visualizer and a belief over Push states
    When: The per-frame belief update runs
    Then: The scatter offsets hold the belief's robot/object positions

    Test type: unit
    """

    @pytest.mark.parametrize("continuous", [False, True])
    def test_belief_scatters_receive_particle_positions(self, continuous):
        """Purpose: Validates both visualizers set scatter offsets from the belief.

        Given: A belief over three Push states
        When: _update_belief_scatters runs for frame 0
        Then: Robot and object scatters hold the matching particle positions

        Test type: unit
        """
        import matplotlib.pyplot as plt

        if continuous:
            env = ContinuousPushPOMDP(discount_factor=0.95, **continuous_push_pinned_kwargs())
            visualizer: Any = ContinuousPushPOMDPVisualizer(env)
        else:
            env = PushPOMDP(discount_factor=0.95, **push_pinned_kwargs())
            visualizer = PushPOMDPVisualizer(env)

        belief = _belief_from(
            [
                np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                np.array([1.5, 2.5, 3.5, 4.5, 5.0, 6.0]),
                np.array([2.0, 3.0, 4.0, 5.0, 5.0, 6.0]),
            ]
        )

        fig, ax = plt.subplots()
        try:
            robot_scatter, object_scatter = visualizer._initialize_belief_scatters(ax)
            visualizer._update_belief_scatters(robot_scatter, object_scatter, [belief], 0)

            np.testing.assert_allclose(
                robot_scatter.get_offsets(), [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]
            )
            np.testing.assert_allclose(
                object_scatter.get_offsets(), [[3.0, 4.0], [3.5, 4.5], [4.0, 5.0]]
            )
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("continuous", [False, True])
    def test_missing_belief_clears_scatters(self, continuous):
        """Purpose: Validates a step without a belief draws no stale particles.

        Given: A frame whose belief is None
        When: _update_belief_scatters runs
        Then: Both scatters end up empty

        Test type: unit
        """
        import matplotlib.pyplot as plt

        if continuous:
            env = ContinuousPushPOMDP(discount_factor=0.95, **continuous_push_pinned_kwargs())
            visualizer: Any = ContinuousPushPOMDPVisualizer(env)
        else:
            env = PushPOMDP(discount_factor=0.95, **push_pinned_kwargs())
            visualizer = PushPOMDPVisualizer(env)

        fig, ax = plt.subplots()
        try:
            robot_scatter, object_scatter = visualizer._initialize_belief_scatters(ax)
            visualizer._update_belief_scatters(robot_scatter, object_scatter, [None], 0)

            assert robot_scatter.get_offsets().shape == (0, 2)
            assert object_scatter.get_offsets().shape == (0, 2)
        finally:
            plt.close(fig)


class TestPushVisualizerBeliefRendering:
    """Test end-to-end GIF rendering with beliefs attached.

    Purpose: Validates the animation renders and stays deterministic

    Given: Episode histories carrying particle beliefs
    When: create_visualization is called
    Then: A GIF is written, and repeated renders are byte-identical

    Test type: integration
    """

    def _push_history(self) -> List[StepData]:
        states = [
            np.array([2.0, 2.0, 3.0, 3.0, 8.0, 8.0]),
            np.array([3.0, 2.0, 3.0, 3.0, 8.0, 8.0]),
            np.array([4.0, 3.0, 4.0, 4.0, 8.0, 8.0]),
        ]
        beliefs = [_spread_belief(s, n=20, spread=0.4 / (i + 1)) for i, s in enumerate(states)]
        return _history(states, ["right", "up"], beliefs)

    def _continuous_push_history(self) -> List[StepData]:
        states = [
            np.array([2.0, 2.0, 3.0, 3.0, 8.0, 8.0]),
            np.array([3.0, 2.0, 3.0, 3.0, 8.0, 8.0]),
            np.array([4.0, 3.0, 4.0, 4.0, 8.0, 8.0]),
        ]
        beliefs = [_spread_belief(s, n=20, spread=0.4 / (i + 1)) for i, s in enumerate(states)]
        actions = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
        return _history(states, actions, beliefs)

    def test_push_visualization_with_belief_is_deterministic(self, tmp_path):
        """Purpose: Validates the discrete Push GIF is reproducible byte-for-byte.

        Given: One history rendered twice by the same visualizer
        When: Both GIFs are hashed
        Then: The hashes match

        Test type: integration
        """
        env = PushPOMDP(
            discount_factor=0.95,
            **push_pinned_kwargs(
                grid_size=10,
                obstacles=[(5.0, 5.0)],
                dangerous_areas=[(6.0, 3.0)],
            ),
        )
        visualizer = PushPOMDPVisualizer(env)
        history = self._push_history()

        first = tmp_path / "push_1.gif"
        second = tmp_path / "push_2.gif"
        visualizer.create_visualization(history, first)
        visualizer.create_visualization(history, second)

        assert first.exists() and second.exists()
        assert _file_hash(first) == _file_hash(second)

    def test_continuous_push_visualization_with_belief_is_deterministic(self, tmp_path):
        """Purpose: Validates the continuous Push GIF is reproducible byte-for-byte.

        Given: One history rendered twice by the same visualizer
        When: Both GIFs are hashed
        Then: The hashes match

        Test type: integration
        """
        env = ContinuousPushPOMDP(
            discount_factor=0.95,
            **continuous_push_pinned_kwargs(
                grid_size=10,
                obstacles=[(5.0, 5.0, 0.5)],
                dangerous_areas=[(6.0, 3.0)],
            ),
        )
        visualizer = ContinuousPushPOMDPVisualizer(env)
        history = self._continuous_push_history()

        first = tmp_path / "continuous_push_1.gif"
        second = tmp_path / "continuous_push_2.gif"
        visualizer.create_visualization(history, first)
        visualizer.create_visualization(history, second)

        assert first.exists() and second.exists()
        assert _file_hash(first) == _file_hash(second)

    def test_history_without_beliefs_still_renders(self, tmp_path):
        """Purpose: Validates belief drawing is optional, not required.

        Given: A history whose steps carry no belief
        When: create_visualization is called
        Then: A non-empty GIF is written

        Test type: integration
        """
        env = PushPOMDP(discount_factor=0.95, **push_pinned_kwargs(grid_size=10))
        visualizer = PushPOMDPVisualizer(env)
        states = [
            np.array([2.0, 2.0, 3.0, 3.0, 8.0, 8.0]),
            np.array([3.0, 2.0, 3.0, 3.0, 8.0, 8.0]),
        ]
        history = _history(states, ["right"], [None, None])

        output = tmp_path / "push_no_belief.gif"
        visualizer.create_visualization(history, output)

        assert output.exists()
        assert output.stat().st_size > 0
