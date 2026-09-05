# SPDX-License-Identifier: MIT

"""Tests for the Battleship visualizer.

The golden-hash test in ``test_environment_visualizations_golden_files.py``
catches *changes* to the rendering, which means it cannot catch a renderer that
was wrong from the first frame. These tests check the things that have to be
true for the picture to mean what its labels say: one frame per recorded step,
an agent panel that carries no information the agent does not have, and a belief
panel that is the belief.
"""

from pathlib import Path

import numpy as np
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.battleship_pomdp import (
    BattleshipBelief,
    BattleshipPOMDP,
)
from POMDPPlanners.environments.battleship_pomdp.battleship_visualizer import (
    BattleshipVisualizer,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import battleship_pinned_kwargs


@pytest.fixture(name="env")
def _env() -> BattleshipPOMDP:
    return BattleshipPOMDP(discount_factor=0.99, **battleship_pinned_kwargs())


@pytest.fixture(name="episode")
def _episode(env: BattleshipPOMDP):
    np.random.seed(21)
    belief = BattleshipBelief.from_environment(env, n_particles=16)
    state = belief.sample()
    history = []
    for action in (0, 1, 2, 12, 13, 24):
        next_state, observation, reward = env.sample_next_step(state, action)
        history.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=reward,
                belief=belief,
            )
        )
        belief = belief.update(action=action, observation=observation, pomdp=env)
        state = next_state
    history.append(
        StepData(state=state, action=None, next_state=None, observation=None, reward=None,
                 belief=belief)
    )
    return history


class TestFrames:
    """What each frame is built from."""

    def test_one_frame_per_recorded_step(self, env, episode) -> None:
        """Purpose: a frame count that drifts from the episode length misreads it.

        Given: a seven-step episode history
        When: frames are built
        Then: there are seven frames, indexed in order

        Test type: unit
        """
        frames = BattleshipVisualizer(env)._build_frames(episode)  # pylint: disable=protected-access
        assert len(frames) == len(episode)
        assert [frame["index"] for frame in frames] == list(range(len(episode)))

    def test_agent_panel_leaks_nothing_about_unprobed_cells(self, env, episode) -> None:
        """Purpose: the agent view must not draw the hidden fleet.

        This is the panel a reviewer uses to judge whether a planner searched
        deliberately. If it showed unprobed ship cells, every planner would look
        clairvoyant and the panel would be worthless.

        Given: each frame of an episode
        When: the agent panel is compared with the true board
        Then: every unprobed cell reads "unknown", whether or not it holds a ship

        Test type: unit
        """
        frames = BattleshipVisualizer(env)._build_frames(episode)  # pylint: disable=protected-access
        for frame, step in zip(frames, episode):
            probed = env.probed(step.state).reshape(env.board_size, env.board_size)
            assert np.all(frame["agent_view"][~probed] == 0)

    def test_agent_panel_marks_hits_and_misses_correctly(self, env, episode) -> None:
        """Purpose: a swapped hit/miss code would invert the whole picture.

        Given: each frame of an episode
        When: the agent panel's probed cells are compared with the true board
        Then: probed ship cells read "hit" and probed water reads "miss"

        Test type: unit
        """
        frames = BattleshipVisualizer(env)._build_frames(episode)  # pylint: disable=protected-access
        for frame, step in zip(frames, episode):
            probed = env.probed(step.state).reshape(env.board_size, env.board_size)
            occupancy = env.occupancy(step.state).reshape(env.board_size, env.board_size)
            assert np.all(frame["agent_view"][probed & occupancy] == 2)
            assert np.all(frame["agent_view"][probed & ~occupancy] == 1)

    def test_truth_panel_is_the_same_board_in_every_frame(self, env, episode) -> None:
        """Purpose: the fleet is fixed, so the ground-truth panel must not move.

        Given: every frame of an episode
        When: the truth panels are compared
        Then: they are identical

        Test type: unit
        """
        frames = BattleshipVisualizer(env)._build_frames(episode)  # pylint: disable=protected-access
        for frame in frames[1:]:
            assert np.array_equal(frame["truth"], frames[0]["truth"])

    def test_belief_panel_is_the_belief_marginal(self, env, episode) -> None:
        """Purpose: the belief panel must be the belief, not a summary of the state.

        Given: each frame
        When: the belief panel is compared with the step's own belief marginal
        Then: they agree cell for cell

        Test type: unit
        """
        frames = BattleshipVisualizer(env)._build_frames(episode)  # pylint: disable=protected-access
        for frame, step in zip(frames, episode):
            expected = step.belief.occupancy_marginal(env).reshape(env.board_size, env.board_size)
            assert np.allclose(frame["belief"], expected)

    def test_belief_panel_falls_back_to_the_particle_mean(self, env) -> None:
        """Purpose: an episode recorded with a generic filter must still render.

        Given: a plain weighted particle belief
        When: its marginal is requested
        Then: the weighted particle mean comes back, in range

        Test type: unit
        """
        from POMDPPlanners.core.belief import WeightedParticleBelief  # pylint: disable=import-outside-toplevel

        particles = np.asarray(BattleshipBelief.from_environment(env, 8).particles)
        plain = WeightedParticleBelief(
            particles=list(particles), log_weights=np.linspace(-1.0, -0.1, 8)
        )
        marginal = BattleshipVisualizer(env)._belief_marginal(plain)  # pylint: disable=protected-access
        assert marginal.shape == (env.num_cells,)
        assert np.all((marginal >= 0.0) & (marginal <= 1.0))


class TestOutput:
    """The file the environment writes."""

    def test_cache_visualization_writes_a_named_gif(self, env, episode, tmp_path: Path) -> None:
        """Purpose: the environment owns the filename; callers pass a directory.

        Given: an episode and an output directory
        When: cache_visualization is called for episode 3
        Then: a non-empty GIF named for that episode exists

        Test type: integration
        """
        env.cache_visualization(episode, tmp_path, 3)
        written = tmp_path / "battleship_board_3.gif"
        assert written.exists()
        assert written.stat().st_size > 0
        assert written.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_empty_history_is_refused(self, env, tmp_path: Path) -> None:
        """Purpose: an empty GIF is a silently broken artifact.

        Given: an empty history
        When: a visualization is requested
        Then: a ValueError says so

        Test type: unit
        """
        with pytest.raises(ValueError, match="empty history"):
            BattleshipVisualizer(env).create_visualization([], tmp_path / "x.gif")

    def test_non_gif_path_is_refused(self, env, episode, tmp_path: Path) -> None:
        """Purpose: the pillow writer is chosen for GIF and nothing else.

        Given: a path that is not a GIF
        When: a visualization is requested
        Then: a ValueError says so

        Test type: unit
        """
        with pytest.raises(ValueError, match=r"\.gif"):
            BattleshipVisualizer(env).create_visualization(episode, tmp_path / "x.mp4")
