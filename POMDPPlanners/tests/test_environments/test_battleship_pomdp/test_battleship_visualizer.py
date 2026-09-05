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
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.colors import to_hex  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib.text import Text  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np
import pytest

from POMDPPlanners.core.simulation import StepData
from POMDPPlanners.environments.battleship_pomdp import (
    BattleshipBelief,
    BattleshipPOMDP,
)
from POMDPPlanners.environments.battleship_pomdp.battleship_visualizer import (
    _HIT,
    _MISS,
    _UNKNOWN,
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


class TestLabels:
    """What the picture says about itself.

    A reader who has not seen the code has only the titles, the legends and the
    caption to go on. These tests pin the wording that carries the meaning, so a
    later edit cannot quietly drop an explanation and leave an unlabelled mark.
    """

    @staticmethod
    def _figure_text(fig) -> str:
        """Every string drawn anywhere on the figure, joined.

        The figure is drawn first: tick labels are only formatted at draw time,
        so reading them off an undrawn figure returns empty strings.
        """
        fig.canvas.draw()
        return "\n".join(
            artist.get_text() for artist in fig.findobj(match=Text)
        )

    @staticmethod
    def _legend_entries(ax) -> dict:
        """``{label: colour}`` for the legend under ``ax``.

        The colour is read off the handle, not the text, because a legend that
        names the right things next to the wrong swatches is exactly as
        misleading as no legend at all.
        """
        legend = ax.get_legend()
        assert legend is not None, "panel has no legend"
        entries = {}
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            colour = handle.get_markerfacecolor()
            if colour in ("none", "None"):
                colour = handle.get_markeredgecolor()
            entries[text.get_text()] = to_hex(colour)
        return entries

    def test_every_panel_says_what_it_shows(self, env) -> None:
        """Purpose: "Belief" alone does not tell a reader what the panel is of.

        Given: a fresh figure
        When: each panel's text is read
        Then: each panel carries its own plain-words description and its own
              row/column axis labels

        Test type: unit
        """
        fig, axes, _ = BattleshipVisualizer(env)._setup_figure()  # pylint: disable=protected-access
        try:
            fig.canvas.draw()
            descriptions = (
                "what the agent has observed so far",
                "estimated chance a cell contains a ship",
                "actual ship layout, hidden from the agent",
            )
            for ax, description in zip(axes, descriptions):
                panel_text = "\n".join(
                    artist.get_text() for artist in ax.findobj(match=Text)
                )
                assert description in panel_text
                assert ax.get_xlabel() == "column"
                assert ax.get_ylabel() == "row"
        finally:
            plt.close(fig)

    def test_every_mark_and_colour_has_a_legend_entry(self, env) -> None:
        """Purpose: an unexplained mark is the defect this change exists to fix.

        The probe ring and the ground-truth crosses used to be drawn with no
        legend at all, so a reader could not tell what either meant.

        Given: a fresh figure
        When: each panel's legend is read
        Then: every mark is named, on the panel that draws it, and each key's
              colour is the colour that mark is actually drawn in

        Test type: unit
        """
        visualizer = BattleshipVisualizer(env)
        fig, axes, artists = visualizer._setup_figure()  # pylint: disable=protected-access
        try:
            agent = self._legend_entries(axes[0])
            assert agent == {
                "not probed yet": "#c9ccd1",
                "probed - water (miss)": "#2f6fb5",
                "probed - ship (hit)": "#c0392b",
                "probing now - result not on the board yet": "#ffd21f",
            }
            # The agent panel's fills are the colormap the panel is drawn with,
            # so a recoloured board cannot drift away from its own legend.
            colormap = artists["agent"].get_cmap()
            for code, label in zip(
                (_UNKNOWN, _MISS, _HIT),
                ("not probed yet", "probed - water (miss)", "probed - ship (hit)"),
            ):
                assert to_hex(colormap(code)) == agent[label]
            assert (
                to_hex(artists["probe_marker"].get_edgecolor()[0])
                == agent["probing now - result not on the board yet"]
            )

            belief = self._legend_entries(axes[1])
            colormap = artists["belief"].get_cmap()
            assert (
                belief["100% - almost certainly a ship"]
                == to_hex(colormap(1.0))
            )
            assert (
                belief["0% - almost certainly water"]
                == to_hex(colormap(0.0))
            )

            truth = self._legend_entries(axes[2])
            colormap = artists["truth"].get_cmap()
            assert truth["ship cell"] == to_hex(colormap(1.0))
            assert truth["water"] == to_hex(colormap(0.0))
            assert (
                truth["already probed"]
                == to_hex(artists["truth_probe"].get_facecolor()[0])
            )
        finally:
            plt.close(fig)

    def test_belief_colourbar_is_labelled_as_a_percentage(self, env) -> None:
        """Purpose: a bar running 0.0 to 1.0 with no unit is not a probability.

        Given: a fresh figure
        When: the belief colourbar's text is read
        Then: it is labelled and ticked from 0% to 100%

        Test type: unit
        """
        fig, _, _ = BattleshipVisualizer(env)._setup_figure()  # pylint: disable=protected-access
        try:
            text = self._figure_text(fig)
            assert "chance the cell contains a ship" in text
            for tick in ("0%", "25%", "50%", "75%", "100%"):
                assert tick in text
        finally:
            plt.close(fig)

    def test_caption_does_not_claim_the_current_probe_is_resolved(
        self, env, episode
    ) -> None:
        """Purpose: the boards in a frame are the state *before* its own action.

        The ring marks a cell the agent has chosen but not yet probed, and the
        hit tally counts only earlier probes. A caption saying the ring is
        already a hit would describe a board the frame does not draw.

        Given: the first frame of an episode
        When: the caption is rendered
        Then: it says the probe is about to happen and dates the tally to before it

        Test type: unit
        """
        visualizer = BattleshipVisualizer(env)
        frames = visualizer._build_frames(episode)  # pylint: disable=protected-access
        fig, axes, artists = visualizer._setup_figure()  # pylint: disable=protected-access
        try:
            visualizer._animation_function(frames, axes, artists)(0)  # pylint: disable=protected-access
            caption = artists["caption"].get_text()
            assert "about to probe row 0, column 0" in caption
            assert "Ship cells found before this probe" in caption
        finally:
            plt.close(fig)

    def test_terminal_frame_describes_no_probe(self, env, episode) -> None:
        """Purpose: the last recorded step has no action, so nothing is pending.

        Given: the terminal frame of an episode
        When: the caption is rendered
        Then: it says the episode is over and reports the final tally plainly

        Test type: unit
        """
        visualizer = BattleshipVisualizer(env)
        frames = visualizer._build_frames(episode)  # pylint: disable=protected-access
        fig, axes, artists = visualizer._setup_figure()  # pylint: disable=protected-access
        try:
            visualizer._animation_function(frames, axes, artists)(len(frames) - 1)  # pylint: disable=protected-access
            caption = artists["caption"].get_text()
            assert "episode over, no further probe" in caption
            assert "before this probe" not in caption
        finally:
            plt.close(fig)


class TestLayout:
    """Where the drawn pieces actually land.

    Measured on the rendered figure, not on the numbers that produced it: the
    left legend once ran wider than its own panel and overlapped the middle
    one, and every string in the picture was still correct while it did.
    """

    _MIN_GAP_PX = 16.0

    @staticmethod
    def _laid_out_figure(env):
        """The figure with the margins the saved GIF is drawn with, plus a caption."""
        visualizer = BattleshipVisualizer(env)
        fig, axes, artists = visualizer._setup_figure()  # pylint: disable=protected-access
        # The longest caption any frame produces, so the test measures the
        # worst case rather than an empty string.
        artists["caption"].set_text(
            "Step 3 of 18 - the agent is about to probe row 2, column 4 (yellow ring)\n"
            "Result of this probe: MISS - water.   Reward: -0.10.   "
            "Ship cells found before this probe: 1 of 7"
        )
        visualizer._apply_layout(fig)  # pylint: disable=protected-access
        fig.canvas.draw()
        return fig, axes, artists

    def test_panel_legends_do_not_overlap_each_other(self, env) -> None:
        """Purpose: overlapping legends were the reported defect.

        Given: the figure laid out exactly as the saved GIF is
        When: the legends' rendered bounding boxes are measured in pixels
        Then: each legend ends at least 16 px left of the next one's start

        Test type: unit
        """
        fig, axes, _ = self._laid_out_figure(env)
        try:
            renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
            boxes = [ax.get_legend().get_window_extent(renderer) for ax in axes]
            for left, right in zip(boxes, boxes[1:]):
                assert right.x0 - left.x1 >= self._MIN_GAP_PX
        finally:
            plt.close(fig)

    def test_nothing_is_clipped_or_written_over_the_caption(self, env) -> None:
        """Purpose: curing the overlap by spreading out could push text off the figure.

        Given: the figure laid out exactly as the saved GIF is
        When: the legends' and caption's rendered boxes are compared to the
              figure bounds and to each other
        Then: every box is inside the figure and no legend reaches the caption

        Test type: unit
        """
        fig, axes, artists = self._laid_out_figure(env)
        try:
            renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
            width, height = fig.canvas.get_width_height()
            caption = artists["caption"].get_window_extent(renderer)
            boxes = [ax.get_legend().get_window_extent(renderer) for ax in axes]
            for box in boxes + [caption]:
                assert 0 <= box.x0 and box.x1 <= width
                assert 0 <= box.y0 and box.y1 <= height
            assert min(box.y0 for box in boxes) > caption.y1
        finally:
            plt.close(fig)


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
