# SPDX-License-Identifier: MIT

"""Tests for the Racetrack world's live frame and its reference track."""

from unittest.mock import Mock

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import RacetrackPOMDP


def test_live_render_frame_contract_is_preserved():
    env = RacetrackPOMDP(discount_factor=0.95, max_tracked_agents=1, render_mode="rgb_array")
    assert env.render_frame() is None

    expected = np.full((3, 4, 3), 17, dtype=np.uint8)
    session = Mock()
    session.render_frame.return_value = expected
    env._session = session  # pylint: disable=protected-access

    assert env.render_frame() is expected
    session.render_frame.assert_called_once_with()


def test_reference_map_matches_simulator_edges():
    """Check the displayed road against world geometry, not the planner model."""
    pytest.importorskip("highway_env")
    from highway_env.envs.racetrack_env import RacetrackEnv
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_visualization.racetrack_reference_track import (
        reference_track_lanes,
    )

    world = RacetrackEnv()
    try:
        lanes = [
            lane
            for ends in world.road.network.graph.values()
            for parts in ends.values()
            for lane in parts
        ]
        rendered = reference_track_lanes()
        assert len(lanes) == len(rendered) == 18
        for lane, painted in zip(lanes, rendered):
            for side, edge in enumerate(painted["lines"]):
                distances = np.linspace(0, lane.length, len(edge["points"]))
                expected = np.array(
                    [lane.position(s, (side - 0.5) * lane.width_at(s)) for s in distances]
                )
                np.testing.assert_allclose(edge["points"], expected, atol=1e-12)
                assert edge["type"] == lane.line_types[side]
            np.testing.assert_array_equal(
                painted["polygon"],
                np.vstack([painted["lines"][0]["points"], painted["lines"][1]["points"][::-1]]),
            )
    finally:
        world.close()
