# SPDX-License-Identifier: MIT

"""Tests for the racetrack model that plans with a map of the circuit.

The base class's dynamics are covered in ``test_racetrack_model_pomdp``; what is left here
is the one thing this subclass adds — curvature by arclength lookup — and the consequence
that makes it worth having: a rollout that reaches a bend starts behaving like a bend.

The last test is the load-bearing one. A model with a frozen curvature slot passes a
straight-line tracking test perfectly and still cannot corner, so tracking the simulator on
a straight proves nothing at all about the change this class exists to make.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import KnownTrackModel
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model import (
    ObservedTrackModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_LAT,
    EGO_SPEED,
    GRID_CELLS,
    ON_ROAD_LAYER,
    STEERING_PRESETS,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry

COAST_STRAIGHT = 13
# The coasting row of the preset table; add a steering index to select a command.
COAST_ROW_BASE = 1 * len(STEERING_PRESETS)


def _two_segment_geometry() -> TrackGeometry:
    """A 100 m lap: 20 m of straight, then a left-hand arc for the rest."""
    return TrackGeometry(
        segment_starts=np.array([0.0, 20.0]),
        segment_curvatures=np.array([0.0, -0.05]),
        total_length_m=100.0,
    )


def _model(**overrides: Any) -> KnownTrackModel:
    settings: Dict[str, Any] = {
        "discount_factor": 0.95,
        "process_noise_std": 0.0,
        "track_geometry": _two_segment_geometry(),
    }
    settings.update(overrides)
    return KnownTrackModel(**settings)


def _cruising_state(model: KnownTrackModel, arclength: float) -> np.ndarray:
    state = np.zeros(model.state_width, dtype=float)
    state[EGO_SPEED] = 10.0
    state[EGO_ARCLENGTH_M] = arclength
    return state


def test_curvature_is_read_from_the_map_at_each_particles_own_arclength():
    """Test that the curvature hook indexes the map per particle, not per model.

    Purpose: Particles in a belief sit at different points on the track, so a hook that
        returned one curvature for the whole batch would put every particle on the same
        piece of road and quietly destroy the spread the filter is maintaining

    Given: A two-segment map, straight until 20 m and curving after it, and an ego block of
        three rows at arclengths 5, 25 and 95 metres
    When: The curvature hook is called on that block
    Then: It returns the straight's zero for the first row and the arc's curvature for the
        other two, matching the map's own lookup exactly

    Test type: unit
    """
    # pylint: disable=protected-access
    model = _model()
    ego = np.zeros((3, 7), dtype=float)
    ego[:, EGO_ARCLENGTH_M] = [5.0, 25.0, 95.0]

    curvature = model._curvature_for(ego)

    assert curvature.shape == (3,)
    np.testing.assert_allclose(curvature, [0.0, -0.05, -0.05], atol=1e-12)


def test_a_rollout_starts_bending_when_it_reaches_the_bend():
    """Test that the curvature a rollout integrates changes underneath it.

    Purpose: This is the whole point of indexing by arclength. A model holding one frozen
        curvature for the horizon predicts that coasting straight keeps the ego on the
        centreline forever, which is exactly the prediction that lets a planner drive into
        a wall

    Given: A two-segment map with the bend at 20 m, and a coasting state 6 m short of it
    When: Twelve decisions of coast-straight are propagated
    Then: The lane-relative angle and lateral offset stay at zero while the ego is on the
        straight, and both grow away from it once it crosses into the arc. The signs follow
        the map's own convention: a negative curvature turns the lane away from a
        straight-driving ego, so its lane-relative angle and offset both go positive

    Test type: unit
    """
    model = _model()
    state = _cruising_state(model, arclength=14.0)
    offsets: List[Tuple[float, float]] = []
    for _ in range(12):
        state = model.sample_next_state(state, COAST_STRAIGHT)
        offsets.append((float(state[EGO_ARCLENGTH_M]), float(state[EGO_LAT])))

    on_straight = [lateral for arclength, lateral in offsets if arclength <= 20.0]
    in_bend = [lateral for arclength, lateral in offsets if arclength > 22.0]

    assert on_straight, "the rollout should spend at least one step on the straight"
    assert all(abs(lateral) < 1e-12 for lateral in on_straight)
    assert in_bend[-1] > 1.0
    assert float(state[EGO_ANG]) > 0.3


def test_the_on_road_layer_is_neither_rendered_nor_scored():
    """Test that this model leaves the observation's on-road layer alone.

    Purpose: With a known track the layer is a function of arclength, which carries no
        process noise, so every particle in a belief would predict the same road. A
        likelihood term identical across particles shifts all the log-weights alike and
        vanishes at normalisation; paying 144 cells of arithmetic per particle per step for
        it would be pure cost

    Given: A POMDP-mode model and two observations that differ *only* in the on-road layer
    When: Observations are sampled, and both candidates are scored
    Then: The sampled layer is all ones, and the two candidates score identically

    Test type: unit
    """
    model = _model()
    state = _cruising_state(model, arclength=5.0)

    drawn = np.asarray(model.sample_observation(state, COAST_STRAIGHT)["occupancy"])
    all_road = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
    all_road[ON_ROAD_LAYER] = 1.0
    no_road = all_road.copy()
    no_road[ON_ROAD_LAYER] = 0.0
    scores = model.observation_log_probability(
        state, COAST_STRAIGHT, [{"occupancy": all_road}, {"occupancy": no_road}]
    )

    np.testing.assert_allclose(drawn[ON_ROAD_LAYER], 1.0)
    assert scores[0] == pytest.approx(scores[1], abs=1e-12)


def _steer_toward_the_centreline(state: np.ndarray) -> int:
    """A crude lane-keeping controller, so a corner can be driven rather than crashed into."""
    target = -0.35 * float(state[EGO_LAT]) - 1.2 * float(state[EGO_ANG])
    return COAST_ROW_BASE + int(np.argmin([abs(preset - target) for preset in STEERING_PRESETS]))


def _drive_through_the_first_corner(model: RacetrackModelPOMDP, steps: int) -> float:
    """Largest lane-offset disagreement between ``model`` and the live world over ``steps``."""
    # Imported here so this module still imports where highway-env is absent.
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (  # pylint: disable=import-outside-toplevel
        RacetrackPOMDP,
    )

    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    truth = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    predicted = truth.copy()
    worst = 0.0
    for _ in range(steps):
        action = _steer_toward_the_centreline(truth)
        predicted = model.sample_next_state(predicted, action)
        truth = np.asarray(world.sample_next_state(truth, action), dtype=float)
        worst = max(worst, abs(float(predicted[EGO_LAT]) - float(truth[EGO_LAT])))
    return worst


def test_the_map_model_tracks_the_live_simulator_through_a_corner(monkeypatch):
    """Test that the predicted lane offset follows the simulator's around a bend.

    Purpose: A straight-line tracking test is passed perfectly by a model that cannot
        corner at all, so it certifies nothing. This drives the ego from its spawn on the
        first straight into the first bend of ``racetrack-v0`` and compares the whole way

    Given: A live world seeded at 0, a noise-free model carrying that world's own track map,
        and a crude lane-keeping controller driving both from the world's true state
    When: 35 decisions are applied, enough to spend roughly 20 of them inside the bend
    Then: The worst lane-offset disagreement stays under 0.5 m, and a model that holds
        curvature at zero — which is what the old frozen-curvature state slot amounted to —
        is at least twice as wrong over the same drive

    Test type: integration
    """
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")
    # Imported here so this module still imports where highway-env is absent.
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (  # pylint: disable=import-outside-toplevel
        RacetrackPOMDP,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (  # pylint: disable=import-outside-toplevel
        geometry_from_world,
    )

    probe = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    probe.initial_state_dist().sample()
    geometry, _ = geometry_from_world(probe)

    mapped = _drive_through_the_first_corner(
        KnownTrackModel(discount_factor=0.95, process_noise_std=0.0, track_geometry=geometry), 35
    )
    # ObservedTrackModel that is never shown an observation holds curvature at zero, which
    # is exactly the behaviour of the model this refactor replaced.
    flat = _drive_through_the_first_corner(
        ObservedTrackModel(discount_factor=0.95, process_noise_std=0.0), 35
    )

    assert mapped < 0.5
    assert flat > 2.0 * mapped
