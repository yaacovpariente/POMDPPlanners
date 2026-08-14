# SPDX-License-Identifier: MIT

"""Tests for the racetrack model that plans with a map of the circuit.

The base class's dynamics are covered in ``test_racetrack_model_pomdp``; what is left here
are the two things this subclass adds, and they are different in kind. ``_curvature_for``
says where the road bends *under* a particle, which is what makes a rollout corner.
``curvature_ahead`` says where it bends *in front of* one, which is what makes the
observation's curvature channel able to tell two particles apart at all — for a mapless model
that prediction comes out of the very reading being scored, so it is identical across the
belief and drops out at normalisation.

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
    EGO_STATE_WIDTH,
    STEERING_PRESETS,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry

COAST_STRAIGHT = 13
# The coasting row of the preset table; add a steering index to select a command.
COAST_ROW_BASE = 1 * len(STEERING_PRESETS)

# The lap the unit tests below are read against: straight until 20 m, then one arc all the
# way round to the finish line at 100 m. Both boundaries matter -- the first is where a
# rollout starts bending, the second is where a lookahead has to wrap.
STRAIGHT_ENDS_M = 20.0
LAP_LENGTH_M = 100.0
ARC_CURVATURE = -0.05


def _two_segment_geometry() -> TrackGeometry:
    """A 100 m lap: 20 m of straight, then a left-hand arc for the rest."""
    return TrackGeometry(
        segment_starts=np.array([0.0, STRAIGHT_ENDS_M]),
        segment_curvatures=np.array([0.0, ARC_CURVATURE]),
        total_length_m=LAP_LENGTH_M,
    )


def _model(**overrides: Any) -> KnownTrackModel:
    settings: Dict[str, Any] = {
        "discount_factor": 0.95,
        "process_noise_std": 0.0,
        "track_geometry": _two_segment_geometry(),
    }
    settings.update(overrides)
    return KnownTrackModel(**settings)


def _cruising_state(
    model: KnownTrackModel, arclength: float, lateral: float = 0.0, angle: float = 0.0
) -> np.ndarray:
    state = np.zeros(model.state_width, dtype=float)
    state[EGO_SPEED] = 10.0
    state[EGO_LAT] = lateral
    state[EGO_ANG] = angle
    state[EGO_ARCLENGTH_M] = arclength
    return state


def _ego_block(*arclengths: float) -> np.ndarray:
    """An ego block of one row per arclength, everything else left at zero."""
    ego = np.zeros((len(arclengths), EGO_STATE_WIDTH), dtype=float)
    ego[:, EGO_ARCLENGTH_M] = arclengths
    return ego


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

    curvature = model._curvature_for(_ego_block(5.0, 25.0, 95.0))

    assert curvature.shape == (3,)
    np.testing.assert_allclose(curvature, [0.0, ARC_CURVATURE, ARC_CURVATURE], atol=1e-12)


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

    on_straight = [lateral for arclength, lateral in offsets if arclength <= STRAIGHT_ENDS_M]
    in_bend = [lateral for arclength, lateral in offsets if arclength > 22.0]

    assert on_straight, "the rollout should spend at least one step on the straight"
    assert all(abs(lateral) < 1e-12 for lateral in on_straight)
    assert in_bend[-1] > 1.0
    assert float(state[EGO_ANG]) > 0.3


def test_curvature_ahead_reports_one_column_per_lookahead_distance():
    """Test the shape of the channel the map predicts, against the channel it is scored on.

    Purpose: The likelihood takes the residual pairwise against the observation's curvature
        samples, so a prediction of the wrong width either raises or -- worse, if it happened
        to broadcast -- compares the curvature at one distance against the curvature at
        another. The width has to follow the configured lookahead distances, not a constant

    Given: Two models over the same lap, one with the default three lookahead distances and
        one configured with two, each handed an ego block of four particles
    When: curvature_ahead is called on that block
    Then: The results are (4, 3) and (4, 2) respectively

    Test type: unit
    """
    ego = _ego_block(0.0, 5.0, 25.0, 95.0)

    default = _model()
    two_samples = _model(curvature_lookahead_m=(5.0, 40.0))

    assert default.curvature_ahead(ego).shape == (4, len(default.curvature_lookahead_m))
    assert default.curvature_ahead(ego).shape == (4, 3)
    assert two_samples.curvature_ahead(ego).shape == (4, 2)


def test_a_particle_on_the_straight_already_reads_the_arc_at_the_lookahead_that_reaches_it():
    """Test that the map's channel leads the particle's own position, exactly as a camera does.

    Purpose: This is the whole reason the map predicts the channel rather than holding one
        value across it. A car 6 m short of a bend is still on a straight, but its camera is
        already looking into the corner; a model that reported the curvature under the ego at
        every distance would predict a flat road there and be unable to score the reading that
        says otherwise

    Given: A lap that is straight until 20 m and curving after it, and a particle at 14 m --
        on the straight, with the first lookahead 10 m ahead of it landing at 24 m
    When: The curvature under the ego and the curvature ahead are both read
    Then: The curvature under the ego is zero while all three lookahead samples already report
        the arc, so the two disagree at the same particle

    Test type: unit
    """
    # pylint: disable=protected-access
    model = _model()
    ego = _ego_block(14.0)

    under = model._curvature_for(ego)
    ahead = model.curvature_ahead(ego)

    assert float(under[0]) == 0.0
    np.testing.assert_allclose(ahead[0], [ARC_CURVATURE] * 3, atol=1e-12)


def test_the_lookahead_wraps_around_the_lap_instead_of_running_off_the_end():
    """Test that a lookahead crossing the finish line reads the start of the lap.

    Purpose: The track is a closed loop and a rollout runs straight through the finish line,
        so a lookahead past the end of the profile has to come back round to the beginning.
        Clipping to the last segment instead would tell every particle approaching the line
        that the road it is about to drive onto is the arc it is already on -- and this lap
        begins with a straight

    Given: A particle 5 m from the end of a 100 m lap whose last 80 m is an arc and whose
        first 20 m is straight, with lookaheads at 10, 20 and 30 m
    When: curvature_ahead is called
    Then: The first two samples land at 5 m and 15 m into the next lap and report the
        straight's zero, the third lands at 25 m and reports the arc, and the whole row equals
        the map's own lookup at the wrapped distances

    Test type: unit
    """
    model = _model()
    geometry = model.track_geometry

    ahead = model.curvature_ahead(_ego_block(95.0))

    np.testing.assert_allclose(ahead[0], [0.0, 0.0, ARC_CURVATURE], atol=1e-12)
    np.testing.assert_allclose(ahead[0], geometry.curvature_at(np.array([5.0, 15.0, 25.0])))


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


def test_the_map_predicts_the_live_circuits_own_curvature_channel(monkeypatch):
    """Test the predicted channel against the reading the world actually emits.

    Purpose: Every unit test above reads the map against itself, with the same sign
        convention and the same lookahead distances on both sides. The world measures its
        curvature channel off its own geometry at its own arclength, so comparing the two is
        the only check that could catch the map being indexed at the wrong place or read with
        the wrong sign — and the channel is worthless to the filter if it is either

    Given: A live world seeded at 0 and driven 20 steps by a lane-keeping controller, and a
        noise-free model carrying that world's own track map
    When: The world's curvature-ahead reading at each step is compared with what the model
        predicts for the world's true state
    Then: Every sample agrees to within the camera's own noise plus the metre of arclength a
        step covers, i.e. well inside 0.01 1/m against arcs running from 0.033 to 0.067

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

    # The camera's noise comes off the global generator, and the threshold below is a bound
    # on a maximum over 60 noisy samples, so the seed is pinned rather than left to the run.
    np.random.seed(0)
    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    truth = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    geometry, _ = geometry_from_world(world)
    model = KnownTrackModel(discount_factor=0.95, process_noise_std=0.0, track_geometry=geometry)

    worst = 0.0
    for _ in range(20):
        action = _steer_toward_the_centreline(truth)
        truth = np.asarray(world.sample_next_state(truth, action), dtype=float)
        observed = np.asarray(world.sample_observation(truth, action).curvature_ahead, dtype=float)
        predicted = model.curvature_ahead(truth[None, :EGO_STATE_WIDTH])[0]
        worst = max(worst, float(np.max(np.abs(observed - predicted))))

    assert worst < 0.01
