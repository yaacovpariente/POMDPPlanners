# SPDX-License-Identifier: MIT

"""Tests for the racetrack model that reads the road out of its observations.

Two kinds of test live here and they are not interchangeable. The pure-NumPy ones build a
corridor with :meth:`ObservedTrackModel._render_on_road_layer` and check the estimator
recovers it — useful, but self-consistent by construction, so they cannot tell you the sign
convention is right. The sign and the rough magnitude are pinned against a *live* circuit
and its :class:`TrackGeometry` profile instead, which is the only comparison that could
catch an across-track axis pointing the other way.
"""

from typing import Any, Dict

import numpy as np
import pytest

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
    PRESENCE_LAYER,
    ObservationMode,
)

COAST_STRAIGHT = 13

# The bends on the shipped racetrack run from 1/30 to 1/15 per metre, so a synthetic
# corridor at 0.05 is a representative one.
_SYNTHETIC_CURVATURE = 0.05


def _model(**overrides: Any) -> ObservedTrackModel:
    settings: Dict[str, Any] = {"discount_factor": 0.95, "process_noise_std": 0.0}
    settings.update(overrides)
    return ObservedTrackModel(**settings)


def _cruising_state(model: ObservedTrackModel, lateral: float = 0.0) -> np.ndarray:
    state = np.zeros(model.state_width, dtype=float)
    state[EGO_SPEED] = 10.0
    state[EGO_LAT] = lateral
    return state


def _corridor_observation(curvature: float, lateral: float = 0.0) -> Dict[str, np.ndarray]:
    """A raw occupancy grid whose on-road layer is one lane of the given curvature."""
    # pylint: disable=protected-access
    renderer = _model()
    renderer._curvature_estimate = curvature
    state = _cruising_state(renderer, lateral=lateral)
    grid = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
    grid[PRESENCE_LAYER, 6, 6] = 1.0
    grid[ON_ROAD_LAYER] = renderer._render_on_road_layer(state)
    return {"occupancy": grid}


def test_curvature_is_zero_before_any_observation_has_been_encoded():
    """Test the estimate a fresh model starts from.

    Purpose: A model that has not yet seen the road has to assume something, and a straight
        is the only defensible assumption. Leaving the attribute unset, or seeded from a
        previous episode, would make the first decision of every episode depend on history
        the model does not have

    Given: A freshly constructed model
    When: Its curvature estimate is read and a state is propagated
    Then: The estimate is exactly zero and the rollout stays on the centreline

    Test type: unit
    """
    model = _model()
    state = _cruising_state(model)

    for _ in range(10):
        state = model.sample_next_state(state, COAST_STRAIGHT)

    assert model.curvature_estimate == 0.0
    assert float(state[EGO_LAT]) == pytest.approx(0.0, abs=1e-12)
    assert float(state[EGO_ARCLENGTH_M]) == pytest.approx(20.0, abs=1e-9)


def test_the_mdp_arm_is_refused_rather_than_silently_driving_straight():
    """Test that a model with nothing to read is rejected at construction.

    Purpose: The MDP observation is a table of vehicle kinematics with no road in it. A
        model built against it would hold curvature at zero for the whole episode and drive
        into the first corner, and the failure would look like a planning bug rather than a
        configuration one

    Given: A request for this model in MDP observation mode
    When: It is constructed
    Then: ValueError names the missing road and points at the known-track model

    Test type: configuration
    """
    with pytest.raises(ValueError, match="on-road layer"):
        _model(observation_mode=ObservationMode.MDP)

    with pytest.raises(ValueError, match="curvature_window_m must be positive"):
        _model(curvature_window_m=0.0)


def test_the_estimate_recovers_a_synthetic_corridors_sign_and_ordering():
    """Test the estimator against corridors of known curvature.

    Purpose: Fixes the estimator's internal consistency — that fitting the corridor centres
        and reading 2a off the quadratic recovers the curvature that drew them, with the
        sign preserved rather than flipped by the across-track axis, and with a bigger bend
        reading as a bigger number

    Given: Occupancy grids whose on-road layer is a single-lane corridor drawn at 0, 0.033,
        0.05 and 0.067 1/m, in both signs
    When: Each is encoded
    Then: The sign is preserved, the magnitudes come out in the same order as the drawn
        curvatures, and a straight corridor estimates below 0.005 1/m

    Note: The scale is deliberately *not* asserted here. A single lane line quantised into
        3 m cells loses roughly a third of the amplitude, and a round trip through this
        model's own renderer would only measure that quantisation. Scale is pinned against
        the live circuit instead, where the corridor is two lane lines and the row mean
        lands between cells

    Test type: unit
    """
    model = _model()
    magnitudes = []
    for curvature in (0.0, 0.033, _SYNTHETIC_CURVATURE, 0.067):
        for sign in (1.0, -1.0):
            model.encode_observation(_corridor_observation(sign * curvature)["occupancy"])
            if curvature > 0.0:
                assert np.sign(model.curvature_estimate) == sign
        magnitudes.append(abs(model.curvature_estimate))

    assert magnitudes[0] < 5e-3
    assert all(later > earlier for earlier, later in zip(magnitudes, magnitudes[1:]))


def test_encoding_an_observation_changes_the_rollout_it_produces():
    """Test that the estimate actually reaches the transition, not just an attribute.

    Purpose: The cache is only worth anything if ``_curvature_for`` hands it to the Frenet
        integration. A model that stored the estimate and then propagated against zero
        would pass an estimator test and still be unable to corner

    Given: One model propagated before and after being shown a curving corridor
    When: Ten coasting decisions are propagated in each case from the same start state
    Then: The pre-observation rollout stays on the centreline and the post-observation one
        does not, and their lane-relative angles differ by more than 0.1 rad

    Test type: unit
    """
    model = _model()
    start = _cruising_state(model)

    blind = start.copy()
    for _ in range(10):
        blind = model.sample_next_state(blind, COAST_STRAIGHT)

    model.encode_observation(_corridor_observation(-_SYNTHETIC_CURVATURE)["occupancy"])
    seeing = start.copy()
    for _ in range(10):
        seeing = model.sample_next_state(seeing, COAST_STRAIGHT)

    assert float(blind[EGO_LAT]) == pytest.approx(0.0, abs=1e-12)
    assert float(blind[EGO_ANG]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(seeing[EGO_LAT])) > 1.0
    # Checked as well as the offset: the estimate enters the dynamics through the lane-angle
    # rate, and the offset only follows from it. A model that moved the ego sideways without
    # turning the lane frame under it would satisfy the offset assertion alone.
    assert abs(float(seeing[EGO_ANG]) - float(blind[EGO_ANG])) > 0.1


def test_the_rendered_on_road_layer_moves_with_the_particles_lane_offset():
    """Test that the predicted on-road layer is a function of the particle, not just the step.

    Purpose: This is what makes the layer worth scoring at all. The curvature estimate is
        shared by every particle in a step, so if the rendered corridor depended on nothing
        else the likelihood term would be a constant and would vanish at normalisation. It
        depends on the particle's own lane offset, which is exactly the quantity the layer
        can measure

    Given: A model shown a straight corridor, and two states 3 m apart laterally
    When: The on-road layer is rendered for each
    Then: Neither layer is all ones, the two differ, and the marked corridor sits on
        opposite sides of the ego's centre column — a particle offset one way sees the
        centreline offset the other

    Test type: unit
    """
    # pylint: disable=protected-access
    model = _model()
    left = model._render_on_road_layer(_cruising_state(model, lateral=-3.0))
    right = model._render_on_road_layer(_cruising_state(model, lateral=3.0))

    assert not np.all(left == 1.0)
    assert not np.array_equal(left, right)
    assert float(np.nonzero(left[6])[0].mean()) > 6.0
    assert float(np.nonzero(right[6])[0].mean()) < 6.0


def test_the_on_road_likelihood_prefers_the_particle_that_is_where_the_road_says():
    """Test that the on-road layer discriminates between particles.

    Purpose: Model B is required to make this layer informative rather than decorative. If
        two particles at different lane offsets scored identically, the extra 144 cells of
        arithmetic would be buying nothing

    Given: An observation rendered for an ego 2 m off the centreline, and three particles at
        -2, 0 and +2 m
    When: observation_log_probability scores each of them against it
    Then: All three scores are finite and the matching particle scores strictly highest

    Test type: unit
    """
    model = _model()
    observation = _corridor_observation(0.0, lateral=2.0)

    scores = np.array(
        [
            model.observation_log_probability(
                _cruising_state(model, lateral=offset), COAST_STRAIGHT, [observation]
            )[0]
            for offset in (-2.0, 0.0, 2.0)
        ]
    )

    assert np.all(np.isfinite(scores))
    assert int(np.argmax(scores)) == 2


def test_sampled_observations_carry_a_corridor_the_model_can_read_back():
    """Test that the model's own sampled observations are ones it could have received.

    Purpose: A planner that samples observations inside a rollout feeds them back through
        the same encoder. If the sampler wrote an all-ones on-road layer, the model would
        read its own sample as a perfectly straight road on the approach to every corner

    Given: A model holding a curving estimate, and a state on the centreline
    When: An observation is sampled and then encoded by a second, fresh model
    Then: The sampled on-road layer is not all ones, and the fresh model recovers a
        curvature of the same sign

    Test type: unit
    """
    model = _model()
    model.encode_observation(_corridor_observation(-_SYNTHETIC_CURVATURE)["occupancy"])
    state = _cruising_state(model)

    sampled = model.sample_observation(state, COAST_STRAIGHT)
    reader = _model()
    reader.encode_observation(sampled["occupancy"])

    assert not np.all(np.asarray(sampled["occupancy"])[ON_ROAD_LAYER] == 1.0)
    assert reader.curvature_estimate < 0.0


def _sweep_the_live_lap(monkeypatch) -> tuple:
    """Estimate the curvature all the way round the real circuit; return it and the truth."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")
    # Imported here so this module still imports where highway-env is absent.
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (  # pylint: disable=import-outside-toplevel
        RacetrackPOMDP,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (  # pylint: disable=import-outside-toplevel
        geometry_from_world,
    )

    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    world.initial_state_dist().sample()
    geometry, lane_index = geometry_from_world(world)
    # pylint: disable=protected-access  # The session is the world's only backend handle.
    backend = world._get_session()._env.unwrapped
    vehicle, network = backend.vehicle, backend.road.network

    lap_offsets, running, current = {}, 0.0, lane_index
    while current not in lap_offsets:
        lane = network.get_lane(current)
        lap_offsets[current] = running
        running += float(lane.length)
        current = network.next_lane(current, position=lane.position(lane.length, 0))

    model = ObservedTrackModel(discount_factor=0.95, process_noise_std=0.0)
    truths, estimates = [], []
    for arclength in range(int(geometry.total_length_m)):
        for index, offset in lap_offsets.items():
            lane = network.get_lane(index)
            if offset <= arclength < offset + lane.length:
                vehicle.position = lane.position(arclength - offset, 0.0)
                vehicle.heading = lane.heading_at(arclength - offset)
                vehicle.lane_index, vehicle.lane = index, lane
                break
        model.encode_observation(np.asarray(backend.observation_type.observe()))
        truths.append(float(geometry.curvature_at(float(arclength))))
        estimates.append(model.curvature_estimate)
    return np.asarray(truths), np.asarray(estimates)


def test_the_live_circuits_curvature_is_recovered_with_the_right_sign_and_scale(monkeypatch):
    """Test the estimator against the real track's own curvature profile.

    Purpose: The across-track axis of the occupancy grid could point either way, and every
        synthetic test above would pass with the sign flipped because it draws the corridor
        with the same convention it reads it back with. Comparing against a profile walked
        out of highway-env's lane graph is the only check that closes that loop

    Given: The ego walked round every metre of ``racetrack-v0``'s lap, on the centreline,
        with the on-road layer read at each point
    When: Each reading is encoded and compared with TrackGeometry's own curvature there
    Then: The estimate agrees in sign on at least 90% of the curved metres, the mean
        absolute error is under 0.015 1/m against bends running from 0.033 to 0.067, the
        regression of estimate on truth has a clearly positive slope, and the straights
        average under 0.015 1/m of spurious curvature

    Note:
        This walks the ego along the centreline, which characterises the estimator but
        flatters it. Measured under a lane-keeper actually driving the lap, the regression
        slope falls from 0.77 to 0.61 and bends come back at roughly three quarters of
        their true magnitude — the ego is off-centre and yawed, and the 12 m fit window
        spends much of every arc straddling a segment boundary. Do not read the thresholds
        here as a claim about what a planner sees

    Test type: integration
    """
    pytest.importorskip("highway_env")

    truths, estimates = _sweep_the_live_lap(monkeypatch)
    curved = truths != 0.0

    assert np.mean(np.sign(estimates[curved]) == np.sign(truths[curved])) > 0.9
    assert np.mean(np.abs(estimates - truths)) < 0.015
    assert float(np.polyfit(truths, estimates, 1)[0]) > 0.6
    assert np.mean(np.abs(estimates[~curved])) < 0.015


def test_the_two_models_disagree_on_the_approach_to_a_corner(monkeypatch):
    """Test that the observed-track model sees a bend before the map model reaches it.

    Purpose: If the two produce the same rollout, the observation is not being read: the
        window looks 18 m ahead while the arclength lookup only reports the road the ego is
        already on, so on the approach to a bend they *must* differ. Identical rollouts here
        would be the signature of a dead estimator

    Given: The live world driven forward until the ego is a few metres short of the first
        bend, its map curvature still exactly zero
    When: The observation there is encoded by the observed-track model, and both models roll
        the same state forward ten coasting decisions
    Then: The map model reports zero curvature under the ego while the observed model
        already reports a non-zero one; three steps later the map model is still exactly on
        the centreline and the observed one has left it; and over ten steps the two
        predictions disagree by more than 0.5 m at their widest

    Test type: integration
    """
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")
    # Imported here so this module still imports where highway-env is absent.
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import (  # pylint: disable=import-outside-toplevel
        KnownTrackModel,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (  # pylint: disable=import-outside-toplevel
        RacetrackPOMDP,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (  # pylint: disable=import-outside-toplevel
        geometry_from_world,
    )

    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    truth = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    geometry, _ = geometry_from_world(world)
    # The first bend begins at 58 m; coast until the ego is within 6 m of it.
    while float(truth[EGO_ARCLENGTH_M]) < 52.0:
        truth = np.asarray(world.sample_next_state(truth, COAST_STRAIGHT), dtype=float)

    observed = ObservedTrackModel(discount_factor=0.95, process_noise_std=0.0)
    observed.encode_observation(world.sample_observation(truth, COAST_STRAIGHT))
    mapped = KnownTrackModel(discount_factor=0.95, process_noise_std=0.0, track_geometry=geometry)

    assert float(geometry.curvature_at(float(truth[EGO_ARCLENGTH_M]))) == 0.0
    assert abs(observed.curvature_estimate) > 0.01

    from_map, from_sight = truth.copy(), truth.copy()
    disagreements = []
    for step in range(10):
        from_map = mapped.sample_next_state(from_map, COAST_STRAIGHT)
        from_sight = observed.sample_next_state(from_sight, COAST_STRAIGHT)
        disagreements.append(abs(float(from_map[EGO_LAT]) - float(from_sight[EGO_LAT])))
        if step == 2:
            assert float(from_map[EGO_LAT]) == pytest.approx(0.0, abs=1e-9)
            assert abs(float(from_sight[EGO_LAT])) > 0.2

    assert max(disagreements) > 0.5
