# SPDX-License-Identifier: MIT

"""Tests for the abstract planner-side racetrack generative model.

The class under test cannot be instantiated — it is abstract precisely because it does not
know where the road bends — so everything here is exercised through the thinnest concrete
subclass available, a
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model.KnownTrackModel`
over a one-segment map. What is being checked is the *base* behaviour: the bicycle
integration, the Frenet coupling, the densities, the rasteriser and the arclength slot.
Behaviour specific to either subclass lives in its own file.

Every test but the last two is pure NumPy and never touches highway-env: the model exists so
a planner can run without the simulator, and a test suite that needed the simulator to check
the model would not be checking that. The last two are the exception on purpose — they are
the only things that can show the reproduced bicycle really tracks the one the world
integrates.
"""

from typing import Any, Dict

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import KnownTrackModel
from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_VY,
    AGENT_REL_X,
    AGENT_REL_Y,
    AGENT_SLOT_WIDTH,
    DEFAULT_ACTION_PRESETS,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_HEADING,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    EGO_X,
    EGO_Y,
    GRID_CELLS,
    MAX_ACCELERATION_MPS2,
    MAX_STEERING_RAD,
    ON_ROAD_LAYER,
    PRESENCE_LAYER,
    ObservationMode,
    racetrack_reward,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry

# Indices into DEFAULT_ACTION_PRESETS, named so the tests read as driving commands. The
# table is the 3 accelerations crossed with the 9 steering angles, acceleration major.
COAST_STRAIGHT = 13
ACCELERATE_STRAIGHT = 4
COAST_FULL_LEFT = 17

DT = 0.2

# A map with one segment and no bends, so the base class's dynamics are exercised without
# a subclass's curvature source doing anything interesting.
_STRAIGHT_LAP_M = 1000.0


def _flat_geometry(curvature: float = 0.0) -> TrackGeometry:
    """A single-segment lap of constant curvature."""
    return TrackGeometry(
        segment_starts=np.array([0.0]),
        segment_curvatures=np.array([curvature]),
        total_length_m=_STRAIGHT_LAP_M,
    )


def _model(**overrides: Any) -> KnownTrackModel:
    """Build a model with deterministic dynamics on a straight map, unless asked otherwise."""
    settings: Dict[str, Any] = {
        "discount_factor": 0.95,
        "process_noise_std": 0.0,
        "track_geometry": _flat_geometry(),
    }
    settings.update(overrides)
    return KnownTrackModel(**settings)


def _ego_state(
    model: RacetrackModelPOMDP,
    *,
    speed: float = 10.0,
    heading: float = 0.0,
    lateral: float = 0.0,
    angle: float = 0.0,
    arclength: float = 0.0,
) -> np.ndarray:
    """A state with the given ego block and every agent slot empty."""
    state = np.zeros(model.state_width, dtype=float)
    state[EGO_SPEED] = speed
    state[EGO_HEADING] = heading
    state[EGO_LAT] = lateral
    state[EGO_ANG] = angle
    state[EGO_ARCLENGTH_M] = arclength
    return state


def _place_agent(
    state: np.ndarray,
    slot: int,
    rel_x: float,
    rel_y: float,
    rel_vx: float = 0.0,
    rel_vy: float = 0.0,
) -> np.ndarray:
    """Return a copy of ``state`` with one agent slot filled in the ego body frame."""
    filled = state.copy()
    base = EGO_STATE_WIDTH + slot * AGENT_SLOT_WIDTH
    filled[base + AGENT_PRESENT] = 1.0
    filled[base + AGENT_REL_X] = rel_x
    filled[base + AGENT_REL_Y] = rel_y
    filled[base + AGENT_REL_VX] = rel_vx
    filled[base + AGENT_REL_VY] = rel_vy
    return filled


def _occupancy(grid: np.ndarray) -> dict:
    """Wrap a presence grid in the two-layer occupancy observation the world emits."""
    occupancy = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
    occupancy[PRESENCE_LAYER] = grid
    occupancy[ON_ROAD_LAYER] = 1.0
    return {"occupancy": occupancy}


def test_the_base_model_cannot_be_instantiated_without_a_curvature_source():
    """Test that the base class is abstract on the one method that matters.

    Purpose: The base class deliberately does not know the track. Leaving it concrete with
        a zero-curvature default would let a caller build a model that silently drives
        straight through every corner, which is the exact bug the split exists to prevent

    Given: The abstract RacetrackModelPOMDP class
    When: It is constructed directly with otherwise valid arguments
    Then: TypeError names the unimplemented curvature hook

    Test type: unit
    """
    # pylint: disable=abstract-class-instantiated
    with pytest.raises(TypeError, match="_curvature_for"):
        RacetrackModelPOMDP(discount_factor=0.95)  # type: ignore[abstract]


def test_actions_index_the_presets_and_hash_action_round_trips():
    """Test that the discrete action set enumerates the shared control presets.

    Purpose: Validates that the planner's action vocabulary is exactly the indices into
        the preset table the world also indexes, and that hashing an action is reversible

    Given: A model built with the default 27-entry preset grid
    When: get_actions() is enumerated and each action is hashed
    Then: The actions are 0..26 in order and each hash equals the action itself

    Test type: unit
    """
    model = _model()

    actions = model.get_actions()

    assert actions == list(range(len(DEFAULT_ACTION_PRESETS)))
    assert [model.hash_action(action) for action in actions] == actions


def test_coasting_straight_advances_x_by_speed_times_dt():
    """Test that a coasting step moves the ego exactly one dt of travel down the track.

    Purpose: Validates the closed form of the bicycle integration in its simplest case,
        where the substeps must sum to exactly speed * dt with no drift in speed or offset

    Given: A model at 10 m/s, heading 0, on the centreline of a straight lane
    When: The coast-straight preset is applied for one 0.2 s decision
    Then: x advances by exactly 2.0 m and speed, y, and lateral offset are unchanged

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)

    next_state = model.sample_next_state(state, COAST_STRAIGHT)

    assert next_state[EGO_X] == pytest.approx(2.0, abs=1e-12)
    assert next_state[EGO_Y] == pytest.approx(0.0, abs=1e-12)
    assert next_state[EGO_SPEED] == pytest.approx(10.0, abs=1e-12)
    assert next_state[EGO_LAT] == pytest.approx(0.0, abs=1e-12)


def test_full_lock_steering_turns_heading_by_the_analytic_bicycle_rate():
    """Test that a full-lock steering step turns the heading by the bicycle model's rate.

    Purpose: Validates that the slip angle and the LENGTH/2 wheelbase term are the ones
        highway-env uses, since a wrong wheelbase is invisible until the model is compared
        against the simulator

    Given: A model at 10 m/s with the steering command saturated at +1
    When: One 0.2 s decision is propagated
    Then: The heading equals speed * sin(arctan(tan(pi/4)/2)) / 2.5 * dt, computed here
        from the same constants rather than hardcoded

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)
    slip = np.arctan(0.5 * np.tan(MAX_STEERING_RAD))
    expected = 10.0 * np.sin(slip) / (model.vehicle_length / 2.0) * DT

    next_state = model.sample_next_state(state, COAST_FULL_LEFT)

    assert next_state[EGO_HEADING] == pytest.approx(float(expected), abs=1e-12)


def test_acceleration_preset_changes_speed_by_accel_times_dt():
    """Test that the throttle preset changes speed by the full commanded acceleration.

    Purpose: Validates that the normalised acceleration is scaled by the shared
        MAX_ACCELERATION_MPS2 and integrated over the whole decision, not one substep

    Given: A model at 10 m/s and the accelerate-straight preset (normalised +1)
    When: One 0.2 s decision is propagated
    Then: Speed rises by MAX_ACCELERATION_MPS2 * dt

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)

    next_state = model.sample_next_state(state, ACCELERATE_STRAIGHT)

    assert next_state[EGO_SPEED] == pytest.approx(10.0 + MAX_ACCELERATION_MPS2 * DT, abs=1e-12)


def test_non_zero_curvature_bends_the_lane_relative_angle():
    """Test that lane curvature rotates the lane frame under a straight-driving ego.

    Purpose: Validates the Frenet coupling: on an arc the lane direction turns even when
        the ego does not, so the heading-versus-lane angle must drift negative on a
        left-curving lane

    Given: Two identical coasting states, propagated by two models whose maps differ only
        in curvature, 0.0 and 0.05 1/m
    When: Each is propagated one decision with no steering
    Then: The straight lane leaves the angle at zero while the curved lane makes it
        negative, at roughly -curvature * speed * dt

    Test type: unit
    """
    curvature = 0.05
    flat = _model()
    curved_model = _model(track_geometry=_flat_geometry(curvature))

    straight = flat.sample_next_state(_ego_state(flat), COAST_STRAIGHT)
    curved = curved_model.sample_next_state(_ego_state(curved_model), COAST_STRAIGHT)

    assert straight[EGO_ANG] == pytest.approx(0.0, abs=1e-12)
    assert curved[EGO_ANG] < 0.0
    assert curved[EGO_ANG] == pytest.approx(-curvature * 10.0 * DT, rel=0.05)


def test_arclength_advances_monotonically_and_its_lookup_wraps_at_the_lap_seam():
    """Test the arclength slot the curvature hook is indexed by.

    Purpose: The slot replaced a frozen curvature value, and the whole point is that it
        moves: a rollout has to walk along the track for a lookup by arclength to ever see
        a different segment. It also has to survive crossing the finish line, which a
        planning horizon on the last segment routinely does

    Given: A one-segment 1000 m map, and a state seeded 4 m short of the lap seam
    When: Ten coasting decisions are propagated from each of arclength 0 and 996
    Then: The slot strictly increases on every step, advances by speed * dt per step on a
        straight, and the map's own lookup wraps a past-the-seam arclength back onto the
        first segment rather than clamping to the last

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)
    history = [float(state[EGO_ARCLENGTH_M])]
    for _ in range(10):
        state = model.sample_next_state(state, COAST_STRAIGHT)
        history.append(float(state[EGO_ARCLENGTH_M]))

    assert all(later > earlier for earlier, later in zip(history, history[1:]))
    assert history[-1] == pytest.approx(10 * 10.0 * DT, abs=1e-9)

    past_seam = _ego_state(model, arclength=_STRAIGHT_LAP_M - 4.0)
    for _ in range(10):
        past_seam = model.sample_next_state(past_seam, COAST_STRAIGHT)
    assert past_seam[EGO_ARCLENGTH_M] > _STRAIGHT_LAP_M
    assert float(model.track_geometry.curvature_at(past_seam[EGO_ARCLENGTH_M])) == 0.0


def test_sample_next_state_batch_matches_a_seeded_loop_over_sample_next_state():
    """Test that the vectorised particle path draws the same states as the scalar path.

    Purpose: Validates that the loop-free batch propagation used by the belief is the same
        kernel as the single-particle one, noise included, so the filter and the planner
        cannot silently disagree

    Given: A noisy model and five distinct particles
    When: The batch is propagated once and the same particles are propagated one at a time
        from the same random seed
    Then: Every entry of the two results matches exactly

    Test type: unit
    """
    model = _model(process_noise_std=0.1)
    particles = np.stack(
        [_ego_state(model, speed=8.0 + index, lateral=0.1 * index) for index in range(5)]
    )

    np.random.seed(7)
    batched = model.sample_next_state_batch(particles, COAST_FULL_LEFT)
    np.random.seed(7)
    looped = np.stack(
        [model.sample_next_state(particle, COAST_FULL_LEFT) for particle in particles]
    )

    assert batched.shape == (5, model.state_width)
    np.testing.assert_array_equal(batched, looped)


def test_process_noise_leaves_the_arclength_slot_alone():
    """Test that the transition's noise stops short of the arclength slot.

    Purpose: Arclength is not an independent coordinate but the integral of the ego's own
        motion, and a known-track model indexes its curvature profile with it. Jittering it
        would scatter particles onto different parts of the circuit for no modelling reason

    Given: A noisy model and one particle propagated many times from the same state
    When: The successors' arclength slots are compared
    Then: Every draw carries exactly the same arclength, while the ego position does vary

    Test type: unit
    """
    model = _model(process_noise_std=0.1)
    state = _ego_state(model, arclength=25.0)

    draws = model.sample_next_state(state, COAST_STRAIGHT, n_samples=32)

    assert len(np.unique(draws[:, EGO_ARCLENGTH_M])) == 1
    assert draws[0, EGO_ARCLENGTH_M] == pytest.approx(25.0 + 10.0 * DT, abs=1e-9)
    assert len(np.unique(draws[:, EGO_X])) > 1


def test_transition_log_probability_is_finite_and_peaks_at_the_deterministic_successor():
    """Test that the transition density is a proper Gaussian centred on the propagation.

    Purpose: Validates the shape contract the belief relies on and that the mode of the
        density is the noise-free successor, which is what makes the weights meaningful

    Given: A noisy model, a coasting state, and three candidate successors — the exact
        deterministic propagation and two increasingly displaced copies of it
    When: transition_log_probability scores all three
    Then: The result has shape (3,), is finite, and decreases with displacement

    Test type: unit
    """
    model = _model(process_noise_std=0.1)
    state = _ego_state(model)
    exact = _model().sample_next_state(state, COAST_STRAIGHT)
    near, far = exact.copy(), exact.copy()
    near[EGO_X] += 0.05
    far[EGO_X] += 0.5

    log_probs = model.transition_log_probability(state, COAST_STRAIGHT, [exact, near, far])

    assert log_probs.shape == (3,)
    assert np.all(np.isfinite(log_probs))
    assert log_probs[0] > log_probs[1] > log_probs[2]


def test_transition_log_probability_rejects_a_noise_free_model():
    """Test that scoring a point-mass transition is refused rather than returning NaN.

    Purpose: A model built without process noise has a degenerate transition; dividing by
        its zero variance would hand the belief a silent NaN weight, which is far harder to
        diagnose than an exception at the call site

    Given: A model with process_noise_std left at zero
    When: transition_log_probability is called
    Then: A ValueError naming the point mass is raised

    Test type: unit
    """
    model = _model()
    state = _ego_state(model)

    with pytest.raises(ValueError, match="point mass"):
        model.transition_log_probability(state, COAST_STRAIGHT, [state])


def test_reward_matches_the_shared_racetrack_reward_on_the_resulting_state():
    """Test that the model scores a transition with the world's own reward function.

    Purpose: The world and the planner call one reward function so the planner cannot be
        optimising a different objective; that only holds if the model feeds it the same
        arguments the world does — the *resulting* lateral offset and the applied command

    Given: A coasting state and its deterministic successor
    When: reward is called with the successor and, separately, with the successor left out
    Then: Both equal racetrack_reward evaluated on the successor's lateral offset with this
        model's own weights, and the off-road case scores exactly zero

    Test type: unit
    """
    model = _model()
    state = _ego_state(model, lateral=0.4)
    successor = model.sample_next_state(state, COAST_FULL_LEFT)
    expected = racetrack_reward(
        float(successor[EGO_LAT]),
        model.action_presets[COAST_FULL_LEFT],
        False,
        True,
        collision_reward=model.collision_reward,
        lane_centering_cost=model.lane_centering_cost,
        lane_centering_reward=model.lane_centering_reward,
        action_reward=model.action_reward,
    )

    assert model.reward(state, COAST_FULL_LEFT, successor) == pytest.approx(expected, abs=1e-12)
    assert model.reward(state, COAST_FULL_LEFT) == pytest.approx(expected, abs=1e-12)

    off_road = _ego_state(model, lateral=model.lane_half_width + 1.0)
    assert model.reward(off_road, COAST_STRAIGHT, off_road) == pytest.approx(0.0, abs=1e-12)


def test_is_terminal_fires_off_road_and_on_a_near_agent_but_not_in_the_clear():
    """Test the two ways the model predicts an episode ending.

    Purpose: Validates that the model can foresee both failure modes the world reports —
        leaving the lane and hitting a vehicle — since a planner blind to either has no
        reason to avoid it

    Given: Three states: on the centreline with no agents, past the lane half-width, and
        centred with an agent 2 m ahead
    When: is_terminal is queried on each
    Then: Only the clear state is non-terminal

    Test type: unit
    """
    model = _model()
    clear = _ego_state(model)
    off_road = _ego_state(model, lateral=model.lane_half_width + 0.1)
    collision = _place_agent(clear, slot=0, rel_x=2.0, rel_y=0.0)

    assert model.is_terminal(clear) is False
    assert model.is_terminal(off_road) is True
    assert model.is_terminal(collision) is True


def test_presence_grid_marks_the_ego_centre_and_places_an_agent_along_then_across():
    """Test the occupancy rasteriser's cell indexing, including the axis order.

    Purpose: Pins the axis convention verified against highway-env 1.12.1 — axis 0 is
        along-track and axis 1 across-track — because a transposed grid scores every
        particle plausibly and is therefore invisible without a test like this one

    Given: A state with a single agent 9 m ahead and 3 m to the right of the ego
    When: The presence grid is rasterised
    Then: The ego occupies the centre cell (6, 6), the agent occupies exactly cell (9, 5),
        and no other cell is marked

    Test type: unit
    """
    # pylint: disable=protected-access
    model = _model()
    state = _place_agent(_ego_state(model), slot=0, rel_x=9.0, rel_y=-3.0)

    grid = model._render_presence_grid(state)

    assert grid.shape == (GRID_CELLS, GRID_CELLS)
    assert bool(grid[6, 6]) is True
    assert bool(grid[9, 5]) is True
    assert int(np.count_nonzero(grid)) == 2


def test_pomdp_observation_log_probability_prefers_the_matching_grid():
    """Test that the occupancy density discriminates between grids.

    Purpose: Validates the Bernoulli likelihood the belief weights particles with: it must
        return one finite score per observation and rank a grid that agrees with the state
        above one that does not

    Given: A model in POMDP mode, a state with an agent ahead, and two observations — the
        grid that state would produce and the same grid with the agent cell moved
    When: observation_log_probability scores both
    Then: The result has shape (2,), is finite, and the matching grid scores strictly higher

    Test type: unit
    """
    # pylint: disable=protected-access
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(_ego_state(model), slot=0, rel_x=9.0, rel_y=-3.0)
    matching = model._render_presence_grid(state)
    mismatched = matching.copy()
    mismatched[9, 5] = False
    mismatched[3, 8] = True

    log_probs = model.observation_log_probability(
        state, COAST_STRAIGHT, [_occupancy(matching), _occupancy(mismatched)]
    )

    assert log_probs.shape == (2,)
    assert np.all(np.isfinite(log_probs))
    assert log_probs[0] > log_probs[1]


def test_pomdp_sample_observation_returns_the_two_layer_grid_the_world_emits():
    """Test the shape and layer count of a sampled occupancy observation.

    Purpose: The belief and the vectorized model both reshape this array by position, so a
        model that emitted a single layer, or the layers transposed, would be caught only
        far downstream

    Given: A POMDP-mode model and a state with one agent ahead
    When: Three observations are drawn
    Then: Each is a dict holding one (2, 12, 12) array whose presence layer is binary

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.POMDP)
    state = _place_agent(_ego_state(model), slot=0, rel_x=9.0, rel_y=-3.0)

    draws = model.sample_observation(state, COAST_STRAIGHT, n_samples=3)

    assert len(draws) == 3
    for draw in draws:
        grid = np.asarray(draw["occupancy"])
        assert grid.shape == (2, GRID_CELLS, GRID_CELLS)
        assert set(np.unique(grid[PRESENCE_LAYER])).issubset({0.0, 1.0})


def test_mdp_encode_observation_rotates_absolute_rows_into_the_ego_body_frame():
    """Test that the MDP encoder converts absolute vehicle rows to ego-relative ones.

    Purpose: Validates the single raw-observation seam for the MDP arm — the state's agent
        slots are body-frame, so an encoder that forgot the rotation would compare
        world-frame observations against body-frame means and score nonsense

    Given: A raw table whose ego drives due north at (10, 5) and a second vehicle 10 m due
        north of it at the same velocity, plus three empty rows
    When: encode_observation is applied
    Then: The ego row is passed through as [x, y, vx, vy], the other vehicle lands 10 m
        straight ahead in the body frame with zero relative velocity, and the empty rows
        stay zero

    Test type: unit
    """
    model = _model(observation_mode=ObservationMode.MDP)
    raw = np.zeros((model.max_tracked_agents + 1, 5), dtype=np.float32)
    raw[0] = [1.0, 10.0, 5.0, 0.0, 4.0]
    raw[1] = [1.0, 10.0, 15.0, 0.0, 4.0]

    encoded = model.encode_observation(raw)

    np.testing.assert_allclose(encoded["ego"], [10.0, 5.0, 0.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(encoded["agents"][0], [1.0, 10.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(encoded["agents"][1:], 0.0, atol=1e-12)


def _track_live_world(action: int, steps: int = 10) -> tuple:
    """Run one action through the live world and the model, returning both end states."""
    # Imported here so this module still imports where highway-env is absent.
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_pomdp import (  # pylint: disable=import-outside-toplevel
        RacetrackPOMDP,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (  # pylint: disable=import-outside-toplevel
        geometry_from_world,
    )

    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    truth = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    model = _model(track_geometry=geometry_from_world(world)[0])
    predicted = truth.copy()
    for _ in range(steps):
        predicted = model.sample_next_state(predicted, action)
        truth = np.asarray(world.sample_next_state(truth, action), dtype=float)
    return predicted, truth


def test_model_tracks_the_live_simulator_over_ten_coasting_steps(monkeypatch):
    """Test that the reproduced bicycle matches highway-env's own integration.

    Purpose: This is the only check that the model is a model *of this world*. Every closed
        form above could be self-consistently wrong; running the same action through both
        and comparing the ego pose is what rules that out

    Given: A live racetrack world seeded at 0 with no extra opponents, and a noise-free
        model started from the world's true state and carrying the world's own track map
    When: Ten coast-straight decisions are applied to both, the model propagating from its
        own prediction rather than being re-synchronised
    Then: The predicted (x, y, heading, lateral offset, arclength) stays within 1e-6 of the
        simulator's. The tolerance is tight on purpose: the ego update is reproduced term
        for term, so the only expected difference is float accumulation over thirty
        substeps — the measured error at this seed is in fact exactly zero, and anything
        near 1e-6 would already be a real disagreement rather than drift

    Note:
        The arclength comparison is only valid because ten steps from the spawn point do
        not reach the lap seam. The world derives arclength from lane offsets, so it wraps
        into ``[0, lap)``; the model integrates it and leaves it unbounded, exactly as
        ``curvature_at`` expects. Extend this drive past a full lap and the two must be
        compared modulo the lap length, the same caveat heading already carries

    Test type: integration
    """
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")

    predicted, truth = _track_live_world(COAST_STRAIGHT)

    assert abs(predicted[EGO_X] - truth[EGO_X]) < 1e-6
    assert abs(predicted[EGO_Y] - truth[EGO_Y]) < 1e-6
    assert abs(predicted[EGO_HEADING] - truth[EGO_HEADING]) < 1e-6
    assert abs(predicted[EGO_LAT] - truth[EGO_LAT]) < 1e-6
    assert abs(predicted[EGO_ARCLENGTH_M] - truth[EGO_ARCLENGTH_M]) < 1e-6


def test_model_tracks_the_live_simulator_through_a_full_lock_arc(monkeypatch):
    """Test that the model tracks the simulator when steering, not just when coasting.

    Purpose: Coasting straight leaves the slip angle at zero, so it cannot catch a wrong
        Frenet rate. Under full lock the slip angle is 0.46 rad and the lateral offset must
        be integrated along the *velocity* direction rather than the heading; getting that
        wrong costs about 2 m over this manoeuvre while everything else still looks right

    Given: The same seeded world and noise-free model
    When: Ten full-lock left decisions are applied to both
    Then: Position, heading and lateral offset all match to 1e-6. Heading is compared
        modulo 2*pi because the model wraps to [-pi, pi) and highway-env does not

    Test type: integration
    """
    pytest.importorskip("highway_env")
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")

    predicted, truth = _track_live_world(COAST_FULL_LEFT)

    heading_error = (predicted[EGO_HEADING] - truth[EGO_HEADING] + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(predicted[EGO_X] - truth[EGO_X]) < 1e-6
    assert abs(predicted[EGO_Y] - truth[EGO_Y]) < 1e-6
    assert abs(heading_error) < 1e-6
    assert abs(predicted[EGO_LAT] - truth[EGO_LAT]) < 1e-6
