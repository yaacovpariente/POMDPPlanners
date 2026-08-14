# SPDX-License-Identifier: MIT

"""Tests for the racetrack model that reads the road out of its observations.

The model has one job the base class does not do: cache the camera's nearest curvature
sample when an observation is encoded, and integrate every rollout against it until the next
one arrives. So the tests come in pairs — one that the number lands in the attribute, and one
that the attribute reaches the transition. A model that stored the estimate and propagated
against zero would pass the first half of every pair and still be unable to corner.

The unit tests hand the model readings built here, which fixes the units and the sign against
a number chosen rather than measured. The sign convention is closed against a *live* circuit
in the last test, where the camera reads a real bend and the map model's arclength lookup
says which way it goes.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pytest

from POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model import (
    ObservedTrackModel,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    DETECTION_SLOT_WIDTH,
    EGO_ANG,
    EGO_ARCLENGTH_M,
    EGO_LAT,
    EGO_SPEED,
    EGO_STATE_WIDTH,
    OBSERVED_EGO_POSE_WIDTH,
    ObservationMode,
)

COAST_STRAIGHT = 13

# The bends on the shipped racetrack run from 1/30 to 1/15 per metre, so a reading at 0.05
# is a representative corner rather than an extreme one.
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


def _reading(
    *curvature_ahead: float,
    speed: float = 10.0,
    lane_pose: Tuple[float, float] = (0.0, 0.0),
    detections: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One raw five-part reading: ego pose, speedometer, lane camera, curvature, detections.

    Built as the world's own tuple rather than as the encoded dictionary, because the tuple is
    the form ``encode_observation`` actually receives in an episode and the split into keys is
    part of what is under test. The ego pose and the detections are left at zero: this model
    reads neither, and a reading that omitted them would not be one the encoder accepts.
    """
    return (
        np.zeros(OBSERVED_EGO_POSE_WIDTH, dtype=np.float32),
        np.array([speed], dtype=np.float32),
        np.array(lane_pose, dtype=np.float32),
        np.array(curvature_ahead, dtype=np.float32),
        np.zeros((detections, DETECTION_SLOT_WIDTH), dtype=np.float32),
    )


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


def test_encoding_a_reading_caches_its_nearest_curvature_sample():
    """Test which of the camera's samples becomes the model's estimate.

    Purpose: The channel reports the road at several distances and only the nearest one is
        the road the next few steps are driven on. Averaging the samples, or taking the
        furthest, would have the model start turning for a corner it has not reached — and on
        the approach to a bend those samples disagree by the whole of the bend

    Given: A reading whose curvature channel is 0.04, 0.02 and 0.0 at the three lookahead
        distances
    When: It is encoded
    Then: The estimate is exactly the nearest sample, 0.04, and the encoder still returns all
        five sensor keys with the whole channel intact

    Test type: unit
    """
    model = _model()

    encoded = model.encode_observation(_reading(0.04, 0.02, 0.0))

    assert model.curvature_estimate == pytest.approx(0.04, abs=1e-7)
    assert set(encoded) == {"ego_pose", "ego_speed", "lane_pose", "curvature_ahead", "detections"}
    np.testing.assert_allclose(encoded["curvature_ahead"], [0.04, 0.02, 0.0], atol=1e-7)


def test_a_new_reading_replaces_the_estimate_rather_than_accumulating_with_it():
    """Test that the estimate is this step's reading and carries no memory of the last.

    Purpose: The road under the car changes as it drives, so an estimate that averaged in
        earlier readings would lag every corner entry and every corner exit, and the lag would
        depend on how long the episode had been running

    Given: A model shown a left-hand bend, then a right-hand one, then a straight
    When: Each reading is encoded in turn
    Then: The estimate is exactly the nearest sample of the most recent reading each time,
        including a return to exactly zero

    Test type: unit
    """
    model = _model()
    seen = []

    for curvature in (-_SYNTHETIC_CURVATURE, _SYNTHETIC_CURVATURE, 0.0):
        model.encode_observation(_reading(curvature, curvature, curvature))
        seen.append(model.curvature_estimate)

    assert seen == pytest.approx([-0.05, 0.05, 0.0], abs=1e-7)


def test_the_cached_estimate_is_the_curvature_every_particle_is_propagated_against():
    """Test that one estimate is handed to the whole batch, which is the known approximation.

    Purpose: The estimate is read once per decision and shared by every particle, so within a
        planning step the belief cannot disagree about where the road goes. That is a real
        approximation rather than a bookkeeping detail, and pinning it here is what keeps it a
        documented limit instead of a surprise: a mapless model has nothing to say about the
        road that it did not read off the observation it is scoring

    Given: A model shown a bend, and an ego block whose particles sit at three very different
        arclengths
    When: The curvature hook is called on that block
    Then: Every row is the same cached estimate, and it is the reading's nearest sample

    Test type: unit
    """
    # pylint: disable=protected-access
    model = _model()
    model.encode_observation(_reading(-_SYNTHETIC_CURVATURE, 0.0, 0.0))
    ego = np.zeros((3, EGO_STATE_WIDTH), dtype=float)
    ego[:, EGO_ARCLENGTH_M] = [0.0, 40.0, 300.0]

    curvature = model._curvature_for(ego)

    assert curvature.shape == (3,)
    np.testing.assert_allclose(curvature, [-_SYNTHETIC_CURVATURE] * 3, atol=1e-7)


def test_encoding_an_observation_changes_the_rollout_it_produces():
    """Test that the estimate actually reaches the transition, not just an attribute.

    Purpose: The cache is only worth anything if ``_curvature_for`` hands it to the Frenet
        integration. A model that stored the estimate and then propagated against zero would
        pass every test above and still be unable to corner

    Given: One model propagated before and after being shown a reading of a left-hand bend
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

    model.encode_observation(_reading(-_SYNTHETIC_CURVATURE, -_SYNTHETIC_CURVATURE, 0.0))
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


def test_a_sampled_observation_is_one_the_model_could_have_received():
    """Test that a self-drawn reading round-trips back through the encoder.

    Purpose: A planner that samples observations inside a rollout feeds them back through the
        same encoder. If the sampler wrote a curvature channel the encoder could not read —
        the wrong width, or a flat road on the approach to every corner — the model would be
        scoring and re-reading two different things

    Given: A model holding a left-hand estimate, and a state on the centreline
    When: An observation is sampled from it and encoded by a second, fresh model
    Then: The channel is the configured width and the fresh model recovers the same bend, to
        within the camera's own noise

    Test type: unit
    """
    np.random.seed(3)
    model = _model()
    model.encode_observation(_reading(-_SYNTHETIC_CURVATURE, -_SYNTHETIC_CURVATURE, 0.0))

    sampled = model.sample_observation(_cruising_state(model), COAST_STRAIGHT)
    reader = _model()
    reader.encode_observation(sampled)

    assert np.asarray(sampled["curvature_ahead"]).shape == (len(model.curvature_lookahead_m),)
    assert reader.curvature_estimate == pytest.approx(-_SYNTHETIC_CURVATURE, abs=0.01)


def test_the_mdp_arm_is_refused_rather_than_silently_driving_straight():
    """Test that a model with no road to read is rejected at construction.

    Purpose: The MDP observation is a table of vehicle kinematics with no road in it. A model
        built against it would hold curvature at zero for the whole episode and drive into the
        first corner, and the failure would look like a planning bug rather than a
        configuration one

    Given: A request for this model in MDP observation mode
    When: It is constructed
    Then: ValueError names the reading it needs and points at the known-track model

    Test type: configuration
    """
    with pytest.raises(ValueError, match="POMDP sensor observation"):
        _model(observation_mode=ObservationMode.MDP)

    with pytest.raises(ValueError, match="KnownTrackModel"):
        _model(observation_mode=ObservationMode.MDP)


def test_the_two_models_disagree_on_the_approach_to_a_corner(monkeypatch):
    """Test that the observed-track model sees a bend before the map model reaches it.

    Purpose: If the two produce the same rollout, the observation is not being read: the
        camera looks 10 m ahead while the arclength lookup only reports the road the ego is
        already on, so on the approach to a bend they *must* differ. Identical rollouts here
        would be the signature of a dead estimate. It is also the test that closes the sign
        convention — the camera's bend and the map's bend have to turn the same way

    Given: The live world driven forward until the ego is a few metres short of the first
        bend, its map curvature still exactly zero
    When: The observation there is encoded by the observed-track model, and both models roll
        the same state forward ten coasting decisions
    Then: The map model reports zero curvature under the ego while the observed model already
        reports a non-zero one of the same sign as the bend the map holds ahead; three steps
        later the map model is still exactly on the centreline and the observed one has left
        it; and over ten steps the two predictions disagree by more than 0.5 m at their widest

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

    np.random.seed(0)
    world = RacetrackPOMDP(discount_factor=0.95, seed=0, other_vehicles=0)
    truth = np.asarray(world.initial_state_dist().sample()[0], dtype=float)
    geometry, _ = geometry_from_world(world)
    # The first bend begins at 58 m; coast until the ego is within 6 m of it.
    while float(truth[EGO_ARCLENGTH_M]) < 52.0:
        truth = np.asarray(world.sample_next_state(truth, COAST_STRAIGHT), dtype=float)

    observed = ObservedTrackModel(discount_factor=0.95, process_noise_std=0.0)
    observed.encode_observation(world.sample_observation(truth, COAST_STRAIGHT))
    mapped = KnownTrackModel(discount_factor=0.95, process_noise_std=0.0, track_geometry=geometry)

    ahead = float(mapped.curvature_ahead(truth[None, :EGO_STATE_WIDTH])[0, 0])
    assert float(geometry.curvature_at(float(truth[EGO_ARCLENGTH_M]))) == 0.0
    assert abs(observed.curvature_estimate) > 0.01
    assert np.sign(observed.curvature_estimate) == np.sign(ahead)

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
