# SPDX-License-Identifier: MIT

"""Tests for the standalone CARLA perception + prediction pipeline.

Covers the swappable :class:`PerceptionModel` / :class:`MotionTracker` interfaces and their
defaults (:class:`LidarCameraPerceptionModel`, :class:`OracleAgentPerceptionModel`,
:class:`AlphaBetaTracker`), plus the composed, immutable :class:`CarlaPerceptionPipeline` that
turns a raw observation into an ego-frame agent block and a fused forward-obstacle distance.
All tests run on hand-built sensor arrays.
"""

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_perception_pipeline import (
    AlphaBetaTracker,
    CarlaPerceptionPipeline,
    Detections,
    LidarCameraPerceptionModel,
    OracleAgentPerceptionModel,
)


def _vehicle_blob(center_x: float, center_y: float = 0.0) -> np.ndarray:
    """A 5x5 grid of vehicle-height lidar points centred at ``(center_x, center_y)``."""
    grid = np.linspace(-0.4, 0.4, 5)
    xx, yy = np.meshgrid(grid, grid)
    points = np.zeros((xx.size, 4))
    points[:, 0] = center_x + xx.ravel()
    points[:, 1] = center_y + yy.ravel()
    points[:, 2] = -1.0
    return points


def _camera_with_red_bulb(size: int = 10) -> np.ndarray:
    """A 32x32 dark RGB frame with a red bulb in the upper region."""
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[2 : 2 + size, 14 : 14 + size, 0] = 255
    return frame


def test_lidar_camera_model_detects_vehicle_and_clear_light():
    """The default model clusters a lidar vehicle and reports a clear (green) light.

    Purpose: Validates single-frame perception yields a vehicle detection and no stop signal.

    Given: A lidar cloud with one vehicle-sized blob 8 m ahead and a dark camera frame
    When: LidarCameraPerceptionModel.detect runs
    Then: One vehicle position is returned and the traffic light is not a stop

    Test type: unit
    """
    obs = {"lidar": _vehicle_blob(8.0), "camera": np.zeros((32, 32, 3), dtype=np.uint8)}

    detections = LidarCameraPerceptionModel().detect(obs)

    assert detections.vehicle_positions.shape == (1, 3)
    assert detections.traffic_light[0] == 0.0


def test_lidar_camera_model_infers_red_light_from_camera():
    """A red bulb in the camera frame is inferred as a stop signal.

    Purpose: Validates the default model infers the traffic light from the image, not a channel.

    Given: An observation with a red bulb in the camera and no traffic_light channel
    When: LidarCameraPerceptionModel.detect runs
    Then: The perceived traffic light signals a stop

    Test type: unit
    """
    detections = LidarCameraPerceptionModel().detect({"camera": _camera_with_red_bulb()})

    assert detections.traffic_light[0] == 1.0


def test_lidar_camera_model_channel_source_reads_traffic_light_key():
    """With traffic_light_source='channel' the model reads the ground-truth light channel.

    Purpose: Validates the channel source bypasses camera inference and reads the channel.

    Given: An observation whose traffic_light channel says stop at 10 m, camera green
    When: A channel-sourced LidarCameraPerceptionModel.detect runs
    Then: The reported light is the channel value, not the camera inference

    Test type: unit
    """
    obs = {
        "camera": np.zeros((32, 32, 3), dtype=np.uint8),
        "traffic_light": np.array([1.0, 10.0]),
    }

    detections = LidarCameraPerceptionModel(traffic_light_source="channel").detect(obs)

    assert list(detections.traffic_light) == [1.0, 10.0]


def test_oracle_model_reads_present_agents_as_positions():
    """The oracle model returns only present agent slots as detections.

    Purpose: Validates ground-truth agent positions are read, padding slots dropped.

    Given: A two-slot agents channel with one present agent 8 m ahead and one empty slot
    When: OracleAgentPerceptionModel.detect runs
    Then: Exactly one detection at (8, 0) is returned

    Test type: unit
    """
    agents = np.array([1.0, 8.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    detections = OracleAgentPerceptionModel(max_tracked_agents=2).detect({"agents": agents})

    assert detections.vehicle_positions.shape == (1, 3)
    assert list(detections.vehicle_positions[0][:2]) == [8.0, 0.0]


def test_alpha_beta_tracker_spawns_track_from_detection():
    """The default tracker spawns a five-column track from a fresh detection.

    Purpose: Validates the MotionTracker default wraps the constant-velocity tracker.

    Given: No prior tracks and one detection 8 m ahead
    When: AlphaBetaTracker.update runs
    Then: One (rel_x, rel_y, vx, vy, confidence) track is returned

    Test type: unit
    """
    tracks = AlphaBetaTracker().update(None, np.array([[8.0, 0.0, 1.0]]), dt=0.05)

    assert tracks.shape == (1, 5)


def test_pipeline_process_produces_agent_block_and_successor():
    """Processing a lidar vehicle fills an agent slot and returns a successor pipeline.

    Purpose: Validates the composed pipeline perceives, tracks, and packs the agent block.

    Given: A pipeline with one agent slot and an observation with a vehicle 8 m ahead
    When: process runs
    Then: The agent block marks a present agent near 8 m and a distinct successor is returned

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(max_tracked_agents=1)

    output = pipeline.process({"lidar": _vehicle_blob(8.0)})

    assert output.agent_rows.shape == (1, 5)
    assert output.agent_rows[0, 0] == 1.0
    assert abs(output.agent_rows[0, 1] - 8.0) < 1.0
    assert output.pipeline is not pipeline


def test_pipeline_reports_red_light_as_forward_obstacle():
    """A red camera light within range is reported as a forward-obstacle distance.

    Purpose: Validates stop_for_traffic_lights folds the inferred light into the obstacle.

    Given: A pipeline with traffic-light stopping and a red bulb 10 m ahead (channel source)
    When: process runs
    Then: The obstacle distance is the light's stop distance

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(
        max_tracked_agents=1,
        perception=LidarCameraPerceptionModel(traffic_light_source="channel"),
        sensor_fusion=False,
        stop_for_traffic_lights=True,
    )

    output = pipeline.process({"traffic_light": np.array([1.0, 10.0])})

    assert output.obstacle_distance == 10.0
    assert output.agent_rows[0, 0] == 1.0  # folded into the agent block as a hazard slot
    assert output.agent_rows[0, 1] == 10.0


def test_pipeline_folds_lidar_obstacle_into_empty_agent_slot():
    """A lidar obstacle no vehicle covers is folded into the agent block as a hazard.

    Purpose: Validates the fused obstacle becomes a present agent slot for the planner.

    Given: A fusion pipeline and a lidar return 6 m ahead with no clustered vehicle
    When: process runs
    Then: One present slot at ~6 m directly ahead appears in the agent block

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(
        max_tracked_agents=2, sensor_fusion=True, stop_for_traffic_lights=False
    )

    output = pipeline.process({"lidar": np.array([[6.0, 0.0, -1.0, 0.5]])})

    present = output.agent_rows[:, 0]
    assert present.sum() == 1.0
    assert output.agent_rows[0, 1] == 6.0


def test_pipeline_does_not_duplicate_a_tracked_obstacle():
    """A lidar hazard already covered by a tracked vehicle adds no extra slot.

    Purpose: Validates obstacle folding deduplicates against the perceived vehicles.

    Given: A fusion pipeline whose lidar shows a vehicle-sized blob 8 m ahead
    When: process runs (the blob is both a tracked vehicle and the corridor obstacle)
    Then: Exactly one slot is present, not a vehicle plus a duplicate hazard

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(
        max_tracked_agents=3, sensor_fusion=True, stop_for_traffic_lights=False
    )

    output = pipeline.process({"lidar": _vehicle_blob(8.0)})

    assert output.agent_rows[:, 0].sum() == 1.0


def test_pipeline_reports_no_obstacle_when_disabled_and_clear():
    """With fusion and light-stopping off, a clear scene reports no obstacle.

    Purpose: Validates the obstacle channel is silent when both sources are disabled.

    Given: A pipeline with sensor_fusion and stop_for_traffic_lights both off
    When: process runs on an empty observation
    Then: The obstacle distance is None

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(
        max_tracked_agents=1, sensor_fusion=False, stop_for_traffic_lights=False
    )

    assert pipeline.process({"lidar": np.zeros((0, 4))}).obstacle_distance is None


def test_pipeline_is_immutable_across_process():
    """process does not mutate the pipeline's own tracker state.

    Purpose: Validates immutability so a belief can thread perception forward safely.

    Given: A fresh pipeline with empty tracks
    When: process runs on an observation with a vehicle
    Then: The original pipeline's tracks stay empty while the successor carries the new track

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(max_tracked_agents=1)

    output = pipeline.process({"lidar": _vehicle_blob(8.0)})

    assert pipeline.tracks.shape == (0, 5)
    assert len(output.pipeline.tracks) == 1


def test_pipeline_coasts_briefly_occluded_vehicle():
    """A vehicle seen then lost is coasted by the tracker through a dropout.

    Purpose: Validates the tracker's coasting keeps a briefly occluded agent in the block.

    Given: A pipeline that perceives a vehicle, then an empty observation
    When: process runs twice, threading the successor pipeline forward
    Then: The agent slot is still present after the dropout step

    Test type: unit
    """
    first = CarlaPerceptionPipeline(max_tracked_agents=1).process({"lidar": _vehicle_blob(8.0)})

    second = first.pipeline.process({"lidar": np.zeros((0, 4))})

    assert second.agent_rows[0, 0] == 1.0


def test_pipeline_detects_object_type_from_model():
    """The pipeline delegates single-frame detection to its perception model.

    Purpose: Validates a swapped-in perception model drives the pipeline's detections.

    Given: A pipeline built with an OracleAgentPerceptionModel and an agents channel
    When: process runs
    Then: The agent block reflects the oracle-perceived vehicle

    Test type: unit
    """
    pipeline = CarlaPerceptionPipeline(
        max_tracked_agents=1, perception=OracleAgentPerceptionModel(max_tracked_agents=1)
    )

    output = pipeline.process({"agents": np.array([1.0, 8.0, 0.0, 0.0, 5.0])})

    assert isinstance(output.pipeline.perception, OracleAgentPerceptionModel)
    assert output.agent_rows[0, 0] == 1.0


def test_detections_dataclass_carries_single_frame_fields():
    """Detections bundles vehicle positions, forward clearance, and a traffic light.

    Purpose: Validates the single-frame perception output structure.

    Given: A hand-built Detections instance
    When: Its fields are read
    Then: The vehicle positions, clearance, and traffic light are returned as given

    Test type: unit
    """
    detections = Detections(np.zeros((0, 3)), 50.0, np.array([0.0, 0.0]))

    assert detections.vehicle_positions.shape == (0, 3)
    assert detections.forward_clearance == 50.0
    assert list(detections.traffic_light) == [0.0, 0.0]
