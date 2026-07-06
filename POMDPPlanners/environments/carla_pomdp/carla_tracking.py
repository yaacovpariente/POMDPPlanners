# SPDX-License-Identifier: MIT

"""Constant-velocity multi-object tracker for lidar vehicle detections.

The temporal (prediction) stage of the perception pipeline: it turns the per-frame
detections from :func:`~POMDPPlanners.environments.carla_pomdp.carla_perception.lidar_vehicle_detections`
into persistent tracks that carry an estimated ego-frame **velocity**, using an alpha-beta
filter (a simplified constant-velocity Kalman) with greedy nearest-neighbour association.
Undetected tracks coast on their velocity and decay in confidence until evicted; unmatched
detections spawn new tracks. This is what supplies the velocity a downstream planner needs
and lets the belief shrink to a thin snapshot of the tracker's estimate rather than doing
the tracking itself.

A track is a row ``[rel_x, rel_y, vx, vy, confidence]`` in the ego frame (``rel_x`` forward,
``rel_y`` left; ``vx, vy`` the agent's velocity relative to the ego; ``confidence`` in
``[0, 1]``). The tracker is a pure function of ``(prior_tracks, detections, dt)`` so it
composes with an immutable belief that carries the track set forward.
"""

from typing import Dict, Optional

import numpy as np

TRACK_WIDTH = 5  # [rel_x, rel_y, vx, vy, confidence]

DEFAULT_GATE = 4.0  # max association distance (m) between a detection and a predicted track
DEFAULT_ALPHA = 0.5  # position blend toward the measurement (alpha-beta filter)
DEFAULT_BETA = 0.3  # velocity blend from the position residual (alpha-beta filter)
DEFAULT_NEW_CONFIDENCE = 0.5  # confidence a freshly spawned track starts with
DEFAULT_CONFIDENCE_GAIN = 0.25  # confidence added on a matched detection
DEFAULT_CONFIDENCE_DECAY = 0.25  # confidence removed on a missed (coasted) track
DEFAULT_MIN_CONFIDENCE = 0.1  # tracks below this confidence are evicted
DEFAULT_TRACK_RANGE = 50.0  # tracks beyond this ego range (m) are evicted


def update_tracks(
    tracks: Optional[np.ndarray],
    detections: Optional[np.ndarray],
    dt: float,
    gate: float = DEFAULT_GATE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    new_confidence: float = DEFAULT_NEW_CONFIDENCE,
    confidence_gain: float = DEFAULT_CONFIDENCE_GAIN,
    confidence_decay: float = DEFAULT_CONFIDENCE_DECAY,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    track_range: float = DEFAULT_TRACK_RANGE,
) -> np.ndarray:
    """Advance the tracker one step and return the new ``(K, 5)`` track set.

    Args:
        tracks: Prior ``(N, 5)`` tracks ``[rel_x, rel_y, vx, vy, confidence]``, or ``None``.
        detections: ``(M, 3)`` detections ``[rel_x, rel_y, confidence]`` from the detector.
        dt: Time step (s) since the previous update.
        gate: Max distance (m) at which a detection is associated with a predicted track.
        alpha: Position correction gain toward the measurement.
        beta: Velocity correction gain from the position residual.
        new_confidence: Confidence a newly spawned track starts with.
        confidence_gain: Confidence added when a track is matched to a detection.
        confidence_decay: Confidence removed when a track is missed (coasted).
        min_confidence: Tracks below this confidence are evicted.
        track_range: Tracks beyond this ego-frame range (m) are evicted.

    Returns:
        The updated ``(K, 5)`` track set (empty ``(0, 5)`` when there is nothing to track).
    """
    predicted = _predict(_as_tracks(tracks), dt)
    dets = (
        np.asarray(detections, dtype=float).reshape(-1, 3) if detections is not None else _EMPTY_DET
    )
    det_to_track = _associate(predicted, dets, gate)
    matched = set(det_to_track.values())
    rows = [
        (
            _matched_track(predicted[det_to_track[i]], dets[i], dt, alpha, beta, confidence_gain)
            if i in det_to_track
            else _new_track(dets[i], new_confidence)
        )
        for i in range(len(dets))
    ]
    rows += [
        _coast(predicted[t], confidence_decay) for t in range(len(predicted)) if t not in matched
    ]
    updated = np.array(rows).reshape(-1, TRACK_WIDTH) if rows else np.zeros((0, TRACK_WIDTH))
    return _evict(updated, min_confidence, track_range)


_EMPTY_DET = np.zeros((0, 3))


def _as_tracks(tracks: Optional[np.ndarray]) -> np.ndarray:
    if tracks is None:
        return np.zeros((0, TRACK_WIDTH))
    return np.asarray(tracks, dtype=float).reshape(-1, TRACK_WIDTH)


def _predict(tracks: np.ndarray, dt: float) -> np.ndarray:
    moved = tracks.copy()
    moved[:, 0] += moved[:, 2] * dt
    moved[:, 1] += moved[:, 3] * dt
    return moved


def _associate(predicted: np.ndarray, dets: np.ndarray, gate: float) -> Dict[int, int]:
    """Greedy nearest-neighbour matching: return ``{detection_index: track_index}``."""
    matches: Dict[int, int] = {}
    if len(predicted) == 0 or len(dets) == 0:
        return matches
    diff = dets[:, None, :2] - predicted[None, :, :2]
    distance = np.hypot(diff[..., 0], diff[..., 1])
    used_tracks = set()
    for flat in np.argsort(distance, axis=None):
        det_idx, track_idx = int(flat // len(predicted)), int(flat % len(predicted))
        if distance[det_idx, track_idx] > gate:
            break
        if det_idx in matches or track_idx in used_tracks:
            continue
        matches[det_idx] = track_idx
        used_tracks.add(track_idx)
    return matches


def _matched_track(track, det, dt, alpha, beta, confidence_gain):
    residual = det[:2] - track[:2]
    position = track[:2] + alpha * residual
    velocity = track[2:4] + (beta / dt) * residual
    confidence = min(1.0, track[4] + confidence_gain)
    return [position[0], position[1], velocity[0], velocity[1], confidence]


def _new_track(det, new_confidence):
    return [det[0], det[1], 0.0, 0.0, new_confidence]


def _coast(track, confidence_decay):
    coasted = track.copy()
    coasted[4] -= confidence_decay
    return coasted


def _evict(tracks: np.ndarray, min_confidence: float, track_range: float) -> np.ndarray:
    if len(tracks) == 0:
        return tracks
    keep = (tracks[:, 4] >= min_confidence) & (np.hypot(tracks[:, 0], tracks[:, 1]) <= track_range)
    return tracks[keep]
