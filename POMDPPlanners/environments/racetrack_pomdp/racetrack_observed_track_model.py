# SPDX-License-Identifier: MIT

"""Planner-side racetrack model for a planner with no map of the circuit.

This is the hard half of the pair. Where
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model.KnownTrackModel`
looks the curvature up by arclength, this model has to *see* it: the only thing it knows
about the road is the on-road layer of the occupancy observation, a 12x12 window of 3 m
cells over +/-18 m, aligned to the ego's own axes.

**The layer leads the state.** At a step where the arclength still sits on a straight, the
on-road cells already show the drivable corridor bending — the road ahead is in the window
before the ego reaches it. That is the whole reason this model can corner at all.

**The estimate is one number per step, shared by every particle.** It is computed once in
:meth:`ObservedTrackModel.encode_observation`, the model's single raw-observation seam, and
handed to every row of every rollout until the next observation arrives. That is a real
approximation and not a bookkeeping detail: within one planning step the belief cannot
disagree about where the road goes, so curvature uncertainty is absent from the search.

**Beyond 18 m the estimate is simply held.** The rollout keeps using the last measured
curvature for the rest of the horizon, so a bend that starts just past the window edge is
invisible. At the shipped rates a rollout covers roughly 12 m, comfortably inside the
window, but this is a true consequence of seeing 18 m rather than a shortcut worth fixing
in the estimator.

**How well this reads the road depends on how well the car is being driven, and the variable
that decides it is lane-relative yaw.** Measured against the live circuit deep inside one
arc, a thousand poses per band::

    |ang| <= 0.15 rad    1.06x true curvature
    0.15 - 0.25          0.97x
    0.25 - 0.35          0.88x
    0.35 - 0.50          0.84x

A lane-keeper holds a mean ``|ang|`` of 0.089 and gets bends back at 0.78x over a lap.
Every figure here scores against the curvature at the ego's arclength. Scoring instead
against the lane object the ego happens to occupy reads about 0.05 lower, because a
wandering car crosses onto the parallel lane, which on an arc has a different radius — a
difference in the yardstick, not in the model.

Those bands are what the fit gives *without* the textbook ``/(1 + b^2)^{3/2}`` slope term,
which :meth:`ObservedTrackModel._fit_curvature` deliberately omits. With it the same bands
read 1.05x, 0.91x, 0.77x and 0.65x — the correction bit hardest at exactly the yaw angles
where the estimate matters most. Removing it buys a great deal at high yaw and almost
nothing at low: over a lane-keeper lap the ratio moves only 0.76x to 0.78x, because a car
that is driving well spends 83% of its steps under 0.15 rad where the term was inert. It
earns its place by helping a car that is already in trouble, not by improving the average.

So there is a loop here worth naming: a weak estimate under-steers, under-steering yaws the
car away from the lane, and a yawed car reads the road worse still. Quote an accuracy for
this model without saying how the car was driven and the number means nothing.

Classes:
    ObservedTrackModel: Generative racetrack model that reads curvature off the observation.
"""

from typing import Any, Dict, Tuple

import numpy as np

from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import (
    OCCUPANCY_KEY,
    RacetrackModelPOMDP,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    EGO_ANG,
    EGO_LAT,
    GRID_CELLS,
    GRID_HALF_EXTENT_M,
    GRID_STEP_M,
    ON_ROAD_LAYER,
    ObservationMode,
)

# The ego sits in the middle cell of the occupancy grid, at (6, 6) for the shipped 12x12.
_GRID_CENTRE = GRID_CELLS // 2

# Offset from a cell index to the metre coordinate of that cell's centre.
_CELL_CENTRE_OFFSET_M = GRID_HALF_EXTENT_M - 0.5 * GRID_STEP_M

# A quadratic needs three points; below that the fit is not determined.
_MIN_FIT_POINTS = 3

# Arclength step used when rasterising the predicted lane centreline. Half a metre is a
# sixth of a cell, fine enough that the marked cells are the ones the curve really crosses.
LANE_SAMPLE_STEP_M = 0.5

# How far along the predicted centreline to walk when rasterising it. Curvature and yaw both
# stretch arclength relative to the grid's along-track axis, so a window's worth of cells
# needs more than a window's worth of centreline.
LANE_SAMPLE_REACH_M = 1.5 * GRID_HALF_EXTENT_M


def _cell_centres_m(indices: Any) -> np.ndarray:
    """Metre coordinate of the centre of each grid cell index, along either axis."""
    return np.asarray(indices, dtype=float) * GRID_STEP_M - _CELL_CENTRE_OFFSET_M


def _cell_indices(metres: np.ndarray) -> np.ndarray:
    """Grid cell index containing each metre coordinate, along either axis."""
    return np.floor((np.asarray(metres, dtype=float) + GRID_HALF_EXTENT_M) / GRID_STEP_M).astype(
        int
    )


class ObservedTrackModel(RacetrackModelPOMDP):
    """Generative racetrack model that reads its curvature out of the on-road layer.

    Every decision, :meth:`encode_observation` fits the drivable corridor visible in the
    observation and stores one signed curvature; the rollout that follows integrates the
    Frenet pair against it. Unlike its known-track counterpart this model also *renders*
    the on-road layer, from that same curvature plus the particle's own lane offset and
    lane-relative angle, and scores it — the layer it reads has to be a layer it can also
    produce, or the observations it samples in a rollout would show an all-clear corridor
    on the approach to every corner.

    Attributes:
        curvature_window_m: Half-extent, along and across, of the cells the fit uses.
        curvature_estimate: Signed curvature in 1/m from the most recent observation; zero
            before any observation has been encoded.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> env = ObservedTrackModel(discount_factor=0.95)
        >>>
        >>> # Before any observation arrives the road is assumed straight
        >>> env.curvature_estimate
        0.0
        >>>
        >>> # A straight-ahead corridor keeps it there
        >>> observation = np.zeros((2, 12, 12), dtype=np.float32)
        >>> observation[1, :, 6] = 1.0
        >>> _ = env.encode_observation(observation)
        >>> abs(env.curvature_estimate) < 1e-9
        True
        >>>
        >>> # Propagate a state cruising straight at the speed limit
        >>> state = np.zeros(env.state_width)
        >>> state[3] = 10.0  # speed, m/s
        >>> next_state = env.sample_next_state(state, action=13)  # coast, straight ahead
        >>> bool(next_state[0] > state[0])  # the ego moved forward
        True

    Note:
        POMDP mode only. The MDP arm's observation is a table of vehicle kinematics with no
        road in it at all, so this model has nothing to read and would silently drive
        straight through every corner; the constructor refuses instead.
    """

    def __init__(
        self,
        discount_factor: float,
        curvature_window_m: float = 12.0,
        **model_kwargs: Any,
    ) -> None:
        """Initialize a racetrack model that infers the road from its observations.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            curvature_window_m: Cells further than this from the ego, along or across, are
                dropped before fitting. Defaults to 12.0 m, which was swept against the
                shipped circuit: the full 18 m window lets an unrelated part of the loop
                passing nearby into the fit, and roughly doubles the error.
            **model_kwargs: Every other argument of
                :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`,
                forwarded unchanged.

        Raises:
            ValueError: If ``curvature_window_m`` is not positive, or if the model is asked
                for the MDP observation arm, whose observation carries no road.
        """
        if curvature_window_m <= 0.0:
            raise ValueError(f"curvature_window_m must be positive, got {curvature_window_m}.")
        self.curvature_window_m = float(curvature_window_m)
        self._curvature_estimate = 0.0
        super().__init__(discount_factor=discount_factor, **model_kwargs)
        if self.observation_mode is not ObservationMode.POMDP:
            raise ValueError(
                "ObservedTrackModel needs the POMDP occupancy observation: its whole premise "
                "is reading the road out of the on-road layer, and the MDP arm's kinematics "
                "table has no road in it. Use KnownTrackModel for the MDP arm."
            )

    @property
    def curvature_estimate(self) -> float:
        """Signed curvature in 1/m inferred from the most recent observation."""
        return self._curvature_estimate

    # ── Curvature from the on-road layer ─────────────────────────────────
    def encode_observation(self, observation: Any) -> Dict[str, np.ndarray]:
        """Encode the raw reading and refresh the model's curvature estimate.

        Extends the base encoder with the one side effect this model is built around:
        the on-road layer of the incoming grid is fitted and the result cached for the
        rollouts that follow. Encoding an observation therefore *changes the transition
        model*, which is unusual and deliberate — it is how a planner with no map learns
        where the road goes.

        Args:
            observation: The raw ``(2, 12, 12)`` occupancy grid emitted by the world.

        Returns:
            ``{"occupancy": (2, 12, 12) float32}``, as the base class does.
        """
        encoded = super().encode_observation(observation)
        self._curvature_estimate = self._fit_curvature(encoded[OCCUPANCY_KEY][ON_ROAD_LAYER])
        return encoded

    def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
        return np.full(len(ego), self._curvature_estimate, dtype=float)

    def _fit_curvature(self, layer: np.ndarray) -> float:
        """Signed curvature of the corridor visible in one on-road layer, in 1/m.

        The fit is a plain quadratic ``across = a * along^2 + b * along + c`` through the
        corridor centres, read as ``2a``.

        The slope term of the textbook curvature formula, ``/(1 + b^2)^{3/2}``, is
        deliberately absent. It divides out a corridor slope on the understanding that the
        slope means the road runs obliquely across the window — but the dominant source of
        ``b`` here is the ego's own yaw tilting the whole corridor, so the correction fires
        when the road is doing nothing. It suppressed the estimate hardest exactly when the
        car was most turned away from the lane, which is when it most needs to be told the
        road bends. See the module docstring for the per-yaw-band measurements.

        Before restoring it, know that two independent harnesses found the same monotonic
        decline with yaw: a hand-placed sweep holding everything but yaw fixed deep inside
        one arc, and a driven-episode sweep whose poses were never chosen at all. They
        disagree on the absolute ratios, because they sample different parts of the track
        and the driven one carries lateral offset and boundary effects the placed one
        excludes, but they agree on the trend and on its direction. One harness showing this
        would be worth doubting; two built on different principles is why it was safe to act
        on. An earlier seven-point version of the placed sweep read as noise and was
        dismissed — single estimates here have a standard deviation near 0.25, so any
        re-test needs hundreds of samples per band, not a handful.
        """
        along, across = self._corridor_centres(np.asarray(layer, dtype=float) > 0.5)
        if len(along) < _MIN_FIT_POINTS:
            return 0.0
        quadratic = float(np.polyfit(along, across, 2)[0])
        return 2.0 * quadratic

    def _corridor_centres(self, on_road: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mean across-track position of the on-road cells in each along-track row, in metres.

        Rows with no on-road cell are skipped rather than filled: an empty row means the
        window ran off the end of what the sensor can see, and inventing a centre for it
        would bend the fit toward whatever value was invented.

        The gate is counted in whole cells out from the ego's *cell*, not from the ego. At
        the shipped 3 m resolution that rounds ``curvature_window_m`` down to a multiple of
        3 and shifts the window half a cell forward and to one side, because the ego sits at
        the corner of its cell rather than the middle of it. A quadratic fitted over a
        shifted window still recovers the same curvature, so this costs nothing on a
        constant-radius arc; it only matters where the curvature changes inside the window,
        and there it is the window's 12 m reach that decides the answer, not its half-cell
        offset.
        """
        window = int(np.floor(self.curvature_window_m / GRID_STEP_M))
        low, high = _GRID_CENTRE - window, _GRID_CENTRE + window
        along, across = [], []
        for row in range(max(low, 0), min(high + 1, GRID_CELLS)):
            columns = np.nonzero(on_road[row])[0]
            columns = columns[(columns >= low) & (columns <= high)]
            if columns.size == 0:
                continue
            along.append(_cell_centres_m(row))
            across.append(_cell_centres_m(columns.mean()))
        return np.asarray(along, dtype=float), np.asarray(across, dtype=float)

    # ── Predicting the on-road layer back ────────────────────────────────
    def _render_on_road_layer(self, state: Any) -> np.ndarray:
        """Rasterise the ego's own lane centreline under the current curvature estimate.

        One lane, not the whole road: the model has no idea how many lanes the circuit has
        or which side they sit on, so it draws the only piece of road it can locate — the
        centreline it is measuring its own offset from. The world's layer additionally shows
        neighbouring lanes, which shows up in :meth:`_on_road_log_prob` as a fixed
        disagreement on those cells rather than as anything that separates particles.
        """
        array = np.asarray(state, dtype=float)
        lateral, angle = float(array[EGO_LAT]), float(array[EGO_ANG])
        arclength = np.arange(
            -LANE_SAMPLE_REACH_M, LANE_SAMPLE_REACH_M + LANE_SAMPLE_STEP_M, LANE_SAMPLE_STEP_M
        )
        # Centreline in the lane frame relative to the ego, then rotated into the body
        # frame by the ego's lane-relative angle. The lane's own lateral axis is the body
        # frame's across-track axis when the two are aligned, which is why a centreline the
        # ego sits ``lateral`` metres from appears at ``-lateral`` across-track.
        lane_across = 0.5 * self._curvature_estimate * arclength**2 - lateral
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        body_along = cos_a * arclength + sin_a * lane_across
        body_across = -sin_a * arclength + cos_a * lane_across

        layer = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)
        rows, columns = _cell_indices(body_along), _cell_indices(body_across)
        inside = (rows >= 0) & (rows < GRID_CELLS) & (columns >= 0) & (columns < GRID_CELLS)
        layer[rows[inside], columns[inside]] = 1.0
        return layer

    def _on_road_log_prob(self, state: Any, observation: Any) -> float:
        observed = np.asarray(observation[OCCUPANCY_KEY], dtype=float)[ON_ROAD_LAYER] > 0.5
        return self._bernoulli_cell_log_prob(self._render_on_road_layer(state) > 0.5, observed)


__all__ = ["ObservedTrackModel"]
