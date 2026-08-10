# SPDX-License-Identifier: MIT

"""Planner-side racetrack model for a planner that knows the circuit.

This is the easy half of the pair. The track is a fixed, closed loop of straights and
constant-radius arcs, so if the planner is allowed to know it, the curvature under a
particle is a table lookup on the particle's own arclength — exact, cheap, and valid as far
ahead as the rollout cares to go. See
:mod:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry` for the table
and how it is walked out of the lane graph.

Knowing the circuit is a real assumption, not a free win: it is the "map-based planner"
baseline. Its counterpart,
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_observed_track_model.ObservedTrackModel`,
has to read the road out of the observation instead, and the gap between the two is what a
map is worth on this problem.

Classes:
    KnownTrackModel: Generative racetrack model whose curvature comes from a track map.
"""

from typing import Any

import numpy as np

from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import RacetrackModelPOMDP
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import EGO_ARCLENGTH_M
from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import TrackGeometry


class KnownTrackModel(RacetrackModelPOMDP):
    """Generative racetrack model that looks its curvature up in a track map.

    Every substep indexes :attr:`track_geometry` with the particle's own arclength slot,
    which the base class advances by the along-track rate as it integrates. A rollout
    therefore drives *through* a corner rather than past it: the curvature the Frenet update
    sees changes exactly where the road changes.

    Attributes:
        track_geometry: Piecewise-constant curvature of the lap, indexed by arclength.

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>>
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
        ...     TrackGeometry,
        ... )
        >>>
        >>> # A lap of one 10 m straight followed by one 10 m left-hand arc
        >>> geometry = TrackGeometry(
        ...     segment_starts=np.array([0.0, 10.0]),
        ...     segment_curvatures=np.array([0.0, 0.05]),
        ...     total_length_m=20.0,
        ... )
        >>> env = KnownTrackModel(discount_factor=0.95, track_geometry=geometry)
        >>>
        >>> # Propagate a state cruising straight at the speed limit
        >>> state = np.zeros(env.state_width)
        >>> state[3] = 10.0  # speed, m/s
        >>> next_state = env.sample_next_state(state, action=13)  # coast, straight ahead
        >>> bool(next_state[0] > state[0])  # the ego moved forward
        True
        >>> bool(next_state[6] > 0.0)  # and its arclength advanced with it
        True

    Note:
        The map is built for one lane. Parallel lanes on the same segment have different
        radii, so a model that ever learns to change lanes would need a per-lane profile.
    """

    def __init__(
        self,
        discount_factor: float,
        track_geometry: TrackGeometry,
        **model_kwargs: Any,
    ) -> None:
        """Initialize a racetrack model over a known circuit.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            track_geometry: Curvature profile of the lap the ego is driving. Build one with
                :func:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry.build_track_geometry`
                or
                :func:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry.geometry_from_world`.
            **model_kwargs: Every other argument of
                :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`
                — observation mode, step rates, action presets, noise scales and the reward
                weights — forwarded unchanged.
        """
        self.track_geometry = track_geometry
        super().__init__(discount_factor=discount_factor, **model_kwargs)

    def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.track_geometry.curvature_at(ego[:, EGO_ARCLENGTH_M]), dtype=float
        ).reshape(len(ego))

    # The base class's all-ones on-road layer and zero on-road likelihood are kept
    # deliberately, and the two halves of that have different reasons.
    #
    # Dropping the *likelihood* term is free. The layer would be a function of the
    # particle's arclength only, and arclength carries no process noise, so every particle
    # in a belief reports the same road; the term adds one constant to every log-weight and
    # vanishes at normalisation, at a cost of 144 cells of arithmetic per particle per step.
    #
    # Leaving the *sampled* layer at all ones is a genuine inaccuracy, not a free one: an
    # observation this model draws in a rollout claims the whole window is drivable, which
    # in a corner is false. It is tolerated because nothing downstream of this model reads
    # that layer -- the belief's tracker works off the presence layer, and unlike
    # ObservedTrackModel this model never feeds its own samples back through an estimator.
    # If a consumer of the on-road layer ever appears, override _render_on_road_layer here;
    # the map makes it straightforward, and ObservedTrackModel already shows the rasteriser.


__all__ = ["KnownTrackModel"]
