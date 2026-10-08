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

**The map also predicts the camera.** The observation reports the lane's curvature at fixed
distances ahead of the bumper, and a model holding the circuit can say what those readings
should be: look the profile up at ``arclength + d``. That is
:meth:`KnownTrackModel.curvature_ahead`, and unlike the base class's held-constant default it
depends on the particle — so it is the one term in the likelihood that scores *where along
the lap* a particle thinks it is.

There is a caveat worth knowing, and it is not the lookup being sloppy: **a lane change
re-bases the arclength, and the map is then read at the wrong place.** The world numbers
arclength from whichever lane the ego first occupied, and re-anchors it the first time the
ego enters a lane it has not visited. Measured over 113 steps of random control on eight
seeds, rebuilding the ego's position from ``(arclength, lat, ang)`` landed within 0.38 m of
the truth while the ego stayed put, and up to 58.7 m out after a re-base. The world rebuilds
its own curvature profile at that same moment, from that same lane, so the two agree — but a
particle carrying a pre-re-base arclength does not.

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
        track_geometry: Curvature and lane layout of the lap, indexed by arclength.

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

    def curvature_ahead(self, ego: np.ndarray) -> np.ndarray:
        """Look the map up at each lookahead distance past every particle's own arclength.

        This is the term that makes the observation's curvature channel worth something to a
        mapped planner. The base class holds one value across the channel, which cannot tell
        two particles apart; here a particle 20 m down the track from the truth reads the
        wrong curvature at every distance at once, and no other channel in the reading says
        where along the lap the car is.

        Args:
            ego: The ego block of the particle batch, shape ``(B, EGO_STATE_WIDTH)``.

        Returns:
            Signed curvature in 1/m, shape ``(B, L)``, wrapping around the lap.
        """
        distances = np.asarray(self.curvature_lookahead_m, dtype=float)
        arclength = ego[:, EGO_ARCLENGTH_M][:, None] + distances[None, :]
        return np.asarray(self.track_geometry.curvature_at(arclength), dtype=float)


__all__ = ["KnownTrackModel"]
