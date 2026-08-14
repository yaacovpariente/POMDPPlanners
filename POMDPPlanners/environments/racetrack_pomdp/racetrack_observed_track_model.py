# SPDX-License-Identifier: MIT

"""Planner-side racetrack model for a planner with no map of the circuit.

This is the hard half of the pair. Where
:class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model.KnownTrackModel`
looks the curvature up by arclength, this model has to take the camera's word for it: the
only thing it knows about the road is the curvature-ahead channel of the observation, a few
noisy samples of where the lane bends in front of the bumper.

**The channel leads the state.** At a step where the arclength still sits on a straight, the
camera is already reporting the bend at 20 and 30 m. That is the whole reason this model can
corner at all.

**The estimate is one number per step, shared by every particle.** It is read once in
:meth:`ObservedTrackModel.encode_observation`, the model's single raw-observation seam, and
handed to every row of every rollout until the next observation arrives. That is a real
approximation and not a bookkeeping detail: within one planning step the belief cannot
disagree about where the road goes, so curvature uncertainty is absent from the search.

**Only the nearest sample drives the rollout.** ``curvature_ahead`` reports at several
distances and this model uses the first, because turning the further samples into a
transition needs the arclength the ego was at *when the reading was taken*, and
``encode_observation`` receives an observation and not a state. The further samples are still
scored — see :meth:`RacetrackModelPOMDP.curvature_ahead` — but for this model that term is
identical across particles and drops out at normalisation, which is the honest outcome for a
prediction derived from the reading being scored. A mapped model is where those samples earn
their place.

**What the redesign changed here.** This model used to fit the curvature itself, out of the
on-road layer of an occupancy grid: a quadratic through the corridor centres, plus a trace of
the ego's own lane and its neighbours to pin the car down laterally. That fit was biased and
the bias depended on how well the car was being driven — measured against the live circuit,
it returned 1.06x the true curvature at lane-relative yaw under 0.15 rad and 0.84x between
0.35 and 0.50, so a weak estimate under-steered, under-steering yawed the car away from the
lane, and a yawed car read the road worse still. None of that survives: a camera reports
curvature directly, at a noise width that does not depend on the driver.

Classes:
    ObservedTrackModel: Generative racetrack model that reads curvature off the observation.
"""

from typing import Any, Dict

import numpy as np

from POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp import (
    CURVATURE_AHEAD_KEY,
    RacetrackModelPOMDP,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import ObservationMode


class ObservedTrackModel(RacetrackModelPOMDP):
    """Generative racetrack model that takes its curvature from the camera.

    Every decision, :meth:`encode_observation` reads the nearest curvature sample out of the
    incoming observation and caches it. The rollout that follows integrates the Frenet pair
    against that one number for the whole horizon.

    Attributes:
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
        >>> # A camera reporting a left-hand bend moves it
        >>> reading = (
        ...     np.zeros(4, dtype=np.float32),               # ego pose
        ...     np.array([10.0], dtype=np.float32),          # speedometer
        ...     np.array([0.0, 0.0], dtype=np.float32),      # lane pose
        ...     np.array([0.04, 0.04, 0.04], dtype=np.float32),  # curvature ahead
        ...     np.zeros((4, 5), dtype=np.float32),          # no detections
        ... )
        >>> _ = env.encode_observation(reading)
        >>> round(env.curvature_estimate, 3)
        0.04
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

    def __init__(self, discount_factor: float, **model_kwargs: Any) -> None:
        """Initialize a racetrack model that takes the road from its observations.

        Args:
            discount_factor: Discount factor for future rewards (0 < d <= 1).
            **model_kwargs: Every other argument of
                :class:`~POMDPPlanners.environments.racetrack_pomdp.racetrack_model_pomdp.RacetrackModelPOMDP`,
                forwarded unchanged.

        Raises:
            ValueError: If the model is asked for the MDP observation arm, whose observation
                carries no road.
        """
        self._curvature_estimate = 0.0
        super().__init__(discount_factor=discount_factor, **model_kwargs)
        if self.observation_mode is not ObservationMode.POMDP:
            raise ValueError(
                "ObservedTrackModel needs the POMDP sensor observation: its whole premise is "
                "taking the road from the camera's curvature channel, and the MDP arm's "
                "kinematics table has no road in it. Use KnownTrackModel for the MDP arm."
            )

    @property
    def curvature_estimate(self) -> float:
        """Signed curvature in 1/m read from the most recent observation."""
        return self._curvature_estimate

    def encode_observation(self, observation: Any) -> Dict[str, np.ndarray]:
        """Encode the raw reading and refresh the model's curvature estimate.

        Extends the base encoder with the one side effect this model is built around: the
        nearest curvature sample of the incoming reading is cached for the rollouts that
        follow. Encoding an observation therefore *changes the transition model*, which is
        unusual and deliberate — it is how a planner with no map learns where the road goes.

        Args:
            observation: The raw five-part sensor reading from the world.

        Returns:
            The five encoded sensor keys, as the base class produces them.
        """
        encoded = super().encode_observation(observation)
        samples = np.asarray(encoded[CURVATURE_AHEAD_KEY], dtype=float).reshape(-1)
        self._curvature_estimate = float(samples[0])
        return encoded

    def _curvature_for(self, ego: np.ndarray) -> np.ndarray:
        return np.full(len(ego), self._curvature_estimate, dtype=float)


__all__ = ["ObservedTrackModel"]
