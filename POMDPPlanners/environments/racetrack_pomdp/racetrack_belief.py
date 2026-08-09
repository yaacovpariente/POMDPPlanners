# SPDX-License-Identifier: MIT

"""Particle belief that stamps tracked opponents onto its particles each step.

The racetrack POMDP hides every velocity. Its observation is a presence/on-road occupancy
grid, so a single frame fixes where the other vehicles are and says nothing about where they
are going. The planner's state, though, carries ``[present, rel_x, rel_y, rel_vx, rel_vy]``
per agent slot, and those velocity terms are what let a rollout predict a closing opponent
instead of re-discovering it on the step it arrives.

:class:`TrackedAgentsBelief` fills those slots. On every update it keeps the previous frame,
hands both frames to :class:`~POMDPPlanners.environments.racetrack_pomdp.
racetrack_occupancy_tracker.OccupancyVelocityTracker`, and writes the resulting blob positions
and relative velocities into every particle's agent block. As with
:class:`~POMDPPlanners.environments.carla_pomdp.carla_belief.PerceivedAgentsBelief`, the agent
block is *trusted* rather than re-filtered as a per-particle latent: a weight-only filter can
propagate the opponents a particle was seeded with but can never acquire one that appears
mid-episode, and a slot seeded empty would stay empty for the whole episode. The ego block is
left to the particle filter, which is where the actual Bayesian work happens.

The same class serves the MDP arm of the matched pair, where the observation already contains
the agent rows and the tracker is never constructed. Running one belief across both arms is
deliberate: it keeps the arms differing in the observation alone, which is the whole point of
the comparison.

Classes:
    TrackedAgentsBelief: Particle belief that stamps tracked agent rows onto particles.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import (
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.racetrack_pomdp.racetrack_occupancy_tracker import (
    OccupancyVelocityTracker,
    TrackedCluster,
)
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_HEADING,
    EGO_STATE_WIDTH,
    ObservationMode,
    wrap_to_pi,
)
from POMDPPlanners.utils.config_to_id import config_to_id

OCCUPANCY_KEY = "occupancy"
AGENTS_KEY = "agents"


class TrackedAgentsBelief(WeightedParticleBeliefReinvigoration):
    """Weighted particle belief that stamps tracked opponent rows onto every particle.

    After the ordinary particle-filter weight update and resample, the reinvigoration step
    derives this step's agent rows — from the occupancy tracker in ``POMDP`` mode, straight
    off the observation in ``MDP`` mode — and writes them into every particle's agent slots
    with per-particle jitter. The returned belief is another ``TrackedAgentsBelief`` carrying
    the current grid and ego heading forward, so the tracking repeats every step. It keeps a
    window of ``tracker.frame_stride`` frames rather than just the last one, so raising the
    stride really does difference frames that far apart instead of dividing a one-step
    displacement by several steps.

    Note:
        What the ``POMDP`` arm recovers is velocity **relative to a moving, turning car**, not
        the opponents' velocity over the ground. That matches what the state's agent slots
        hold, so nothing is inconsistent — but it is not the obvious reading of "the belief
        infers the other vehicles' velocities", and a rollout that treats those numbers as
        absolute will be wrong by the ego's own motion. The magnitudes are coarse on top of
        that: a vehicle marks one 3 m cell per 0.2 s step, so the smallest non-zero reading is
        15 m/s. ``agent_velocity_jitter`` exists to cover that quantisation and defaults an
        order of magnitude above ``agent_pose_jitter`` for exactly that reason.

    Note:
        The frames carried between steps live on the private ``_previous_frames`` attribute.
        That is not a style choice: ``config_id`` deliberately ignores underscore-prefixed
        attributes, so a public grid would fold hundreds of floats of per-step observation
        into the belief's identity and into ``__hash__``, and two beliefs that a planner
        should treat as the same node would stop comparing equal. Private attributes are
        ordinary ``__dict__`` entries, so plain pickling still carries them to a worker.

        The flip side is the accepted cost: two beliefs holding identical particles but
        different tracking histories do hash alike even though their next step differs. The
        histories are one step of observation, not configuration, and the alternative -- an
        identity that changes on every frame -- defeats the caching the id exists for.

    Attributes:
        observation_mode: Which arm of the matched pair this belief is reading.
        max_tracked_agents: Number of fixed agent slots carried in each particle.
        agent_pose_jitter: Std of the Gaussian noise added to each stamped ``(rel_x, rel_y)``.
        agent_velocity_jitter: Std of the Gaussian noise added to each stamped
            ``(rel_vx, rel_vy)``.
        tracker: The occupancy tracker, or ``None`` in ``MDP`` mode where none is needed.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
        ...     EGO_STATE_WIDTH, GRID_CELLS)
        >>> width = EGO_STATE_WIDTH + 1 * 5
        >>> particles = np.zeros((4, width))
        >>> belief = TrackedAgentsBelief(  # jitter off so the stamped row is exact
        ...     particles=particles,
        ...     log_weights=np.log(np.ones(4) / 4),
        ...     max_tracked_agents=1,
        ...     agent_pose_jitter=0.0,
        ...     agent_velocity_jitter=0.0,
        ... )
        >>> grid = np.zeros((2, GRID_CELLS, GRID_CELLS), dtype=np.float32)
        >>> grid[0, 6, 6] = 1.0  # the ego, always written to the centre cell
        >>> grid[0, 9, 6] = 1.0  # an opponent ~9 m ahead
        >>> base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
        >>> refreshed = belief.reinvigorate("noop", {"occupancy": grid}, None, base)
        >>> np.asarray(refreshed.particles)[0, EGO_STATE_WIDTH:]  # present, ahead, no velocity
        array([ 1. , 10.5,  1.5,  0. ,  0. ])
        >>> isinstance(refreshed, TrackedAgentsBelief)  # stamping repeats next step
        True
    """

    def __init__(
        self,
        particles: Any,
        log_weights: np.ndarray,
        observation_mode: ObservationMode = ObservationMode.POMDP,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        agent_pose_jitter: float = 0.5,
        agent_velocity_jitter: float = 1.0,
        tracker: Optional[OccupancyVelocityTracker] = None,
        resampling: bool = True,
        ess_factor: float = 0.5,
        previous_frames: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ):
        """Initialize the tracked-agents belief.

        Args:
            particles: State particles ``[ego(7) | agent slots(K*5)]``.
            log_weights: Log-weights for the particles.
            observation_mode: Which arm of the matched pair the observation comes from.
                Defaults to ``ObservationMode.POMDP``.
            max_tracked_agents: Number of fixed agent slots carried per particle. Defaults
                to 4.
            agent_pose_jitter: Std of the noise added to each stamped ``(rel_x, rel_y)``, in
                metres. Defaults to 0.5.
            agent_velocity_jitter: Std of the noise added to each stamped
                ``(rel_vx, rel_vy)``, in m/s. Defaults to 1.0.
            tracker: Occupancy tracker to use in ``POMDP`` mode. Defaults to None, which
                builds one with the shipped grid geometry; in ``MDP`` mode the default stays
                ``None`` because no tracking is done.
            resampling: Enable automatic resampling when ESS drops. Defaults to True.
            ess_factor: Effective-sample-size threshold factor. Defaults to 0.5.
            previous_frames: Up to ``tracker.frame_stride`` earlier ``(grid, ego_heading)``
                pairs, oldest first. The oldest is what the next observation is differenced
                against, so a stride of 3 needs three of them before any velocity is
                reported. Defaults to None, meaning the next step is the episode's first and
                will report positions only. This is how :meth:`reinvigorate` hands its frames
                to its successor; they are stored privately so ``config_id`` never sees
                them.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        self.observation_mode = observation_mode
        self.max_tracked_agents = max_tracked_agents
        self.agent_pose_jitter = agent_pose_jitter
        self.agent_velocity_jitter = agent_velocity_jitter
        self.tracker = _resolve_tracker(observation_mode, tracker)
        self._previous_frames: Tuple[Tuple[np.ndarray, float], ...] = tuple(previous_frames or ())

    @property
    def config_id(self) -> str:
        """Deterministic identifier covering the particles and the stamping configuration.

        Two beliefs holding identical particles but jittering them differently behave
        differently on the next step, so the jitter widths belong in the identity. The carried
        frames do not: they are observation, not configuration.
        """
        return config_to_id(
            {
                "particles": super().config_id,
                "observation_mode": self.observation_mode.value,
                "max_tracked_agents": self.max_tracked_agents,
                "agent_pose_jitter": self.agent_pose_jitter,
                "agent_velocity_jitter": self.agent_velocity_jitter,
            }
        )

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBelief",
    ) -> "TrackedAgentsBelief":
        """Stamp this step's tracked agent rows onto every particle."""
        del action, pomdp
        particles = np.array(belief.particles, dtype=float, copy=True)
        heading = _mean_heading(particles, belief.normalized_weights)
        grid: Optional[np.ndarray] = None
        if self.observation_mode is ObservationMode.MDP:
            rows = self._mdp_rows(observation)
        else:
            grid = self._require_grid(observation)
            rows = self._pomdp_rows(grid, heading)
        self._stamp(particles, rows)
        return self._successor(particles, belief, grid, heading)

    def _mdp_rows(self, observation: Any) -> np.ndarray:
        if not _has_key(observation, AGENTS_KEY):
            raise ValueError(
                f"ObservationMode.MDP expects an '{AGENTS_KEY}' block in the observation, got "
                f"keys {_observation_keys(observation)}. Silently falling back to the "
                f"occupancy grid would leave the MDP arm reading a degraded observation."
            )
        rows = np.asarray(observation[AGENTS_KEY], dtype=float)
        expected = self.max_tracked_agents * AGENT_SLOT_WIDTH
        if rows.size != expected:
            raise ValueError(
                f"Observed '{AGENTS_KEY}' block has {rows.size} values but "
                f"max_tracked_agents={self.max_tracked_agents} needs {expected}."
            )
        return rows.reshape(self.max_tracked_agents, AGENT_SLOT_WIDTH)

    def _require_grid(self, observation: Any) -> np.ndarray:
        if not _has_key(observation, OCCUPANCY_KEY):
            raise ValueError(
                f"ObservationMode.POMDP expects an '{OCCUPANCY_KEY}' grid in the observation, "
                f"got keys {_observation_keys(observation)}. Degrading to zero agent rows "
                f"here would produce a velocity-blind planner that still looks like it works."
            )
        return np.asarray(observation[OCCUPANCY_KEY])

    def _pomdp_rows(self, grid: np.ndarray, heading: float) -> np.ndarray:
        tracker = self.tracker
        if tracker is None:
            raise ValueError("ObservationMode.POMDP requires a tracker, but tracker is None.")
        if len(self._previous_frames) < tracker.frame_stride:
            # Not enough history yet: report where the opponents are and admit that their
            # velocity is unknown, rather than differencing against a frame that does not
            # exist or one closer than the configured baseline. With the default stride of 1
            # this is the episode's first step only.
            clusters = tracker.detect_clusters(grid)
        else:
            reference_grid, reference_heading = self._previous_frames[0]
            clusters = tracker.track(reference_grid, grid, wrap_to_pi(heading - reference_heading))
        return self._rows_from_clusters(clusters)

    def _next_frames(
        self, grid: Optional[np.ndarray], heading: float
    ) -> Tuple[Tuple[np.ndarray, float], ...]:
        # The window holds exactly `frame_stride` frames, so its oldest entry is always the
        # one `frame_stride` steps back. Keeping only the immediately preceding frame would
        # make a stride above 1 divide a one-step displacement by several steps and
        # under-report every velocity by that factor -- the knob would silently lie instead
        # of widening the baseline.
        if grid is None or self.tracker is None:
            return ()
        # Copied, not referenced: highway-env writes each observation into the same buffer,
        # so holding the array would leave the reference frame equal to the current one and
        # difference every opponent to a standstill.
        window = self._previous_frames + ((grid.copy(), heading),)
        return window[-self.tracker.frame_stride :]

    def _rows_from_clusters(self, clusters: List[TrackedCluster]) -> np.ndarray:
        rows = np.zeros((self.max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
        nearest = sorted(clusters, key=lambda cluster: float(np.linalg.norm(cluster.centre)))
        for slot, cluster in enumerate(nearest[: self.max_tracked_agents]):
            rows[slot] = (
                1.0,
                float(cluster.centre[0]),
                float(cluster.centre[1]),
                float(cluster.velocity[0]),
                float(cluster.velocity[1]),
            )
        return rows

    def _stamp(self, particles: np.ndarray, rows: np.ndarray) -> None:
        expected = EGO_STATE_WIDTH + rows.size
        if particles.shape[1] != expected:
            raise ValueError(
                f"Particles are {particles.shape[1]} wide but max_tracked_agents="
                f"{self.max_tracked_agents} needs {expected}. The agent block is written "
                f"straight into the fixed slots, so the two must agree."
            )
        block = np.broadcast_to(rows.reshape(-1), (len(particles), rows.size))
        particles[:, EGO_STATE_WIDTH:] = block + self._jitter(len(particles), rows)

    def _jitter(self, count: int, rows: np.ndarray) -> np.ndarray:
        # Two widths, not one: the position estimate is good to about a cell while the
        # velocity estimate is quantised at 15 m/s, so a shared std would either erase the
        # position information or pretend the velocity is precise.
        noise = np.zeros((count, self.max_tracked_agents, AGENT_SLOT_WIDTH))
        present = np.flatnonzero(rows[:, AGENT_PRESENT] == 1.0)
        if len(present) == 0:
            return noise.reshape(count, -1)
        widths = (
            (AGENT_REL_X, self.agent_pose_jitter),
            (AGENT_REL_VX, self.agent_velocity_jitter),
        )
        for start, width in widths:
            if width > 0.0:
                noise[:, present, start : start + 2] = np.random.normal(
                    0.0, width, size=(count, len(present), 2)
                )
        return noise.reshape(count, -1)

    def _successor(
        self,
        particles: np.ndarray,
        belief: "WeightedParticleBelief",
        grid: Optional[np.ndarray],
        heading: float,
    ) -> "TrackedAgentsBelief":
        return TrackedAgentsBelief(
            particles=particles,
            log_weights=np.array(belief.log_weights, dtype=float, copy=True),
            observation_mode=self.observation_mode,
            max_tracked_agents=self.max_tracked_agents,
            agent_pose_jitter=self.agent_pose_jitter,
            agent_velocity_jitter=self.agent_velocity_jitter,
            tracker=self.tracker,
            resampling=belief.resampling,
            ess_factor=belief.ess_factor,
            previous_frames=self._next_frames(grid, heading),
        )


def _resolve_tracker(
    observation_mode: ObservationMode, tracker: Optional[OccupancyVelocityTracker]
) -> Optional[OccupancyVelocityTracker]:
    if tracker is not None:
        return tracker
    if observation_mode is ObservationMode.MDP:
        return None
    return OccupancyVelocityTracker()


def _mean_heading(particles: np.ndarray, weights: np.ndarray) -> float:
    # Weighted circular mean, on two counts. Circular because headings wrap, so averaging the
    # raw radians puts the mean near zero whenever the particles straddle +/-pi -- a bogus yaw
    # delta of about pi that would de-rotate the whole reference frame backwards. Weighted
    # because the filter only resamples when ESS drops, so between resamples the particle
    # cloud can be dominated by a handful of heavy particles and an unweighted mean would let
    # near-zero-probability headings steer the de-rotation.
    headings = particles[:, EGO_HEADING]
    sin_mean = float(np.sum(weights * np.sin(headings)))
    cos_mean = float(np.sum(weights * np.cos(headings)))
    return float(np.arctan2(sin_mean, cos_mean))


def _has_key(observation: Any, key: str) -> bool:
    try:
        return key in observation
    except TypeError:
        return False


def _observation_keys(observation: Any) -> Any:
    if isinstance(observation, dict):
        return sorted(observation.keys())
    return type(observation).__name__
