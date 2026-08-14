# SPDX-License-Identifier: MIT

"""Particle belief that stamps this step's detections onto its particles.

The planner's state carries ``[present, rel_x, rel_y, rel_vx, rel_vy]`` per agent slot, and
those velocity terms are what let a rollout predict a closing opponent instead of
re-discovering it on the step it arrives. :class:`TrackedAgentsBelief` fills those slots from
the observation.

**What a detection gives, and what it does not.** A detection reports a vehicle's relative
position and its *whole* relative velocity, so both components go straight into the slot and
nothing has to be resolved along a line of sight. ``agent_pose_jitter`` and
``agent_velocity_jitter`` then cover the measurement noise on what was reported, and nothing
more — the arm's hidden state is a vehicle that produced no detection at all, not a component
of one that did.

That is why a slot with no detection behind it is left empty rather than jittered: there is
no reading to spread around. Recovering such a vehicle is the weight update's job, through
the model's prediction that a slot inside sensor range *should* have been reported.

As with
:class:`~POMDPPlanners.environments.carla_pomdp.carla_belief.PerceivedAgentsBelief`, the
agent block is *trusted* rather than re-filtered as a per-particle latent: a weight-only
filter can propagate the opponents a particle was seeded with but can never acquire one that
appears mid-episode, and a slot seeded empty would stay empty for the whole episode. The ego
block is left to the particle filter, which is where the actual Bayesian work happens.

**The observed ego speed is deliberately not stamped**, even though the same argument might
seem to apply to it. Stamping the reading into every particle and then scoring the particles
on agreeing with it is double-counting, and it would flatten the speed spread to zero — which
makes the likelihood term identical across particles and therefore worthless. The model's
process noise spreads the particles across the ego block, the speed slot included, so the
speedometer has something real to discriminate on and does its work through the weights
instead. Unlike the opponents, the ego cannot fail to be acquired, so there is nothing here
that a weight-only filter cannot reach.

The same class serves the MDP arm of the matched pair, where the observation already contains
the agent rows. Running one belief across both arms is deliberate: it keeps the arms
differing in the observation alone, which is the whole point of the comparison.

Classes:
    TrackedAgentsBelief: Particle belief that stamps observed agent rows onto particles.
"""

import warnings
from typing import Any

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import (
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
    AGENT_PRESENT,
    AGENT_REL_VX,
    AGENT_REL_X,
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    DETECTION_PRESENT,
    DETECTION_REL_VX,
    DETECTION_REL_X,
    DETECTION_SLOT_WIDTH,
    EGO_STATE_WIDTH,
    ObservationMode,
)
from POMDPPlanners.utils.config_to_id import config_to_id

DETECTIONS_KEY = "detections"
AGENTS_KEY = "agents"


class TrackedAgentsBelief(WeightedParticleBeliefReinvigoration):
    """Weighted particle belief that stamps observed opponent rows onto every particle.

    After the ordinary particle-filter weight update and resample, the reinvigoration step
    derives this step's agent rows — from the detections in ``POMDP`` mode, straight off the
    observation in ``MDP`` mode — and writes them into every particle's agent slots with
    per-particle jitter. The returned belief is another ``TrackedAgentsBelief``, so the
    stamping repeats every step.

    Note:
        In ``POMDP`` mode a detection carries the vehicle's full relative velocity, so both
        components are stamped as reported. A vehicle crossing the ego's path at 6 m/s
        directly abeam is stamped at 6 m/s across, and the particles agree about where it
        will be next step to within the jitter. What they cannot agree about is a vehicle no
        detection mentions — beyond ``max_detection_range_m``, or behind a closer one — and
        that is the only inference this arm asks for.

    Note:
        What is recovered is velocity **relative to a moving, turning car**, not the
        opponents' velocity over the ground. That matches what the state's agent slots hold,
        so nothing is inconsistent — but a rollout treating those numbers as absolute will be
        wrong by the ego's own motion.

    Attributes:
        observation_mode: Which arm of the matched pair this belief is reading.
        max_tracked_agents: Number of fixed agent slots carried in each particle.
        agent_pose_jitter: Std of the Gaussian noise added to each stamped ``(rel_x, rel_y)``.
        agent_velocity_jitter: Std of the Gaussian noise added to each stamped
            ``(rel_vx, rel_vy)``.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
        ...     EGO_STATE_WIDTH)
        >>> width = EGO_STATE_WIDTH + 1 * 5
        >>> particles = np.zeros((4, width))
        >>> belief = TrackedAgentsBelief(  # jitter off so the stamped row is exact
        ...     particles=particles,
        ...     log_weights=np.log(np.ones(4) / 4),
        ...     max_tracked_agents=1,
        ...     agent_pose_jitter=0.0,
        ...     agent_velocity_jitter=0.0,
        ... )
        >>> # 10 m ahead, closing at 2 m/s and crossing left at 3 m/s
        >>> detections = np.array([[1.0, 10.0, 0.0, -2.0, 3.0]])
        >>> base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
        >>> refreshed = belief.reinvigorate("noop", {"detections": detections}, None, base)
        >>> np.asarray(refreshed.particles)[0, EGO_STATE_WIDTH:]
        array([ 1., 10.,  0., -2.,  3.])
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
        resampling: bool = True,
        ess_factor: float = 0.5,
    ):
        """Initialize the tracked-agents belief.

        Args:
            particles: State particles ``[ego(7) | agent slots(K*5)]``.
            log_weights: Log-weights for the particles.
            observation_mode: Which arm of the matched pair the observation comes from.
                Defaults to ``ObservationMode.POMDP``.
            max_tracked_agents: Number of fixed agent slots carried per particle. Defaults
                to 4. A reading carrying more detections than there are slots warns and drops
                the furthest, because those become empty road to the planner rather than an
                error it can see.
            agent_pose_jitter: Std of the noise added to each stamped ``(rel_x, rel_y)``, in
                metres. Defaults to 0.5, matching the radar's own position noise.
            agent_velocity_jitter: Std of the noise added to each stamped
                ``(rel_vx, rel_vy)``, in m/s. Defaults to 1.0. Both components are now
                measured, so this covers observation noise plus slack for the
                constant-velocity drift the model propagates them with — it is wider than
                the sensor's own velocity width on purpose, and it is a tuning knob rather
                than a sensor constant.
            resampling: Enable automatic resampling when ESS drops. Defaults to True.
            ess_factor: Effective-sample-size threshold factor. Defaults to 0.5.
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

    @property
    def config_id(self) -> str:
        """Deterministic identifier covering the particles and the stamping configuration.

        Two beliefs holding identical particles but jittering them differently behave
        differently on the next step, so the jitter widths belong in the identity. The
        observation does not: it is one step's reading, not configuration, and folding it in
        would give an identity that changes every frame and defeats the caching the id exists
        for.
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
        """Stamp this step's observed agent rows onto every particle."""
        del action, pomdp
        particles = np.array(belief.particles, dtype=float, copy=True)
        if self.observation_mode is ObservationMode.MDP:
            rows = self._mdp_rows(observation)
        else:
            rows = self._detection_rows(observation)
        self._stamp(particles, rows)
        return self._successor(particles, belief)

    def _mdp_rows(self, observation: Any) -> np.ndarray:
        if not _has_key(observation, AGENTS_KEY):
            raise ValueError(
                f"ObservationMode.MDP expects an '{AGENTS_KEY}' block in the observation, got "
                f"keys {_observation_keys(observation)}. Silently falling back to the "
                f"detections would leave the MDP arm reading a degraded observation."
            )
        rows = np.asarray(observation[AGENTS_KEY], dtype=float)
        expected = self.max_tracked_agents * AGENT_SLOT_WIDTH
        if rows.size != expected:
            raise ValueError(
                f"Observed '{AGENTS_KEY}' block has {rows.size} values but "
                f"max_tracked_agents={self.max_tracked_agents} needs {expected}."
            )
        return rows.reshape(self.max_tracked_agents, AGENT_SLOT_WIDTH)

    def _detection_rows(self, observation: Any) -> np.ndarray:
        """Turn ``(D, 5)`` detections into ``(K, 5)`` agent slots, nearest first.

        A detection and an agent slot now carry the same four numbers in the same frame, so
        this is a copy rather than a reconstruction: the relative position and both
        components of relative velocity go straight across, and only the presence flag has to
        be filled in.
        """
        detections = self._require_detections(observation)
        reported = detections[detections[:, DETECTION_PRESENT] > 0.5]
        # Nearest-K is the policy, and dropping the rest is intended -- but doing it quietly
        # is not. In traffic heavier than anything this has been run against the planner would
        # simply stop seeing the overflow and treat that road as empty. That is a false
        # negative in the one channel the POMDP arm has, and it looks exactly like success.
        if len(reported) > self.max_tracked_agents:
            warnings.warn(
                f"Observed {len(reported)} detections but only max_tracked_agents="
                f"{self.max_tracked_agents} slots are available; the "
                f"{len(reported) - self.max_tracked_agents} furthest were dropped and the "
                f"planner will treat that road as empty. Raise max_tracked_agents.",
                UserWarning,
                stacklevel=3,
            )
        rows = np.zeros((self.max_tracked_agents, AGENT_SLOT_WIDTH), dtype=float)
        kept = reported[: self.max_tracked_agents]
        filled = len(kept)
        rows[:filled, AGENT_PRESENT] = 1.0
        rows[:filled, AGENT_REL_X : AGENT_REL_X + 2] = kept[
            :, DETECTION_REL_X : DETECTION_REL_X + 2
        ]
        rows[:filled, AGENT_REL_VX : AGENT_REL_VX + 2] = kept[
            :, DETECTION_REL_VX : DETECTION_REL_VX + 2
        ]
        return rows

    def _require_detections(self, observation: Any) -> np.ndarray:
        if not _has_key(observation, DETECTIONS_KEY):
            raise ValueError(
                f"ObservationMode.POMDP expects a '{DETECTIONS_KEY}' block in the "
                f"observation, got keys {_observation_keys(observation)}. Degrading to zero "
                f"agent rows here would produce a planner that sees no traffic at all and "
                f"still looks like it works."
            )
        return np.asarray(observation[DETECTIONS_KEY], dtype=float).reshape(
            -1, DETECTION_SLOT_WIDTH
        )

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
        # Two widths, not one: the two blocks are measured to different accuracies and
        # propagated with different amounts of model error, so a shared std would either
        # erase the position information or under-spread the velocity.
        noise = np.zeros((count, self.max_tracked_agents, AGENT_SLOT_WIDTH))
        # Thresholded, not compared to exactly 1.0: in MDP mode these rows come from the
        # observation rather than from this package, and a presence flag that arrived as
        # 0.999 would silently leave a real vehicle unjittered.
        present = np.flatnonzero(rows[:, AGENT_PRESENT] > 0.5)
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
        self, particles: np.ndarray, belief: "WeightedParticleBelief"
    ) -> "TrackedAgentsBelief":
        return TrackedAgentsBelief(
            particles=particles,
            log_weights=np.array(belief.log_weights, dtype=float, copy=True),
            observation_mode=self.observation_mode,
            max_tracked_agents=self.max_tracked_agents,
            agent_pose_jitter=self.agent_pose_jitter,
            agent_velocity_jitter=self.agent_velocity_jitter,
            resampling=belief.resampling,
            ess_factor=belief.ess_factor,
        )


def _has_key(observation: Any, key: str) -> bool:
    try:
        return key in observation
    except TypeError:
        return False


def _observation_keys(observation: Any) -> Any:
    if isinstance(observation, dict):
        return sorted(observation.keys())
    return type(observation).__name__
