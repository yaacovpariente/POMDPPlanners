# SPDX-License-Identifier: MIT

"""Plain particle belief that stamps the observed agent block onto its particles.

Perception lives in the world's observation model: a
:class:`~POMDPPlanners.environments.carla_pomdp.carla_perception.carla_pipeline.CarlaPerceptionPipeline`
run inside :class:`~POMDPPlanners.environments.carla_pomdp.carla_pomdp.CarlaPOMDP` turns the raw
sensors into the *perceived* observation, so the ``agents`` channel the belief receives is
already the tracked object list. The belief therefore does no perception at all.

It still cannot be a bare particle filter, though: a weight-only update can reweight and
propagate the agents a particle was seeded with, but it can never *acquire* a vehicle that
appears mid-episode, and a slot seeded empty stays empty forever. :class:`PerceivedAgentsBelief`
closes that gap in the minimal way — after the ordinary particle-filter weight update and
resample, it replaces every particle's ego-frame agent block with the observation's ``agents``
block (plus optional per-particle pose jitter for diversity), leaving the ego block to the
filter. The perceived agents are the observation's estimate, *trusted* rather than re-filtered as
a per-particle latent.

The returned belief is itself a :class:`PerceivedAgentsBelief`, so the stamping repeats on every
step of the episode.

Note:
    The belief's ``max_tracked_agents`` must match the width of the observation's ``agents``
    block, since that block is written straight into each particle's fixed agent slots.

Classes:
    PerceivedAgentsBelief: Particle belief that stamps the observed agent block onto particles.
"""

from typing import Any

import numpy as np

from POMDPPlanners.core.belief.particle_beliefs import (
    WeightedParticleBelief,
    WeightedParticleBeliefReinvigoration,
)
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
    AGENT_SLOT_WIDTH,
    DEFAULT_MAX_TRACKED_AGENTS,
    EGO_STATE_WIDTH,
)


class PerceivedAgentsBelief(WeightedParticleBeliefReinvigoration):
    """Weighted particle belief that stamps the observation's agent block onto every particle.

    After the standard particle-filter weight update and resample, the reinvigoration step
    writes the current observation's ``agents`` block into every particle's agent slots (plus
    optional per-particle jitter), leaving the ego block to the filter. The belief holds no
    perception state — perception is the world's observation model — so a plain observation with
    a perceived ``agents`` block is all it needs.

    Attributes:
        max_tracked_agents: Number of fixed agent slots carried in each particle.
        agent_pose_jitter: Std of Gaussian noise added to each stamped agent's
            ``[rel_x, rel_y, rel_yaw, rel_speed]`` pose, for particle diversity.

    Example:
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> from POMDPPlanners.environments.carla_pomdp.carla_pomdp import (
        ...     AGENT_SLOT_WIDTH, EGO_STATE_WIDTH)
        >>> width = EGO_STATE_WIDTH + 1 * AGENT_SLOT_WIDTH
        >>> particles = [np.zeros(width) for _ in range(4)]
        >>> belief = PerceivedAgentsBelief(
        ...     particles=particles,
        ...     log_weights=np.log(np.ones(4) / 4),
        ...     max_tracked_agents=1,
        ... )
        >>> observation = {  # a perceived agent 8 m ahead
        ...     "gnss": np.zeros(2),
        ...     "agents": np.array([1.0, 8.0, 0.0, 0.0, 5.0]),
        ... }
        >>> base = WeightedParticleBelief(particles=particles, log_weights=belief.log_weights)
        >>> refreshed = belief.reinvigorate("noop", observation, None, base)
        >>> bool(np.asarray(refreshed.particles)[0, EGO_STATE_WIDTH] == 1.0)  # slot now present
        True
    """

    def __init__(
        self,
        particles: Any,
        log_weights: np.ndarray,
        max_tracked_agents: int = DEFAULT_MAX_TRACKED_AGENTS,
        agent_pose_jitter: float = 0.3,
        resampling: bool = True,
        ess_factor: float = 0.5,
    ):
        """Initialize the perceived-agents belief.

        Args:
            particles: State particles ``[ego(7) | agent slots(K*5)]``.
            log_weights: Log-weights for the particles.
            max_tracked_agents: Number of fixed agent slots carried per particle; must match the
                width of the observation's ``agents`` block.
            agent_pose_jitter: Std of Gaussian noise added to each stamped agent pose, for
                particle diversity.
            resampling: Enable automatic resampling when ESS drops. Defaults to True.
            ess_factor: Effective-sample-size threshold factor. Defaults to 0.5.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        self.max_tracked_agents = max_tracked_agents
        self.agent_pose_jitter = agent_pose_jitter

    def reinvigorate(  # type: ignore[override]
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        belief: "WeightedParticleBelief",
    ) -> "PerceivedAgentsBelief":
        """Stamp the observation's perceived agent block onto every particle."""
        del action, pomdp
        agent_rows = np.asarray(observation["agents"], dtype=float).reshape(
            self.max_tracked_agents, AGENT_SLOT_WIDTH
        )
        particles = np.array(belief.particles, dtype=float, copy=True)
        for index in range(len(particles)):
            particles[index, EGO_STATE_WIDTH:] = self._stamped_agent_block(agent_rows)
        return PerceivedAgentsBelief(
            particles=particles,
            log_weights=np.array(belief.log_weights, dtype=float, copy=True),
            max_tracked_agents=self.max_tracked_agents,
            agent_pose_jitter=self.agent_pose_jitter,
            resampling=belief.resampling,
            ess_factor=belief.ess_factor,
        )

    def _stamped_agent_block(self, agent_rows: np.ndarray) -> np.ndarray:
        """One particle's agent block: the observed rows plus per-particle pose jitter."""
        stamped = agent_rows.copy()
        present = stamped[:, 0] == 1.0
        count = int(present.sum())
        if self.agent_pose_jitter > 0.0 and count > 0:
            jitter = np.random.normal(
                0.0, self.agent_pose_jitter, size=(count, AGENT_SLOT_WIDTH - 1)
            )
            stamped[present, 1:] += jitter
        return stamped.reshape(-1)
