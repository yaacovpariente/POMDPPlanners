"""Append-only weighted particle belief with inline cumulative-weight CDF.

A drop-in alternative to
:class:`POMDPPlanners.core.belief.WeightedParticleBeliefStateUpdate` for
in-tree MCTS belief updates. Internally the belief holds two parallel
lists — particles and a cumulative-weight CDF — so:

* ``inplace_update`` is amortized O(1) (one append per list).
* ``sample`` is O(log K) via :func:`bisect.bisect_left` on the CDF.

Compare to the existing class, which does
``self.particles + [state]`` on every update — an O(particles)
reallocation that dominates per-simulation cost in POMCPOW with
moderately-sized beliefs.

Mirrors the design of Julia POMCPOW.jl's ``POWNodeBelief.dist`` field
(``CategoricalVector{Tuple{S,Float64}}``).

Trade-offs vs ``WeightedParticleBeliefStateUpdate``:

* No resampling / ESS-based reweighting — append-only.
* No ``to_unique_support_distribution`` helper.
* No native (C++) backend.
* Particles list is mutated in place; callers must not assume
  ``belief.particles`` is a stable snapshot across updates.
"""

import bisect
import random
from typing import Any, List, Optional

import numpy as np

from POMDPPlanners.core.belief.base_belief import Belief
from POMDPPlanners.core.environment import Environment


class CDFParticleBelief(Belief):
    """Particle belief with inline cumulative-weight CDF.

    Attributes:
        particles: List of state particles (mutated in place by
            :meth:`inplace_update`).
        cdf: Cumulative-weight list aligned with ``particles``;
            ``cdf[-1]`` is the running total weight.
    """

    def __init__(
        self,
        particles: Optional[list] = None,
        weights: Optional[list] = None,
    ) -> None:
        if particles is None:
            particles = []
        if weights is None:
            weights = []
        if len(particles) != len(weights):
            raise ValueError("particles and weights must have the same length")
        self.particles: list = list(particles)
        self.cdf: List[float] = []
        running = 0.0
        for w in weights:
            running += float(w)
            self.cdf.append(running)

    def push_weighted(self, particle: Any, weight: float) -> None:
        """Append ``particle`` with the given ``weight``; O(1) amortized."""
        prev = self.cdf[-1] if self.cdf else 0.0
        self.particles.append(particle)
        self.cdf.append(prev + float(weight))

    def inplace_update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> None:
        """Append ``state`` weighted by P(observation | state, action).

        Mirrors :meth:`WeightedParticleBeliefStateUpdate.inplace_update` but
        avoids the O(particles) list reallocation by extending the inline
        CDF instead.
        """
        if state is None:
            raise ValueError("state cannot be None")
        if action is None:
            raise ValueError("action cannot be None")
        if observation is None:
            raise ValueError("observation cannot be None")
        if not isinstance(pomdp, Environment):
            # Runtime guard for callers that bypass static typing.
            raise TypeError(
                "pomdp must be an instance of Environment"
            )  # pyright: ignore[reportUnreachable]
        observation_probability = pomdp.observation_model(
            next_state=state, action=action
        ).probability([observation])[0]
        weight = float(
            observation_probability.item()
            if hasattr(observation_probability, "item")
            else observation_probability
        )
        self.push_weighted(state, weight)

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "CDFParticleBelief":
        """Return a fresh belief equal to this one with ``state`` appended.

        Provided to satisfy the abstract :class:`Belief` interface; the
        in-tree MCTS path uses :meth:`inplace_update` instead, which is
        cheaper.
        """
        # Reconstruct independent particle/weight lists from the CDF.
        new_weights: List[float] = []
        prev = 0.0
        for cumulative in self.cdf:
            new_weights.append(cumulative - prev)
            prev = cumulative
        new = CDFParticleBelief(particles=list(self.particles), weights=new_weights)
        new.inplace_update(action=action, observation=observation, pomdp=pomdp, state=state)
        return new

    def sample(self) -> Any:
        """Sample a particle proportional to its weight via O(log K) bisect."""
        if not self.particles:
            raise ValueError("Cannot sample from an empty belief")
        total = self.cdf[-1]
        if total == 0.0:
            raise ValueError("Cannot sample from a belief with zero total weight")
        target = random.random() * total
        idx = bisect.bisect_left(self.cdf, target)
        if idx >= len(self.particles):
            idx = len(self.particles) - 1
        particle = self.particles[idx]
        if isinstance(particle, list):
            # Match WeightedParticleBeliefStateUpdate.sample defensive cast.
            particle = np.array(particle)
        return particle
