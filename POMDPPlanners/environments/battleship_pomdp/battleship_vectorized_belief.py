# SPDX-License-Identifier: MIT

"""Vectorized particle belief over Battleship fleet layouts.

The batched twin of
:class:`~POMDPPlanners.environments.battleship_pomdp.battleship_belief.BattleshipBelief`:
same posterior, expressed through the
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
interface so vectorized planners can hold it.

The updater is the environment's own model, batched and nothing more: a probe
sets one probe bit, and a reading has likelihood 1 on the layouts whose probed
cell agrees with it and 0 on the rest. Those zeroes are the whole problem with
a particle filter here. Battleship's sensor is deterministic, so every reading
splits the particles into the ones that agree and the ones that are *impossible*,
and after a handful of probes nothing drawn from the prior is still consistent.
A generic filter floors the weights, resamples impossible layouts back over the
whole set, and goes on planning against boards the observations already ruled
out. Nothing raises; the belief just stops meaning anything.

So the reweighting here is the generic one, and the *resampling* is not. Instead
of drawing from the surviving particles, :meth:`BattleshipVectorizedWeightedParticleBelief.update`
redraws from the rows of the enumerated layout table that agree with every probe
so far. That set is tracked rather than sampled, so it cannot deplete, and a
uniform draw from it is an i.i.d. draw from the exact posterior — the same
posterior the scalar belief carries, and every particle is a legal fleet.

Classes:
    BattleshipVectorizedUpdater: The batched transition and likelihood.
    BattleshipVectorizedWeightedParticleBelief: The belief that redraws exactly.

Functions:
    create_battleship_belief: Factory used by the top-level belief factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from POMDPPlanners.core.belief.belief_utils import get_initial_belief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.utils.belief_factory import BeliefType
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.core.belief.base_belief import Belief
    from POMDPPlanners.core.environment import Environment
    from POMDPPlanners.environments.battleship_pomdp.battleship_layouts import (
        FleetLayoutTable,
    )
    from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import (
        BattleshipPOMDP,
    )


class BattleshipVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Batched transition and observation likelihood for Battleship.

    A state is ``[occupancy | probed]``, each half ``num_cells`` long. Probing
    cell ``a`` sets ``probed[a]`` and touches nothing else, and the sensor
    reports ``occupancy[a]`` exactly, so both batch paths are one column of the
    particle array.

    Attributes:
        num_cells: Cells on the board, so half the state width.
        layouts: The enumerated legal fleet placements, shared with the
            environment and used for the exact redraw.
        board_size: Board side length, carried for the identity.
        ship_lengths: The fleet, carried for the identity.
        allow_adjacent_ships: Whether ships may touch, carried for the identity.
    """

    def __init__(
        self,
        num_cells: int,
        layouts: "FleetLayoutTable",
        board_size: int,
        ship_lengths: Any,
        allow_adjacent_ships: bool,
    ) -> None:
        """Initialize the updater.

        Args:
            num_cells: Cells on the board.
            layouts: The environment's layout table.
            board_size: Board side length.
            ship_lengths: The fleet's ship lengths.
            allow_adjacent_ships: Whether ships may touch.
        """
        self.num_cells = int(num_cells)
        self.layouts = layouts
        self.board_size = int(board_size)
        self.ship_lengths = tuple(int(length) for length in ship_lengths)
        self.allow_adjacent_ships = bool(allow_adjacent_ships)

    @classmethod
    def from_environment(cls, env: "BattleshipPOMDP") -> "BattleshipVectorizedUpdater":
        """Construct an updater from a :class:`BattleshipPOMDP` instance."""
        return cls(
            num_cells=env.num_cells,
            layouts=env.layouts,
            board_size=env.board_size,
            ship_lengths=env.ship_lengths,
            allow_adjacent_ships=env.allow_adjacent_ships,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        """Mark the probed cell on every particle.

        Args:
            particles: ``(N, 2 * num_cells)`` particles.
            action: The probed cell index.

        Returns:
            ``(N, 2 * num_cells)`` successors. The fleet never moves, so only the
            probe half changes.
        """
        next_particles = np.array(particles, dtype=np.float64, copy=True)
        if next_particles.size == 0:
            return next_particles
        next_particles[:, self.num_cells + int(action)] = 1.0
        return next_particles

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: Any,
        observation: Any,
    ) -> np.ndarray:
        """Score the reading against every particle's board.

        Args:
            next_particles: ``(N, 2 * num_cells)`` transitioned particles.
            action: The probed cell index.
            observation: ``1`` for a hit, ``0`` for a miss.

        Returns:
            ``(N,)`` log-likelihoods: ``0.0`` where the particle's board agrees
            with the reading and ``-inf`` where it does not. The zero
            likelihoods are exact -- the sensor cannot lie -- and
            :class:`BattleshipVectorizedWeightedParticleBelief` is built to
            survive them.
        """
        particles = np.asarray(next_particles, dtype=np.float64)
        hit = bool(int(np.asarray(observation).ravel()[0]))
        occupied = particles[:, int(action)] > 0.5
        return np.where(occupied == hit, 0.0, -np.inf)

    @property
    def config_id(self) -> str:
        """Return a deterministic identifier for this updater configuration.

        The layout table is a pure function of the three geometry parameters, so
        hashing them is hashing it -- and hashing the table itself would put tens
        of thousands of rows through the identity for no added information.
        """
        return config_to_id(
            {
                "class": "BattleshipVectorizedUpdater",
                "board_size": self.board_size,
                "ship_lengths": list(self.ship_lengths),
                "allow_adjacent_ships": self.allow_adjacent_ships,
            }
        )

    # ------------------------------------------------------------------
    # Exact-posterior support
    # ------------------------------------------------------------------

    def consistent_indices(self, probe_half: np.ndarray) -> np.ndarray:
        """Layout rows that agree with everything one particle has probed.

        The history is fully recoverable from any single particle of a belief
        this class produced, because they all share the probe half and agree on
        the occupancy of every probed cell.

        Args:
            probe_half: One particle's full ``(2 * num_cells,)`` state.

        Returns:
            ``int64`` row indices into the layout table.
        """
        reference = np.asarray(probe_half, dtype=np.float64)
        probed_cells = np.flatnonzero(reference[self.num_cells :] > 0.5)
        revealed = (reference[: self.num_cells][probed_cells] > 0.5).astype(np.uint8)
        return self.layouts.consistent_indices(probed_cells, revealed)

    def draw_particles(
        self, indices: np.ndarray, probed: np.ndarray, n_particles: int
    ) -> np.ndarray:
        """Draw states uniformly from the given layout rows.

        Drawn with replacement, because the posterior is a distribution over
        layouts and i.i.d. draws from it are what a particle belief is. Without
        replacement it could not represent a posterior narrower than the particle
        count -- which is exactly the endgame.

        Args:
            indices: Layout rows to draw from.
            probed: The ``(num_cells,)`` probe half every drawn state shares.
            n_particles: How many particles to draw.

        Returns:
            ``(n_particles, 2 * num_cells)`` ``float64`` particles.
        """
        drawn = indices[np.random.randint(0, indices.size, size=n_particles)]
        particles = np.empty((n_particles, 2 * self.num_cells), dtype=np.float64)
        particles[:, : self.num_cells] = self.layouts.masks[drawn]
        particles[:, self.num_cells :] = probed
        return particles


class BattleshipVectorizedWeightedParticleBelief(VectorizedWeightedParticleBelief):
    """Vectorized particle belief that redraws from the exact posterior.

    Attributes:
        updater: The :class:`BattleshipVectorizedUpdater` the belief updates
            through, and the owner of the layout table it redraws from.
    """

    def __init__(
        self,
        particles: np.ndarray,
        log_weights: np.ndarray,
        updater: BattleshipVectorizedUpdater,
        resampling: bool = True,
        ess_factor: float = 0.5,
        consistent_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the belief.

        Args:
            particles: ``(N, 2 * num_cells)`` particles sharing one probe half.
            log_weights: One log-weight per particle.
            updater: The batched updater.
            resampling: Accepted for interface compatibility. The redraw is
                unconditional, so this does not gate it. Defaults to ``True``.
            ess_factor: Accepted for interface compatibility.
            consistent_indices: The layout rows the particles were drawn from,
                when the caller already knows them. Defaults to ``None``, meaning
                recompute from the particles on first use.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            updater=updater,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        # Private, so it stays out of ``config_id``: it is derived from the
        # particles' own probe half and adds nothing to the belief's identity,
        # while a 12k-element index array would dominate the hash.
        self._consistent_indices = (
            None if consistent_indices is None else np.asarray(consistent_indices, dtype=np.int64)
        )

    @classmethod
    def from_environment(
        cls, env: "BattleshipPOMDP", n_particles: int = 200
    ) -> "BattleshipVectorizedWeightedParticleBelief":
        """Build the prior belief for ``env``.

        Args:
            env: The Battleship environment.
            n_particles: How many particles to carry. Defaults to 200.

        Returns:
            A belief over every legal layout, with nothing probed.

        Raises:
            ValueError: If ``n_particles`` is not positive.
        """
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        updater = BattleshipVectorizedUpdater.from_environment(env)
        indices = np.arange(updater.layouts.num_layouts, dtype=np.int64)
        probed = np.zeros(updater.num_cells, dtype=np.float64)
        return cls(
            particles=updater.draw_particles(indices, probed, n_particles),
            log_weights=_uniform_log_weights(n_particles),
            updater=updater,
            consistent_indices=indices,
        )

    @property
    def consistent_indices(self) -> np.ndarray:
        """Layout rows still consistent with every probe so far.

        Recomputed from the particles when not already known, which is what makes
        the belief safe to pickle, deepcopy or rebuild from its particles alone.
        """
        if self._consistent_indices is None:
            self._consistent_indices = self.updater.consistent_indices(self.particles[0])
        return self._consistent_indices

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Optional["Environment"] = None,
        state: Optional[Any] = None,
    ) -> "BattleshipVectorizedWeightedParticleBelief":
        """Condition on one probe outcome and redraw the particles.

        Args:
            action: The probed cell index.
            observation: The hit/miss reading.
            pomdp: Unused; the updater owns everything the update needs.
            state: Ignored, so the true state cannot leak into the belief.

        Returns:
            The exact posterior belief.

        Raises:
            ValueError: If no legal layout agrees with the history. That cannot
                happen while the belief is conditioned on readings this
                environment produced, so it means the belief and the world have
                come apart -- another board's observations, or a mismatched
                fleet. Raised rather than worked around: a belief supported on
                nothing is the failure this class exists to prevent, and quietly
                reinitialising it would hide the cause.
        """
        del pomdp, state
        cell = int(action)
        reading = np.uint8(1 if int(observation) else 0)

        next_particles = self.updater.batch_transition(self.particles, action)

        indices = self.consistent_indices
        # A repeat probe re-reveals a cell that is already resolved, so this is a
        # no-op on it rather than a second application of the same evidence.
        indices = indices[self.updater.layouts.masks[indices, cell] == reading]
        if indices.size == 0:
            raise ValueError(
                f"no legal Battleship layout is consistent with probing cell {cell} "
                f"and observing {int(observation)}; the belief and the world disagree"
            )
        probed = np.asarray(next_particles[0], dtype=np.float64)[self.updater.num_cells :].copy()
        n_particles = self.particles.shape[0]
        return BattleshipVectorizedWeightedParticleBelief(
            particles=self.updater.draw_particles(indices, probed, n_particles),
            log_weights=_uniform_log_weights(n_particles),
            updater=self.updater,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
            consistent_indices=indices,
        )

    def occupancy_marginal(self, pomdp: Optional["Environment"] = None) -> np.ndarray:
        """Per-cell posterior probability that the cell holds a ship.

        Computed from the consistent-layout set rather than from the particles,
        so it is the exact marginal and not a Monte Carlo estimate of it. This is
        what the visualizer draws.

        Args:
            pomdp: Unused; accepted so the visualizer can call this the same way
                it calls the scalar belief's.

        Returns:
            ``(num_cells,)`` ``float64`` probabilities.
        """
        del pomdp
        return self.updater.layouts.masks[self.consistent_indices].mean(axis=0).astype(np.float64)


def create_battleship_belief(
    env: "BattleshipPOMDP",
    belief_type: BeliefType = BeliefType.VECTORIZED_PARTICLE,
    n_particles: int = 200,
    **kwargs: Any,
) -> "Belief":
    """Create a belief for the Battleship POMDP.

    Args:
        env: The environment.
        belief_type: ``PARTICLE`` or ``VECTORIZED_PARTICLE``.
        n_particles: Number of particles. Defaults to 200.
        **kwargs: Reserved for future use.

    Returns:
        A configured belief.

    Raises:
        ValueError: If *belief_type* is not supported.
    """
    del kwargs
    if belief_type == BeliefType.PARTICLE:
        return get_initial_belief(env, n_particles)
    if belief_type == BeliefType.VECTORIZED_PARTICLE:
        return BattleshipVectorizedWeightedParticleBelief.from_environment(env, n_particles)
    raise ValueError(f"BattleshipPOMDP does not support belief type {belief_type!r}")


def _uniform_log_weights(n_particles: int) -> np.ndarray:
    """Equal log-weights for ``n_particles`` particles.

    The redraw already applies the posterior, so re-weighting the particles on
    top of it would apply the same evidence twice.

    Args:
        n_particles: Number of particles.

    Returns:
        ``(n_particles,)`` ``float64`` equal log-weights.
    """
    return np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)


__all__ = [
    "BattleshipVectorizedUpdater",
    "BattleshipVectorizedWeightedParticleBelief",
    "create_battleship_belief",
]
