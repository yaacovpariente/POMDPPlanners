# SPDX-License-Identifier: MIT

"""Exact belief over Battleship fleet layouts.

Battleship's observations are deterministic, and that breaks a generic particle
filter in a way that does not announce itself. Each probe splits the particle
set into the layouts that agree with the reading and the ones that do not, and
the ones that do not have likelihood zero. After a handful of probes every
particle drawn from the prior is inconsistent; the filter's ``eps`` floor keeps
the weights finite, resampling then spreads those impossible layouts back over
the whole set, and the planner goes on searching against boards that the
observations have already ruled out. Nothing raises. The belief just stops
meaning anything.

This module avoids the problem instead of patching it. The set of legal fleet
placements is enumerated once
(:class:`~POMDPPlanners.environments.battleship_pomdp.battleship_layouts.FleetLayoutTable`),
and the belief carries the indices of the layouts still consistent with every
probe so far. Conditioning on a new reading is one boolean filter over one
column of that table, which is exact — the posterior really is "the prior
restricted to the layouts that match" — and cannot deplete, because the
consistent set is tracked rather than sampled. Particles are then redrawn
uniformly from that set on every update, so they are i.i.d. draws from the exact
posterior and every one of them is a legal fleet.

Classes:
    BattleshipBelief: The exact belief.
"""

from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.battleship_pomdp.battleship_pomdp import (
        BattleshipPOMDP,
    )


def _as_battleship(pomdp: Environment) -> "BattleshipPOMDP":
    """Narrow the base ``Environment`` the Belief API hands us to a Battleship one.

    The ``Belief`` interface types this argument as ``Environment`` and every
    caller passes whatever environment the belief is being held over, so
    narrowing the parameter itself would be a Liskov violation. The cast is
    checked once, at construction, by :meth:`BattleshipBelief.from_environment`.

    Args:
        pomdp: The environment supplied by the caller.

    Returns:
        The same object, typed as a Battleship environment.
    """
    return cast("BattleshipPOMDP", pomdp)


class BattleshipBelief(WeightedParticleBelief):
    """Uniform particle belief over the fleet layouts consistent with the history.

    Subclasses :class:`WeightedParticleBelief` rather than :class:`Belief`
    directly so that everything downstream that dispatches on belief type — the
    expected-reward helper, the terminal-belief check, the arena tree — keeps
    working unchanged. The particles and their (uniform) weights are a genuine
    weighted particle belief; only :meth:`update` is replaced, because the
    generic weight-and-resample update is the part that is wrong here.

    Attributes:
        particles: ``(n_particles, 2 * num_cells)`` ``float64`` array of states,
            drawn uniformly from the consistent layouts and sharing one probe
            half.
        log_weights: Uniform log-weights. The redraw already applies the
            posterior, so re-weighting the particles on top of it would apply
            the same evidence twice.
    """

    def __init__(
        self,
        particles: Any,
        log_weights: np.ndarray,
        resampling: bool = False,
        ess_factor: float = 0.5,
        consistent_indices: Optional[np.ndarray] = None,
        layout_key: Optional[tuple] = None,
    ):
        """Initialize the belief.

        Args:
            particles: State particles, all sharing one probe half.
            log_weights: Log-weights, one per particle.
            resampling: Kept for interface compatibility; this belief redraws
                from the exact posterior every update, so weight-based
                resampling never applies. Defaults to ``False``.
            ess_factor: Kept for interface compatibility.
            consistent_indices: Rows of the layout table still consistent with
                the history, when the caller already knows them. Defaults to
                ``None``, meaning "recompute from the particles' probe half on
                first use".
            layout_key: The board geometry ``consistent_indices`` was computed
                against, as ``(board_size, ship_lengths, allow_adjacent_ships)``.
                Defaults to ``None``, which forces a recompute on first use.
        """
        super().__init__(
            particles=particles,
            log_weights=log_weights,
            resampling=resampling,
            ess_factor=ess_factor,
        )
        # Private, so it stays out of config_id: it is derived from the
        # particles' own probe half and adds no information to the belief's
        # identity, while a 12k-element index array would dominate the hash.
        self._consistent_indices = (
            None if consistent_indices is None else np.asarray(consistent_indices, dtype=np.int64)
        )
        # Which layout table those indices index into. They are row numbers, so
        # they mean nothing against a table built for another geometry — and a
        # wrong-but-in-range row would be read as a legal board rather than
        # raising. Recording the geometry lets the cache invalidate itself
        # instead of silently answering for the wrong board.
        self._layout_key: Optional[tuple] = layout_key

    @classmethod
    def from_environment(
        cls, pomdp: Environment, n_particles: int = 100
    ) -> "BattleshipBelief":
        """Build the prior belief for ``pomdp``.

        Args:
            pomdp: A :class:`BattleshipPOMDP`.
            n_particles: How many particles to carry. Defaults to 100.

        Returns:
            A belief over every legal layout, with nothing probed.

        Raises:
            TypeError: If ``pomdp`` is not a Battleship environment.
            ValueError: If ``n_particles`` is not positive.
        """
        if not hasattr(pomdp, "layouts"):
            raise TypeError(
                f"BattleshipBelief needs a BattleshipPOMDP, got {type(pomdp).__name__}"
            )
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        env = _as_battleship(pomdp)
        indices = np.arange(env.layouts.num_layouts, dtype=np.int64)
        probed = np.zeros(env.num_cells, dtype=np.float64)
        return cls(
            particles=_draw_particles(env, indices, probed, n_particles),
            log_weights=_uniform_log_weights(n_particles),
            consistent_indices=indices,
            layout_key=_layout_key(env),
        )

    @property
    def config_id(self) -> str:
        """Identity of this belief, distinct from a generic particle belief's.

        The inherited identity is built from the particles and their weights
        alone, with no class in it. That is enough to tell two particle beliefs
        apart, but not enough to tell *this* belief from a
        :class:`WeightedParticleBelief` holding the same particles — and the
        episode result cache keys on the initial belief's ``config_id``, so the
        two would share cached episodes. Comparing the exact belief against a
        generic filter is the first thing anyone will want to do on this
        environment, and it is exactly the comparison that collision would
        silently answer from the wrong run.

        Returns:
            The inherited identity, qualified by this class.
        """
        return config_to_id(
            {"class": f"{type(self).__module__}.{type(self).__qualname__}",
             "particles": super().config_id}
        )

    def consistent_indices(self, pomdp: Environment) -> np.ndarray:
        """Rows of the layout table consistent with everything probed so far.

        Recomputed from the particles when not already known, which is what
        makes the belief safe to pickle, deepcopy or rebuild from its particles:
        the history is fully recoverable from any single particle, because all
        particles share the probe half and agree on the occupancy of every
        probed cell.

        Args:
            pomdp: The Battleship environment the belief is held over.

        Returns:
            ``int64`` array of row indices.
        """
        env = _as_battleship(pomdp)
        layout_key = _layout_key(env)
        if self._consistent_indices is None or self._layout_key != layout_key:
            reference = np.asarray(self.particles[0], dtype=np.float64)
            probed_cells = np.flatnonzero(reference[env.num_cells :] > 0.5)
            revealed = (reference[: env.num_cells][probed_cells] > 0.5).astype(np.uint8)
            self._consistent_indices = env.layouts.consistent_indices(probed_cells, revealed)
            self._layout_key = layout_key
        return self._consistent_indices

    def update(
        self,
        action: Any,
        observation: Any,
        pomdp: Environment,
        state: Optional[Any] = None,
    ) -> "BattleshipBelief":
        """Condition on one probe outcome and redraw the particles.

        Args:
            action: The probed cell index.
            observation: The hit/miss reading.
            pomdp: The Battleship environment.
            state: Unused; the true state must not leak into the belief.

        Returns:
            The exact posterior belief.

        Raises:
            ValueError: If no legal layout agrees with the history. That cannot
                happen while the belief is conditioned on readings this
                environment produced, so it means the belief and the world have
                come apart — a caller feeding it another board's observations,
                or a mismatched fleet configuration. It is raised rather than
                worked around because a belief supported on nothing is exactly
                the failure this class exists to prevent, and silently
                reinitialising it would hide the cause.
        """
        del state
        env = _as_battleship(pomdp)
        cell = int(action)
        reading = np.uint8(1 if int(observation) else 0)

        indices = self.consistent_indices(pomdp)
        # A repeat probe re-reveals a cell that is already resolved, so the
        # filter is a no-op on it rather than a second application of the same
        # evidence. Running it anyway is correct and costs one pass, so it is
        # left in place rather than special-cased.
        indices = indices[env.layouts.masks[indices, cell] == reading]
        if indices.size == 0:
            raise ValueError(
                f"no legal Battleship layout is consistent with probing cell {cell} "
                f"and observing {int(observation)}; the belief and the world disagree"
            )

        reference = np.asarray(self.particles[0], dtype=np.float64)
        probed = reference[env.num_cells :].copy()
        probed[cell] = 1.0

        n_particles = len(self.particles)
        return BattleshipBelief(
            particles=_draw_particles(env, indices, probed, n_particles),
            log_weights=_uniform_log_weights(n_particles),
            consistent_indices=indices,
            layout_key=_layout_key(env),
        )

    def occupancy_marginal(self, pomdp: Environment) -> np.ndarray:
        """Per-cell posterior probability that the cell holds a ship.

        Computed from the consistent-layout set rather than from the particles,
        so it is the exact marginal and not a Monte Carlo estimate of it. This
        is what the visualizer draws.

        Args:
            pomdp: The Battleship environment.

        Returns:
            ``(num_cells,)`` ``float64`` array of probabilities.
        """
        env = _as_battleship(pomdp)
        indices = self.consistent_indices(pomdp)
        return env.layouts.masks[indices].mean(axis=0).astype(np.float64)


def _layout_key(env: "BattleshipPOMDP") -> tuple:
    """Return the geometry that identifies ``env``'s layout table.

    Args:
        env: The Battleship environment.

    Returns:
        ``(board_size, ship_lengths, allow_adjacent_ships)``.
    """
    return (env.board_size, env.ship_lengths, env.allow_adjacent_ships)


def _uniform_log_weights(n_particles: int) -> np.ndarray:
    """Return finite, equal log-weights for ``n_particles`` particles.

    ``log(1/n)`` is the honest value, but for ``n == 1`` it is exactly zero and
    :class:`WeightedParticleBelief` rejects an all-zero weight vector. Weights
    are normalised by subtracting their maximum, so any constant works; ``-1.0``
    is used for the singleton case and nothing downstream can tell.

    Args:
        n_particles: Number of particles.

    Returns:
        ``(n_particles,)`` ``float64`` array of equal, non-zero log-weights.
    """
    if n_particles == 1:
        return np.array([-1.0], dtype=np.float64)
    return np.full(n_particles, -float(np.log(n_particles)), dtype=np.float64)


def _draw_particles(
    pomdp: "BattleshipPOMDP", indices: np.ndarray, probed: np.ndarray, n_particles: int
) -> np.ndarray:
    """Draw ``n_particles`` states uniformly from the consistent layouts.

    Drawn with replacement: the posterior is a distribution over layouts, and
    i.i.d. draws from it are what a particle belief is supposed to be. Sampling
    without replacement would refuse to represent a posterior narrower than the
    particle count — exactly the endgame, where only a few layouts remain.

    Args:
        pomdp: The Battleship environment.
        indices: Rows of the layout table to draw from.
        probed: The probe half every drawn state shares.
        n_particles: How many particles to draw.

    Returns:
        ``(n_particles, 2 * num_cells)`` ``float64`` array.
    """
    drawn = indices[np.random.randint(0, indices.size, size=n_particles)]
    particles = np.empty((n_particles, 2 * pomdp.num_cells), dtype=np.float64)
    particles[:, : pomdp.num_cells] = pomdp.layouts.masks[drawn]
    particles[:, pomdp.num_cells :] = probed
    return particles
