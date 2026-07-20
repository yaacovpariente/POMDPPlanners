# SPDX-License-Identifier: MIT

"""Batched, on-device weighted particle belief for vectorized POMDP planners.

This module provides :class:`BatchedParticleBelief`, a torch-based belief
representation that holds a *batch* of ``B`` independent weighted particle
beliefs as on-device tensors and performs every step of the
predict-reweight-resample cycle as a single batched call to a
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`.

It is the belief-side counterpart of the vectorized generative model: the
model batches the per-particle kernels (transition, observation, likelihood),
and this class batches the per-belief filtering algebra on top of them, so a
vectorized planner can propagate, reweight, and resample many beliefs at once
with no Python loop over beliefs or particles and no host/device
synchronization.

Like the model protocol it consumes, this class is opt-in and standalone: it
does not subclass :class:`~POMDPPlanners.core.belief.base_belief.Belief`,
whose scalar ``update(action, observation, pomdp)`` / ``sample() -> state``
interface is built around a single belief over NumPy states. The batched
equivalent of ``update`` is :meth:`BatchedParticleBelief.update`; the batched
equivalent of ``sample`` is :meth:`BatchedParticleBelief.sample_states`.

Conventions (shared with the vectorized model protocol):
    * ``B`` is the number of beliefs in the batch, ``Np`` the number of
      particles per belief, ``ds``/``do`` the state/observation dimensions.
    * ``particles`` is ``[B, Np, ds]`` and ``log_weights`` is ``[B, Np]``;
      both live on the model's device.
    * Actions are per-belief integer indices of shape ``[B]``; observations
      are per-belief tensors of shape ``[B, do]``.

Classes:
    BatchedParticleBelief: Batch of weighted particle beliefs on a device.
"""

import math

import torch
from torch import Tensor

from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)


class BatchedParticleBelief:
    """Batch of ``B`` weighted particle beliefs stored as on-device tensors.

    Every operation is expressed as batched torch tensor algebra plus at most
    one flattened ``[B * Np, ...]`` call into the vectorized generative model,
    so the full filtering cycle for all ``B`` beliefs runs in a constant
    number of kernels regardless of the batch size.

    Attributes:
        particles: Particle tensor of shape ``[B, Np, ds]``.
        log_weights: Log-weight tensor of shape ``[B, Np]``.
        model: The :class:`VectorizedGenerativeModel` providing the kernels.
        resampling: Whether :meth:`update` applies per-belief ESS resampling.
        ess_factor: Fraction of ``Np`` used as the ESS resampling threshold.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> class DriftModel:
        ...     @property
        ...     def device(self):
        ...         return torch.device("cpu")
        ...     def sample_next_states(self, states, actions):
        ...         return states + actions.to(states.dtype).unsqueeze(1)
        ...     def sample_observations(self, next_states, actions):
        ...         return next_states.clone()
        ...     def rewards(self, states, actions, next_states):
        ...         return torch.zeros(states.shape[0])
        ...     def terminal_mask(self, states):
        ...         return torch.zeros(states.shape[0], dtype=torch.bool)
        ...     def observation_log_probs(self, next_states, actions, observations):
        ...         return -((next_states - observations) ** 2).sum(dim=1)
        ...     def action_keys(self, actions):
        ...         return actions.to(torch.int64)
        ...     def observation_keys(self, observations):
        ...         return torch.zeros(observations.shape[0], dtype=torch.int64)
        >>> belief = BatchedParticleBelief(
        ...     particles=torch.zeros(3, 8, 2),
        ...     log_weights=torch.zeros(3, 8),
        ...     model=DriftModel(),
        ... )
        >>> actions = torch.ones(3, dtype=torch.int64)
        >>> observations = torch.ones(3, 2)
        >>> updated = belief.update(actions, observations)
        >>> updated.particles.shape
        torch.Size([3, 8, 2])
        >>> updated.sample_states(4).shape
        torch.Size([3, 4, 2])
    """

    def __init__(
        self,
        particles: Tensor,
        log_weights: Tensor,
        model: VectorizedGenerativeModel,
        resampling: bool = False,
        ess_factor: float = 0.5,
    ):
        """Initialize a batched particle belief.

        Args:
            particles: Tensor of shape ``[B, Np, ds]``.
            log_weights: Tensor of shape ``[B, Np]``.
            model: Vectorized generative model providing the batched kernels.
            resampling: Enable per-belief ESS-based resampling inside
                :meth:`update`. Defaults to False.
            ess_factor: ESS threshold as a fraction of ``Np``. Defaults to 0.5.

        Raises:
            ValueError: If shapes are inconsistent or log_weights contains
                NaN or +inf entries.
        """
        self._validate_init_args(particles, log_weights)
        self.particles = particles.to(model.device)
        self.log_weights = log_weights.to(model.device)
        self.model = model
        self.resampling = resampling
        self.ess_factor = ess_factor
        self.ess_threshold = particles.shape[1] * ess_factor

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_root(
        cls,
        root_particles: Tensor,
        model: VectorizedGenerativeModel,
        batch_size: int = 1,
        resampling: bool = False,
        ess_factor: float = 0.5,
    ) -> "BatchedParticleBelief":
        """Build a batch of identical uniform-weight beliefs from one root.

        Args:
            root_particles: Tensor of shape ``[Np, ds]`` holding one belief's
                particles (e.g. the planner's root belief).
            model: Vectorized generative model providing the batched kernels.
            batch_size: Number of belief copies ``B``. Defaults to 1.
            resampling: Passed through to the constructor. Defaults to False.
            ess_factor: Passed through to the constructor. Defaults to 0.5.

        Returns:
            A :class:`BatchedParticleBelief` with ``[B, Np, ds]`` particles
            and uniform log-weights.

        Raises:
            ValueError: If root_particles is not 2-D or batch_size < 1.
        """
        if root_particles.dim() != 2:
            raise ValueError(f"root_particles must be 2-D [Np, ds], got {root_particles.dim()}-D")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        particles = root_particles.to(model.device).unsqueeze(0).repeat(batch_size, 1, 1)
        num_particles = root_particles.shape[0]
        log_weights = torch.full(
            (batch_size, num_particles), -math.log(num_particles), device=model.device
        )
        return cls(particles, log_weights, model, resampling=resampling, ess_factor=ess_factor)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def propagate(self, actions: Tensor) -> "BatchedParticleBelief":
        """Push every particle through the transition model (predict step).

        Args:
            actions: ``[B]`` integer action indices, one per belief.

        Returns:
            A new belief whose particles are the sampled next states; the
            log-weights are carried over unchanged.
        """
        self._validate_actions(actions)
        next_flat = self.model.sample_next_states(
            self._flat_particles(), self._per_particle_actions(actions)
        )
        next_particles = next_flat.reshape(self.batch_size, self.num_particles, -1)
        return self._derived(next_particles, self.log_weights.clone())

    def reweight(self, actions: Tensor, observations: Tensor) -> "BatchedParticleBelief":
        """Add each particle's observation log-likelihood to its log-weight.

        Call on an already-propagated belief: the current particles are
        treated as the post-transition states that emitted the observation.

        Args:
            actions: ``[B]`` integer action indices, one per belief.
            observations: ``[B, do]`` received observations, one per belief.

        Returns:
            A new belief with the same particles and updated log-weights.
        """
        self._validate_actions(actions)
        self._validate_observations(observations)
        flat_observations = observations.to(self.device).repeat_interleave(
            self.num_particles, dim=0
        )
        log_probs = self.model.observation_log_probs(
            self._flat_particles(), self._per_particle_actions(actions), flat_observations
        )
        next_log_weights = self.log_weights + log_probs.reshape(self.batch_size, self.num_particles)
        return self._derived(self.particles, next_log_weights)

    def update(self, actions: Tensor, observations: Tensor) -> "BatchedParticleBelief":
        """Full Bayesian belief update: propagate, reweight, and resample.

        This is the batched equivalent of
        :meth:`POMDPPlanners.core.belief.base_belief.Belief.update`. The
        resample step runs only when the belief was constructed with
        ``resampling=True``, and then only for the rows whose effective
        sample size fell below ``ess_factor * Np``.

        Args:
            actions: ``[B]`` integer action indices, one per belief.
            observations: ``[B, do]`` received observations, one per belief.

        Returns:
            A new :class:`BatchedParticleBelief` reflecting the posterior.
        """
        posterior = self.propagate(actions).reweight(actions, observations)
        if self.resampling:
            return posterior.resample_degenerate_rows()
        return posterior

    def sample_states(self, num_samples: int) -> Tensor:
        """Sample states from every belief in the batch.

        Args:
            num_samples: Number of states to draw per belief.

        Returns:
            ``[B, num_samples, ds]`` states drawn with replacement according
            to each belief's normalized weights.
        """
        indices = torch.multinomial(self.normalized_weights, num_samples, replacement=True)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.state_dim)
        return torch.gather(self.particles, 1, gather_indices)

    def sample_observations(self, actions: Tensor) -> Tensor:
        """Sample one observation per particle from the observation model.

        Call on an already-propagated belief: the current particles are
        treated as the post-transition states that emit the observations.

        Args:
            actions: ``[B]`` integer action indices, one per belief.

        Returns:
            ``[B, Np, do]`` sampled observations.
        """
        self._validate_actions(actions)
        flat_observations = self.model.sample_observations(
            self._flat_particles(), self._per_particle_actions(actions)
        )
        return flat_observations.reshape(self.batch_size, self.num_particles, -1)

    def resample(self) -> "BatchedParticleBelief":
        """Systematically resample every belief back to uniform weights.

        Returns:
            A new belief whose particles are drawn (per belief, with low
            variance) according to the current weights and whose log-weights
            are uniform.
        """
        indices = self._systematic_resample_indices(self.normalized_weights)
        return self._derived(self._gather_particles(indices), self._uniform_log_weights())

    def resample_degenerate_rows(self) -> "BatchedParticleBelief":
        """Resample only the beliefs whose ESS fell below the threshold.

        The selection is branchless — rows at or above ``ess_factor * Np``
        keep their particles and weights via a batched ``where`` — so no
        host/device synchronization is needed to decide whether any row
        resamples.

        Returns:
            A new belief in which degenerate rows are systematically
            resampled to uniform weights and healthy rows are unchanged.
        """
        needs_resample = (self.effective_sample_size() < self.ess_threshold).unsqueeze(1)
        resampled_indices = self._systematic_resample_indices(self.normalized_weights)
        identity_indices = (
            torch.arange(self.num_particles, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )
        indices = torch.where(needs_resample, resampled_indices, identity_indices)
        log_weights = torch.where(needs_resample, self._uniform_log_weights(), self.log_weights)
        return self._derived(self._gather_particles(indices), log_weights)

    def effective_sample_size(self) -> Tensor:
        """Return each belief's effective sample size ``1 / sum(w_i^2)``.

        Returns:
            ``[B]`` effective sample sizes, in ``(0, Np]``.
        """
        weights = self.normalized_weights
        return 1.0 / (weights**2).sum(dim=1)

    @property
    def normalized_weights(self) -> Tensor:
        """Per-belief probability weights of shape ``[B, Np]``.

        Rows whose log-weights are all ``-inf`` (every particle assigns zero
        likelihood) carry no information and fall back to uniform weights.
        """
        finite_rows = torch.isfinite(self.log_weights).any(dim=1, keepdim=True)
        safe_log_weights = torch.where(
            finite_rows, self.log_weights, torch.zeros_like(self.log_weights)
        )
        return torch.softmax(safe_log_weights, dim=1)

    @property
    def batch_size(self) -> int:
        """Number of beliefs ``B`` in the batch."""
        return self.particles.shape[0]

    @property
    def num_particles(self) -> int:
        """Number of particles ``Np`` per belief."""
        return self.particles.shape[1]

    @property
    def state_dim(self) -> int:
        """State dimensionality ``ds``."""
        return self.particles.shape[2]

    @property
    def device(self) -> torch.device:
        """Device the belief tensors live on."""
        return self.model.device

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_init_args(particles: Tensor, log_weights: Tensor) -> None:
        if particles.dim() != 3:
            raise ValueError(f"particles must be 3-D [B, Np, ds], got {particles.dim()}-D")
        if log_weights.dim() != 2:
            raise ValueError(f"log_weights must be 2-D [B, Np], got {log_weights.dim()}-D")
        if particles.shape[:2] != log_weights.shape:
            raise ValueError(
                f"particles batch/particle shape {tuple(particles.shape[:2])} does not "
                f"match log_weights shape {tuple(log_weights.shape)}"
            )
        if bool(torch.isnan(log_weights).any()) or bool(torch.isposinf(log_weights).any()):
            raise ValueError("log_weights must not contain NaN or +inf values")

    def _validate_actions(self, actions: Tensor) -> None:
        if actions.dim() != 1 or actions.shape[0] != self.batch_size:
            raise ValueError(
                f"actions must be 1-D with one entry per belief "
                f"[{self.batch_size}], got shape {tuple(actions.shape)}"
            )

    def _validate_observations(self, observations: Tensor) -> None:
        if observations.dim() != 2 or observations.shape[0] != self.batch_size:
            raise ValueError(
                f"observations must be 2-D with one row per belief "
                f"[{self.batch_size}, do], got shape {tuple(observations.shape)}"
            )

    def _flat_particles(self) -> Tensor:
        return self.particles.reshape(self.batch_size * self.num_particles, self.state_dim)

    def _per_particle_actions(self, actions: Tensor) -> Tensor:
        return actions.to(self.device).repeat_interleave(self.num_particles)

    def _derived(self, particles: Tensor, log_weights: Tensor) -> "BatchedParticleBelief":
        return BatchedParticleBelief(
            particles=particles,
            log_weights=log_weights,
            model=self.model,
            resampling=self.resampling,
            ess_factor=self.ess_factor,
        )

    def _uniform_log_weights(self) -> Tensor:
        return torch.full_like(self.log_weights, -math.log(self.num_particles))

    def _systematic_resample_indices(self, weights: Tensor) -> Tensor:
        cumulative = torch.cumsum(weights, dim=1)
        offsets = torch.rand(self.batch_size, 1, device=self.device)
        strata = torch.arange(self.num_particles, device=self.device).unsqueeze(0)
        positions = (strata + offsets) / self.num_particles
        indices = torch.searchsorted(cumulative.contiguous(), positions)
        return indices.clamp_(max=self.num_particles - 1)

    def _gather_particles(self, indices: Tensor) -> Tensor:
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, self.state_dim)
        return torch.gather(self.particles, 1, gather_indices)
