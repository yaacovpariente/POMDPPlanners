# SPDX-License-Identifier: MIT

"""Tests for the batched, on-device weighted particle belief.

The suite drives :class:`BatchedParticleBelief` with a deterministic mock
vectorized generative model so every step of the predict-reweight-resample
cycle can be pinned to exact tensor values, including that per-belief actions
and observations are applied row-wise and never leak across the batch.
"""

import math

import pytest
import torch
from torch import Tensor

from POMDPPlanners.core.belief import BatchedParticleBelief


class DriftModel:
    """Deterministic, fully controllable vectorized generative model.

    Transitions add the (float-cast) action index to every state coordinate,
    observations echo the next state, and the observation log-likelihood is a
    scaled negative squared distance ``-scale * ||s' - o||^2``, so weights
    after a reweight are known in closed form.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self._device = torch.device("cpu")
        self._scale = scale

    @property
    def device(self) -> torch.device:
        return self._device

    def sample_next_states(self, states: Tensor, actions: Tensor) -> Tensor:
        return states + actions.to(states.dtype).unsqueeze(1)

    def sample_observations(self, next_states: Tensor, actions: Tensor) -> Tensor:
        del actions
        return next_states.clone()

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        del actions, next_states
        return torch.zeros(states.shape[0], device=self._device)

    def terminal_mask(self, states: Tensor) -> Tensor:
        return torch.zeros(states.shape[0], dtype=torch.bool, device=self._device)

    def observation_log_probs(
        self, next_states: Tensor, actions: Tensor, observations: Tensor
    ) -> Tensor:
        del actions
        return -self._scale * ((next_states - observations) ** 2).sum(dim=1)

    def action_keys(self, actions: Tensor) -> Tensor:
        return actions.to(torch.int64)

    def observation_keys(self, observations: Tensor) -> Tensor:
        return torch.zeros(observations.shape[0], dtype=torch.int64, device=self._device)


def _line_belief(
    batch_size: int = 3,
    num_particles: int = 8,
    state_dim: int = 2,
    **kwargs,
) -> BatchedParticleBelief:
    """Uniform-weight belief whose particle i in every row is the point (i, i, ...)."""
    line = torch.arange(num_particles, dtype=torch.float32).unsqueeze(1).repeat(1, state_dim)
    particles = line.unsqueeze(0).repeat(batch_size, 1, 1)
    log_weights = torch.zeros(batch_size, num_particles)
    return BatchedParticleBelief(particles, log_weights, DriftModel(), **kwargs)


def test_init_rejects_malformed_tensors():
    """Test that construction validates particle and log-weight shapes.

    Purpose: Validates that malformed inputs fail fast with a clear error

    Given: Particle/log-weight tensors that are 2-D, shape-mismatched, or
        contain NaN log-weights
    When: A BatchedParticleBelief is constructed from each of them
    Then: A ValueError is raised in every case

    Test type: unit
    """
    model = DriftModel()
    with pytest.raises(ValueError, match="3-D"):
        BatchedParticleBelief(torch.zeros(4, 2), torch.zeros(1, 4), model)
    with pytest.raises(ValueError, match="2-D"):
        BatchedParticleBelief(torch.zeros(1, 4, 2), torch.zeros(4), model)
    with pytest.raises(ValueError, match="does not"):
        BatchedParticleBelief(torch.zeros(2, 4, 2), torch.zeros(2, 5), model)
    with pytest.raises(ValueError, match="NaN"):
        BatchedParticleBelief(torch.zeros(1, 2, 2), torch.tensor([[0.0, float("nan")]]), model)


def test_from_root_replicates_particles_with_uniform_weights():
    """Test that from_root broadcasts one particle set across the batch.

    Purpose: Validates the convenience constructor used to fan a planner's
        root belief out into a batch

    Given: A single [Np, ds] particle set and batch_size=4
    When: from_root builds the batched belief
    Then: Particles are [4, Np, ds] with every row equal to the root set and
        log-weights are uniform at -log(Np)

    Test type: unit
    """
    root = torch.randn(6, 3)
    belief = BatchedParticleBelief.from_root(root, DriftModel(), batch_size=4)
    assert belief.particles.shape == (4, 6, 3)
    assert torch.equal(belief.particles[2], root)
    assert torch.allclose(belief.log_weights, torch.full((4, 6), -math.log(6)))


def test_propagate_applies_per_belief_actions_row_wise():
    """Test that propagate shifts each belief by its own action only.

    Purpose: Validates that the flatten-call-reshape round trip keeps each
        belief's action on its own row of the batch

    Given: Three identical beliefs and per-belief actions [0, 1, 2] under a
        deterministic drift transition
    When: propagate is called
    Then: Row b's particles are shifted by exactly b and log-weights are
        unchanged

    Test type: unit
    """
    belief = _line_belief(batch_size=3)
    actions = torch.tensor([0, 1, 2], dtype=torch.int64)
    propagated = belief.propagate(actions)
    for row, shift in enumerate([0.0, 1.0, 2.0]):
        assert torch.equal(propagated.particles[row], belief.particles[row] + shift)
    assert torch.equal(propagated.log_weights, belief.log_weights)


def test_reweight_adds_observation_log_likelihoods_per_row():
    """Test that reweight adds each row's own observation log-likelihood.

    Purpose: Validates the Bayesian weighting step and its row alignment

    Given: Beliefs with particles at (i, i) and per-belief observations equal
        to particle 0 in row 0 and particle 1 in row 1
    When: reweight is called
    Then: New log-weights equal old log-weights plus -||p - o||^2 computed
        against each row's own observation

    Test type: unit
    """
    belief = _line_belief(batch_size=2)
    actions = torch.zeros(2, dtype=torch.int64)
    observations = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    reweighted = belief.reweight(actions, observations)
    expected = -((belief.particles - observations.unsqueeze(1)) ** 2).sum(dim=2)
    assert torch.allclose(reweighted.log_weights, expected)
    assert torch.equal(reweighted.particles, belief.particles)


def test_update_matches_propagate_then_reweight():
    """Test that update composes the predict and reweight steps.

    Purpose: Validates that the one-call update equals the explicit
        propagate + reweight pipeline when resampling is off

    Given: A belief batch, per-belief actions, and per-belief observations
    When: update is called and separately propagate followed by reweight
    Then: Both paths produce identical particles and log-weights

    Test type: unit
    """
    belief = _line_belief(batch_size=3)
    actions = torch.tensor([1, 0, 2], dtype=torch.int64)
    observations = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    updated = belief.update(actions, observations)
    explicit = belief.propagate(actions).reweight(actions, observations)
    assert torch.equal(updated.particles, explicit.particles)
    assert torch.allclose(updated.log_weights, explicit.log_weights)


def test_update_with_resampling_resamples_only_degenerate_rows():
    """Test per-belief ESS-gated resampling inside update.

    Purpose: Validates that resampling triggers row-wise: collapsed rows are
        resampled to uniform weights while healthy rows keep their weights

    Given: A resampling-enabled belief where row 0's observation matches one
        particle overwhelmingly (ESS ~ 1) and row 1's particles are identical
        so all its weights stay equal (ESS = Np)
    When: update is called
    Then: Row 0 has uniform log-weights and (almost) all particles equal the
        favoured one; row 1 keeps its particles and non-uniform-constant
        log-weights untouched

    Test type: unit
    """
    torch.manual_seed(0)
    num_particles = 8
    line = torch.arange(num_particles, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    particles = torch.stack([line, torch.ones(num_particles, 2)])
    log_weights = torch.zeros(2, num_particles)
    belief = BatchedParticleBelief(particles, log_weights, DriftModel(scale=100.0), resampling=True)
    actions = torch.zeros(2, dtype=torch.int64)
    observations = torch.tensor([[3.0, 3.0], [0.0, 0.0]])
    updated = belief.update(actions, observations)

    uniform = torch.full((num_particles,), -math.log(num_particles))
    assert torch.allclose(updated.log_weights[0], uniform)
    assert torch.equal(updated.particles[0], torch.full((num_particles, 2), 3.0))
    assert torch.equal(updated.particles[1], particles[1])
    assert torch.allclose(updated.log_weights[1], torch.full((num_particles,), -200.0))


def test_sample_states_shape_and_degenerate_support():
    """Test batched state sampling shapes and weight fidelity.

    Purpose: Validates that sample_states draws per row from that row's own
        weight distribution

    Given: Two beliefs whose weights put all mass on particle 2 (row 0) and
        particle 5 (row 1)
    When: sample_states(16) is called
    Then: The result is [2, 16, ds] and every sample equals the row's
        full-mass particle

    Test type: unit
    """
    belief = _line_belief(batch_size=2)
    log_weights = torch.full((2, belief.num_particles), -float("inf"))
    log_weights[0, 2] = 0.0
    log_weights[1, 5] = 0.0
    peaked = BatchedParticleBelief(belief.particles, log_weights, DriftModel())
    samples = peaked.sample_states(16)
    assert samples.shape == (2, 16, 2)
    assert torch.equal(samples[0], torch.full((16, 2), 2.0))
    assert torch.equal(samples[1], torch.full((16, 2), 5.0))


def test_sample_observations_shape_and_values():
    """Test per-particle observation generation.

    Purpose: Validates the observation fan-out used to generate candidate
        futures during planning

    Given: A belief batch and per-belief actions under an echo observation
        model
    When: sample_observations is called
    Then: The result is [B, Np, do] and equals the particles themselves

    Test type: unit
    """
    belief = _line_belief(batch_size=3)
    observations = belief.sample_observations(torch.zeros(3, dtype=torch.int64))
    assert observations.shape == (3, belief.num_particles, 2)
    assert torch.equal(observations, belief.particles)


def test_resample_returns_uniform_weights_from_weighted_support():
    """Test full systematic resampling of the batch.

    Purpose: Validates that resample redraws particles by weight and resets
        weights to uniform

    Given: A belief whose row puts all mass on particle 4
    When: resample is called
    Then: All resampled particles equal particle 4 and log-weights are
        uniform at -log(Np)

    Test type: unit
    """
    torch.manual_seed(0)
    belief = _line_belief(batch_size=1)
    log_weights = torch.full((1, belief.num_particles), -float("inf"))
    log_weights[0, 4] = 0.0
    peaked = BatchedParticleBelief(belief.particles, log_weights, DriftModel())
    resampled = peaked.resample()
    assert torch.equal(resampled.particles[0], torch.full((belief.num_particles, 2), 4.0))
    uniform = torch.full((1, belief.num_particles), -math.log(belief.num_particles))
    assert torch.allclose(resampled.log_weights, uniform)


def test_effective_sample_size_uniform_and_degenerate():
    """Test the per-belief effective sample size.

    Purpose: Validates the ESS statistic that gates resampling

    Given: A batch whose row 0 has uniform weights and row 1 has all mass on
        one particle
    When: effective_sample_size is called
    Then: Row 0 yields Np and row 1 yields 1

    Test type: unit
    """
    belief = _line_belief(batch_size=2)
    log_weights = torch.zeros(2, belief.num_particles)
    log_weights[1] = -float("inf")
    log_weights[1, 0] = 0.0
    mixed = BatchedParticleBelief(belief.particles, log_weights, DriftModel())
    ess = mixed.effective_sample_size()
    assert torch.allclose(ess, torch.tensor([float(belief.num_particles), 1.0]))


def test_normalized_weights_degenerate_row_falls_back_to_uniform():
    """Test the all--inf log-weight fallback.

    Purpose: Validates that a belief row carrying zero likelihood everywhere
        degrades to uniform weights instead of NaNs, per row

    Given: A batch whose row 0 log-weights are all -inf and row 1 weights are
        peaked on particle 1
    When: normalized_weights is read
    Then: Row 0 is uniform and row 1 keeps its peaked distribution

    Test type: unit
    """
    belief = _line_belief(batch_size=2)
    log_weights = torch.full((2, belief.num_particles), -float("inf"))
    log_weights[1, 1] = 0.0
    degenerate = BatchedParticleBelief(belief.particles, log_weights, DriftModel())
    weights = degenerate.normalized_weights
    uniform = torch.full((belief.num_particles,), 1.0 / belief.num_particles)
    assert torch.allclose(weights[0], uniform)
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0))
    assert weights[1, 1] == pytest.approx(1.0)


def test_batched_methods_reject_misaligned_actions_and_observations():
    """Test the per-call shape validation of actions and observations.

    Purpose: Validates that batch-size mismatches fail fast instead of
        silently broadcasting

    Given: A batch of 3 beliefs
    When: propagate receives 2 actions and reweight receives observations
        with 2 rows
    Then: Both calls raise ValueError

    Test type: unit
    """
    belief = _line_belief(batch_size=3)
    with pytest.raises(ValueError, match="actions"):
        belief.propagate(torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="observations"):
        belief.reweight(torch.zeros(3, dtype=torch.int64), torch.zeros(2, 2))


def test_batched_particle_belief_usage_example():
    """Test the usage example from the BatchedParticleBelief docstring.

    Purpose: Validates that the documented example executes and produces the
        shapes it claims

    Given: The exact model and belief construction from the class docstring
    When: update and sample_states are called as in the example
    Then: The resulting shapes match the docstring output

    Test type: example
    """
    torch.manual_seed(0)
    belief = BatchedParticleBelief(
        particles=torch.zeros(3, 8, 2),
        log_weights=torch.zeros(3, 8),
        model=DriftModel(),
    )
    actions = torch.ones(3, dtype=torch.int64)
    observations = torch.ones(3, 2)
    updated = belief.update(actions, observations)
    assert updated.particles.shape == torch.Size([3, 8, 2])
    assert updated.sample_states(4).shape == torch.Size([3, 4, 2])
