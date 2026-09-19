# SPDX-License-Identifier: MIT

"""Serializing a belief into a trace, for any belief the package defines.

The belief is the field an episode trace exists to carry, and it is the one a
reader is most likely to get wrong: a cloud regenerated from the true state
looks convincing and says nothing. So it is written from the belief object the
run actually held.

That serialization belongs here rather than in any one environment's exporter,
because :class:`~POMDPPlanners.core.belief.base_belief.Belief` is already a
core abstraction with a closed family of implementations. Dispatching on the
class instead of on attribute names has two consequences worth stating:

* every particle belief serializes to the same payload whatever its concrete
  class — the plain, the reinvigorating, the incremental, the vectorized and
  the batched ones are one shape to a reader, so a viewer draws particles once
  rather than once per class;
* an environment's exporter hands a ``Belief`` here and gets a payload back,
  so adding an environment means writing its world and its states, never its
  belief.

``unsupported`` remains, but only as what it says: a belief class nobody has
written yet. It names the class, so the gap is diagnosable rather than silent.
"""

from enum import Enum
from typing import Any, Dict, List

import numpy as np

from POMDPPlanners.core.belief import (
    BatchedParticleBelief,
    GaussianBelief,
    GaussianMixtureBelief,
    UnweightedParticleBelief,
    UnweightedParticleBeliefStateUpdate,
    VectorizedWeightedParticleBelief,
    WeightedParticleBelief,
    WeightedParticleBeliefStateUpdate,
)
from POMDPPlanners.core.simulation.traces import to_jsonable

# How many particles of one cloud are written. A vectorized or batched belief
# can carry tens of thousands, which would make a trace hundreds of megabytes
# and a browser scene unusable. The payload records both counts, so a reader
# can say what it is showing rather than imply it has the whole cloud.
MAX_PAYLOAD_PARTICLES = 400


class BeliefPayloadKind(str, Enum):
    """The shapes a belief is written in, and what a reader switches on.

    A kind is a drawing contract, not a class name: every particle belief is
    ``PARTICLES`` regardless of which concrete class produced it.
    """

    PARTICLES = "particles"
    PARTICLE_BATCH = "particle_batch"
    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    UNSUPPORTED = "unsupported"


def _as_float_array(values: Any) -> np.ndarray:
    """Coerce weights to a 1-D float array, whatever container they arrive in."""
    return np.asarray(values, dtype=float).reshape(-1)


def _particle_payload(
    particles: Any,
    weights: np.ndarray,
    weighted: bool,
    belief_class: str,
    max_particles: int,
) -> Dict[str, Any]:
    """Build the shared particle payload from a cloud and its weights.

    Args:
        particles: The particles, in any sequence or array form. They are
            written as they are, so a non-numeric particle (a Tiger state
            string, say) survives; deciding what a particle means is the
            environment's business, not this module's.
        weights: One weight per particle, already aligned with them.
        weighted: False when the weights are uniform by construction. A reader
            that shades particles by weight must not imply structure an
            unweighted belief does not have.
        belief_class: Name of the concrete class, for diagnosis.
        max_particles: Cap on how many particles are written.

    Returns:
        The payload, with weights renormalized over what was written.
    """
    items: List[Any] = list(particles)
    mass = _as_float_array(weights)
    total = len(items)

    if total > max_particles:
        if weighted:
            # Keep the cloud's mass and drop the tail. Taking a random subset
            # would lose exactly the particles a reader cares about most.
            keep = np.argsort(mass)[::-1][:max_particles]
            keep.sort()
        else:
            # Every particle carries the same weight, so any subset is an
            # unbiased sample; the first N is the one that is reproducible.
            keep = np.arange(max_particles)
        items = [items[int(i)] for i in keep]
        mass = mass[keep]

    written_total = float(mass.sum())
    if written_total > 0:
        mass = mass / written_total
    elif len(mass):
        # Degenerate weights (all zero, or an empty-sum incremental belief)
        # still have to produce a distribution a reader can draw.
        mass = np.full(len(mass), 1.0 / len(mass))

    return {
        "kind": BeliefPayloadKind.PARTICLES.value,
        "belief_class": belief_class,
        "weighted": bool(weighted),
        "particles": [to_jsonable(item) for item in items],
        "weights": [float(w) for w in mass],
        "num_particles": int(total),
        "num_written": int(len(items)),
    }


def _uniform_weights(count: int) -> np.ndarray:
    return np.full(count, 1.0 / count) if count else np.empty(0)


def _gaussian_payload(belief: GaussianBelief) -> Dict[str, Any]:
    return {
        "kind": BeliefPayloadKind.GAUSSIAN.value,
        "belief_class": type(belief).__name__,
        "mean": [float(v) for v in np.asarray(belief.mean, dtype=float).reshape(-1)],
        "covariance": [
            [float(v) for v in row]
            for row in np.asarray(belief.covariance, dtype=float).reshape(len(belief.mean), -1)
        ],
    }


def _gaussian_mixture_payload(belief: GaussianMixtureBelief) -> Dict[str, Any]:
    """Write a mixture as its components, not as a single blob.

    Collapsing a mixture to one mean and covariance would erase the very thing
    it represents: a belief that is genuinely multi-modal would be drawn as a
    wide unimodal cloud centred between its modes, where nothing is.
    """
    components = []
    for weight, mean, covariance in zip(belief.weights, belief.means, belief.covariances):
        mean_array = np.asarray(mean, dtype=float).reshape(-1)
        components.append(
            {
                "weight": float(weight),
                "mean": [float(v) for v in mean_array],
                "covariance": [
                    [float(v) for v in row]
                    for row in np.asarray(covariance, dtype=float).reshape(len(mean_array), -1)
                ],
            }
        )
    return {
        "kind": BeliefPayloadKind.GAUSSIAN_MIXTURE.value,
        "belief_class": type(belief).__name__,
        "components": components,
    }


def _batched_payload(belief: BatchedParticleBelief, max_particles: int) -> Dict[str, Any]:
    """Write a batch as the batch it is: several clouds, not one.

    ``BatchedParticleBelief`` holds ``B`` independent beliefs as one tensor, so
    flattening it into a single cloud would merge beliefs that were never the
    same belief. A reader is told the batch size and gets each member in the
    ordinary particle shape, and can then draw one, all, or refuse.
    """
    particles = belief.particles.detach().cpu().numpy()
    log_weights = belief.log_weights.detach().cpu().numpy()

    members = []
    for index in range(particles.shape[0]):
        row = log_weights[index]
        # Shift before exponentiating: log-weights are unnormalized and can sit
        # far below zero, where a direct exp underflows the whole row to zero.
        shifted = np.exp(row - np.max(row))
        members.append(
            _particle_payload(
                particles=list(particles[index]),
                weights=shifted,
                weighted=True,
                belief_class=type(belief).__name__,
                max_particles=max_particles,
            )
        )

    return {
        "kind": BeliefPayloadKind.PARTICLE_BATCH.value,
        "belief_class": type(belief).__name__,
        "batch_size": int(particles.shape[0]),
        "beliefs": members,
    }


# One branch per belief class in the package, which is the point: a shorter
# dispatch would be one that guesses at a class rather than naming it.
# pylint: disable-next=too-many-return-statements
def belief_to_payload(belief: Any, max_particles: int = MAX_PAYLOAD_PARTICLES) -> Dict[str, Any]:
    """Serialize any of the package's beliefs into JSON-compatible data.

    Args:
        belief: The belief recorded on a ``StepData``, or any other belief
            object.
        max_particles: Cap on how many particles of one cloud are written.

    Returns:
        A payload whose ``kind`` is one of :class:`BeliefPayloadKind`. Weights
        in a particle payload are normalized and sum to one, so a reader never
        has to know which concrete class produced them.
    """
    # Ordered most-derived first where the hierarchy overlaps.
    # ``WeightedParticleBeliefReinvigoration`` subclasses ``WeightedParticleBelief``
    # and is covered by it, deliberately: it is the same shape to a reader.
    if isinstance(belief, GaussianMixtureBelief):
        return _gaussian_mixture_payload(belief)
    if isinstance(belief, GaussianBelief):
        return _gaussian_payload(belief)
    if isinstance(belief, BatchedParticleBelief):
        return _batched_payload(belief, max_particles)

    if isinstance(belief, (WeightedParticleBelief, VectorizedWeightedParticleBelief)):
        # Both maintain ``normalized_weights`` alongside their log-weights.
        return _particle_payload(
            particles=belief.particles,
            weights=belief.normalized_weights,
            weighted=True,
            belief_class=type(belief).__name__,
            max_particles=max_particles,
        )
    if isinstance(belief, WeightedParticleBeliefStateUpdate):
        # Raw observation likelihoods, accumulated and never normalized.
        return _particle_payload(
            particles=belief.particles,
            weights=_as_float_array(belief.weights),
            weighted=True,
            belief_class=type(belief).__name__,
            max_particles=max_particles,
        )
    if isinstance(belief, (UnweightedParticleBelief, UnweightedParticleBeliefStateUpdate)):
        return _particle_payload(
            particles=belief.particles,
            weights=_uniform_weights(len(belief.particles)),
            weighted=False,
            belief_class=type(belief).__name__,
            max_particles=max_particles,
        )

    return {
        "kind": BeliefPayloadKind.UNSUPPORTED.value,
        "belief_class": type(belief).__name__,
    }
