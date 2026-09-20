# SPDX-License-Identifier: MIT

"""Tests for serializing each of the package's beliefs into a trace.

The belief is what an episode trace exists to carry and the field most easily
faked, so every concrete ``Belief`` in the package gets its own test here. The
dispatch is on the class, not on attribute names, and the point of that is
that all the particle classes come out as one shape: a reader draws particles
once rather than once per class.
"""

from typing import Any, List

import numpy as np
import pytest

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
from POMDPPlanners.core.belief.gaussian_belief_updaters import GaussianBeliefUpdater
from POMDPPlanners.core.belief.gaussian_mixture_belief import GaussianMixtureBeliefUpdater
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.utils.weighted_particle_beliefs import (
    WeightedParticleBeliefContinuousLightDarkFullCoverage,
)
from POMDPPlanners.core.simulation.belief_payloads import (
    MAX_PAYLOAD_PARTICLES,
    BeliefPayloadKind,
    belief_to_payload,
)
from POMDPPlanners.core.simulation.traces import EpisodeTrace

POINTS = [[0.0, 5.0], [1.0, 4.0], [2.0, 6.0], [3.0, 5.5]]


class _IdentityVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Smallest updater that satisfies the vectorized belief's constructor."""

    def batch_transition(self, particles: np.ndarray, action: Any) -> np.ndarray:
        return particles

    def batch_observation_log_likelihood(
        self, next_particles: np.ndarray, action: Any, observation: Any
    ) -> np.ndarray:
        return np.zeros(len(next_particles))

    @property
    def config_id(self) -> str:
        return "identity"


class _IdentityGaussianUpdater(GaussianBeliefUpdater):
    """Smallest updater that satisfies the Gaussian belief's constructor."""

    def update(self, mean, covariance, action, observation):
        return mean, covariance

    @property
    def config_id(self) -> str:
        return "identity"


class _IdentityMixtureUpdater(GaussianMixtureBeliefUpdater):
    """Smallest updater that satisfies the mixture belief's constructor."""

    def update(self, means, covariances, weights, action, observation):
        return means, covariances, weights

    @property
    def config_id(self) -> str:
        return "identity"


class _PlainUnweightedBelief(UnweightedParticleBelief):
    """Smallest concrete ``UnweightedParticleBelief``.

    The base class is abstract because reinvigoration is environment-specific;
    nothing about serializing it is.
    """

    def _reinvigoration_pertubation(self, action, observation, pomdp):
        return observation


def _particle_arrays() -> List[np.ndarray]:
    return [np.asarray(p, dtype=float) for p in POINTS]


def _assert_particle_payload(payload, expected_class: str, expected_count: int):
    """Every particle belief must produce the same shape, whatever its class."""
    assert payload["kind"] == BeliefPayloadKind.PARTICLES.value
    assert payload["belief_class"] == expected_class
    assert payload["num_particles"] == expected_count
    assert payload["num_written"] == min(expected_count, MAX_PAYLOAD_PARTICLES)
    assert len(payload["particles"]) == payload["num_written"]
    assert len(payload["weights"]) == payload["num_written"]
    # Normalized here so no reader has to know which class produced them.
    assert sum(payload["weights"]) == pytest.approx(1.0)


# -- the particle family ---------------------------------------------------


def test_weighted_particle_belief_writes_its_particles_and_weights():
    """WeightedParticleBelief writes the cloud and its normalized weights.

    Given: A four-particle belief with deliberately uneven weights.
    When: It is serialized.
    Then: The payload is the shared particle shape and the weights match the
        belief's own ``normalized_weights``.
    """
    belief = WeightedParticleBelief(
        particles=_particle_arrays(), log_weights=np.log([0.4, 0.3, 0.2, 0.1])
    )
    payload = belief_to_payload(belief)

    _assert_particle_payload(payload, "WeightedParticleBelief", 4)
    assert payload["weighted"] is True
    assert payload["particles"] == POINTS
    assert payload["weights"] == pytest.approx(list(belief.normalized_weights))


def test_weighted_particle_belief_reinvigoration_is_the_same_shape():
    """A reinvigorating belief is a WeightedParticleBelief to a reader.

    Purpose: It subclasses WeightedParticleBelief and differs only in how it
    updates. Dispatching on the class rather than the attribute names has to
    cover the subclass through its parent, or the viewer would need a branch
    for a difference that does not exist at drawing time.

    Given: The reinvigorating belief Light-Dark itself runs with.
    When: It is serialized.
    Then: It produces the shared particle payload, naming its own class.
    """
    belief = WeightedParticleBeliefContinuousLightDarkFullCoverage(
        particles=_particle_arrays(), log_weights=np.log(np.ones(4) / 4)
    )
    payload = belief_to_payload(belief)

    _assert_particle_payload(payload, "WeightedParticleBeliefContinuousLightDarkFullCoverage", 4)
    assert payload["weighted"] is True
    assert payload["particles"] == POINTS


def test_vectorized_weighted_particle_belief_is_the_same_shape():
    """The vectorized belief stores an (N, d) array and still writes particles.

    Purpose: It is a different class with a different internal layout and the
    same meaning. Attribute-name duck-typing happened to cover it; class
    dispatch covers it on purpose.
    """
    belief = VectorizedWeightedParticleBelief(
        particles=np.asarray(POINTS, dtype=float),
        log_weights=np.log(np.asarray([0.4, 0.3, 0.2, 0.1])),
        updater=_IdentityVectorizedUpdater(),
    )
    payload = belief_to_payload(belief)

    _assert_particle_payload(payload, "VectorizedWeightedParticleBelief", 4)
    assert payload["particles"] == POINTS
    assert payload["weights"] == pytest.approx(list(belief.normalized_weights))


def test_weighted_particle_belief_state_update_normalizes_its_raw_weights():
    """The incremental belief keeps raw observation likelihoods; they normalize.

    Purpose: Its ``weights`` are accumulated likelihoods that never sum to one.
    Writing them unchanged would hand a viewer a distribution that is not one.
    """
    belief = WeightedParticleBeliefStateUpdate(
        particles=_particle_arrays(), weights=[4.0, 3.0, 2.0, 1.0]
    )
    payload = belief_to_payload(belief)

    _assert_particle_payload(payload, "WeightedParticleBeliefStateUpdate", 4)
    assert payload["weighted"] is True
    assert payload["weights"] == pytest.approx([0.4, 0.3, 0.2, 0.1])


def test_unweighted_particle_belief_is_marked_uniform():
    """An unweighted belief says so, so a viewer does not shade it.

    Purpose: Shading a uniform cloud by weight would draw structure the belief
    does not have — the one failure mode a belief display must not have.
    """
    belief = _PlainUnweightedBelief(particles=_particle_arrays())
    payload = belief_to_payload(belief)

    _assert_particle_payload(payload, "_PlainUnweightedBelief", 4)
    assert payload["weighted"] is False
    assert payload["weights"] == pytest.approx([0.25] * 4)


def test_unweighted_particle_belief_state_update_is_marked_uniform():
    """The incremental unweighted belief is uniform too."""
    belief = UnweightedParticleBeliefStateUpdate(particles=_particle_arrays())
    payload = belief_to_payload(belief)

    _assert_particle_payload(payload, "UnweightedParticleBeliefStateUpdate", 4)
    assert payload["weighted"] is False


def test_non_numeric_particles_survive():
    """A discrete state is written as itself, not coerced into coordinates.

    Purpose: Tiger's particles are strings. What a particle means is the
    environment's business; core's job is to not lose it.
    """
    belief = _PlainUnweightedBelief(particles=["tiger_left", "tiger_right"])
    payload = belief_to_payload(belief)

    assert payload["particles"] == ["tiger_left", "tiger_right"]


def test_a_large_cloud_is_trimmed_to_its_heaviest_particles():
    """A big weighted belief keeps its mass and reports what it dropped.

    Purpose: A vectorized belief can carry tens of thousands of particles.
    Dropping the light tail keeps the cloud; dropping at random would not.
    """
    count = MAX_PAYLOAD_PARTICLES * 3
    weights = np.linspace(1.0, 2.0, count)
    belief = WeightedParticleBelief(
        particles=[np.array([float(i), 0.0]) for i in range(count)],
        log_weights=np.log(weights / weights.sum()),
    )
    payload = belief_to_payload(belief)

    assert payload["num_particles"] == count
    assert payload["num_written"] == MAX_PAYLOAD_PARTICLES
    assert sum(payload["weights"]) == pytest.approx(1.0)
    # The heaviest particles are the high indices, so the tail is what went.
    assert payload["particles"][0][0] >= count - MAX_PAYLOAD_PARTICLES


def test_the_cap_is_a_parameter():
    """The trim is configurable, because the right cap is not universal."""
    belief = WeightedParticleBelief(
        particles=_particle_arrays(), log_weights=np.log(np.ones(4) / 4)
    )
    payload = belief_to_payload(belief, max_particles=2)
    assert payload["num_particles"] == 4
    assert payload["num_written"] == 2


# -- the Gaussian family ---------------------------------------------------


def test_gaussian_belief_writes_its_mean_and_covariance():
    """GaussianBelief writes exactly what it holds."""
    belief = GaussianBelief(
        mean=np.array([1.0, 2.0]),
        covariance=np.array([[0.5, 0.1], [0.1, 0.3]]),
        updater=_IdentityGaussianUpdater(),
    )
    payload = belief_to_payload(belief)

    assert payload["kind"] == BeliefPayloadKind.GAUSSIAN.value
    assert payload["belief_class"] == "GaussianBelief"
    assert payload["mean"] == pytest.approx([1.0, 2.0])
    assert np.allclose(payload["covariance"], [[0.5, 0.1], [0.1, 0.3]])


def test_gaussian_mixture_belief_writes_every_component():
    """A mixture is written as its components, not collapsed into one blob.

    Purpose: A genuinely bimodal belief collapsed to a single mean and
    covariance would be drawn as a wide cloud centred between its modes —
    exactly where the belief says nothing is.

    Given: A two-component mixture with modes far apart.
    When: It is serialized.
    Then: Both components survive with their own weight, mean and covariance.
    """
    belief = GaussianMixtureBelief(
        means=[np.array([0.0, 0.0]), np.array([9.0, 9.0])],
        covariances=[np.eye(2) * 0.5, np.eye(2) * 2.0],
        weights=np.array([0.7, 0.3]),
        updater=_IdentityMixtureUpdater(),
    )
    payload = belief_to_payload(belief)

    assert payload["kind"] == BeliefPayloadKind.GAUSSIAN_MIXTURE.value
    assert payload["belief_class"] == "GaussianMixtureBelief"
    assert len(payload["components"]) == 2
    assert [c["weight"] for c in payload["components"]] == pytest.approx([0.7, 0.3])
    assert payload["components"][1]["mean"] == pytest.approx([9.0, 9.0])
    assert np.allclose(payload["components"][1]["covariance"], [[2.0, 0.0], [0.0, 2.0]])


# -- the batched belief ----------------------------------------------------


def test_batched_particle_belief_is_written_as_a_batch():
    """A batch is B beliefs, and is written as B clouds rather than merged.

    Purpose: ``BatchedParticleBelief`` holds independent beliefs in one tensor.
    Flattening it would produce a cloud that was never anyone's belief.

    Given: A batch of three beliefs over eight particles each.
    When: It is serialized.
    Then: The batch size is recorded and each member is an ordinary particle
        payload.
    """
    # pylint: disable-next=import-outside-toplevel
    import torch

    class _Model:
        @property
        def device(self):
            return torch.device("cpu")

    belief = BatchedParticleBelief(
        particles=torch.arange(3 * 8 * 2, dtype=torch.float32).reshape(3, 8, 2),
        log_weights=torch.zeros(3, 8),
        model=_Model(),  # type: ignore[arg-type]
    )
    payload = belief_to_payload(belief)

    assert payload["kind"] == BeliefPayloadKind.PARTICLE_BATCH.value
    assert payload["belief_class"] == "BatchedParticleBelief"
    assert payload["batch_size"] == 3
    assert len(payload["beliefs"]) == 3
    for member in payload["beliefs"]:
        assert member["kind"] == BeliefPayloadKind.PARTICLES.value
        assert member["num_particles"] == 8
        assert sum(member["weights"]) == pytest.approx(1.0)


def test_batched_log_weights_do_not_underflow():
    """Very negative log-weights still produce a usable distribution.

    Purpose: Log-weights are unnormalized and routinely sit far below zero.
    Exponentiating them directly underflows the row to all zeros, which would
    write a belief with no mass anywhere.
    """
    # pylint: disable-next=import-outside-toplevel
    import torch

    class _Model:
        @property
        def device(self):
            return torch.device("cpu")

    belief = BatchedParticleBelief(
        particles=torch.zeros(1, 4, 2),
        log_weights=torch.tensor([[-900.0, -901.0, -902.0, -903.0]]),
        model=_Model(),  # type: ignore[arg-type]
    )
    member = belief_to_payload(belief)["beliefs"][0]

    assert sum(member["weights"]) == pytest.approx(1.0)
    assert member["weights"][0] > member["weights"][-1]


# -- the fallback ----------------------------------------------------------


def test_an_unknown_belief_class_is_named_rather_than_approximated():
    """A belief core has no payload for is declared, not invented.

    Purpose: The fallback exists for a class nobody has written yet. Naming it
    makes the gap diagnosable; drawing a plausible cloud instead would be the
    one failure this whole module is written to prevent.
    """

    class _FutureBelief:
        pass

    payload = belief_to_payload(_FutureBelief())
    assert payload["kind"] == BeliefPayloadKind.UNSUPPORTED.value
    assert payload["belief_class"] == "_FutureBelief"


# -- round trip through a trace file ---------------------------------------


@pytest.mark.parametrize(
    "belief_factory, expected_kind",
    [
        (
            lambda: WeightedParticleBelief(
                particles=_particle_arrays(), log_weights=np.log(np.ones(4) / 4)
            ),
            BeliefPayloadKind.PARTICLES,
        ),
        (
            lambda: _PlainUnweightedBelief(particles=_particle_arrays()),
            BeliefPayloadKind.PARTICLES,
        ),
        (
            lambda: WeightedParticleBeliefStateUpdate(
                particles=_particle_arrays(), weights=[1.0, 1.0, 1.0, 1.0]
            ),
            BeliefPayloadKind.PARTICLES,
        ),
        (
            lambda: UnweightedParticleBeliefStateUpdate(particles=_particle_arrays()),
            BeliefPayloadKind.PARTICLES,
        ),
        (
            lambda: VectorizedWeightedParticleBelief(
                particles=np.asarray(POINTS, dtype=float),
                log_weights=np.zeros(4),
                updater=_IdentityVectorizedUpdater(),
            ),
            BeliefPayloadKind.PARTICLES,
        ),
        (
            lambda: GaussianBelief(
                mean=np.array([1.0, 2.0]),
                covariance=np.eye(2),
                updater=_IdentityGaussianUpdater(),
            ),
            BeliefPayloadKind.GAUSSIAN,
        ),
        (
            lambda: GaussianMixtureBelief(
                means=[np.zeros(2), np.ones(2)],
                covariances=[np.eye(2), np.eye(2)],
                weights=np.array([0.5, 0.5]),
                updater=_IdentityMixtureUpdater(),
            ),
            BeliefPayloadKind.GAUSSIAN_MIXTURE,
        ),
    ],
)
def test_every_belief_class_round_trips_through_a_trace_file(
    tmp_path, belief_factory, expected_kind
):
    """Each belief survives being written to a trace and read back.

    Purpose: The payload's only job is to cross from Python into a browser
    through a JSON file. A field that does not survive the file is a field the
    viewer silently never sees.

    Given: One belief of each class in the package.
    When: Its payload is written inside a trace and read back.
    Then: The payload is unchanged and carries the expected kind.
    """
    payload = belief_to_payload(belief_factory())
    trace = EpisodeTrace(
        environment="Env",
        payload_kind="x.v1",
        episode_index=0,
        discount_factor=0.95,
        payload={"beliefs": [payload]},
    )
    restored = EpisodeTrace.read(trace.write(tmp_path / "trace.json"))

    assert restored.payload["beliefs"][0] == payload
    assert restored.payload["beliefs"][0]["kind"] == expected_kind.value
