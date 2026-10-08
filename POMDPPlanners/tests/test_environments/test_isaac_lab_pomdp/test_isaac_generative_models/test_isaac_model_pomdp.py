# SPDX-License-Identifier: MIT

"""Unit tests for the channel schema and the abstract factored Isaac model.

The schema exists to replace raw offsets with names, so its tests are mostly about the failure
modes offsets have: a misnamed channel must raise, not quietly read the wrong slice.
"""

from typing import Any, Mapping

import numpy as np
import pytest

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
    IsaacChannelSchema,
    IsaacModelPOMDP,
)
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception import (
    GaussianChannelObservationModel,
    IsaacObservationModel,
    LatentTypeSignalObservationModel,
)

SCHEMA = IsaacChannelSchema((("base_pose", 3), ("hazard_type", 2)))


class _StaticModel(IsaacModelPOMDP):
    """Concrete model with fixed dynamics, so the base's own behaviour is what is tested."""

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        del action
        vector = np.asarray(state, dtype=float)
        return vector if n_samples == 1 else np.stack([vector] * n_samples)

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del state, action
        return np.zeros(np.atleast_2d(np.asarray(next_states)).shape[0])

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del state, action, next_state
        return 0.0

    def is_terminal(self, state: Any) -> bool:
        del state
        return False


class _CountingSignalModel(IsaacObservationModel):
    """A sample-only channel that reports how many times it was asked to perceive."""

    channel = "counter"
    state_channels = ("hazard_type",)

    def __init__(self) -> None:
        self.calls = 0

    def perceive(self, clean_state: Mapping[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        return np.asarray(clean_state["hazard_type"], dtype=float) + self.calls


def _model(**overrides: Any) -> _StaticModel:
    settings: dict = {
        "state_schema": SCHEMA,
        "action_presets": [np.zeros(3), np.ones(3)],
        "discount_factor": 0.99,
        "observation_models": {
            "base_pose": GaussianChannelObservationModel(channel="base_pose", noise_std=0.1)
        },
    }
    settings.update(overrides)
    return _StaticModel(**settings)


# ── The channel schema ───────────────────────────────────────────────────


def test_schema_packs_and_slices_by_name() -> None:
    """Names replace offsets, so a round trip through them must be exact.

    Purpose: Validates pack, block and split against a two-channel schema

    Given: A schema of a 3-wide pose and a 2-wide hazard type
    When: Blocks are packed and read back by name
    Then: The widths, order and values all round-trip

    Test type: unit
    """
    state = SCHEMA.pack({"base_pose": [1.0, 2.0, 0.5], "hazard_type": [0.0, 1.0]})
    assert state.shape == (5,)
    assert SCHEMA.total_dim == 5
    assert SCHEMA.block(state, "base_pose") == pytest.approx([1.0, 2.0, 0.5])
    assert SCHEMA.block(state, "hazard_type") == pytest.approx([0.0, 1.0])
    assert SCHEMA.split(state)["hazard_type"] == pytest.approx([0.0, 1.0])


def test_schema_slices_a_batch_without_losing_the_batch_dimension() -> None:
    """The belief propagates particles in batches, so slicing must be batch-safe.

    Purpose: Validates that block preserves a leading batch dimension

    Given: A batch of four states
    When: A channel is sliced out
    Then: The result is (4, width)

    Test type: unit
    """
    batch = np.tile(SCHEMA.pack({"base_pose": np.zeros(3), "hazard_type": [0.0, 1.0]}), (4, 1))
    assert SCHEMA.block(batch, "hazard_type").shape == (4, 2)


def test_schema_broadcasts_an_unbatched_block_over_a_batched_one() -> None:
    """A per-episode latent block has to ride along with a batch of sampled robot states.

    Purpose: Validates the broadcasting rule in pack

    Given: A batch of five pose blocks and one unbatched hazard block
    When: They are packed
    Then: The result is (5, 5) with the hazard block repeated

    Test type: unit
    """
    packed = SCHEMA.pack({"base_pose": np.zeros((5, 3)), "hazard_type": [0.0, 1.0]})
    assert packed.shape == (5, 5)
    assert np.all(packed[:, 3:] == np.array([0.0, 1.0]))


def test_schema_indices_follow_the_order_requested() -> None:
    """The driven-block index array is built from this, so order is load-bearing.

    Purpose: Validates indices_of against a reordered channel list

    Given: A two-channel schema
    When: Indices are requested for the trailing channel then the leading one
    Then: They come back in that order, not in packing order

    Test type: unit
    """
    assert SCHEMA.indices_of(["hazard_type", "base_pose"]).tolist() == [3, 4, 0, 1, 2]


def test_schema_rejects_an_unknown_channel_name() -> None:
    """A typo must raise rather than read a plausible-looking wrong slice.

    Purpose: Validates the unknown-name guard

    Given: A schema without a "lidar" channel
    When: That channel is sliced
    Then: KeyError is raised listing the known names

    Test type: unit
    """
    with pytest.raises(KeyError, match="unknown channel"):
        SCHEMA.block(np.zeros(5), "lidar")


def test_schema_rejects_a_block_of_the_wrong_width() -> None:
    """Packing a mis-sized block would shift every channel after it.

    Purpose: Validates the width check in pack

    Given: A pose block of width 2 against a schema declaring 3
    When: The state is packed
    Then: ValueError is raised naming the channel and both widths

    Test type: unit
    """
    with pytest.raises(ValueError, match="base_pose"):
        SCHEMA.pack({"base_pose": [1.0, 2.0], "hazard_type": [0.0, 1.0]})


def test_schema_rejects_a_missing_channel() -> None:
    """A silently omitted channel would produce a short vector every consumer misreads.

    Purpose: Validates the completeness check in pack

    Given: Blocks for only one of the two channels
    When: The state is packed
    Then: KeyError is raised naming the missing channel

    Test type: unit
    """
    with pytest.raises(KeyError, match="hazard_type"):
        SCHEMA.pack({"base_pose": np.zeros(3)})


@pytest.mark.parametrize(
    "channels, message",
    [
        ((), "at least one channel"),
        ((("a", 2), ("a", 3)), "unique"),
        ((("a", 0),), "positive width"),
    ],
)
def test_schema_rejects_an_invalid_declaration(channels: tuple, message: str) -> None:
    """A malformed schema must fail where it is written, not where it is used.

    Purpose: Validates the construction-time guards on the schema

    Given: An empty, duplicated or zero-width channel declaration
    When: The schema is constructed
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match=message):
        IsaacChannelSchema(channels)


# ── The abstract model's observation composition ─────────────────────────


def test_observation_is_a_channel_mapping_not_a_flat_vector() -> None:
    """The whole point of the refactor: state and observation need not share a width.

    Purpose: Validates that sample_observation returns only the configured channels

    Given: A 5-wide state and a single 3-wide observation channel
    When: An observation is sampled
    Then: It is a mapping carrying just that channel, of width 3

    Test type: unit
    """
    np.random.seed(0)
    model = _model()
    state = SCHEMA.pack({"base_pose": [1.0, 2.0, 0.5], "hazard_type": [0.0, 1.0]})
    observation = model.sample_observation(state, action=None)
    assert sorted(observation) == ["base_pose"]
    assert np.asarray(observation["base_pose"]).shape == (3,)


def test_a_state_block_with_no_channel_stays_hidden() -> None:
    """A latent type is latent precisely because no observation channel is derived from it.

    Purpose: Validates that unobserved state blocks never leak into the observation

    Given: A schema with a hazard_type block and no channel reading it
    When: An observation is sampled
    Then: No channel carries the hazard type

    Test type: unit
    """
    np.random.seed(0)
    state = SCHEMA.pack({"base_pose": np.zeros(3), "hazard_type": [0.0, 1.0]})
    observation = _model().sample_observation(state, action=None)
    assert "hazard_type" not in observation


def test_batched_observation_draws_are_independent() -> None:
    """Repeating one draw would make a planner's observation widening degenerate.

    Purpose: Validates that n_samples draws are sampled independently

    Given: A Gaussian channel with appreciable noise
    When: Eight observations are drawn in one call
    Then: They are not all identical

    Test type: unit
    """
    np.random.seed(0)
    state = SCHEMA.pack({"base_pose": np.zeros(3), "hazard_type": [0.0, 1.0]})
    draws = _model().sample_observation(state, action=None, n_samples=8)
    assert len(draws) == 8
    stacked = np.stack([draw["base_pose"] for draw in draws])
    assert not np.allclose(stacked[0], stacked[1])


def test_observation_log_probability_sums_the_channel_densities() -> None:
    """A composed observation's density is the product of its channels', by construction.

    Purpose: Validates that the composed density equals the sum of the per-channel ones

    Given: A model with two density-carrying channels
    When: One observation is scored by the model and by the channels directly
    Then: The two agree

    Test type: unit
    """
    model = _model(
        observation_models={
            "base_pose": GaussianChannelObservationModel(channel="base_pose", noise_std=0.1),
            "echo_type": GaussianChannelObservationModel(
                channel="echo_type", state_channel="hazard_type", noise_std=0.2
            ),
        }
    )
    state = SCHEMA.pack({"base_pose": [1.0, 0.0, 0.0], "hazard_type": [0.0, 1.0]})
    observation = {"base_pose": np.array([1.1, 0.0, 0.0]), "echo_type": np.array([0.1, 0.9])}
    clean = model.clean_state(state)
    models = model.observation_models or {}
    expected = sum(
        channel_model.log_probability(clean, observation[channel])
        for channel, channel_model in models.items()
    )
    assert float(model.observation_log_probability(state, None, observation)[0]) == pytest.approx(
        expected
    )


def test_observation_log_probability_scores_a_list_elementwise() -> None:
    """The belief filter scores a batch of candidate observations in one call.

    Purpose: Validates the list form of the observation density

    Given: Two candidate observations, one matching the state and one far from it
    When: They are scored together
    Then: Two scores come back and the matching one is higher

    Test type: unit
    """
    model = _model()
    state = SCHEMA.pack({"base_pose": np.zeros(3), "hazard_type": [0.0, 1.0]})
    scores = model.observation_log_probability(
        state, None, [{"base_pose": np.zeros(3)}, {"base_pose": np.full(3, 5.0)}]
    )
    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_a_sample_only_channel_is_rejected_for_belief_updates() -> None:
    """A belief update that silently dropped an unscoreable channel would be wrong, not partial.

    Purpose: Validates the density-capability check across channels

    Given: A model holding one channel with no density
    When: An observation is scored
    Then: NotImplementedError is raised naming that channel

    Test type: unit
    """
    model = _model(observation_models={"counter": _CountingSignalModel()})
    state = SCHEMA.pack({"base_pose": np.zeros(3), "hazard_type": [0.0, 1.0]})
    with pytest.raises(NotImplementedError, match="counter"):
        model.observation_log_probability(state, None, {"counter": np.zeros(2)})


def test_encode_splits_a_flat_world_observation_by_the_raw_schema() -> None:
    """An Isaac world emits a flat sensor vector; the model has to name its parts.

    Purpose: Validates encode_observation against a flat raw reading

    Given: A raw observation schema declaring a 3-wide pose and a 2-wide signal
    When: A flat 5-wide world reading is encoded
    Then: It becomes the configured channels, with the values untouched

    Test type: unit
    """
    raw_schema = IsaacChannelSchema((("base_pose", 3), ("hazard_signal", 2)))
    model = _model(raw_observation_schema=raw_schema)
    encoded = model.encode_observation(np.array([1.0, 2.0, 0.5, 0.0, 1.0]))
    assert sorted(encoded) == ["base_pose"]
    assert encoded["base_pose"] == pytest.approx([1.0, 2.0, 0.5])


def test_encode_takes_a_mapping_world_observation_channel_by_channel() -> None:
    """A world whose extractor already emits channels needs no schema to split it.

    Purpose: Validates encode_observation against a mapping raw reading

    Given: A world reading already shaped as a channel mapping
    When: It is encoded
    Then: The configured channel passes through unchanged

    Test type: unit
    """
    encoded = _model().encode_observation(
        {"base_pose": np.array([1.0, 2.0, 0.5]), "hazard_signal": np.array([0.0, 1.0])}
    )
    assert encoded["base_pose"] == pytest.approx([1.0, 2.0, 0.5])


def test_encoding_a_flat_observation_without_a_schema_raises() -> None:
    """Guessing the split would silently mis-assign every channel.

    Purpose: Validates the error path when a flat reading has no schema to split it

    Given: A model with no raw_observation_schema
    When: A flat world reading is encoded
    Then: RuntimeError is raised explaining both remedies

    Test type: unit
    """
    with pytest.raises(RuntimeError, match="raw_observation_schema"):
        _model().encode_observation(np.zeros(5))


def test_a_channel_reading_an_undeclared_state_block_is_rejected_at_construction() -> None:
    """Left to run time this surfaces as a KeyError deep inside the search tree.

    Purpose: Validates the construction-time check of channels against the schema

    Given: A channel configured to read a state block the schema does not declare
    When: The model is constructed
    Then: ValueError is raised naming the channel and the block

    Test type: unit
    """
    with pytest.raises(ValueError, match="lidar"):
        _model(
            observation_models={
                "lidar": GaussianChannelObservationModel(channel="lidar", state_channel="missing")
            }
        )


def test_seed_reaches_channels_that_carry_their_own_sampler() -> None:
    """Channels with private randomness must be reseeded, and the rest left alone.

    Purpose: Validates that seed forwards to every channel exposing one

    Given: A model holding a channel with a seed method and one without
    When: The model is seeded
    Then: The seeded channel's stream repeats and no error is raised for the other

    Test type: unit
    """
    signal = LatentTypeSignalObservationModel(
        channel="hazard_signal",
        type_channel="hazard_type",
        position_channel="base_pose",
        zone_centers=[(0.0, 0.0), (5.0, 0.0)],
        zone_radii=[1.0, 1.0],
    )
    model = _model(
        observation_models={
            "base_pose": GaussianChannelObservationModel(channel="base_pose", noise_std=0.1),
            "hazard_signal": signal,
        }
    )
    clean = {"base_pose": np.zeros(3), "hazard_type": np.array([1.0, 0.0])}

    model.seed(11)
    first = np.stack([signal.perceive(clean) for _ in range(20)])
    model.seed(11)
    second = np.stack([signal.perceive(clean) for _ in range(20)])
    assert np.array_equal(first, second)


def test_observations_hash_and_compare_channel_by_channel() -> None:
    """A planner keys its observation nodes on these, so equality must be structural.

    Purpose: Validates observation hashing and equality over the channel mapping

    Given: Two equal observations and one differing in a single channel
    When: They are compared and hashed
    Then: Equal ones agree and the differing one does not

    Test type: unit
    """
    model = _model()
    first = {"base_pose": np.array([1.0, 2.0, 0.5])}
    same = {"base_pose": np.array([1.0, 2.0, 0.5])}
    other = {"base_pose": np.array([1.0, 2.0, 0.6])}
    assert model.is_equal_observation(first, same)
    assert model.hash_observation(first) == model.hash_observation(same)
    assert not model.is_equal_observation(first, other)
    assert model.hash_observation(first) != model.hash_observation(other)


def test_observations_with_different_channel_sets_are_not_equal() -> None:
    """A missing channel changes what was observed, so it cannot compare equal.

    Purpose: Validates that equality checks the channel set, not just shared keys

    Given: Two observations agreeing on one channel, one carrying an extra channel
    When: They are compared
    Then: They are unequal

    Test type: unit
    """
    model = _model()
    assert not model.is_equal_observation(
        {"base_pose": np.zeros(3)}, {"base_pose": np.zeros(3), "lidar": np.zeros(4)}
    )


def test_initial_distributions_direct_the_caller_to_the_world() -> None:
    """A model with an invented prior would seed the belief somewhere the world never is.

    Purpose: Validates that both initial-distribution hooks refuse

    Given: A factored model
    When: Either initial distribution is requested
    Then: NotImplementedError points at the world's initial observation

    Test type: unit
    """
    model = _model()
    with pytest.raises(NotImplementedError, match="world's initial observation"):
        model.initial_state_dist()
    with pytest.raises(NotImplementedError, match="world's initial observation"):
        model.initial_observation_dist()


def test_actions_are_returned_as_copies() -> None:
    """A caller mutating a returned action would silently redefine the action set.

    Purpose: Validates that get_actions does not hand out the stored presets

    Given: A model with two action presets
    When: A returned action is mutated
    Then: The next call is unaffected

    Test type: unit
    """
    model = _model()
    model.get_actions()[0][0] = 99.0
    assert model.get_actions()[0] == pytest.approx(np.zeros(3))
