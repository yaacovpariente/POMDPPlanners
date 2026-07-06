# SPDX-License-Identifier: MIT

"""Tests for the CarDreamer/DreamerV3 adapter's framework-agnostic packing helpers.

The JAX-backed :class:`CarDreamerWorldModel` itself needs ``jax``/``dreamerv3`` and a
trained checkpoint, so it is not exercised here. The latent pack/unpack helpers that bridge
the flat planner latent and the DreamerV3 ``{deter, stoch}`` state dict are pure NumPy and
are covered directly.
"""

import numpy as np

from POMDPPlanners.environments.carla_pomdp.carla_generative_models.cardreamer_world_model import (
    _pack_latents,
    _unpack_latents,
)


def test_pack_unpack_latents_round_trip_categorical_stoch():
    """Packing then unpacking a latent recovers the deterministic and stochastic parts.

    Purpose: Validates the flat-latent <-> RSSM-state bridge is lossless for categorical stoch.

    Given: A batch of deterministic vectors and categorical (2-D per row) stochastic tensors.
    When: They are packed into a flat latent and unpacked with the same dimensions.
    Then: The recovered deter and stoch equal the originals with the expected latent width.

    Test type: unit
    """
    deter_dim = 4
    stoch_shape = (3, 2)
    deter = np.arange(2 * deter_dim, dtype=np.float32).reshape(2, deter_dim)
    stoch = np.arange(2 * 6, dtype=np.float32).reshape((2, *stoch_shape))

    packed = _pack_latents(deter, stoch)
    recovered_deter, recovered_stoch = _unpack_latents(packed, deter_dim, stoch_shape)

    assert packed.shape == (2, deter_dim + int(np.prod(stoch_shape)))
    np.testing.assert_array_equal(recovered_deter, deter)
    np.testing.assert_array_equal(recovered_stoch, stoch)


def test_unpack_latents_promotes_single_vector_to_batch():
    """A 1-D latent is treated as a batch of one before splitting.

    Purpose: Validates unpacking accepts an unbatched latent vector.

    Given: A single flat latent vector concatenating a 2-wide deter and a length-3 stoch.
    When: It is unpacked with those dimensions.
    Then: The result carries a leading batch axis of one and the correct split.

    Test type: unit
    """
    deter_dim = 2
    stoch_shape = (3,)
    latent = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    deter, stoch = _unpack_latents(latent, deter_dim, stoch_shape)

    assert deter.shape == (1, deter_dim)
    assert stoch.shape == (1, *stoch_shape)
    np.testing.assert_array_equal(deter[0], [1.0, 2.0])
    np.testing.assert_array_equal(stoch[0], [3.0, 4.0, 5.0])
