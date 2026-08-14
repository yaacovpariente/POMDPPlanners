# SPDX-License-Identifier: MIT

"""Tests for the numeric helpers the IsaacLab stack shares."""

import numpy as np
import pytest
import torch

from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_helpers import (
    first_row,
    spatial_hash_primes,
)


class TestFirstRow:
    """Detaching a batched environment reading down to one flat host vector."""

    def test_first_row_returns_leading_row_of_numpy_batch(self):
        """Test that a numpy batch is reduced to its first row.

        Purpose: Validates the common case of a single-environment reading in a batched array

        Given: A (2, 2) array whose rows differ
        When: first_row is called on it
        Then: The leading row is returned and the trailing row is discarded

        Test type: unit
        """
        assert first_row(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [1.0, 2.0]

    def test_first_row_detaches_torch_tensor_to_numpy(self):
        """Test that a torch tensor carrying grad is detached to a numpy array.

        Purpose: Validates that a live IsaacLab reading crosses to the host without a grad error

        Given: A (1, 3) torch tensor with requires_grad set
        When: first_row is called on it
        Then: A numpy array of the row is returned, which numpy conversion would refuse without
            the detach

        Test type: unit
        """
        tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        result = first_row(tensor)

        assert isinstance(result, np.ndarray)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_first_row_flattens_multi_dimensional_row(self):
        """Test that a row with its own inner axes is flattened to one dimension.

        Purpose: Validates the reshape(-1), which callers rely on to index by a flat offset

        Given: A (1, 2, 2) array, so the first environment's reading is itself 2-D
        When: first_row is called on it
        Then: A flat 4-entry vector is returned

        Test type: unit
        """
        result = first_row(np.arange(4.0).reshape(1, 2, 2))

        assert result.shape == (4,)
        assert result.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_first_row_on_empty_batch_raises(self):
        """Test that a batch with no environments raises rather than returning empty.

        Purpose: Validates that a degenerate reading surfaces instead of propagating silently

        Given: A (0, 3) array, which is a batch over zero environments
        When: first_row is called on it
        Then: IndexError is raised

        Test type: unit
        """
        with pytest.raises(IndexError):
            first_row(np.zeros((0, 3)))


class TestSpatialHashPrimes:
    """The per-dimension weights a vectorized model hashes observations with."""

    def test_spatial_hash_primes_returns_requested_count_of_distinct_primes(self):
        """Test that the table has the requested width and no repeats.

        Purpose: Validates that each observation dimension gets its own weight

        Given: A request for 12 primes
        When: spatial_hash_primes is called
        Then: 12 entries are returned, all distinct, all prime

        Test type: unit
        """
        primes = spatial_hash_primes(12)

        assert primes.shape == (12,)
        assert len(set(primes.tolist())) == 12
        for value in primes.tolist():
            assert all(value % divisor != 0 for divisor in range(2, int(value**0.5) + 1))

    def test_spatial_hash_primes_is_deterministic_across_calls(self):
        """Test that two calls produce the same table.

        Purpose: Validates the property the hoist exists to protect — two models that hash the
            same observation must agree on the weights, or they bucket it differently

        Given: Two independent calls for the same width
        When: Their results are compared
        Then: They are equal entry for entry

        Test type: unit
        """
        assert np.array_equal(spatial_hash_primes(8), spatial_hash_primes(8))

    def test_spatial_hash_primes_is_a_prefix_as_the_width_grows(self):
        """Test that a wider request extends the narrower table rather than replacing it.

        Purpose: Validates that a dimension's weight depends on its index alone, so widening a
            state does not silently rekey the dimensions that were already there

        Given: A 4-wide and a 9-wide table
        When: The 9-wide table is truncated to its first four entries
        Then: It equals the 4-wide table

        Test type: unit
        """
        assert np.array_equal(spatial_hash_primes(9)[:4], spatial_hash_primes(4))

    def test_spatial_hash_primes_returns_int64(self):
        """Test that the table is int64.

        Purpose: Validates the dtype the torch key tensors are built from; a narrower integer
            would overflow when the primes are multiplied by quantized coordinates

        Given: A request for 3 primes
        When: The result's dtype is read
        Then: It is int64

        Test type: unit
        """
        assert spatial_hash_primes(3).dtype == np.int64

    def test_spatial_hash_primes_of_zero_returns_empty_table(self):
        """Test that a zero-width request returns an empty table rather than looping.

        Purpose: Validates the boundary a zero-dimensional state would hit

        Given: A request for 0 primes
        When: spatial_hash_primes is called
        Then: An empty int64 array is returned

        Test type: unit
        """
        primes = spatial_hash_primes(0)

        assert primes.shape == (0,)
        assert primes.dtype == np.int64
