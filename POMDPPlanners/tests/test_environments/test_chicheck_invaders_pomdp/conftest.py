# SPDX-License-Identifier: MIT

"""Seed the Chicheck Invaders native RNG before every test in this directory.

The environment's sampling runs in C++ and draws from that extension's own
RNG, which ``np.random.seed`` does not reach and which otherwise starts from
``std::random_device``. Without this fixture a test that seeds NumPy to pin an
episode would still see a different episode on every run.
"""

import pytest

from POMDPPlanners.environments.chicheck_invaders_pomdp import (  # pylint: disable=no-name-in-module
    _native,
)


@pytest.fixture(autouse=True)
def seed_chicheck_invaders_native_rng():
    """Seed the native RNG to a fixed value before each test."""
    _native.set_seed(0)
