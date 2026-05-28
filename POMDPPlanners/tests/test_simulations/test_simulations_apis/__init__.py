# SPDX-License-Identifier: MIT

"""Tests for simulation API implementations.

This package contains tests for all simulation API implementations
(LocalSimulationsAPI, DaskSimulationsAPI, PBSSimulationsAPI) using
shared test mixins to ensure consistent behavior across implementations.
"""

import pytest

pytest_plugins = ["POMDPPlanners.tests.test_simulations.test_simulations_apis.api_test_fixtures"]
