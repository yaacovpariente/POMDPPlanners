"""Tests for Light Dark POMDP observation models.

This module tests the observation models from light_dark_observation_models.py, focusing on:
- Base observation model functionality
- Continuous light dark normal noise observation model
- Continuous light dark normal noise no observation in dark model
"""

# pylint: disable=protected-access  # Tests need to access protected members

import random
from typing import List, Union

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from POMDPPlanners.environments.light_dark_pomdp.light_dark_pomdp_utils.light_dark_observation_models import (
    BaseLightDarkObservationModel,
    ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel,
    ContinuousLightDarkNormalNoiseObservationModel,
)

# Set seeds for reproducible tests
np.random.seed(42)
random.seed(42)


class TestBaseLightDarkObservationModel:
    """Test cases for base observation model."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseLightDarkObservationModel(  # type: ignore  # pylint: disable=abstract-class-instantiated
                next_state=np.array([1.0, 2.0]),
                action=np.array([0.5, -0.3]),
                observation_cov_matrix=np.eye(2),
                grid_size=10,
                beacons=np.array([[0, 5], [0, 5]]),
                beacon_radius=1.0,
            )

    def test_near_beacon_detection(self):
        """Test that near_beacon is correctly detected."""

        # Create a concrete implementation for testing
        class ConcreteObservationModel(BaseLightDarkObservationModel):
            def sample(self, n_samples: int = 1) -> List[np.ndarray]:
                return [np.array([0.0, 0.0])] * n_samples

        # State near beacon
        next_state_near = np.array([0.5, 0.5])
        beacons = np.array([[0, 5], [0, 5]])  # Beacon at (0,0) and (5,5)
        beacon_radius = 1.0

        model_near = ConcreteObservationModel(
            next_state=next_state_near,
            action=np.array([0, 0]),
            observation_cov_matrix=np.eye(2),
            grid_size=10,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        assert model_near.near_beacon == True, "Should detect proximity to beacon"

        # State far from beacon
        next_state_far = np.array([3.0, 3.0])
        model_far = ConcreteObservationModel(
            next_state=next_state_far,
            action=np.array([0, 0]),
            observation_cov_matrix=np.eye(2),
            grid_size=10,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        assert model_far.near_beacon == False, "Should not detect proximity when far from beacons"


class TestContinuousLightDarkNormalNoiseObservationModel:
    """Test cases for ContinuousLightDarkNormalNoiseObservationModel."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.next_state = np.array([5.0, 3.0])
        self.action = np.array([0, 0])
        self.observation_cov_matrix = np.eye(2) * 0.25
        self.grid_size = 11
        self.beacons = np.array([[0, 10], [0, 10]])  # Beacons at (0,0) and (10,10)
        self.beacon_radius = 1.0

    def test_initialization_near_beacon_reduces_covariance(self):
        """Test initialization when state is near a beacon reduces covariance."""
        next_state_near = np.array([0.5, 0.5])  # Near beacon at (0,0)
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        assert obs_model.near_beacon == True, "Should detect proximity to beacon"
        expected_cov = self.observation_cov_matrix * 0.5
        assert np.array_equal(
            obs_model.observation_cov_matrix, expected_cov
        ), f"Covariance should be reduced by 0.5 when near beacon. Expected {expected_cov}, got {obs_model.observation_cov_matrix}"

    def test_initialization_far_from_beacon_preserves_covariance(self):
        """Test initialization when state is far from beacons preserves covariance."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        assert obs_model.near_beacon == False, "Should not detect proximity when far from beacons"
        assert np.array_equal(
            obs_model.observation_cov_matrix, self.observation_cov_matrix
        ), "Covariance should remain unchanged when far from beacons"

    def test_sample_always_returns_observations(self):
        """Test that sampling always returns actual observations (never None)."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Sample multiple observations
        observations = obs_model.sample(n_samples=10)

        assert len(observations) == 10, "Should return requested number of samples"
        assert all(obs is not None for obs in observations), "All observations should be non-None"
        assert all(
            isinstance(obs, np.ndarray) for obs in observations
        ), "All observations should be numpy arrays"
        assert all(
            obs.shape == (2,) for obs in observations
        ), "All observations should be 2D vectors"

    def test_sample_near_beacon_returns_observations(self):
        """Test that sampling near beacon returns actual observations."""
        next_state_near = np.array([0.5, 0.5])  # Near beacon at (0,0)
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Sample multiple observations
        observations = obs_model.sample(n_samples=10)

        assert len(observations) == 10, "Should return requested number of samples"
        assert all(
            obs is not None for obs in observations
        ), "All observations should be non-None when near beacon"
        assert all(
            isinstance(obs, np.ndarray) for obs in observations
        ), "All observations should be numpy arrays"

        # Check that observations are within reasonable range (3 standard deviations)
        # Note: covariance is reduced by 0.5, so std is sqrt(0.5 * 0.25) = sqrt(0.125) ≈ 0.35
        reduced_cov = self.observation_cov_matrix * 0.5
        std = np.sqrt(np.diag(reduced_cov))[0]
        for obs in observations:
            assert np.allclose(
                obs, next_state_near, atol=3 * std
            ), f"Observation {obs} should be close to next_state {next_state_near}"

    def test_sample_single_observation(self):
        """Test sampling a single observation."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        observations = obs_model.sample(n_samples=1)

        assert len(observations) == 1, "Should return one observation"
        assert observations[0] is not None, "Observation should not be None"
        assert isinstance(observations[0], np.ndarray), "Observation should be numpy array"
        assert observations[0].shape == (2,), "Observation should be 2D vector"

    def test_sample_observations_clipped_to_grid(self):
        """Test that observations are clipped to grid boundaries."""
        next_state_near = np.array([0.5, 0.5])
        # Use large covariance to ensure some samples go outside grid
        large_cov = np.eye(2) * 10.0
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=large_cov,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        observations = obs_model.sample(n_samples=100)

        for obs in observations:
            assert obs is not None, "Observations should not be None"
            assert np.all(obs >= 0), f"Observation {obs} should be >= 0"
            assert np.all(obs <= self.grid_size), f"Observation {obs} should be <= grid_size"

    def test_probability_single_observation(self):
        """Test probability calculation for single observation."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Test single observation value
        observation_values = [np.array([5.0, 3.0])]  # Same as next_state (mean)
        probabilities = obs_model.probability(observation_values)

        # Calculate expected probability using scipy
        expected_prob = multivariate_normal.pdf(
            observation_values[0], mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
        )

        assert isinstance(probabilities, np.ndarray), "Probability should return numpy array"
        assert len(probabilities) == 1, "Should return one probability for one observation"
        assert np.isclose(
            probabilities[0], expected_prob, rtol=1e-10
        ), f"Probability {probabilities[0]} should match expected {expected_prob}"

    def test_probability_multiple_observations(self):
        """Test probability calculation for multiple observations."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Test multiple observation values
        observation_values = [
            np.array([5.0, 3.0]),  # At mean
            np.array([5.5, 3.5]),  # Offset from mean
            np.array([4.0, 2.0]),  # Different offset
        ]
        probabilities = obs_model.probability(observation_values)

        # Calculate expected probabilities
        expected_probs = multivariate_normal.pdf(
            observation_values, mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
        )

        assert len(probabilities) == 3, "Should return three probabilities for three observations"
        assert np.allclose(
            probabilities, expected_probs, rtol=1e-10
        ), f"Probabilities {probabilities} should match expected {expected_probs}"

    def test_probability_decreases_with_distance_from_mean(self):
        """Test that probability decreases with distance from mean."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        close_obs = np.array([5.1, 3.1])  # Close to mean
        far_obs = np.array([6.0, 4.0])  # Far from mean

        close_prob = obs_model.probability([close_obs])[0]
        far_prob = obs_model.probability([far_obs])[0]

        assert (
            close_prob > far_prob
        ), f"Probability for closer observation {close_prob} should be higher than farther {far_prob}"

    def test_probability_with_reduced_covariance_near_beacon(self):
        """Test that probability uses reduced covariance when near beacon."""
        next_state_near = np.array([0.5, 0.5])  # Close to beacon at (0,0)
        observation_cov_matrix = np.eye(2) * 1.0
        beacons = np.array([[0, 5], [0, 5]])  # Beacon at (0,0) and (5,5)
        beacon_radius = 1.0

        # Create observation model near beacon (covariance will be reduced)
        obs_model_near = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Create observation model far from beacon (covariance unchanged)
        next_state_far = np.array([3.0, 3.0])
        obs_model_far = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_far,
            action=self.action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Test probability for same observation value
        observation_value = np.array([0.6, 0.6])
        prob_near = obs_model_near.probability([observation_value])[0]
        prob_far = obs_model_far.probability([observation_value])[0]

        # Near-beacon model should have higher probability density due to reduced covariance
        # (tighter distribution means higher density at the same point)
        assert (
            prob_near > prob_far
        ), f"Near-beacon probability {prob_near} should be higher than far-beacon {prob_far} due to reduced covariance"

    def test_covariance_reduction_affects_sampling(self):
        """Test that reduced covariance near beacon affects observation distribution."""
        next_state_near = np.array([0.5, 0.5])
        observation_cov_matrix = np.eye(2) * 2.0
        beacons = np.array([[0], [0]])
        beacon_radius = 1.0

        # Model near beacon (reduced covariance)
        obs_model_near = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Model far from beacon (normal covariance)
        next_state_far = np.array([5.0, 5.0])
        obs_model_far = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_far,
            action=self.action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Sample many observations
        obs_near = obs_model_near.sample(n_samples=1000)
        obs_far = obs_model_far.sample(n_samples=1000)

        # Calculate variance of observations
        obs_near_array = np.array(obs_near)
        obs_far_array = np.array(obs_far)

        var_near = np.var(obs_near_array, axis=0)
        var_far = np.var(obs_far_array, axis=0)

        # Near-beacon observations should have lower variance due to reduced covariance
        assert np.all(
            var_near < var_far
        ), f"Near-beacon variance {var_near} should be lower than far-beacon {var_far}"

    def test_beacon_proximity_with_multiple_beacons(self):
        """Test beacon proximity detection with multiple beacons."""
        observation_cov_matrix = np.eye(2) * 2.0
        grid_size = 11
        # Multiple beacons: (0,0), (5,5), (10,10)
        beacons = np.array([[0, 5, 10], [0, 5, 10]])
        beacon_radius = 1.5
        action = np.array([0, 0])

        # Test near first beacon (0,0)
        near_first_beacon = np.array([1.0, 1.0])
        obs_model_1 = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=near_first_beacon,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Test near middle beacon (5,5)
        near_middle_beacon = np.array([5.0, 6.0])
        obs_model_2 = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=near_middle_beacon,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Test far from all beacons
        far_state = np.array([3.0, 3.0])
        obs_model_3 = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=far_state,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Verify proximity detection
        assert obs_model_1.near_beacon == True, "Should detect proximity to first beacon"
        assert obs_model_2.near_beacon == True, "Should detect proximity to middle beacon"
        assert obs_model_3.near_beacon == False, "Should not detect proximity when far"

        # Verify covariance reduction
        expected_reduced_cov = observation_cov_matrix * 0.5
        expected_normal_cov = observation_cov_matrix.copy()

        assert np.array_equal(
            obs_model_1.observation_cov_matrix, expected_reduced_cov
        ), "Covariance should be reduced near beacon 1"
        assert np.array_equal(
            obs_model_2.observation_cov_matrix, expected_reduced_cov
        ), "Covariance should be reduced near beacon 2"
        assert np.array_equal(
            obs_model_3.observation_cov_matrix, expected_normal_cov
        ), "Covariance should remain normal when far from beacons"

    def test_edge_case_exactly_on_beacon_radius(self):
        """Test behavior when state is exactly on beacon radius boundary."""
        next_state_on_boundary = np.array([1.0, 0.0])  # Exactly 1.0 from beacon at (0,0)
        beacons = np.array([[0], [0]])
        beacon_radius = 1.0

        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_on_boundary,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Should be considered near beacon (distance == radius)
        assert obs_model.near_beacon == True, "Should detect proximity when exactly on radius"
        expected_cov = self.observation_cov_matrix * 0.5
        assert np.array_equal(
            obs_model.observation_cov_matrix, expected_cov
        ), "Covariance should be reduced when on boundary"

    def test_edge_case_just_outside_beacon_radius(self):
        """Test behavior when state is just outside beacon radius."""
        # Use sqrt(2) + epsilon to be just outside radius 1.0
        epsilon = 0.01
        next_state_outside = np.array([1.0 + epsilon, 1.0 + epsilon])
        beacons = np.array([[0], [0]])
        beacon_radius = 1.0

        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=next_state_outside,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Should not be considered near beacon
        assert obs_model.near_beacon == False, "Should not detect proximity when outside radius"
        assert np.array_equal(
            obs_model.observation_cov_matrix, self.observation_cov_matrix
        ), "Covariance should remain unchanged when outside boundary"

    def test_vectorized_probability_calculation(self):
        """Test that probability calculation works correctly for vectorized input."""
        obs_model = ContinuousLightDarkNormalNoiseObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Test with many observations
        n_obs = 100
        observation_values = [
            np.array([5.0 + np.random.normal(0, 0.5), 3.0 + np.random.normal(0, 0.5)])
            for _ in range(n_obs)
        ]

        probabilities = obs_model.probability(observation_values)

        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == n_obs, f"Should return {n_obs} probabilities"
        assert np.all(probabilities >= 0), "All probabilities should be non-negative"
        assert np.all(np.isfinite(probabilities)), "All probabilities should be finite"


class TestContinuousLightDarkNormalNoiseNoObsInDarkObservationModel:
    """Test cases for ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.next_state = np.array([5.0, 3.0])
        self.action = np.array([0, 0])
        self.observation_cov_matrix = np.eye(2) * 0.25
        self.grid_size = 11
        self.beacons = np.array([[0, 10], [0, 10]])  # Beacons at (0,0) and (10,10)
        self.beacon_radius = 1.0

    def test_initialization_near_beacon(self):
        """Test initialization when state is near a beacon."""
        next_state_near = np.array([0.5, 0.5])  # Near beacon at (0,0)
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        assert obs_model.near_beacon == True, "Should detect proximity to beacon"
        assert np.array_equal(
            obs_model.observation_cov_matrix, self.observation_cov_matrix
        ), "Covariance should not be modified (unlike normal noise model)"

    def test_initialization_far_from_beacon(self):
        """Test initialization when state is far from beacons."""
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        assert obs_model.near_beacon == False, "Should not detect proximity when far from beacons"
        assert np.array_equal(
            obs_model.observation_cov_matrix, self.observation_cov_matrix
        ), "Covariance should remain unchanged"

    def test_sample_near_beacon_returns_observations(self):
        """Test that sampling near beacon returns actual observations."""
        next_state_near = np.array([0.5, 0.5])  # Near beacon at (0,0)
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Sample multiple observations
        observations = obs_model.sample(n_samples=10)

        assert len(observations) == 10, "Should return requested number of samples"
        assert all(
            obs is not None for obs in observations
        ), "All observations should be non-None when near beacon"
        assert all(
            isinstance(obs, np.ndarray) for obs in observations
        ), "All observations should be numpy arrays"
        assert all(
            obs.shape == (2,) for obs in observations
        ), "All observations should be 2D vectors"

        # Check that observations are within reasonable range (3 standard deviations)
        for obs in observations:
            assert obs is not None, "Observation should not be None"
            assert np.allclose(
                obs, next_state_near, atol=3.0
            ), f"Observation {obs} should be close to next_state {next_state_near}"

    def test_sample_far_from_beacon_returns_none(self):
        """Test that sampling far from beacon returns None."""
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Sample multiple observations
        observations = obs_model.sample(n_samples=10)

        assert len(observations) == 10, "Should return requested number of samples"
        assert all(
            obs is None for obs in observations
        ), "All observations should be None when far from beacon"

    def test_sample_single_observation_near_beacon(self):
        """Test sampling a single observation near beacon."""
        next_state_near = np.array([0.5, 0.5])
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        observations = obs_model.sample(n_samples=1)

        assert len(observations) == 1, "Should return one observation"
        assert observations[0] is not None, "Observation should not be None"
        assert isinstance(observations[0], np.ndarray), "Observation should be numpy array"

    def test_sample_single_observation_far_from_beacon(self):
        """Test sampling a single observation far from beacon."""
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        observations = obs_model.sample(n_samples=1)

        assert len(observations) == 1, "Should return one observation"
        assert observations[0] is None, "Observation should be None when far from beacon"

    def test_sample_observations_clipped_to_grid(self):
        """Test that observations are clipped to grid boundaries."""
        next_state_near = np.array([0.5, 0.5])
        # Use large covariance to ensure some samples go outside grid
        large_cov = np.eye(2) * 10.0
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=large_cov,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        observations = obs_model.sample(n_samples=100)

        for obs in observations:
            assert obs is not None, "Observations should not be None near beacon"
            assert np.all(obs >= 0), f"Observation {obs} should be >= 0"
            assert np.all(obs <= self.grid_size), f"Observation {obs} should be <= grid_size"

    def test_probability_none_value_when_far_from_beacon(self):
        """Test probability calculation for None when far from beacon."""
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Probability of None when far from beacon should be 1
        probabilities = obs_model.probability([None])

        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == 1, "Should return one probability"
        assert np.isclose(
            probabilities[0], 1.0
        ), f"Probability of None when far from beacon should be 1.0, got {probabilities[0]}"

    def test_probability_none_value_when_near_beacon(self):
        """Test probability calculation for None when near beacon."""
        next_state_near = np.array([0.5, 0.5])
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Probability of None when near beacon should be 0
        probabilities = obs_model.probability([None])

        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == 1, "Should return one probability"
        assert np.isclose(
            probabilities[0], 0.0
        ), f"Probability of None when near beacon should be 0.0, got {probabilities[0]}"

    def test_probability_actual_observation_when_near_beacon(self):
        """Test probability calculation for actual observation when near beacon."""
        next_state_near = np.array([5.0, 3.0])
        # Move beacon closer to next_state
        beacons_near = np.array([[5, 10], [3, 10]])
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons_near,
            beacon_radius=self.beacon_radius,
        )

        # Test probability of observation at mean
        observation_values = [np.array([5.0, 3.0])]
        probabilities = obs_model.probability(observation_values)  # type: ignore

        # Calculate expected probability using scipy
        expected_prob = multivariate_normal.pdf(
            observation_values[0], mean=next_state_near, cov=self.observation_cov_matrix  # type: ignore
        )

        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == 1, "Should return one probability"
        assert np.isclose(
            probabilities[0], expected_prob, rtol=1e-10
        ), f"Probability {probabilities[0]} should match expected {expected_prob}"

    def test_probability_actual_observation_when_far_from_beacon(self):
        """Test probability calculation for actual observation when far from beacon."""
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Test probability of actual observation when far from beacon
        # Should still calculate normal probability (even though sampling returns None)
        observation_values = [np.array([5.0, 3.0])]
        probabilities = obs_model.probability(observation_values)  # type: ignore

        # Calculate expected probability using scipy
        expected_prob = multivariate_normal.pdf(
            observation_values[0], mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
        )

        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == 1, "Should return one probability"
        assert np.isclose(
            probabilities[0], expected_prob, rtol=1e-10
        ), f"Probability {probabilities[0]} should match expected {expected_prob}"

    def test_probability_mixed_none_and_observations(self):
        """Test probability calculation with mixed None and actual observations."""
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=self.next_state,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=self.beacons,
            beacon_radius=self.beacon_radius,
        )

        # Mix of None and actual observations
        observation_values: List[Union[np.ndarray, None]] = [
            None,
            np.array([5.0, 3.0]),
            None,
            np.array([5.5, 3.5]),
        ]
        probabilities = obs_model.probability(observation_values)

        assert isinstance(probabilities, np.ndarray), "Should return numpy array"
        assert len(probabilities) == 4, "Should return four probabilities"

        # First None should have probability 1 (far from beacon)
        assert np.isclose(probabilities[0], 1.0), "First None should have probability 1.0"

        # Second observation should have normal probability
        assert observation_values[1] is not None, "Second observation should not be None"
        expected_prob_2 = multivariate_normal.pdf(
            observation_values[1], mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
        )
        assert np.isclose(
            probabilities[1], expected_prob_2, rtol=1e-10
        ), "Second observation should have normal probability"

        # Third None should have probability 1 (far from beacon)
        assert np.isclose(probabilities[2], 1.0), "Third None should have probability 1.0"

        # Fourth observation should have normal probability
        assert observation_values[3] is not None, "Fourth observation should not be None"
        expected_prob_4 = multivariate_normal.pdf(
            observation_values[3], mean=self.next_state, cov=self.observation_cov_matrix  # type: ignore
        )
        assert np.isclose(
            probabilities[3], expected_prob_4, rtol=1e-10
        ), "Fourth observation should have normal probability"

    def test_probability_multiple_observations_near_beacon(self):
        """Test probability calculation for multiple observations when near beacon."""
        next_state_near = np.array([5.0, 3.0])
        beacons_near = np.array([[5, 10], [3, 10]])
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons_near,
            beacon_radius=self.beacon_radius,
        )

        observation_values = [
            np.array([5.0, 3.0]),  # At mean
            np.array([5.5, 3.5]),  # Offset from mean
            np.array([4.0, 2.0]),  # Different offset
        ]
        probabilities = obs_model.probability(observation_values)  # type: ignore

        # Calculate expected probabilities
        expected_probs = multivariate_normal.pdf(
            observation_values, mean=next_state_near, cov=self.observation_cov_matrix  # type: ignore
        )

        assert len(probabilities) == 3, "Should return three probabilities"
        assert np.allclose(
            probabilities, expected_probs, rtol=1e-10
        ), f"Probabilities {probabilities} should match expected {expected_probs}"

    def test_probability_decreases_with_distance_from_mean(self):
        """Test that probability decreases with distance from mean."""
        next_state_near = np.array([5.0, 3.0])
        beacons_near = np.array([[5, 10], [3, 10]])
        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_near,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons_near,
            beacon_radius=self.beacon_radius,
        )

        close_obs = np.array([5.1, 3.1])  # Close to mean
        far_obs = np.array([6.0, 4.0])  # Far from mean

        close_prob = obs_model.probability([close_obs])[0]
        far_prob = obs_model.probability([far_obs])[0]

        assert (
            close_prob > far_prob
        ), f"Probability for closer observation {close_prob} should be higher than farther {far_prob}"

    def test_beacon_proximity_with_multiple_beacons(self):
        """Test beacon proximity detection with multiple beacons."""
        observation_cov_matrix = np.eye(2) * 2.0
        grid_size = 11
        # Multiple beacons: (0,0), (5,5), (10,10)
        beacons = np.array([[0, 5, 10], [0, 5, 10]])
        beacon_radius = 1.5
        action = np.array([0, 0])

        # Test near first beacon (0,0)
        near_first_beacon = np.array([1.0, 1.0])
        obs_model_1 = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=near_first_beacon,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Test near middle beacon (5,5)
        near_middle_beacon = np.array([5.0, 6.0])
        obs_model_2 = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=near_middle_beacon,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Test far from all beacons
        far_state = np.array([3.0, 3.0])
        obs_model_3 = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=far_state,
            action=action,
            observation_cov_matrix=observation_cov_matrix,
            grid_size=grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Verify proximity detection
        assert obs_model_1.near_beacon == True, "Should detect proximity to first beacon"
        assert obs_model_2.near_beacon == True, "Should detect proximity to middle beacon"
        assert obs_model_3.near_beacon == False, "Should not detect proximity when far"

        # Verify sampling behavior
        obs_1 = obs_model_1.sample(n_samples=10)
        obs_2 = obs_model_2.sample(n_samples=10)
        obs_3 = obs_model_3.sample(n_samples=10)

        assert all(o is not None for o in obs_1), "Should return observations near beacon 1"
        assert all(o is not None for o in obs_2), "Should return observations near beacon 2"
        assert all(o is None for o in obs_3), "Should return None when far from beacons"

    def test_edge_case_exactly_on_beacon_radius(self):
        """Test behavior when state is exactly on beacon radius boundary."""
        next_state_on_boundary = np.array([1.0, 0.0])  # Exactly 1.0 from beacon at (0,0)
        beacons = np.array([[0], [0]])
        beacon_radius = 1.0

        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_on_boundary,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Should be considered near beacon (distance == radius)
        assert obs_model.near_beacon == True, "Should detect proximity when exactly on radius"
        observations = obs_model.sample(n_samples=5)
        assert all(
            o is not None for o in observations
        ), "Should return observations when on boundary"

    def test_edge_case_just_outside_beacon_radius(self):
        """Test behavior when state is just outside beacon radius."""
        # Use sqrt(2) + epsilon to be just outside radius 1.0
        epsilon = 0.01
        next_state_outside = np.array([1.0 + epsilon, 1.0 + epsilon])
        beacons = np.array([[0], [0]])
        beacon_radius = 1.0

        obs_model = ContinuousLightDarkNormalNoiseNoObsInDarkObservationModel(
            next_state=next_state_outside,
            action=self.action,
            observation_cov_matrix=self.observation_cov_matrix,
            grid_size=self.grid_size,
            beacons=beacons,
            beacon_radius=beacon_radius,
        )

        # Should not be considered near beacon
        assert obs_model.near_beacon == False, "Should not detect proximity when outside radius"
        observations = obs_model.sample(n_samples=5)
        assert all(o is None for o in observations), "Should return None when outside boundary"
