"""NumPy type hints for POMDPPlanners.

This module provides comprehensive type hints for numpy arrays commonly used
throughout the POMDPPlanners codebase. These type hints are designed to pass
pyright checks and improve type safety.

Usage:
    from type_hints import StateVector, ObservationVector, ProbabilityArray
    from numpy.typing import NDArray
    import numpy as np

    def process_state(state: StateVector) -> StateVector:
        return state

    def compute_probabilities(obs: ObservationVector) -> ProbabilityArray:
        return np.array([0.1, 0.2, 0.7])
"""

from typing import Any, TypeVar, Union
import numpy as np
from numpy.typing import NDArray

# Type variables for generic array dimensions
T = TypeVar("T", bound=np.generic)

# =============================================================================
# Basic NumPy Array Type Aliases
# =============================================================================

# Generic array types
Array = NDArray[Any]
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]

# Specific dtype arrays
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]

# Additional numeric array types
NumericArray = NDArray[Union[np.floating[Any], np.integer[Any]]]  # Any numeric array
ComplexArray = NDArray[np.complexfloating[Any]]  # Complex number arrays
UnsignedIntArray = NDArray[np.unsignedinteger[Any]]  # Unsigned integer arrays
SignedIntArray = NDArray[np.signedinteger[Any]]  # Signed integer arrays

# =============================================================================
# POMDP-Specific Array Types
# =============================================================================

# State vectors (commonly 1D or 2D)
StateVector = NDArray[np.floating[Any]]  # 1D state vector
StateVector2D = NDArray[np.floating[Any]]  # 2D state vector (x, y coordinates)
StateVector4D = NDArray[np.floating[Any]]  # 4D state vector (position + velocity)
StateVector6D = NDArray[np.floating[Any]]  # 6D state vector (complex state)

# Observation vectors
ObservationVector = NDArray[np.floating[Any]]  # 1D observation
ObservationVector2D = NDArray[np.floating[Any]]  # 2D observation (x, y)
ObservationVector4D = NDArray[np.floating[Any]]  # 4D observation
ObservationVector6D = NDArray[np.floating[Any]]  # 6D observation

# Probability arrays
ProbabilityArray = NDArray[np.floating[Any]]  # Array of probabilities
LogProbabilityArray = NDArray[np.floating[Any]]  # Array of log probabilities

# =============================================================================
# Coordinate and Geometric Arrays
# =============================================================================

# 2D coordinate arrays (x, y pairs)
Coordinate2D = NDArray[np.floating[Any]]  # Single (x, y) coordinate
Coordinates2D = NDArray[np.floating[Any]]  # Multiple (x, y) coordinates

# Beacon and obstacle arrays (2xN format: first row x, second row y)
BeaconArray = NDArray[np.floating[Any]]  # 2xN array of beacon coordinates
ObstacleArray = NDArray[np.floating[Any]]  # 2xN array of obstacle coordinates

# =============================================================================
# Matrix and Covariance Arrays
# =============================================================================

# Covariance matrices (2x2 for 2D, 4x4 for 4D, etc.)
CovarianceMatrix2D = NDArray[np.floating[Any]]  # 2x2 covariance matrix
CovarianceMatrix4D = NDArray[np.floating[Any]]  # 4x4 covariance matrix
CovarianceMatrix = NDArray[np.floating[Any]]  # Generic covariance matrix

# Transition matrices
TransitionMatrix = NDArray[np.floating[Any]]  # State transition matrix
ObservationMatrix = NDArray[np.floating[Any]]  # Observation matrix

# =============================================================================
# Particle and Belief Arrays
# =============================================================================

# Particle arrays for particle filters
ParticleArray = NDArray[np.floating[Any]]  # Array of particles
ParticleWeights = NDArray[np.floating[Any]]  # Particle weights
LogParticleWeights = NDArray[np.floating[Any]]  # Log particle weights

# Belief representation arrays
BeliefArray = NDArray[np.floating[Any]]  # Belief state representation

# =============================================================================
# Action and Reward Arrays
# =============================================================================

# Action arrays
ActionArray = NDArray[np.integer[Any]]  # Discrete action indices
ContinuousActionArray = NDArray[np.floating[Any]]  # Continuous action values

# Reward arrays
RewardArray = NDArray[np.floating[Any]]  # Array of rewards
ValueArray = NDArray[np.floating[Any]]  # Array of state/action values

# =============================================================================
# Utility Arrays
# =============================================================================

# Distance and metric arrays
DistanceArray = NDArray[np.floating[Any]]  # Array of distances
MetricArray = NDArray[np.floating[Any]]  # Array of metric values

# Index arrays
IndexArray = NDArray[np.integer[Any]]  # Array of indices
BooleanMask = NDArray[np.bool_]  # Boolean mask array

# =============================================================================
# Environment-Specific Arrays
# =============================================================================

# CartPole arrays
CartPoleState = NDArray[np.floating[Any]]  # [position, velocity, angle, angular_velocity]
CartPoleObservation = NDArray[np.floating[Any]]  # [position, velocity, angle, angular_velocity]

# Light-Dark arrays
LightDarkState = NDArray[np.floating[Any]]  # [x, y] position
LightDarkObservation = NDArray[np.floating[Any]]  # [x, y] position with noise

# Mountain Car arrays
MountainCarState = NDArray[np.floating[Any]]  # [position, velocity]
MountainCarObservation = NDArray[np.floating[Any]]  # [position, velocity] with noise

# Push arrays
PushState = NDArray[np.floating[Any]]  # [agent_x, agent_y, object_x, object_y, goal_x, goal_y]
PushObservation = NDArray[np.floating[Any]]  # 6D observation with noise

# Safety Ant Velocity arrays
SafeAntState = NDArray[np.floating[Any]]  # [x, y, vx, vy] position and velocity
SafeAntObservation = NDArray[np.floating[Any]]  # [x, y, vx, vy] with noise

# =============================================================================
# Union Types for Flexibility
# =============================================================================

# Flexible state/observation types
StateLike = Union[StateVector, StateVector2D, StateVector4D, StateVector6D]
ObservationLike = Union[
    ObservationVector, ObservationVector2D, ObservationVector4D, ObservationVector6D
]

# Flexible coordinate types
CoordinateLike = Union[Coordinate2D, Coordinates2D, BeaconArray, ObstacleArray]

# Flexible probability types
ProbabilityLike = Union[ProbabilityArray, LogProbabilityArray, ParticleWeights, LogParticleWeights]

# =============================================================================
# Type Guards and Validation
# =============================================================================


def is_state_vector(arr: Array) -> bool:
    """Check if array is a valid state vector."""
    return arr.ndim == 1 and arr.dtype.kind in "fc"  # float or complex


def is_2d_coordinate_array(arr: Array) -> bool:
    """Check if array is a valid 2D coordinate array."""
    return arr.ndim == 2 and arr.shape[0] == 2 and arr.dtype.kind in "fc"


def is_probability_array(arr: Array) -> bool:
    """Check if array contains valid probabilities."""
    return arr.ndim == 1 and arr.dtype.kind in "fc" and bool(np.all((arr >= 0) & (arr <= 1)))


def is_covariance_matrix(arr: Array) -> bool:
    """Check if array is a valid covariance matrix."""
    return (
        arr.ndim == 2
        and arr.shape[0] == arr.shape[1]
        and arr.dtype.kind in "fc"
        and np.allclose(arr, arr.T)
    )  # Symmetric


# =============================================================================
# Example Functions Using All Types
# =============================================================================


def process_generic_array(arr: Array) -> Array:
    """Process any numpy array."""
    return arr


def process_float_array(arr: FloatArray) -> FloatArray:
    """Process floating point array."""
    return arr * 2.0


def process_int_array(arr: IntArray) -> IntArray:
    """Process integer array."""
    return arr + 1


def process_bool_array(arr: BoolArray) -> BoolArray:
    """Process boolean array."""
    return ~arr


def process_float32_array(arr: Float32Array) -> Float32Array:
    """Process 32-bit float array."""
    return arr.astype(np.float32)


def process_float64_array(arr: Float64Array) -> Float64Array:
    """Process 64-bit float array."""
    return arr.astype(np.float64)


def process_int32_array(arr: Int32Array) -> Int32Array:
    """Process 32-bit integer array."""
    return arr.astype(np.int32)


def process_int64_array(arr: Int64Array) -> Int64Array:
    """Process 64-bit integer array."""
    return arr.astype(np.int64)


def process_numeric_array(arr: NumericArray) -> NumericArray:
    """Process any numeric array (float or int)."""
    return arr


def process_complex_array(arr: ComplexArray) -> ComplexArray:
    """Process complex number array."""
    return arr.conj()


def process_unsigned_int_array(arr: UnsignedIntArray) -> UnsignedIntArray:
    """Process unsigned integer array."""
    return arr


def process_signed_int_array(arr: SignedIntArray) -> SignedIntArray:
    """Process signed integer array."""
    return arr


def process_state_vector(state: StateVector) -> StateVector:
    """Process 1D state vector."""
    return state


def process_state_vector_2d(state: StateVector2D) -> StateVector2D:
    """Process 2D state vector."""
    return state


def process_state_vector_4d(state: StateVector4D) -> StateVector4D:
    """Process 4D state vector."""
    return state


def process_state_vector_6d(state: StateVector6D) -> StateVector6D:
    """Process 6D state vector."""
    return state


def process_observation_vector(obs: ObservationVector) -> ObservationVector:
    """Process 1D observation vector."""
    return obs


def process_observation_vector_2d(obs: ObservationVector2D) -> ObservationVector2D:
    """Process 2D observation vector."""
    return obs


def process_observation_vector_4d(obs: ObservationVector4D) -> ObservationVector4D:
    """Process 4D observation vector."""
    return obs


def process_observation_vector_6d(obs: ObservationVector6D) -> ObservationVector6D:
    """Process 6D observation vector."""
    return obs


def process_probability_array(probs: ProbabilityArray) -> ProbabilityArray:
    """Process probability array."""
    return probs / np.sum(probs)


def process_log_probability_array(log_probs: LogProbabilityArray) -> LogProbabilityArray:
    """Process log probability array."""
    return log_probs - np.max(log_probs)


def process_coordinate_2d(coord: Coordinate2D) -> Coordinate2D:
    """Process single 2D coordinate."""
    return coord


def process_coordinates_2d(coords: Coordinates2D) -> Coordinates2D:
    """Process multiple 2D coordinates."""
    return coords


def process_beacon_array(beacons: BeaconArray) -> BeaconArray:
    """Process beacon array."""
    return beacons


def process_obstacle_array(obstacles: ObstacleArray) -> ObstacleArray:
    """Process obstacle array."""
    return obstacles


def process_covariance_matrix_2d(cov: CovarianceMatrix2D) -> CovarianceMatrix2D:
    """Process 2D covariance matrix."""
    return cov


def process_covariance_matrix_4d(cov: CovarianceMatrix4D) -> CovarianceMatrix4D:
    """Process 4D covariance matrix."""
    return cov


def process_covariance_matrix(cov: CovarianceMatrix) -> CovarianceMatrix:
    """Process generic covariance matrix."""
    return cov


def process_transition_matrix(trans: TransitionMatrix) -> TransitionMatrix:
    """Process transition matrix."""
    return trans


def process_observation_matrix(obs_mat: ObservationMatrix) -> ObservationMatrix:
    """Process observation matrix."""
    return obs_mat


def process_particle_array(particles: ParticleArray) -> ParticleArray:
    """Process particle array."""
    return particles


def process_particle_weights(weights: ParticleWeights) -> ParticleWeights:
    """Process particle weights."""
    return weights / np.sum(weights)


def process_log_particle_weights(log_weights: LogParticleWeights) -> LogParticleWeights:
    """Process log particle weights."""
    return log_weights - np.max(log_weights)


def process_belief_array(belief: BeliefArray) -> BeliefArray:
    """Process belief array."""
    return belief


def process_action_array(actions: ActionArray) -> ActionArray:
    """Process discrete action array."""
    return actions


def process_continuous_action_array(actions: ContinuousActionArray) -> ContinuousActionArray:
    """Process continuous action array."""
    return actions


def process_reward_array(rewards: RewardArray) -> RewardArray:
    """Process reward array."""
    return rewards


def process_value_array(values: ValueArray) -> ValueArray:
    """Process value array."""
    return values


def process_distance_array(distances: DistanceArray) -> DistanceArray:
    """Process distance array."""
    return distances


def process_metric_array(metrics: MetricArray) -> MetricArray:
    """Process metric array."""
    return metrics


def process_index_array(indices: IndexArray) -> IndexArray:
    """Process index array."""
    return indices


def process_boolean_mask(mask: BooleanMask) -> BooleanMask:
    """Process boolean mask."""
    return mask


def process_cartpole_state(state: CartPoleState) -> CartPoleState:
    """Process CartPole state."""
    return state


def process_cartpole_observation(obs: CartPoleObservation) -> CartPoleObservation:
    """Process CartPole observation."""
    return obs


def process_light_dark_state(state: LightDarkState) -> LightDarkState:
    """Process Light-Dark state."""
    return state


def process_light_dark_observation(obs: LightDarkObservation) -> LightDarkObservation:
    """Process Light-Dark observation."""
    return obs


def process_mountain_car_state(state: MountainCarState) -> MountainCarState:
    """Process Mountain Car state."""
    return state


def process_mountain_car_observation(obs: MountainCarObservation) -> MountainCarObservation:
    """Process Mountain Car observation."""
    return obs


def process_push_state(state: PushState) -> PushState:
    """Process Push state."""
    return state


def process_push_observation(obs: PushObservation) -> PushObservation:
    """Process Push observation."""
    return obs


def process_safe_ant_state(state: SafeAntState) -> SafeAntState:
    """Process Safety Ant Velocity state."""
    return state


def process_safe_ant_observation(obs: SafeAntObservation) -> SafeAntObservation:
    """Process Safety Ant Velocity observation."""
    return obs


def process_state_like(state: StateLike) -> StateLike:
    """Process flexible state type."""
    return state


def process_observation_like(obs: ObservationLike) -> ObservationLike:
    """Process flexible observation type."""
    return obs


def process_coordinate_like(coord: CoordinateLike) -> CoordinateLike:
    """Process flexible coordinate type."""
    return coord


def process_probability_like(prob: ProbabilityLike) -> ProbabilityLike:
    """Process flexible probability type."""
    return prob


# =============================================================================
# Backward Compatibility
# =============================================================================

# Legacy type aliases for backward compatibility
NDArrayFloat = FloatArray
NDArrayInt = IntArray
NDArrayBool = BoolArray

# Common aliases used in the codebase
StateArray = StateVector
ObservationArray = ObservationVector
ProbArray = ProbabilityArray
