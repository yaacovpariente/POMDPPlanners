"""Type stubs for the native C++ Continuous LaserTag sampling extension.

Declares the Python-visible API of the ``_native`` module so pyright can
type-check modules that import from it. The runtime implementation lives
in ``_cpp/continuous_laser_tag.cpp``.
"""

# pylint: disable=unused-argument,unnecessary-ellipsis

from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

def set_seed(seed: int) -> None:
    """Seed the module-level RNG used by ``sample()`` / batch entry points."""
    ...

def reward_batch(
    states: NDArray[np.floating],
    action: NDArray[np.floating],
    tag_radius: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    dangerous_areas: NDArray[np.floating],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
) -> NDArray[np.float64]:
    """Vectorised reward kernel; see ContinuousLaserTagPOMDP.reward_batch."""
    ...

def lasertag_discrete_reward_batch(
    states: NDArray[np.floating],
    action: int,
    rows: int,
    cols: int,
    walls_flat: NDArray[np.integer],
    n_walls: int,
    dangerous_areas: NDArray[np.floating],
    n_dangerous: int,
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    action_directions: NDArray[np.integer],
    next_states: NDArray[np.floating] = ...,
    reward_variant_code: int = ...,
    penalty_decay: float = ...,
) -> NDArray[np.float64]:
    """Vectorised reward kernel for the discrete LaserTagPOMDP.

    Mirrors ``LaserTagRewardModel.compute_reward_batch`` across all three
    :class:`RewardModelType` variants. ``walls_flat`` is the flattened
    ``(row, col)`` wall list (length ``2 * n_walls``). ``action_directions``
    is a ``(4, 2)`` int64 array mapping action index ``0..3`` to its
    ``(dr, dc)`` cell delta. When ``next_states`` is shape ``(N, 5)``, the
    danger / wall penalty is scored against ``next_states[:, :2]``; when it
    is empty (the default) the legacy intended-position fallback is used.
    ``reward_variant_code`` selects the variant: ``0`` = CONSTANT_HAZARD_PENALTY,
    ``1`` = ZERO_MEAN_HAZARD_SHOCK, ``2`` = DISTANCE_DECAYED_HAZARD_PENALTY.
    ``penalty_decay`` is the decay length used by the
    ``DISTANCE_DECAYED_HAZARD_PENALTY`` variant (ignored otherwise). Returns shape
    ``(N,)`` float64.
    """
    ...

class ContinuousLaserTagTransitionCpp:
    """Native state transition sampler (robot + opponent Gaussian steps)."""

    state: Tuple[float, float, float, float, float]
    action: Tuple[float, float, float]

    def __init__(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
        action: Union[Sequence[float], NDArray[np.floating]],
        robot_covariance: NDArray[np.floating],
        opponent_covariance: NDArray[np.floating],
        evasion_speed: float,
        walls: NDArray[np.floating],
        grid_size: NDArray[np.floating],
        robot_radius: float,
        opponent_radius: float,
        tag_radius: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_sample(self, particles: NDArray[np.floating]) -> NDArray[np.float64]: ...
    def set_state(
        self,
        state: Union[Sequence[float], NDArray[np.floating]],
    ) -> None: ...

class ContinuousLaserTagObservationCpp:
    """Native 8-direction laser observation sampler."""

    next_state: Tuple[float, float, float, float, float]
    action: Tuple[float, float, float]
    mean: NDArray[np.float64]

    def __init__(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
        action: Union[Sequence[float], NDArray[np.floating]],
        measurement_noise: float,
        walls: NDArray[np.floating],
        grid_size: NDArray[np.floating],
        opponent_radius: float,
    ) -> None: ...
    def sample(self, n_samples: int = 1) -> List[NDArray[np.float64]]: ...
    def probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def log_probability(
        self,
        values: Union[Sequence[NDArray[np.floating]], NDArray[np.floating]],
    ) -> NDArray[np.float64]: ...
    def batch_log_likelihood(
        self,
        next_particles: NDArray[np.floating],
        observation: NDArray[np.floating],
    ) -> NDArray[np.float64]: ...
    def set_next_state(
        self,
        next_state: Union[Sequence[float], NDArray[np.floating]],
    ) -> None: ...

# ── Discrete LaserTag native rollout ─────────────────────────────────────────

def simulate_rollout_discrete(
    initial_state: NDArray[np.floating],
    max_depth: int,
    discount: float,
    initial_depth: int,
    rows: int,
    cols: int,
    walls_flat: NDArray[np.integer],
    dangerous_areas: NDArray[np.floating],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    transition_error_prob: float,
    reward_variant_code: int = ...,
    penalty_decay: float = ...,
) -> float:
    """Run a full random-action rollout for the discrete LaserTagPOMDP in one C++ frame.

    Actions are drawn uniformly from ``{0,1,2,3,4}`` using the module-level
    mt19937_64 RNG. Seed via :func:`set_seed` before calling for reproducible
    results. ``reward_variant_code`` selects the reward variant:
    ``0`` = CONSTANT_HAZARD_PENALTY, ``1`` = ZERO_MEAN_HAZARD_SHOCK,
    ``2`` = DISTANCE_DECAYED_HAZARD_PENALTY. ``penalty_decay`` is consulted only by
    the ``DISTANCE_DECAYED_HAZARD_PENALTY`` variant. Returns the discounted sum of
    immediate rewards along the sampled trajectory.
    """
    ...

# ── Discrete LaserTag belief-update kernels ──────────────────────────────────

def belief_batch_transition_discrete(
    particles: NDArray[np.floating],
    action_idx: int,
    transition_error_prob: float,
    valid_cell_flat: NDArray[np.unsignedinteger],
    rows: int,
    cols: int,
) -> NDArray[np.float64]:
    """Native port of LaserTagVectorizedUpdater.batch_transition.

    Returns the (N, 5) float64 array of next particles. Uses the module-level
    mt19937_64 RNG; seed via set_seed() before calling for reproducibility.
    """
    ...

def belief_batch_obs_log_likelihood_discrete(
    next_particles: NDArray[np.floating],
    observation: NDArray[np.floating],
    wall_dist_table_flat: NDArray[np.integer],
    rows: int,
    cols: int,
    log_norm_1d: float,
    inv_2var: float,
) -> NDArray[np.float64]:
    """Native port of LaserTagVectorizedUpdater.batch_observation_log_likelihood.

    Returns the (N,) float64 array of per-particle log-likelihoods.
    Terminal handling matches the Python wrapper: terminal observations
    return 0.0 for terminal particles and -inf for non-terminal, and
    non-terminal observations return -inf for terminal particles.
    """
    ...

# ── Discrete LaserTag single-step kernels ───────────────────────────────────

def sample_next_state_step(
    state: NDArray[np.floating],
    actual_action: int,
    opp_uniform: float,
    rows: int,
    cols: int,
    walls_flat: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Single-step transition for the discrete LaserTagPOMDP.

    The Python wrapper resolves the actual_action (handling the optional
    transition error in numpy) and pre-draws ``opp_uniform`` via
    ``np.random.random()`` so byte-identical numpy RNG state is preserved
    across the original Python path and this native fast path.
    """
    ...

def sample_observation_step(
    next_state: NDArray[np.floating],
    noise: NDArray[np.floating],
    rows: int,
    cols: int,
    walls_flat: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Single-step observation for the discrete LaserTagPOMDP.

    ``noise`` must be a length-8 float64 array of pre-drawn N(0, sigma)
    samples. Returns the noisy 8-direction laser observation.
    """
    ...

def observation_log_probability_step(
    next_state: NDArray[np.floating],
    observations: NDArray[np.floating],
    measurement_noise: float,
    rows: int,
    cols: int,
    walls_flat: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Per-observation log-probability for the discrete LaserTagPOMDP.

    Mirrors LaserTagPOMDP.observation_log_probability semantics.
    """
    ...

# ── ContinuousLaserTag native rollout (added by perf agent) ──────────────────

def cont_simulate_rollout(
    initial_state: NDArray[np.floating],
    actions_buffer: NDArray[np.floating],
    start_depth: int,
    max_depth: int,
    discount_factor: float,
    robot_covariance: NDArray[np.floating],
    opponent_covariance: NDArray[np.floating],
    evasion_speed: float,
    walls: NDArray[np.floating],
    grid_size: NDArray[np.floating],
    robot_radius: float,
    opponent_radius: float,
    tag_radius: float,
    tag_reward: float,
    tag_penalty: float,
    step_cost: float,
    dangerous_areas: NDArray[np.floating],
    dangerous_area_radius: float,
    dangerous_area_penalty: float,
) -> float:
    """Run a full random rollout for ContinuousLaserTagPOMDP in one C++ frame.

    ``actions_buffer`` must be shape (N, 3) float64 with N >= max_depth - start_depth.
    Returns the discounted sum of immediate rewards along the sampled trajectory.
    """
    ...
