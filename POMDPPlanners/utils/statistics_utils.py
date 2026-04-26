# pylint: disable=fixme
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import binom

from POMDPPlanners.core.distributions import Distribution


def cvar_estimator(vec: np.ndarray, alpha: float) -> float:
    """Calculate Conditional Value at Risk (CVaR) for risk-sensitive POMDP evaluation.

    CVaR measures the expected value of the worst-case outcomes, providing a
    risk-sensitive performance metric that goes beyond simple mean rewards.
    This is particularly valuable for safety-critical applications where
    tail risk matters more than average performance.

    Mathematical Definition:
        CVaR_α(X) = E[X | X ≥ VaR_α(X)]

    Where VaR_α is the Value at Risk at confidence level α.

    The implementation uses a vectorized approach for computational efficiency,
    calculating CVaR by integrating over the tail distribution above the α-quantile.

    Args:
        vec: Array of values (typically returns, costs, or performance metrics)
        alpha: Confidence level (0 < α ≤ 1), where higher values focus on worse outcomes

    Returns:
        CVaR value representing expected worst-case performance

    Raises:
        ValueError: If alpha not in [0,1] or vec is empty

    Example:
        Risk analysis of POMDP algorithm performance:

        >>> import numpy as np
        >>> from POMDPPlanners.utils.statistics_utils import cvar_estimator

        >>> # Simulate algorithm returns from multiple episodes
        >>> returns = np.array([12.5, 8.3, 15.7, -2.1, 9.8, 13.2, 6.4, 11.0, -1.5, 14.3])
        >>> len(returns)
        10

        >>> # Calculate risk metrics
        >>> mean_return = np.mean(returns)
        >>> bool(mean_return > 8.0)  # Check reasonable mean
        True
        >>> cvar_90 = cvar_estimator(returns, alpha=0.9)  # Worst 10% outcomes
        >>> cvar_95 = cvar_estimator(returns, alpha=0.95) # Worst 5% outcomes
        >>> isinstance(cvar_90, (float, np.floating))
        True
        >>> isinstance(cvar_95, (float, np.floating))
        True
        >>> cvar_95 <= cvar_90  # CVaR should be lower for higher alpha
        True

    Example:
        Comparing algorithm risk profiles:

        >>> # Algorithm performance data from experiments
        >>> pomcp_returns = np.array([10.2, 12.8, 9.5, 11.3, 8.7, 12.1, 10.9, 9.8, 11.5, 10.4])
        >>> pft_returns = np.array([15.1, 7.2, 14.8, 13.3, 6.9, 15.5, 8.1, 14.2, 12.7, 9.3])
        >>> pomcp_cvar = cvar_estimator(pomcp_returns, alpha=0.9)
        >>> pft_cvar = cvar_estimator(pft_returns, alpha=0.9)

    Risk Assessment Applications:
        **Portfolio Analysis**: Compare multiple algorithms' risk-return profiles

        **Safety-Critical Systems**: Evaluate worst-case performance guarantees

        **Robust Planning**: Select algorithms with acceptable tail risk

        **Performance Bounds**: Establish confidence intervals for worst-case scenarios

    Mathematical Properties:
        - **Monotonic**: CVaR_α ≥ VaR_α (CVaR is always at least as large as VaR)
        - **Coherent**: Satisfies subadditivity, monotonicity, positive homogeneity
        - **Tail Sensitivity**: Lower α values emphasize extreme outcomes more
        - **Computational**: More stable than VaR, especially for small samples
    """
    # Calculate the Conditional Value at Risk (CVaR) for a given vector of values.
    # CVaR is the expected value of the worst (1-alpha)% of cases, where "worst"
    # means the highest values (assuming these represent costs or risks).
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if len(vec) == 0:
        raise ValueError("Input vector must not be empty")

    if len(vec) == 1:
        return float(vec[0])

    sorted_vec = np.sort(vec)
    n = len(sorted_vec)

    if alpha == 1.0:
        return float(sorted_vec[-1])
    if alpha == 0.0:
        return float(sorted_vec[0])

    n = len(sorted_vec)  # Get the number of elements

    # Create indices and weights vectorized
    indices = np.arange(1, n)  # 1 to n-1
    weights = np.maximum(0, (indices / n) - (1 - alpha))

    # Calculate differences vectorized
    diffs = sorted_vec[1:] - sorted_vec[:-1]

    # Calculate final result vectorized
    s = sorted_vec[-1] - np.sum(weights * diffs) / alpha

    return float(s)


def confidence_interval(data, confidence=0.95) -> Tuple[float, float]:
    """Calculate confidence interval for the mean using t-distribution.

    Computes confidence intervals for algorithm performance means, providing
    statistical bounds on the true expected performance. This is essential
    for making statistically sound comparisons between POMDP algorithms.

    Uses the t-distribution to account for sample size uncertainty, which is
    more appropriate than normal distribution for small to moderate sample sizes
    common in POMDP experiments.

    Args:
        data: Sample data (algorithm returns, rewards, or performance metrics)
        confidence: Confidence level (0 < confidence < 1, typically 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval

    Raises:
        ValueError: If insufficient data or contains NaN values

    Example:
        Statistical comparison of algorithm performance:

        >>> import numpy as np
        >>> # Algorithm performance from multiple runs
        >>> pomcp_rewards = [12.3, 11.8, 13.1, 12.7, 11.9, 12.5, 13.0, 12.1, 12.8, 12.4]
        >>> pft_rewards = [11.5, 13.2, 12.8, 11.9, 12.3, 13.5, 12.1, 12.9, 11.7, 12.6]
        >>>
        >>> # Calculate 95% confidence intervals
        >>> pomcp_ci = confidence_interval(pomcp_rewards, confidence=0.95)
        >>> pft_ci = confidence_interval(pft_rewards, confidence=0.95)
        >>>
        >>> # Verify confidence intervals are tuples with two elements
        >>> isinstance(pomcp_ci, tuple) and len(pomcp_ci) == 2
        True
        >>> isinstance(pft_ci, tuple) and len(pft_ci) == 2
        True
        >>>
        >>> # Verify confidence intervals contain the mean
        >>> pomcp_mean = np.mean(pomcp_rewards)
        >>> pft_mean = np.mean(pft_rewards)
        >>> bool(pomcp_ci[0] <= pomcp_mean <= pomcp_ci[1])
        True
        >>> bool(pft_ci[0] <= pft_mean <= pft_ci[1])
        True
        >>>
        >>> # Verify lower bound is less than upper bound
        >>> pomcp_ci[0] < pomcp_ci[1]
        True
        >>> pft_ci[0] < pft_ci[1]
        True

    Calculate the confidence interval for the mean of a dataset using the t-distribution.

    Parameters:
        data (array-like): Sample data
        confidence (float): Confidence level (default 0.95 for 95%)

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval

    Raises:
        ValueError: If data contains NaN values or has insufficient samples
    """
    data = np.array(data)
    if len(data) == 0:
        raise ValueError("Data must contain at least one element")

    # For single data point, return the value as both bounds
    if len(data) == 1:
        return (-np.inf, np.inf)

    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    # If all values are the same, return the value as both bounds
    if np.all(data == data[0]) and len(data) > 1:
        return (data[0], data[0])

    mean = np.mean(data)
    sem = float(stats.sem(data))  # Standard error of the mean
    df = len(data) - 1  # Degrees of freedom

    # Confidence interval
    ci = stats.t.interval(confidence, df, loc=mean, scale=sem)
    return (float(ci[0]), float(ci[1]))


def cvar_confidence_interval(
    data,
    alpha=0.95,
    delta=0.05,
    dist_lower_bound: Optional[float] = None,
    dist_upper_bound: Optional[float] = None,
):
    """
    Calculate the confidence interval for the CVaR of a dataset using the t-distribution.

    Args:
        data: Array of values
        alpha: Confidence level (default 0.95 for 95%)
        delta: Significance level for the probabilistic bounds (default 0.05)
        dist_lower_bound: Known lower bound of the distribution support.
            If None, uses min(data) as a conservative data-driven fallback.
        dist_upper_bound: Known upper bound of the distribution support.
            If None, uses max(data) as a conservative data-driven fallback.

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval

    Raises:
        ValueError: If data contains NaN values or has insufficient samples
    """

    if not 0 <= alpha <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if len(data) == 0:
        raise ValueError("Input vector must not be empty")

    # For single data point, return the value as both bounds
    if len(data) == 1:
        cvar_value = cvar_estimator(data, alpha)
        return (cvar_value, cvar_value)

    data_arr = np.asarray(data)
    effective_lower = dist_lower_bound if dist_lower_bound is not None else float(np.min(data_arr))
    effective_upper = dist_upper_bound if dist_upper_bound is not None else float(np.max(data_arr))

    lower_bound = cvar_probabilistic_lower_bound_thomas(
        vec=data, alpha=alpha, delta=delta / 2, dist_lower_bound=effective_lower
    )
    upper_bound = cvar_probabilistic_upper_bound_thomas(
        vec=data, alpha=alpha, delta=delta / 2, dist_upper_bound=effective_upper
    )
    return lower_bound, upper_bound


def cvar_probabilistic_lower_bound_thomas(
    vec: np.ndarray, alpha: float, delta: float, dist_lower_bound: float
) -> float:
    """
    Calculate a probabilistic lower bound for CVaR using Thomas's method.

    Args:
        vec: Array of values
        alpha: Confidence level (between 0 and 1)
        delta: Probability of the bound holding (between 0 and 1)
        dist_lower_bound: Lower bound of the distribution

    Returns:
        float: The probabilistic lower bound for CVaR

    Raises:
        ValueError: If alpha or delta are not between 0 and 1, or if vec is empty
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")
    if len(vec) == 0:
        raise ValueError("Input vector must not be empty")

    sorted_vec = np.sort(vec)
    n = len(sorted_vec)

    # Calculate weights vectorized
    indices = np.arange(n)  # 0 to n-1
    weights = np.maximum(
        0,
        np.minimum(1, (indices / n) + np.sqrt(np.log(1 / delta) / (2 * n))) - (1 - alpha),
    )

    # Calculate differences vectorized
    diffs = sorted_vec[1:] - sorted_vec[:-1]

    # Calculate final result vectorized
    s = sorted_vec[-1] - np.sum(weights[1:] * diffs) / alpha

    # Add the lower bound term
    s -= weights[0] * (sorted_vec[0] - dist_lower_bound) / alpha

    return s


def cvar_probabilistic_upper_bound_thomas(
    vec: np.ndarray, alpha: float, delta: float, dist_upper_bound: float
) -> float:
    """
    Calculate a probabilistic upper bound for CVaR using Thomas's method.

    Args:
        vec: Array of values
        alpha: Confidence level (between 0 and 1)
        delta: Probability of the bound holding (between 0 and 1)
        dist_upper_bound: Upper bound of the distribution

    Returns:
        float: The probabilistic upper bound for CVaR

    Raises:
        ValueError: If alpha or delta are not between 0 and 1, or if vec is empty
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1")
    if len(vec) == 0:
        raise ValueError("Input vector must not be empty")

    sorted_vec = np.sort(vec)
    n = len(sorted_vec)

    # Calculate weights vectorized
    indices = np.arange(1, n + 1)  # 1 to n
    weights = np.maximum(0, (indices / n) - np.sqrt(np.log(1 / delta) / (2 * n)) - (1 - alpha))

    # Calculate differences vectorized
    diffs = sorted_vec[1:] - sorted_vec[:-1]

    # Calculate initial term
    initial_weight = weights[-1]
    s = dist_upper_bound - (dist_upper_bound - sorted_vec[-1]) * initial_weight / alpha

    # Calculate final result vectorized
    s -= np.sum(weights[:-1] * diffs) / alpha

    return s


def quantile_confidence_interval(data, alpha=0.95, conf_level=0.95):
    """
    data: 1D array-like of samples
    alpha: target quantile (e.g. 0.95 for 95% VaR)
    conf_level: overall coverage (e.g. 0.95 for 95% CI)
    Returns: (lower_value, upper_value, k1, k2)
    """
    x = np.sort(np.asarray(data))
    n = len(x)

    # For single data point, return the value with indices
    if n == 1:
        return x[0], x[0], 1, 1

    delta = 1 - conf_level

    # find smallest k1 such that P(Y < k1) >= delta/2
    k1 = 0
    while binom.cdf(k1 - 1, n, alpha) < delta / 2 and k1 <= n:
        k1 += 1

    # find largest k2 such that P(Y <= k2) <= 1 - delta/2
    k2 = n
    while binom.cdf(k2, n, alpha) > 1 - delta / 2 and k2 >= 0:
        k2 -= 1

    # clip to valid index range and convert to 0-based indexing
    k1 = max(1, min(k1, n))  # at least 1
    k2 = max(1, min(k2, n))  # at least 1

    return x[k1 - 1], x[k2 - 1], k1, k2


@lru_cache(maxsize=2048)
def get_min_and_max_cost(
    min_immediate_cost: float,
    max_immediate_cost: float,
    depth: int,
    max_depth: int,
    gamma: float,
) -> tuple[float, float]:
    """
    Calculate the minimum and maximum costs over a time horizon using a discount factor.

    Args:
        min_immediate_cost: Minimum immediate cost
        max_immediate_cost: Maximum immediate cost
        depth: Current depth in the search tree
        max_depth: Maximum depth of the search tree
        gamma: Discount factor (between 0 and 1)

    Returns:
        tuple: (min_cost, max_cost) representing the minimum and maximum costs over the time horizon

    Raises:
        ValueError: If gamma is not between 0 and 1
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")

    # Closed-form: sum_{k=0..h} gamma^k = (1 - gamma^(h+1)) / (1 - gamma) for gamma < 1, else h+1.
    horizon = max_depth - depth + 1
    if gamma == 1.0:
        gamma_geometric_sum = float(horizon)
    else:
        gamma_geometric_sum = (1.0 - gamma**horizon) / (1.0 - gamma)

    return min_immediate_cost * gamma_geometric_sum, max_immediate_cost * gamma_geometric_sum


def cvar_bound_const_eps(
    y_samp: np.ndarray, y_sup: float, y_inf: float, eps: float, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Calculate bounds for CVaR using a constant epsilon parameter.

    Args:
        y_samp: Array of sample values
        y_sup: Upper bound of the distribution
        y_inf: Lower bound of the distribution
        eps: Epsilon parameter for the bound calculation
        alpha: Confidence level (default 0.05)

    Returns:
        tuple: (lower_bound, upper_bound) representing the bounds for CVaR

    Raises:
        ValueError: If alpha or eps are not between 0 and 1, or if y_samp is empty
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between 0 and 1")
    if len(y_samp) == 0:
        raise ValueError("Input vector must not be empty")

    # Calculate lower bound
    if eps + alpha < 1:
        lower_bound = float(
            (alpha + eps) / alpha * cvar_estimator(y_samp, alpha + eps)
            - eps / alpha * cvar_estimator(y_samp, eps)
        )
    else:
        y_samp_mean = np.mean(y_samp)
        lower_bound = (
            (alpha + eps - 1) * y_inf
            + float(y_samp_mean)
            - eps * float(cvar_estimator(y_samp, eps))
        )
        lower_bound = float(lower_bound / alpha)

    # Calculate upper bound
    if eps < alpha:
        upper_bound = (alpha - eps) / alpha * float(
            cvar_estimator(y_samp, alpha - eps)
        ) + eps / alpha * y_sup
        upper_bound = float(upper_bound)
    else:
        upper_bound = float(y_sup)

    return lower_bound, upper_bound


def aggregate_weights_for_duplicate_values(
    values: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate weights for duplicate values to ensure unique values.

    When the same value appears multiple times in the values array with different
    weights, this function combines them into a single entry with the sum of all
    weights for that value. This is useful for discrete distributions where
    duplicate values should be treated as a single outcome with aggregated probability.

    Args:
        values: Array of values (may contain duplicates)
        weights: Array of corresponding weights/probabilities

    Returns:
        Tuple of (unique_values, aggregated_weights) where:
        - unique_values: Array of unique values (sorted)
        - aggregated_weights: Array of weights corresponding to unique values,
          normalized to sum to 1

    Raises:
        ValueError: If arrays are empty or have mismatched lengths

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.utils.statistics_utils import aggregate_weights_for_duplicate_values
        >>> values = np.array([1.0, 2.0, 2.0, 3.0])
        >>> weights = np.array([0.3, 0.2, 0.3, 0.2])
        >>> unique_vals, agg_weights = aggregate_weights_for_duplicate_values(values, weights)
        >>> unique_vals
        array([1., 2., 3.])
        >>> bool(np.isclose(agg_weights, np.array([0.3, 0.5, 0.2])).all())
        True
        >>> bool(np.isclose(np.sum(agg_weights), 1.0))
        True
    """
    if len(values) == 0 or len(weights) == 0:
        raise ValueError("Input arrays must not be empty")
    if len(values) != len(weights):
        raise ValueError("Values and weights arrays must have the same length")

    # Get unique values and aggregate weights for each
    unique_values = np.unique(values)
    aggregated_weights = np.zeros(len(unique_values))
    for i, unique_val in enumerate(unique_values):
        # Handle NaN values specially since NaN != NaN
        if np.isnan(unique_val):
            mask = np.isnan(values)
        else:
            mask = values == unique_val
        aggregated_weights[i] = np.sum(weights[mask])

    # Normalize weights to ensure they still sum to 1 (handles floating point precision)
    aggregated_weights = aggregated_weights / np.sum(aggregated_weights)

    return unique_values, aggregated_weights


def cvar_estimator_from_dist(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) from a discrete distribution.

    Args:
        values: Array of values
        weights: Array of corresponding weights/probabilities
        alpha: Confidence level (between 0 and 1)

    Returns:
        float: The CVaR value

    Raises:
        ValueError: If alpha is not between 0 and 1, if arrays are empty, or if weights don't sum to 1
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if len(values) == 0 or len(weights) == 0:
        raise ValueError("Input arrays must not be empty")
    if len(values) != len(weights):
        raise ValueError("Values and weights arrays must have the same length")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Weights must sum to 1")

    if len(values) == 1:
        return float(values[0])

    if len(np.unique(values)) < len(values):
        values, weights = aggregate_weights_for_duplicate_values(values, weights)

    # Sort values and weights
    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]
    # TODO: fix the logic here
    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(sorted_weights)

    # Find value at risk index
    var_idx = np.where(cumulative_probs == (1 - alpha))[0]
    if len(var_idx) == 0:
        var_idx = np.where(cumulative_probs < (1 - alpha))[0]
        if len(var_idx) > 0:
            var_idx = var_idx[-1] + 1
        else:
            var_idx = 0
    else:
        var_idx = var_idx[0]

    value_at_risk = float(sorted_values[var_idx])

    # Calculate correction value
    value_at_risk_tail_cdf_prob: float = float(np.sum(sorted_weights[var_idx:]))
    correction_val = (value_at_risk_tail_cdf_prob - alpha) * value_at_risk

    # Calculate CVaR
    mask = values >= value_at_risk
    conditional_values = values[mask]
    conditional_weights = weights[mask]
    cvar = (np.sum(conditional_values * conditional_weights) - correction_val) / alpha

    return float(cvar)


def cvar_estimator_from_dist_fast(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Fast CVaR over a discrete distribution. Caller guarantees weights sum to 1.

    Hot-path variant of :func:`cvar_estimator_from_dist` for use inside MCTS
    backups where ``values`` is small (≤ ~30 elements) and ``weights`` is
    already normalized. Skips the input validation, the duplicate-value
    aggregation, and the equality-search branch of the slow path; runs a
    single ``argsort`` + cumulative sum + tail aggregation.

    Args:
        values: Array of values.
        weights: Array of normalized probabilities (must sum to 1.0).
        alpha: Confidence level in (0, 1].

    Returns:
        CVaR_alpha — the conditional expectation of ``values`` over the
        worst-alpha probability mass (upper tail when interpreting values
        as costs).
    """
    n = values.shape[0]
    if n == 1:
        return float(values[0])

    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]

    # Find the smallest index whose cumulative probability reaches 1 - alpha.
    threshold = 1.0 - alpha
    cumulative_probs = np.cumsum(sorted_weights)
    var_idx = int(np.searchsorted(cumulative_probs, threshold, side="left"))
    if var_idx >= n:
        var_idx = n - 1

    value_at_risk = sorted_values[var_idx]
    tail_weight = float(np.sum(sorted_weights[var_idx:]))
    tail_sum = float(np.sum(sorted_values[var_idx:] * sorted_weights[var_idx:]))
    correction = (tail_weight - alpha) * value_at_risk
    return (tail_sum - correction) / alpha


def tv_distance_grid(
    p: Distribution,
    q: Distribution,
    x_min: float = -5.0,
    x_max: float = 5.0,
    n_points: int = 10000,
) -> float:
    """Compute TV distance using grid-based numerical integration.

    This method has zero sampling variance and is deterministic.
    Works well for continuous distributions with known support.

    Args:
        p: First distribution
        q: Second distribution
        x_min: Lower bound of integration range
        x_max: Upper bound of integration range
        n_points: Number of grid points (higher = more accurate)

    Returns:
        TV distance between p and q
    """
    # Create uniform grid
    x_points = np.linspace(x_min, x_max, n_points)

    # Evaluate densities at grid points
    p_probs = p.probability(x_points.tolist())
    q_probs = q.probability(x_points.tolist())

    # Numerical integration using trapezoidal rule
    dx = (x_max - x_min) / n_points
    tv_dist = 0.5 * np.sum(np.abs(p_probs - q_probs)) * dx

    return float(tv_dist)


def tv_distance_averaged(
    p: Distribution, q: Distribution, n_samples: int = 1000, n_runs: int = 10
) -> float:
    """Compute TV distance by averaging multiple independent estimates.

    Reduces variance by sqrt(n_runs) compared to single run.

    Args:
        p: First distribution
        q: Second distribution
        n_samples: Number of samples per run
        n_runs: Number of independent runs to average

    Returns:
        Average TV distance estimate
    """
    estimates = []

    for _ in range(n_runs):
        # Sample half from p, half from q (similar to mixture sampling)
        n_half = n_samples // 2
        p_sample = p.sample(n_half)
        q_sample = q.sample(n_half)
        all_samples = p_sample + q_sample

        p_probs = p.probability(all_samples)
        q_probs = q.probability(all_samples)

        # TV distance = 0.5 * ∫|p(x) - q(x)| dx
        # Estimate using samples from both distributions
        tv_est = 0.5 * np.mean(np.abs(p_probs - q_probs))
        estimates.append(tv_est)

    return float(np.mean(estimates))


def tv_distance_mixture_sampling(p: Distribution, q: Distribution, n_samples: int = 2000) -> float:
    """Compute TV distance using mixture sampling for better coverage.

    Samples from mixture (p + q) / 2 to ensure good coverage of both
    distributions' support.

    Args:
        p: First distribution
        q: Second distribution
        n_samples: Number of samples from mixture

    Returns:
        TV distance estimate
    """
    # Sample half from p, half from q
    n_half = n_samples // 2
    p_sample = p.sample(n_half)
    q_sample = q.sample(n_half)

    # Combine samples
    all_samples = p_sample + q_sample

    # Evaluate both distributions at all points
    p_probs = p.probability(all_samples)
    q_probs = q.probability(all_samples)

    # Direct TV distance estimate
    # TV = 0.5 * ∫|p(x) - q(x)| dx
    # Estimate using importance sampling from mixture
    tv_est = 0.5 * np.mean(np.abs(p_probs - q_probs))

    return float(tv_est)


def tv_distance_monte_carlo(p: Distribution, q: Distribution, n_samples: int = 1000) -> float:
    """Compute TV distance using basic Monte Carlo sampling (original method).

    This is the original implementation that samples randomly from both
    distributions. Has higher variance than other methods.

    Args:
        p: First distribution
        q: Second distribution
        n_samples: Number of samples

    Returns:
        TV distance estimate
    """
    # Sample half from p, half from q (similar to mixture sampling)
    n_half = n_samples // 2
    p_sample = p.sample(n_half)
    q_sample = q.sample(n_half)

    # Combine samples
    all_samples = p_sample + q_sample

    # Evaluate both distributions at all points
    p_probs = p.probability(all_samples)
    q_probs = q.probability(all_samples)

    # Avoid division by zero by adding a small epsilon
    denominator = p_probs + q_probs
    denominator = np.where(denominator == 0, 1e-10, denominator)

    tv_distance_val = 0.5 * np.mean(np.abs(p_probs - q_probs) / denominator * 2)

    return float(tv_distance_val)


def tv_distance(
    p: Distribution,
    q: Distribution,
    n_samples: int = 1000,
    method: str = "grid",
    **kwargs,
) -> float:
    """Compute Total Variation distance between two distributions.

    Total Variation distance measures how different two probability distributions are,
    with values ranging from 0 (identical) to 1 (completely different).

    Args:
        p: First distribution
        q: Second distribution
        n_samples: Number of samples (method-dependent usage)
        method: Estimation method - "grid", "monte_carlo", "averaged", "mixture"
        **kwargs: Additional method-specific parameters:
            For "grid": x_min, x_max, n_points
            For "averaged": n_runs
            For "mixture": (uses n_samples directly)
            For "monte_carlo": (uses n_samples directly)

    Returns:
        TV distance estimate

    Example:
        >>> import numpy as np
        >>> from POMDPPlanners.core.distributions import DiscreteDistribution
        >>> np.random.seed(42)
        >>> # Create two similar discrete distributions
        >>> values = [0, 1, 2, 3, 4]
        >>> p_probs = np.array([0.2, 0.3, 0.25, 0.15, 0.1])
        >>> q_probs = np.array([0.25, 0.25, 0.25, 0.15, 0.1])
        >>> p = DiscreteDistribution(values, p_probs)
        >>> q = DiscreteDistribution(values, q_probs)
        >>> tv = tv_distance(p, q, method="monte_carlo", n_samples=10000)
        >>> isinstance(tv, float)
        True
        >>> 0.0 <= tv <= 1.0
        True
    """
    if method == "grid":
        # Use grid-based integration (deterministic, best for continuous)
        x_min = kwargs.get("x_min", -5.0)
        x_max = kwargs.get("x_max", 5.0)
        n_points = kwargs.get("n_points", n_samples)
        return tv_distance_grid(p, q, x_min, x_max, n_points)

    if method == "averaged":
        # Use multiple runs averaging
        n_runs = kwargs.get("n_runs", 10)
        return tv_distance_averaged(p, q, n_samples, n_runs)

    if method == "mixture":
        # Use mixture sampling
        return tv_distance_mixture_sampling(p, q, n_samples)

    if method == "monte_carlo":
        # Original implementation
        return tv_distance_monte_carlo(p, q, n_samples)

    raise ValueError(
        f"Unknown method '{method}'. Choose from: 'grid', 'monte_carlo', 'averaged', 'mixture'"
    )
