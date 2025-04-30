import numpy as np
from scipy import stats


def cvar_estimator(vec: np.ndarray, alpha: float) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) for a given vector of values.
    CVaR is the expected value of the worst (1-alpha)% of cases, where "worst"
    means the highest values (assuming these represent costs or risks).

    Args:
        vec: Array of values
        alpha: Confidence level (between 0 and 1)

    Returns:
        float: The CVaR value

    Raises:
        ValueError: If alpha is not between 0 and 1 or if vec is empty
    """
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

    return s


def confidence_interval(data, confidence=0.95):
    """
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
    if len(data) <= 1:
        raise ValueError("Data must contain at least two elements")
        
    if np.any(np.isnan(data)):
        raise ValueError("Data contains NaN values")

    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    df = len(data) - 1  # Degrees of freedom

    # Confidence interval
    ci = stats.t.interval(confidence, df, loc=mean, scale=sem)
    return ci


def cvar_probabilistic_lower_bound_thomas(vec: np.ndarray, alpha: float, delta: float, dist_lower_bound: float) -> float:
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
    weights = np.maximum(0, np.minimum(1, (indices / n) + np.sqrt(np.log(1/delta) / (2 * n))) - (1 - alpha))
    
    # Calculate differences vectorized
    diffs = sorted_vec[1:] - sorted_vec[:-1]
    
    # Calculate final result vectorized
    s = sorted_vec[-1] - np.sum(weights[1:] * diffs) / alpha
    
    # Add the lower bound term
    s -= weights[0] * (sorted_vec[0] - dist_lower_bound) / alpha
    
    return s


def cvar_probabilistic_upper_bound_thomas(vec: np.ndarray, alpha: float, delta: float, dist_upper_bound: float) -> float:
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
    weights = np.maximum(0, (indices / n) - np.sqrt(np.log(1/delta) / (2 * n)) - (1 - alpha))
    
    # Calculate differences vectorized
    diffs = sorted_vec[1:] - sorted_vec[:-1]
    
    # Calculate initial term
    initial_weight = weights[-1]
    s = dist_upper_bound - (dist_upper_bound - sorted_vec[-1]) * initial_weight / alpha
    
    # Calculate final result vectorized
    s -= np.sum(weights[:-1] * diffs) / alpha
    
    return s


def get_min_and_max_cost(min_immediate_cost: float, max_immediate_cost: float, depth: int, max_depth: int, gamma: float) -> tuple[float, float]:
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
        
    # Create array of powers of gamma
    powers = np.arange(max_depth - depth + 1)
    gamma_powers = gamma ** powers
    
    # Calculate min and max costs using vectorized operations
    min_cost = np.sum(min_immediate_cost * gamma_powers)
    max_cost = np.sum(max_immediate_cost * gamma_powers)
    
    return min_cost, max_cost


def cvar_bound_const_eps(y_samp: np.ndarray, y_sup: float, y_inf: float, eps: float, alpha: float = 0.05) -> tuple[float, float]:
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
        lower_bound = (alpha + eps) / alpha * cvar_estimator(y_samp, alpha + eps) - eps / alpha * cvar_estimator(y_samp, eps)
    else:
        y_samp_mean = np.mean(y_samp)
        lower_bound = (alpha + eps - 1) * y_inf + y_samp_mean - eps * cvar_estimator(y_samp, eps)
        lower_bound /= alpha
        
    # Calculate upper bound
    if eps < alpha:
        upper_bound = (alpha - eps) / alpha * cvar_estimator(y_samp, alpha - eps) + eps / alpha * y_sup
    else:
        upper_bound = y_sup
        
    return lower_bound, upper_bound


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
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Weights must sum to 1")
        
    # Sort values and weights
    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]
    
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
        
    value_at_risk = sorted_values[var_idx]
    
    # Calculate correction value
    value_at_risk_tail_cdf_prob = np.sum(sorted_weights[var_idx:])
    correction_val = (value_at_risk_tail_cdf_prob - alpha) * value_at_risk
    
    # Calculate CVaR
    mask = values >= value_at_risk
    conditional_values = values[mask]
    conditional_weights = weights[mask]
    cvar = (np.sum(conditional_values * conditional_weights) - correction_val) / alpha
    
    return float(cvar)

