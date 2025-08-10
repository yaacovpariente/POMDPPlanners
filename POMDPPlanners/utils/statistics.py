import numpy as np
from scipy import stats
from scipy.stats import binom

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
        Risk analysis of POMDP algorithm performance::
        
            import numpy as np
            from POMDPPlanners.utils.statistics import cvar_estimator
            
            # Simulate algorithm returns from multiple episodes  
            returns = np.array([12.5, 8.3, 15.7, -2.1, 9.8, 13.2, 6.4, 11.0, -1.5, 14.3])
            
            # Calculate risk metrics
            mean_return = np.mean(returns)
            cvar_90 = cvar_estimator(returns, alpha=0.9)  # Worst 10% outcomes
            cvar_95 = cvar_estimator(returns, alpha=0.95) # Worst 5% outcomes
            
            print(f"Mean return: {mean_return:.2f}")
            print(f"CVaR (90%): {cvar_90:.2f}")  # Focus on worst 10%
            print(f"CVaR (95%): {cvar_95:.2f}")  # Focus on worst 5% 
            
            # Lower CVaR indicates higher tail risk
            
    Example:
        Comparing algorithm risk profiles::
        
            # Algorithm performance data from experiments
            pomcp_returns = np.array([10.2, 12.8, 9.5, 11.3, 8.7, 12.1, 10.9, 9.8, 11.5, 10.4])
            pft_returns = np.array([15.1, 7.2, 14.8, 13.3, 6.9, 15.5, 8.1, 14.2, 12.7, 9.3])
            
            # Compare risk-adjusted performance
            pomcp_mean = np.mean(pomcp_returns)
            pft_mean = np.mean(pft_returns) 
            
            pomcp_cvar = cvar_estimator(pomcp_returns, alpha=0.9)
            pft_cvar = cvar_estimator(pft_returns, alpha=0.9)
            
            print("Algorithm Risk Comparison:")
            print(f"POMCP - Mean: {pomcp_mean:.2f}, CVaR(90%): {pomcp_cvar:.2f}")
            print(f"PFT-DPW - Mean: {pft_mean:.2f}, CVaR(90%): {pft_cvar:.2f}")
            
            # Interpretation:
            # - Higher mean = better average performance  
            # - Higher CVaR = better worst-case performance (lower risk)
            
    Example:
        Safety-critical application analysis::
        
            # Safety constraint violations (negative rewards)
            safety_violations = np.array([-50, -20, -10, -100, -5, -30, -15, -25, -40, -60])
            
            # Assess safety risk with different confidence levels
            risk_metrics = {}
            for alpha in [0.8, 0.9, 0.95, 0.99]:
                risk_metrics[alpha] = cvar_estimator(safety_violations, alpha)
                
            for alpha, cvar in risk_metrics.items():
                worst_pct = (1 - alpha) * 100
                print(f"CVaR({alpha:.0%}): {cvar:.1f} (worst {worst_pct:.0f}% avg)")
                
            # Use CVaR to set safety thresholds and compare algorithms
            
    Risk Assessment Applications:
        **Portfolio Analysis**: Compare multiple algorithms' risk-return profiles
        
        **Safety-Critical Systems**: Evaluate worst-case performance guarantees
        
        **Robust Planning**: Select algorithms with acceptable tail risk
        
        **Performance Bounds**: Establish confidence intervals for worst-case scenarios
        
    Mathematical Properties:
        - **Monotonic**: CVaR_α ≥ VaR_α (CVaR is always at least as large as VaR)
        - **Coherent**: Satisfies subadditivity, monotonicity, positive homogeneity
        - **Tail Sensitivity**: Higher α values emphasize extreme outcomes more
        - **Computational**: More stable than VaR, especially for small samples
    """
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
        Statistical comparison of algorithm performance::
        
            import numpy as np
            from POMDPPlanners.utils.statistics import confidence_interval
            
            # Algorithm performance from multiple runs
            pomcp_rewards = [12.3, 11.8, 13.1, 12.7, 11.9, 12.5, 13.0, 12.1, 12.8, 12.4]
            pft_rewards = [11.5, 13.2, 12.8, 11.9, 12.3, 13.5, 12.1, 12.9, 11.7, 12.6]
            
            # Calculate 95% confidence intervals
            pomcp_ci = confidence_interval(pomcp_rewards, confidence=0.95)
            pft_ci = confidence_interval(pft_rewards, confidence=0.95)
            
            print(f"POMCP mean: {np.mean(pomcp_rewards):.2f}")
            print(f"POMCP 95% CI: [{pomcp_ci[0]:.2f}, {pomcp_ci[1]:.2f}]")
            print(f"PFT-DPW mean: {np.mean(pft_rewards):.2f}")  
            print(f"PFT-DPW 95% CI: [{pft_ci[0]:.2f}, {pft_ci[1]:.2f}]")
            
            # Check for statistical significance
            if pomcp_ci[1] < pft_ci[0]:
                print("PFT-DPW significantly outperforms POMCP")
            elif pft_ci[1] < pomcp_ci[0]:
                print("POMCP significantly outperforms PFT-DPW") 
            else:
                print("No statistically significant difference")
                
    Example:
        Multi-algorithm performance study::
        
            # Results from comparative study
            algorithms = {
                'POMCP': [15.2, 14.8, 15.5, 14.9, 15.1, 15.3, 14.7, 15.0, 15.2, 14.8],
                'PFT_DPW': [16.1, 15.9, 16.3, 15.7, 16.0, 16.2, 15.8, 16.1, 15.9, 16.0],
                'SparsePFT': [14.5, 14.9, 14.7, 14.3, 14.8, 14.6, 14.4, 14.7, 14.5, 14.6],
                'OpenLoop': [12.3, 12.7, 12.1, 12.5, 12.4, 12.6, 12.2, 12.8, 12.3, 12.5]
            }
            
            # Statistical analysis with confidence intervals
            print("Algorithm Performance Analysis (95% CI):")
            print("-" * 50)
            for name, rewards in algorithms.items():
                mean_reward = np.mean(rewards)
                ci_lower, ci_upper = confidence_interval(rewards, confidence=0.95)
                ci_width = ci_upper - ci_lower
                
                print(f"{name:12}: {mean_reward:.2f} [{ci_lower:.2f}, {ci_upper:.2f}] (±{ci_width/2:.2f})")
                
    Example:
        Sample size analysis for experiment design::
        
            # Analyze how confidence interval width changes with sample size
            true_mean = 12.0
            std_dev = 2.0
            
            sample_sizes = [5, 10, 20, 50, 100]
            print("Sample Size Effect on Confidence Interval Width:")
            print("-" * 50)
            
            for n in sample_sizes:
                # Simulate data
                np.random.seed(42)  # For reproducibility
                data = np.random.normal(true_mean, std_dev, n)
                
                ci_lower, ci_upper = confidence_interval(data, confidence=0.95)
                ci_width = ci_upper - ci_lower
                
                print(f"n={n:3d}: CI width = {ci_width:.3f}")
                
            # Shows that larger samples give narrower, more precise intervals
            
    Statistical Interpretation:
        **Confidence Level**: 95% confidence means that if we repeated the
        experiment many times, 95% of computed intervals would contain the true mean.
        
        **Interval Width**: Narrower intervals indicate more precise estimates.
        Width depends on sample size, variance, and confidence level.
        
        **Overlapping Intervals**: If confidence intervals overlap significantly,
        there may not be a statistically meaningful difference between algorithms.
        
        **Non-overlapping Intervals**: Strong evidence of a real performance difference.
        
    Experimental Design Guidelines:
        **Sample Size**: Larger samples give narrower confidence intervals
        - n ≥ 30: Generally adequate for robust estimates  
        - n ≥ 10: Minimum for t-distribution approximation
        - n < 10: Use with caution, consider bootstrap methods
        
        **Confidence Level Selection**:
        - 90%: Less stringent, wider intervals, higher power
        - 95%: Standard for most scientific applications  
        - 99%: More stringent, narrower intervals, lower power
        
        **Multiple Comparisons**: When comparing many algorithms, consider
        Bonferroni correction or other methods to control family-wise error rate.
    """
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

    # If all values are the same, return the value as both bounds
    if np.all(data == data[0]):
        return (data[0], data[0])

    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    df = len(data) - 1  # Degrees of freedom

    # Confidence interval
    ci = stats.t.interval(confidence, df, loc=mean, scale=sem)
    return ci


def cvar_confidence_interval(data, alpha=0.95, delta=0.05):
    """
    Calculate the confidence interval for the CVaR of a dataset using the t-distribution.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval
        
    Raises:
        ValueError: If data contains NaN values or has insufficient samples
    """

    if not 0 <= alpha <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if len(data) == 0:
        raise ValueError("Input vector must not be empty")
    
    lower_bound = cvar_probabilistic_lower_bound_thomas(vec=data, alpha=alpha, delta=delta/2, dist_lower_bound=0)
    upper_bound = cvar_probabilistic_upper_bound_thomas(vec=data, alpha=alpha, delta=delta/2, dist_upper_bound=0)
    return lower_bound, upper_bound

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


def quantile_confidence_interval(data, alpha=0.95, conf_level=0.95):
    """
    data: 1D array-like of samples
    alpha: target quantile (e.g. 0.95 for 95% VaR)
    conf_level: overall coverage (e.g. 0.95 for 95% CI)
    Returns: (lower_value, upper_value, k1, k2)
    """
    x = np.sort(np.asarray(data))
    n = len(x)
    delta = 1 - conf_level

    # find smallest k1 such that P(Y < k1) >= delta/2
    k1 = 0
    while binom.cdf(k1-1, n, alpha) < delta/2 and k1 <= n:
        k1 += 1

    # find largest k2 such that P(Y <= k2) <= 1 - delta/2
    k2 = n
    while binom.cdf(k2, n, alpha) > 1 - delta/2 and k2 >= 0:
        k2 -= 1

    # clip to valid index range and convert to 0-based indexing
    k1 = max(1, min(k1, n))      # at least 1
    k2 = max(1, min(k2, n))      # at least 1

    return x[k1-1], x[k2-1], k1, k2

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
        
    if len(values) == 1:
        return values[0]

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

