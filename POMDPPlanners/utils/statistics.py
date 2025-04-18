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
    # TODO: test this function
    """
    Calculate the confidence interval for the mean of a dataset using the t-distribution.

    Parameters:
        data (array-like): Sample data
        confidence (float): Confidence level (default 0.95 for 95%)

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval
    """
    data = np.array(data)
    if len(data) <= 1:
        raise ValueError("Data must contain at least two elements")
    
    mean = np.mean(data)
    sem = stats.sem(data, nan_policy='omit')  # Standard error of the mean
    df = len(data) - 1  # Degrees of freedom

    # Confidence interval
    ci = stats.t.interval(confidence, df, loc=mean, scale=sem)
    return ci    

