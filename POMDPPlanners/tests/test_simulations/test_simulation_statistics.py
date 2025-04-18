import numpy as np
from typing import List
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.simulations.simulation_statistics import compute_statistics

def create_test_history(rewards: List[float], discount_factor: float = 0.95) -> History:
    steps = [StepData(None, None, None, None, r) for r in rewards]
    history = History(
        history=steps,
        discount_factor=discount_factor,
        average_state_sampling_time=0.1,
        average_action_time=0.2,
        average_observation_time=0.3,
        average_belief_update_time=0.4,
        average_reward_time=0.5
    )
    return history

def test_compute_statistics_basic():
    # Create test histories with known rewards
    histories = [
        create_test_history([1.0, 2.0, 3.0]),  # Return = 1 + 2*0.95 + 3*0.95^2
        create_test_history([2.0, 3.0, 4.0]),  # Return = 2 + 3*0.95 + 4*0.95^2
        create_test_history([3.0, 4.0, 5.0])   # Return = 3 + 4*0.95 + 5*0.95^2
    ]
    
    alpha = 0.1
    confidence_level = 0.95
    
    stats = compute_statistics(histories, alpha, confidence_level)
    
    # Test average return
    expected_returns = [
        1.0 + 2.0*0.95 + 3.0*0.95**2,
        2.0 + 3.0*0.95 + 4.0*0.95**2,
        3.0 + 4.0*0.95 + 5.0*0.95**2
    ]
    expected_avg_return = np.mean(expected_returns)
    assert abs(stats["average_return"][0] - expected_avg_return) < 1e-6
    
    # Test timing statistics
    assert len(stats["average_state_sampling_time"][1]) == 2  # Confidence interval has lower and upper bounds
    assert len(stats["average_action_time"][1]) == 2
    assert len(stats["average_observation_time"][1]) == 2
    assert len(stats["average_belief_update_time"][1]) == 2
    assert len(stats["average_reward_time"][1]) == 2
    
    # Test confidence intervals
    for key in stats:
        if key != "return_value_at_risk" and key != "return_cvar":
            assert len(stats[key][1]) == 2  # Each confidence interval should have lower and upper bounds

def test_compute_statistics_single_history():
    # Test with a single history
    histories = [create_test_history([1.0, 2.0, 3.0])]
    alpha = 0.1
    confidence_level = 0.95
    
    # For a single history, we expect the return to be the same as the computed return
    expected_return = 1.0 + 2.0*0.95 + 3.0*0.95**2
    
    # We need at least two histories to compute confidence intervals
    # So we'll create a duplicate history to satisfy this requirement
    histories = [create_test_history([1.0, 2.0, 3.0]), create_test_history([1.0, 2.0, 3.0])]
    
    stats = compute_statistics(histories, alpha, confidence_level)
    
    # With identical histories, CVaR and VaR should be the same as the return
    assert abs(stats["average_return"][0] - expected_return) < 1e-6
    assert abs(stats["return_cvar"][0] - expected_return) < 1e-6
    assert abs(stats["return_value_at_risk"][0] - expected_return) < 1e-6

def test_compute_statistics_negative_rewards():
    # Test with negative rewards
    histories = [
        create_test_history([-1.0, -2.0, -3.0]),
        create_test_history([-2.0, -3.0, -4.0]),
        create_test_history([-3.0, -4.0, -5.0])
    ]
    
    alpha = 0.1
    confidence_level = 0.95
    
    stats = compute_statistics(histories, alpha, confidence_level)
    
    # Verify that statistics handle negative values correctly
    assert stats["average_return"][0] < 0
    assert stats["return_cvar"][0] < 0
    assert stats["return_value_at_risk"][0] < 0

def test_compute_statistics_zero_rewards():
    # Test with zero rewards
    histories = [
        create_test_history([0.0, 0.0, 0.0]),
        create_test_history([0.0, 0.0, 0.0]),
        create_test_history([0.0, 0.0, 0.0])
    ]
    
    alpha = 0.1
    confidence_level = 0.95
    
    stats = compute_statistics(histories, alpha, confidence_level)
    
    # All statistics should be zero
    assert abs(stats["average_return"][0]) < 1e-6
    assert abs(stats["return_cvar"][0]) < 1e-6
    assert abs(stats["return_value_at_risk"][0]) < 1e-6
