"""Reusable test utilities for verifying metric name consistency.

This module provides generic test functions to verify that the metric names
declared by environments and policies match the actual metrics they produce.
"""

from typing import List, Type

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import History


def verify_environment_metric_consistency(
    environment: Environment,
    histories: List[History],
) -> None:
    """Verify that environment's declared metric names match actual produced metrics.

    This function checks that:
    1. get_metric_names() returns the exact set of metric names
    2. compute_metrics() produces MetricValue objects with those exact names
    3. No extra or missing metrics

    Args:
        environment: Environment instance to test
        histories: List of episode histories to pass to compute_metrics()

    Raises:
        AssertionError: If declared and actual metric names don't match

    Example:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.simulations.episodes import run_episode
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.utils.logger import get_logger
        >>>
        >>> # Set up environment and run episodes
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> policy = POMCP(environment=env, discount_factor=0.95, depth=5,
        ...                exploration_constant=1.0, name="test", n_simulations=10)
        >>> belief = get_initial_belief(env, n_particles=10)
        >>> logger = get_logger("test", debug=False)
        >>> history = run_episode(env, policy, belief, num_steps=3, logger=logger)
        >>>
        >>> # Verify consistency
        >>> verify_environment_metric_consistency(env, [history])  # Should pass without error
    """
    # Get declared metric names
    declared_names = set(environment.get_metric_names())

    # Get actual metric names from compute_metrics()
    actual_metrics = environment.compute_metrics(histories)
    actual_names = set(metric.name for metric in actual_metrics)

    # Verify exact match
    missing_metrics = declared_names - actual_names
    extra_metrics = actual_names - declared_names

    error_messages = []

    if missing_metrics:
        error_messages.append(
            f"Environment {environment.__class__.__name__} declares metrics in get_metric_names() "
            f"that are not produced by compute_metrics(): {sorted(missing_metrics)}"
        )

    if extra_metrics:
        error_messages.append(
            f"Environment {environment.__class__.__name__} produces metrics in compute_metrics() "
            f"that are not declared in get_metric_names(): {sorted(extra_metrics)}"
        )

    if error_messages:
        raise AssertionError("\n".join(error_messages))


def verify_policy_metric_consistency(
    policy: Policy,
    belief: Belief,
) -> None:
    """Verify that policy's declared info variable names match actual produced names.

    This function checks that:
    1. get_info_variable_names() returns the exact set of info variable names
    2. action() produces PolicyInfoVariable objects with those exact names
    3. No extra or missing info variables

    Args:
        policy: Policy instance to test
        belief: Belief state to pass to policy.action()

    Raises:
        AssertionError: If declared and actual info variable names don't match

    Example:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>>
        >>> # Set up policy
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> policy = POMCP(environment=env, discount_factor=0.95, depth=5,
        ...                exploration_constant=1.0, name="test", n_simulations=10)
        >>> belief = get_initial_belief(env, n_particles=10)
        >>>
        >>> # Verify consistency
        >>> verify_policy_metric_consistency(policy, belief)  # Should pass without error
    """
    # Get declared info variable names
    declared_names = set(policy.__class__.get_info_variable_names())

    # Get actual info variable names from action()
    _, policy_run_data = policy.action(belief)
    actual_names = set(info_var.name for info_var in policy_run_data.info_variables)

    # Verify exact match
    missing_variables = declared_names - actual_names
    extra_variables = actual_names - declared_names

    error_messages = []

    if missing_variables:
        error_messages.append(
            f"Policy {policy.__class__.__name__} declares info variables in get_info_variable_names() "
            f"that are not produced by action(): {sorted(missing_variables)}"
        )

    if extra_variables:
        error_messages.append(
            f"Policy {policy.__class__.__name__} produces info variables in action() "
            f"that are not declared in get_info_variable_names(): {sorted(extra_variables)}"
        )

    if error_messages:
        raise AssertionError("\n".join(error_messages))


def verify_policy_class_metric_consistency(
    policy_class: Type[Policy],
    policy_instance: Policy,
    belief: Belief,
) -> None:
    """Verify policy class and instance metric consistency.

    This is a convenience wrapper around verify_policy_metric_consistency()
    that also checks the policy class directly.

    Args:
        policy_class: Policy class to test
        policy_instance: Instance of the policy class
        belief: Belief state to pass to policy.action()

    Raises:
        AssertionError: If declared and actual info variable names don't match

    Example:
        >>> from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
        >>> from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
        >>> from POMDPPlanners.core.belief import get_initial_belief
        >>>
        >>> env = TigerPOMDP(discount_factor=0.95)
        >>> policy = POMCP(environment=env, discount_factor=0.95, depth=5,
        ...                exploration_constant=1.0, name="test", n_simulations=10)
        >>> belief = get_initial_belief(env, n_particles=10)
        >>>
        >>> verify_policy_class_metric_consistency(POMCP, policy, belief)
    """
    # Verify the instance
    verify_policy_metric_consistency(policy_instance, belief)

    # Also verify that the class method returns the same as instance's class
    assert (
        policy_class.get_info_variable_names()
        == policy_instance.__class__.get_info_variable_names()
    ), f"Policy class {policy_class.__name__} class method doesn't match instance class method"
