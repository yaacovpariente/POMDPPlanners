"""Layer 1: Environment-only benchmarks.

Measures environment operations in isolation so that improvements or
regressions can be attributed to environment code, not planner code.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# TigerPOMDP benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="env-sample_next_step")
def test_bench_tiger_sample_next_step(benchmark, tiger_state_action):
    """Benchmark TigerPOMDP.sample_next_step.

    Purpose: Measure full environment step performance for TigerPOMDP.

    Given: A TigerPOMDP environment with a valid initial state and action.
    When: sample_next_step is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = tiger_state_action
    np.random.seed(42)
    benchmark(env.sample_next_step, state, action)


@pytest.mark.benchmark(group="env-state_transition")
def test_bench_tiger_state_transition(benchmark, tiger_state_action):
    """Benchmark TigerPOMDP state transition model.

    Purpose: Measure state transition sampling in isolation.

    Given: A TigerPOMDP environment with a valid state and action.
    When: state_transition_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = tiger_state_action
    np.random.seed(42)

    def run():
        return env.state_transition_model(state=state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-observation_model")
def test_bench_tiger_observation_model(benchmark, tiger_state_action):
    """Benchmark TigerPOMDP observation model.

    Purpose: Measure observation sampling in isolation.

    Given: A TigerPOMDP environment with a sampled next state and action.
    When: observation_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = tiger_state_action
    np.random.seed(42)
    next_state = env.state_transition_model(state=state, action=action).sample()[0]

    def run():
        return env.observation_model(next_state=next_state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-reward")
def test_bench_tiger_reward(benchmark, tiger_state_action):
    """Benchmark TigerPOMDP reward computation.

    Purpose: Measure reward function performance.

    Given: A TigerPOMDP environment with a valid state and action.
    When: reward is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = tiger_state_action
    benchmark(env.reward, state=state, action=action)


@pytest.mark.benchmark(group="env-is_terminal")
def test_bench_tiger_is_terminal(benchmark, tiger_state_action):
    """Benchmark TigerPOMDP terminal check.

    Purpose: Measure terminal state check performance.

    Given: A TigerPOMDP environment with a valid state.
    When: is_terminal is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, _ = tiger_state_action
    benchmark(env.is_terminal, state)


# ---------------------------------------------------------------------------
# DiscreteLightDarkPOMDP benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="env-sample_next_step")
def test_bench_discrete_ld_sample_next_step(benchmark, discrete_ld_state_action):
    """Benchmark DiscreteLightDarkPOMDP.sample_next_step.

    Purpose: Measure full environment step performance for DiscreteLightDarkPOMDP.

    Given: A DiscreteLightDarkPOMDP environment with a valid initial state and action.
    When: sample_next_step is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = discrete_ld_state_action
    np.random.seed(42)
    benchmark(env.sample_next_step, state, action)


@pytest.mark.benchmark(group="env-state_transition")
def test_bench_discrete_ld_state_transition(benchmark, discrete_ld_state_action):
    """Benchmark DiscreteLightDarkPOMDP state transition model.

    Purpose: Measure state transition sampling in isolation.

    Given: A DiscreteLightDarkPOMDP with a valid state and action.
    When: state_transition_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = discrete_ld_state_action
    np.random.seed(42)

    def run():
        return env.state_transition_model(state=state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-observation_model")
def test_bench_discrete_ld_observation_model(benchmark, discrete_ld_state_action):
    """Benchmark DiscreteLightDarkPOMDP observation model.

    Purpose: Measure observation sampling in isolation.

    Given: A DiscreteLightDarkPOMDP with a sampled next state and action.
    When: observation_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = discrete_ld_state_action
    np.random.seed(42)
    next_state = env.state_transition_model(state=state, action=action).sample()[0]

    def run():
        return env.observation_model(next_state=next_state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-reward")
def test_bench_discrete_ld_reward(benchmark, discrete_ld_state_action):
    """Benchmark DiscreteLightDarkPOMDP reward computation.

    Purpose: Measure reward function performance.

    Given: A DiscreteLightDarkPOMDP with a valid state and action.
    When: reward is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = discrete_ld_state_action
    benchmark(env.reward, state=state, action=action)


@pytest.mark.benchmark(group="env-is_terminal")
def test_bench_discrete_ld_is_terminal(benchmark, discrete_ld_state_action):
    """Benchmark DiscreteLightDarkPOMDP terminal check.

    Purpose: Measure terminal state check performance.

    Given: A DiscreteLightDarkPOMDP with a valid state.
    When: is_terminal is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, _ = discrete_ld_state_action
    benchmark(env.is_terminal, state)


# ---------------------------------------------------------------------------
# RockSamplePOMDP benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="env-sample_next_step")
def test_bench_rock_sample_sample_next_step(benchmark, rock_sample_state_action):
    """Benchmark RockSamplePOMDP.sample_next_step.

    Purpose: Measure full environment step performance for RockSamplePOMDP.

    Given: A RockSamplePOMDP environment with a valid initial state and action.
    When: sample_next_step is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = rock_sample_state_action
    np.random.seed(42)
    benchmark(env.sample_next_step, state, action)


@pytest.mark.benchmark(group="env-state_transition")
def test_bench_rock_sample_state_transition(benchmark, rock_sample_state_action):
    """Benchmark RockSamplePOMDP state transition model.

    Purpose: Measure state transition sampling in isolation.

    Given: A RockSamplePOMDP with a valid state and action.
    When: state_transition_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = rock_sample_state_action
    np.random.seed(42)

    def run():
        return env.state_transition_model(state=state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-observation_model")
def test_bench_rock_sample_observation_model(benchmark, rock_sample_state_action):
    """Benchmark RockSamplePOMDP observation model.

    Purpose: Measure observation sampling in isolation.

    Given: A RockSamplePOMDP with a sampled next state and action.
    When: observation_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = rock_sample_state_action
    np.random.seed(42)
    next_state = env.state_transition_model(state=state, action=action).sample()[0]

    def run():
        return env.observation_model(next_state=next_state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-reward")
def test_bench_rock_sample_reward(benchmark, rock_sample_state_action):
    """Benchmark RockSamplePOMDP reward computation.

    Purpose: Measure reward function performance.

    Given: A RockSamplePOMDP with a valid state and action.
    When: reward is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = rock_sample_state_action
    benchmark(env.reward, state=state, action=action)


@pytest.mark.benchmark(group="env-is_terminal")
def test_bench_rock_sample_is_terminal(benchmark, rock_sample_state_action):
    """Benchmark RockSamplePOMDP terminal check.

    Purpose: Measure terminal state check performance.

    Given: A RockSamplePOMDP with a valid state.
    When: is_terminal is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, _ = rock_sample_state_action
    benchmark(env.is_terminal, state)


# ---------------------------------------------------------------------------
# LaserTagPOMDP benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="env-sample_next_step")
def test_bench_laser_tag_sample_next_step(benchmark, laser_tag_state_action):
    """Benchmark LaserTagPOMDP.sample_next_step.

    Purpose: Measure full environment step performance for LaserTagPOMDP.

    Given: A LaserTagPOMDP environment with a valid initial state and action.
    When: sample_next_step is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = laser_tag_state_action
    np.random.seed(42)
    benchmark(env.sample_next_step, state, action)


@pytest.mark.benchmark(group="env-state_transition")
def test_bench_laser_tag_state_transition(benchmark, laser_tag_state_action):
    """Benchmark LaserTagPOMDP state transition model.

    Purpose: Measure state transition sampling in isolation.

    Given: A LaserTagPOMDP with a valid state and action.
    When: state_transition_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = laser_tag_state_action
    np.random.seed(42)

    def run():
        return env.state_transition_model(state=state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-observation_model")
def test_bench_laser_tag_observation_model(benchmark, laser_tag_state_action):
    """Benchmark LaserTagPOMDP observation model.

    Purpose: Measure observation sampling in isolation.

    Given: A LaserTagPOMDP with a sampled next state and action.
    When: observation_model is called and sampled repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = laser_tag_state_action
    np.random.seed(42)
    next_state = env.state_transition_model(state=state, action=action).sample()[0]

    def run():
        return env.observation_model(next_state=next_state, action=action).sample()[0]

    benchmark(run)


@pytest.mark.benchmark(group="env-reward")
def test_bench_laser_tag_reward(benchmark, laser_tag_state_action):
    """Benchmark LaserTagPOMDP reward computation.

    Purpose: Measure reward function performance.

    Given: A LaserTagPOMDP with a valid state and action.
    When: reward is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = laser_tag_state_action
    benchmark(env.reward, state=state, action=action)


@pytest.mark.benchmark(group="env-is_terminal")
def test_bench_laser_tag_is_terminal(benchmark, laser_tag_state_action):
    """Benchmark LaserTagPOMDP terminal check.

    Purpose: Measure terminal state check performance.

    Given: A LaserTagPOMDP with a valid state.
    When: is_terminal is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, _ = laser_tag_state_action
    benchmark(env.is_terminal, state)
