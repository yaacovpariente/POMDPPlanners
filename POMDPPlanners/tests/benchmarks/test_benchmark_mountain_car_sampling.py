"""MountainCar sampling hot-path benchmarks.

Captures the per-call cost of ``env.sample_next_state`` /
``env.sample_observation`` and the ``observation_log_probability`` /
``transition_log_probability`` paths, plus ``sample_next_step`` on the
POMDP. Used as the before/after baseline for the pybind11 C++ port of the
sampling hot path (see plan-the-implementation-and-recursive-pelican.md).

Run the baseline::

    pytest POMDPPlanners/tests/benchmarks/test_benchmark_mountain_car_sampling.py \
        -m benchmark --benchmark-save=0009_before_mc_cpp -v

And after the port::

    pytest POMDPPlanners/tests/benchmarks/test_benchmark_mountain_car_sampling.py \
        -m benchmark --benchmark-save=0010_after_mc_cpp -v
    pytest-benchmark compare 0009 0010
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]

_N_SAMPLES_BATCH = 100


# ---------------------------------------------------------------------------
# sample(n=1) — per-call cost including model instantiation
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="mc-transition-sample-n1")
def test_bench_mc_transition_sample_n1(benchmark, mountain_car_state_action):
    """Benchmark env.sample_next_state with n=1.

    Purpose: Measures per-call transition sampling cost (n=1).

    Given: A MountainCarPOMDP and a valid (state, action).
    When: env.sample_next_state(state, action) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)

    def run():
        return env.sample_next_state(state=state, action=action)

    benchmark(run)


@pytest.mark.benchmark(group="mc-observation-sample-n1")
def test_bench_mc_observation_sample_n1(benchmark, mountain_car_state_action):
    """Benchmark env.sample_observation with n=1.

    Purpose: Measures per-call observation sampling cost (n=1).

    Given: A MountainCarPOMDP and a sampled next state + action.
    When: env.sample_observation(next_state, action) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    next_state = env.sample_next_state(state=state, action=action)

    def run():
        return env.sample_observation(next_state=next_state, action=action)

    benchmark(run)


# ---------------------------------------------------------------------------
# sample(n=100) — amortised per-batch cost
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="mc-transition-sample-n100")
def test_bench_mc_transition_sample_n100(benchmark, mountain_car_state_action):
    """Benchmark env.sample_next_state with n=100.

    Purpose: Measures amortised transition sampling cost for n=100 samples
    per call, which stresses the inner Gaussian loop.

    Given: A MountainCarPOMDP and a valid (state, action).
    When: env.sample_next_state(state, action, n_samples=100) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)

    def run():
        return env.sample_next_state(state=state, action=action, n_samples=_N_SAMPLES_BATCH)

    benchmark(run)


@pytest.mark.benchmark(group="mc-observation-sample-n100")
def test_bench_mc_observation_sample_n100(benchmark, mountain_car_state_action):
    """Benchmark env.sample_observation with n=100.

    Purpose: Measures amortised observation sampling cost for n=100.

    Given: A MountainCarPOMDP and a sampled next state + action.
    When: env.sample_observation(next_state, action, n_samples=100) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    next_state = env.sample_next_state(state=state, action=action)

    def run():
        return env.sample_observation(
            next_state=next_state, action=action, n_samples=_N_SAMPLES_BATCH
        )

    benchmark(run)


# ---------------------------------------------------------------------------
# probability(n=1) and probability(n=100)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="mc-transition-probability-n1")
def test_bench_mc_transition_probability_n1(benchmark, mountain_car_state_action):
    """Benchmark env.transition_log_probability for a single value.

    Purpose: Measures per-call cost of the PDF evaluation for a single next-state.

    Given: A MountainCarPOMDP, a valid (state, action), and one sampled next state.
    When: env.transition_log_probability(state, action, [value]) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    sample_value = env.sample_next_state(state=state, action=action)
    values = [sample_value]

    def run():
        return env.transition_log_probability(state=state, action=action, next_states=values)

    benchmark(run)


@pytest.mark.benchmark(group="mc-transition-probability-n100")
def test_bench_mc_transition_probability_n100(benchmark, mountain_car_state_action):
    """Benchmark env.transition_log_probability for 100 values.

    Purpose: Measures amortised PDF evaluation cost for 100 next-states.

    Given: A MountainCarPOMDP, a valid (state, action), and 100 sampled next states.
    When: env.transition_log_probability(state, action, values) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    values = env.sample_next_state(state=state, action=action, n_samples=_N_SAMPLES_BATCH)

    def run():
        return env.transition_log_probability(state=state, action=action, next_states=values)

    benchmark(run)


@pytest.mark.benchmark(group="mc-observation-probability-n1")
def test_bench_mc_observation_probability_n1(benchmark, mountain_car_state_action):
    """Benchmark env.observation_log_probability for a single observation.

    Purpose: Measures per-call cost of the observation likelihood for one value.

    Given: A MountainCarPOMDP, a valid (state, action), a next state, and a sampled observation.
    When: env.observation_log_probability(next_state, action, [value]) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    next_state = env.sample_next_state(state=state, action=action)
    sample_value = env.sample_observation(next_state=next_state, action=action)
    values = [sample_value]

    def run():
        return env.observation_log_probability(
            next_state=next_state, action=action, observations=values
        )

    benchmark(run)


@pytest.mark.benchmark(group="mc-observation-probability-n100")
def test_bench_mc_observation_probability_n100(benchmark, mountain_car_state_action):
    """Benchmark env.observation_log_probability for 100 observations.

    Purpose: Measures amortised observation likelihood cost for 100 values.

    Given: A MountainCarPOMDP, a valid (state, action), a next state, and 100 sampled observations.
    When: env.observation_log_probability(next_state, action, values) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    next_state = env.sample_next_state(state=state, action=action)
    values = env.sample_observation(
        next_state=next_state, action=action, n_samples=_N_SAMPLES_BATCH
    )

    def run():
        return env.observation_log_probability(
            next_state=next_state, action=action, observations=values
        )

    benchmark(run)


# ---------------------------------------------------------------------------
# Full sample_next_step — the path the planner uses
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="mc-sample_next_step")
def test_bench_mc_sample_next_step(benchmark, mountain_car_state_action):
    """Benchmark MountainCarPOMDP.sample_next_step.

    Purpose: Measures the full per-step cost used by planner rollouts
    (transition sample + observation sample + reward lookup).

    Given: A MountainCarPOMDP and a valid (state, action).
    When: env.sample_next_step(state, action) is called repeatedly.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)
    benchmark(env.sample_next_step, state, action)


@pytest.mark.benchmark(group="mc-sample_next_step-loop1000")
def test_bench_mc_sample_next_step_loop1000(benchmark, mountain_car_state_action):
    """Benchmark 1000 iterations of sample_next_step from the same state.

    Purpose: Integration-level throughput measurement; amortises
    per-call pytest-benchmark overhead so the reported time reflects
    steady-state hot-path cost as experienced inside a planner loop.

    Given: A MountainCarPOMDP and a valid (state, action).
    When: sample_next_step is called 1000 times per benchmark iteration.
    Then: Execution time is recorded for regression tracking.

    Test type: performance
    """
    env, state, action = mountain_car_state_action
    np.random.seed(42)

    def run():
        current_state = state
        for _ in range(1000):
            current_state = env.sample_next_step(current_state, action)[0]
        return current_state

    benchmark(run)
