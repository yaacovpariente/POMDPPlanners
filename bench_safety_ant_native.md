# Safety Ant Velocity — pre-port vs native C++ benchmark

Workload: one `WeightedParticleBelief.update(action, observation, pomdp)`
call, N = 100 particles, same action and observation within each case.
Measured with `pytest-benchmark` via
`POMDPPlanners/tests/benchmarks/test_benchmark_particle_belief_update.py`
on this branch.

Columns: mean / median / std-dev over `rounds` timed reps.

| case                                  | belief class                    | path       | mean      | median    | std-dev  | rounds |
| ------------------------------------- | ------------------------------- | ---------- | --------- | --------- | -------- | -----: |
| safety-ant-generic-python (pre-port)  | `WeightedParticleBelief`        | per-particle Python loop | 1471 μs   | 1462 μs   | 47.9 μs  |    593 |
| safety-ant-generic-cpp (post-port)    | `WeightedParticleBelief`        | C++ batch via shim       |   53.3 μs |   45.9 μs | 15.8 μs  |  5 084 |
| safety-ant-vectorized-numpy (pre-port)| `VectorizedWeightedParticleBelief` | NumPy batch          |  105 μs   |   95.8 μs | 98.8 μs  |  1 112 |
| safety-ant-vectorized-cpp (post-port) | `VectorizedWeightedParticleBelief` | C++ batch            |   24.3 μs |   23.6 μs |  2.36 μs | 19 802 |

## Speedups

- Generic `WeightedParticleBelief.update`: **~27.6× faster** (1471 μs → 53.3 μs, mean).
- Vectorized `VectorizedWeightedParticleBelief.update`: **~4.3× faster**
  (105 μs → 24.3 μs, mean).

Both numbers compare the pre-port reference implementations (kept verbatim in the
benchmark module as `_PrePortSafeAntVelocityTransition` / the numpy-only
vectorized updater closure) to the shipped post-port classes routed through
`_native.SafeAntVelocityTransitionCpp.batch_sample` and
`_native.SafeAntVelocityObservationCpp.batch_log_likelihood`.
