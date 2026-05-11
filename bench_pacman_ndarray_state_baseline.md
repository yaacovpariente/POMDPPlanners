# pacman native C++ port: runtime comparison

Measurement setup (same as PR #87): numpy seed 0 + `_native.set_seed(0)`, warmup=100, measure=1000, rollout episodes=1000 cap=50, belief particles=200 updates=50, POMCPOW budget=30s. 7x7 maze, 2 ghosts, 4 pellets (`ghost_coordination="independent"`, `ghost_aggressiveness=2.0`).

Run from the worktree with:

```
source .venv/bin/activate
python bench_pacman_ndarray_state.py
```

## Full history (develop → ndarray → native)

| case | Python (pre-#87) | ndarray (post-#87) | native (this PR) | native vs Python | native vs ndarray |
|---|---|---|---|---|---|
| `reward` per-call | 37.07 µs | 44.45 µs | **5.36 µs** | **6.9× faster** | 8.3× faster |
| `state_transition_model.sample()` per-call | 35.84 µs | 44.10 µs | **3.94 µs** | **9.1× faster** | 11.2× faster |
| `observation_model.sample()` per-call | n/a | n/a | **3.87 µs** | new | new |
| random-policy rollout median / total (1000 eps) | 0.767 ms / 0.884 s | 0.855 ms / 0.981 s | **0.105 ms / 0.121 s** | **7.3× faster** | 8.1× faster |
| `VectorizedWeightedParticleBelief.update` per-call | 0.199 ms | 0.187 ms | **0.030 ms** | **6.6× faster** | 6.2× faster |
| `make_state` per-call | n/a | 1.95 µs | 1.85 µs | new | ~same |
| POMCPOW sims/sec (30s budget, 200 particles) | n/a | n/a | **3,817 sims/s** | new | new |
| mixed-strategy `sample()` per-call | n/a | n/a | **3.77 µs** | new | new |
| mixed-strategy belief.update per-call | n/a | n/a | **0.024 ms** | new | new |

## Reading the numbers

- **Every scalar per-call hot path drops ~7–11×** — the Python-scalar ndarray indexing tax from PR #87 is erased. `reward`, `state_transition_model.sample()`, `observation_model.sample()` all finish in ~4–5 µs where they previously took 35–44 µs. Random-policy rollouts drop from ~0.9 ms/episode to ~0.1 ms/episode (7.3× faster than pre-#87).
- **Vectorized belief update drops 6.6×** — the batched kernel (`PacManVectorizedUpdater.batch_transition` / `batch_observation_log_likelihood`) dispatches straight into `PacManTransitionCpp.batch_sample` / `PacManObservationCpp.batch_log_likelihood`. Each 200-particle update is ~30 µs vs 200 µs before.
- **POMCPOW sims/sec = 3,817** on a 30-second budget with the default POMCPOW hyperparameters (`k_a = k_o = 10.0`, `α = 0.5`, `c = 1.0`, `depth = 20`, `γ = 0.95`, 200 particles).
- Mixed-strategy rows confirm that the non-independent + patrol branches in C++ do not regress vs. the independent + aggressive configuration (both ~4 µs / sample, ~25 µs / belief update).

The `make_state` row is unchanged from PR #87 since it's purely a ndarray builder (no transition work).
