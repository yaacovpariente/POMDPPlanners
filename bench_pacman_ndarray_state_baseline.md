# pacman ndarray-state refactor: runtime comparison

Measurement setup: numpy seed 0, warmup=100, measure=1000, rollout episodes=1000 cap=50, belief particles=200 updates=50. Run from the worktree with:

```
source .venv/bin/activate
PYTHONPATH=. python bench_pacman_ndarray_state.py
```

| case | before (`origin/develop`) | after (`refactor/pacman-ndarray-state`) | Δ |
|---|---|---|---|
| `reward` per-call | 37.07 µs | 44.45 µs | **+19.9%** |
| `state_transition_model.sample()` per-call | 35.84 µs | 44.10 µs | **+23.0%** |
| random-policy rollout median / total (1000 eps) | 0.767 ms / 0.884 s | 0.855 ms / 0.981 s | **+11.0%** |
| `VectorizedWeightedParticleBelief.update` per-call | 0.199 ms | 0.187 ms | **−6.0%** |
| `make_state` per-call | n/a | 1.95 µs | (new API) |

## Reading the numbers

- **Vectorized belief update is ~6% faster** — the env's canonical state representation now matches the particle-array layout, so the update path no longer needs the old `state_to_array` / `array_to_state` conversion at the env↔belief boundary. This is the headline improvement.
- **Scalar paths (`reward`, `sample`, random-policy rollout) regressed 11–23%** — pure-Python numpy scalar indexing (`int(state[idx])`, `float(state[idx])`) is inherently slower than Python-native tuple/dataclass attribute reads. This tax is expected and unavoidable without either numba/cython or, more directly, the planned C++ port of the PacMan env (the actual motivation for this refactor).

The trade-off is intentional: the structural alignment with the vectorized belief and the C++ port path is worth the temporary scalar-path slowdown.
