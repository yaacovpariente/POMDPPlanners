# SPDX-License-Identifier: MIT

"""Collect rollouts from the true LightDark and fit the transition and observation MLPs."""

import json
import time
from pathlib import Path

import numpy as np

from experiments.add_env_model_gate_revision import data as D
from experiments.add_env_model_gate_revision import train as T
from experiments.add_env_model_gate_revision import world as W
from experiments.add_env_model_gate_revision.models import GaussianMLPObservation, GaussianMLPTransition

OUT = Path("results/add-env-model-gate-revision/learned_model")
NUM_EPISODES = 400
NUM_STEPS = 40
HOLD_STEPS = 3
RANDOM_START_FRACTION = 0.5
HOLDOUT_FRACTION = 0.2
SEED = 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    world = W.make_world()
    t0 = time.time()
    episodes = D.collect_episodes(world, NUM_EPISODES, NUM_STEPS, HOLD_STEPS, RANDOM_START_FRACTION, SEED)
    collect_seconds = time.time() - t0
    train_eps, holdout_eps = D.split_episodes(episodes, HOLDOUT_FRACTION, SEED)
    np.savez(
        OUT / "rollouts.npz",
        **{f"train_{k}": np.concatenate([ep[k] for ep in train_eps]) for k in ("states", "actions", "next_states", "observations")},
        **{f"holdout_{k}": np.concatenate([ep[k] for ep in holdout_eps]) for k in ("states", "actions", "next_states", "observations")},
        train_episode_index=np.array([int(ep["index"]) for ep in train_eps]),
        holdout_episode_index=np.array([int(ep["index"]) for ep in holdout_eps]),
    )
    tb, hb = D.transition_batch(train_eps), D.transition_batch(holdout_eps)
    ob, hob = D.observation_batch(train_eps), D.observation_batch(holdout_eps)

    t0 = time.time()
    t_core, t_rec = T.fit_gaussian_mlp(
        np.concatenate([tb.states, tb.actions], axis=1), tb.deltas,
        np.concatenate([hb.states, hb.actions], axis=1), hb.deltas, seed=SEED,
    )
    t_rec["fit_seconds"] = time.time() - t0
    t0 = time.time()
    o_core, o_rec = T.fit_gaussian_mlp(
        ob.next_states, ob.observations - ob.next_states,
        hob.next_states, hob.observations - hob.next_states, seed=SEED,
    )
    o_rec["fit_seconds"] = time.time() - t0

    transition = GaussianMLPTransition(t_core, t_rec)
    observation = GaussianMLPObservation(o_core, o_rec)
    transition.save(OUT / "transition.npz")
    observation.save(OUT / "observation.npz")
    summary = {
        "data": {
            "episodes_collected": len(episodes), "train_episodes": len(train_eps), "holdout_episodes": len(holdout_eps),
            "train_rows": len(tb), "holdout_rows": len(hb), "num_steps": NUM_STEPS, "hold_steps": HOLD_STEPS,
            "random_start_fraction": RANDOM_START_FRACTION, "presets": W.PRESETS.tolist(), "seed": SEED,
            "collect_seconds": collect_seconds,
        },
        "transition": {**t_rec, "fingerprint": transition.fingerprint, "input": "(x, y, ax, ay)", "target": "next - state"},
        "observation": {**o_rec, "fingerprint": observation.fingerprint, "input": "(x, y) of next state", "target": "obs - next"},
    }
    (OUT / "training.json").write_text(json.dumps(summary, indent=2))
    for name, rec in (("transition", t_rec), ("observation", o_rec)):
        print(f"{name}: rows {rec['num_train_rows']}/{rec['num_holdout_rows']} epochs {rec['epochs_run']} best {rec['best_epoch']} "
              f"stopped_early {rec['stopped_early']} holdout NLL(raw) {rec['best_holdout_nll_raw']:.4f} fit {rec['fit_seconds']:.1f}s")


if __name__ == "__main__":
    main()
