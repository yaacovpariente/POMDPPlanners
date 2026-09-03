# SPDX-License-Identifier: MIT

"""How much the ranking gate's bar moves with its own seed -- context for a verdict, not a gate.

``preset_ranking_agreement`` scores every preset by a 32-sample mean reward
under the model and under the world, and the truth against the world is
itself below 1. The gate compares a candidate to that number with no
tolerance, so the size of its seed-to-seed swing is what decides whether a
pass or fail means anything.
"""

import json
import time
from pathlib import Path

import numpy as np

from POMDPPlanners.training.model_learning import TransitionBatch
from POMDPPlanners.training.model_learning.diagnostics import preset_ranking_agreement

from experiments.add_env_model_gate_revision import world as W
from experiments.add_env_model_gate_revision.models import GaussianMLPTransition, bad_transition, truth_transition

OUT = Path("results/add-env-model-gate-revision")
SEEDS = range(8)
NUM_START_STATES = 64


def main() -> None:
    start = time.time()
    world = W.make_world()
    view = W.WorldPositionView(world)
    with np.load(OUT / "learned_model" / "rollouts.npz") as payload:
        holdout = TransitionBatch(payload["holdout_states"], payload["holdout_actions"], payload["holdout_next_states"])
    models = {
        "truth": truth_transition(world.state_transition_cov_matrix),
        "learned": GaussianMLPTransition.load(OUT / "learned_model" / "transition.npz"),
        "bad": bad_transition(world.state_transition_cov_matrix),
    }
    table = {name: {"32_samples": [], "256_samples": []} for name in models}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(holdout), size=NUM_START_STATES, replace=False)
        starts = [holdout.states[i] for i in chosen]
        for name, model in models.items():
            np.random.seed(seed)
            table[name]["32_samples"].append(preset_ranking_agreement(model, view, view, starts, W.PRESETS, num_samples=32))
            np.random.seed(seed)
            table[name]["256_samples"].append(preset_ranking_agreement(model, view, view, starts, W.PRESETS, num_samples=256))
    summary = {
        name: {key: {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))} for key, vals in d.items()}
        for name, d in table.items()
    }
    summary["wall_seconds"] = time.time() - start
    (OUT / "ranking_noise.json").write_text(json.dumps(summary, indent=2))
    for name in models:
        for key in ("32_samples", "256_samples"):
            s = summary[name][key]
            print(f"{name:8s} {key}: mean {s['mean']:.3f} std {s['std']:.3f} values {np.round(s['values'], 3).tolist()}")


if __name__ == "__main__":
    main()
