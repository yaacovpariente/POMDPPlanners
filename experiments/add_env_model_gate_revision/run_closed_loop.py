# SPDX-License-Identifier: MIT

"""Stage 5: the three arms in the true world, through the simulations API.

``calibrate`` runs three episodes per arm on replicate 0 and reports decision
time and simulations per decision; those episodes are cached under the same
keys the full run uses, so they are not paid twice. ``run`` runs the full
design: 50 episodes x 3 replicates per arm, same seeds in every arm.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from experiments.add_env_model_gate_revision import closed_loop as CL
from experiments.add_env_model_gate_revision import world as W
from experiments.add_env_model_gate_revision.models import GaussianMLPObservation, GaussianMLPTransition

OUT = Path("results/add-env-model-gate-revision")
CACHE = OUT / "cache" / "closed_loop"
REPLICATES = (0, 1, 2)
EPISODES_PER_REPLICATE = 50
N_JOBS = int(os.environ.get("GATE_REVISION_N_JOBS", "16"))


def arms(world: Any) -> Dict[str, Any]:
    learned_dir = OUT / "learned_model"
    return {
        "truth": W.make_truth_model(world),
        "learned": W.make_learned_model(
            GaussianMLPTransition.load(learned_dir / "transition.npz"),
            GaussianMLPObservation.load(learned_dir / "observation.npz"),
        ),
        "bad": W.make_bad_model(world),
    }


def run_arm(arm: str, model: Any, world: Any, replicates: Any, episodes: int, tag: str) -> pd.DataFrame:
    planner = CL.make_planner(model)
    start = time.time()
    batch = CL.run_batch(f"{tag}-{arm}", [planner], replicates, episodes, CACHE, N_JOBS)
    elapsed = time.time() - start
    rows = CL.episode_records(batch["results"], world, arm=arm)
    out_dir = OUT / tag / arm
    CL.write_records(rows, out_dir / "episodes.csv")
    batch["stats"].to_csv(out_dir / "api_statistics.csv", index=False)
    CL.write_provenance(out_dir / "models.jsonl", arm, model, world, planner)
    (out_dir / "timing.json").write_text(json.dumps({
        "wall_seconds": elapsed, "episodes": len(rows), "replicates": list(replicates), "n_jobs": N_JOBS,
        "episodes_from_cache_possible": True,
    }))
    frame = pd.DataFrame(rows)
    print(f"[{arm}] {len(rows)} episodes in {elapsed:.0f}s wall; mean return {frame.discounted_return.mean():.3f} "
          f"goal {frame.goal_reached.mean():.2f} steps {frame.steps.mean():.1f} "
          f"action {frame.mean_action_seconds.median():.3f}s (p90 {frame.mean_action_seconds.quantile(0.9):.3f}) "
          f"sims/decision {frame.mean_simulations_per_decision.mean():.0f}", flush=True)
    return frame


def merge(arm: str) -> pd.DataFrame:
    """Concatenate the per-replicate records of one arm into closed_loop/<arm>/episodes.csv."""
    parts = [pd.read_csv(OUT / "closed_loop" / f"{arm}-rep{k}" / "episodes.csv") for k in REPLICATES]
    frame = pd.concat(parts, ignore_index=True)
    out_dir = OUT / "closed_loop" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "episodes.csv", index=False)
    with (out_dir / "models.jsonl").open("w", encoding="utf-8") as handle:
        for k in REPLICATES:
            handle.write((OUT / "closed_loop" / f"{arm}-rep{k}" / "models.jsonl").read_text())
    timings = {k: json.loads((OUT / "closed_loop" / f"{arm}-rep{k}" / "timing.json").read_text()) for k in REPLICATES}
    (out_dir / "timing.json").write_text(json.dumps({"per_replicate": timings, "wall_seconds": sum(t["wall_seconds"] for t in timings.values())}))
    return frame


def main(argv: Any) -> None:
    mode = argv[0]
    world = W.make_world()
    if mode == "run":
        # One arm and one replicate per invocation: the cache is flushed on
        # every normal exit, so a call has to be small enough to finish.
        arm, replicate = argv[1], int(argv[2])
        run_arm(arm, arms(world)[arm], world, replicates=(replicate,), episodes=EPISODES_PER_REPLICATE, tag="closed_loop")
        (OUT / "closed_loop" / f"{arm}-rep{replicate}").mkdir(parents=True, exist_ok=True)
        for name in ("episodes.csv", "api_statistics.csv", "models.jsonl", "timing.json"):
            (OUT / "closed_loop" / arm / name).rename(OUT / "closed_loop" / f"{arm}-rep{replicate}" / name)
    elif mode == "merge":
        for arm in ("truth", "learned", "bad"):
            frame = merge(arm)
            print(arm, len(frame), "episodes", "mean return", round(frame.discounted_return.mean(), 3))
    else:
        raise SystemExit("usage: run_closed_loop.py run <arm> <seed> | merge")


if __name__ == "__main__":
    main(sys.argv[1:])
