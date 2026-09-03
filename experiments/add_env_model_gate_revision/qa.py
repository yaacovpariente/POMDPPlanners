# SPDX-License-Identifier: MIT

"""env-qa for the closed-loop stage: does PFT-DPW beat random on the pinned LightDark, and at what cost?

Runs a bounded ladder of exploration constants against a random baseline on one
replicate, through the simulations API. The histories also give the decision
time and simulation count per decision -- the calibration numbers.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.add_env_model_gate_revision import closed_loop as CL
from experiments.add_env_model_gate_revision import world as W

OUT = Path("results/add-env-model-gate-revision")


def main(num_episodes: int = 20, n_jobs: int = 8) -> None:
    world = W.make_world()
    truth = W.make_truth_model(world)
    policies = [
        CL.make_planner(truth, exploration_constant=1.0, name="PFT_DPW_c1"),
        CL.make_planner(truth, exploration_constant=10.0, name="PFT_DPW_c10"),
        CL.RandomPolicy(truth, name="Random"),
    ]
    start = time.time()
    batch = CL.run_batch("qa", policies, replicate_seeds=[0], num_episodes=num_episodes, cache_dir=OUT / "cache" / "qa", n_jobs=n_jobs)
    elapsed = time.time() - start
    rows = CL.episode_records(batch["results"], world, arm="qa")
    CL.write_records(rows, OUT / "qa" / "episodes.csv")
    frame = pd.DataFrame(rows)
    summary = frame.groupby("policy").agg(
        episodes=("episode", "count"),
        mean_return=("discounted_return", "mean"),
        se_return=("discounted_return", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        goal_rate=("goal_reached", "mean"),
        hazard_rate=("hazard_hit", "mean"),
        mean_steps=("steps", "mean"),
        median_action_s=("mean_action_seconds", "median"),
        p90_action_s=("mean_action_seconds", lambda v: v.quantile(0.9)),
        mean_belief_update_s=("mean_belief_update_seconds", "mean"),
        sims_per_decision=("mean_simulations_per_decision", "mean"),
    )
    (OUT / "qa").mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "qa" / "summary.csv")
    batch["stats"].to_csv(OUT / "qa" / "api_statistics.csv", index=False)
    (OUT / "qa" / "timing.json").write_text(json.dumps({"wall_seconds": elapsed, "num_episodes": num_episodes, "n_jobs": n_jobs}))
    print(summary.to_string())
    print(f"wall {elapsed:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
