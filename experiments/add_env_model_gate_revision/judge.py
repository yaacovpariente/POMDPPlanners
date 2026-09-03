# SPDX-License-Identifier: MIT

"""Apply the evidence and control gates to the closed-loop records, and collect every gate into one table.

The control rule is the one in ``training/model_learning/acceptance.py``
(git-ignored; its constants are restated here in case it is absent): a
one-sided Welch test on per-seed mean returns, candidate >= incumbent - 10 %
of |incumbent|, alpha 0.05; and the task metric (goal-reaching rate) not lower.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

try:
    from POMDPPlanners.training.model_learning import acceptance as A

    ALPHA, PARITY_MARGIN_FRACTION, MIN_EPISODES_PER_SEED, MIN_SEEDS = (
        A.ALPHA, A.PARITY_MARGIN_FRACTION, A.MIN_EPISODES_PER_SEED, A.MIN_SEEDS,
    )
    ACCEPTANCE_SOURCE = "POMDPPlanners/training/model_learning/acceptance.py"
except ImportError:  # the file is git-ignored and may be absent
    ALPHA, PARITY_MARGIN_FRACTION, MIN_EPISODES_PER_SEED, MIN_SEEDS = 0.05, 0.10, 50, 3
    ACCEPTANCE_SOURCE = "restated in add-env-model SKILL.md (acceptance.py absent)"

OUT = Path("results/add-env-model-gate-revision")
ARMS = ("truth", "learned", "bad")
INCUMBENT = "truth"


def _welch_greater(first: np.ndarray, second: np.ndarray, shift: float = 0.0) -> float:
    if first.size < 2 or second.size < 2:
        return float("nan")
    return float(stats.ttest_ind(first, second - shift, equal_var=False, alternative="greater").pvalue)


def load_arms() -> Dict[str, pd.DataFrame]:
    return {arm: pd.read_csv(OUT / "closed_loop" / arm / "episodes.csv") for arm in ARMS}


def closed_loop_table(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, frame in frames.items():
        seed_means = frame.groupby("replicate")["discounted_return"].mean()
        rows.append({
            "arm": arm,
            "mean_return": frame.discounted_return.mean(),
            "se_over_episodes": frame.discounted_return.std(ddof=1) / np.sqrt(len(frame)),
            "se_over_seeds": seed_means.std(ddof=1) / np.sqrt(len(seed_means)),
            "std_over_episodes": frame.discounted_return.std(ddof=1),
            "per_seed_means": [round(float(v), 3) for v in seed_means],
            "episodes": len(frame),
            "seeds": frame.replicate.nunique(),
            "episodes_per_seed": int(frame.groupby("replicate").size().min()),
            "goal_rate": frame.goal_reached.mean(),
            "out_of_grid_rate": frame.out_of_grid.mean(),
            "mean_steps": frame.steps.mean(),
            "median_action_s": frame.mean_action_seconds.median(),
            "p90_action_s": frame.mean_action_seconds.quantile(0.9),
            "sims_per_decision": frame.mean_simulations_per_decision.mean(),
        })
    return pd.DataFrame(rows).set_index("arm")


def control_gates(frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    incumbent = frames[INCUMBENT]
    inc_means = incumbent.groupby("replicate")["discounted_return"].mean()
    inc_goal = incumbent.groupby("replicate")["goal_reached"].mean()
    margin = PARITY_MARGIN_FRACTION * abs(inc_means.mean())
    verdicts: Dict[str, Dict[str, Any]] = {}
    for arm, frame in frames.items():
        means = frame.groupby("replicate")["discounted_return"].mean()
        goal = frame.groupby("replicate")["goal_reached"].mean()
        seeds_match = set(means.index) == set(inc_means.index)
        evidence = bool(
            len(means) >= MIN_SEEDS and frame.groupby("replicate").size().min() >= MIN_EPISODES_PER_SEED and seeds_match
        )
        p_parity = _welch_greater(means.to_numpy(), inc_means.to_numpy(), shift=margin)
        p_worse = _welch_greater(inc_means.to_numpy(), means.to_numpy())  # is the candidate actually worse?
        goal_not_lower = bool(goal.mean() >= inc_goal.mean())
        verdicts[arm] = {
            "evidence": {"seeds": int(len(means)), "episodes_per_seed": int(frame.groupby("replicate").size().min()),
                         "threshold": f">= {MIN_SEEDS} seeds, >= {MIN_EPISODES_PER_SEED} episodes/seed", "pass": evidence},
            "control_return": {"candidate_mean": float(means.mean()), "incumbent_mean": float(inc_means.mean()),
                               "margin": float(margin), "threshold": float(inc_means.mean() - margin),
                               "p_parity": p_parity, "p_candidate_worse_than_incumbent": p_worse, "alpha": ALPHA,
                               "per_seed_means": [float(v) for v in means], "pass": bool(p_parity < ALPHA)},
            "control_task_metric": {"candidate_goal_rate": float(goal.mean()), "incumbent_goal_rate": float(inc_goal.mean()),
                                    "pass": goal_not_lower},
        }
        verdicts[arm]["control"] = {"pass": bool(verdicts[arm]["control_return"]["pass"] and goal_not_lower)}
    return verdicts


def main() -> Dict[str, Any]:
    frames = load_arms()
    table = closed_loop_table(frames)
    stage4 = json.loads((OUT / "diagnostics.json").read_text())
    contract = json.loads((OUT / "contract.json").read_text())
    if not contract.get("pass"):
        raise RuntimeError("contract tests did not pass; refusing to judge gates")
    control = control_gates(frames)
    gates: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        s4 = stage4["verdicts"][arm]
        gates[arm] = {
            "contract": contract,
            "calibration": s4["calibration"], "likelihood": s4["likelihood"],
            "observation": s4["observation"], "ranking": s4["ranking"],
            "evidence": control[arm]["evidence"],
            "control": {**control[arm]["control"], "return": control[arm]["control_return"], "task_metric": control[arm]["control_task_metric"]},
        }
        failed = [g for g in ("contract", "calibration", "likelihood", "observation", "ranking", "evidence", "control") if not gates[arm][g]["pass"]]
        gates[arm]["verdict"] = "pass" if not failed else "fail"
        gates[arm]["failed_gates"] = failed
    out = {"acceptance_constants_source": ACCEPTANCE_SOURCE, "gates": gates,
           "closed_loop": json.loads(table.reset_index().to_json(orient="records"))}
    (OUT / "gates.json").write_text(json.dumps(out, indent=2))
    table.to_csv(OUT / "closed_loop_table.csv")
    print(table.to_string())
    for arm in ARMS:
        c = control[arm]
        print(arm, gates[arm]["verdict"], gates[arm]["failed_gates"],
              "p_parity", round(c["control_return"]["p_parity"], 4), "p_worse", round(c["control_return"]["p_candidate_worse_than_incumbent"], 4),
              "goal", c["control_task_metric"]["candidate_goal_rate"])
    return out


if __name__ == "__main__":
    main()
