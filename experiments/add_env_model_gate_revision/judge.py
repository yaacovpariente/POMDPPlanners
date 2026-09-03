# SPDX-License-Identifier: MIT

"""Apply the evidence and control gates to the closed-loop records, and collect every gate into one table.

The control rule is the one in ``training/model_learning/acceptance.py``
(git-ignored; its constants are restated here in case it is absent): a
one-sided Welch test on per-seed mean returns, candidate >= incumbent - 10 %
of |incumbent|, alpha 0.05; and the task metric (goal-reaching rate) not lower.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

import POMDPPlanners

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
#: The worktree this experiment package sits in; the git facts are read from it.
REPO = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path = REPO) -> str:
    """Run git in ``cwd``, or return the error instead of raising.

    A provenance record that fails to write is worse than one that records why a
    field is missing, so a git failure is captured as the field's value.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - environment
        return f"unavailable: {error}"


def write_provenance(path: Path) -> Dict[str, Any]:
    """Record what produced the gates, read from code rather than transcribed.

    The previous provenance file was hand-written, and its revision named the
    parent commit rather than the experiment's, so nothing on disk said which
    version of this code ran. Every field below is read at run time: the git
    facts from this worktree, the settings from the module constants the run
    actually used. Fields the hand-written file carried that code cannot know --
    the remote user and path, the transfer note, the wall-clock timestamps --
    are dropped rather than guessed.
    """
    # Imported here, not at module import, so the judge still loads when the
    # closed-loop dependencies are unavailable.
    from experiments.add_env_model_gate_revision import closed_loop as CL
    from experiments.add_env_model_gate_revision import revised_diagnostics as RD
    from experiments.add_env_model_gate_revision import run_closed_loop as RC
    from experiments.add_env_model_gate_revision import world as W

    package_dir = Path(POMDPPlanners.__file__).resolve().parent
    record = {
        "written_by": "experiments/add_env_model_gate_revision/judge.py:write_provenance",
        "host": platform.node(),
        "python": sys.executable,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "worktree": str(REPO),
        "package": {
            "version": POMDPPlanners.__version__,
            "path": str(package_dir),
            # Resolved against the package's own directory, not this worktree:
            # the run imports POMDPPlanners from the venv's checkout, which can
            # sit at a different revision than the experiment code.
            "revision": _git("rev-parse", "HEAD", cwd=package_dir),
        },
        "planner": {
            "simulations_per_decision": CL.N_SIMULATIONS,
            "depth": W.PLANNING_DEPTH,
            "particles": CL.NUM_PARTICLES,
            "horizon": CL.NUM_STEPS,
            "exploration_constant": CL.EXPLORATION_CONSTANT,
            "policy_name": CL.POLICY_NAME,
        },
        "closed_loop": {
            "arms": list(ARMS),
            "seeds": list(RC.REPLICATES),
            "episodes_per_seed": RC.EPISODES_PER_REPLICATE,
            "total_episodes": len(ARMS) * len(RC.REPLICATES) * RC.EPISODES_PER_REPLICATE,
        },
        "diagnostics": {
            "seeds": list(RD.SEEDS),
            "num_start_states": RD.NUM_START_STATES,
            "ranking_samples_per_seed": RD.RANKING_SAMPLES,
            "calibration_band": list(RD.CALIBRATION_BAND),
            "slack_nats": RD.SLACK_NATS,
        },
        "acceptance_constants_source": ACCEPTANCE_SOURCE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return record


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
            "goal_reaching_rate": frame.goal_reaching_rate.mean(),
            "ended_by_goal": frame.ended_by_goal.mean(),
            "ended_by_failure": frame.ended_by_failure.mean(),
            "ended_by_timeout": frame.ended_by_timeout.mean(),
            "average_episode_length": frame.average_episode_length.mean(),
            "out_of_grid_rate": frame.out_of_grid_rate.mean(),
            "obstacle_hit_rate": frame.obstacle_hit_rate.mean(),
            "avg_obstacle_hit_counter": frame.avg_obstacle_hit_counter.mean(),
            "avg_high_variance_states_counter": frame.avg_high_variance_states_counter.mean(),
            "final_distance_to_goal": frame.final_distance_to_goal.mean(),
            "steps_in_dark": frame.steps_in_dark.mean(),
            "localization_error_at_goal": frame.localization_error_at_goal.mean(),
            "median_action_s": frame.mean_action_seconds.median(),
            "p90_action_s": frame.mean_action_seconds.quantile(0.9),
            "mean_return": frame.discounted_return.mean(),
            "se_over_episodes": frame.discounted_return.std(ddof=1) / np.sqrt(len(frame)),
            "se_over_seeds": seed_means.std(ddof=1) / np.sqrt(len(seed_means)),
            "std_over_episodes": frame.discounted_return.std(ddof=1),
            "per_seed_means": [round(float(v), 3) for v in seed_means],
            "episodes": len(frame),
            "seeds": frame.replicate.nunique(),
            "episodes_per_seed": int(frame.groupby("replicate").size().min()),
            "sims_per_decision": frame.mean_simulations_per_decision.mean(),
        })
    return pd.DataFrame(rows).set_index("arm")


def control_gates(frames: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    incumbent = frames[INCUMBENT]
    inc_means = incumbent.groupby("replicate")["discounted_return"].mean()
    inc_goal = incumbent.groupby("replicate")["goal_reaching_rate"].mean()
    margin = PARITY_MARGIN_FRACTION * abs(inc_means.mean())
    verdicts: Dict[str, Dict[str, Any]] = {}
    for arm, frame in frames.items():
        means = frame.groupby("replicate")["discounted_return"].mean()
        goal = frame.groupby("replicate")["goal_reaching_rate"].mean()
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
            "control_task_metric": {
                "candidate_task_completion_rate": float(goal.mean()),
                "truth_task_completion_rate": float(inc_goal.mean()),
                "pass": goal_not_lower,
            },
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
            "control": {
                **control[arm]["control"],
                "return": control[arm]["control_return"],
                "task_completion_rate": control[arm]["control_task_metric"],
            },
        }
        failed = [g for g in ("contract", "calibration", "likelihood", "observation", "ranking", "evidence", "control") if not gates[arm][g]["pass"]]
        gates[arm]["verdict"] = "pass" if not failed else "fail"
        gates[arm]["failed_gates"] = failed
    out = {"acceptance_constants_source": ACCEPTANCE_SOURCE, "gates": gates,
           "closed_loop": json.loads(table.reset_index().to_json(orient="records"))}
    (OUT / "gates.json").write_text(json.dumps(out, indent=2))
    table.to_csv(OUT / "closed_loop_table.csv")
    write_provenance(OUT / "provenance.json")
    print(table.to_string())
    for arm in ARMS:
        c = control[arm]
        print(arm, gates[arm]["verdict"], gates[arm]["failed_gates"],
              "p_parity", round(c["control_return"]["p_parity"], 4), "p_worse", round(c["control_return"]["p_candidate_worse_than_incumbent"], 4),
              "goal", c["control_task_metric"]["candidate_task_completion_rate"])
    return out


if __name__ == "__main__":
    main()
