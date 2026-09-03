# SPDX-License-Identifier: MIT

"""Stage-4 gates: calibration, likelihood and ranking, for the three candidates on one held-out batch.

The floor for the drift gate is the world rolled against itself at the
planning horizon, from the same start states and action sequences the gate
uses. The observation models are scored beside the gates, because no gate
looks at them.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from POMDPPlanners.training.model_learning import TransitionBatch, evaluate_model, horizon_drift_ratio
from POMDPPlanners.training.model_learning.diagnostics import held_out_log_likelihood

from experiments.add_env_model_gate_revision import world as W
from experiments.add_env_model_gate_revision.models import (
    GaussianMLPObservation,
    GaussianMLPTransition,
    bad_transition,
    truth_transition,
)

OUT = Path("results/add-env-model-gate-revision")
LEARNED = OUT / "learned_model"
NUM_START_STATES = 64
SEED = 0
LIKELIHOOD_SLACK_NATS = 0.1
MAX_DRIFT_RATIO = 1.5


def _drift_floor(view: W.WorldPositionView, holdout: TransitionBatch, seed: int) -> float:
    """Exactly evaluate_model's start-state draw and action sequences, with the world as the model."""
    rng = np.random.default_rng(seed)
    count = min(NUM_START_STATES, len(holdout))
    chosen = rng.choice(len(holdout), size=count, replace=False)
    starts = [holdout.states[index] for index in chosen]
    return float(horizon_drift_ratio(view, view, starts, W.PRESETS, W.PLANNING_DEPTH, rng))


def _observation_scores(world: Any, holdout_next: np.ndarray, holdout_obs: np.ndarray) -> Dict[str, Any]:
    truth = W.true_observation(world)
    learned = GaussianMLPObservation.load(LEARNED / "observation.npz")
    near = truth.near(holdout_next)
    scores: Dict[str, Any] = {"holdout_rows": int(len(holdout_next)), "near_beacon_fraction": float(near.mean())}
    for name, model in (("truth", truth), ("learned", learned)):
        ll = np.array([float(model.log_probability(ns, o[None, :])[0]) for ns, o in zip(holdout_next, holdout_obs)])
        std = model.std(holdout_next)
        scores[name] = {
            "held_out_log_likelihood": float(ll.mean()),
            "mean_std_near_beacon": float(std[near].mean()),
            "mean_std_far_from_beacon": float(std[~near].mean()),
        }
    return scores


def main() -> Dict[str, Any]:
    world = W.make_world()
    view = W.WorldPositionView(world)
    with np.load(LEARNED / "rollouts.npz") as payload:
        holdout = TransitionBatch(payload["holdout_states"], payload["holdout_actions"], payload["holdout_next_states"])
        holdout_obs = payload["holdout_observations"]
    candidates = {
        "truth": truth_transition(world.state_transition_cov_matrix),
        "learned": GaussianMLPTransition.load(LEARNED / "transition.npz"),
        "bad": bad_transition(world.state_transition_cov_matrix, std_factor=2.0),
    }
    floors = {seed: _drift_floor(view, holdout, seed) for seed in (SEED, SEED + 1, SEED + 2)}
    floor = floors[SEED]
    metrics = {
        name: evaluate_model(
            model, view, holdout, W.PRESETS, horizon=W.PLANNING_DEPTH, reward_model=view,
            num_start_states=NUM_START_STATES, seed=SEED,
        )
        for name, model in candidates.items()
    }
    incumbent = metrics["truth"]
    verdicts: Dict[str, Dict[str, Any]] = {}
    for name, m in metrics.items():
        ll = m["held_out_log_likelihood"]
        verdicts[name] = {
            "calibration": {
                "value": m["horizon_drift_ratio"], "floor": floor, "ratio_to_floor": m["horizon_drift_ratio"] / floor,
                "threshold": MAX_DRIFT_RATIO, "pass": bool(np.isfinite(m["horizon_drift_ratio"]) and m["horizon_drift_ratio"] <= MAX_DRIFT_RATIO * floor),
            },
            "likelihood": {
                "value": ll, "incumbent": incumbent["held_out_log_likelihood"],
                "threshold": incumbent["held_out_log_likelihood"] - LIKELIHOOD_SLACK_NATS,
                "pass": bool(np.isfinite(ll) and ll >= incumbent["held_out_log_likelihood"] - LIKELIHOOD_SLACK_NATS),
            },
            "ranking": {
                "value": m["preset_ranking_agreement"], "threshold": incumbent["preset_ranking_agreement"],
                "pass": bool(m["preset_ranking_agreement"] >= incumbent["preset_ranking_agreement"]),
            },
        }
    report = {
        "settings": {"horizon": W.PLANNING_DEPTH, "num_start_states": NUM_START_STATES, "seed": SEED,
                     "presets": W.PRESETS.tolist(), "holdout_rows": len(holdout)},
        "drift_floor_by_seed": floors,
        "metrics": metrics,
        "verdicts": verdicts,
        "observation_models": _observation_scores(world, holdout.next_states, holdout_obs),
    }
    (OUT / "stage4_diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
