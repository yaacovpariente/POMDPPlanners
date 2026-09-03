# SPDX-License-Identifier: MIT
"""Run the revised LightDark diagnostic gates on the reused held-out split.

The calibration ratio divides each candidate's horizon drift by the world's
drift against itself at the same seed. ``_seed_all`` resets all three noise
sources before the floor and before every candidate, so the seed fixes the
start states, the action sequence and both draw streams, and a rerun on the
same seeds reproduces the ratios. It does not make the floor a *paired*
measurement: the floor rolls the world twice, so per step it takes one native
draw for the world and sixteen more for the world-as-model, while a candidate
takes only the world's one and draws its own from ``np.random``. The world
trajectories therefore differ between the floor and the candidate even at the
same seed, which is why the recorded key is
``mean_ratio_to_seed_matched_floor`` and not ``..._to_paired_floor``.
"""

import json
import time
from pathlib import Path

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp import _native as _ld_native  # pylint: disable=no-name-in-module
from POMDPPlanners.training.model_learning import TransitionBatch, evaluate_model, horizon_drift_ratio
from POMDPPlanners.training.model_learning.diagnostics import preset_ranking_agreement

from experiments.add_env_model_gate_revision import world as W
from experiments.add_env_model_gate_revision.models import (
    GaussianMLPObservation,
    GaussianMLPTransition,
    bad_transition,
    truth_transition,
)

OUT = Path("results/add-env-model-gate-revision")
LEARNED = OUT / "learned_model"
SEEDS = tuple(range(8))
NUM_START_STATES = 64
RANKING_SAMPLES = 256
SLACK_NATS = 0.1
CALIBRATION_BAND = (1.0 / 1.5, 1.5)


def _seed_all(seed):
    """Reset every noise source the drift measurement draws from to ``seed``.

    ``horizon_drift_ratio`` takes a ``Generator``, but that generator only picks
    the action sequence and which of the 16 model draws to advance on. The draws
    themselves come from two other places: the candidate transitions all sample
    through global ``np.random`` (``GaussianMLPTransition.sample_next_state``, and
    ``CovarianceParameterizedMultivariateNormal.sample`` under the truth and the
    doubled-noise models), and the world side runs on the LightDark C++ kernel,
    whose only entry point is ``_native.set_seed`` -- the same call the native
    equivalence tests use. Seeding one of the three is not enough for a rerun to
    reproduce the ratios, so all three are set here, to the same value, and this
    is called immediately before the floor draw and before each candidate's.

    Returns:
        A fresh ``Generator`` on ``seed``; a ``Generator`` cannot be reseeded in
        place, so the caller uses the one returned here.
    """
    np.random.seed(seed)
    _ld_native.set_seed(seed)
    return np.random.default_rng(seed)


def _starts(holdout, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(holdout), size=NUM_START_STATES, replace=False)
    return [holdout.states[index] for index in chosen]


def _floor(view, holdout, seed):
    starts = _starts(holdout, seed)
    rng = _seed_all(seed)
    # Consume the same start-selection draw as evaluate_model before drift, so
    # the generator enters the drift loop in the same state on both sides.
    rng.choice(len(holdout), size=NUM_START_STATES, replace=False)
    return float(horizon_drift_ratio(view, view, starts, W.PRESETS, W.PLANNING_DEPTH, rng))


def _observation_scores(world, next_states, observations):
    truth = W.true_observation(world)
    learned = GaussianMLPObservation.load(LEARNED / "observation.npz")
    near = truth.near(next_states)
    result = {"holdout_rows": int(len(next_states)), "near_beacon_fraction": float(near.mean())}
    for name, model in (("truth", truth), ("learned", learned), ("bad", truth)):
        ll = np.array([model.log_probability(state, obs[None, :])[0] for state, obs in zip(next_states, observations)])
        std = model.std(next_states)
        result[name] = {
            "held_out_log_likelihood": float(ll.mean()),
            "mean_std_near_beacon": float(std[near].mean()),
            "mean_std_far_from_beacon": float(std[~near].mean()),
        }
    return result


def main():
    started = time.time()
    world = W.make_world()
    view = W.WorldPositionView(world)
    with np.load(LEARNED / "rollouts.npz") as payload:
        holdout = TransitionBatch(payload["holdout_states"], payload["holdout_actions"], payload["holdout_next_states"])
        holdout_observations = payload["holdout_observations"]
    models = {
        "truth": truth_transition(world.state_transition_cov_matrix),
        "learned": GaussianMLPTransition.load(LEARNED / "transition.npz"),
        "bad": bad_transition(world.state_transition_cov_matrix, std_factor=2.0),
    }
    floors = {seed: _floor(view, holdout, seed) for seed in SEEDS}
    evaluated = {name: {} for name in models}
    rankings = {name: [] for name in models}
    for seed in SEEDS:
        starts = _starts(holdout, seed)
        for name, model in models.items():
            # Same three sources, same value, immediately before this candidate's
            # drift draw. ``evaluate_model`` builds its own Generator on ``seed``
            # and its held-out likelihood sweep is a deterministic density
            # evaluation, so the drift loop starts where the floor's did.
            _seed_all(seed)
            evaluated[name][seed] = evaluate_model(
                model, view, holdout, W.PRESETS, horizon=W.PLANNING_DEPTH,
                reward_model=view, num_start_states=NUM_START_STATES, seed=seed,
            )
            # The ranking gate needs the same reset: evaluate_model has just
            # advanced the native stream by a model-dependent number of draws,
            # so seeding only np.random here would leave the score dependent on
            # which arm ran before it.
            _seed_all(seed)
            rankings[name].append(float(preset_ranking_agreement(
                model, view, view, starts, W.PRESETS, num_samples=RANKING_SAMPLES,
            )))
    observations = _observation_scores(world, holdout.next_states, holdout_observations)
    transition_ll = {name: float(evaluated[name][0]["held_out_log_likelihood"]) for name in models}
    truth_rank_mean = float(np.mean(rankings["truth"]))
    truth_rank_std = float(np.std(rankings["truth"], ddof=1))
    rank_threshold = truth_rank_mean - truth_rank_std
    verdicts = {}
    for name in models:
        drifts = [float(evaluated[name][seed]["horizon_drift_ratio"]) for seed in SEEDS]
        ratios = [drift / floors[seed] for seed, drift in zip(SEEDS, drifts)]
        mean_ratio = float(np.mean(ratios))
        obs_ll = observations[name]["held_out_log_likelihood"]
        rank_mean = float(np.mean(rankings[name]))
        verdicts[name] = {
            "calibration": {"drift_by_seed": drifts, "floor_by_seed": [floors[s] for s in SEEDS],
                            "ratio_by_seed": ratios, "mean_ratio_to_seed_matched_floor": mean_ratio,
                            "band": list(CALIBRATION_BAND),
                            "pass": bool(CALIBRATION_BAND[0] <= mean_ratio <= CALIBRATION_BAND[1])},
            "likelihood": {"value": transition_ll[name], "incumbent": transition_ll["truth"],
                           "threshold": transition_ll["truth"] - SLACK_NATS,
                           "pass": bool(np.isfinite(transition_ll[name]) and transition_ll[name] >= transition_ll["truth"] - SLACK_NATS)},
            "observation": {"value": obs_ll, "incumbent": observations["truth"]["held_out_log_likelihood"],
                            "threshold": observations["truth"]["held_out_log_likelihood"] - SLACK_NATS,
                            "pass": bool(np.isfinite(obs_ll) and obs_ll >= observations["truth"]["held_out_log_likelihood"] - SLACK_NATS)},
            "ranking": {"values": rankings[name], "mean": rank_mean, "truth_mean": truth_rank_mean,
                        "truth_seed_std": truth_rank_std, "threshold": rank_threshold,
                        "pass": bool(rank_mean >= rank_threshold)},
        }
    report = {
        "settings": {"seeds": list(SEEDS), "holdout_rows": len(holdout), "horizon": W.PLANNING_DEPTH,
                     "num_start_states": NUM_START_STATES, "ranking_samples_per_seed": RANKING_SAMPLES},
        "drift_floor_by_seed": floors, "seed_metrics": evaluated,
        "observation_models": observations, "verdicts": verdicts,
        "wall_seconds": time.time() - started,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
