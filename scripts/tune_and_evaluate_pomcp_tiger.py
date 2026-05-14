"""Simple parameter-tuning + evaluation demo for POMCP on TigerPOMDP.

Two phases, each a separate :class:`POMDPSimulator` instance — and therefore
two distinct runs in the progress DB / Slack channel:

1. **Tuning phase**: compare POMCP with several values of ``n_simulations``
   side-by-side on a modest number of episodes; pick the best by mean
   discounted return.
2. **Evaluation phase**: re-run only the winning configuration on a larger
   number of episodes to get a tighter estimate of its performance.

Run with::

    source .venv/bin/activate
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."  # optional
    python scripts/tune_and_evaluate_pomcp_tiger.py

If ``SLACK_WEBHOOK_URL`` is set, you will see four messages in the channel
(two ``run_started`` / ``run_finished`` pairs). If it is unset, the script
runs silently w.r.t. Slack but still writes the run history into the
progress DB at ``~/.cache/POMDPPlanners/progress.db``.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.history import history_to_discounted_return_value
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)

CACHE_DIR = Path("./cache/pomcp_tiger_demo")
TUNING_EPISODES = 30
EVALUATION_EPISODES = 100
NUM_STEPS = 5
N_PARTICLES = 50
DISCOUNT_FACTOR = 0.95
EXPLORATION_CONSTANT = 1.0
ROLLOUT_DEPTH = 5
CANDIDATE_N_SIMULATIONS = [5, 20, 50, 100]


def main() -> None:
    """Run the two-phase tuning + evaluation experiment."""
    tiger = TigerPOMDP(discount_factor=DISCOUNT_FACTOR)
    initial_belief = get_initial_belief(tiger, n_particles=N_PARTICLES)
    api = LocalSimulationsAPI(cache_dir_path=CACHE_DIR)

    best_n_sims = _run_tuning_phase(api, tiger, initial_belief)
    _run_evaluation_phase(api, tiger, initial_belief, best_n_sims)


def _run_tuning_phase(
    api: LocalSimulationsAPI,
    tiger: TigerPOMDP,
    initial_belief,
) -> int:
    print(
        f"[tuning] Comparing POMCP with n_simulations in "
        f"{CANDIDATE_N_SIMULATIONS} over {TUNING_EPISODES} episodes each"
    )
    candidate_policies = [
        POMCP(
            environment=tiger,
            discount_factor=DISCOUNT_FACTOR,
            depth=ROLLOUT_DEPTH,
            exploration_constant=EXPLORATION_CONSTANT,
            n_simulations=n_sims,
            name=f"POMCP_n{n_sims}",
        )
        for n_sims in CANDIDATE_N_SIMULATIONS
    ]
    env_params = [
        EnvironmentRunParams(
            environment=tiger,
            belief=initial_belief,
            policies=candidate_policies,
            num_episodes=TUNING_EPISODES,
            num_steps=NUM_STEPS,
        )
    ]

    results, _ = api.run_multiple_environments_and_policies(
        environment_run_params=env_params,
        alpha=0.1,
        confidence_interval_level=0.95,
        experiment_name="tune_pomcp_tiger",
        n_jobs=1,
        cache_dir_path=CACHE_DIR / "tuning",
    )

    return _pick_best_n_simulations(results, tiger.name)


def _pick_best_n_simulations(results: dict, environment_name: str) -> int:
    print("[tuning] Mean discounted returns:")
    mean_returns: dict[int, float] = {}
    for policy_name, histories in results[environment_name].items():
        returns = [history_to_discounted_return_value(h) for h in histories]
        mean_returns[_parse_n_sims_from_name(policy_name)] = statistics.fmean(returns)
        print(f"  {policy_name}: {statistics.fmean(returns):.3f}")

    best_n_sims = max(mean_returns, key=lambda k: mean_returns[k])
    print(f"[tuning] Winner: n_simulations={best_n_sims}")
    return best_n_sims


def _parse_n_sims_from_name(policy_name: str) -> int:
    # Policy names are constructed as f"POMCP_n{n_sims}".
    return int(policy_name.rsplit("_n", 1)[-1])


def _run_evaluation_phase(
    api: LocalSimulationsAPI,
    tiger: TigerPOMDP,
    initial_belief,
    best_n_sims: int,
) -> None:
    print(
        f"[eval] Re-running winner (n_simulations={best_n_sims}) "
        f"on {EVALUATION_EPISODES} episodes"
    )
    winning_policy = POMCP(
        environment=tiger,
        discount_factor=DISCOUNT_FACTOR,
        depth=ROLLOUT_DEPTH,
        exploration_constant=EXPLORATION_CONSTANT,
        n_simulations=best_n_sims,
        name=f"POMCP_n{best_n_sims}_eval",
    )
    env_params = [
        EnvironmentRunParams(
            environment=tiger,
            belief=initial_belief,
            policies=[winning_policy],
            num_episodes=EVALUATION_EPISODES,
            num_steps=NUM_STEPS,
        )
    ]

    results, _ = api.run_multiple_environments_and_policies(
        environment_run_params=env_params,
        alpha=0.1,
        confidence_interval_level=0.95,
        experiment_name="evaluate_pomcp_tiger",
        n_jobs=1,
        cache_dir_path=CACHE_DIR / "evaluation",
    )

    histories = results[tiger.name][winning_policy.name]
    returns = [history_to_discounted_return_value(h) for h in histories]
    print(
        f"[eval] n_simulations={best_n_sims}: "
        f"mean={statistics.fmean(returns):.3f}, "
        f"stdev={statistics.stdev(returns):.3f} "
        f"over {len(returns)} episodes"
    )


if __name__ == "__main__":
    main()
