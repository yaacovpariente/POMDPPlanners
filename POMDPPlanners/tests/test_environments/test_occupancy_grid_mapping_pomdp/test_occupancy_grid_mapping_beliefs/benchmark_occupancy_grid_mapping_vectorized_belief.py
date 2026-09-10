# SPDX-License-Identifier: MIT

"""Benchmark: PFT-DPW on occupancy-grid mapping with the scalar vs the vectorized filter.

Two comparisons, both run through ``LocalSimulationsAPI`` so the cache, the
per-episode records and the seeding are the production ones:

* **Fixed work.** PFT-DPW with a fixed simulation count per decision. The
  decision time and the belief-update time are read from each episode's
  history; their ratio is the speed-up.
* **Fixed time.** PFT-DPW with the QA time budget per decision. The number of
  simulations a decision completes (``root_visit_count``) is the throughput;
  return and completion say whether the extra search bought anything.

Episodes are seeded from the environment name, the policy name and the
episode index, so both filters face the same worlds. The search itself still
draws its own randomness, so trajectories differ.

Run manually, for example::

    python -m POMDPPlanners.tests.test_environments.test_occupancy_grid_mapping_pomdp\
.test_occupancy_grid_mapping_beliefs.benchmark_occupancy_grid_mapping_vectorized_belief \
        --episodes 5 --n-simulations 200 --time-out 1

Results go under ``results/`` and are never committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.history import history_to_discounted_return_value
from POMDPPlanners.environments.occupancy_grid_mapping_pomdp import (
    OccupancyGridMappingPOMDP,
)
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)
from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
    occupancy_grid_mapping_pinned_kwargs,
)
from POMDPPlanners.tests.test_utils.env_qa_planner_configs import (
    occupancy_grid_mapping_qa_belief_particles,
    occupancy_grid_mapping_qa_pft_dpw_kwargs,
)
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief
from POMDPPlanners.utils.tree_statistics import TreeMetrics

BELIEF_TYPES = {
    "scalar": BeliefType.PARTICLE,
    "vectorized": BeliefType.VECTORIZED_PARTICLE,
}


def _planner(env: OccupancyGridMappingPOMDP, **overrides) -> PFT_DPW:
    kwargs = occupancy_grid_mapping_qa_pft_dpw_kwargs(**overrides)
    return PFT_DPW(
        environment=env,
        discount_factor=env.discount_factor,
        name="PFT_DPW",
        action_sampler=DiscreteActionSampler(env.get_actions()),
        **kwargs,
    )


def _summarize(histories) -> Dict[str, float]:
    """Per-episode means of the timing, throughput and outcome fields."""
    visits: List[float] = []
    for history in histories:
        for run_data in history.policy_run_data:
            for variable in run_data.info_variables:
                if variable.name == TreeMetrics.ROOT_VISIT_COUNT.value:
                    visits.append(float(variable.value))
    return {
        "decision_s": float(np.mean([h.average_action_time for h in histories])),
        "belief_update_s": float(np.mean([h.average_belief_update_time for h in histories])),
        "reward_s": float(np.mean([h.average_reward_time for h in histories])),
        "simulations_per_decision": float(np.mean(visits)) if visits else float("nan"),
        "steps": float(np.mean([h.actual_num_steps for h in histories])),
        "discounted_return": float(
            np.mean([history_to_discounted_return_value(h) for h in histories])
        ),
    }


def _run(
    label: str,
    belief_type: BeliefType,
    episodes: int,
    n_jobs: int,
    results_dir: Path,
    **planner_overrides,
) -> Dict[str, float]:
    env = OccupancyGridMappingPOMDP(discount_factor=0.95, **occupancy_grid_mapping_pinned_kwargs())
    np.random.seed(0)
    belief = create_environment_belief(
        env, belief_type, n_particles=occupancy_grid_mapping_qa_belief_particles()
    )
    params = EnvironmentRunParams(
        environment=env,
        belief=belief,
        policies=[_planner(env, **planner_overrides)],
        num_episodes=episodes,
        num_steps=env.max_steps,
    )
    results, _ = LocalSimulationsAPI().run_multiple_environments_and_policies(
        environment_run_params=[params],
        alpha=0.05,
        confidence_interval_level=0.95,
        experiment_name=f"occupancy_belief_benchmark_{label}",
        n_jobs=n_jobs,
        cache_dir_path=results_dir / label,
    )
    histories = [h for policies in results.values() for hs in policies.values() for h in hs]
    return _summarize(histories)


def _print_table(title: str, rows: Dict[str, Dict[str, float]]) -> None:
    keys = list(next(iter(rows.values())).keys())
    print(f"\n--- {title} ---")
    print(f"{'belief':>12} | " + " | ".join(f"{k:>24}" for k in keys))
    for label, values in rows.items():
        print(f"{label:>12} | " + " | ".join(f"{values[k]:>24.4f}" for k in keys))
    if {"scalar", "vectorized"} <= rows.keys():
        scalar, vectorized = rows["scalar"], rows["vectorized"]
        print(
            f"speed-up: decision x{scalar['decision_s'] / vectorized['decision_s']:.2f}, "
            f"belief update x{scalar['belief_update_s'] / vectorized['belief_update_s']:.2f}, "
            f"simulations per decision x"
            f"{vectorized['simulations_per_decision'] / scalar['simulations_per_decision']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--n-simulations", type=int, default=200)
    parser.add_argument("--time-out", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results") / "occupancy_belief_benchmark"
    )
    parser.add_argument(
        "--skip-fixed-time", action="store_true", help="Run only the fixed-work comparison."
    )
    args = parser.parse_args()

    fixed_work = {
        label: _run(
            f"fixed_work_{label}",
            belief_type,
            args.episodes,
            args.n_jobs,
            args.results_dir,
            n_simulations=args.n_simulations,
            time_out_in_seconds=None,
        )
        for label, belief_type in BELIEF_TYPES.items()
    }
    _print_table(f"fixed work: {args.n_simulations} simulations per decision", fixed_work)

    if not args.skip_fixed_time:
        fixed_time = {
            label: _run(
                f"fixed_time_{label}",
                belief_type,
                args.episodes,
                args.n_jobs,
                args.results_dir,
                time_out_in_seconds=args.time_out,
            )
            for label, belief_type in BELIEF_TYPES.items()
        }
        _print_table(f"fixed time: {args.time_out}s per decision", fixed_time)


if __name__ == "__main__":
    main()
