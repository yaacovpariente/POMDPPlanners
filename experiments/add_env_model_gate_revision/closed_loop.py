# SPDX-License-Identifier: MIT

"""Closed-loop runs through ``LocalSimulationsAPI``: the world executes, the candidate plans.

One API call per arm. The policy is named the same in every arm so the harness
derives the same per-episode seed for every arm (the seed is a hash of the
world name, the policy name and the episode index); the cache separates arms
through the policy's ``config_id``, which carries the model's label and
fingerprints. Three replicates are three copies of the world that differ only
by name, so each replicate gets its own seeds and all arms share them.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.policy import Policy, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.history import history_to_discounted_return_value
from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.utils.action_samplers import UnitCircleActionSampler

from experiments.add_env_model_gate_revision import world as W

POLICY_NAME = "PFT_DPW"
#: MCTS wall-clock budget per decision, seconds. The planner-calibration band is
#: 1-2 s; the planner takes an integer timeout and hits it exactly.
N_SIMULATIONS = 2849
NUM_PARTICLES = 100
NUM_STEPS = 40
EXPLORATION_CONSTANT = 10.0


class RandomPolicy(Policy):
    """Uniform actions on the unit disc -- the env-qa baseline."""

    def __init__(self, environment: Any, name: str = "Random") -> None:
        super().__init__(environment=environment, discount_factor=environment.discount_factor, name=name)
        self.max_action_magnitude = 1.0

    def action(self, belief: Any) -> Any:
        del belief
        return [UnitCircleActionSampler(self.max_action_magnitude).sample()], PolicyRunData(info_variables=[])

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(action_space=SpaceType.CONTINUOUS, observation_space=SpaceType.CONTINUOUS)

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return []


def make_planner(model: Any, exploration_constant: float = EXPLORATION_CONSTANT, name: str = POLICY_NAME) -> PFT_DPW:
    return PFT_DPW(
        environment=model,
        discount_factor=W.DISCOUNT,
        depth=W.PLANNING_DEPTH,
        name=name,
        action_sampler=UnitCircleActionSampler(max_action_magnitude=1.0),
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=exploration_constant,
        n_simulations=N_SIMULATIONS,
        min_visit_count_per_action=1,
    )


def run_batch(
    label: str,
    policies_for_model: Sequence[Any],
    replicate_seeds: Sequence[int],
    num_episodes: int,
    cache_dir: Path,
    n_jobs: int,
    num_steps: int = NUM_STEPS,
) -> Dict[str, Any]:
    """Run every policy on every replicate world; return the raw histories and the stats frame."""
    params = []
    for seed in replicate_seeds:
        world = W.make_world(name=f"ContinuousLightDarkPOMDP-rep{seed}")
        params.append(
            EnvironmentRunParams(
                environment=world,
                belief=get_initial_belief(policies_for_model[0].environment, n_particles=NUM_PARTICLES, resampling=True),
                policies=list(policies_for_model),
                num_episodes=num_episodes,
                num_steps=num_steps,
            )
        )
    api = LocalSimulationsAPI(cache_dir_path=cache_dir)
    results, stats = api.run_multiple_environments_and_policies(
        environment_run_params=params,
        alpha=0.05,
        confidence_interval_level=0.95,
        experiment_name=f"add-env-model-gate-revision-{label}",
        n_jobs=n_jobs,
        cache_dir_path=cache_dir,
        clear_cache_on_start=False,
    )
    return {"results": results, "stats": stats, "params": params}


def episode_records(results: Dict[str, Dict[str, list]], world: Any, arm: str) -> List[Dict[str, Any]]:
    """One row per episode, from the world's own history: the numbers every table is built from."""
    rows: List[Dict[str, Any]] = []
    for env_name, by_policy in results.items():
        replicate = int(env_name.rsplit("rep", 1)[-1])
        for policy_name, histories in by_policy.items():
            for episode_index, history in enumerate(histories):
                final_state = np.asarray(history.history[-1].state, dtype=float)
                positions = np.array([np.asarray(step.state, dtype=float)[:2] for step in history.history])
                goal = float(np.linalg.norm(final_state[:2] - world.goal_state) <= world.goal_state_radius)
                hazard = float(final_state.shape[0] > 2 and final_state[2] > 0.5)
                out_of_grid = float(np.any((positions < 0.0) | (positions > world.grid_size)))
                rewards = [step.reward for step in history.history if step.reward is not None]
                sims = [
                    float(var.value)
                    for run in history.policy_run_data
                    for var in getattr(run, "info_variables", [])
                    if getattr(var, "name", "") == "root_visit_count"
                ]
                rows.append(
                    {
                        "arm": arm,
                        "policy": policy_name,
                        "replicate": replicate,
                        "episode": episode_index,
                        "discounted_return": history_to_discounted_return_value(history),
                        "undiscounted_return": float(np.sum(rewards)),
                        "steps": int(history.actual_num_steps),
                        "terminated": bool(history.reach_terminal_state),
                        "goal_reached": goal,
                        "hazard_hit": hazard,
                        "out_of_grid": out_of_grid,
                        "mean_action_seconds": float(history.average_action_time),
                        "mean_belief_update_seconds": float(history.average_belief_update_time),
                        "mean_simulations_per_decision": float(np.mean(sims)) if sims else float("nan"),
                    }
                )
    return rows


def write_records(rows: List[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_provenance(path: Path, arm: str, model: Any, world: Any, planner: Any) -> None:
    """The line realistic-simulations asks for: which model each policy planned with."""
    line = {
        "arm": arm,
        "policy": planner.name,
        "policy_config_id": planner.config_id,
        "model_class": type(model).__name__,
        "model_label": getattr(model, "model_label", None),
        "model_config_id": model.config_id,
        "transition_class": type(getattr(model, "transition", None)).__name__,
        "transition_fingerprint": getattr(model, "transition_fingerprint", None),
        "observation_class": type(getattr(model, "observation", None)).__name__,
        "observation_fingerprint": getattr(model, "observation_fingerprint", None),
        "world_class": type(world).__name__,
        "world_config_id": world.config_id,
        "world_scenario": "pinned kwargs from POMDPPlanners.tests.test_utils.env_pinned_kwargs.continuous_light_dark_pinned_kwargs",
        "discount_check": "EpisodeRunner asserts equal discount factors when world and model differ",
        "planner": {
            "depth": planner.depth, "k_a": planner.k_a, "alpha_a": planner.alpha_a, "k_o": planner.k_o,
            "alpha_o": planner.alpha_o, "exploration_constant": planner.exploration_constant,
            "n_simulations": N_SIMULATIONS, "num_particles": NUM_PARTICLES, "num_steps": NUM_STEPS,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")
