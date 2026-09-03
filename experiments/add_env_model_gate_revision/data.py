# SPDX-License-Identifier: MIT

"""Held-random-preset rollouts of the true LightDark, split held-out by episode."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from POMDPPlanners.environments.light_dark_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.training.model_learning import TransitionBatch, collect_random_preset_episode

from experiments.add_env_model_gate_revision.world import PRESETS


@dataclass(frozen=True)
class ObservationBatch:
    """Next positions and the observations the world emitted at them."""

    next_states: np.ndarray
    observations: np.ndarray

    def __len__(self) -> int:
        return int(self.next_states.shape[0])


def collect_episodes(
    world: Any,
    num_episodes: int,
    num_steps: int,
    hold_steps: int,
    random_start_fraction: float,
    seed: int,
) -> List[Dict[str, np.ndarray]]:
    """Roll out held random presets; half the starts are the task start, the rest uniform in-grid.

    The task start alone covers only a band of the grid, and the planner's
    beliefs go everywhere the tree searches, so part of the data starts from
    uniform positions. Boundary rows are dropped by the collector.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    _native.set_seed(seed)
    episodes: List[Dict[str, np.ndarray]] = []
    for index in range(num_episodes):
        initial = None
        if rng.random() < random_start_fraction:
            pos = rng.uniform(0.0, float(world.grid_size), size=2)
            initial = world._pad_terminal_slot(pos)  # pylint: disable=protected-access
        states, actions, next_states = collect_random_preset_episode(
            world, PRESETS, num_steps=num_steps, rng=rng, hold_steps=hold_steps, initial_state=initial
        )
        if len(states) == 0:
            continue
        observations = np.stack(
            [np.asarray(world.sample_observation(ns, a), dtype=float) for ns, a in zip(next_states, actions)]
        )
        episodes.append(
            {
                "index": np.array(index),
                "states": states[:, :2],
                "actions": actions,
                "next_states": next_states[:, :2],
                "observations": observations,
            }
        )
    return episodes


def split_episodes(
    episodes: List[Dict[str, np.ndarray]], holdout_fraction: float, seed: int
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Episode-wise split: a row-wise one would put a row's neighbours in the training set."""
    rng = np.random.default_rng(seed)
    flags = rng.random(len(episodes)) < holdout_fraction
    train = [ep for ep, held in zip(episodes, flags) if not held]
    held = [ep for ep, held in zip(episodes, flags) if held]
    return train, held


def transition_batch(episodes: List[Dict[str, np.ndarray]]) -> TransitionBatch:
    return TransitionBatch(
        np.concatenate([ep["states"] for ep in episodes]),
        np.concatenate([ep["actions"] for ep in episodes]),
        np.concatenate([ep["next_states"] for ep in episodes]),
    )


def observation_batch(episodes: List[Dict[str, np.ndarray]]) -> ObservationBatch:
    return ObservationBatch(
        np.concatenate([ep["next_states"] for ep in episodes]),
        np.concatenate([ep["observations"] for ep in episodes]),
    )
