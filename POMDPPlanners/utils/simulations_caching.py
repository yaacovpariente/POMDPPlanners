# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Optional

import diskcache

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.policy import Policy
from POMDPPlanners.core.simulation import History
from POMDPPlanners.utils.logger import get_logger

logger = get_logger(__name__)


def get_cache_key(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    general_config: dict = {},
) -> str:
    # String key for diskcache; include env/policy/belief IDs and sorted config
    return "|".join(
        [
            environment.config_id,
            policy.config_id,
            initial_belief.config_id,
            str(sorted(general_config.items())),
        ]
    )


def get_cache_dir_path(
    cache_dir_path: Path,
) -> Path:
    return cache_dir_path / "simulations_cache"


def cache_episode_simulation_results(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    results: List[History],
    cache_dir_path: Path,
    general_config: dict = {},
) -> None:
    cache_dir_path = get_cache_dir_path(cache_dir_path)
    cache = diskcache.Cache(cache_dir_path)

    key = get_cache_key(environment, policy, initial_belief, general_config)
    if key in cache:
        logger.info("Simulation results for %s already cached", key)
    else:
        cache[key] = results


def load_episode_simulation_results(
    environment: Environment,
    policy: Policy,
    initial_belief: Belief,
    cache_dir_path: Path,
    general_config: dict = {},
) -> List[History]:
    cache_dir_path = get_cache_dir_path(cache_dir_path)
    cache = diskcache.Cache(cache_dir_path)

    key = get_cache_key(environment, policy, initial_belief, general_config)

    if key in cache:
        return cache[key]  # type: ignore
    logger.debug("Simulation results for %s not found in cache", key)
    return []
