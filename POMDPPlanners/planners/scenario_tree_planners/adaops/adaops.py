# SPDX-License-Identifier: MIT

"""Adaptive Online Packing-guided Search (AdaOPS).

This implementation follows Wu et al., NeurIPS 2021, Algorithms 2, 5 and 6.
Beliefs are weighted particles. Sibling posteriors initially share propagated
particles, KLD-sized resampling is triggered by design effect, and observation
branches at L1 distance at most ``delta`` are packed. Search descends through
the action with the largest upper bound and the packed child with the largest
probability-weighted excess uncertainty. It returns the root action with the
largest lower bound.

Paper: https://papers.nips.cc/paper_files/paper/2021/file/
ef41d488755367316f04fc0e0e9dc9fc-Paper.pdf
Author code inspected: https://github.com/JuliaPOMDP/AdaOPS.jl (main, 2026-09-08).
The author code is MIT licensed. This is a Python implementation from the
paper's equations; no Julia source was copied.

Deliberate differences: bounds use this repository's finite ``depth`` contract;
KLD adaptation needs an explicit ``state_binner`` and is otherwise explicitly
disabled; fixed trials and wall-clock timeout are mutually exclusive so cached
runs have one unambiguous compute control.
"""

import copy
import importlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from POMDPPlanners.core.belief import Belief, WeightedParticleBelief
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, SpaceType
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.core.tree.arena import ACTION, BELIEF, Tree
from POMDPPlanners.planners.planners_utils.scenario_streams import ScenarioRandomStreams
from POMDPPlanners.planners.scenario_tree_planners.adaops.adaptive_particles import (
    adaptive_resample,
    design_effect,
    effective_sample_size,
    l1_weight_distance,
    normalize_weights,
)
from POMDPPlanners.planners.scenario_tree_planners.despot import (
    DESPOT,
    TERMINAL_OBSERVATION,
    TINY,
)
from POMDPPlanners.utils.config_to_id import config_to_id
from POMDPPlanners.utils.tree_statistics import TreeMetrics, compute_arena_tree_metrics


class AdaOPSMetrics(Enum):
    """Numeric diagnostics reset for every :meth:`AdaOPS.action` call."""

    N_TRIALS = "adaops_n_trials"
    ROOT_LOWER_BOUND = "adaops_root_lower_bound"
    ROOT_UPPER_BOUND = "adaops_root_upper_bound"
    ROOT_GAP = "adaops_root_gap"
    N_BELIEF_NODES = "adaops_n_belief_nodes"
    N_ACTION_NODES = "adaops_n_action_nodes"
    PACKED_BRANCHES = "adaops_packed_branches"
    MERGED_BRANCHES = "adaops_merged_branches"
    RESAMPLE_COUNT = "adaops_resample_count"
    MIN_PARTICLES = "adaops_min_particles"
    MAX_PARTICLES = "adaops_max_particles"
    MEAN_PARTICLES = "adaops_mean_particles"
    MEAN_ESS = "adaops_mean_effective_sample_size"
    MAX_DESIGN_EFFECT = "adaops_max_design_effect"
    BOUND_CORRECTIONS = "adaops_bound_corrections"
    ZERO_WEIGHT_CORRECTIONS = "adaops_zero_weight_corrections"
    STOPPED_BY_GAP = "adaops_stopped_by_gap"
    STOPPED_BY_TIMEOUT = "adaops_stopped_by_timeout"
    STOPPED_BY_TRIALS = "adaops_stopped_by_trials"
    STOPPED_BY_DEPTH = "adaops_stopped_by_depth"
    STALLED = "adaops_stalled"


@dataclass
class _AdaBeliefData:
    depth: int
    weights: np.ndarray
    default_value: float
    expanded: bool = False
    terminal: bool = False
    in_tree: bool = False
    best_upper_action_id: Optional[int] = None
    scenario_ids: List[int] = field(default_factory=list)
    observation_probability: float = 1.0
    packed_observations: List[Any] = field(default_factory=list)
    occupied_bins: int = 0
    resampled: bool = False


BoundProvider = Callable[[Sequence[Any], np.ndarray, int], float]


class AdaOPS(DESPOT):
    """AdaOPS planner for discrete actions and scoreable observations.

    ``state_binner`` maps a state to a hashable bin. Passing ``None`` disables
    KLD adaptation explicitly and resampling uses ``max_particles``. A callable
    requires ``state_binner_id`` so its semantics enter ``config_id``.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals

    def __init__(
        self,
        environment: DiscreteActionsEnvironment,
        discount_factor: float,
        depth: int,
        name: str,
        epsilon_0: float = 1e-3,
        xi: float = 0.95,
        delta: float = 0.1,
        min_particles: int = 30,
        max_particles: int = 200,
        zeta: float = 0.05,
        kld_confidence: float = 0.95,
        design_effect_threshold: float = 2.0,
        state_binner: Optional[Any] = None,
        state_binner_id: Optional[str] = None,
        lower_bound: Optional[BoundProvider] = None,
        upper_bound: Optional[BoundProvider] = None,
        max_reward: Optional[float] = None,
        min_reward: Optional[float] = None,
        rollout_depth: Optional[int] = None,
        random_seed: int = 0,
        time_out_in_seconds: Optional[int] = None,
        n_simulations: Optional[int] = None,
        reserve_capacity: int = 0,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
    ):
        self._validate_adaops_params(
            epsilon_0,
            xi,
            delta,
            min_particles,
            max_particles,
            zeta,
            kld_confidence,
            design_effect_threshold,
            state_binner,
            state_binner_id,
        )
        super().__init__(
            environment=environment,
            discount_factor=discount_factor,
            depth=depth,
            name=name,
            n_scenarios=max_particles,
            eta=xi,
            max_reward=max_reward,
            min_reward=min_reward,
            rollout_depth=rollout_depth,
            scenario_seed=random_seed,
            use_determinized_scenarios=False,
            time_out_in_seconds=time_out_in_seconds,
            n_simulations=n_simulations,
            reserve_capacity=reserve_capacity,
            log_path=log_path,
            debug=debug,
            use_queue_logger=use_queue_logger,
        )
        self.epsilon_0 = float(epsilon_0)
        self.xi = float(xi)
        self.delta = float(delta)
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.zeta = float(zeta)
        self.kld_confidence = float(kld_confidence)
        self.design_effect_threshold = float(design_effect_threshold)
        self._state_binner, persisted_binner = self._resolve_state_binner(
            state_binner, state_binner_id
        )
        self.state_binner = persisted_binner
        self.state_binner_id = persisted_binner
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.random_seed = random_seed
        self._particle_rng = np.random.default_rng(random_seed)
        self._reset_adaops_counters()
        self._last_snapshot: Optional[Dict[str, Any]] = None
        self._stable_config_id = config_to_id(
            {
                "environment": environment.config_id,
                "discount_factor": self.discount_factor,
                "planner_class": type(self).__name__,
                "name": self.name,
                "depth": self.depth,
                "algorithm": {
                    "epsilon_0": self.epsilon_0,
                    "xi": self.xi,
                    "delta": self.delta,
                    "min_particles": self.min_particles,
                    "max_particles": self.max_particles,
                    "zeta": self.zeta,
                    "kld_confidence": self.kld_confidence,
                    "design_effect_threshold": self.design_effect_threshold,
                    "state_binner_id": self.state_binner_id,
                    "lower_bound": self._callable_id(self.lower_bound),
                    "upper_bound": self._callable_id(self.upper_bound),
                    "max_reward": self.max_reward,
                    "min_reward": self.min_reward,
                    "rollout_depth": self.rollout_depth,
                    "random_seed": self.random_seed,
                },
                "budget": {
                    "n_simulations": self.n_simulations,
                    "time_out_in_seconds": self.time_out_in_seconds,
                },
            }
        )

    @staticmethod
    def _callable_id(value: Optional[BoundProvider]) -> Optional[str]:
        if value is None:
            return None
        return f"{value.__module__}.{value.__qualname__}"

    @staticmethod
    def _resolve_state_binner(
        value: Optional[Any], identifier: Optional[str]
    ) -> Tuple[Optional[Callable[[Any], Any]], Optional[str]]:
        if value is None:
            return None, None
        path = value if isinstance(value, str) else identifier
        if not path or "." not in path:
            raise ValueError("state_binner_id must be an importable 'module.function' path")
        module_name, attribute = path.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_name), attribute)
        if not callable(imported):
            raise ValueError(f"state_binner_id does not name a callable: {path}")
        if callable(value) and imported is not value:
            raise ValueError(f"state_binner_id does not resolve to state_binner: {path}")
        return imported, path

    @property
    def config_id(self) -> str:
        """Stable constructor identity; search and mutable environment state are excluded."""
        return self._stable_config_id

    @staticmethod
    def _validate_adaops_params(
        epsilon_0: float,
        xi: float,
        delta: float,
        minimum: int,
        maximum: int,
        zeta: float,
        confidence: float,
        threshold: float,
        state_binner: Optional[Any],
        state_binner_id: Optional[str],
    ) -> None:
        if epsilon_0 < 0.0:
            raise ValueError("epsilon_0 must be non-negative")
        if not 0.0 < xi < 1.0:
            raise ValueError("xi must be strictly between 0 and 1")
        if delta < 0.0 or delta > 2.0:
            raise ValueError("delta must be between 0 and 2")
        if minimum <= 0 or maximum < minimum:
            raise ValueError("particle limits must satisfy 0 < min_particles <= max_particles")
        if zeta <= 0.0:
            raise ValueError("zeta must be positive")
        if not 0.5 < confidence < 1.0:
            raise ValueError("kld_confidence must be between 0.5 and 1")
        if threshold < 1.0:
            raise ValueError("design_effect_threshold must be at least 1")
        if callable(state_binner) and not state_binner_id:
            raise ValueError("state_binner_id is required when state_binner is provided")
        if state_binner is None and state_binner_id is not None:
            raise ValueError("state_binner_id requires state_binner")
        if isinstance(state_binner, str) and state_binner_id not in (None, state_binner):
            raise ValueError("saved state_binner and state_binner_id must match")

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        return PolicySpaceInfo(
            action_space=SpaceType.DISCRETE, observation_space=SpaceType.CONTINUOUS
        )

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return [metric.value for metric in TreeMetrics] + [metric.value for metric in AdaOPSMetrics]

    def _reset_adaops_counters(self) -> None:
        self._packed_branches = 0
        self._merged_branches = 0
        self._resample_count = 0
        self._zero_weight_corrections = 0
        self._stopped_by_timeout = False
        self._stopped_by_trials = False
        self._stopped_by_depth = False

    def action(self, belief: Belief) -> Tuple[List[Any], PolicyRunData]:
        if self._is_terminal_belief(belief=belief):
            return [self._sample_random_action(belief=belief)], PolicyRunData(info_variables=[])
        tree, root_id = self._learn_tree(belief)
        self._last_tree, self._last_root_id = tree, root_id
        chosen = (
            self._sample_random_action(belief=belief)
            if not tree.children_ids[root_id]
            else self._action_of_best_lower_bound(tree, root_id)
        )
        metrics = compute_arena_tree_metrics(tree=tree, root_id=root_id)
        metrics.extend(self._adaops_metrics(tree, root_id))
        self._last_snapshot = copy.deepcopy(self._snapshot(tree, root_id, chosen))
        if self._search_state_dump_dir is not None:
            self._write_search_state_dump(tree, root_id)
        return [chosen], PolicyRunData(info_variables=metrics)

    def _learn_tree(self, belief: Belief) -> Tuple[Tree, int]:
        tree = Tree()
        capacity = self._effective_reserve_capacity()
        if capacity > 0:
            tree.reserve(capacity)
        self._call_index += 1
        self._particle_rng = np.random.default_rng(self.random_seed + self._call_index)
        self._scenario_rng = self._particle_rng
        self._streams = ScenarioRandomStreams(
            n_scenarios=self.max_particles,
            max_depth=max(self.depth, self.rollout_depth),
            base_seed=self.random_seed + self._call_index,
        )
        self._n_trials = 0
        self._max_trial_depth = 0
        self._gap_closed = False
        self._stalled = False
        self._bound_clamps = 0
        self._reset_adaops_counters()
        particles, weights = self._root_particles(belief)
        root_id = self._add_weighted_belief_node(
            tree, particles, weights, 0, None, None, None, 1.0, list(range(len(particles)))
        )
        self._root_id = root_id
        if self.n_simulations is not None:
            for _ in range(self.n_simulations):
                if self._should_stop(tree, root_id):
                    break
                self._simulate_path(tree, root_id, 0)
            self._stopped_by_trials = not self._gap_closed and self._n_trials >= self.n_simulations
        else:
            if self.time_out_in_seconds is None:
                raise ValueError("one search budget is required")
            start = time.monotonic()
            while time.monotonic() - start < self.time_out_in_seconds:
                if self._should_stop(tree, root_id):
                    break
                self._simulate_path(tree, root_id, 0)
            self._stopped_by_timeout = not self._gap_closed and not self._stalled
        final_gap = tree.upper_confidence_bound[root_id] - tree.lower_confidence_bound[root_id]
        if final_gap <= self.epsilon_0 + TINY:
            self._gap_closed = True
            self._stopped_by_trials = False
            self._stopped_by_timeout = False
        self._last_tree_size = len(tree)
        return tree, root_id

    def _root_particles(self, belief: Belief) -> Tuple[List[Any], np.ndarray]:
        particles = getattr(belief, "particles", None)
        if particles is None or len(particles) == 0:
            # Inherited so the draw is seeded from the planner's own generator
            # and the caller's global RNG streams are restored afterwards.
            particles = self._sample_from_belief_sampler(belief, self.max_particles)
            weights = np.full(len(particles), 1.0 / len(particles))
        else:
            particles = list(particles)
            weights = self._particle_weights(belief, len(particles))
        sampled, sampled_weights, _ = adaptive_resample(
            particles,
            weights,
            self._particle_rng,
            self.min_particles,
            self.max_particles,
            self.zeta,
            self._state_binner,
            self.kld_confidence,
        )
        self._resample_count += 1
        return sampled, sampled_weights

    def _maybe_resample(
        self, particles: Sequence[Any], weights: Sequence[float]
    ) -> Tuple[List[Any], np.ndarray, int, bool]:
        normalized = normalize_weights(weights)
        deff = design_effect(normalized)
        if deff <= self.design_effect_threshold:
            return list(particles), normalized, 0, False
        sampled, sampled_weights, occupied = adaptive_resample(
            particles,
            normalized,
            self._particle_rng,
            self.min_particles,
            self.max_particles,
            self.zeta,
            self._state_binner,
            self.kld_confidence,
        )
        self._resample_count += 1
        return sampled, sampled_weights, occupied, True

    def _should_stop(self, tree: Tree, root_id: int) -> bool:
        if self._stalled:
            return True
        gap = tree.upper_confidence_bound[root_id] - tree.lower_confidence_bound[root_id]
        if gap <= self.epsilon_0 + TINY:
            self._gap_closed = True
            return True
        return False

    def _expand(self, tree: Tree, belief_id: int) -> None:
        data: _AdaBeliefData = tree.data[belief_id]
        particles = list(cast(WeightedParticleBelief, tree.get_belief(belief_id)).particles)
        weights = data.weights
        parent_mass = tree.weight[belief_id]
        best_upper = -np.inf
        best_action_id: Optional[int] = None
        environment = cast(DiscreteActionsEnvironment, self.environment)
        for action in environment.get_actions():
            next_states: List[Any] = []
            observations: List[Any] = []
            rewards: List[float] = []
            unique: Dict[Any, Any] = {}
            generated_mass: Dict[Any, float] = {}
            for state, weight in zip(particles, weights):
                if self.environment.is_terminal(state=state):
                    next_state, observation, reward = state, TERMINAL_OBSERVATION, 0.0
                    key = TERMINAL_OBSERVATION
                else:
                    next_state, observation, reward = self.environment.sample_next_step(
                        state, action
                    )
                    key = self._observation_key(observation)
                next_states.append(next_state)
                observations.append(observation)
                rewards.append(float(reward))
                unique.setdefault(key, observation)
                generated_mass[key] = generated_mass.get(key, 0.0) + float(weight)
            action_id = tree.add_action_node(
                action=action,
                parent_id=belief_id,
                action_key=self.environment.hash_action(action),
            )
            tree.set_immediate_reward(action_id, float(np.dot(weights, rewards)))
            retained: List[Tuple[int, np.ndarray]] = []
            for key, observation in unique.items():
                if key is TERMINAL_OBSERVATION:
                    posterior = normalize_weights(
                        [
                            weight if obs is TERMINAL_OBSERVATION else 0.0
                            for weight, obs in zip(weights, observations)
                        ]
                    )
                else:
                    log_likelihood = self.environment.observation_log_probability_per_state(
                        next_states=next_states, action=action, observation=observation
                    )
                    likelihood = np.exp(np.asarray(log_likelihood) - np.max(log_likelihood))
                    raw = weights * likelihood
                    if not np.all(np.isfinite(raw)) or float(raw.sum()) <= 0.0:
                        self._zero_weight_corrections += 1
                        continue
                    posterior = normalize_weights(raw)
                merged_into: Optional[int] = None
                for child_id, retained_weights in retained:
                    if l1_weight_distance(posterior, retained_weights) <= self.delta + TINY:
                        merged_into = child_id
                        break
                branch_probability = generated_mass[key]
                if merged_into is not None:
                    tree.weight[merged_into] += parent_mass * branch_probability
                    tree.data[merged_into].observation_probability += branch_probability
                    tree.data[merged_into].packed_observations.append(observation)
                    tree.recompute_children_cdf(action_id)
                    self._merged_branches += 1
                    continue
                child_particles, child_weights, occupied, resampled = self._maybe_resample(
                    next_states, posterior
                )
                child_id = self._add_weighted_belief_node(
                    tree,
                    child_particles,
                    child_weights,
                    data.depth + 1,
                    action_id,
                    observation,
                    key,
                    parent_mass * branch_probability,
                    list(range(len(child_particles))),
                    branch_probability,
                    occupied,
                    resampled,
                )
                retained.append((child_id, posterior))
                self._packed_branches += 1
            self._update_action_bounds(tree, action_id)
            upper = tree.upper_confidence_bound[action_id]
            if upper > best_upper + TINY:
                best_upper, best_action_id = upper, action_id
        data.expanded = True
        data.best_upper_action_id = best_action_id

    def _add_weighted_belief_node(
        self,
        tree: Tree,
        particles: Sequence[Any],
        weights: Sequence[float],
        depth: int,
        parent_id: Optional[int],
        observation: Any,
        obs_key: Any,
        mass: float,
        scenario_ids: List[int],
        observation_probability: float = 1.0,
        occupied_bins: int = 0,
        resampled: bool = False,
    ) -> int:
        normalized = normalize_weights(weights)
        log_weights = np.log(np.maximum(normalized, 1e-300))
        if not np.any(log_weights != 0.0):
            # The repository belief class rejects an all-zero log vector even
            # though it is the valid representation of a one-particle belief.
            # A shared additive offset leaves normalized weights unchanged.
            log_weights = log_weights - 1.0
        belief = WeightedParticleBelief(particles=list(particles), log_weights=log_weights)
        node_id = tree.add_belief_node(
            belief=belief,
            observation=observation,
            weight=mass,
            parent_id=parent_id,
            obs_key=obs_key,
        )
        tree.sample[node_id] = list(scenario_ids)
        terminal = all(self.environment.is_terminal(state=state) for state in particles)
        lower, upper = self._weighted_initial_bounds(
            particles, normalized, scenario_ids, depth, terminal
        )
        tree.lower_confidence_bound[node_id] = lower
        tree.upper_confidence_bound[node_id] = upper
        tree.v_value[node_id] = lower
        tree.data[node_id] = _AdaBeliefData(
            depth=depth,
            weights=normalized,
            default_value=lower,
            terminal=terminal,
            scenario_ids=scenario_ids,
            observation_probability=observation_probability,
            packed_observations=[observation] if observation is not None else [],
            occupied_bins=occupied_bins,
            resampled=resampled,
        )
        return node_id

    def _weighted_initial_bounds(
        self,
        states: Sequence[Any],
        weights: np.ndarray,
        scenario_ids: Sequence[int],
        depth: int,
        terminal: bool,
    ) -> Tuple[float, float]:
        if terminal or depth >= self.depth:
            self._stopped_by_depth = self._stopped_by_depth or depth >= self.depth
            return 0.0, 0.0
        if self.lower_bound is not None:
            lower = float(self.lower_bound(states, weights.copy(), depth))
        else:
            actions = self._default_policy_actions(int(scenario_ids[0]), depth)
            values = [
                self._default_policy_value(int(sid), state, depth, actions)
                for sid, state in zip(scenario_ids, states)
            ]
            lower = float(np.dot(weights, values))
        if self.upper_bound is not None:
            upper = float(self.upper_bound(states, weights.copy(), depth))
        else:
            remaining = self.depth - depth
            values = [
                0.0
                if self.environment.is_terminal(state=state)
                else self.max_reward * self._discount_sum(remaining)
                for state in states
            ]
            upper = float(np.dot(weights, values))
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("bound providers must return finite values")
        return lower, self._reconcile_bounds(lower, upper)

    def _backup(self, tree: Tree, belief_id: int) -> None:
        """Bellman backup: maximize each bound over all action children."""
        action_ids = tree.get_children_ids(belief_id)
        if not action_ids:
            return
        for action_id in action_ids:
            self._update_action_bounds(tree, action_id)
        best_upper = max(action_ids, key=lambda node: tree.upper_confidence_bound[node])
        best_lower = max(action_ids, key=lambda node: tree.lower_confidence_bound[node])
        lower = max(
            tree.lower_confidence_bound[belief_id],
            tree.lower_confidence_bound[best_lower],
        )
        upper = self._reconcile_bounds(lower, float(tree.upper_confidence_bound[best_upper]))
        tree.lower_confidence_bound[belief_id] = lower
        tree.upper_confidence_bound[belief_id] = upper
        tree.v_value[belief_id] = lower
        tree.data[belief_id].best_upper_action_id = best_upper

    def _adaops_metrics(self, tree: Tree, root_id: int) -> List[PolicyInfoVariable]:
        belief_ids = [node_id for node_id in range(len(tree)) if tree.kind[node_id] == BELIEF]
        action_count = sum(tree.kind[node_id] == ACTION for node_id in range(len(tree)))
        particle_counts = [
            len(cast(WeightedParticleBelief, tree.get_belief(node_id)).particles)
            for node_id in belief_ids
        ]
        ess_values = [effective_sample_size(tree.data[node_id].weights) for node_id in belief_ids]
        design_values = [design_effect(tree.data[node_id].weights) for node_id in belief_ids]
        lower = float(tree.lower_confidence_bound[root_id])
        upper = float(tree.upper_confidence_bound[root_id])
        values = {
            AdaOPSMetrics.N_TRIALS: self._n_trials,
            AdaOPSMetrics.ROOT_LOWER_BOUND: lower,
            AdaOPSMetrics.ROOT_UPPER_BOUND: upper,
            AdaOPSMetrics.ROOT_GAP: upper - lower,
            AdaOPSMetrics.N_BELIEF_NODES: len(belief_ids),
            AdaOPSMetrics.N_ACTION_NODES: action_count,
            AdaOPSMetrics.PACKED_BRANCHES: self._packed_branches,
            AdaOPSMetrics.MERGED_BRANCHES: self._merged_branches,
            AdaOPSMetrics.RESAMPLE_COUNT: self._resample_count,
            AdaOPSMetrics.MIN_PARTICLES: min(particle_counts),
            AdaOPSMetrics.MAX_PARTICLES: max(particle_counts),
            AdaOPSMetrics.MEAN_PARTICLES: float(np.mean(particle_counts)),
            AdaOPSMetrics.MEAN_ESS: float(np.mean(ess_values)),
            AdaOPSMetrics.MAX_DESIGN_EFFECT: max(design_values),
            AdaOPSMetrics.BOUND_CORRECTIONS: self._bound_clamps,
            AdaOPSMetrics.ZERO_WEIGHT_CORRECTIONS: self._zero_weight_corrections,
            AdaOPSMetrics.STOPPED_BY_GAP: int(self._gap_closed),
            AdaOPSMetrics.STOPPED_BY_TIMEOUT: int(self._stopped_by_timeout),
            AdaOPSMetrics.STOPPED_BY_TRIALS: int(self._stopped_by_trials),
            AdaOPSMetrics.STOPPED_BY_DEPTH: int(self._stopped_by_depth),
            AdaOPSMetrics.STALLED: int(self._stalled),
        }
        return [
            PolicyInfoVariable(name=metric.value, value=values[metric]) for metric in AdaOPSMetrics
        ]

    def get_last_search_state(self) -> Dict[str, Any]:
        """Return an immutable deep copy of the last decision snapshot."""
        if self._last_snapshot is None:
            raise ValueError("no search state available; call action() first")
        return copy.deepcopy(self._last_snapshot)

    def export_search_state(
        self, path: Path, tree: Optional[Tree] = None, root_id: Optional[int] = None
    ) -> Path:
        if tree is None or root_id is None:
            if self._last_snapshot is None:
                raise ValueError("no search state to export; call action() first")
            payload = copy.deepcopy(self._last_snapshot)
        else:
            selected = (
                self._action_of_best_lower_bound(tree, root_id)
                if tree.children_ids[root_id]
                else None
            )
            payload = self._snapshot(tree, root_id, selected)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def _snapshot(self, tree: Tree, root_id: int, selected: Any) -> Dict[str, Any]:
        nodes = []
        for node_id in range(len(tree)):
            common = {
                "id": node_id,
                "kind": "belief" if tree.kind[node_id] == BELIEF else "action",
                "parent_id": tree.parent_id[node_id],
                "children_ids": list(tree.children_ids[node_id]),
                "depth": tree.data[node_id].depth if tree.kind[node_id] == BELIEF else None,
                "probability_mass": tree.weight[node_id],
                "lower_bound": tree.lower_confidence_bound[node_id],
                "upper_bound": tree.upper_confidence_bound[node_id],
                "visits": tree.visit_count[node_id],
            }
            if tree.kind[node_id] == BELIEF:
                data: _AdaBeliefData = tree.data[node_id]
                common.update(
                    {
                        "observation": repr(tree.observation[node_id]),
                        "packed_observations": [repr(value) for value in data.packed_observations],
                        "particles": [
                            repr(value)
                            for value in cast(
                                WeightedParticleBelief, tree.get_belief(node_id)
                            ).particles
                        ],
                        "weights": data.weights.tolist(),
                        "effective_sample_size": effective_sample_size(data.weights),
                        "design_effect": design_effect(data.weights),
                        "occupied_bins": data.occupied_bins,
                        "resampled": data.resampled,
                    }
                )
            else:
                common.update(
                    {
                        "action": repr(tree.action[node_id]),
                        "immediate_reward": tree.immediate_reward[node_id],
                    }
                )
            nodes.append(common)
        return {
            "planner": self.name,
            "planner_class": type(self).__name__,
            "config_id": self.config_id,
            "selected_action": repr(selected),
            "root_id": root_id,
            "nodes": nodes,
        }

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # User callables are configuration, but many local functions are not
        # pickleable. Require pickleable providers instead of silently losing them.
        return state
