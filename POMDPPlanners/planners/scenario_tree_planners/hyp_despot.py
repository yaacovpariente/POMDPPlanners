# SPDX-License-Identifier: MIT
"""HyP-DESPOT: shared CPU search with batched CUDA leaf expansion.

This is a clean implementation from HyP-DESPOT (Cai et al., RSS 2018,
arXiv:1802.06215).  It does not copy the unlicensed author C++/CUDA source.
CPU workers traverse one scenario tree using scenario-weighted PO-UCT and
temporary observation-branch virtual loss.  A queue combines discovered
leaves; transition, reward, observation grouping, terminal masking, and leaf
bounds then execute over the leaf/action/scenario Cartesian batch on CUDA.
Final choice uses backed-up lower values, never the exploration score.
"""
# pylint: disable=too-many-instance-attributes,too-many-arguments
import copy
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast
import torch
from torch import Tensor
from POMDPPlanners.core.environment import SpaceType
from POMDPPlanners.core.environment.hyp_despot_cuda_model import HypDESPOTCUDAmodel
from POMDPPlanners.core.policy import Policy, PolicyInfoVariable, PolicyRunData, PolicySpaceInfo
from POMDPPlanners.planners.scenario_tree_planners.hyp_despot_cuda import (
    CUDAExpansionResult,
    HypDESPOTCompatibilityError,
    expand_cuda_leaves,
    validate_cuda_contract,
)


class HypDESPOTMetrics(Enum):
    TRAVERSALS = "hyp_despot_traversals"
    CUDA_BATCHES = "hyp_despot_cuda_batches"
    CUDA_LEAVES = "hyp_despot_cuda_leaves"
    CUDA_SCENARIOS = "hyp_despot_cuda_scenarios"
    CUDA_ACTIONS = "hyp_despot_cuda_actions"
    MEAN_BATCH_SIZE = "hyp_despot_mean_batch_size"
    MAX_BATCH_SIZE = "hyp_despot_max_batch_size"
    QUEUE_TIME = "hyp_despot_queue_time_seconds"
    KERNEL_TIME = "hyp_despot_kernel_time_seconds"
    TRANSFER_TIME = "hyp_despot_transfer_time_seconds"
    TRAVERSAL_TIME = "hyp_despot_traversal_time_seconds"
    BACKUP_TIME = "hyp_despot_backup_time_seconds"
    TOTAL_TIME = "hyp_despot_total_time_seconds"
    VIRTUAL_LOSS = "hyp_despot_virtual_loss_applications"
    LOCK_CONTENTION = "hyp_despot_lock_contention_seconds"
    ROOT_LOWER = "hyp_despot_root_lower_bound"
    ROOT_UPPER = "hyp_despot_root_upper_bound"
    ROOT_GAP = "hyp_despot_root_gap"
    GPU_DEVICE = "hyp_despot_gpu_device"
    PEAK_MEMORY = "hyp_despot_peak_memory_bytes"


@dataclass
class ObservationBranch:
    key: int
    child: int
    weight: float
    virtual_loss: float = 0.0


@dataclass
class ActionBranch:
    action_index: int
    visits: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    immediate_reward: float = 0.0
    observations: Dict[int, ObservationBranch] = field(default_factory=dict)


@dataclass
class BeliefNode:
    node_id: int
    parent_action: Optional[Tuple[int, int]]
    depth: int
    states: Tensor
    scenario_ids: Tensor
    weights: Tensor
    visits: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    expanded: bool = False
    terminal: bool = False
    actions: Dict[int, ActionBranch] = field(default_factory=dict)


class HypDESPOT(Policy):
    """Hybrid parallel DESPOT requiring a deterministic CUDA scenario model.

    The environment must expose ``hyp_despot_cuda_model`` after JSON reload.
    ``expansion_actions`` is an explicit finite list; its indices are passed to
    CUDA. ``n_traversals`` is a deterministic work budget.
    """

    def __init__(
        self,
        environment: Any,
        discount_factor: float,
        depth: int,
        name: str,
        n_scenarios: int = 64,
        n_traversals: int = 64,
        n_workers: int = 4,
        cuda_batch_size: int = 8,
        exploration_constant: float = 1.0,
        virtual_loss: float = 1.0,
        scenario_seed: int = 0,
        expansion_actions: Optional[Sequence[Any]] = None,
        device: str = "cuda:0",
        search_state_dump_dir: Optional[Path] = None,
        log_path: Optional[Path] = None,
        debug: bool = False,
        use_queue_logger: bool = False,
        model: Any = None,
    ):
        if (
            depth <= 0
            or n_scenarios <= 0
            or n_traversals <= 0
            or n_workers <= 0
            or cuda_batch_size <= 0
        ):
            raise ValueError(
                "depth, scenarios, traversals, workers, and CUDA batch size must be positive"
            )
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        if exploration_constant < 0 or virtual_loss < 0:
            raise ValueError("exploration_constant and virtual_loss must be non-negative")
        self.depth = depth
        self.n_scenarios = n_scenarios
        self.n_traversals = n_traversals
        self.n_workers = n_workers
        self.cuda_batch_size = cuda_batch_size
        self.exploration_constant = float(exploration_constant)
        self.virtual_loss = float(virtual_loss)
        self.scenario_seed = scenario_seed
        self.expansion_actions = list(expansion_actions) if expansion_actions is not None else None
        self.device = device
        self.search_state_dump_dir = Path(search_state_dump_dir) if search_state_dump_dir else None
        self._model_override = model
        self._tree_lock = threading.RLock()
        self._last_snapshot = None
        super().__init__(environment, discount_factor, name, log_path, debug, use_queue_logger)
        self._actions = self.expansion_actions or self._resolve_actions(environment)
        self.expansion_actions = list(self._actions)
        resolved_model = (
            model if model is not None else getattr(environment, "hyp_despot_cuda_model", None)
        )
        if resolved_model is None:
            raise HypDESPOTCompatibilityError(
                "environment must expose hyp_despot_cuda_model or model must be supplied"
            )
        self._model: HypDESPOTCUDAmodel = cast(HypDESPOTCUDAmodel, resolved_model)
        validate_cuda_contract(self._model, self._actions, torch.device(device))
        self._reset_transient()

    @staticmethod
    def _resolve_actions(environment: Any) -> List[Any]:
        getter = getattr(environment, "get_actions", None)
        if not callable(getter):
            raise HypDESPOTCompatibilityError(
                "continuous actions require an explicit finite expansion_actions set"
            )
        actions = list(cast(Iterable[Any], getter()))
        if not actions:
            raise HypDESPOTCompatibilityError("finite expansion action set must not be empty")
        return actions

    @classmethod
    def get_space_info(cls) -> PolicySpaceInfo:
        # Continuous environments are legal only when the caller supplies a
        # finite representative expansion set; constructor validation enforces
        # that condition more precisely than the generic compatibility check.
        return PolicySpaceInfo(SpaceType.MIXED, SpaceType.MIXED)

    @classmethod
    def get_info_variable_names(cls) -> List[str]:
        return [x.value for x in HypDESPOTMetrics]

    def _reset_transient(self) -> None:
        self._nodes: Dict[int, BeliefNode] = {}
        self._next_node_id = 0
        self._counts = {metric: 0.0 for metric in HypDESPOTMetrics}
        self._batch_sizes = []

    @staticmethod
    def scenario_po_uct(
        action_visits: float,
        parent_visits: float,
        scenario_weight: float,
        value: float,
        exploration_constant: float,
        virtual_loss: float = 0.0,
    ) -> float:
        """Paper PO-UCT: value plus exploration scaled by scenario mass."""
        if action_visits <= 0:
            return math.inf
        bonus = exploration_constant * math.sqrt(
            max(0.0, scenario_weight) * math.log(max(1.0, parent_visits)) / action_visits
        )
        return value + bonus - virtual_loss

    def _acquire(self) -> float:
        start = time.perf_counter()
        self._tree_lock.acquire()
        waited = time.perf_counter() - start
        self._counts[HypDESPOTMetrics.LOCK_CONTENTION] += waited
        return waited

    def _new_node(
        self,
        states: Tensor,
        ids: Tensor,
        weights: Tensor,
        depth: int,
        parent: Optional[Tuple[int, int]],
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        self._nodes[node_id] = BeliefNode(node_id, parent, depth, states, ids, weights)
        return node_id

    def _root_particles(self, belief: Any) -> Tuple[Tensor, Tensor, Tensor]:
        particles = getattr(belief, "particles", belief)
        if not isinstance(particles, Tensor):
            raise HypDESPOTCompatibilityError(
                "HypDESPOT belief particles must be a CUDA torch.Tensor; scalar Python models are unsupported"
            )
        device = torch.device(self.device)
        if particles.device != device:
            raise HypDESPOTCompatibilityError(
                f"belief particles must already be on {device}; hidden copies are forbidden"
            )
        if particles.dim() != 2 or not particles.dtype.is_floating_point:
            raise HypDESPOTCompatibilityError(
                "belief particles must be floating [scenario, state] tensors"
            )
        if particles.shape[0] < self.n_scenarios:
            raise HypDESPOTCompatibilityError("belief has fewer particles than n_scenarios")
        states = particles[: self.n_scenarios]
        ids = torch.arange(self.n_scenarios, device=device, dtype=torch.int64)
        raw = getattr(belief, "weights", None)
        if raw is None:
            weights = torch.full(
                (self.n_scenarios,), 1 / self.n_scenarios, device=device, dtype=states.dtype
            )
        else:
            if (
                not isinstance(raw, Tensor)
                or raw.device != device
                or raw.shape[0] < self.n_scenarios
            ):
                raise HypDESPOTCompatibilityError(
                    "belief weights must be a same-device tensor aligned with particles"
                )
            weights = raw[: self.n_scenarios]
            total = weights.sum()
            if (
                not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any())
                or float(total) <= 0
            ):
                raise HypDESPOTCompatibilityError(
                    "belief weights must be finite, non-negative, and have positive mass"
                )
            weights = weights / total
        return states, ids, weights

    def _select_leaf(self, root: int) -> Tuple[int, List[Tuple[int, int, int]]]:
        path: List[Tuple[int, int, int]] = []
        self._acquire()
        try:
            node = self._nodes[root]
            while node.expanded and node.depth < self.depth and node.actions:
                parent = max(1.0, node.visits)
                total_weight = float(node.weights.sum())
                action = max(
                    node.actions.values(),
                    key=lambda a: self.scenario_po_uct(
                        a.visits,
                        parent,
                        total_weight,
                        a.upper,
                        self.exploration_constant,
                        sum(o.virtual_loss for o in a.observations.values()),
                    ),
                )
                if not action.observations:
                    break
                obs = max(action.observations.values(), key=lambda o: o.weight - o.virtual_loss)
                obs.virtual_loss += self.virtual_loss
                self._counts[HypDESPOTMetrics.VIRTUAL_LOSS] += 1
                path.append((node.node_id, action.action_index, obs.key))
                node = self._nodes[obs.child]
            return node.node_id, path
        except Exception:
            for node_id, action, key in path:
                branch = self._nodes[node_id].actions[action].observations[key]
                branch.virtual_loss = max(0.0, branch.virtual_loss - self.virtual_loss)
            raise
        finally:
            self._tree_lock.release()

    def _release_virtual_loss(self, path: List[Tuple[int, int, int]]) -> None:
        self._acquire()
        try:
            for node_id, action, key in path:
                branch = self._nodes[node_id].actions[action].observations[key]
                branch.virtual_loss = max(0.0, branch.virtual_loss - self.virtual_loss)
        finally:
            self._tree_lock.release()

    def _integrate(self, leaf_ids: List[int], result: CUDAExpansionResult) -> None:
        start = time.perf_counter()
        self._acquire()
        try:
            for local_leaf, node_id in enumerate(leaf_ids):
                node = self._nodes[node_id]
                if node.expanded:
                    continue
                rows = result.leaf_indices == local_leaf
                for action_index in range(len(self._actions)):
                    mask = rows & (result.action_indices == action_index)
                    rewards = result.rewards[mask]
                    terminal = result.terminal[mask]
                    source_indices = result.scenario_indices[mask]
                    row_weights = node.weights[source_indices]
                    normalized_weights = row_weights / row_weights.sum()
                    lower = torch.where(
                        terminal, torch.zeros_like(result.lower[mask]), result.lower[mask]
                    )
                    upper = torch.where(
                        terminal, torch.zeros_like(result.upper[mask]), result.upper[mask]
                    )
                    branch = ActionBranch(
                        action_index,
                        visits=self._host_float(node.weights.sum()),
                        immediate_reward=self._host_float((rewards * normalized_weights).sum()),
                    )
                    branch.lower = (
                        branch.immediate_reward
                        + self.discount_factor
                        * self._host_float((lower * normalized_weights).sum())
                    )
                    branch.upper = (
                        branch.immediate_reward
                        + self.discount_factor
                        * self._host_float((upper * normalized_weights).sum())
                    )
                    keys = result.observation_keys[mask]
                    next_states = result.next_states[mask]
                    for key_tensor in torch.unique(keys):
                        key = int(key_tensor.item())
                        group = keys == key_tensor
                        indices = source_indices[group]
                        child_weights = node.weights[indices]
                        mass = self._host_float(child_weights.sum())
                        child_normalized = child_weights / child_weights.sum()
                        child = self._new_node(
                            next_states[group],
                            node.scenario_ids[indices],
                            child_weights,
                            node.depth + 1,
                            (node_id, action_index),
                        )
                        self._nodes[child].lower = self._host_float(
                            (lower[group] * child_normalized).sum()
                        )
                        self._nodes[child].upper = self._host_float(
                            (upper[group] * child_normalized).sum()
                        )
                        self._nodes[child].terminal = bool(terminal[group].all())
                        branch.observations[key] = ObservationBranch(key, child, mass)
                    node.actions[action_index] = branch
                node.expanded = True
                self._backup_node(node)
        finally:
            self._tree_lock.release()
            self._counts[HypDESPOTMetrics.BACKUP_TIME] += time.perf_counter() - start

    def _host_float(self, value: Tensor) -> float:
        """Copy one scalar needed by the CPU tree and account for that transfer."""
        start = time.perf_counter()
        result = float(value.item())
        self._counts[HypDESPOTMetrics.TRANSFER_TIME] += time.perf_counter() - start
        return result

    def _backup_node(self, node: BeliefNode) -> None:
        if not node.actions:
            return
        node.lower = max(a.lower for a in node.actions.values())
        node.upper = max(a.upper for a in node.actions.values())

    @staticmethod
    def _select_final_action(node: BeliefNode) -> int:
        """Choose only from backed-up lower values, excluding PO-UCT bonuses."""
        if not node.actions:
            raise RuntimeError("HyP-DESPOT root has no backed-up actions")
        return max(node.actions.values(), key=lambda action: action.lower).action_index

    def _backup_path(self, path: List[Tuple[int, int, int]]) -> None:
        start = time.perf_counter()
        self._acquire()
        try:
            for node_id, action_index, _ in reversed(path):
                node = self._nodes[node_id]
                action = node.actions[action_index]
                action.visits += float(node.weights.sum())
                node.visits += float(node.weights.sum())
                if action.observations:
                    mass = sum(o.weight for o in action.observations.values()) or 1.0
                    action.lower = (
                        action.immediate_reward
                        + self.discount_factor
                        * sum(
                            o.weight * self._nodes[o.child].lower
                            for o in action.observations.values()
                        )
                        / mass
                    )
                    action.upper = (
                        action.immediate_reward
                        + self.discount_factor
                        * sum(
                            o.weight * self._nodes[o.child].upper
                            for o in action.observations.values()
                        )
                        / mass
                    )
                self._backup_node(node)
        finally:
            self._tree_lock.release()
            self._counts[HypDESPOTMetrics.BACKUP_TIME] += time.perf_counter() - start

    def action(self, belief: Any) -> Tuple[List[Any], PolicyRunData]:
        validate_cuda_contract(self._model, self._actions, torch.device(self.device))
        self._reset_transient()
        torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        total_start = time.perf_counter()
        states, ids, weights = self._root_particles(belief)
        root = self._new_node(states, ids, weights, 0, None)
        remaining = self.n_traversals
        while remaining:
            count = min(self.cuda_batch_size, remaining)
            traverse_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                selected = list(pool.map(lambda _: self._select_leaf(root), range(count)))
            self._counts[HypDESPOTMetrics.TRAVERSAL_TIME] += time.perf_counter() - traverse_start
            queue_start = time.perf_counter()
            try:
                unique = []
                for leaf, _ in selected:
                    candidate = self._nodes[leaf]
                    if (
                        leaf not in unique
                        and not candidate.expanded
                        and not candidate.terminal
                        and candidate.depth < self.depth
                    ):
                        unique.append(leaf)
                if unique:
                    batch_depth = self._nodes[unique[0]].depth
                    # A model call has one deterministic scenario depth.
                    unique = [leaf for leaf in unique if self._nodes[leaf].depth == batch_depth]
                self._counts[HypDESPOTMetrics.QUEUE_TIME] += time.perf_counter() - queue_start
                if unique:
                    kernel_start = time.perf_counter()
                    result = expand_cuda_leaves(
                        self._model,
                        [self._nodes[x].states for x in unique],
                        [self._nodes[x].scenario_ids for x in unique],
                        len(self._actions),
                        batch_depth,
                        self.depth,
                        self.scenario_seed,
                    )
                    torch.cuda.synchronize(torch.device(self.device))
                    self._counts[HypDESPOTMetrics.KERNEL_TIME] += time.perf_counter() - kernel_start
                    self._integrate(unique, result)
                    self._counts[HypDESPOTMetrics.CUDA_BATCHES] += 1
                    self._counts[HypDESPOTMetrics.CUDA_LEAVES] += len(unique)
                    self._batch_sizes.append(len(unique))
                    self._counts[HypDESPOTMetrics.CUDA_SCENARIOS] += result.rewards.numel() / len(
                        self._actions
                    )
                    self._counts[HypDESPOTMetrics.CUDA_ACTIONS] += result.rewards.numel()
                    self._counts[HypDESPOTMetrics.GPU_DEVICE] = result.kernel_device_index
                for _, path in selected:
                    self._backup_path(path)
            finally:
                for _, path in selected:
                    self._release_virtual_loss(path)
            self._counts[HypDESPOTMetrics.TRAVERSALS] += count
            remaining -= count
        root_node = self._nodes[root]
        if not root_node.actions:
            raise RuntimeError("HyP-DESPOT completed no CUDA leaf expansion")
        chosen = self._select_final_action(root_node)
        self._counts[HypDESPOTMetrics.ROOT_LOWER] = root_node.lower
        self._counts[HypDESPOTMetrics.ROOT_UPPER] = root_node.upper
        self._counts[HypDESPOTMetrics.ROOT_GAP] = root_node.upper - root_node.lower
        self._counts[HypDESPOTMetrics.MEAN_BATCH_SIZE] = sum(self._batch_sizes) / len(
            self._batch_sizes
        )
        self._counts[HypDESPOTMetrics.MAX_BATCH_SIZE] = max(self._batch_sizes)
        self._counts[HypDESPOTMetrics.PEAK_MEMORY] = torch.cuda.max_memory_allocated(
            torch.device(self.device)
        )
        self._counts[HypDESPOTMetrics.TOTAL_TIME] = time.perf_counter() - total_start
        self._last_snapshot = self._snapshot(root, chosen)
        if self.search_state_dump_dir:
            self.search_state_dump_dir.mkdir(parents=True, exist_ok=True)
            path = self.search_state_dump_dir / f"hyp_despot_{time.time_ns()}.json"
            path.write_text(json.dumps(self._last_snapshot, indent=2), encoding="utf-8")
        info = [
            PolicyInfoVariable(metric.value, float(self._counts[metric]))
            for metric in HypDESPOTMetrics
        ]
        return [self._actions[chosen]], PolicyRunData(info)

    def _snapshot(self, root: int, chosen: int) -> dict:
        nodes = []
        for node in self._nodes.values():
            nodes.append(
                {
                    "id": node.node_id,
                    "parent_action": node.parent_action,
                    "depth": node.depth,
                    "scenario_ids": node.scenario_ids.detach().cpu().tolist(),
                    "scenario_weights": node.weights.detach().cpu().tolist(),
                    "visits": node.visits,
                    "lower": node.lower,
                    "upper": node.upper,
                    "expanded": node.expanded,
                    "terminal": node.terminal,
                    "actions": [
                        {
                            "action_index": a.action_index,
                            "visits": a.visits,
                            "lower": a.lower,
                            "upper": a.upper,
                            "immediate_reward": a.immediate_reward,
                            "observations": [
                                {
                                    "key": o.key,
                                    "child": o.child,
                                    "weight": o.weight,
                                    "virtual_loss": o.virtual_loss,
                                }
                                for o in a.observations.values()
                            ],
                        }
                        for a in node.actions.values()
                    ],
                }
            )
        return {
            "root": root,
            "selected_action_index": chosen,
            "nodes": nodes,
            "worker_count": self.n_workers,
            "cuda_batch_sizes": list(self._batch_sizes),
            "metrics": {m.value: float(v) for m, v in self._counts.items()},
        }

    def get_search_snapshot(self) -> Optional[dict]:
        return copy.deepcopy(self._last_snapshot)

    def __getstate__(self):
        """Drop the lock and the tree; keep an explicit model override.

        ``_model`` itself is re-resolved on unpickling rather than stored,
        which avoids a second reference to the same object -- the environment
        is in the state and already carries it. An explicit ``model=``
        argument is different: it is the only record of which model this
        planner was built with, so it has to travel, or the restored planner
        silently binds to whatever the environment happens to expose. The one
        exception is the case where the two are the same object, where storing
        it would pickle the model twice and break the ``is`` relationship on
        the way back.
        """
        state = self.__dict__.copy()
        for key in ("_tree_lock", "_nodes", "_model", "_last_snapshot"):
            state.pop(key, None)
        if state.get("_model_override") is not None and state["_model_override"] is getattr(
            self.environment, "hyp_despot_cuda_model", None
        ):
            state["_model_override"] = None
            state["_model_override_from_environment"] = True
        else:
            state["_model_override_from_environment"] = False
        return state

    def __setstate__(self, state):
        for key, value in dict(state).items():
            setattr(self, key, value)
        self._tree_lock = threading.RLock()
        self._last_snapshot = None
        from_environment = getattr(self, "_model_override_from_environment", False)
        override = getattr(self, "_model_override", None)
        resolved_model = (
            override
            if override is not None
            else getattr(self.environment, "hyp_despot_cuda_model", None)
        )
        if resolved_model is None:
            raise HypDESPOTCompatibilityError(
                "unpickled environment must expose hyp_despot_cuda_model"
            )
        if from_environment:
            self._model_override = resolved_model
        self._model = cast(HypDESPOTCUDAmodel, resolved_model)
        validate_cuda_contract(self._model, self._actions, torch.device(self.device))
        self._reset_transient()


__all__ = [
    "HypDESPOT",
    "HypDESPOTMetrics",
    "HypDESPOTCompatibilityError",
    "BeliefNode",
    "ActionBranch",
    "ObservationBranch",
]
