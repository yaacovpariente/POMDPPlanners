"""Column-store implementation of the MCTS tree.

A single :class:`Tree` instance holds one Python ``list`` per node attribute.
A node is identified by an integer ID — its index into every column. There
are no node objects; operations on the tree are methods of this class that
take ``(node_id, ...)`` and read or write the columns directly.

Use::

    from POMDPPlanners.core.tree.arena import Tree, BELIEF, ACTION

    tree = Tree()
    root_id = tree.add_belief_node(some_belief)
    action_id = tree.add_action_node("up", parent_id=root_id)
    child_id = tree.add_belief_node(next_belief, observation="o", parent_id=action_id)
    tree.visit_count[root_id] += 1

Schema is declared once in ``__init__``. Adding a new per-node attribute
means adding a line to ``__init__`` (the empty column) and one matching
line to ``_allocate`` (the default value at allocation). Direct attribute
access is used in ``_allocate`` rather than a generic ``setattr`` loop
because the per-allocation cost is on the hot path.

This layout follows JuliaPOMDP/POMCPOW.jl: contiguous typed vectors instead
of a graph of Python objects, integer IDs instead of object references,
and a single allocation point so per-node overhead is one append per
column rather than a Python class instantiation plus GC bookkeeping.

In addition to the columns, the tree maintains two per-parent indexes:

* ``children_cdf[parent]`` — cumulative distribution function over child
  weights, in the same order as ``children_ids[parent]``. Maintained
  incrementally on every ``add_*_node``. Enables O(log K) weighted
  sampling via ``sample_belief_child``.
* ``obs_child_lookup`` and ``action_child_lookup`` — ``(parent_id, key) →
  child_id`` dicts populated when the observation/action is hashable.
  Enable O(1) child lookup via ``get_belief_child_indexed`` and
  ``get_action_child_indexed``.

If the user mutates ``weight[id]`` after ``add_belief_node`` the CDF for
the parent becomes stale; call ``recompute_children_cdf(parent_id)`` to
rebuild. Indexed lookup is unavailable for unhashable
observations/actions (e.g. ``np.ndarray``); use the linear-scan
``get_belief_child`` / ``get_action_child`` for those.
"""

import bisect
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from POMDPPlanners.core.belief import Belief
from POMDPPlanners.core.environment import Environment


# Node kind tags.
BELIEF: int = 0
ACTION: int = 1


class Tree:
    """Column-store search tree."""

    # pylint: disable=too-many-instance-attributes
    # The columns are the schema; one attribute per column is the design.

    def __init__(self) -> None:
        # Topology
        self.kind: List[int] = []
        self.parent_id: List[Optional[int]] = []
        self.children_ids: List[List[int]] = []
        # Common metrics
        self.visit_count: List[int] = []
        self.q_value: List[float] = []
        self.v_value: List[float] = []
        self.lower_confidence_bound: List[float] = []
        self.upper_confidence_bound: List[float] = []
        self.immediate_cost: List[Optional[float]] = []
        self.immediate_reward: List[Optional[float]] = []
        self.weight: List[float] = []
        # Per-kind payloads
        self.action: List[Any] = []
        self.observation: List[Any] = []
        self.belief: List[Any] = []
        self.data: List[Any] = []
        self.sample: List[List[Any]] = []
        # Per-parent CDF over children's weights, aligned with children_ids[parent].
        self.children_cdf: List[List[float]] = []
        # Hash-indexed children lookups; populated when the key is hashable.
        self.obs_child_lookup: Dict[Tuple[int, Any], int] = {}
        self.action_child_lookup: Dict[Tuple[int, Any], int] = {}
        # Logical node count. May be less than ``len(self.kind)`` after a
        # call to :meth:`reserve` pre-fills the columns with sentinels.
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def reserve(self, capacity: int) -> None:
        """Pre-allocate ``capacity`` slots in every per-node column.

        Mirrors Julia's ``sizehint!`` / ``Vector{T}(undef, n)``. After this
        call ``len(tree)`` is unchanged, but ``capacity`` slots are
        physically resident in each column so the first ``capacity``
        subsequent allocations write at a cursor instead of triggering
        ``list.append`` (and the periodic O(N) realloc bursts that come
        with it). Useful when the maximum tree size is known in advance
        (e.g., ``2 * n_simulations * depth`` for POMCPOW-style planners).

        Allocations beyond ``capacity`` fall back to ``append`` and grow
        the columns normally. Calling :meth:`reserve` more than once is
        idempotent in the sense that columns are re-padded up to
        ``capacity`` if more headroom is needed; existing entries are
        preserved.
        """
        # Pad every column up to ``capacity`` slots, leaving any existing
        # nodes (positions ``[0, _size)``) untouched. The sentinel value
        # is a placeholder; ``_allocate`` overwrites it when the cursor
        # advances over the slot, so the concrete value never matters at
        # runtime — but we annotate columns as ``Any`` here to avoid
        # variance complaints when extending an int/float-typed list with
        # ``None`` (and vice versa).
        target_len = max(self._size, capacity)
        # Each column's element type is enforced by ``_allocate``'s writes;
        # the sentinel exists only to grow the underlying buffer.
        columns: List[Any] = [
            self.kind,
            self.parent_id,
            self.children_ids,
            self.visit_count,
            self.q_value,
            self.v_value,
            self.lower_confidence_bound,
            self.upper_confidence_bound,
            self.immediate_cost,
            self.immediate_reward,
            self.weight,
            self.action,
            self.observation,
            self.belief,
            self.data,
            self.sample,
            self.children_cdf,
        ]
        for column in columns:
            current_len = len(column)
            if target_len > current_len:
                column.extend([None] * (target_len - current_len))

    def _allocate(self, kind: int, parent_id: Optional[int]) -> int:
        node_id = self._size
        if node_id < len(self.kind):
            # Reserved slot exists — overwrite in place. This path avoids
            # ``list.append``'s amortised growth and the periodic O(N)
            # realloc bursts on every column.
            self.kind[node_id] = kind
            self.parent_id[node_id] = parent_id
            self.children_ids[node_id] = []
            self.visit_count[node_id] = 0
            self.q_value[node_id] = 0.0
            self.v_value[node_id] = 0.0
            self.lower_confidence_bound[node_id] = 0.0
            self.upper_confidence_bound[node_id] = 0.0
            self.immediate_cost[node_id] = None
            self.immediate_reward[node_id] = None
            self.weight[node_id] = 1.0
            self.action[node_id] = None
            self.observation[node_id] = None
            self.belief[node_id] = None
            self.data[node_id] = None
            self.sample[node_id] = []
            self.children_cdf[node_id] = []
        else:
            # Past the reserved zone — fall back to append.
            self.kind.append(kind)
            self.parent_id.append(parent_id)
            self.children_ids.append([])
            self.visit_count.append(0)
            self.q_value.append(0.0)
            self.v_value.append(0.0)
            self.lower_confidence_bound.append(0.0)
            self.upper_confidence_bound.append(0.0)
            self.immediate_cost.append(None)
            self.immediate_reward.append(None)
            self.weight.append(1.0)
            self.action.append(None)
            self.observation.append(None)
            self.belief.append(None)
            self.data.append(None)
            self.sample.append([])
            self.children_cdf.append([])
        self._size += 1
        if parent_id is not None:
            self.children_ids[parent_id].append(node_id)
        return node_id

    # --- construction ---

    def add_belief_node(
        self,
        belief: Belief,
        observation: Any = None,
        weight: float = 1.0,
        parent_id: Optional[int] = None,
        data: Any = None,
        obs_key: Optional[Hashable] = None,
    ) -> int:
        """Allocate a belief node and return its ID.

        If ``parent_id`` is provided, also: (a) extend the parent's CDF by
        ``weight``; (b) register ``(parent_id, key) → node_id`` in
        ``obs_child_lookup`` where ``key`` is the explicit ``obs_key`` if
        provided (caller guarantees hashable), or the raw ``observation``
        (silently dropped if unhashable).
        """
        if not isinstance(belief, Belief):
            # Runtime guard for callers that bypass static typing.
            raise TypeError(
                "belief must be a Belief instance"
            )  # pyright: ignore[reportUnreachable]
        node_id = self._allocate(BELIEF, parent_id)
        self.belief[node_id] = belief
        self.observation[node_id] = observation
        weight_f = float(weight)
        self.weight[node_id] = weight_f
        self.data[node_id] = data
        if parent_id is not None:
            self._extend_cdf(parent_id, weight_f)
            self._register_obs_child(parent_id, observation, obs_key, node_id)
        return node_id

    def _register_obs_child(
        self,
        parent_id: int,
        observation: Any,
        obs_key: Optional[Hashable],
        node_id: int,
    ) -> None:
        if obs_key is not None:
            self.obs_child_lookup[(parent_id, obs_key)] = node_id
            return
        try:
            self.obs_child_lookup[(parent_id, observation)] = node_id
        except TypeError:
            pass  # unhashable observation; rely on linear-scan get_belief_child

    def add_action_node(
        self,
        action: Any,
        parent_id: int,
        data: Any = None,
        action_key: Optional[Hashable] = None,
    ) -> int:
        """Allocate an action node under ``parent_id`` and return its ID.

        Also: (a) extend the parent's CDF by 1.0 (action nodes default to
        unit weight); (b) register ``(parent_id, key) → node_id`` in
        ``action_child_lookup`` where ``key`` is the explicit ``action_key``
        if provided (caller guarantees hashable), or the raw ``action``
        (silently dropped if unhashable).
        """
        node_id = self._allocate(ACTION, parent_id)
        self.action[node_id] = action
        self.data[node_id] = data
        self._extend_cdf(parent_id, 1.0)
        self._register_action_child(parent_id, action, action_key, node_id)
        return node_id

    def _register_action_child(
        self,
        parent_id: int,
        action: Any,
        action_key: Optional[Hashable],
        node_id: int,
    ) -> None:
        if action_key is not None:
            self.action_child_lookup[(parent_id, action_key)] = node_id
            return
        try:
            self.action_child_lookup[(parent_id, action)] = node_id
        except TypeError:
            pass  # unhashable action; rely on linear-scan get_action_child

    def _extend_cdf(self, parent_id: int, weight: float) -> None:
        cdf = self.children_cdf[parent_id]
        prev_total = cdf[-1] if cdf else 0.0
        cdf.append(prev_total + weight)

    def recompute_children_cdf(self, parent_id: int) -> None:
        """Rebuild the parent's CDF from current ``weight`` values.

        Use this when a caller has mutated ``weight[id]`` for one or more
        children after they were added.
        """
        running_total = 0.0
        new_cdf: List[float] = []
        for cid in self.children_ids[parent_id]:
            running_total += self.weight[cid]
            new_cdf.append(running_total)
        self.children_cdf[parent_id] = new_cdf

    def increment_weight(self, child_id: int, delta: float) -> None:
        """Increment ``weight[child_id]`` by ``delta`` and patch the parent's CDF.

        Cheaper than :meth:`recompute_children_cdf` for the common case of a
        single weight bump (e.g. POMCPOW's observation widening): walks the
        parent's CDF only from this child's position onward, adding
        ``delta`` to each remaining entry. O(K - position) where K is the
        number of siblings.

        If ``child_id`` is the root (no parent), only the weight is
        updated; there is no parent CDF to patch.
        """
        self.weight[child_id] += delta
        parent_id = self.parent_id[child_id]
        if parent_id is None:
            return
        siblings = self.children_ids[parent_id]
        cdf = self.children_cdf[parent_id]
        position = siblings.index(child_id)
        for index in range(position, len(cdf)):
            cdf[index] += delta

    # --- queries ---

    def depth(self, node_id: int) -> int:
        """Number of edges from this node to the root."""
        result = 0
        current = self.parent_id[node_id]
        while current is not None:
            result += 1
            current = self.parent_id[current]
        return result

    def get_belief_child(self, action_id: int, observation: Any, env: Environment) -> Optional[int]:
        """Return the ID of the belief child of ``action_id`` matching ``observation``.

        Linear scan using ``env.is_equal_observation``. Use this when the
        observation may not be hashable (e.g. ``np.ndarray``) or when
        equality is custom; otherwise prefer
        :meth:`get_belief_child_indexed` for O(1) lookup.
        """
        for cid in self.children_ids[action_id]:
            if env.is_equal_observation(self.observation[cid], observation):
                return cid
        return None

    def get_belief_child_indexed(
        self,
        action_id: int,
        observation: Any = None,
        obs_key: Optional[Hashable] = None,
    ) -> Optional[int]:
        """O(1) lookup of a belief child of ``action_id`` by hashable key.

        If ``obs_key`` is provided the caller guarantees it is hashable; the
        lookup uses it directly. Otherwise, the raw ``observation`` is tried
        and ``None`` is returned for unhashable values.
        """
        if obs_key is not None:
            return self.obs_child_lookup.get((action_id, obs_key))
        try:
            return self.obs_child_lookup.get((action_id, observation))
        except TypeError:
            return None

    def get_action_child(self, belief_id: int, action: Any) -> Optional[int]:
        """Return the ID of the action child of ``belief_id`` matching ``action``.

        Linear scan; compares numpy arrays by content, everything else by
        ``==``. Use when ``action`` may not be hashable; otherwise prefer
        :meth:`get_action_child_indexed` for O(1) lookup.
        """
        for cid in self.children_ids[belief_id]:
            child_action = self.action[cid]
            if isinstance(child_action, np.ndarray) and isinstance(action, np.ndarray):
                if np.array_equal(child_action, action):
                    return cid
            elif child_action == action:
                return cid
        return None

    def get_action_child_indexed(
        self,
        belief_id: int,
        action: Any = None,
        action_key: Optional[Hashable] = None,
    ) -> Optional[int]:
        """O(1) lookup of an action child of ``belief_id`` by hashable key.

        If ``action_key`` is provided the caller guarantees it is hashable;
        the lookup uses it directly. Otherwise, the raw ``action`` is tried
        and ``None`` is returned for unhashable values.
        """
        if action_key is not None:
            return self.action_child_lookup.get((belief_id, action_key))
        try:
            return self.action_child_lookup.get((belief_id, action))
        except TypeError:
            return None

    def sample_belief_child(self, action_id: int) -> int:
        """Sample one belief child of ``action_id`` proportional to its weight.

        Uses the maintained CDF: O(log K) per sample via ``bisect``. If the
        caller has mutated ``weight[id]`` after the children were added,
        call :meth:`recompute_children_cdf` first to refresh the CDF.
        """
        children = self.children_ids[action_id]
        if not children:
            raise ValueError("no belief children to sample from")
        cdf = self.children_cdf[action_id]
        total = cdf[-1]
        if total <= 0.0:
            raise ValueError("total weight is non-positive; cannot sample")
        target = float(np.random.uniform(0.0, total))
        idx = bisect.bisect_left(cdf, target)
        if idx >= len(children):
            idx = len(children) - 1
        return children[idx]

    # --- mutations ---

    def update_belief(
        self,
        belief_id: int,
        action: Any,
        observation: Any,
        pomdp: Environment,
        **kwargs: Any,
    ) -> None:
        """Replace the belief at ``belief_id`` with the result of ``belief.update``."""
        self.belief[belief_id] = self.belief[belief_id].update(
            action=action, observation=observation, pomdp=pomdp, **kwargs
        )

    def set_immediate_cost(self, node_id: int, value: Optional[float]) -> None:
        """Set ``immediate_cost`` and mirror to ``immediate_reward = -value``."""
        self.immediate_cost[node_id] = value
        if value is not None:
            self.immediate_reward[node_id] = -value

    def set_immediate_reward(self, node_id: int, value: Optional[float]) -> None:
        """Set ``immediate_reward`` and mirror to ``immediate_cost = -value``."""
        self.immediate_reward[node_id] = value
        if value is not None:
            self.immediate_cost[node_id] = -value

    # --- visualisation ---

    def print(self, node_id: int = 0) -> None:
        """Print the subtree rooted at ``node_id`` as an indented tree."""
        self._render(node_id, prefix="")

    def best_action_by_reward(self, belief_id: int) -> Any:
        """Return the action of the highest-q_value action child of ``belief_id``.

        Arena equivalent of :func:`get_optimal_action_reward_setting` from
        ``anytree_based``.
        """
        children = self.children_ids[belief_id]
        if not children:
            raise ValueError("belief node has no action children")
        best_id = max(children, key=lambda cid: self.q_value[cid])
        return self.action[best_id]

    def best_action_by_cost(self, belief_id: int) -> Any:
        """Return the action of the lowest-q_value action child of ``belief_id``.

        Arena equivalent of :func:`get_optimal_action_cost_setting` from
        ``anytree_based``.
        """
        children = self.children_ids[belief_id]
        if not children:
            raise ValueError("belief node has no action children")
        best_id = min(children, key=lambda cid: self.q_value[cid])
        return self.action[best_id]

    def _render(self, node_id: int, prefix: str) -> None:
        label = "BeliefNode" if self.kind[node_id] == BELIEF else "ActionNode"
        payload = (
            f"obs={self.observation[node_id]}"
            if self.kind[node_id] == BELIEF
            else f"action={self.action[node_id]}"
        )
        print(
            f"{prefix}{label}[{node_id}] {payload} "
            f"visits={self.visit_count[node_id]} "
            f"q={self.q_value[node_id]:.3f} v={self.v_value[node_id]:.3f}"
        )
        children = self.children_ids[node_id]
        for index, cid in enumerate(children):
            is_last = index == len(children) - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{prefix}{branch}", end="")
            self._render(cid, next_prefix)
