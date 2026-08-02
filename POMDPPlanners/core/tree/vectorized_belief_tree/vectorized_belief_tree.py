# SPDX-License-Identifier: MIT

"""Vectorized belief tree stored in flat PyTorch tensors.

This module implements :class:`VectorizedBeliefTree`, a reusable belief-tree
data structure for online POMDP planning. It captures the tensor tree layout
of GPU-vectorized planners (e.g. VOPP / PORPP): rather than a graph of Python
node objects, the whole tree is a small set of flat, preallocated tensors and
a node is an integer index into every column.

The tree alternates two node kinds::

    belief node --a--> action node --o--> belief node

An action node is uniquely identified by ``(parent_belief_index, action_key)``
and a successor belief node by ``(parent_action_index, observation_key)``.
Both keys are integers; converting continuous actions/observations into
integer keys (binning, hashing, nearest-neighbour, progressive widening) is
the caller's responsibility.

The class stores only topology, integer keys, and statistics. The concrete
belief attached to a belief node lives in an external
:class:`~POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_store.BeliefStore`
keyed by the integer node index. No planning-policy logic (UCB, softmax
action selection, log-sum-exp value backups, PORPP preference updates) lives
here — only the batched structural and statistical primitives such
algorithms are built from.

Example:
    Basic usage on CPU or CUDA::

        >>> import torch
        >>> from POMDPPlanners.core.tree.vectorized_belief_tree import (
        ...     VectorizedBeliefTree,
        ... )
        >>> tree = VectorizedBeliefTree(device=torch.device("cpu"))
        >>> root = tree.root_index
        >>> parents = torch.tensor([root, root, root, root])
        >>> actions = torch.tensor([0, 1, 1, 2])
        >>> action_nodes, created = tree.get_or_create_actions(parents, actions)
        >>> bool(action_nodes[1] == action_nodes[2])  # duplicate pair -> one node
        True
        >>> tree.update_action_statistics(action_nodes, torch.tensor([1.0, 2.0, 3.0, -1.0]))
        >>> observations = torch.tensor([4, 2, 2, 7])
        >>> beliefs, belief_created = tree.get_or_create_beliefs(action_nodes, observations)
        >>> tree.validate()
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree_fields import (
    FieldRegistry,
    FieldSpec,
)
from POMDPPlanners.core.tree.vectorized_belief_tree.vectorized_belief_tree_lookup import (
    csr_children,
    match_pairs,
    scatter_or,
    segment_reduce,
    unique_pairs,
)

_ROOT_INDEX = 0

_RESERVED_BELIEF_NAMES = frozenset({"parent_action", "parent_observation", "depth", "terminal"})
_RESERVED_ACTION_NAMES = frozenset({"parent_belief", "key", "depth", "visit_count", "reward_sum"})


class VectorizedBeliefTree:
    """Flat-tensor belief tree with batched insertion, lookup, and reduction.

    All persistent tree data is stored in preallocated tensors on
    ``self.device``. Belief-node and action-node columns are stored
    separately. Insertion, lookup, and statistic accumulation are batched and
    free of Python per-row loops so the structure runs on a GPU without
    per-simulation synchronization.

    Attributes:
        device: The device every persistent tensor lives on.
        index_dtype: Integer dtype for indices and keys (default ``int64``).
        value_dtype: Floating dtype for real-valued statistics (default
            ``float32``).
        growth_factor: Geometric factor used when a capacity is exceeded.

    Example:
        See the module-level docstring for a runnable example.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    # pylint: disable=attribute-defined-outside-init
    # The flat columns are the schema (one attribute per column) and the
    # batched primitives are the public API; both are the design, not sprawl.
    # Columns are allocated by ``_allocate_columns`` (called from ``__init__``
    # and reused by ``load_state_dict``) rather than assigned inline.

    def __init__(
        self,
        device: Optional[torch.device] = None,
        belief_capacity: int = 1024,
        action_capacity: int = 1024,
        index_dtype: torch.dtype = torch.int64,
        value_dtype: torch.dtype = torch.float32,
        growth_factor: float = 2.0,
    ) -> None:
        """Initialise an empty tree containing only the root belief.

        Args:
            device: Target device; defaults to CPU.
            belief_capacity: Initial preallocated belief-node capacity.
            action_capacity: Initial preallocated action-node capacity.
            index_dtype: Integer dtype for indices, keys, depths, and visits.
            value_dtype: Floating dtype for reward sums and float fields.
            growth_factor: Capacity multiplier on overflow (must exceed 1).

        Raises:
            ValueError: If a capacity is non-positive or ``growth_factor`` is
                not greater than 1.
        """
        if belief_capacity <= 0 or action_capacity <= 0:
            raise ValueError("capacities must be positive")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than 1")
        # Normalise to a concrete, indexed device (e.g. ``cuda`` -> ``cuda:0``)
        # so device-equality checks against materialised tensors succeed.
        self.device = torch.empty(0, device=device).device
        self.index_dtype = index_dtype
        self.value_dtype = value_dtype
        self.growth_factor = growth_factor
        self._belief_capacity = belief_capacity
        self._action_capacity = action_capacity
        self._num_beliefs = 0
        self._num_actions = 0
        self._belief_fields = FieldRegistry(self.device, belief_capacity)
        self._action_fields = FieldRegistry(self.device, action_capacity)
        self._allocate_columns()
        self._init_root()

    # ------------------------------------------------------------------ #
    # Construction / column allocation
    # ------------------------------------------------------------------ #

    def _allocate_columns(self) -> None:
        cap_b = self._belief_capacity
        cap_a = self._action_capacity
        self.belief_parent_action = self._new_int_column(cap_b, -1)
        self.belief_parent_observation = self._new_int_column(cap_b, -1)
        self.belief_depth = self._new_int_column(cap_b, 0)
        self.belief_terminal = torch.zeros(cap_b, dtype=torch.bool, device=self.device)
        self.action_parent_belief = self._new_int_column(cap_a, -1)
        self.action_key = self._new_int_column(cap_a, -1)
        self.action_depth = self._new_int_column(cap_a, 0)
        self.action_visit_count = self._new_int_column(cap_a, 0)
        self.action_reward_sum = torch.zeros(cap_a, dtype=self.value_dtype, device=self.device)

    def _new_int_column(self, capacity: int, fill: int) -> Tensor:
        return torch.full((capacity,), fill, dtype=self.index_dtype, device=self.device)

    def _init_root(self) -> None:
        self._num_beliefs = 1
        self.belief_parent_action[_ROOT_INDEX] = -1
        self.belief_parent_observation[_ROOT_INDEX] = -1
        self.belief_depth[_ROOT_INDEX] = 0
        self.belief_terminal[_ROOT_INDEX] = False
        self._belief_fields.reset_rows(0, 1)

    # ------------------------------------------------------------------ #
    # Simple accessors
    # ------------------------------------------------------------------ #

    @property
    def root_index(self) -> int:
        """Index of the root belief node (always ``0``)."""
        return _ROOT_INDEX

    @property
    def num_belief_nodes(self) -> int:
        """Number of active belief nodes."""
        return self._num_beliefs

    @property
    def num_action_nodes(self) -> int:
        """Number of active action nodes."""
        return self._num_actions

    @property
    def belief_capacity(self) -> int:
        """Current preallocated belief-node capacity."""
        return self._belief_capacity

    @property
    def action_capacity(self) -> int:
        """Current preallocated action-node capacity."""
        return self._action_capacity

    # ------------------------------------------------------------------ #
    # Action-node lookup / insertion
    # ------------------------------------------------------------------ #

    def find_actions(
        self,
        parent_belief_indices: Tensor,
        action_keys: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Look up action nodes by ``(parent_belief, action_key)`` without mutating.

        Args:
            parent_belief_indices: ``[batch]`` parent belief indices.
            action_keys: ``[batch]`` integer action keys.

        Returns:
            ``(action_indices, found_mask)``. Missing entries hold ``-1`` in
            ``action_indices`` and ``False`` in ``found_mask``. Order follows
            the input batch.
        """
        self._validate_pair(parent_belief_indices, action_keys, "parent_belief_indices")
        matched = match_pairs(
            self.action_parent_belief[: self._num_actions],
            self.action_key[: self._num_actions],
            parent_belief_indices,
            action_keys,
        )
        return matched, matched >= 0

    def get_or_create_actions(
        self,
        parent_belief_indices: Tensor,
        action_keys: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return one action-node index per input row, creating missing nodes.

        Duplicate ``(parent_belief, action_key)`` pairs — whether already in
        the tree or repeated within the batch — resolve to a single node.

        Args:
            parent_belief_indices: ``[batch]`` parent belief indices.
            action_keys: ``[batch]`` integer action keys.

        Returns:
            ``(action_indices, created_mask)`` in input order. ``created_mask``
            is ``True`` for rows whose node did not exist before this call.
        """
        self._validate_pair(parent_belief_indices, action_keys, "parent_belief_indices")
        matched = match_pairs(
            self.action_parent_belief[: self._num_actions],
            self.action_key[: self._num_actions],
            parent_belief_indices,
            action_keys,
        )
        result = matched.clone()
        created_mask = matched < 0
        new_parents = parent_belief_indices[created_mask]
        new_keys = action_keys[created_mask]
        unique_parents, unique_keys, inverse = unique_pairs(new_parents, new_keys)
        new_ids = self._append_actions(unique_parents, unique_keys)
        result[created_mask] = new_ids[inverse]
        return result, created_mask

    def _append_actions(self, parent_beliefs: Tensor, action_keys: Tensor) -> Tensor:
        count = parent_beliefs.shape[0]
        start = self._num_actions
        end = start + count
        self._ensure_action_capacity(end)
        self.action_parent_belief[start:end] = parent_beliefs.to(self.index_dtype)
        self.action_key[start:end] = action_keys.to(self.index_dtype)
        self.action_depth[start:end] = self.belief_depth[parent_beliefs]
        self.action_visit_count[start:end] = 0
        self.action_reward_sum[start:end] = 0
        self._action_fields.reset_rows(start, end)
        self._num_actions = end
        return torch.arange(start, end, dtype=self.index_dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Belief-node lookup / insertion
    # ------------------------------------------------------------------ #

    def find_beliefs(
        self,
        parent_action_indices: Tensor,
        observation_keys: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Look up belief nodes by ``(parent_action, observation_key)``.

        Args:
            parent_action_indices: ``[batch]`` parent action indices.
            observation_keys: ``[batch]`` integer observation keys.

        Returns:
            ``(belief_indices, found_mask)`` with ``-1`` for missing entries,
            in input order.
        """
        self._validate_pair(parent_action_indices, observation_keys, "parent_action_indices")
        matched = match_pairs(
            self.belief_parent_action[: self._num_beliefs],
            self.belief_parent_observation[: self._num_beliefs],
            parent_action_indices,
            observation_keys,
        )
        return matched, matched >= 0

    def get_or_create_beliefs(
        self,
        parent_action_indices: Tensor,
        observation_keys: Tensor,
        *,
        terminal_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return one successor-belief index per input row, creating missing nodes.

        Duplicate ``(parent_action, observation_key)`` pairs resolve to a
        single node. A new belief's depth is its parent action's depth plus
        one.

        Terminal handling: if ``terminal_mask`` is supplied, terminal flags are
        combined with logical OR — both when several batch rows map to the same
        new node and when a row targets a belief that already exists (an
        existing ``False`` flag can be promoted to ``True``, never the reverse).

        Args:
            parent_action_indices: ``[batch]`` parent action indices.
            observation_keys: ``[batch]`` integer observation keys.
            terminal_mask: Optional ``[batch]`` boolean terminal flags.

        Returns:
            ``(belief_indices, created_mask)`` in input order.
        """
        self._validate_pair(parent_action_indices, observation_keys, "parent_action_indices")
        if terminal_mask is not None:
            self._validate_bool(terminal_mask, parent_action_indices, "terminal_mask")
        matched = match_pairs(
            self.belief_parent_action[: self._num_beliefs],
            self.belief_parent_observation[: self._num_beliefs],
            parent_action_indices,
            observation_keys,
        )
        return self._resolve_beliefs(
            parent_action_indices, observation_keys, matched, terminal_mask
        )

    def _resolve_beliefs(
        self,
        parent_actions: Tensor,
        observation_keys: Tensor,
        matched: Tensor,
        terminal_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        result = matched.clone()
        created_mask = matched < 0
        if terminal_mask is not None:
            found_mask = ~created_mask
            self.belief_terminal |= scatter_or(
                matched[found_mask], terminal_mask[found_mask], self._belief_capacity
            )
        unique_parents, unique_obs, inverse = unique_pairs(
            parent_actions[created_mask], observation_keys[created_mask]
        )
        new_ids = self._append_beliefs(unique_parents, unique_obs)
        result[created_mask] = new_ids[inverse]
        if terminal_mask is not None:
            new_terminal = scatter_or(inverse, terminal_mask[created_mask], unique_parents.shape[0])
            self.belief_terminal[new_ids] = self.belief_terminal[new_ids] | new_terminal
        return result, created_mask

    def _append_beliefs(self, parent_actions: Tensor, observation_keys: Tensor) -> Tensor:
        count = parent_actions.shape[0]
        start = self._num_beliefs
        end = start + count
        self._ensure_belief_capacity(end)
        self.belief_parent_action[start:end] = parent_actions.to(self.index_dtype)
        self.belief_parent_observation[start:end] = observation_keys.to(self.index_dtype)
        self.belief_depth[start:end] = self.action_depth[parent_actions] + 1
        self.belief_terminal[start:end] = False
        self._belief_fields.reset_rows(start, end)
        self._num_beliefs = end
        return torch.arange(start, end, dtype=self.index_dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Statistic accumulation
    # ------------------------------------------------------------------ #

    def increment_action_visits(
        self,
        action_indices: Tensor,
        counts: Optional[Tensor] = None,
    ) -> None:
        """Add visit counts to action nodes, summing duplicate indices.

        Args:
            action_indices: ``[batch]`` action node indices (may repeat).
            counts: Optional ``[batch]`` integer increments; defaults to one.
        """
        self._validate_index(action_indices, "action_indices")
        if counts is None:
            counts = torch.ones_like(action_indices)
        else:
            self._validate_same_length(counts, action_indices, "counts")
        self.action_visit_count.index_add_(0, action_indices, counts.to(self.index_dtype))

    def add_action_rewards(self, action_indices: Tensor, rewards: Tensor) -> None:
        """Add rewards to action nodes' reward sums, summing duplicate indices.

        Args:
            action_indices: ``[batch]`` action node indices (may repeat).
            rewards: ``[batch]`` real-valued rewards.
        """
        self._validate_index(action_indices, "action_indices")
        self._validate_same_length(rewards, action_indices, "rewards")
        self.action_reward_sum.index_add_(0, action_indices, rewards.to(self.value_dtype))

    def update_action_statistics(
        self,
        action_indices: Tensor,
        rewards: Tensor,
        visit_weights: Optional[Tensor] = None,
    ) -> None:
        """Accumulate reward sums and visit counts for action nodes in one call.

        Args:
            action_indices: ``[batch]`` action node indices (may repeat).
            rewards: ``[batch]`` real-valued rewards.
            visit_weights: Optional ``[batch]`` integer visit increments;
                defaults to one per row.
        """
        self.add_action_rewards(action_indices, rewards)
        self.increment_action_visits(action_indices, visit_weights)

    # ------------------------------------------------------------------ #
    # Registered fields
    # ------------------------------------------------------------------ #

    def register_belief_field(
        self,
        name: str,
        shape: Tuple[int, ...] = (),
        *,
        dtype: Optional[torch.dtype] = None,
        default: float = 0,
    ) -> None:
        """Register an extra per-belief-node field backed by its own tensor.

        Args:
            name: Field name; must not shadow a built-in belief column.
            shape: Trailing per-node shape (``()`` for scalar).
            dtype: Storage dtype; defaults to ``value_dtype``.
            default: Value used to initialise new rows.
        """
        spec = FieldSpec(name, tuple(shape), dtype or self.value_dtype, default)
        self._belief_fields.register(spec, _RESERVED_BELIEF_NAMES)

    def register_action_field(
        self,
        name: str,
        shape: Tuple[int, ...] = (),
        *,
        dtype: Optional[torch.dtype] = None,
        default: float = 0,
    ) -> None:
        """Register an extra per-action-node field backed by its own tensor.

        Args:
            name: Field name; must not shadow a built-in action column.
            shape: Trailing per-node shape (``()`` for scalar).
            dtype: Storage dtype; defaults to ``value_dtype``.
            default: Value used to initialise new rows.
        """
        spec = FieldSpec(name, tuple(shape), dtype or self.value_dtype, default)
        self._action_fields.register(spec, _RESERVED_ACTION_NAMES)

    def belief_field(self, name: str) -> Tensor:
        """Return the active-rows view of a registered belief field.

        The returned view spans ``[0, num_belief_nodes)`` and is valid until
        the next structural mutation (which may reallocate the backing tensor).
        """
        return self._belief_fields.tensor(name)[: self._num_beliefs]

    def action_field(self, name: str) -> Tensor:
        """Return the active-rows view of a registered action field.

        The returned view spans ``[0, num_action_nodes)`` and is valid until
        the next structural mutation (which may reallocate the backing tensor).
        """
        return self._action_fields.tensor(name)[: self._num_actions]

    # ------------------------------------------------------------------ #
    # Traversal and indexing
    # ------------------------------------------------------------------ #

    def belief_nodes_at_depth(self, depth: int) -> Tensor:
        """Return the indices of every active belief node at ``depth``."""
        active = self.belief_depth[: self._num_beliefs]
        return torch.nonzero(active == depth, as_tuple=False).flatten()

    def action_nodes_at_depth(self, depth: int) -> Tensor:
        """Return the indices of every active action node at ``depth``."""
        active = self.action_depth[: self._num_actions]
        return torch.nonzero(active == depth, as_tuple=False).flatten()

    def parent_actions(self, belief_indices: Tensor) -> Tensor:
        """Return the parent action index of each queried belief node.

        The root belief maps to ``-1``.
        """
        self._validate_index(belief_indices, "belief_indices")
        return self.belief_parent_action[belief_indices]

    def parent_beliefs(self, action_indices: Tensor) -> Tensor:
        """Return the parent belief index of each queried action node."""
        self._validate_index(action_indices, "action_indices")
        return self.action_parent_belief[action_indices]

    def action_children(self, belief_indices: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the action children of the queried belief nodes in CSR form.

        Args:
            belief_indices: ``[M]`` parent belief node indices.

        Returns:
            ``(flat_child_indices, offsets)`` where the children of
            ``belief_indices[m]`` are ``flat_child_indices[offsets[m]:offsets[m + 1]]``.
        """
        self._validate_index(belief_indices, "belief_indices")
        return csr_children(self.action_parent_belief[: self._num_actions], belief_indices)

    def belief_children(self, action_indices: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the belief children of the queried action nodes in CSR form.

        Args:
            action_indices: ``[M]`` parent action node indices.

        Returns:
            ``(flat_child_indices, offsets)`` where the children of
            ``action_indices[m]`` are ``flat_child_indices[offsets[m]:offsets[m + 1]]``.
        """
        self._validate_index(action_indices, "action_indices")
        return csr_children(self.belief_parent_action[: self._num_beliefs], action_indices)

    def reduce_belief_children(
        self,
        parent_action_indices: Tensor,
        child_values: Tensor,
        *,
        reduction: str,
    ) -> Tensor:
        """Reduce child values grouped by parent action node.

        This is a generic segmented reduction — it does not implement any
        algorithm-specific backup. ``parent_action_indices[k]`` is the parent
        action node of the ``k``-th child and ``child_values[k]`` its value.

        Args:
            parent_action_indices: ``[K]`` parent action indices per child.
            child_values: ``[K]`` per-child values.
            reduction: One of ``"sum"``, ``"mean"``, ``"max"``, ``"min"``.

        Returns:
            ``[num_action_nodes]`` reduced value per action node. Action nodes
            with no contributing child hold ``0``.
        """
        self._validate_index(parent_action_indices, "parent_action_indices")
        self._validate_same_length(child_values, parent_action_indices, "child_values")
        return segment_reduce(parent_action_indices, child_values, self._num_actions, reduction)

    # ------------------------------------------------------------------ #
    # Capacity management
    # ------------------------------------------------------------------ #

    def _ensure_belief_capacity(self, required: int) -> None:
        if required <= self._belief_capacity:
            return
        new_capacity = self._grown_capacity(self._belief_capacity, required)
        self.belief_parent_action = self._grow_column(self.belief_parent_action, new_capacity, -1)
        self.belief_parent_observation = self._grow_column(
            self.belief_parent_observation, new_capacity, -1
        )
        self.belief_depth = self._grow_column(self.belief_depth, new_capacity, 0)
        self.belief_terminal = self._grow_column(self.belief_terminal, new_capacity, False)
        self._belief_fields.grow(new_capacity)
        self._belief_capacity = new_capacity

    def _ensure_action_capacity(self, required: int) -> None:
        if required <= self._action_capacity:
            return
        new_capacity = self._grown_capacity(self._action_capacity, required)
        self.action_parent_belief = self._grow_column(self.action_parent_belief, new_capacity, -1)
        self.action_key = self._grow_column(self.action_key, new_capacity, -1)
        self.action_depth = self._grow_column(self.action_depth, new_capacity, 0)
        self.action_visit_count = self._grow_column(self.action_visit_count, new_capacity, 0)
        self.action_reward_sum = self._grow_column(self.action_reward_sum, new_capacity, 0)
        self._action_fields.grow(new_capacity)
        self._action_capacity = new_capacity

    def _grown_capacity(self, current: int, required: int) -> int:
        new_capacity = current
        while new_capacity < required:
            new_capacity = int(new_capacity * self.growth_factor) + 1
        return new_capacity

    def _grow_column(self, column: Tensor, new_capacity: int, fill: float) -> Tensor:
        grown = torch.full(
            (new_capacity, *column.shape[1:]),
            fill,
            dtype=column.dtype,
            device=self.device,
        )
        grown[: column.shape[0]] = column
        return grown

    # ------------------------------------------------------------------ #
    # Reset / clear
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Remove every node except the root, preserving capacity and fields.

        Registered field definitions survive; their rows reset to defaults.
        """
        self._num_actions = 0
        self._init_root()

    def reset_statistics(self) -> None:
        """Reset visit counts, reward sums, and registered fields; keep topology."""
        self.action_visit_count[: self._num_actions] = 0
        self.action_reward_sum[: self._num_actions] = 0
        self._belief_fields.reset_rows(0, self._num_beliefs)
        self._action_fields.reset_rows(0, self._num_actions)

    # ------------------------------------------------------------------ #
    # Serialization / device movement
    # ------------------------------------------------------------------ #

    def to(self, device: torch.device) -> "VectorizedBeliefTree":
        """Move every built-in and registered tensor to ``device`` in place.

        Returns:
            ``self``, for chaining.
        """
        self.device = torch.empty(0, device=device).device
        for name, tensor in self._named_columns().items():
            setattr(self, name, tensor.to(self.device))
        self._belief_fields.to(self.device)
        self._action_fields.to(self.device)
        return self

    def state_dict(self) -> Dict[str, object]:
        """Serialise the active tree (topology, statistics, and fields).

        Returns:
            A picklable dict holding the active tensor rows, field
            definitions, node counters, and reconstruction configuration.
        """
        return {
            "config": self._config_dict(),
            "num_beliefs": self._num_beliefs,
            "num_actions": self._num_actions,
            "columns": {
                name: tensor[: self._active_len(name)].clone()
                for name, tensor in self._named_columns().items()
            },
            "belief_fields": self._field_state(self._belief_fields, self._num_beliefs),
            "action_fields": self._field_state(self._action_fields, self._num_actions),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore the tree from a :meth:`state_dict` payload."""
        config = _as_dict(state["config"])
        self.index_dtype = _as_dtype(config["index_dtype"])
        self.value_dtype = _as_dtype(config["value_dtype"])
        self.growth_factor = _as_float(config["growth_factor"])
        self._num_beliefs = int(_as_int(state["num_beliefs"]))
        self._num_actions = int(_as_int(state["num_actions"]))
        self._belief_capacity = max(self._num_beliefs, 1)
        self._action_capacity = max(self._num_actions, 1)
        self._allocate_columns()
        self._restore_columns(_as_dict(state["columns"]))
        self._belief_fields = self._restore_fields(
            _as_dict(state["belief_fields"]), self._belief_capacity
        )
        self._action_fields = self._restore_fields(
            _as_dict(state["action_fields"]), self._action_capacity
        )

    def _restore_columns(self, columns: Dict[str, object]) -> None:
        for name, tensor in columns.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"column '{name}' must be a tensor")
            moved = tensor.to(self.device)
            getattr(self, name)[: moved.shape[0]] = moved

    def _restore_fields(self, field_state: Dict[str, object], capacity: int) -> FieldRegistry:
        registry = FieldRegistry(self.device, capacity)
        specs = field_state["specs"]
        values = _as_dict(field_state["values"])
        if not isinstance(specs, (list, tuple)):
            raise TypeError("field specs must be a sequence")
        for spec in specs:
            if not isinstance(spec, FieldSpec):
                raise TypeError("field spec must be a FieldSpec")
            registry.register(spec, frozenset())
            stored = values[spec.name]
            if not isinstance(stored, Tensor):
                raise TypeError(f"field '{spec.name}' must be a tensor")
            registry.tensor(spec.name)[: stored.shape[0]] = stored.to(self.device)
        return registry

    # ------------------------------------------------------------------ #
    # Validation of invariants
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Assert the structural invariants of the tree.

        Raises:
            AssertionError: If any invariant is violated.
        """
        self._validate_root()
        self._validate_parent_links()
        self._validate_depths()
        self._validate_unique_pairs()
        self._validate_capacities()
        self._validate_devices()

    def _validate_root(self) -> None:
        assert self._num_beliefs >= 1, "tree must contain the root belief"
        assert bool(self.belief_parent_action[_ROOT_INDEX] == -1), "root has no parent action"
        assert int(self.belief_depth[_ROOT_INDEX]) == 0, "root depth must be 0"

    def _validate_parent_links(self) -> None:
        if self._num_beliefs > 1:
            parents = self.belief_parent_action[1 : self._num_beliefs]
            assert bool((parents >= 0).all()), "non-root belief has no parent action"
            assert bool((parents < self._num_actions).all()), "belief parent action out of range"
        if self._num_actions > 0:
            parents = self.action_parent_belief[: self._num_actions]
            assert bool((parents >= 0).all()), "action node has no parent belief"
            assert bool((parents < self._num_beliefs).all()), "action parent belief out of range"

    def _validate_depths(self) -> None:
        if self._num_actions > 0:
            parents = self.action_parent_belief[: self._num_actions]
            expected = self.belief_depth[parents]
            assert bool(
                (self.action_depth[: self._num_actions] == expected).all()
            ), "action depth must equal parent belief depth"
        if self._num_beliefs > 1:
            parents = self.belief_parent_action[1 : self._num_beliefs]
            expected = self.action_depth[parents] + 1
            assert bool(
                (self.belief_depth[1 : self._num_beliefs] == expected).all()
            ), "belief depth must equal parent action depth plus one"

    def _validate_unique_pairs(self) -> None:
        self._assert_unique(
            self.action_parent_belief[: self._num_actions],
            self.action_key[: self._num_actions],
            "duplicate (belief, action) pair",
        )
        self._assert_unique(
            self.belief_parent_action[1 : self._num_beliefs],
            self.belief_parent_observation[1 : self._num_beliefs],
            "duplicate (action, observation) pair",
        )

    @staticmethod
    def _assert_unique(first: Tensor, second: Tensor, message: str) -> None:
        if first.shape[0] == 0:
            return
        pairs = torch.stack([first, second], dim=1)
        unique_tensor = torch.unique(pairs, dim=0)
        assert unique_tensor.shape[0] == first.shape[0], message

    def _validate_capacities(self) -> None:
        assert self._num_beliefs <= self._belief_capacity, "belief count exceeds capacity"
        assert self._num_actions <= self._action_capacity, "action count exceeds capacity"

    def _validate_devices(self) -> None:
        for name, tensor in self._named_columns().items():
            assert tensor.device == self.device, f"column '{name}' is on the wrong device"

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _named_columns(self) -> Dict[str, Tensor]:
        return {
            "belief_parent_action": self.belief_parent_action,
            "belief_parent_observation": self.belief_parent_observation,
            "belief_depth": self.belief_depth,
            "belief_terminal": self.belief_terminal,
            "action_parent_belief": self.action_parent_belief,
            "action_key": self.action_key,
            "action_depth": self.action_depth,
            "action_visit_count": self.action_visit_count,
            "action_reward_sum": self.action_reward_sum,
        }

    def _active_len(self, column_name: str) -> int:
        return self._num_actions if column_name.startswith("action") else self._num_beliefs

    def _config_dict(self) -> Dict[str, object]:
        return {
            "index_dtype": self.index_dtype,
            "value_dtype": self.value_dtype,
            "growth_factor": self.growth_factor,
        }

    @staticmethod
    def _field_state(registry: FieldRegistry, active: int) -> Dict[str, object]:
        return {
            "specs": list(registry.specs()),
            "values": {name: registry.tensor(name)[:active].clone() for name in registry.names()},
        }

    def _validate_pair(self, first: Tensor, second: Tensor, first_name: str) -> None:
        self._validate_index(first, first_name)
        self._validate_index(second, "keys")
        self._validate_same_length(second, first, "keys")

    def _validate_index(self, tensor: Tensor, name: str) -> None:
        if not isinstance(tensor, Tensor):
            # Runtime guard for callers that bypass static typing.
            raise TypeError(f"{name} must be a torch.Tensor")  # pyright: ignore[reportUnreachable]
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on device {self.device}, got {tensor.device}")
        if tensor.dim() != 1:
            raise ValueError(f"{name} must be 1-dimensional")
        if not _is_integer_dtype(tensor.dtype):
            raise TypeError(f"{name} must have an integer dtype")

    def _validate_bool(self, tensor: Tensor, reference: Tensor, name: str) -> None:
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on device {self.device}")
        if tensor.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean tensor")
        self._validate_same_length(tensor, reference, name)

    @staticmethod
    def _validate_same_length(tensor: Tensor, reference: Tensor, name: str) -> None:
        if tensor.shape[0] != reference.shape[0]:
            raise ValueError(f"{name} must have the same batch size as its companion tensor")


def _is_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _as_dict(value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a dict in the serialized state")
    return value


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("expected an int in the serialized state")
    return value


def _as_dtype(value: object) -> torch.dtype:
    if not isinstance(value, torch.dtype):
        raise TypeError("expected a torch.dtype in the serialized state")
    return value


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("expected a float in the serialized state")
    return float(value)
