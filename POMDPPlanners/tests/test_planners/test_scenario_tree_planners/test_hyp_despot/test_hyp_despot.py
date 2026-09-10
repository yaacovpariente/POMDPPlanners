# SPDX-License-Identifier: MIT
"""Deterministic CPU and CUDA correctness checks for HyP-DESPOT."""
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
import torch
from POMDPPlanners.core.environment import SpaceInfo, SpaceType
from POMDPPlanners.planners.scenario_tree_planners.hyp_despot import (
    ActionBranch,
    BeliefNode,
    HypDESPOT,
    ObservationBranch,
)
from POMDPPlanners.planners.scenario_tree_planners.hyp_despot.hyp_despot_cuda import (
    CUDAExpansionResult,
    HypDESPOTCompatibilityError,
    expand_cuda_leaves,
    validate_cuda_contract,
)


class _Model:
    supports_factored_step = False

    def __init__(self, device):
        self.device = torch.device(device)
        self.calls = []

    def scenario_randomness(self, ids, depth, seed):
        self.calls.append("random")
        return (ids + depth + seed).to(torch.float32).unsqueeze(1)

    def sample_next_states(self, states, actions, random_values):
        self.calls.append("transition")
        return states + actions.unsqueeze(1) + random_values * 0.01

    def sample_observations(self, next_states, actions, random_values):
        self.calls.append("observation")
        return next_states[:, :1].floor()

    def rewards(self, states, actions, next_states):
        self.calls.append("reward")
        return next_states[:, 0]

    def terminal_mask(self, states):
        self.calls.append("terminal")
        return states[:, 0] >= 9

    def observation_keys(self, observations):
        self.calls.append("keys")
        return observations[:, 0].to(torch.int64)

    def leaf_bounds(self, states, remaining_depth):
        self.calls.append("bounds")
        lower = states[:, 0] * remaining_depth
        return lower, lower + 1


class _Environment:
    name = "cuda-test"
    config_id = "cuda-test-config"
    space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.DISCRETE)

    def __init__(self):
        self.hyp_despot_cuda_model = _Model("cuda:0")

    def get_actions(self):
        return ["left", "right"]


def test_scenario_po_uct_uses_scenario_weight_and_differs_from_uct():
    weighted = HypDESPOT.scenario_po_uct(2, 16, 0.01, 3, 2)
    ordinary = 3 + 2 * (torch.log(torch.tensor(16.0)) / 2).sqrt().item()
    assert weighted < ordinary
    assert HypDESPOT.scenario_po_uct(0, 16, 1, 0, 1) == float("inf")


@pytest.mark.parametrize(
    "device,message", [("cpu", "CUDA device"), ("mps", "Apple MPS"), ("cuda:1", "one GPU")]
)
def test_contract_rejects_wrong_devices(monkeypatch, device, message):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    with pytest.raises(HypDESPOTCompatibilityError, match=message):
        validate_cuda_contract(_Model(device), [0], torch.device(device))


def test_contract_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(HypDESPOTCompatibilityError, match="requires CUDA PyTorch"):
        validate_cuda_contract(_Model("cpu"), [0], torch.device("cpu"))


def test_contract_rejects_empty_actions_missing_bounds_and_factored(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    model = _Model("cuda:0")
    with pytest.raises(HypDESPOTCompatibilityError, match="finite expansion"):
        validate_cuda_contract(model, [], torch.device("cuda:0"))
    model.supports_factored_step = True
    with pytest.raises(HypDESPOTCompatibilityError, match="factored"):
        validate_cuda_contract(model, [0], torch.device("cuda:0"))
    model.supports_factored_step = False
    model.leaf_bounds = None  # type: ignore[assignment]
    with pytest.raises(HypDESPOTCompatibilityError, match="leaf_bounds"):
        validate_cuda_contract(model, [0], torch.device("cuda:0"))


def test_configuration_identity_pickle_and_transient_reset(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    first = HypDESPOT(_Environment(), 0.9, 3, "hyp", n_scenarios=2, n_traversals=2)
    same = HypDESPOT(_Environment(), 0.9, 3, "hyp", n_scenarios=2, n_traversals=2)
    changed = HypDESPOT(_Environment(), 0.9, 4, "hyp", n_scenarios=2, n_traversals=2)
    first._nodes[9] = object()
    restored = pickle.loads(pickle.dumps(first))
    assert first.config_id == same.config_id != changed.config_id
    assert restored.config_id == first.config_id
    assert restored._nodes == {} and restored.get_search_snapshot() is None


def _bare_planner():
    planner = HypDESPOT.__new__(HypDESPOT)
    planner.virtual_loss = 2
    planner.exploration_constant = 1
    planner.depth = 3
    planner._tree_lock = threading.RLock()
    from POMDPPlanners.planners.scenario_tree_planners.hyp_despot import HypDESPOTMetrics

    planner._counts = {metric: 0.0 for metric in HypDESPOTMetrics}
    states = torch.zeros(1, 1)
    ids = torch.zeros(1, dtype=torch.int64)
    weights = torch.ones(1)
    child = BeliefNode(1, (0, 0), 1, states, ids, weights)
    action = ActionBranch(0, 1, 0, 1, 0, {4: ObservationBranch(4, 1, 1)})
    root = BeliefNode(0, None, 0, states, ids, weights, 1, 0, 1, True, False, {0: action})
    planner._nodes = {0: root, 1: child}
    return planner


def test_virtual_loss_is_applied_and_released():
    planner = _bare_planner()
    leaf, path = planner._select_leaf(0)
    assert leaf == 1 and planner._nodes[0].actions[0].observations[4].virtual_loss == 2
    planner._release_virtual_loss(path)
    assert planner._nodes[0].actions[0].observations[4].virtual_loss == 0


def test_virtual_loss_makes_workers_choose_different_action_branches():
    planner = _bare_planner()
    second = BeliefNode(
        2, (0, 1), 1, torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64), torch.ones(1)
    )
    planner._nodes[2] = second
    planner._nodes[0].actions[1] = ActionBranch(1, 1, 0, 1, 0, {5: ObservationBranch(5, 2, 1)})
    _, first_path = planner._select_leaf(0)
    _, second_path = planner._select_leaf(0)
    assert first_path[0][1] != second_path[0][1]
    planner._release_virtual_loss(first_path)
    planner._release_virtual_loss(second_path)


def test_virtual_loss_is_released_when_selection_raises(monkeypatch):
    planner = _bare_planner()

    class Broken(dict):
        def values(self):
            raise RuntimeError("boom")

    # Fail after a virtual loss by making the selected child expansion access fail.
    original = planner._nodes

    class Nodes(dict):
        def __getitem__(self, key):
            if key == 1:
                raise RuntimeError("boom")
            return super().__getitem__(key)

    planner._nodes = Nodes(original)
    with pytest.raises(RuntimeError, match="boom"):
        planner._select_leaf(0)
    assert original[0].actions[0].observations[4].virtual_loss == 0


def test_locked_backup_is_thread_safe_and_final_choice_ignores_exploration():
    planner = _bare_planner()
    planner.discount_factor = 0.5
    planner._nodes[1].lower = 4
    planner._nodes[1].upper = 6
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: planner._backup_path([(0, 0, 4)]), range(20)))
    action = planner._nodes[0].actions[0]
    assert action.visits == 21 and action.lower == 2 and action.upper == 3


def test_final_choice_uses_backed_up_value_not_exploration_score():
    planner = _bare_planner()
    root = planner._nodes[0]
    root.actions[0].lower = 1
    root.actions[0].visits = 100
    root.actions[1] = ActionBranch(1, visits=0, lower=0, upper=100)
    assert HypDESPOT.scenario_po_uct(0, 100, 1, 0, 1) == float("inf")
    assert HypDESPOT._select_final_action(root) == 0


def test_concurrent_leaf_expansion_is_exactly_once_and_weighted():
    planner = _bare_planner()
    planner._actions = [0, 1]
    planner.discount_factor = 0.5
    planner._nodes = {}
    planner._next_node_id = 0
    root = planner._new_node(
        torch.tensor([[0.0], [0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([0.75, 0.25]),
        0,
        None,
    )
    result = CUDAExpansionResult(
        next_states=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        observation_keys=torch.tensor([7, 8, 7, 8]),
        rewards=torch.tensor([2.0, 10.0, 6.0, 14.0]),
        terminal=torch.tensor([False, False, False, False]),
        lower=torch.tensor([4.0, 8.0, 12.0, 16.0]),
        upper=torch.tensor([6.0, 10.0, 14.0, 18.0]),
        leaf_indices=torch.zeros(4, dtype=torch.int64),
        action_indices=torch.tensor([0, 1, 0, 1]),
        scenario_indices=torch.tensor([0, 0, 1, 1]),
        kernel_device_index=0,
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: planner._integrate([root], result), range(8)))
    assert len(planner._nodes) == 3
    assert len(planner._nodes[root].actions) == 2
    # Action 0 reward = .75*2 + .25*6; lower adds .5*(.75*4 + .25*12).
    assert planner._nodes[root].actions[0].lower == pytest.approx(6.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_batch_has_real_leaf_action_scenario_work_and_masks():
    device = torch.device("cuda:0")
    model = _Model(device)
    states = [
        torch.tensor([[1.0], [9.0]], device=device),
        torch.tensor([[2.0], [3.0]], device=device),
    ]
    ids = [torch.tensor([0, 1], device=device), torch.tensor([2, 3], device=device)]
    result = expand_cuda_leaves(model, states, ids, 3, 0, 3, 7)
    assert result.rewards.shape == (12,)
    assert set(result.leaf_indices.cpu().tolist()) == {0, 1}
    assert set(result.action_indices.cpu().tolist()) == {0, 1, 2}
    assert result.next_states.is_cuda
    assert {"transition", "reward", "bounds"}.issubset(model.calls)
    expected = states[0][0, 0] + 0 + (0 + 7) * 0.01
    assert result.rewards[0].item() == pytest.approx(expected.item(), abs=1e-6)
    assert bool(result.terminal.any())


# ---------------------------------------------------------------------------
# Regression tests for the 2026-09-08 review findings
# ---------------------------------------------------------------------------


class _EnvironmentWithoutModel(_Environment):
    """Environment that carries no ``hyp_despot_cuda_model`` of its own.

    The documented way to plan on such an environment is the explicit
    ``model=`` argument, so this is the fixture that exposes whether the
    planner survives a pickle round trip on that path.
    """

    def __init__(self):  # pylint: disable=super-init-not-called
        pass


class _ForeignModel(_Model):
    """A second, distinguishable model, to catch a silent substitution."""


def test_normalize_cuda_device_matches_indexed_and_bare_cuda():
    """``cuda`` and ``cuda:0`` name the same GPU (review finding 4)."""
    from POMDPPlanners.planners.scenario_tree_planners.hyp_despot.hyp_despot_cuda import (
        normalize_cuda_device,
    )

    assert normalize_cuda_device(torch.device("cuda")) == torch.device("cuda", 0)
    assert normalize_cuda_device("cuda") == normalize_cuda_device(torch.device("cuda:0"))
    # Non-CUDA devices are returned untouched, so the CPU/MPS rejections stand.
    assert normalize_cuda_device(torch.device("cpu")) == torch.device("cpu")
    assert normalize_cuda_device(torch.device("cuda:1")) == torch.device("cuda", 1)


def test_contract_accepts_bare_cuda_planner_device_against_indexed_model(monkeypatch):
    """A ``device="cuda"`` planner and a ``cuda:0`` model are a valid pair."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    validate_cuda_contract(_Model("cuda:0"), [0], torch.device("cuda"))
    validate_cuda_contract(_Model("cuda"), [0], torch.device("cuda:0"))


def test_pickle_keeps_the_explicit_model_when_the_environment_has_none(monkeypatch):
    """An explicit ``model=`` must survive pickling (review finding 3)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    model = _Model("cuda:0")
    planner = HypDESPOT(
        _EnvironmentWithoutModel(), 0.9, 3, "hyp", n_scenarios=2, n_traversals=2, model=model
    )
    restored = pickle.loads(pickle.dumps(planner))
    assert isinstance(restored._model, _Model)
    assert restored._model_override is not None


def test_pickle_does_not_swap_an_explicit_model_for_an_unrelated_environment_one(monkeypatch):
    """The environment's model must not silently replace an explicit one."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    environment = _Environment()  # carries its own, different, model
    explicit = _ForeignModel("cuda:0")
    planner = HypDESPOT(environment, 0.9, 3, "hyp", n_scenarios=2, n_traversals=2, model=explicit)
    assert planner._model is explicit
    restored = pickle.loads(pickle.dumps(planner))
    assert isinstance(restored._model, _ForeignModel), "explicit model was swapped on unpickle"


def test_pickle_still_re_resolves_the_environment_supplied_model(monkeypatch):
    """When the model came from the environment it is re-resolved, not copied."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    environment = _Environment()
    planner = HypDESPOT(
        environment,
        0.9,
        3,
        "hyp",
        n_scenarios=2,
        n_traversals=2,
        model=environment.hyp_despot_cuda_model,
    )
    state = planner.__getstate__()
    assert state["_model_override"] is None, "environment-owned model should not travel"
    restored = pickle.loads(pickle.dumps(planner))
    assert restored._model is restored.environment.hyp_despot_cuda_model


def test_bare_cuda_resolves_to_the_process_current_device_not_hard_coded_zero(monkeypatch):
    """A bare ``cuda`` names the current device, so the one-GPU guard still bites.

    On a box where ``torch.cuda.set_device(1)`` has run, ``torch.device("cuda")``
    means ``cuda:1``. Filling in ``0`` regardless would let that configuration
    pass the "one GPU only" check and then fail later against tensors on the
    other GPU, with a message naming the wrong device.
    """
    from POMDPPlanners.planners.scenario_tree_planners.hyp_despot.hyp_despot_cuda import (
        normalize_cuda_device,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert normalize_cuda_device(torch.device("cuda")) == torch.device("cuda", 1)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    with pytest.raises(HypDESPOTCompatibilityError, match="one GPU"):
        validate_cuda_contract(_Model("cuda:1"), [0], torch.device("cuda"))
