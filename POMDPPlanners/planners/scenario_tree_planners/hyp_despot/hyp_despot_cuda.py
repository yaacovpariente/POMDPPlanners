# SPDX-License-Identifier: MIT
"""Contract checks and leaf/action/scenario CUDA batching for HyP-DESPOT."""
from dataclasses import dataclass
from typing import List, Sequence
import torch
from torch import Tensor
from POMDPPlanners.core.environment.hyp_despot_cuda_model import HypDESPOTCUDAmodel


class HypDESPOTCompatibilityError(ValueError):
    """The supplied model cannot perform required HyP-DESPOT CUDA work."""


@dataclass(frozen=True)
class CUDAExpansionResult:
    next_states: Tensor
    observation_keys: Tensor
    rewards: Tensor
    terminal: Tensor
    lower: Tensor
    upper: Tensor
    leaf_indices: Tensor
    action_indices: Tensor
    scenario_indices: Tensor
    kernel_device_index: int


def normalize_cuda_device(device: torch.device | str) -> torch.device:
    """Give a CUDA device an explicit index so ``==`` compares like for like.

    ``torch.device("cuda")`` carries index ``None`` while every tensor's
    ``.device`` carries an explicit ``0``; the two name the same GPU but are
    not equal in PyTorch. Normalizing once at the boundary keeps the contract
    checks from rejecting a valid configuration.

    A bare ``cuda`` resolves to the process's *current* device, not to ``0``,
    so that is what is filled in. Hard-coding ``0`` would let a planner on a
    box where ``torch.cuda.set_device(1)`` has run slip past the single-GPU
    guard below and then fail much later with a message naming the wrong GPU.
    """
    resolved: torch.device = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        try:
            index = int(torch.cuda.current_device())
        except (RuntimeError, AssertionError):
            # No usable CUDA runtime. The availability check rejects this
            # anyway; assume 0 so the comparison stays well defined.
            index = 0
        return torch.device("cuda", index)
    return resolved


def validate_cuda_contract(
    model: HypDESPOTCUDAmodel, actions: Sequence[object], device: torch.device
) -> None:
    device = normalize_cuda_device(device)
    if not torch.cuda.is_available():
        raise HypDESPOTCompatibilityError("HypDESPOT requires CUDA PyTorch and an NVIDIA GPU")
    if torch.version.cuda is None:
        raise HypDESPOTCompatibilityError("PyTorch was built without a supported CUDA runtime")
    if device.type == "mps":
        raise HypDESPOTCompatibilityError("HypDESPOT does not support Apple MPS")
    if device.type != "cuda":
        raise HypDESPOTCompatibilityError("HypDESPOT model.device must be a CUDA device")
    if device.index not in (None, 0):
        raise HypDESPOTCompatibilityError("HypDESPOT supports one GPU only (cuda:0)")
    model_device = getattr(model, "device", None)
    if model_device is None or normalize_cuda_device(model_device) != device:
        raise HypDESPOTCompatibilityError("model.device must exactly match the planner CUDA device")
    if not actions:
        raise HypDESPOTCompatibilityError(
            "HypDESPOT needs an explicit non-empty finite expansion action set"
        )
    if getattr(model, "supports_factored_step", False):
        raise HypDESPOTCompatibilityError(
            "HyP-DESPOT does not support within-step factored CUDA kernels"
        )
    required = (
        "scenario_randomness",
        "sample_next_states",
        "sample_observations",
        "rewards",
        "terminal_mask",
        "observation_keys",
        "leaf_bounds",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise HypDESPOTCompatibilityError("HypDESPOT CUDA model is missing: " + ", ".join(missing))


def _tensor(
    value: object, name: str, device: torch.device, rows: int, dtype: torch.dtype | None = None
) -> Tensor:
    if not isinstance(value, Tensor):
        raise HypDESPOTCompatibilityError(
            f"{name} must be a torch.Tensor; host values are forbidden"
        )
    tensor: Tensor = value
    device = normalize_cuda_device(device)
    if normalize_cuda_device(tensor.device) != device:
        raise HypDESPOTCompatibilityError(
            f"{name} must remain on {device}; hidden host/device copies are forbidden"
        )
    if tensor.shape[0] != rows:
        raise HypDESPOTCompatibilityError(
            f"{name} first dimension must be {rows}, got {tuple(tensor.shape)}"
        )
    if dtype is not None and tensor.dtype != dtype:
        raise HypDESPOTCompatibilityError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    return tensor


def expand_cuda_leaves(
    model: HypDESPOTCUDAmodel,
    leaf_states: List[Tensor],
    leaf_scenario_ids: List[Tensor],
    num_actions: int,
    depth: int,
    horizon: int,
    scenario_seed: int,
) -> CUDAExpansionResult:
    """Evaluate all Cartesian leaf/action/scenario rows in one device batch."""
    if not leaf_states:
        raise ValueError("leaf_states must not be empty")
    device, width = normalize_cuda_device(model.device), leaf_states[0].shape[1]
    states_parts, id_parts, leaf_parts, action_parts, scenario_parts = [], [], [], [], []
    for leaf_index, (states, ids) in enumerate(zip(leaf_states, leaf_scenario_ids)):
        _tensor(states, "leaf states", device, states.shape[0])
        _tensor(ids, "scenario ids", device, states.shape[0], torch.int64)
        if states.dim() != 2 or states.shape[1] != width:
            raise HypDESPOTCompatibilityError(
                "leaf states must share a 2-D [scenario, state] shape"
            )
        count = states.shape[0]
        states_parts.append(states.repeat_interleave(num_actions, 0))
        id_parts.append(ids.repeat_interleave(num_actions))
        leaf_parts.append(
            torch.full((count * num_actions,), leaf_index, device=device, dtype=torch.int64)
        )
        action_parts.append(torch.arange(num_actions, device=device).repeat(count))
        scenario_parts.append(torch.arange(count, device=device).repeat_interleave(num_actions))
    states, ids = torch.cat(states_parts), torch.cat(id_parts)
    leaves, actions, scenarios = (
        torch.cat(leaf_parts),
        torch.cat(action_parts),
        torch.cat(scenario_parts),
    )
    rows = states.shape[0]
    random_values = _tensor(
        model.scenario_randomness(ids, depth, scenario_seed), "scenario randomness", device, rows
    )
    repeated_randomness = _tensor(
        model.scenario_randomness(ids, depth, scenario_seed), "scenario randomness", device, rows
    )
    if not torch.equal(random_values, repeated_randomness):
        raise HypDESPOTCompatibilityError(
            "scenario_randomness must be deterministic for the same ids, depth, and seed"
        )
    next_states = _tensor(
        model.sample_next_states(states, actions, random_values), "next states", device, rows
    )
    observations = _tensor(
        model.sample_observations(next_states, actions, random_values), "observations", device, rows
    )
    rewards = _tensor(model.rewards(states, actions, next_states), "rewards", device, rows)
    terminal = _tensor(model.terminal_mask(next_states), "terminal mask", device, rows, torch.bool)
    keys = _tensor(
        model.observation_keys(observations), "observation keys", device, rows, torch.int64
    )
    repeated_keys = _tensor(
        model.observation_keys(observations), "observation keys", device, rows, torch.int64
    )
    if not torch.equal(keys, repeated_keys):
        raise HypDESPOTCompatibilityError(
            "observation_keys must be stable for identical observation tensors"
        )
    if keys.dim() != 1:
        raise HypDESPOTCompatibilityError("observation keys must be a groupable 1-D int64 tensor")
    lower, upper = model.leaf_bounds(next_states, max(0, horizon - depth - 1))
    lower, upper = _tensor(lower, "leaf lower bounds", device, rows), _tensor(
        upper, "leaf upper bounds", device, rows
    )
    for name, value in (
        ("rewards", rewards),
        ("leaf lower bounds", lower),
        ("leaf upper bounds", upper),
    ):
        if (
            value.dim() != 1
            or not value.dtype.is_floating_point
            or not bool(torch.isfinite(value).all())
        ):
            raise HypDESPOTCompatibilityError(f"{name} must be finite 1-D floating CUDA values")
    if bool((lower > upper).any()):
        raise HypDESPOTCompatibilityError("CUDA leaf lower bounds must not exceed upper bounds")
    return CUDAExpansionResult(
        next_states,
        keys,
        rewards,
        terminal,
        lower,
        upper,
        leaves,
        actions,
        scenarios,
        next_states.device.index or 0,
    )


__all__ = [
    "CUDAExpansionResult",
    "HypDESPOTCompatibilityError",
    "expand_cuda_leaves",
    "normalize_cuda_device",
    "validate_cuda_contract",
]
