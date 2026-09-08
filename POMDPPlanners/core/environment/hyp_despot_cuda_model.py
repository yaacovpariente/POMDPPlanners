# SPDX-License-Identifier: MIT
"""Strict deterministic CUDA model protocol used by HyP-DESPOT."""
from typing import Protocol, Tuple, runtime_checkable
import torch
from torch import Tensor


@runtime_checkable
class HypDESPOTCUDAmodel(Protocol):
    """All leaf transition, observation, reward, terminal, and bound kernels."""

    @property
    def device(self) -> torch.device:
        ...

    @property
    def supports_factored_step(self) -> bool:
        ...

    def scenario_randomness(self, scenario_ids: Tensor, depth: int, seed: int) -> Tensor:
        ...

    def sample_next_states(self, states: Tensor, actions: Tensor, random_values: Tensor) -> Tensor:
        ...

    def sample_observations(
        self, next_states: Tensor, actions: Tensor, random_values: Tensor
    ) -> Tensor:
        ...

    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        ...

    def terminal_mask(self, states: Tensor) -> Tensor:
        ...

    def observation_keys(self, observations: Tensor) -> Tensor:
        ...

    def leaf_bounds(self, states: Tensor, remaining_depth: int) -> Tuple[Tensor, Tensor]:
        ...


__all__ = ["HypDESPOTCUDAmodel"]
