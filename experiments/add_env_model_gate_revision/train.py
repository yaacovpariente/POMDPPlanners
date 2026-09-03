# SPDX-License-Identifier: MIT

"""Fit a Gaussian MLP by held-out-monitored maximum likelihood, then export it to numpy."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from experiments.add_env_model_gate_revision.models import LOG_STD_MAX, LOG_STD_MIN, GaussianMLPCore

_MIN_SCALE = 1e-6


def _stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), np.maximum(values.std(axis=0), _MIN_SCALE)


class _GaussianHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        width = input_dim
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, 2 * output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, raw = out[:, : self.output_dim], out[:, self.output_dim :]
        upper = LOG_STD_MAX - nn.functional.softplus(LOG_STD_MAX - raw)
        log_std = LOG_STD_MIN + nn.functional.softplus(upper - LOG_STD_MIN)
        return mean, log_std


def _nll(model: _GaussianHead, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean, log_std = model(x)
    return (0.5 * ((y - mean) / log_std.exp()) ** 2 + log_std + 0.5 * np.log(2 * np.pi)).sum(dim=1).mean()


def fit_gaussian_mlp(
    inputs: np.ndarray,
    targets: np.ndarray,
    holdout_inputs: np.ndarray,
    holdout_targets: np.ndarray,
    hidden_sizes: Sequence[int] = (64, 64),
    max_epochs: int = 300,
    patience: int = 25,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> Tuple[GaussianMLPCore, Dict[str, Any]]:
    """Fit on normalized data; stop when the held-out NLL has not improved for ``patience`` epochs.

    Returns the numpy core at the best held-out epoch and the training record.
    The record says whether the fit stopped on the holdout or hit the cap, which
    the acceptance rules distinguish.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_mean, in_scale = _stats(inputs)
    out_mean, out_scale = _stats(targets)
    x = torch.as_tensor((inputs - in_mean) / in_scale, dtype=torch.float32)
    y = torch.as_tensor((targets - out_mean) / out_scale, dtype=torch.float32)
    xh = torch.as_tensor((holdout_inputs - in_mean) / in_scale, dtype=torch.float32)
    yh = torch.as_tensor((holdout_targets - out_mean) / out_scale, dtype=torch.float32)

    model = _GaussianHead(inputs.shape[1], targets.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_curve: List[float] = []
    holdout_curve: List[float] = []
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_holdout = float("inf")
    best_epoch = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(x), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size])
            optimizer.zero_grad()
            loss = _nll(model, x[idx], y[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            held = float(_nll(model, xh, yh))
        train_curve.append(float(np.mean(losses)))
        holdout_curve.append(held)
        epochs_run = epoch
        if held < best_holdout - 1e-4:
            best_holdout = held
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    layers = [
        (module.weight.detach().numpy().astype(float), module.bias.detach().numpy().astype(float))
        for module in model.net
        if isinstance(module, nn.Linear)
    ]
    core = GaussianMLPCore(layers, in_mean, in_scale, out_mean, out_scale)
    # Held-out NLL in raw units: the normalized loss plus the log of the target scale.
    raw_offset = float(np.sum(np.log(out_scale)))
    record = {
        "hidden_sizes": list(hidden_sizes),
        "epochs_run": epochs_run,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "stopped_early": epochs_run < max_epochs,
        "train_nll_normalized": train_curve,
        "holdout_nll_normalized": holdout_curve,
        "best_holdout_nll_raw": best_holdout + raw_offset,
        "num_train_rows": int(len(inputs)),
        "num_holdout_rows": int(len(holdout_inputs)),
        "seed": seed,
    }
    return core, record
