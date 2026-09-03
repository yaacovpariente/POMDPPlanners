# SPDX-License-Identifier: MIT

"""Fit the learned model class on the batch-parity suite's synthetic linear system.

The tracked suite hardcodes a 3-D state and a 2-D action; the LightDark
instance is 2-D, so the *class* joins the suite through this fit and the
LightDark *instances* are checked by the local copy of the same assertions.
"""

from typing import Any

import numpy as np

from experiments.add_env_model_gate_revision.models import GaussianMLPTransition
from experiments.add_env_model_gate_revision.train import fit_gaussian_mlp


def fit_synthetic_transition(dim: int = 3, action_dim: int = 2, seed: int = 11) -> Any:
    """Same generating system as ``test_batched_model_parity._fitted_ensemble``."""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(600, dim))
    actions = rng.normal(size=(600, action_dim))
    next_states = states * 0.9 + actions @ np.ones((action_dim, dim)) * 0.1
    next_states = next_states + rng.normal(scale=0.03, size=(600, dim))
    inputs = np.concatenate([states, actions], axis=1)
    core, record = fit_gaussian_mlp(
        inputs[:480], (next_states - states)[:480], inputs[480:], (next_states - states)[480:],
        hidden_sizes=(32, 32), max_epochs=15, patience=15, seed=0,
    )
    model = GaussianMLPTransition(core, record)
    model.dim = dim
    model.action_dim = action_dim
    return model
