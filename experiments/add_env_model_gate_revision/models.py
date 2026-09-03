# SPDX-License-Identifier: MIT

"""The three transition candidates and the two observation candidates for LightDark.

Truth and bad are the package's own ``LinearGaussianTransition`` with
``next = state + action + N(0, Sigma)``; bad doubles the noise std. The learned
pair are small Gaussian MLPs whose forward pass runs in numpy, so the planner's
scalar calls do not pay torch overhead; torch is used only to fit them.
"""

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from POMDPPlanners.core.environment import TransitionModel
from POMDPPlanners.core.environment.array_backend import as_backend, as_rows, is_tensor
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
    LinearGaussianTransition,
)

DIM = 2
ACTION_DIM = 2
LOG_STD_MIN = -6.0
LOG_STD_MAX = 1.0
_LOG_2PI = math.log(2.0 * math.pi)


def truth_transition(covariance: np.ndarray) -> LinearGaussianTransition:
    """LightDark's own transition: ``next = state + action + N(0, covariance)``."""
    return LinearGaussianTransition(
        weight_state=np.eye(DIM),
        weight_action=np.eye(ACTION_DIM),
        bias=np.zeros(DIM),
        covariance=np.asarray(covariance, dtype=float),
    )


def bad_transition(covariance: np.ndarray, std_factor: float = 2.0) -> LinearGaussianTransition:
    """The truth with its noise standard deviation multiplied by ``std_factor``."""
    return truth_transition(np.asarray(covariance, dtype=float) * std_factor**2)


def _to_numpy(value: Any) -> np.ndarray:
    if is_tensor(value):
        return value.detach().cpu().numpy().astype(float)
    return np.asarray(value, dtype=float)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def bounded_log_std(raw: np.ndarray) -> np.ndarray:
    """Squeeze a raw log-std between the fixed bounds, smoothly (numpy)."""
    upper = LOG_STD_MAX - _softplus(LOG_STD_MAX - raw)
    return LOG_STD_MIN + _softplus(upper - LOG_STD_MIN)


class GaussianMLPCore:
    """A fitted MLP with a Gaussian head, evaluated in numpy.

    ``layers`` are ``(weight, bias)`` pairs with weight shaped ``(out, in)`` as
    torch stores them; SiLU between layers; the last layer emits
    ``[mean, raw_log_std]`` in normalized target units.
    """

    def __init__(
        self,
        layers: Sequence[Tuple[np.ndarray, np.ndarray]],
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
    ) -> None:
        self.layers = [(np.asarray(w, float), np.asarray(b, float)) for w, b in layers]
        self.input_mean = np.asarray(input_mean, float)
        self.input_scale = np.asarray(input_scale, float)
        self.target_mean = np.asarray(target_mean, float)
        self.target_scale = np.asarray(target_scale, float)
        self.output_dim = int(self.target_mean.size)

    @property
    def hidden_sizes(self) -> Tuple[int, ...]:
        return tuple(int(w.shape[0]) for w, _ in self.layers[:-1])

    def predict(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and std of the target in raw units, one row per input row."""
        x = (np.atleast_2d(inputs) - self.input_mean) / self.input_scale
        for weight, bias in self.layers[:-1]:
            x = x @ weight.T + bias
            x = x / (1.0 + np.exp(-x))  # SiLU
        weight, bias = self.layers[-1]
        out = x @ weight.T + bias
        mean_n = out[:, : self.output_dim]
        log_std_n = bounded_log_std(out[:, self.output_dim :])
        return mean_n * self.target_scale + self.target_mean, np.exp(log_std_n) * self.target_scale

    def digest(self, salt: str) -> str:
        digest = hashlib.sha256(salt.encode())
        for weight, bias in self.layers:
            digest.update(weight.tobytes())
            digest.update(bias.tobytes())
        for stat in (self.input_mean, self.input_scale, self.target_mean, self.target_scale):
            digest.update(stat.tobytes())
        return digest.hexdigest()

    def to_npz_dict(self, prefix: str) -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {
            f"{prefix}input_mean": self.input_mean,
            f"{prefix}input_scale": self.input_scale,
            f"{prefix}target_mean": self.target_mean,
            f"{prefix}target_scale": self.target_scale,
            f"{prefix}num_layers": np.array(len(self.layers)),
        }
        for index, (weight, bias) in enumerate(self.layers):
            payload[f"{prefix}w{index}"] = weight
            payload[f"{prefix}b{index}"] = bias
        return payload

    @classmethod
    def from_npz_dict(cls, payload: Any, prefix: str) -> "GaussianMLPCore":
        count = int(payload[f"{prefix}num_layers"])
        layers = [(payload[f"{prefix}w{i}"], payload[f"{prefix}b{i}"]) for i in range(count)]
        return cls(
            layers,
            payload[f"{prefix}input_mean"],
            payload[f"{prefix}input_scale"],
            payload[f"{prefix}target_mean"],
            payload[f"{prefix}target_scale"],
        )


def _gaussian_log_density(residual: np.ndarray, std: np.ndarray) -> np.ndarray:
    var = std**2
    return -0.5 * (
        np.sum(residual**2 / var, axis=-1) + np.sum(np.log(var), axis=-1) + residual.shape[-1] * _LOG_2PI
    )


class GaussianMLPTransition(TransitionModel):
    """``next = state + delta``, ``delta ~ N(mlp_mean(state, action), mlp_std(state, action)^2)``.

    Follows the batch contract of the package's fitted transitions: a single
    state keeps the legacy shapes, a batch pairs rows with actions and with
    candidate successors, and a tensor state gets a tensor back.
    """

    def __init__(self, core: GaussianMLPCore, training_summary: Optional[Dict[str, Any]] = None):
        self._core = core
        self.dim = DIM
        self.action_dim = ACTION_DIM
        self.training_summary = dict(training_summary or {})

    @property
    def core(self) -> GaussianMLPCore:
        return self._core

    @property
    def fingerprint(self) -> str:
        return self._core.digest(type(self).__name__)

    def _predict(self, rows: np.ndarray, action: Any) -> Tuple[np.ndarray, np.ndarray]:
        actions, _ = as_rows(_to_numpy(action), self.action_dim)
        actions = np.broadcast_to(actions, (rows.shape[0], self.action_dim))
        return self._core.predict(np.concatenate([rows, actions], axis=1))

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        rows, batched = as_rows(state, self.dim)
        rows_np = _to_numpy(rows)
        mean, std = self._predict(rows_np, action)
        noise = np.random.standard_normal((rows_np.shape[0], n_samples, self.dim))
        samples = rows_np[:, None, :] + mean[:, None, :] + noise * std[:, None, :]
        if not batched:
            result = samples[0, 0] if n_samples == 1 else samples[0]
        else:
            result = samples[:, 0] if n_samples == 1 else samples
        return as_backend(result, state) if is_tensor(state) else result

    def log_probability(self, state: Any, action: Any, next_states: Any) -> Any:
        rows, batched = as_rows(state, self.dim)
        rows_np = _to_numpy(rows)
        candidates = np.atleast_2d(_to_numpy(next_states))
        mean, std = self._predict(rows_np, action)
        if batched:
            residual = candidates - rows_np - mean
            result = _gaussian_log_density(residual, std)
        else:
            residual = candidates - rows_np[0] - mean[0]
            result = _gaussian_log_density(residual, np.broadcast_to(std[0], residual.shape))
        return as_backend(result, state) if is_tensor(state) else result

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._core.to_npz_dict("t_"))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GaussianMLPTransition":
        with np.load(Path(path)) as payload:
            return cls(GaussianMLPCore.from_npz_dict(payload, "t_"))


class ObservationCandidate:
    """What the substituted LightDark model needs from an observation model.

    Observations are 2-D noisy positions. ``sample`` and ``log_probability``
    take one next state; ``log_probability_per_state`` takes many next states
    and one observation, which is the particle filter's call.
    """

    fingerprint: Optional[str] = None

    def sample(self, next_state: np.ndarray, n_samples: int = 1) -> np.ndarray:
        raise NotImplementedError

    def log_probability(self, next_state: np.ndarray, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_probability_per_state(self, next_states: np.ndarray, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TrueLightDarkObservation(ObservationCandidate):
    """LightDark's NORMAL_NOISE observation: ``obs = next + N(0, Sigma)``, Sigma halved near a beacon."""

    def __init__(self, beacons: np.ndarray, beacon_radius: float, covariance: np.ndarray) -> None:
        self.beacons = np.asarray(beacons, dtype=float)  # (2, K)
        self.beacon_radius = float(beacon_radius)
        self.cov_far = np.asarray(covariance, dtype=float)
        self.cov_near = self.cov_far * 0.5
        self._chol_t = {"far": np.linalg.cholesky(self.cov_far).T, "near": np.linalg.cholesky(self.cov_near).T}
        self._prec = {"far": np.linalg.inv(self.cov_far), "near": np.linalg.inv(self.cov_near)}
        self._lognorm = {
            key: -0.5 * (DIM * _LOG_2PI + np.linalg.slogdet(cov)[1])
            for key, cov in (("far", self.cov_far), ("near", self.cov_near))
        }
        self.fingerprint = None

    def near(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        diff = points[:, None, :] - self.beacons.T[None, :, :]
        min_sq = np.min(np.sum(diff * diff, axis=-1), axis=1)
        return min_sq < self.beacon_radius**2

    def std(self, points: np.ndarray) -> np.ndarray:
        near = self.near(points)
        far_std = np.sqrt(np.diag(self.cov_far))
        near_std = np.sqrt(np.diag(self.cov_near))
        return np.where(near[:, None], near_std[None, :], far_std[None, :])

    def sample(self, next_state: np.ndarray, n_samples: int = 1) -> np.ndarray:
        point = np.asarray(next_state, dtype=float).reshape(-1)[:DIM]
        key = "near" if bool(self.near(point)[0]) else "far"
        z = np.random.standard_normal((n_samples, DIM))
        samples = point + z @ self._chol_t[key]
        return samples[0] if n_samples == 1 else samples

    def _log_pdf(self, points: np.ndarray, observations: np.ndarray) -> np.ndarray:
        near = self.near(points)
        residual = observations - points
        out = np.empty(residual.shape[0])
        for key, mask in (("near", near), ("far", ~near)):
            if np.any(mask):
                r = residual[mask]
                out[mask] = self._lognorm[key] - 0.5 * np.sum((r @ self._prec[key]) * r, axis=-1)
        return out

    def log_probability(self, next_state: np.ndarray, observations: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(np.asarray(observations, dtype=float))
        points = np.broadcast_to(np.asarray(next_state, dtype=float).reshape(-1)[:DIM], obs.shape)
        return self._log_pdf(points, obs)

    def log_probability_per_state(self, next_states: np.ndarray, observation: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(next_states, dtype=float))[:, :DIM]
        obs = np.broadcast_to(np.asarray(observation, dtype=float).reshape(-1)[:DIM], points.shape)
        return self._log_pdf(points, obs)


class GaussianMLPObservation(ObservationCandidate):
    """``obs = next + r``, ``r ~ N(mlp_mean(next), mlp_std(next)^2)``; the MLP sees only the position."""

    def __init__(self, core: GaussianMLPCore, training_summary: Optional[Dict[str, Any]] = None):
        self._core = core
        self.training_summary = dict(training_summary or {})
        self.fingerprint = core.digest(type(self).__name__)

    @property
    def core(self) -> GaussianMLPCore:
        return self._core

    def std(self, points: np.ndarray) -> np.ndarray:
        return self._core.predict(np.atleast_2d(points)[:, :DIM])[1]

    def sample(self, next_state: np.ndarray, n_samples: int = 1) -> np.ndarray:
        point = np.asarray(next_state, dtype=float).reshape(-1)[:DIM]
        mean, std = self._core.predict(point[None, :])
        z = np.random.standard_normal((n_samples, DIM))
        samples = point + mean + z * std
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, next_state: np.ndarray, observations: np.ndarray) -> np.ndarray:
        point = np.asarray(next_state, dtype=float).reshape(-1)[:DIM]
        obs = np.atleast_2d(np.asarray(observations, dtype=float))
        mean, std = self._core.predict(point[None, :])
        residual = obs - point - mean
        return _gaussian_log_density(residual, np.broadcast_to(std, residual.shape))

    def log_probability_per_state(self, next_states: np.ndarray, observation: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(next_states, dtype=float))[:, :DIM]
        obs = np.asarray(observation, dtype=float).reshape(-1)[:DIM]
        mean, std = self._core.predict(points)
        return _gaussian_log_density(obs[None, :] - points - mean, std)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._core.to_npz_dict("o_"))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GaussianMLPObservation":
        with np.load(Path(path)) as payload:
            return cls(GaussianMLPCore.from_npz_dict(payload, "o_"))
