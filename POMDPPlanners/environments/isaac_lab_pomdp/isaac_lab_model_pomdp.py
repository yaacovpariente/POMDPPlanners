# SPDX-License-Identifier: MIT

"""Planner-side generative model for the IsaacLab POMDP.

:class:`~POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp.IsaacLabPOMDP`
is a forward-only *world*: it steps a live IsaacLab task and emits the real
observation, but it cannot resample from an arbitrary state or score observation
densities. A belief filter and a planner such as POMCPOW therefore need a
separate *generative model* to search inside — this module provides it.

Design (see the project discussion that motivated it): rather than hand-splitting
each of IsaacLab's ~200 tasks into privileged-state vs sensor-observation terms,
the model keeps **state and observation in the same space** and treats the
observation as the state seen through additive Gaussian sensor noise —
``observation = state + N(0, Sigma)``. This makes the observation model generic
(one model for every task), trivially samplable from any state, and guarantees
the state explains the observation by construction. Truly unmeasurable *constant*
parameters (friction, added mass, actuator gains) are handled as domain
randomization on the world side, not as observation channels.

The Normal-noise mechanism reuses
:class:`~POMDPPlanners.utils.multivariate_normal.CovarianceParameterizedMultivariateNormal`
— the same fixed-covariance-varying-mean multivariate normal the continuous
light-dark POMDP uses for its ``NORMAL_NOISE`` observation model and its state
transition — so the Cholesky factorization is computed once and reused.

Transition model — a swappable seam. IsaacLab's true contact dynamics are not
analytic, so the transition is injected via the :class:`TransitionModel`
interface. Two concretes ship here:

* :class:`GaussianRandomWalkTransition` — the trivial, action-ignoring default
  (``next = state + N(0, Sigma)``); planning is cosmetic under it.
* :class:`LinearGaussianTransition` — a first-order **learned** dynamics model
  ``next = A @ state + B @ action + b + N(0, Sigma)`` whose parameters are fit
  from ``(state, action, next_state)`` rollouts via ridge least squares. It is
  action-conditioned, so POMCPOW's lookahead sees that actions move the state.

A learned, history-conditioned world model (RSSM/Dreamer) can be dropped in as a
third :class:`TransitionModel` without touching the observation model or the
planner.

Reward model — the objective POMCPOW optimizes. The forward-only world cannot be
queried for the reward of a hypothetical in-tree transition, so the reward is also
injected, via :class:`RewardModel`. :class:`LinearRewardModel` is fit from the same
warm-up rollouts as the transition; without it the model's reward is a flat zero
and the planner has no objective, so the task is never solved.

The :class:`~POMDPPlanners.core.environment.transition_model.TransitionModel` and
:class:`~POMDPPlanners.core.environment.transition_model.RewardModel` interfaces
themselves live in core, because fitting one from rollouts is generic work that
must not depend on an environment package. They are re-exported here so every
existing import path keeps resolving.

Classes:
    TransitionModel: Interface for a state-transition model (re-exported from core).
    GaussianRandomWalkTransition: Action-ignoring Gaussian random-walk transition.
    LinearGaussianTransition: Fit-from-data linear-Gaussian action-conditioned transition.
    IsaacLabSimulatorTransition: Steps the IsaacLab simulator as the true transition.
    RewardModel: Interface for a reward model over a transition (re-exported from core).
    LinearRewardModel: Fit-from-data linear reward model POMCPOW optimizes.
    GaussianObservationModel: Additive-Normal observation model (obs = state + noise).
    IsaacLabModelPOMDP: Discrete-action generative model POMCPOW searches inside.
"""

import hashlib
import math
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    DiscreteActionsEnvironment,
    RewardModel,
    SpaceInfo,
    SpaceType,
    TransitionModel,
)

# Reuse the world module's launch seam so both share the single per-process
# ``SimulationApp`` (IsaacLab permits exactly one) and so tests can monkeypatch
# ``_build_isaac_env`` here to inject a fake batched env.
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import (
    _build_isaac_env,
    _to_numpy,
)
from POMDPPlanners.core.environment.array_backend import (
    BackendParameters,
    as_backend,
    as_rows,
    is_tensor,
    standard_normal,
)
from POMDPPlanners.utils.multivariate_normal import (
    CovarianceParameterizedMultivariateNormal,
)

NoiseStd = Union[float, np.ndarray]

# A ``state_writer`` writes a ``(batch, dim)`` block of flat state vectors back
# into the live IsaacLab env (the inverse of the world's state extractor).
StateWriter = Callable[[Any, np.ndarray], None]
StateReader = Callable[[Any], np.ndarray]


def _diagonal_covariance(dim: int, std: NoiseStd) -> np.ndarray:
    """Build a diagonal covariance from a scalar or per-channel standard deviation.

    Args:
        dim: Dimensionality of the state/observation vector.
        std: A scalar standard deviation (isotropic noise) or a length-``dim``
            array of per-channel standard deviations.

    Returns:
        A ``(dim, dim)`` diagonal covariance matrix.

    Raises:
        ValueError: If any standard deviation is non-positive or ``std`` has the
            wrong length.
    """
    std_vector = np.broadcast_to(np.asarray(std, dtype=float), (dim,)).astype(float)
    if np.any(std_vector <= 0.0):
        raise ValueError("noise standard deviations must be strictly positive")
    return np.diag(std_vector**2)


def _is_legacy_call(state: Any) -> bool:
    """Whether this call is the pre-batching case: a numpy/list 1-D state.

    Those calls keep the exact code path they always took -- the shared
    :class:`~POMDPPlanners.utils.multivariate_normal.CovarianceParameterizedMultivariateNormal`,
    triangular solve and all. Routing them through the batched kernel instead
    would be mathematically equivalent and numerically slightly different, and
    "no existing result moved" is worth more than one less branch.
    """
    if is_tensor(state):
        return False
    return np.asarray(state, dtype=float).ndim == 1


def _shape_samples(samples: Any, batched: bool, n_samples: int) -> Any:
    """Give a ``(N, n_samples, dim)`` draw back in the shape the caller asked for."""
    if not batched:
        return samples[0, 0] if n_samples == 1 else samples[0]
    return samples[:, 0] if n_samples == 1 else samples


class _BatchedGaussian:
    """Fixed-covariance normal over a batch of means, following its input's backend.

    The scalar models already have a normal --
    :class:`~POMDPPlanners.utils.multivariate_normal.CovarianceParameterizedMultivariateNormal`
    -- and it stays the path every pre-existing call takes, bit for bit. This is
    the batched counterpart: one mean *per row* rather than one mean for the whole
    call, and tensors out when handed tensors, which is what lets a GPU planner
    take a step without a host round trip.

    Two normals rather than one widened normal is deliberate. The shared one is
    used by several environments; widening it would put all of them on new
    numerics for the sake of this one.

    Args:
        covariance: Positive-definite ``(dim, dim)`` covariance.
    """

    def __init__(self, covariance: np.ndarray) -> None:
        matrix = np.asarray(covariance, dtype=float)
        self.dim = int(matrix.shape[0])
        self._params = BackendParameters(
            cholesky_t=np.linalg.cholesky(matrix).T,
            precision=np.linalg.inv(matrix),
        )
        self._log_normalizer = float(
            -0.5 * (self.dim * math.log(2.0 * math.pi) + np.linalg.slogdet(matrix)[1])
        )

    def sample(self, means: Any, n_samples: int) -> Any:
        """Draw ``n_samples`` per row.

        Args:
            means: ``(N, dim)`` means, one per row.
            n_samples: Draws per row.

        Returns:
            ``(N, n_samples, dim)`` samples.
        """
        params = self._params.matching(means)
        noise = standard_normal((int(means.shape[0]), n_samples, self.dim), means)
        return means[:, None, :] + noise @ params["cholesky_t"]

    def log_pdf(self, means: Any, values: Any) -> Any:
        """Row-wise log-density, broadcasting a single mean across many values.

        Args:
            means: ``(N, dim)`` or ``(1, dim)`` means.
            values: ``(N, dim)`` points.

        Returns:
            ``(N,)`` log-densities.
        """
        params = self._params.matching(means)
        residual = values - means
        mahalanobis = ((residual @ params["precision"]) * residual).sum(-1)
        return self._log_normalizer - 0.5 * mahalanobis


class GaussianRandomWalkTransition(TransitionModel):
    """Action-ignoring Gaussian random walk: ``next_state = state + N(0, Sigma)``.

    The trivial placeholder transition. Because it ignores the action, a planner's
    lookahead cannot distinguish actions under it — use it only for wiring/tests
    or when a learned transition is not yet available.

    Example:
        Draw a next state near the current one::

            transition = GaussianRandomWalkTransition(dim=4, process_noise_std=0.05)
            nxt = transition.sample_next_state([0.0, 1.0, 2.0, 3.0], action=None)
    """

    def __init__(self, dim: int, process_noise_std: NoiseStd = 0.05) -> None:
        """Initialize the random-walk transition.

        Args:
            dim: Dimensionality of the state vector.
            process_noise_std: Scalar or per-channel std of the process noise.
        """
        self.dim = int(dim)
        covariance = _diagonal_covariance(self.dim, process_noise_std)
        self._normal = CovarianceParameterizedMultivariateNormal(covariance)
        self._batched = _BatchedGaussian(covariance)

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample successors, accepting either one state or a batch of them.

        Shapes and backend conventions are shared with
        :meth:`LinearGaussianTransition.sample_next_state`.
        """
        del action  # random walk ignores the action
        if _is_legacy_call(state):
            mean = np.asarray(state, dtype=float).reshape(-1)
            samples = self._normal.sample(mean, n_samples=n_samples)
            return samples[0] if n_samples == 1 else samples
        rows, batched = as_rows(state, self.dim)
        return _shape_samples(self._batched.sample(rows, n_samples), batched, n_samples)

    def log_probability(self, state: Any, action: Any, next_states: Any) -> Any:
        """Log-density of the successors, accepting either one state or a batch."""
        del action
        if _is_legacy_call(state):
            mean = np.asarray(state, dtype=float).reshape(-1)
            return self._normal.log_pdf(np.asarray(next_states, dtype=float), mean)
        rows, _ = as_rows(state, self.dim)
        candidates, _ = as_rows(as_backend(next_states, rows), self.dim)
        return self._batched.log_pdf(rows, candidates)


class LinearGaussianTransition(TransitionModel):
    """Learned linear-Gaussian transition ``next = A @ state + B @ action + b + N(0, Sigma)``.

    A first-order, action-conditioned dynamics model whose parameters are fit from
    ``(state, action, next_state)`` rollouts (see :meth:`fit`). Analytic, so both
    sampling and log-density reuse the pre-factorized normal. It is deliberately a
    system-identification baseline — a linear approximation of the true nonlinear
    contact dynamics — but, unlike the random walk, it makes actions move the
    predicted state so POMCPOW's planning is meaningful.

    Attributes:
        dim: State dimensionality.
        action_dim: Action dimensionality.

    Example:
        Fit from rollouts, then sample::

            transition = LinearGaussianTransition.fit(states, actions, next_states)
            nxt = transition.sample_next_state(states[0], actions[0])
    """

    def __init__(
        self,
        weight_state: np.ndarray,
        weight_action: np.ndarray,
        bias: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """Initialize the linear-Gaussian transition from explicit parameters.

        Args:
            weight_state: State matrix ``A`` of shape ``(dim, dim)``.
            weight_action: Action matrix ``B`` of shape ``(dim, action_dim)``.
            bias: Offset vector ``b`` of shape ``(dim,)``.
            covariance: Residual covariance of shape ``(dim, dim)``.
        """
        self._weight_state = np.asarray(weight_state, dtype=float)
        self._weight_action = np.asarray(weight_action, dtype=float)
        self._bias = np.asarray(bias, dtype=float).reshape(-1)
        self.dim = self._weight_state.shape[0]
        self.action_dim = self._weight_action.shape[1]
        self._covariance = np.asarray(covariance, dtype=float)
        self._normal = CovarianceParameterizedMultivariateNormal(self._covariance)
        self._batched = _BatchedGaussian(self._covariance)
        self._params = BackendParameters(
            weight_state=self._weight_state,
            weight_action=self._weight_action,
            bias=self._bias,
        )

    def _mean(self, state: Any, action: Any) -> np.ndarray:
        state_vector = np.asarray(state, dtype=float).reshape(-1)
        action_vector = np.asarray(action, dtype=float).reshape(-1)
        return self._weight_state @ state_vector + self._weight_action @ action_vector + self._bias

    def _mean_rows(self, state: Any, action: Any) -> Any:
        """Per-row transition means, in the backend of ``state``."""
        states, batched = as_rows(state, self.dim)
        actions, _ = as_rows(as_backend(action, states), self.action_dim)
        params = self._params.matching(states)
        mean = (
            states @ params["weight_state"].T
            + actions @ params["weight_action"].T
            + params["bias"]
        )
        return mean, batched

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        """Sample successors, accepting either one state or a batch of them.

        Args:
            state: A ``(dim,)`` state, or a ``(N, dim)`` batch -- one row per
                particle, each with its own action.
            action: A ``(action_dim,)`` action, or a ``(N, action_dim)`` batch.
                A single action broadcasts across a batch of states.
            n_samples: Draws per state.

        Returns:
            For a single state: ``(dim,)`` when ``n_samples == 1``, else
            ``(n_samples, dim)`` -- unchanged from before batching existed. For a
            batch: ``(N, dim)`` when ``n_samples == 1``, else
            ``(N, n_samples, dim)``.

            The result follows the *state's* backend: a numpy state gives a numpy
            array, a tensor gives a tensor on that tensor's device. That is what
            lets a vectorized planner take a step with no host round trip.
        """
        if _is_legacy_call(state):
            mean = self._mean(state, action)
            samples = self._normal.sample(mean, n_samples=n_samples)
            return samples[0] if n_samples == 1 else samples
        means, batched = self._mean_rows(state, action)
        return _shape_samples(self._batched.sample(means, n_samples), batched, n_samples)

    def log_probability(self, state: Any, action: Any, next_states: Any) -> Any:
        """Log-density of the successors, accepting either one state or a batch.

        Args:
            state: A ``(dim,)`` state, or a ``(N, dim)`` batch.
            action: The action, or one per row.
            next_states: For a single state, a ``(dim,)`` candidate or an
                ``(n, dim)`` batch scored against that one state. For a batch of
                states, an ``(N, dim)`` array paired **row-wise** -- candidate
                ``i`` is scored under state ``i``, which is what a particle filter
                needs and what the single-state form cannot express.

        Returns:
            ``(n,)`` or ``(N,)`` log-densities, in the state's backend.
        """
        if _is_legacy_call(state):
            mean = self._mean(state, action)
            return self._normal.log_pdf(np.asarray(next_states, dtype=float), mean)
        means, _ = self._mean_rows(state, action)
        candidates, _ = as_rows(as_backend(next_states, means), self.dim)
        return self._batched.log_pdf(means, candidates)

    @property
    def fingerprint(self) -> str:
        """Hash of the fitted parameters -- two different fits never share one.

        The model that plans with a transition holds it privately, and an
        environment's ``config_id`` skips private attributes, so a refitted
        transition leaves the cache key untouched and the next run is served the
        previous fit's episodes. An environment planning with a fitted
        transition exposes this string to move the key with the parameters.

        Returns:
            A hex digest over ``A``, ``B``, ``b`` and the residual covariance.
        """
        digest = hashlib.sha256(type(self).__name__.encode())
        for parameter in (
            self._weight_state,
            self._weight_action,
            self._bias,
            self._covariance,
        ):
            digest.update(np.asarray(parameter, dtype=float).tobytes())
        return digest.hexdigest()

    def save(self, path: Union[str, Path]) -> Path:
        """Write the fitted parameters to one ``.npz`` file.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weight_state=self._weight_state,
            weight_action=self._weight_action,
            bias=self._bias,
            covariance=self._covariance,
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearGaussianTransition":
        """Rebuild a transition written by :meth:`save`.

        Args:
            path: File written by :meth:`save`.

        Returns:
            The transition.
        """
        with np.load(Path(path)) as payload:
            return cls(
                weight_state=payload["weight_state"],
                weight_action=payload["weight_action"],
                bias=payload["bias"],
                covariance=payload["covariance"],
            )

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        regularization: float = 1e-4,
        min_variance: float = 1e-6,
    ) -> "LinearGaussianTransition":
        """Fit ``A``, ``B``, ``b`` and a diagonal residual covariance via ridge least squares.

        Args:
            states: Array of shape ``(N, dim)`` of source states.
            actions: Array of shape ``(N, action_dim)`` of applied actions.
            next_states: Array of shape ``(N, dim)`` of resulting states.
            regularization: Ridge penalty added to the normal-equations diagonal.
            min_variance: Floor on each residual variance to keep the covariance
                positive definite.

        Returns:
            A fitted :class:`LinearGaussianTransition`.

        Raises:
            ValueError: If fewer than two transitions are supplied or shapes disagree.
        """
        states_2d = np.atleast_2d(np.asarray(states, dtype=float))
        actions_2d = np.atleast_2d(np.asarray(actions, dtype=float))
        next_2d = np.atleast_2d(np.asarray(next_states, dtype=float))
        if states_2d.shape[0] < 2:
            raise ValueError("fitting a linear transition needs at least two transitions")
        if not states_2d.shape[0] == actions_2d.shape[0] == next_2d.shape[0]:
            raise ValueError("states, actions, and next_states must have equal length")

        dim = states_2d.shape[1]
        action_dim = actions_2d.shape[1]
        ones = np.ones((states_2d.shape[0], 1))
        design = np.hstack([states_2d, actions_2d, ones])  # (N, dim + action_dim + 1)

        gram = design.T @ design + regularization * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ next_2d)  # (dim + action_dim + 1, dim)

        weight_state = weights[:dim].T
        weight_action = weights[dim : dim + action_dim].T
        bias = weights[-1]
        residuals = next_2d - design @ weights
        variances = np.maximum(np.var(residuals, axis=0), min_variance)
        return cls(weight_state, weight_action, bias, np.diag(variances))


class IsaacLabSimulatorTransition(TransitionModel):
    """State-transition model that steps the IsaacLab simulator itself.

    ``next_state = f_sim(state, action) + N(0, Sigma)``: the flat state vector is
    written back into the physics engine, the sim is advanced one step under
    ``action``, and the resulting state is read out as the transition **mean**. A
    small diagonal process noise is added so the transition is a proper, samplable
    density — the bare physics step is a deterministic point mass with no density
    for the belief filter to weight against.

    Unlike :class:`LinearGaussianTransition`, this is the *true* nonlinear
    dynamics rather than a linear fit, so POMCPOW plans against exact contact
    physics. The cost is one GPU sim step per query.

    Constraints:
        * **One SimulationApp per process.** This transition builds its own
          batched IsaacLab env through the same launch singleton the world uses,
          so it cannot coexist with an IsaacLab *world* in the same process — pair
          it with a non-IsaacLab world (real robot or a cheaper simulator).
        * **State writer is task-specific.** Writing a flat state back into the
          sim is the inverse of the world's state extractor and depends on the
          asset/frame conventions, so it is injected via ``state_writer`` rather
          than guessed.

    Attributes:
        dim: Dimensionality of the shared state/observation vector.
        num_envs: Number of parallel sim envs the batched model env is built with.

    Example:
        Wrap a task, supplying the inverse of the world's state extractor::

            def write_state(env, states):  # states: (batch, dim)
                env.unwrapped.write_root_state_to_sim(to_torch(states))

            transition = IsaacLabSimulatorTransition(
                task_id="Isaac-Reach-Franka-v0", dim=18, state_writer=write_state,
                device="cpu", process_noise_std=0.01,
            )
            nxt = transition.sample_next_state(state, action)
    """

    def __init__(
        self,
        task_id: str,
        dim: int,
        state_writer: StateWriter,
        num_envs: int = 1,
        device: str = "cuda",
        env_cfg_kwargs: Optional[Dict[str, Any]] = None,
        headless: bool = True,
        process_noise_std: NoiseStd = 0.01,
        state_reader: Optional[StateReader] = None,
        state_asset: str = "robot",
        action_space_type: SpaceType = SpaceType.CONTINUOUS,
    ) -> None:
        """Initialize the simulator-backed transition.

        Args:
            task_id: Registered IsaacLab task id (e.g. ``"Isaac-Reach-Franka-v0"``).
            dim: Dimensionality of the flat state/observation vector.
            state_writer: Callback ``(env, states) -> None`` writing a ``(batch, dim)``
                block of flat states into the live env — the inverse of the world's
                state extractor.
            num_envs: Parallel sim envs the model env is built with. Defaults to 1.
            device: Torch device the simulator runs on. Defaults to ``"cuda"``.
            env_cfg_kwargs: Extra keyword args for ``parse_env_cfg``. Defaults to none.
            headless: Launch the simulator without a GUI. Defaults to True.
            process_noise_std: Scalar or per-channel std of the additive process
                noise around the physics step. Defaults to 0.01.
            state_reader: Optional ``env -> (num_envs, dim)`` reader of the resulting
                states. Defaults to concatenating root pose/velocity and joint state.
            state_asset: Scene key for the ground-truth articulation. Defaults to
                ``"robot"``.
            action_space_type: Action space category. Defaults to CONTINUOUS.
        """
        self.task_id = task_id
        self.dim = int(dim)
        self.num_envs = int(num_envs)
        self.device = device
        self.env_cfg_kwargs: Dict[str, Any] = dict(env_cfg_kwargs) if env_cfg_kwargs else {}
        self.headless = headless
        self.state_asset = state_asset
        self.action_space_type = action_space_type
        self._state_writer = state_writer
        self._state_reader = state_reader
        self._env: Optional[Any] = None
        self._normal = CovarianceParameterizedMultivariateNormal(
            _diagonal_covariance(self.dim, process_noise_std)
        )

    def _get_env(self) -> Any:
        if self._env is None:
            self._env = _build_isaac_env(
                self.task_id, self.num_envs, self.device, self.env_cfg_kwargs, self.headless
            )
        return self._env

    def _to_isaac_action(self, action: Any, batch: int) -> Any:
        import torch  # pylint: disable=import-outside-toplevel

        if self.action_space_type == SpaceType.DISCRETE:
            return torch.as_tensor([[int(action)]] * batch, device=self.device)
        row = np.asarray(action, dtype=np.float32).reshape(-1)
        return torch.as_tensor(np.tile(row, (batch, 1)), device=self.device)

    def _read_states(self, env: Any) -> np.ndarray:
        if self._state_reader is not None:
            return np.atleast_2d(np.asarray(self._state_reader(env), dtype=float))
        return self._default_read_states(env)

    def _default_read_states(self, env: Any) -> np.ndarray:
        """Read the ground-truth state of every env as a ``(num_envs, dim)`` block."""
        data = env.unwrapped.scene[self.state_asset].data
        components = (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "joint_pos",
            "joint_vel",
        )
        parts = [
            _to_numpy(getattr(data, attr)).reshape(self.num_envs, -1)
            for attr in components
            if getattr(data, attr, None) is not None
        ]
        if not parts:
            raise RuntimeError(
                f"No ground-truth state fields found on scene asset "
                f"'{self.state_asset}'; pass a custom state_reader."
            )
        return np.concatenate(parts, axis=1)

    def _sim_step_mean(self, state: Any, action: Any) -> np.ndarray:
        """Write ``state`` into the sim, step once under ``action``, read the result."""
        env = self._get_env()
        self._state_writer(env, np.asarray(state, dtype=float).reshape(1, -1))
        env.step(self._to_isaac_action(action, batch=1))
        return self._read_states(env)[0]

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        mean = self._sim_step_mean(state, action)
        samples = self._normal.sample(mean, n_samples=n_samples)
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        mean = self._sim_step_mean(state, action)
        return self._normal.log_pdf(np.asarray(next_states, dtype=float), mean)


class LinearRewardModel(RewardModel):
    """Learned linear reward ``r = w_s . state + w_a . action + w_n . next_state + b``.

    Fit from ``(state, action, next_state, reward)`` rollouts via ridge least
    squares (see :meth:`fit`). It is a first-order approximation of the task's true
    (often nonlinear) reward, but it gives POMCPOW a real objective to optimize —
    without it the planner has no signal and produces undirected behavior.

    Example:
        Fit from rollouts, then score a transition::

            reward_model = LinearRewardModel.fit(states, actions, next_states, rewards)
            value = reward_model.reward(states[0], actions[0], next_states[0])
    """

    def __init__(
        self,
        weight_state: np.ndarray,
        weight_action: np.ndarray,
        weight_next_state: np.ndarray,
        bias: float,
    ) -> None:
        """Initialize the linear reward model from explicit parameters.

        Args:
            weight_state: Coefficients on the state, shape ``(dim,)``.
            weight_action: Coefficients on the action, shape ``(action_dim,)``.
            weight_next_state: Coefficients on the next state, shape ``(dim,)``.
            bias: Scalar offset.
        """
        self._weight_state = np.asarray(weight_state, dtype=float).reshape(-1)
        self._weight_action = np.asarray(weight_action, dtype=float).reshape(-1)
        self._weight_next_state = np.asarray(weight_next_state, dtype=float).reshape(-1)
        self._bias = float(bias)
        self._params = BackendParameters(
            weight_state=self._weight_state,
            weight_action=self._weight_action,
            weight_next_state=self._weight_next_state,
        )

    def reward(self, state: Any, action: Any, next_state: Any) -> Any:
        """Score a transition, or a whole batch of them.

        Args:
            state: A ``(dim,)`` state, or a ``(N, dim)`` batch.
            action: The action, or one per row.
            next_state: The successor, or one per row.

        Returns:
            A Python ``float`` for a single transition -- unchanged from before
            batching existed -- or an ``(N,)`` array in the state's backend for a
            batch.

            Batching the reward matters as much as batching the transition: a
            vectorized rollout that steps on device and then drops to the host to
            score the reward pays the sync it was built to avoid, once per step.
        """
        if _is_legacy_call(state):
            state_vector = np.asarray(state, dtype=float).reshape(-1)
            action_vector = np.asarray(action, dtype=float).reshape(-1)
            next_vector = np.asarray(next_state, dtype=float).reshape(-1)
            return float(
                self._weight_state @ state_vector
                + self._weight_action @ action_vector
                + self._weight_next_state @ next_vector
                + self._bias
            )
        states, _ = as_rows(state, self._weight_state.shape[0])
        actions, _ = as_rows(as_backend(action, states), self._weight_action.shape[0])
        next_states, _ = as_rows(as_backend(next_state, states), self._weight_next_state.shape[0])
        params = self._params.matching(states)
        return (
            states @ params["weight_state"]
            + actions @ params["weight_action"]
            + next_states @ params["weight_next_state"]
            + self._bias
        )

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        regularization: float = 1e-4,
    ) -> "LinearRewardModel":
        """Fit the reward coefficients via ridge least squares.

        Args:
            states: Array of shape ``(N, dim)`` of source states.
            actions: Array of shape ``(N, action_dim)`` of applied actions.
            next_states: Array of shape ``(N, dim)`` of resulting states.
            rewards: Array of shape ``(N,)`` of observed rewards.
            regularization: Ridge penalty added to the normal-equations diagonal.

        Returns:
            A fitted :class:`LinearRewardModel`.

        Raises:
            ValueError: If fewer than two transitions are supplied or shapes disagree.
        """
        states_2d = np.atleast_2d(np.asarray(states, dtype=float))
        actions_2d = np.atleast_2d(np.asarray(actions, dtype=float))
        next_2d = np.atleast_2d(np.asarray(next_states, dtype=float))
        rewards_1d = np.asarray(rewards, dtype=float).reshape(-1)
        if states_2d.shape[0] < 2:
            raise ValueError("fitting a linear reward needs at least two transitions")
        if not states_2d.shape[0] == actions_2d.shape[0] == next_2d.shape[0] == rewards_1d.shape[0]:
            raise ValueError("states, actions, next_states, and rewards must have equal length")

        dim = states_2d.shape[1]
        action_dim = actions_2d.shape[1]
        ones = np.ones((states_2d.shape[0], 1))
        design = np.hstack([states_2d, actions_2d, next_2d, ones])
        gram = design.T @ design + regularization * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ rewards_1d)
        return cls(
            weight_state=weights[:dim],
            weight_action=weights[dim : dim + action_dim],
            weight_next_state=weights[dim + action_dim : 2 * dim + action_dim],
            bias=float(weights[-1]),
        )


class GaussianObservationModel:
    """Additive-Normal observation model: ``observation = state + N(0, Sigma)``.

    A fixed-covariance multivariate normal whose mean is the state, mirroring the
    continuous light-dark ``NORMAL_NOISE`` model. The covariance is diagonal and
    parameterized per channel, so proprioceptive channels can carry tight noise
    and exteroceptive ones looser noise without any per-task code.

    Attributes:
        dim: Dimensionality of the observation/state vector.

    Example:
        Sample and score an observation for a 4-D state::

            model = GaussianObservationModel(observation_dim=4, noise_std=0.1)
            obs = model.sample([0.0, 1.0, 2.0, 3.0])
            log_p = model.log_probability([0.0, 1.0, 2.0, 3.0], obs)
    """

    def __init__(self, observation_dim: int, noise_std: NoiseStd = 0.1) -> None:
        """Initialize the additive-Normal observation model.

        Args:
            observation_dim: Dimensionality of the observation/state vector.
            noise_std: Scalar (isotropic) or per-channel standard deviation of the
                additive Gaussian sensor noise. Defaults to 0.1.
        """
        self.dim = int(observation_dim)
        covariance = _diagonal_covariance(self.dim, noise_std)
        self._normal = CovarianceParameterizedMultivariateNormal(covariance)

    def sample(self, state: Any, n_samples: int = 1) -> np.ndarray:
        """Draw ``observation = state + noise`` for the given state.

        Args:
            state: The (next) state to observe, a length-``dim`` vector.
            n_samples: Number of observations to draw. Defaults to 1.

        Returns:
            A single ``(dim,)`` observation when ``n_samples == 1``, else a
            ``(n_samples, dim)`` array.
        """
        mean = np.asarray(state, dtype=float).reshape(-1)
        samples = self._normal.sample(mean, n_samples=n_samples)
        return samples[0] if n_samples == 1 else samples

    def log_probability(self, state: Any, observations: Any) -> np.ndarray:
        """Gaussian log-density of ``observations`` centered on ``state``.

        Args:
            state: The (next) state, a length-``dim`` vector (the Gaussian mean).
            observations: A single ``(dim,)`` observation or a ``(n, dim)`` batch.

        Returns:
            A ``(n,)`` array of log-densities, one per observation.
        """
        mean = np.asarray(state, dtype=float).reshape(-1)
        return self._normal.log_pdf(np.asarray(observations, dtype=float), mean)


class IsaacLabModelPOMDP(DiscreteActionsEnvironment):
    """Discrete-action generative model POMCPOW searches inside for IsaacLab.

    State and observation share one space; the observation is the state seen
    through :class:`GaussianObservationModel`, and the transition is any
    :class:`TransitionModel` (defaulting to :class:`GaussianRandomWalkTransition`).
    Actions are a finite set of continuous control vectors applied verbatim to the
    world.

    Attributes:
        observation_dim: Dimensionality of the shared state/observation vector.
        action_presets: The finite set of action vectors the planner chooses among.

    Example:
        Build a model over a 4-D observation with three 2-D action presets::

            import numpy as np
            presets = [np.zeros(2), np.ones(2), -np.ones(2)]
            model = IsaacLabModelPOMDP(observation_dim=4, action_presets=presets,
                                       discount_factor=0.99)
            actions = model.get_actions()
    """

    def __init__(
        self,
        observation_dim: int,
        action_presets: List[np.ndarray],
        discount_factor: float,
        observation_noise_std: NoiseStd = 0.1,
        process_noise_std: NoiseStd = 0.05,
        transition: Optional[TransitionModel] = None,
        reward_model: Optional[RewardModel] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the planner-side generative model.

        Args:
            observation_dim: Dimensionality of the shared state/observation vector.
            action_presets: Finite list of continuous action vectors to plan over.
            discount_factor: POMDP discount factor (shared with the world).
            observation_noise_std: Scalar or per-channel std of the observation noise.
            process_noise_std: Scalar or per-channel std of the default random-walk
                transition. Ignored when an explicit ``transition`` is given.
            transition: The state-transition model to plan with. Defaults to a
                :class:`GaussianRandomWalkTransition` built from ``process_noise_std``.
            reward_model: The reward model POMCPOW optimizes. Defaults to ``None``,
                which yields a flat zero reward (undirected planning) — supply a
                fitted :class:`LinearRewardModel` to make the planner solve the task.
            name: Model name (also used to label planner output).
        """
        super().__init__(
            discount_factor=discount_factor,
            name=name if name is not None else type(self).__name__,
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.CONTINUOUS,
            ),
        )
        self.observation_dim = int(observation_dim)
        self.action_presets = [np.asarray(a, dtype=float).reshape(-1) for a in action_presets]
        self._observation_model = GaussianObservationModel(
            self.observation_dim, observation_noise_std
        )
        self._transition = (
            transition
            if transition is not None
            else GaussianRandomWalkTransition(self.observation_dim, process_noise_std)
        )
        self._reward_model = reward_model

    def get_actions(self) -> List[np.ndarray]:
        """Return the finite set of action vectors the planner chooses among."""
        return [preset.copy() for preset in self.action_presets]

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        return self._transition.sample_next_state(state, action, n_samples=n_samples)

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        return self._transition.log_probability(state, action, next_states)

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> np.ndarray:
        del action
        return self._observation_model.sample(next_state, n_samples=n_samples)

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        return self._observation_model.log_probability(next_state, observations)

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        if self._reward_model is None:
            return 0.0  # no reward model → undirected planning (see __init__ docstring)
        resulting = state if next_state is None else next_state
        return self._reward_model.reward(state, action, resulting)

    def is_terminal(self, state: Any) -> bool:
        del state
        return False

    def initial_state_dist(self) -> Distribution:
        raise NotImplementedError(
            "IsaacLabModelPOMDP has no initial-state prior; seed the belief from "
            "the world's initial observation."
        )

    def initial_observation_dist(self) -> Distribution:
        raise NotImplementedError(
            "IsaacLabModelPOMDP has no initial-observation prior; seed the belief "
            "from the world's initial observation."
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return np.array_equal(np.asarray(observation1), np.asarray(observation2))

    def hash_observation(self, observation: Any) -> Hashable:
        return np.asarray(observation).tobytes()

    def hash_action(self, action: Any) -> Hashable:
        return np.asarray(action).tobytes()
