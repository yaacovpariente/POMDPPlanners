# SPDX-License-Identifier: MIT

"""CarDreamer/DreamerV3 adapter for the :class:`DreamerWorldModel` protocol.

:class:`DreamerCarlaModelPOMDP` plans inside *any* object satisfying the
:class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_dreamer_model_pomdp.DreamerWorldModel`
protocol. This module supplies one concrete backing: a trained **DreamerV3** world model
from the CarDreamer project (https://github.com/ucd-dare/CarDreamer), whose RSSM is a
JAX/`ninjax` module.

The adapter bridges two representations:

* The planner-side *latent* the POMDP carries is a **flat 1-D float vector** — this module
  packs it as ``concat(deter, stoch.flatten())`` (the DreamerV3 recurrent state), and
  unpacks it back into the ``{deter, stoch}`` state dict the RSSM and heads consume.
* Each protocol method is *batched* over ``(batch, latent_dim)`` and runs the relevant
  DreamerV3 component (encoder, ``rssm.obs_step`` / ``rssm.img_step``, or a head) through a
  single ``ninjax.pure`` call, converting NumPy in and NumPy out.

JAX, ``ninjax``, and ``dreamerv3`` are imported **lazily** (inside the constructor and the
factory), so importing this module — and the example script that references it — never
requires the deep-learning stack. Only actually *building* a
:class:`CarDreamerWorldModel` does.

Precondition — schema alignment:
    The trained checkpoint's observation space must expose the CARLA schema keys this
    POMDP uses (``gnss`` and ``agents``) with matching shapes, and its action space must
    accept the ``(throttle, steer, brake)`` control triple. A CarDreamer task configured
    with those observation handlers satisfies this by construction; a checkpoint trained
    on, e.g., birds-eye-view images does not and cannot be plugged in unchanged.

Classes:
    CarDreamerWorldModel: DreamerV3-backed implementation of ``DreamerWorldModel``.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# The observation heads decoded back into the CARLA observation dict.
_OBSERVATION_KEYS: Tuple[str, ...] = ("gnss", "agents")


def _unpack_latents(
    latents: np.ndarray, deter_dim: int, stoch_shape: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a ``(batch, latent_dim)`` packed latent into ``(deter, stoch)`` arrays."""
    array = np.asarray(latents, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    deter = array[:, :deter_dim]
    stoch = array[:, deter_dim:].reshape((array.shape[0], *stoch_shape))
    return deter, stoch


def _pack_latents(deter: np.ndarray, stoch: np.ndarray) -> np.ndarray:
    """Pack ``(deter, stoch)`` back into a ``(batch, latent_dim)`` flat latent."""
    deter_array = np.asarray(deter, dtype=np.float32)
    stoch_array = np.asarray(stoch, dtype=np.float32)
    batch = deter_array.shape[0]
    return np.concatenate([deter_array.reshape(batch, -1), stoch_array.reshape(batch, -1)], axis=1)


class CarDreamerWorldModel:
    """Trained CarDreamer DreamerV3 world model exposed as a ``DreamerWorldModel``.

    Wraps a constructed DreamerV3 JAX agent (the object holding the fitted parameters in
    ``agent.varibs`` and the world model in ``agent.agent.wm``) and routes every protocol
    method onto its RSSM and prediction heads. Build one from a training checkpoint with
    :meth:`from_checkpoint`.

    Attributes:
        latent_dim: Width of a packed latent vector (``deter`` width + flattened ``stoch``
            width), matching the flat state the planner carries.

    Note:
        This class requires ``jax``, ``ninjax``, and the CarDreamer ``dreamerv3`` package
        importable in the running environment. It is intentionally not unit-tested against
        a live model here; the framework-agnostic
        :class:`~POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_dreamer_model_pomdp.DreamerWorldModel`
        protocol is covered by a lightweight fake instead.
    """

    def __init__(
        self,
        agent: Any,
        action_dim: int = 3,
        rng_seed: int = 0,
    ) -> None:
        """Wrap an already-constructed DreamerV3 JAX agent.

        Args:
            agent: A built DreamerV3 agent exposing the fitted parameters as ``agent.varibs``
                and the world model as ``agent.agent.wm`` (encoder, ``rssm``, and the
                ``decoder``/``reward``/``cont`` heads).
            action_dim: Width of the control vector fed to the RSSM (the CARLA
                ``(throttle, steer, brake)`` triple is 3-wide).
            rng_seed: Seed for the JAX PRNG driving the (sampling) RSSM/head calls.
        """
        # Heavy optional deps: import on build, not at module load.
        # pylint: disable=import-outside-toplevel,import-error
        import jax  # type: ignore[import]

        self._agent = agent
        self._world_model = agent.agent.wm
        self._action_dim = action_dim
        self._jax = jax
        self._rng_key = jax.random.PRNGKey(rng_seed)

        initial_state = self._run(lambda: self._world_model.rssm.initial(1))
        deter = np.asarray(initial_state["deter"])
        stoch = np.asarray(initial_state["stoch"])
        self._deter_dim: int = int(deter.reshape(1, -1).shape[1])
        self._stoch_shape: Tuple[int, ...] = tuple(stoch.shape[1:])
        self.latent_dim: int = self._deter_dim + int(np.prod(self._stoch_shape))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        obs_space: Mapping[str, Any],
        act_space: Mapping[str, Any],
        config_size: str = "medium",
        config_updates: Optional[Mapping[str, Any]] = None,
        step: int = 0,
        action_dim: int = 3,
        rng_seed: int = 0,
    ) -> "CarDreamerWorldModel":
        """Build the adapter from a CarDreamer/DreamerV3 training checkpoint.

        Constructs the DreamerV3 config (defaults + the named size preset + any overrides),
        instantiates the agent over the given observation/action spaces, and restores the
        fitted parameters from ``checkpoint_path`` via ``embodied.Checkpoint``.

        Args:
            checkpoint_path: Path to a DreamerV3 ``checkpoint.ckpt`` written during training.
            obs_space: The agent's observation space (``{name: embodied.Space}``); must
                include the CARLA schema keys ``gnss`` and ``agents``.
            act_space: The agent's action space (``{name: embodied.Space}``).
            config_size: DreamerV3 config size preset to load (e.g. ``"small"``,
                ``"medium"``, ``"large"``); must match the size the checkpoint was trained at.
            config_updates: Optional additional ``{"dreamerv3": {...}}`` config overrides.
            step: The environment step counter to seed the agent with.
            action_dim: Width of the control vector fed to the RSSM.
            rng_seed: Seed for the JAX PRNG driving the RSSM/head calls.

        Returns:
            A :class:`CarDreamerWorldModel` wrapping the restored agent.
        """
        # Heavy optional deps: import on build, not at module load.
        # pylint: disable=import-outside-toplevel,import-error
        import dreamerv3  # type: ignore[import]
        import embodied  # type: ignore[import]
        from ruamel import yaml  # type: ignore[import]

        config_root = embodied.Path(dreamerv3.__file__).parent / "dreamerv3.yaml"
        presets = yaml.YAML(typ="safe").load(config_root.read())
        config = embodied.Config({"dreamerv3": presets["defaults"]})
        config = config.update({"dreamerv3": presets[config_size]})
        if config_updates is not None:
            config = config.update(dict(config_updates))
        dreamer_config = config["dreamerv3"]

        agent = dreamerv3.Agent(dict(obs_space), dict(act_space), step, dreamer_config)
        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(checkpoint_path, keys=["agent"])
        return cls(agent, action_dim=action_dim, rng_seed=rng_seed)

    # ── DreamerWorldModel protocol ───────────────────────────────────────
    def encode(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Encode a real observation into a latent via the RSSM posterior (belief seed)."""
        batched = {key: np.asarray(value)[None, ...] for key, value in observation.items()}

        def posterior() -> Any:
            wm = self._world_model
            obs = {key: self._jax.numpy.asarray(value) for key, value in batched.items()}
            prev_state = wm.rssm.initial(1)
            prev_action = self._jax.numpy.zeros((1, self._action_dim))
            is_first = self._jax.numpy.ones((1,), dtype=bool)
            embed = wm.encoder(obs)
            post, _ = wm.rssm.obs_step(prev_state, prev_action, embed, is_first)
            return post

        post = self._run(posterior)
        return _pack_latents(np.asarray(post["deter"]), np.asarray(post["stoch"]))[0]

    def imagine(self, latents: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Advance ``(batch, latent_dim)`` latents under ``(batch, action_dim)`` controls."""
        controls_array = np.asarray(controls, dtype=np.float32)

        def step(state: Mapping[str, Any]) -> Any:
            actions = self._jax.numpy.asarray(controls_array)
            return self._world_model.rssm.img_step(state, actions)

        prior = self._run_over_state(latents, step)
        return _pack_latents(np.asarray(prior["deter"]), np.asarray(prior["stoch"]))

    def decode(self, latents: np.ndarray) -> Dict[str, np.ndarray]:
        """Decode ``(batch, latent_dim)`` latents to ``{gnss, agents}`` observation heads."""
        distributions = self._run_over_state(latents, self._decoder_distributions)
        return {key: np.asarray(distributions[key].mode()) for key in _OBSERVATION_KEYS}

    def decode_log_prob(
        self, latents: np.ndarray, observation: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Log-density of one observation under each of ``(batch, latent_dim)`` latents."""
        batch = _unpack_latents(latents, self._deter_dim, self._stoch_shape)[0].shape[0]
        targets = {
            key: np.broadcast_to(
                np.asarray(observation[key]), (batch, *np.asarray(observation[key]).shape)
            )
            for key in _OBSERVATION_KEYS
        }

        def log_prob(state: Mapping[str, Any]) -> Any:
            distributions = self._decoder_distributions(state)
            total = self._jax.numpy.zeros((batch,))
            for key in _OBSERVATION_KEYS:
                total = total + distributions[key].log_prob(self._jax.numpy.asarray(targets[key]))
            return total

        return np.asarray(self._run_over_state(latents, log_prob), dtype=float)

    def reward(self, latents: np.ndarray) -> np.ndarray:
        """Predicted reward for each of ``(batch, latent_dim)`` latents."""
        means = self._run_over_state(
            latents, lambda state: self._world_model.heads["reward"](state).mean()
        )
        return np.asarray(means, dtype=float).reshape(-1)

    def continue_prob(self, latents: np.ndarray) -> np.ndarray:
        """Probability the episode continues for each of ``(batch, latent_dim)`` latents."""
        means = self._run_over_state(
            latents, lambda state: self._world_model.heads["cont"](state).mean()
        )
        return np.asarray(means, dtype=float).reshape(-1)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _decoder_distributions(self, state: Mapping[str, Any]) -> Any:
        return self._world_model.heads["decoder"](state)

    def _state_from_latents(self, latents: np.ndarray) -> Dict[str, Any]:
        deter, stoch = _unpack_latents(latents, self._deter_dim, self._stoch_shape)
        return {
            "deter": self._jax.numpy.asarray(deter),
            "stoch": self._jax.numpy.asarray(stoch),
        }

    def _run_over_state(self, latents: np.ndarray, function: Any) -> Any:
        state = self._state_from_latents(latents)
        return self._run(lambda: function(state))

    def _run(self, function: Any) -> Any:
        # Heavy optional dep: import on call, not at module load.
        # pylint: disable=import-outside-toplevel,import-error
        import ninjax as nj  # type: ignore[import]

        self._rng_key, subkey = self._jax.random.split(self._rng_key)
        output, _ = nj.pure(function)(self._agent.varibs, subkey)
        return self._jax.device_get(output)


def build_cardreamer_model(
    checkpoint_path: str,
    obs_space: Mapping[str, Any],
    act_space: Mapping[str, Any],
    action_presets: Optional[Sequence[Tuple[float, float, float]]] = None,
    **kwargs: Any,
) -> CarDreamerWorldModel:
    """Convenience wrapper around :meth:`CarDreamerWorldModel.from_checkpoint`.

    Derives the RSSM action width from ``action_presets`` (defaulting to the 3-wide CARLA
    control triple) and forwards the rest to the checkpoint loader.

    Args:
        checkpoint_path: Path to a DreamerV3 training checkpoint.
        obs_space: The agent's observation space.
        act_space: The agent's action space.
        action_presets: Discrete control triples; only their width (3) is used here.
        **kwargs: Forwarded to :meth:`CarDreamerWorldModel.from_checkpoint`.

    Returns:
        The constructed :class:`CarDreamerWorldModel`.
    """
    action_dim = 3 if action_presets is None else len(action_presets[0])
    return CarDreamerWorldModel.from_checkpoint(
        checkpoint_path, obs_space, act_space, action_dim=action_dim, **kwargs
    )
