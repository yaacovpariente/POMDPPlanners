"""Vectorized particle belief updater for the PacMan POMDP.

Batched state transitions and observation log-likelihoods dispatch to the
native C++ kernels in :mod:`POMDPPlanners.environments.pacman_pomdp._native`
(``PacManTransitionCpp.batch_sample`` and
``PacManObservationCpp.batch_log_likelihood``). The Python side pre-computes
and caches the per-env ctor kwargs once in :meth:`__init__` so the per-batch
construction of the native objects is O(1).

Classes:
    PacManVectorizedUpdater: Batched updater for the PacMan POMDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.environments.pacman_pomdp import _native  # pylint: disable=no-name-in-module
from POMDPPlanners.utils.config_to_id import config_to_id

if TYPE_CHECKING:
    from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP


class PacManVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Vectorized particle belief updater for PacMan POMDP, native-backed.

    Performs all-particle transitions and observation log-likelihoods by
    dispatching to the ``PacManTransitionCpp`` and ``PacManObservationCpp``
    native kernels.

    Attributes:
        maze_size: Grid dimensions (rows, cols).
        num_ghosts: Number of ghosts.
        num_pellets: Number of initial pellets.
        state_dim: Dimensionality of the array state.
        ghost_aggressiveness: Softmax temperature for ghost pursuit.
        ghost_coordination: Ghost coordination mode.
        ghost_strategies: Per-ghost strategy list.
        observation_noise_factor: Multiplier for observation noise.
        max_observation_noise: Maximum observation noise std.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        maze_size: Tuple[int, int],
        num_ghosts: int,
        num_pellets: int,
        state_dim: int,
        ghost_aggressiveness: float,
        ghost_coordination: str,
        ghost_strategies: List[str],
        observation_noise_factor: float,
        max_observation_noise: float,
        transition_ctor_kwargs: Dict[str, Any],
        observation_ctor_kwargs: Dict[str, Any],
        patrol_dir_state: np.ndarray,
    ):
        self.maze_size = maze_size
        self.num_ghosts = num_ghosts
        self.num_pellets = num_pellets
        self.state_dim = state_dim
        self.ghost_aggressiveness = ghost_aggressiveness
        self.ghost_coordination = ghost_coordination
        self.ghost_strategies = ghost_strategies
        self.observation_noise_factor = observation_noise_factor
        self.max_observation_noise = max_observation_noise
        self._transition_ctor_kwargs = transition_ctor_kwargs
        self._observation_ctor_kwargs = observation_ctor_kwargs
        self._patrol_dir_state = patrol_dir_state
        # Layout indices mirrored on the updater for diagnostics and tests;
        # all pulled from the transition ctor kwargs so they stay in sync.
        self._idx_pac_row: int = transition_ctor_kwargs["idx_pac_row"]
        self._idx_pac_col: int = transition_ctor_kwargs["idx_pac_col"]
        self._idx_ghosts_start: int = transition_ctor_kwargs["idx_ghosts_start"]
        self._idx_pellets_start: int = transition_ctor_kwargs["idx_pellets_start"]
        self._idx_pellets_end: int = transition_ctor_kwargs["idx_pellets_end"]
        self._idx_score: int = transition_ctor_kwargs["idx_score"]
        self._idx_terminal: int = transition_ctor_kwargs["idx_terminal"]
        # Scratch zero-state used as the throwaway ``state`` / ``next_state``
        # argument to the native ctor on the batch paths (the batch methods
        # ignore per-instance state; only the batch tensor matters).
        self._scratch_state = np.zeros(state_dim, dtype=np.float64)
        self._scratch_obs = np.zeros(2 * num_ghosts, dtype=np.float64)

    @classmethod
    def from_environment(cls, env: "PacManPOMDP") -> "PacManVectorizedUpdater":
        """Construct an updater from a PacManPOMDP instance."""
        return cls(
            maze_size=env.maze_size,
            num_ghosts=env.num_ghosts,
            num_pellets=env._num_initial_pellets,  # pylint: disable=protected-access
            state_dim=env._state_dim,  # pylint: disable=protected-access
            ghost_aggressiveness=env.ghost_aggressiveness,
            ghost_coordination=env.ghost_coordination,
            ghost_strategies=list(env.ghost_strategies),
            observation_noise_factor=env.observation_noise_factor,
            max_observation_noise=env.max_observation_noise,
            transition_ctor_kwargs=env.get_transition_cpp_ctor_kwargs(),
            observation_ctor_kwargs=env.get_observation_cpp_ctor_kwargs(),
            patrol_dir_state=env.ghost_patrol_directions,
        )

    # ------------------------------------------------------------------
    # VectorizedParticleBeliefUpdater interface
    # ------------------------------------------------------------------

    def batch_transition(  # pylint: disable=arguments-renamed
        self, particles: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        particles_f = np.ascontiguousarray(particles, dtype=np.float64)
        if particles_f.shape[0] == 0:
            return particles_f
        ref_state = particles_f[0] if particles_f.shape[0] > 0 else self._scratch_state
        tm = _native.PacManTransitionCpp(
            state=ref_state,
            action=int(action),
            **self._transition_ctor_kwargs,
            patrol_dir_state=self._patrol_dir_state,
        )
        return tm.batch_sample(particles_f)

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,  # pylint: disable=unused-argument
        observation: np.ndarray,
    ) -> np.ndarray:
        del action  # observation likelihood is action-independent for PacMan
        particles_f = np.ascontiguousarray(next_particles, dtype=np.float64)
        obs_f = np.ascontiguousarray(observation, dtype=np.float64).ravel()
        ref_state = particles_f[0] if particles_f.shape[0] > 0 else self._scratch_state
        obs_model = _native.PacManObservationCpp(
            next_state=ref_state,
            action=0,  # unused on the batch path
            **self._observation_ctor_kwargs,
        )
        return obs_model.batch_log_likelihood(particles_f, obs_f)

    @property
    def config_id(self) -> str:
        config_dict = {
            "class": "PacManVectorizedUpdater",
            "maze_size": list(self.maze_size),
            "num_ghosts": self.num_ghosts,
            "num_pellets": self.num_pellets,
            "ghost_aggressiveness": self.ghost_aggressiveness,
            "ghost_coordination": self.ghost_coordination,
            "ghost_strategies": self.ghost_strategies,
            "observation_noise_factor": self.observation_noise_factor,
            "max_observation_noise": self.max_observation_noise,
        }
        return config_to_id(config_dict)
