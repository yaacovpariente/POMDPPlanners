"""Benchmarks for WeightedParticleBelief / VectorizedWeightedParticleBelief
update paths across natively-ported (MountainCar, CartPole) envs.

Measures the cases laid out in the pomdp_native port plan:

    | Case                  | Env        | Belief class                        | Path            |
    |-----------------------|------------|-------------------------------------|-----------------|
    | MC-generic-cpp        | MountainCar| WeightedParticleBelief.update       | C++ batch       |
    | MC-vectorized-cpp     | MountainCar| VectorizedWeightedParticleBelief    | C++ batch       |
    | CP-generic-python     | CartPole   | WeightedParticleBelief.update       | pre-port Python |
    | CP-vectorized-numpy   | CartPole   | VectorizedWeightedParticleBelief    | pre-port numpy  |
    | CP-generic-cpp        | CartPole   | WeightedParticleBelief.update       | C++ batch       |
    | CP-vectorized-cpp     | CartPole   | VectorizedWeightedParticleBelief    | C++ batch       |

Same N=100 particles, same action, same observation across all cases of
the same env. The two CartPole "python" / "numpy" cases snapshot the
pre-port implementations (reproduced inline below) so they keep running
the pure-Python reference code even after the native port replaces the
shipped CartPole model classes. They give the baseline numbers the port
replaces; the two "cpp" cases sit on the shipped (post-port) module path.

Use ``pytest-benchmark compare`` across the cases to report:
    1. MC-generic-cpp vs MC-vectorized-cpp   -- auto-dispatch parity with
        the explicit vectorized path on a native env.
    2. CP-generic-python vs CP-vectorized-numpy -- the pre-port baseline
        gap (vectorization win before any native port).
    3. CP-generic-cpp vs CP-generic-python   -- headline speedup the
        native port buys for callers who use plain WeightedParticleBelief.
    4. CP-vectorized-cpp vs CP-vectorized-numpy -- speedup the native
        port buys for callers already on the vectorized path.
    5. CP-generic-cpp vs CP-vectorized-cpp   -- auto-dispatch overhead.

Run::

    pytest POMDPPlanners/tests/benchmarks/test_benchmark_particle_belief_update.py \\
        -m benchmark --benchmark-save=layer2_batch_dispatch -v
"""

import math
from typing import Any, List

import numpy as np
import pytest

from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_particle_belief_updater import (
    VectorizedParticleBeliefUpdater,
)
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.environment import ObservationModel, StateTransitionModel
from POMDPPlanners.environments.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp_beliefs import (
    CartPoleVectorizedUpdater,
)
from POMDPPlanners.environments.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp_beliefs import (
    MountainCarVectorizedUpdater,
)
from POMDPPlanners.utils.multivariate_normal import CovarianceParameterizedMultivariateNormal

pytestmark = [pytest.mark.slow]

_N_PARTICLES = 100


# ---------------------------------------------------------------------------
# MountainCar (native) cases
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="belief-update-mc-generic-cpp")
def test_bench_mc_generic_belief_update(benchmark):
    """Benchmark WeightedParticleBelief.update on MountainCar (auto-dispatch).

    Purpose: Measures the generic per-particle-looking belief update path
    on an env whose transition/observation models expose native batch
    entry points. WeightedParticleBelief._update_weights sniffs the batch
    interface and dispatches to C++ batch_sample / batch_log_likelihood in
    a single round-trip per update.

    Given: MountainCarPOMDP + WeightedParticleBelief with N=100 particles.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    particles = list(env.initial_state_dist().sample(n_samples=_N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    observation = np.array([-0.5, 0.0])

    def run():
        return belief.update(action=1, observation=observation, pomdp=env)

    benchmark(run)


@pytest.mark.benchmark(group="belief-update-mc-vectorized-cpp")
def test_bench_mc_vectorized_belief_update(benchmark):
    """Benchmark VectorizedWeightedParticleBelief.update on MountainCar.

    Purpose: Measures the explicit vectorized belief path on MountainCar.
    Its updater (MountainCarVectorizedUpdater) delegates batch_transition
    and batch_observation_log_likelihood directly to the native C++ batch
    methods.

    Given: MountainCarPOMDP + VectorizedWeightedParticleBelief with N=100.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = MountainCarPOMDP(discount_factor=0.99)
    updater = MountainCarVectorizedUpdater.from_environment(env)
    particles = np.array(env.initial_state_dist().sample(n_samples=_N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = VectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
    )
    observation = np.array([-0.5, 0.0])

    def run():
        return belief.update(action=1, observation=observation, pomdp=env)

    benchmark(run)


# ---------------------------------------------------------------------------
# CartPole cases
# ---------------------------------------------------------------------------
#
# The pre-port CartPole "python" / "numpy" benchmarks snapshot the Python /
# numpy implementations that used to ship before the native port, so the
# port's speedup stays measurable after the shipped classes switched to
# C++. These reference classes are lifted verbatim from the pre-port
# cartpole_pomdp.py / cartpole_pomdp_beliefs.py; they are test-only and
# intentionally do not reach into the shipped CartPoleStateTransition /
# CartPoleObservation (which are now C++-backed).


def _cartpole_initial_particles(env: CartPolePOMDP, n: int) -> Any:
    """Draw n initial CartPole state particles (ndarray or list format)."""
    return env.initial_state_dist().sample(n_samples=n)


class _PrePortCartPoleTransition(StateTransitionModel):
    # Pre-port Python reference kept for baseline benchmarking only.
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        state: np.ndarray,
        action: int,
        force_mag: float,
        total_mass: float,
        polemass_length: float,
        gravity: float,
        length: float,
        kinematics_integrator: str,
        tau: float,
        masspole: float,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
    ):
        super().__init__(state, action)
        self.force_mag = force_mag
        self.total_mass = total_mass
        self.polemass_length = polemass_length
        self.gravity = gravity
        self.length = length
        self.kinematics_integrator = kinematics_integrator
        self.tau = tau
        self.masspole = masspole
        self._state_transition_dist = state_transition_dist

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        deterministic = self._compute_deterministic_next_state()
        noise = self._state_transition_dist.sample(np.zeros(4), n_samples=n_samples)
        return [deterministic + noise[i] for i in range(n_samples)]

    def _compute_deterministic_next_state(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if self.action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        return np.array([x, x_dot, theta, theta_dot])

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        deterministic = self._compute_deterministic_next_state()
        values_array = np.array(values)
        return self._state_transition_dist.pdf(values_array, deterministic)


class _PrePortCartPoleObservation(ObservationModel):
    # Pre-port Python reference kept for baseline benchmarking only.
    def __init__(
        self,
        next_state: np.ndarray,
        action: int,
        obs_dist: CovarianceParameterizedMultivariateNormal,
    ):
        super().__init__(next_state=next_state, action=action)
        self.obs_dist = obs_dist

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        samples = self.obs_dist.sample(self.next_state, n_samples)
        return [samples[i] for i in range(n_samples)]

    def probability(self, values: List[np.ndarray]) -> np.ndarray:
        if len(values) == 0:
            return np.array([])
        values_array = np.array(values)
        return self.obs_dist.pdf(values_array, self.next_state)


class _PrePortCartPolePOMDP(CartPolePOMDP):
    """Pre-port CartPolePOMDP whose factories return the Python reference models."""

    def state_transition_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, state: np.ndarray, action: int
    ) -> StateTransitionModel:
        return _PrePortCartPoleTransition(
            state=state,
            action=action,
            force_mag=self.force_mag,
            total_mass=self.total_mass,
            polemass_length=self.polemass_length,
            gravity=self.gravity,
            length=self.length,
            kinematics_integrator=self.kinematics_integrator,
            tau=self.tau,
            masspole=self.masspole,
            state_transition_dist=self._state_transition_dist,
        )

    def observation_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, next_state: np.ndarray, action: int
    ) -> ObservationModel:
        return _PrePortCartPoleObservation(
            next_state=next_state, action=action, obs_dist=self._obs_dist
        )


class _PrePortCartPoleVectorizedUpdater(VectorizedParticleBeliefUpdater):
    """Pre-port numpy-only vectorized updater preserved for baseline benchmarks."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        state_transition_dist: CovarianceParameterizedMultivariateNormal,
        obs_dist: CovarianceParameterizedMultivariateNormal,
        force_mag: float,
        gravity: float,
        masspole: float,
        total_mass: float,
        length: float,
        polemass_length: float,
        tau: float,
        kinematics_integrator: str,
    ):
        self.state_transition_dist = state_transition_dist
        self.obs_dist = obs_dist
        self.force_mag = force_mag
        self.gravity = gravity
        self.masspole = masspole
        self.total_mass = total_mass
        self.length = length
        self.polemass_length = polemass_length
        self.tau = tau
        self.kinematics_integrator = kinematics_integrator

    @classmethod
    def from_environment(cls, env: CartPolePOMDP) -> "_PrePortCartPoleVectorizedUpdater":
        # pylint: disable=protected-access
        return cls(
            state_transition_dist=env._state_transition_dist,
            obs_dist=env._obs_dist,
            force_mag=env.force_mag,
            gravity=env.gravity,
            masspole=env.masspole,
            total_mass=env.total_mass,
            length=env.length,
            polemass_length=env.polemass_length,
            tau=env.tau,
            kinematics_integrator=env.kinematics_integrator,
        )

    def batch_transition(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        next_particles = self._deterministic_next_state(particles, action)
        noise = self.state_transition_dist.sample(np.zeros(4), n_samples=particles.shape[0])
        return next_particles + noise

    def _deterministic_next_state(self, particles: np.ndarray, action: np.ndarray) -> np.ndarray:
        force = self.force_mag if action == 1 else -self.force_mag
        theta = particles[:, 2]
        theta_dot = particles[:, 3]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass
        next_particles = np.empty_like(particles)
        if self.kinematics_integrator == "euler":
            next_particles[:, 0] = particles[:, 0] + self.tau * particles[:, 1]
            next_particles[:, 1] = particles[:, 1] + self.tau * x_acc
            next_particles[:, 2] = particles[:, 2] + self.tau * particles[:, 3]
            next_particles[:, 3] = particles[:, 3] + self.tau * theta_acc
        else:
            next_particles[:, 1] = particles[:, 1] + self.tau * x_acc
            next_particles[:, 0] = particles[:, 0] + self.tau * next_particles[:, 1]
            next_particles[:, 3] = particles[:, 3] + self.tau * theta_acc
            next_particles[:, 2] = particles[:, 2] + self.tau * next_particles[:, 3]
        return next_particles

    def batch_observation_log_likelihood(
        self,
        next_particles: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        observation_arr = np.asarray(observation, dtype=float).ravel()
        return self.obs_dist.log_pdf(next_particles, observation_arr)

    @property
    def config_id(self) -> str:
        return "_PrePortCartPoleVectorizedUpdater"


@pytest.mark.benchmark(group="belief-update-cp-generic-python")
def test_bench_cp_generic_belief_update_python(benchmark):
    """Benchmark WeightedParticleBelief.update on CartPole (pre-port Python).

    Purpose: Pre-port baseline. Measures the generic per-particle Python
    loop using the pre-port reference CartPole transition / observation
    classes (kept inline in this module). These models do NOT expose
    batch_sample, so WeightedParticleBelief._update_weights falls back to
    its per-particle ``state_transition_model().sample()`` loop.

    Given: Pre-port CartPolePOMDP + WeightedParticleBelief with N=100.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = _PrePortCartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    particles = list(_cartpole_initial_particles(env, _N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    action = env.get_actions()[0]
    observation = env.initial_observation_dist().sample(n_samples=1)[0]

    def run():
        return belief.update(action=action, observation=observation, pomdp=env)

    benchmark(run)


@pytest.mark.benchmark(group="belief-update-cp-vectorized-numpy")
def test_bench_cp_vectorized_belief_update_numpy(benchmark):
    """Benchmark VectorizedWeightedParticleBelief.update on CartPole (pre-port numpy).

    Purpose: Pre-port baseline. Measures the explicit vectorized belief
    path using the numpy-only reference updater (kept inline in this
    module). ``batch_transition`` and ``batch_observation_log_likelihood``
    run entirely in numpy -- no C++ involved.

    Given: CartPolePOMDP + VectorizedWeightedParticleBelief with N=100.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    updater = _PrePortCartPoleVectorizedUpdater.from_environment(env)
    particles = np.array(_cartpole_initial_particles(env, _N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = VectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
    )
    action = env.get_actions()[0]
    observation = env.initial_observation_dist().sample(n_samples=1)[0]

    def run():
        return belief.update(action=action, observation=observation, pomdp=env)

    benchmark(run)


@pytest.mark.benchmark(group="belief-update-cp-generic-cpp")
def test_bench_cp_generic_belief_update_cpp(benchmark):
    """Benchmark WeightedParticleBelief.update on CartPole (C++ auto-dispatch).

    Purpose: Measures the generic per-particle-looking belief update path
    on the post-port CartPole, where the shipped transition / observation
    models expose native batch entry points.
    WeightedParticleBelief._update_weights sniffs the batch interface and
    dispatches to C++ batch_sample / batch_log_likelihood in a single
    round-trip per update.

    Given: CartPolePOMDP + WeightedParticleBelief with N=100 particles.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    particles = list(_cartpole_initial_particles(env, _N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights)
    action = env.get_actions()[0]
    observation = env.initial_observation_dist().sample(n_samples=1)[0]

    def run():
        return belief.update(action=action, observation=observation, pomdp=env)

    benchmark(run)


@pytest.mark.benchmark(group="belief-update-cp-vectorized-cpp")
def test_bench_cp_vectorized_belief_update_cpp(benchmark):
    """Benchmark VectorizedWeightedParticleBelief.update on CartPole (C++).

    Purpose: Measures the explicit vectorized belief path on the post-port
    CartPole. Its updater (CartPoleVectorizedUpdater) now delegates
    batch_transition and batch_observation_log_likelihood directly to the
    native C++ batch methods.

    Given: CartPolePOMDP + VectorizedWeightedParticleBelief with N=100.
    When: belief.update(action, observation, pomdp) is called repeatedly.
    Then: Execution time is recorded.

    Test type: performance
    """
    env = CartPolePOMDP(discount_factor=0.99, noise_cov=np.diag([0.1, 0.1, 0.1, 0.1]))
    updater = CartPoleVectorizedUpdater.from_environment(env)
    particles = np.array(_cartpole_initial_particles(env, _N_PARTICLES))
    log_weights = np.log(np.ones(_N_PARTICLES) / _N_PARTICLES)
    belief = VectorizedWeightedParticleBelief(
        particles=particles,
        log_weights=log_weights,
        updater=updater,
    )
    action = env.get_actions()[0]
    observation = env.initial_observation_dist().sample(n_samples=1)[0]

    def run():
        return belief.update(action=action, observation=observation, pomdp=env)

    benchmark(run)
