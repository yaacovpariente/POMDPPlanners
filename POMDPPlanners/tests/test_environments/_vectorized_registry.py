# SPDX-License-Identifier: MIT

"""Registry of every vectorized duplicate of an environment's logic.

Most environments here are written more than once. The scalar
:class:`~POMDPPlanners.core.environment.Environment` is the reference; a
:class:`~POMDPPlanners.core.belief.vectorized_particle_belief_updater.VectorizedParticleBeliefUpdater`
re-expresses its transition and observation model over a particle array for
the belief, and a
:class:`~POMDPPlanners.core.environment.vectorized_generative_model.VectorizedGenerativeModel`
re-expresses the same kernels in torch for VOPP. Each copy is hand-written, so
each is a place for the two to drift apart.

This module lists every such copy with the few callbacks a generic test needs
to compare it with its scalar environment, and nothing about what the
environment does:

* :data:`BELIEF_SPECS` -- one entry per environment variant with a vectorized
  belief. The belief and its updater are built through the public
  :func:`~POMDPPlanners.utils.belief_factory.create_environment_belief`, so the
  test exercises exactly what a simulation runs. The per-environment
  knowledge is how an observation is handed to the updater (mirroring the
  belief's own ``update``), and flags saying which comparisons can be exact:
  whether the batch and scalar paths share an RNG order, and whether the
  belief's update is the plain reweighting algorithm.
* :data:`MODEL_SPECS` -- one entry per vectorized generative model, with the
  conversions between the model's tensors and the scalar environment's values
  (action index to action, state row to state, observation row to
  observation).

:func:`discover_vectorized_classes` scans the source tree for every concrete
updater, vectorized belief and generative model, and the coverage tests in
``test_vectorized_belief_conformance.py`` and
``test_vectorized_model_conformance.py`` require each one to be either
registered here or on a commented exclusion list. A new vectorized copy cannot
escape the conformance suite silently.

Known disagreements are recorded on the allowlists below and run as
``xfail(strict=True)``: the suite turns red the moment one is fixed, so the
list cannot outlive the bug.
"""

import ast
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.tests.test_environments.test_env_api_conformance import ENV_BUILDERS

EnvBuilder = Callable[[], Environment]

# A recorded disagreement: the reason, and the exception the failing test
# raises. The exception type is passed to ``xfail(raises=...)`` so that an
# unrelated new failure in the same test is not absorbed by an old entry.
Disagreement = Tuple[str, Type[BaseException]]
ObservationForUpdater = Callable[[Any], Any]

_ENV_BUILDER_BY_ID: Dict[str, EnvBuilder] = dict(ENV_BUILDERS)


# ---------------------------------------------------------------------------
# Belief-side registry
# ---------------------------------------------------------------------------


def _float_array(observation: Any) -> np.ndarray:
    """Default observation hand-off: what ``VectorizedWeightedParticleBelief.update`` does."""
    return np.asarray(observation, dtype=float)


def _raw(observation: Any) -> Any:
    """Hand the observation over unchanged, for beliefs whose ``update`` does so."""
    return observation


def _rock_sample_observation(observation: Any) -> np.ndarray:
    # Mirrors RockSampleVectorizedWeightedParticleBelief.update: the string
    # reading is encoded to its integer label before it reaches the updater.
    from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp_beliefs import (
        rocksample_belief_factory,
    )

    if isinstance(observation, str):
        # pylint: disable-next=protected-access
        observation = rocksample_belief_factory._OBS_ENCODING[observation]
    return np.asarray(observation, dtype=float)


def _maze_observation(observation: Any) -> np.ndarray:
    # Mirrors MazeVectorizedWeightedParticleBelief.update.
    from POMDPPlanners.environments.maze_pomdp.maze_pomdp_beliefs import maze_belief_factory

    if isinstance(observation, str):
        # pylint: disable-next=protected-access
        observation = maze_belief_factory._OBS_ENCODING.get(
            observation,
            maze_belief_factory._UNKNOWN_OBSERVATION,  # pylint: disable=protected-access
        )
    return np.asarray(observation, dtype=np.float64)


def _occupancy_grid_mapping_belief(particles: List[Any], log_weights: np.ndarray) -> Any:
    # The occupancy reading is a whole range scan that a freshly predicted
    # particle never reproduces exactly, so a plain predict-and-reweight filter
    # loses all support on the first step. The environment's own scalar
    # filter conditions on the reading instead; it is the general
    # implementation the vectorized belief duplicates.
    from POMDPPlanners.environments.occupancy_grid_mapping_pomdp.occupancy_grid_mapping_belief import (
        OccupancyGridMappingBelief,
    )

    return OccupancyGridMappingBelief(particles=particles, log_weights=log_weights)


ReferenceBelief = Callable[[List[Any], np.ndarray], Any]


@dataclass(frozen=True)
class BeliefSpec:
    """One environment variant whose belief has a vectorized implementation.

    Attributes:
        env_id: Test id. For variants in ``ENV_BUILDERS`` it is that registry's
            id and the environment is built by its builder.
        updater_class: Name of the updater class the belief factory must
            return. Pinned so the coverage test can see which classes are
            exercised, and so a factory that silently swaps updaters fails.
        belief_class: Name of the vectorized belief class the factory returns.
        observation_for_updater: Converts an environment observation into the
            argument the belief's ``update`` passes to
            ``batch_observation_log_likelihood``.
        transition_shares_rng: ``True`` when ``batch_transition`` consumes the
            seeded RNGs exactly as a loop over ``sample_next_state`` does, so the
            two can be compared value for value. ``False`` means only the
            distributions are comparable. This is a statement about RNG order,
            not about correctness.
        update_matches_reference: ``True`` when the vectorized ``update`` runs
            the same algorithm as ``reference_belief`` -- so with a shared seed
            and a shared RNG order the two produce the same particles and
            weights. Beliefs that redraw, condition on part of the reading,
            stratify resampling or reinvigorate are compared on the posterior
            distribution only.
        reference_belief: Builds the environment's own general (scalar)
            belief from a particle list and log-weights, for environments whose
            reading makes a plain particle filter degenerate. ``None`` means
            ``WeightedParticleBelief`` in the seeded update check and a plain
            particle filter on the scalar interface in the posterior check.
        prior_holds_true_state: ``True`` puts the true state on half of the
            prior's particles in the posterior check. Needed when the reading
            is sharp relative to the prior: without it only a handful of
            particles explain the reading, the reference posterior's effective
            sample size is tiny, and the comparison has no power to fail (the
            check asserts it has). Off by default, because it changes the
            prior, and a belief that redraws from an exact posterior
            (Battleship) ignores the prior's weights.
        builder: Builds the environment when the variant is not in
            ``ENV_BUILDERS``.
    """

    env_id: str
    updater_class: str
    belief_class: str
    observation_for_updater: ObservationForUpdater = _float_array
    transition_shares_rng: bool = True
    update_matches_reference: bool = True
    reference_belief: Optional[ReferenceBelief] = None
    prior_holds_true_state: bool = False
    builder: Optional[EnvBuilder] = None

    def build_env(self) -> Environment:
        """Build the scalar environment."""
        if self.builder is not None:
            return self.builder()
        return _ENV_BUILDER_BY_ID[self.env_id]()


_PLAIN = "VectorizedWeightedParticleBelief"


# The light-dark observation models each have their own updater class, but
# ENV_BUILDERS only builds the default one. These builders add the others.
def _continuous_light_dark_with(observation_model: str) -> EnvBuilder:
    def build() -> Environment:
        from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
            ContinuousLightDarkPOMDP,
            ObservationModelType,
        )
        from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
            continuous_light_dark_pinned_kwargs,
        )

        return ContinuousLightDarkPOMDP(
            discount_factor=0.95,
            **continuous_light_dark_pinned_kwargs(
                observation_model_type=ObservationModelType[observation_model]
            ),
        )

    return build


def _discrete_light_dark_with(observation_model: str) -> EnvBuilder:
    def build() -> Environment:
        from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
            DiscreteLightDarkPOMDP,
            ObservationModelType,
        )
        from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
            discrete_light_dark_pinned_kwargs,
        )

        return DiscreteLightDarkPOMDP(
            discount_factor=0.95,
            **discrete_light_dark_pinned_kwargs(
                observation_model_type=ObservationModelType[observation_model]
            ),
        )

    return build


# Continuous light-dark: position noise comes from the native RNG and the
# obstacle-hit roll from numpy, in a different order than the scalar step.
# Discrete light-dark: the batch path draws slips with random() + choice(), the
# scalar step with one rand() and a searchsorted. Neither is a bug.
_LIGHT_DARK_RNG = {"transition_shares_rng": False}

# The reading is sharp relative to the prior (a precise range scan, a
# low-noise position fix, a hundred noisy cells at once): see
# BeliefSpec.prior_holds_true_state. Every entry whose reference posterior's
# effective sample size fell below what the check needs on some seed has it.
_SHARP_READING = {"prior_holds_true_state": True}

BELIEF_SPECS: List[BeliefSpec] = [
    BeliefSpec(
        "BattleshipPOMDP",
        "BattleshipVectorizedUpdater",
        "BattleshipVectorizedWeightedParticleBelief",
        # Redraws every particle from the exact posterior over legal layouts.
        update_matches_reference=False,
    ),
    BeliefSpec(
        "CaptureTheFlagPOMDP",
        "CaptureTheFlagVectorizedUpdater",
        "CaptureTheFlagVectorizedBelief",
        # Both teams move stochastically; the draws are made in another order.
        transition_shares_rng=False,
        # Writes the exactly-observed components into the particles and
        # resamples within each flag candidate.
        update_matches_reference=False,
        **_SHARP_READING,
    ),
    BeliefSpec("CartPolePOMDP", "CartPoleVectorizedUpdater", _PLAIN),
    BeliefSpec(
        "ChicheckInvadersPOMDP",
        "ChicheckInvadersVectorizedUpdater",
        "ChicheckInvadersVectorizedBelief",
        # Re-draws the unreported chickens of a fraction of the particles.
        update_matches_reference=False,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "ChicheckInvadersPOMDP[fully_observable]",
        "ChicheckInvadersVectorizedUpdater",
        "ChicheckInvadersVectorizedBelief",
        # Collapses onto the observed state.
        update_matches_reference=False,
        # The reading is the state itself.
        **_SHARP_READING,
    ),
    BeliefSpec(
        "ContinuousLaserTagPOMDP",
        "ContinuousLaserTagVectorizedUpdater",
        _PLAIN,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "ContinuousLaserTagPOMDPDiscreteActions",
        "ContinuousLaserTagVectorizedUpdater",
        _PLAIN,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "ContinuousLightDarkPOMDP",
        "ContinuousLightDarkVectorizedUpdater",
        _PLAIN,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "ContinuousLightDarkPOMDP[distance_based]",
        "ContinuousLightDarkDistanceBasedVectorizedUpdater",
        _PLAIN,
        builder=_continuous_light_dark_with("DISTANCE_BASED"),
        # The updater scores the env's "None" (no reading) label directly.
        observation_for_updater=_raw,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "ContinuousLightDarkPOMDP[no_obs_in_dark]",
        "ContinuousLightDarkNoObsInDarkVectorizedUpdater",
        _PLAIN,
        builder=_continuous_light_dark_with("NORMAL_NOISE_NO_OBS_IN_DARK"),
        observation_for_updater=_raw,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "ContinuousLightDarkPOMDPDiscreteActions",
        "ContinuousLightDarkVectorizedUpdater",
        _PLAIN,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "ContinuousMazePOMDP",
        "ContinuousMazeVectorizedUpdater",
        "MazeVectorizedWeightedParticleBelief",
        observation_for_updater=_maze_observation,
    ),
    BeliefSpec("ContinuousPushPOMDP", "ContinuousPushVectorizedUpdater", _PLAIN, **_SHARP_READING),
    BeliefSpec(
        "ContinuousPushPOMDPDiscreteActions",
        "ContinuousPushVectorizedUpdater",
        _PLAIN,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "DiscreteLightDarkPOMDP",
        "DiscreteLightDarkVectorizedUpdater",
        _PLAIN,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "DiscreteLightDarkPOMDP[distance_based]",
        "DiscreteLightDarkDistanceBasedVectorizedUpdater",
        _PLAIN,
        builder=_discrete_light_dark_with("DISTANCE_BASED"),
        observation_for_updater=_raw,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "DiscreteLightDarkPOMDP[no_obs_in_dark]",
        "DiscreteLightDarkNoObsInDarkVectorizedUpdater",
        _PLAIN,
        builder=_discrete_light_dark_with("NO_OBS_IN_DARK"),
        observation_for_updater=_raw,
        **_LIGHT_DARK_RNG,
    ),
    BeliefSpec(
        "DiscreteMazePOMDP",
        "DiscreteMazeVectorizedUpdater",
        "MazeVectorizedWeightedParticleBelief",
        observation_for_updater=_maze_observation,
    ),
    BeliefSpec(
        "FirefightingPOMDP",
        "FirefightingVectorizedUpdater",
        "FirefightingVectorizedBelief",
        # Slip, spread and burn-out are drawn in another order.
        transition_shares_rng=False,
        # Writes the noiseless robot fields into the particles and resamples
        # within each wind value.
        update_matches_reference=False,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "FirefightingPOMDP[3 robots]",
        "FirefightingVectorizedUpdater",
        "FirefightingVectorizedBelief",
        transition_shares_rng=False,
        update_matches_reference=False,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "LaserTagPOMDP",
        "LaserTagVectorizedUpdater",
        _PLAIN,
        # The batch path draws the opponent's move from the native RNG; the
        # scalar step draws it from numpy.
        transition_shares_rng=False,
        **_SHARP_READING,
    ),
    BeliefSpec("MountainCarPOMDP", "MountainCarVectorizedUpdater", _PLAIN),
    BeliefSpec(
        "OccupancyGridMappingPOMDP",
        "OccupancyGridMappingVectorizedUpdater",
        "OccupancyGridMappingVectorizedBelief",
        reference_belief=_occupancy_grid_mapping_belief,
        **_SHARP_READING,
    ),
    BeliefSpec(
        "OccupancyGridMappingPOMDP[truncated_normal]",
        "OccupancyGridMappingVectorizedUpdater",
        "OccupancyGridMappingVectorizedBelief",
        reference_belief=_occupancy_grid_mapping_belief,
        **_SHARP_READING,
    ),
    BeliefSpec("PacManPOMDP", "PacManVectorizedUpdater", _PLAIN),
    BeliefSpec("PushPOMDP", "PushVectorizedUpdater", _PLAIN, **_SHARP_READING),
    BeliefSpec(
        "RockSamplePOMDP",
        "RockSampleVectorizedUpdater",
        "RockSampleVectorizedWeightedParticleBelief",
        observation_for_updater=_rock_sample_observation,
    ),
    BeliefSpec(
        "SafeAntVelocityPOMDP", "SafetyAntVelocityVectorizedUpdater", _PLAIN, **_SHARP_READING
    ),
    BeliefSpec(
        "SnakePOMDP",
        "SnakeVectorizedUpdater",
        "SnakeVectorizedWeightedParticleBelief",
        observation_for_updater=_raw,
        # Food respawn is drawn with random() * count; the scalar step uses
        # randint.
        transition_shares_rng=False,
        # Redraws the food when a reading rules out every particle.
        update_matches_reference=False,
        **_SHARP_READING,
    ),
]


# Environments in ENV_BUILDERS with no vectorized belief, and why. The belief
# coverage test checks this list against the registry, so an environment that
# gains a vectorized belief must move from here into BELIEF_SPECS.
ENVS_WITHOUT_VECTORIZED_BELIEF: Dict[str, str] = {
    "SanityPOMDP": "scalar int states; the generic WeightedParticleBelief is the belief",
    "TMazePOMDP": "no vectorized updater written; the factory falls back to WeightedParticleBelief",
    "TigerPOMDP": "string states; the generic WeightedParticleBelief is the belief",
}


# Concrete updater / belief classes found in the source tree that no
# BeliefSpec exercises, and why. Each is a decision, not a gap.
BELIEF_CLASS_EXCLUSIONS: Dict[str, str] = {
    # Abstract: holds the shared maze kernels; both concrete subclasses are
    # registered.
    "BaseMazeVectorizedUpdater": "abstract base of the two registered maze updaters",
}


# ---------------------------------------------------------------------------
# Belief-side known disagreements (xfail(strict=True)).
#
# Each entry is a real disagreement between a vectorized copy and the scalar
# environment, found by this suite. The comment says what disagrees. Remove the
# entry when the fix lands; strict xfail makes the test fail until you do.
# ---------------------------------------------------------------------------

OBSERVATION_LIKELIHOOD_DISAGREEMENTS: Dict[str, Disagreement] = {}

# The light-dark "no reading" models emit the string "None" when the agent is
# in the dark. Their updaters score "None" correctly, but the belief factory
# wraps them in a plain VectorizedWeightedParticleBelief, whose update runs
# np.asarray(observation, dtype=float) first and raises ValueError. A
# simulation on either config with the default belief crashes on the first
# dark step.
_LIGHT_DARK_NONE_READING: Disagreement = (
    "VectorizedWeightedParticleBelief.update converts the env's 'None' reading "
    "with np.asarray(..., dtype=float) and raises ValueError; the updater itself "
    "handles 'None'",
    ValueError,
)
OBSERVATION_HANDLING_DISAGREEMENTS: Dict[str, Disagreement] = {
    "ContinuousLightDarkPOMDP[distance_based]": _LIGHT_DARK_NONE_READING,
    "ContinuousLightDarkPOMDP[no_obs_in_dark]": _LIGHT_DARK_NONE_READING,
    "DiscreteLightDarkPOMDP[distance_based]": _LIGHT_DARK_NONE_READING,
    "DiscreteLightDarkPOMDP[no_obs_in_dark]": _LIGHT_DARK_NONE_READING,
}
TRANSITION_DISTRIBUTION_DISAGREEMENTS: Dict[str, Disagreement] = {}
TRANSITION_SHARED_SEED_DISAGREEMENTS: Dict[str, Disagreement] = {}
BELIEF_UPDATE_DISAGREEMENTS: Dict[str, Disagreement] = {}
# No static entry for the light-dark "None" crash: whether the posterior
# test's seeded draw lands in the dark depends on the platform's RNG stream
# (it does on Linux for the continuous variants and not on macOS). The test
# confirms the crash on a "None" reading and xfails at runtime instead; see
# OBSERVATION_HANDLING_DISAGREEMENTS.
POSTERIOR_DISTRIBUTION_DISAGREEMENTS: Dict[str, Disagreement] = {}


# ---------------------------------------------------------------------------
# Source-tree discovery
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_ENVIRONMENTS_ROOT = _PACKAGE_ROOT / "environments"

UPDATER_ROOT = "VectorizedParticleBeliefUpdater"
BELIEF_ROOT = "VectorizedWeightedParticleBelief"

# A class satisfies the VectorizedGenerativeModel protocol structurally, so it
# is recognised by the methods it defines rather than by a base class.
_MODEL_METHODS = frozenset(
    {
        "sample_next_states",
        "sample_observations",
        "rewards",
        "terminal_mask",
        "observation_log_probs",
    }
)


@dataclass
class DiscoveredClasses:
    """Vectorized classes defined under ``POMDPPlanners/environments``.

    Attributes:
        updaters: Concrete and abstract subclasses of the updater ABC.
        beliefs: Subclasses of ``VectorizedWeightedParticleBelief``.
        models: Classes defining every ``VectorizedGenerativeModel`` kernel.
    """

    updaters: Set[str] = field(default_factory=set)
    beliefs: Set[str] = field(default_factory=set)
    models: Set[str] = field(default_factory=set)


def discover_vectorized_classes() -> DiscoveredClasses:
    """Find every vectorized class in the environments package by parsing source.

    Parsing rather than importing keeps the scan complete when an optional
    simulator (Isaac, CARLA, highway-env) is missing: a module that cannot be
    imported still has its classes found.

    Returns:
        The discovered class names, grouped by kind.
    """
    bases_of: Dict[str, Set[str]] = {}
    methods_of: Dict[str, Set[str]] = {}
    for path in sorted(_ENVIRONMENTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases_of[node.name] = {_base_name(base) for base in node.bases}
                methods_of[node.name] = {
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }

    found = DiscoveredClasses()
    found.updaters = _descendants(UPDATER_ROOT, bases_of)
    found.beliefs = _descendants(BELIEF_ROOT, bases_of)
    found.models = {name for name, methods in methods_of.items() if _MODEL_METHODS <= methods}
    return found


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return ""


def _descendants(root: str, bases_of: Dict[str, Set[str]]) -> Set[str]:
    known = {root}
    changed = True
    while changed:
        changed = False
        for name, bases in bases_of.items():
            if name not in known and bases & known:
                known.add(name)
                changed = True
    return known - {root}


# ---------------------------------------------------------------------------
# Model-side registry
# ---------------------------------------------------------------------------


def _copy_row(env: Any, row: np.ndarray) -> np.ndarray:
    """Default row-to-value conversion: the value is the row itself."""
    del env
    return row.copy()


def _float_row(env: Any, value: Any) -> np.ndarray:
    """Default value-to-row conversion: the value's flattened float64 array."""
    del env
    return np.asarray(value, dtype=np.float64).ravel()


@dataclass(frozen=True)
class ModelSpec:
    """One vectorized generative model and its scalar environment.

    The conversions are the only environment knowledge the model conformance
    test uses. They translate between the model's tensor encoding and the
    scalar environment's values; none of them re-implements any dynamics.

    Attributes:
        model_id: Test id.
        model_class: Name of the model class, for the coverage test.
        build: Returns ``(env, model)``, the model on CPU in float64. Imports
            what it needs lazily, so a missing optional simulator surfaces as
            ``ImportError`` and the test skips.
        action_of_index: ``(env, model, index) -> action`` for the scalar env.
        state_of_row: ``(env, row) -> state``; ``row`` is a 1-D float64 array.
        row_of_state: ``(env, state) -> row`` as a 1-D float64 array.
        observation_of_row: ``(env, row) -> observation`` for the scalar env.
        row_of_observation: ``(env, observation) -> row`` as a 1-D float64 array.
        num_actions: ``(env, model) -> int``.
        required_modules: Optional simulator modules; the test skips when one
            is missing.
        probe_states: ``env -> states`` checked first, before the rollout
            states. For a branch short random rollouts rarely reach; it is
            data, and the checks treat it like any other state.
        observation_not_comparable: Why the model's observation kernels have
            no scalar counterpart, or ``None``. The two observation checks
            skip with this reason; the other kernels are still compared.
        initial_state: ``(env, rng) -> state`` for a model env with no
            initial-state prior. Those envs are seeded from the real world's
            first observation, so the test needs some other plausible state to
            start from; ``None`` means ``env.initial_state_dist()``.
    """

    model_id: str
    model_class: str
    build: Callable[[], Tuple[Any, Any]]
    action_of_index: Callable[[Any, Any, int], Any]
    num_actions: Callable[[Any, Any], int]
    state_of_row: Callable[[Any, np.ndarray], Any] = _copy_row
    row_of_state: Callable[[Any, Any], np.ndarray] = _float_row
    observation_of_row: Callable[[Any, np.ndarray], Any] = _copy_row
    row_of_observation: Callable[[Any, Any], np.ndarray] = _float_row
    required_modules: Tuple[str, ...] = ()
    initial_state: Optional[Callable[[Any, np.random.Generator], Any]] = None
    observation_not_comparable: Optional[str] = None
    probe_states: Optional[Callable[[Any], List[Any]]] = None


def _cpu64() -> Dict[str, Any]:
    import torch  # pylint: disable=import-outside-toplevel

    return {"device": torch.device("cpu"), "dtype": torch.float64}


def _discrete_action(env: Any, model: Any, index: int) -> Any:
    del model
    return env.get_actions()[index]  # type: ignore[attr-defined]


def _discrete_action_count(env: Any, model: Any) -> int:
    del model
    return len(env.get_actions())  # type: ignore[attr-defined]


def _from_builder(env_id: str, model_path: str, model_name: str) -> Callable[[], Tuple[Any, Any]]:
    """Build the env from ENV_BUILDERS and the model with ``Model(env, cpu, float64)``."""

    def build() -> Tuple[Any, Any]:
        env = _ENV_BUILDER_BY_ID[env_id]()
        model_class = getattr(importlib.import_module(model_path), model_name)
        return env, model_class(env, **_cpu64())

    return build


def _build_continuous_light_dark() -> Tuple[Any, Any]:
    # The torch model declines the hazard-terminal slot (is_obstacle_hit_terminal,
    # the environment's default) with NotImplementedError, so it is built with
    # the slot off -- the only configuration VOPP can plan on. See
    # test_vectorized_config_contract.py.
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ContinuousLightDarkPOMDP,
    )
    from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
        ContinuousLightDarkVectorizedModel,
    )
    from POMDPPlanners.tests.test_utils.env_pinned_kwargs import (
        continuous_light_dark_pinned_kwargs,
    )

    env = ContinuousLightDarkPOMDP(
        discount_factor=0.95,
        **continuous_light_dark_pinned_kwargs(is_obstacle_hit_terminal=False),
    )
    return env, ContinuousLightDarkVectorizedModel(env, **_cpu64())


def _tiger_labels() -> Tuple[List[str], List[str], List[str]]:
    from POMDPPlanners.environments.tiger_pomdp.tiger_pomdp import ACTIONS, OBSERVATIONS, STATES

    return list(STATES), list(ACTIONS), list(OBSERVATIONS)


def _rock_sample_labels() -> Tuple[str, ...]:
    # The torch model's observation code is an index into this tuple.
    return ("none", "good", "bad")


def _build_racetrack() -> Tuple[Any, Any]:
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_known_track_model import (
        KnownTrackModel,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_track_geometry import (
        TrackGeometry,
    )
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_vectorized_model import (
        RacetrackVectorizedModel,
    )

    # A short lap with three curvature changes, as in the racetrack parity test,
    # so a few steps cross segment boundaries.
    geometry = TrackGeometry(
        segment_starts=np.array([0.0, 12.0, 30.0, 45.0]),
        segment_curvatures=np.array([0.0, 0.05, -0.03, 0.0]),
        total_length_m=60.0,
    )
    env = KnownTrackModel(track_geometry=geometry, discount_factor=0.95)
    return env, RacetrackVectorizedModel(env, **_cpu64())


def _racetrack_keys() -> Tuple[str, ...]:
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_sensor_model import (
        CURVATURE_AHEAD_KEY,
        DETECTIONS_KEY,
        EGO_POSE_KEY,
        EGO_SPEED_KEY,
        LANE_POSE_KEY,
    )

    # The torch row's channel order, from racetrack_schema.
    return (EGO_POSE_KEY, EGO_SPEED_KEY, LANE_POSE_KEY, CURVATURE_AHEAD_KEY, DETECTIONS_KEY)


def _racetrack_observation_of_row(env: Any, row: np.ndarray) -> Dict[str, np.ndarray]:
    from POMDPPlanners.environments.racetrack_pomdp.racetrack_schema import (
        DETECTION_SLOT_WIDTH,
        OBSERVED_EGO_POSE_WIDTH,
        OBSERVED_EGO_SPEED_WIDTH,
        OBSERVED_LANE_POSE_WIDTH,
    )

    widths = (
        OBSERVED_EGO_POSE_WIDTH,
        OBSERVED_EGO_SPEED_WIDTH,
        OBSERVED_LANE_POSE_WIDTH,
        len(env.curvature_lookahead_m),
    )
    keys = _racetrack_keys()
    observation: Dict[str, np.ndarray] = {}
    start = 0
    for key, width in zip(keys, widths):
        observation[key] = row[start : start + width].copy()
        start += width
    observation[keys[-1]] = row[start:].reshape(-1, DETECTION_SLOT_WIDTH).copy()
    return observation


def _racetrack_row_of_observation(env: Any, observation: Dict[str, Any]) -> np.ndarray:
    del env
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float64).ravel() for key in _racetrack_keys()]
    )


def _build_carla_kinematic() -> Tuple[Any, Any]:
    from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_model_pomdp import (
        KinematicCarlaModelPOMDP,
    )
    from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_vectorized_model import (
        CarlaKinematicVectorizedModel,
    )

    env = KinematicCarlaModelPOMDP(discount_factor=0.95, dt=0.05)
    return env, CarlaKinematicVectorizedModel(env, **_cpu64())


def _build_isaac_lab_surrogate() -> Tuple[Any, Any]:
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_model_pomdp import (
        IsaacLabModelPOMDP,
        LinearGaussianTransition,
        LinearRewardModel,
    )
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_vectorized_model import (
        IsaacLabVectorizedModel,
    )

    # A fitted surrogate with arbitrary (seeded) coefficients: the model is
    # built from the same fitted components the scalar env plans with.
    rng = np.random.default_rng(0)
    state_dim, action_dim = 6, 2
    env = IsaacLabModelPOMDP(
        observation_dim=state_dim,
        action_presets=list(rng.standard_normal((4, action_dim))),
        discount_factor=0.95,
        observation_noise_std=0.1,
        transition=LinearGaussianTransition(
            np.eye(state_dim) + 0.05 * rng.standard_normal((state_dim, state_dim)),
            0.3 * rng.standard_normal((state_dim, action_dim)),
            0.1 * rng.standard_normal(state_dim),
            np.diag(0.01 + 0.02 * rng.random(state_dim)),
        ),
        reward_model=LinearRewardModel(
            rng.standard_normal(state_dim),
            rng.standard_normal(action_dim),
            rng.standard_normal(state_dim),
            float(rng.standard_normal()),
        ),
    )
    # pylint: disable=protected-access
    model = IsaacLabVectorizedModel(
        env._transition,
        env._observation_model,
        env._reward_model,
        np.stack(env.action_presets),
        **_cpu64(),
    )
    return env, model


def _gaussian_channel_observations(schema: Any, noise_std: float) -> Dict[str, Any]:
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_perception.observation_models.proprioception_models import (
        GaussianChannelObservationModel,
    )

    # The torch models observe the whole state through one isotropic Gaussian;
    # per-channel Gaussians of the same width are the scalar equivalent.
    return {
        name: GaussianChannelObservationModel(channel=name, noise_std=noise_std)
        for name in schema.names
    }


_ISAAC_OBSERVATION_NOISE = 0.1


def _build_isaac_manipulator() -> Tuple[Any, Any]:
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        IsaacChannelSchema,
        ManipulatorIsaacModel,
        franka_panda_chain,
    )
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_manipulator_vectorized_model import (
        ManipulatorVectorizedModel,
    )

    schema = IsaacChannelSchema(
        (("joint_pos", 7), ("joint_vel", 7), ("command", 7), ("last_action", 7))
    )
    env = ManipulatorIsaacModel(
        state_schema=schema,
        action_presets=[
            np.zeros(7),
            np.full(7, 0.4),
            np.full(7, -0.4),
            np.linspace(-0.5, 0.5, 7),
        ],
        discount_factor=0.99,
        step_dt=0.1,
        tracking_gain=0.4,
        chain=franka_panda_chain(),
        default_joint_positions=np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]),
        action_scale=0.5,
        observation_models=_gaussian_channel_observations(schema, _ISAAC_OBSERVATION_NOISE),
    )
    model = ManipulatorVectorizedModel(
        env, observation_noise_std=_ISAAC_OBSERVATION_NOISE, **_cpu64()
    )
    return env, model


def _build_isaac_navigation() -> Tuple[Any, Any]:
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models import (
        NavigationIsaacModel,
        navigation_state_schema,
    )
    from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_navigation_vectorized_model import (
        NavigationVectorizedModel,
    )

    schema = navigation_state_schema()
    env = NavigationIsaacModel(
        state_schema=schema,
        action_presets=[
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.6, -0.4, 0.5]),
        ],
        discount_factor=0.99,
        step_dt=0.2,
        linear_scale=0.8,
        angular_scale=0.6,
        observation_models=_gaussian_channel_observations(schema, _ISAAC_OBSERVATION_NOISE),
    )
    model = NavigationVectorizedModel(
        env, observation_noise_std=_ISAAC_OBSERVATION_NOISE, **_cpu64()
    )
    return env, model


# Plausible starting states for the model envs that are seeded from the real
# world's first observation and so have no initial-state prior. Ranges follow
# each model's own parity test.


def _carla_initial_state(env: Any, rng: np.random.Generator) -> np.ndarray:
    agents = env.max_tracked_agents
    ego = np.concatenate(
        [
            rng.uniform(-20.0, 20.0, size=2),
            rng.uniform(-180.0, 180.0, size=1),
            rng.uniform(-5.0, 10.0, size=2),
            rng.uniform(-3.0, 3.0, size=1),
            rng.uniform(-0.5, 0.5, size=1),
        ]
    )
    slots = np.stack(
        [
            (rng.uniform(size=agents) < 0.6).astype(float),
            rng.uniform(-60.0, 60.0, size=agents),
            rng.uniform(-8.0, 8.0, size=agents),
            rng.uniform(-np.pi, np.pi, size=agents),
            rng.uniform(0.0, 10.0, size=agents),
        ],
        axis=1,
    )
    return np.concatenate([ego, slots.ravel()])


def _carla_occlusion_probe(env: Any) -> List[np.ndarray]:
    """One agent directly ahead and a second at the edge of its shadow.

    The second agent sits where the first one's occlusion disc just touches the
    ego's line of sight to it, so whether it is seen turns on the occlusion rule
    itself. Rollouts from random states almost never land there.
    """
    from POMDPPlanners.environments.carla_pomdp.carla_generative_models.carla_kinematic_vectorized_model import (  # noqa: E501  pylint: disable=line-too-long
        CarlaKinematicVectorizedModel,
    )

    radius = CarlaKinematicVectorizedModel(
        env, **_cpu64()
    )._occlusion_radius  # pylint: disable=protected-access
    state = np.zeros(7 + 5 * env.max_tracked_agents)
    state[3] = 5.0
    agents = state[7:].reshape(env.max_tracked_agents, 5)
    agents[0] = [1.0, 20.0, 0.0, 0.0, 5.0]
    agents[1] = [1.0, 40.0, 2.0 * radius, 0.0, 5.0]
    return [state]


def _isaac_surrogate_initial_state(env: Any, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(env.observation_dim)


def _manipulator_initial_state(env: Any, rng: np.random.Generator) -> np.ndarray:
    schema = env.state_schema
    state = np.zeros(schema.total_dim)
    for name, low, high in (
        ("joint_pos", -0.6, 0.6),
        ("joint_vel", -1.0, 1.0),
        ("command", -0.6, 0.6),
        ("last_action", -1.0, 1.0),
    ):
        block = schema.slice_of(name)
        state[block] = rng.uniform(low, high, size=block.stop - block.start)
    return state


def _navigation_initial_state(env: Any, rng: np.random.Generator) -> np.ndarray:
    schema = env.state_schema
    state = np.zeros(schema.total_dim)
    state[schema.slice_of("base_lin_vel")] = rng.uniform(-1.0, 1.0, size=3)
    state[schema.slice_of("projected_gravity")] = rng.uniform(-0.2, 0.2, size=3)
    goal = schema.slice_of("pose_command")
    state[goal.start : goal.start + 2] = rng.uniform(-3.0, 3.0, size=2)
    state[goal.start + 2] = rng.uniform(-0.05, 0.05)
    state[goal.start + 3] = rng.uniform(-np.pi, np.pi)
    return state


def _schema_observation_of_row(env: Any, row: np.ndarray) -> Dict[str, np.ndarray]:
    return env.state_schema.split(row.copy())


def _schema_row_of_observation(env: Any, observation: Dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(observation[name], dtype=np.float64).ravel() for name in env.state_schema.names]
    )


_ENVS = "POMDPPlanners.environments"

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        "CarlaKinematicVectorizedModel",
        "CarlaKinematicVectorizedModel",
        _build_carla_kinematic,
        action_of_index=lambda env, model, i: int(i),
        num_actions=lambda env, model: len(env.action_presets),
        observation_of_row=lambda env, row: {"gnss": row[:2].copy(), "agents": row[2:].copy()},
        row_of_observation=lambda env, o: np.concatenate(
            [np.asarray(o["gnss"], dtype=np.float64).ravel(), np.asarray(o["agents"]).ravel()]
        ),
        initial_state=_carla_initial_state,
        probe_states=_carla_occlusion_probe,
    ),
    ModelSpec(
        "CartPoleVectorizedModel",
        "CartPoleVectorizedModel",
        _from_builder(
            "CartPolePOMDP",
            f"{_ENVS}.cartpole_pomdp.cartpole_vectorized_model",
            "CartPoleVectorizedModel",
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
    ),
    ModelSpec(
        "ContinuousLightDarkVectorizedModel",
        "ContinuousLightDarkVectorizedModel",
        _build_continuous_light_dark,
        # The model plans over a fixed set of unit moves; the scalar env takes
        # the move vector itself.
        action_of_index=lambda env, model, i: model.action_vectors[i].cpu().numpy().copy(),
        num_actions=lambda env, model: int(model.action_vectors.shape[0]),
    ),
    ModelSpec(
        "IsaacLabVectorizedModel",
        "IsaacLabVectorizedModel",
        _build_isaac_lab_surrogate,
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        initial_state=_isaac_surrogate_initial_state,
    ),
    ModelSpec(
        "LaserTagVectorizedModel",
        "LaserTagVectorizedModel",
        _from_builder(
            "LaserTagPOMDP",
            f"{_ENVS}.laser_tag_pomdp.laser_tag_vectorized_model",
            "LaserTagVectorizedModel",
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        observation_of_row=lambda env, row: tuple(float(v) for v in row),
    ),
    ModelSpec(
        "ManipulatorVectorizedModel",
        "ManipulatorVectorizedModel",
        _build_isaac_manipulator,
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        observation_of_row=_schema_observation_of_row,
        row_of_observation=_schema_row_of_observation,
        initial_state=_manipulator_initial_state,
    ),
    ModelSpec(
        "MountainCarVectorizedModel",
        "MountainCarVectorizedModel",
        _from_builder(
            "MountainCarPOMDP",
            f"{_ENVS}.mountain_car_pomdp.mountain_car_vectorized_model",
            "MountainCarVectorizedModel",
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
    ),
    ModelSpec(
        "NavigationVectorizedModel",
        "NavigationVectorizedModel",
        _build_isaac_navigation,
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        observation_of_row=_schema_observation_of_row,
        row_of_observation=_schema_row_of_observation,
        initial_state=_navigation_initial_state,
        # NavigationIsaacModel ships without an observation model, and the torch
        # model wraps the pose_command heading residual into (-pi, pi]. The
        # per-channel Gaussians configured here match every channel except
        # that one, and no scalar observation model observes the whole state
        # with a wrapped heading (GoalRelativePoseObservationModel wraps, but
        # emits a 3-vector). So there is no scalar twin to compare with.
        observation_not_comparable=(
            "no scalar observation model observes the navigation state with the "
            "heading residual wrapped, as the torch model does"
        ),
    ),
    ModelSpec(
        "PacManVectorizedModel",
        "PacManVectorizedModel",
        _from_builder(
            "PacManPOMDP", f"{_ENVS}.pacman_pomdp.pacman_vectorized_model", "PacManVectorizedModel"
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        observation_of_row=lambda env, row: env.array_to_observation(row.copy()),
        row_of_observation=lambda env, o: np.asarray(env.observation_to_array(o), dtype=np.float64),
    ),
    ModelSpec(
        "PushVectorizedModel",
        "PushVectorizedModel",
        _from_builder(
            "PushPOMDP", f"{_ENVS}.push_pomdp.push_vectorized_model", "PushVectorizedModel"
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
    ),
    ModelSpec(
        "RacetrackVectorizedModel",
        "RacetrackVectorizedModel",
        _build_racetrack,
        action_of_index=lambda env, model, i: int(i),
        num_actions=lambda env, model: len(env.action_presets),
        observation_of_row=_racetrack_observation_of_row,
        row_of_observation=_racetrack_row_of_observation,
    ),
    ModelSpec(
        "RockSampleVectorizedModel",
        "RockSampleVectorizedModel",
        _from_builder(
            "RockSamplePOMDP",
            f"{_ENVS}.rock_sample_pomdp.rocksample_vectorized_model",
            "RockSampleVectorizedModel",
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        observation_of_row=lambda env, row: _rock_sample_labels()[int(row[0])],
        row_of_observation=lambda env, o: np.array([float(_rock_sample_labels().index(str(o)))]),
    ),
    ModelSpec(
        "SafetyAntVelocityVectorizedModel",
        "SafetyAntVelocityVectorizedModel",
        _from_builder(
            "SafeAntVelocityPOMDP",
            f"{_ENVS}.safety_ant_velocity_pomdp.safety_ant_velocity_vectorized_model",
            "SafetyAntVelocityVectorizedModel",
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
    ),
    ModelSpec(
        "SanityVectorizedModel",
        "SanityVectorizedModel",
        _from_builder(
            "SanityPOMDP",
            f"{_ENVS}.sanity_pomdp.sanity_pomdp_vectorized_model",
            "SanityVectorizedModel",
        ),
        action_of_index=_discrete_action,
        num_actions=_discrete_action_count,
        state_of_row=lambda env, row: int(row[0]),
        observation_of_row=lambda env, row: int(row[0]),
    ),
    ModelSpec(
        "TigerVectorizedModel",
        "TigerVectorizedModel",
        _from_builder(
            "TigerPOMDP",
            f"{_ENVS}.tiger_pomdp.tiger_pomdp_vectorized_model",
            "TigerVectorizedModel",
        ),
        action_of_index=lambda env, model, i: _tiger_labels()[1][i],
        num_actions=lambda env, model: len(_tiger_labels()[1]),
        state_of_row=lambda env, row: _tiger_labels()[0][int(row[0])],
        row_of_state=lambda env, s: np.array([float(_tiger_labels()[0].index(s))]),
        observation_of_row=lambda env, row: _tiger_labels()[2][int(row[0])],
        row_of_observation=lambda env, o: np.array([float(_tiger_labels()[2].index(o))]),
    ),
]


# Model classes found in the source tree that no ModelSpec exercises, and why.
MODEL_CLASS_EXCLUSIONS: Dict[str, str] = {}

OBSERVATION_LOG_PROB_DISAGREEMENTS: Dict[str, Disagreement] = {}
# The torch model scores the speed of next_states; the scalar reward() and
# reward_batch() score the speed of the current state and ignore next_state.
# The model's own parity test passes the same array as both states and
# next_states, which hides the difference.
REWARD_DISAGREEMENTS: Dict[str, Disagreement] = {
    "SafetyAntVelocityVectorizedModel": (
        "rewards() scores next_states' speed; the scalar reward() scores the current "
        "state's speed and ignores next_state",
        AssertionError,
    ),
}
TERMINAL_MASK_DISAGREEMENTS: Dict[str, Disagreement] = {}
NEXT_STATE_DISTRIBUTION_DISAGREEMENTS: Dict[str, Disagreement] = {}
# The scalar FactoredAgentObservationModel.render decides each slot's occlusion
# against the rows it is rewriting in place: an earlier slot has already had
# pose noise added (or been zeroed) when a later slot's sight line is tested.
# So an agent just behind another is reported or dropped at random, while the
# torch sampler -- and both log-likelihoods -- test occlusion on the clean
# state. A zeroed earlier slot also stops occluding. The registry's probe
# state puts one agent at the occlusion boundary behind another, so the
# disagreement shows on every run: the scalar sensor reports it in about half
# the draws, the torch sampler never.
OBSERVATION_DISTRIBUTION_DISAGREEMENTS: Dict[str, Disagreement] = {
    "CarlaKinematicVectorizedModel": (
        "scalar CARLA agent sensor tests occlusion against rows it has already "
        "noised or zeroed, so an agent at the occlusion boundary is dropped at "
        "random; the torch sampler tests the clean state",
        AssertionError,
    ),
}


def missing_module(modules: Tuple[str, ...]) -> Optional[str]:
    """Return the first of ``modules`` that cannot be imported, or ``None``."""
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            return module
    return None
