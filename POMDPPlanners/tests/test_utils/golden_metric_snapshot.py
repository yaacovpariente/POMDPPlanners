# SPDX-License-Identifier: MIT

"""Deterministic ``compute_metrics`` snapshots for every instantiable environment.

This module builds fixed, seeded episode histories for each environment that
overrides :meth:`~POMDPPlanners.core.environment.environment.Environment.compute_metrics`,
and reduces the result to a plain ``{env_slug: {metric_name: value}}`` mapping.

The mapping is frozen in :mod:`POMDPPlanners.tests.metric_golden_values` and
asserted by :mod:`POMDPPlanners.tests.test_metric_golden_snapshot`. Its purpose is
characterization: a change to the shared metrics plumbing must leave every
existing environment's metric names *and* values untouched, and this snapshot is
what makes that provable rather than assumed.

Histories are **frozen**, not regenerated. ``compute_metrics`` was verified to be a
pure function of ``(environment, histories)``, but generating the histories is not
reproducible: several environments (Pacman, both light-dark variants) consume a
data-dependent number of draws inside numba kernels, whose RNG state a
Python-level ``np.random.seed`` cannot pin. Regenerating on every run would
therefore produce a flaky baseline that fails for reasons unrelated to the
metrics code.

So the rollouts are generated once by :func:`generate_snapshot_histories_fixture`
and pickled; the tests replay that fixture. This isolates exactly what the
baseline is meant to characterize — the metrics code — from environment
transition randomness, and as a side effect the committed pickle also pins
``StepData`` wire compatibility, since it was written before the ``info`` field
was added.

Functions:
    build_registry: Construct the env-slug -> environment factory registry.
    build_snapshot_histories: Roll out fresh histories (fixture generation only).
    generate_snapshot_histories_fixture: Write the pickled history fixture.
    load_snapshot_histories: Load the pickled histories for one environment.
    attach_step_info: Replay an environment's step_info over pre-built histories.
    append_terminal_step: Derive the terminated-episode shape of a history list.
    compute_metric_snapshot: Reduce one environment's metrics to name -> value.
    compute_metric_order: Collect one environment's produced metric names, in order.
    compute_all_metric_snapshots: Snapshot every environment from the fixture.
    compute_all_metric_snapshots_with_terminal: Snapshot the terminated-episode shape.
    compute_all_metric_orders: Collect produced and declared name orders per environment.
    compute_all_metric_names: Collect every environment's declared metric names.
"""

import dataclasses
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numba import njit

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.cartpole_pomdp.cartpole_pomdp import CartPolePOMDP
from POMDPPlanners.environments.laser_tag_pomdp.continuous_laser_tag_pomdp import (
    ContinuousLaserTagPOMDP,
)
from POMDPPlanners.environments.laser_tag_pomdp.laser_tag_pomdp import LaserTagPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.environments.light_dark_pomdp.discrete_light_dark_pomdp import (
    DiscreteLightDarkPOMDP,
)
from POMDPPlanners.environments.mountain_car_pomdp.mountain_car_pomdp import MountainCarPOMDP
from POMDPPlanners.environments.pacman_pomdp.pacman_pomdp import PacManPOMDP
from POMDPPlanners.environments.push_pomdp.continuous_push_pomdp import ContinuousPushPOMDP
from POMDPPlanners.environments.push_pomdp.push_pomdp import PushPOMDP
from POMDPPlanners.environments.rock_sample_pomdp.rock_sample_pomdp import RockSamplePOMDP
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.tests.test_utils import env_pinned_kwargs as pinned
from POMDPPlanners.tests.test_utils.history_builders import build_test_history

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_metric_histories.pkl"

# Pinned so the snapshot never moves with an unrelated default change.
SNAPSHOT_DISCOUNT_FACTOR = 0.95
SNAPSHOT_NUM_EPISODES = 3
SNAPSHOT_NUM_STEPS = 5
SNAPSHOT_SEED = 1234

# Action cycle for continuous-action environments, which expose no ``get_actions``.
_CONTINUOUS_ACTION_CYCLE: List[np.ndarray] = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([-1.0, 0.0]),
    np.array([0.0, -1.0]),
]


def build_registry() -> Dict[str, Callable[[], Environment]]:
    """Build the env-slug -> zero-argument environment factory registry.

    Covers every environment that overrides ``compute_metrics`` and can be
    constructed without an external simulator. CARLA, nuPlan, IsaacLab and
    SafetyAnt are excluded: they need a live backend or an optional heavy
    dependency, so they cannot contribute a hermetic baseline.

    Returns:
        Mapping from a stable slug to a factory producing a fresh environment.
    """
    discount = SNAPSHOT_DISCOUNT_FACTOR
    return {
        "tiger": lambda: TigerPOMDP(discount_factor=discount, **pinned.tiger_pinned_kwargs()),
        "cartpole": lambda: CartPolePOMDP(
            discount_factor=discount,
            noise_cov=np.eye(4) * 0.01,
            **pinned.cartpole_pinned_kwargs(),
        ),
        "mountain_car": lambda: MountainCarPOMDP(
            discount_factor=discount, **pinned.mountain_car_pinned_kwargs()
        ),
        "laser_tag": lambda: LaserTagPOMDP(
            discount_factor=discount, **pinned.laser_tag_pinned_kwargs()
        ),
        "continuous_laser_tag": lambda: ContinuousLaserTagPOMDP(
            discount_factor=discount, **pinned.continuous_laser_tag_pinned_kwargs()
        ),
        "discrete_light_dark": lambda: DiscreteLightDarkPOMDP(
            discount_factor=discount, **pinned.discrete_light_dark_pinned_kwargs()
        ),
        "continuous_light_dark": lambda: ContinuousLightDarkPOMDP(
            discount_factor=discount, **pinned.continuous_light_dark_pinned_kwargs()
        ),
        "push": lambda: PushPOMDP(discount_factor=discount, **pinned.push_pinned_kwargs()),
        "continuous_push": lambda: ContinuousPushPOMDP(
            discount_factor=discount, **pinned.continuous_push_pinned_kwargs()
        ),
        "pacman": lambda: PacManPOMDP(discount_factor=discount, **pinned.pacman_pinned_kwargs()),
        "rock_sample": lambda: RockSamplePOMDP(
            discount_factor=discount, **pinned.rock_sample_pinned_kwargs()
        ),
    }


def _snapshot_actions(environment: Environment) -> List[Any]:
    get_actions = getattr(environment, "get_actions", None)
    if get_actions is None:
        return list(_CONTINUOUS_ACTION_CYCLE)
    return list(get_actions())


def _snapshot_belief(state: Any) -> WeightedParticleBelief:
    # Two particles with distinct log-weights: a single all-zero weight vector is
    # rejected by WeightedParticleBelief's validation.
    return WeightedParticleBelief(particles=[state, state], log_weights=np.array([0.0, -0.1]))


@njit(cache=True)
def _seed_numba_rng(seed: int) -> None:
    # Numba keeps its own RNG state inside nopython mode; a Python-level
    # ``np.random.seed`` does not reach it. Environments whose transitions run in
    # jitted kernels (Pacman, continuous light-dark) are otherwise irreproducible.
    np.random.seed(seed)


def _seed_all_rngs(seed: int) -> None:
    np.random.seed(seed)
    _seed_numba_rng(seed)


def _build_single_history(environment: Environment, actions: List[Any], seed: int) -> History:
    _seed_all_rngs(seed)
    state = environment.initial_state_dist().sample()[0]
    steps: List[StepData] = []
    for step_index in range(SNAPSHOT_NUM_STEPS):
        action = actions[step_index % len(actions)]
        # Re-seed before every transition: some environments consume a
        # data-dependent number of draws, so seeding once per episode leaves the
        # stream position (and therefore the snapshot) irreproducible.
        _seed_all_rngs(seed + step_index)
        next_state, observation, reward = environment.sample_next_step(state, action)
        steps.append(
            StepData(
                state=state,
                action=action,
                next_state=next_state,
                observation=observation,
                reward=float(reward),
                belief=_snapshot_belief(state),
            )
        )
        state = next_state
        if environment.is_terminal(state):
            break
    return build_test_history(
        steps=steps,
        reach_terminal=environment.is_terminal(state),
        discount_factor=SNAPSHOT_DISCOUNT_FACTOR,
    )


def build_snapshot_histories(environment: Environment) -> List[History]:
    """Build the pinned set of episode histories used for one environment's snapshot.

    Args:
        environment: The environment to roll out. Its own generative model supplies
            transitions, observations and rewards.

    Returns:
        A list of ``SNAPSHOT_NUM_EPISODES`` histories, each at most
        ``SNAPSHOT_NUM_STEPS`` steps long and truncated early on a terminal state.
    """
    actions = _snapshot_actions(environment)
    return [
        _build_single_history(environment, actions, SNAPSHOT_SEED + episode_index)
        for episode_index in range(SNAPSHOT_NUM_EPISODES)
    ]


def generate_snapshot_histories_fixture(fixture_path: Path = FIXTURE_PATH) -> Path:
    """Roll out and pickle the frozen histories for every registry environment.

    This is fixture-generation, not test code: it is run once, deliberately, and
    its output is committed. Re-running it will produce *different* histories
    (see the module docstring) and therefore invalidate the committed baseline,
    so only call it when the baseline is being reset on purpose.

    Args:
        fixture_path: Destination pickle path. Defaults to the committed fixture.

    Returns:
        The path written.
    """
    histories = {
        slug: build_snapshot_histories(factory()) for slug, factory in build_registry().items()
    }
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with fixture_path.open("wb") as handle:
        pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return fixture_path


def load_snapshot_histories(fixture_path: Path = FIXTURE_PATH) -> Dict[str, List[History]]:
    """Load the committed frozen histories.

    Args:
        fixture_path: Pickle to read. Defaults to the committed fixture.

    Returns:
        Mapping from environment slug to its frozen list of histories.

    Raises:
        FileNotFoundError: If the fixture has not been generated.
    """
    with fixture_path.open("rb") as handle:
        return pickle.load(handle)


def attach_step_info(environment: Environment, histories: List[History]) -> List[History]:
    """Replay an environment's ``step_info`` over pre-built histories.

    This is test infrastructure, not a backward-compatibility path. The frozen
    fixture and the hand-built histories in the environment test suites never
    went through :class:`~POMDPPlanners.simulations.episodes.EpisodeRunner`, so
    they carry no ``StepData.info`` at all. An environment whose metrics are
    derived from the per-step channel would report nothing for them, and the
    characterization tests would pass vacuously.

    The call mirrors ``EpisodeRunner._record_step`` exactly, including its
    ``info or None`` normalization. One expression covers the terminal
    bookkeeping step too, because that step carries ``action is None`` and
    ``next_state is None`` -- which is precisely what
    ``EpisodeRunner._add_terminal_step`` passes.

    Args:
        environment: The environment whose ``step_info`` supplies the channels.
        histories: Histories to measure. Not mutated; ``StepData`` is a
            ``NamedTuple``, so each step is replaced rather than modified.

    Returns:
        New histories whose steps carry the measured ``info`` mappings.
    """
    measured: List[History] = []
    for history in histories:
        steps = [
            step._replace(
                info=environment.step_info(step.state, step.action, step.next_state) or None
            )
            for step in history.history
        ]
        measured.append(dataclasses.replace(history, history=steps))
    return measured


def append_terminal_step(histories: List[History]) -> List[History]:
    """Derive the terminated-episode shape of a list of histories.

    The frozen fixture stops after the last recorded transition, but a real
    episode that reaches a terminal state also carries a bookkeeping step whose
    ``state`` is the final state and whose ``action`` / ``next_state`` /
    ``observation`` / ``reward`` are all ``None``
    (``EpisodeRunner._add_terminal_step``). Metrics that scan ``step.state``
    over the whole history therefore see one more state in a real run than they
    do on the fixture, and the fixture alone cannot detect a regression in how
    that final state is counted.

    This derivation is deterministic -- it reads the last step's ``next_state``
    and reuses its belief -- so it needs no rollout and no RNG.

    Note:
        This is a *shape probe*, not a sample of the observed distribution: the
        real runner only appends a terminal step when ``is_terminal`` fires,
        whereas this appends one to every history. That does not weaken the
        characterization argument, since the pre-change and post-change
        implementations are compared on identical inputs; it only bounds what
        the probe covers.

    Args:
        histories: Histories to extend. Not mutated. Histories with no steps are
            passed through unchanged, having no final state to append.

    Returns:
        New histories, each with a terminal bookkeeping step appended and
        ``reach_terminal_state`` set.
    """
    extended: List[History] = []
    for history in histories:
        if not history.history:
            extended.append(history)
            continue
        last = history.history[-1]
        terminal_step = StepData(
            state=last.next_state,
            action=None,
            next_state=None,
            observation=None,
            reward=None,
            belief=last.belief,
        )
        extended.append(
            dataclasses.replace(
                history,
                history=list(history.history) + [terminal_step],
                reach_terminal_state=True,
            )
        )
    return extended


def compute_metric_snapshot(environment: Environment, histories: List[History]) -> Dict[str, float]:
    """Compute one environment's ``compute_metrics`` output as a name -> value mapping.

    Args:
        environment: The environment to snapshot.
        histories: The frozen histories to feed it. They are measured through
            :func:`attach_step_info` first, so an environment reporting through
            the per-step channel sees the same input it would in a real run.

    Returns:
        Mapping from metric name to point estimate. Confidence bounds are
        deliberately excluded: with three episodes several are infinite, which
        makes them useless as a stable baseline.
    """
    metrics = environment.compute_metrics(attach_step_info(environment, histories))
    return {metric.name: float(metric.value) for metric in metrics}


def compute_metric_order(environment: Environment, histories: List[History]) -> List[str]:
    """Collect one environment's produced metric names, in emission order.

    Order is part of the contract:
    :func:`~POMDPPlanners.simulations.simulation_statistics.get_metric_names_from_environment_policy_pair`
    feeds hyperparameter-tuning objective selection, and
    :func:`compute_metric_snapshot` cannot see order because it builds a dict.

    Args:
        environment: The environment to inspect.
        histories: The frozen histories to feed it.

    Returns:
        The produced metric names, in the order ``compute_metrics`` emitted them.
    """
    metrics = environment.compute_metrics(attach_step_info(environment, histories))
    return [metric.name for metric in metrics]


def compute_all_metric_snapshots() -> Dict[str, Dict[str, float]]:
    """Snapshot every registry environment against the frozen histories.

    Returns:
        Mapping from environment slug to that environment's metric snapshot.
    """
    frozen = load_snapshot_histories()
    return {
        slug: compute_metric_snapshot(factory(), frozen[slug])
        for slug, factory in build_registry().items()
    }


def compute_all_metric_snapshots_with_terminal() -> Dict[str, Dict[str, float]]:
    """Snapshot every registry environment against the terminated-episode shape.

    Complements :func:`compute_all_metric_snapshots`, which uses the frozen
    histories as-is. This one appends the terminal bookkeeping step first, so
    the baseline also pins how each environment counts the final state -- the
    one thing the raw fixture structurally cannot show.

    Returns:
        Mapping from environment slug to that environment's metric snapshot.
    """
    frozen = load_snapshot_histories()
    return {
        slug: compute_metric_snapshot(factory(), append_terminal_step(frozen[slug]))
        for slug, factory in build_registry().items()
    }


def compute_all_metric_orders() -> Dict[str, Tuple[List[str], List[str]]]:
    """Collect every registry environment's produced and declared name orders.

    Returns:
        Mapping from environment slug to ``(produced_names, declared_names)``,
        both unsorted. ``compute_all_metric_names`` sorts, which is what makes it
        blind to a reordering; these two lists are what pin the order itself.
    """
    frozen = load_snapshot_histories()
    orders: Dict[str, Tuple[List[str], List[str]]] = {}
    for slug, factory in build_registry().items():
        environment = factory()
        orders[slug] = (
            compute_metric_order(environment, frozen[slug]),
            list(environment.get_metric_names()),
        )
    return orders


def compute_all_metric_names() -> Dict[str, List[str]]:
    """Collect every registry environment's declared metric names.

    Snapshotting the declared names separately from the values is what catches a
    rename: renaming a metric does not crash anything, it silently invalidates
    saved MLflow runs and Optuna objective configs that reference the old name.

    Returns:
        Mapping from environment slug to its sorted ``get_metric_names()`` output.
    """
    return {
        slug: sorted(factory().get_metric_names()) for slug, factory in build_registry().items()
    }
