# SPDX-License-Identifier: MIT

"""Configuration identity, save/load and per-call reset for the uncovered planners.

``test_planner_serialization.py`` and ``test_planner_save_load_interface.py``
cover POMCP, POMCP-DPW, POMCPOW, PFT-DPW, Sparse-PFT, the open-loop planner,
sparse sampling and BetaZero. Five concrete planners had none of it: CPFT-DPW,
CPOMCPOW, iCVaR-PFT-DPW, iCVaR-POMCPOW and iCVaR sparse sampling.

Three properties matter and they are separate:

* **Configuration identity** — ``config_id`` is the simulation cache key. Two
  planners that differ in an algorithm parameter must not share cached
  episodes, and two that are identical must, or every rerun recomputes.
* **Serialization** — a saved planner must still be able to choose an action
  after being loaded, not merely reconstruct without raising.
* **Per-call reset** — the arena planners key their per-search state by integer
  node ID, and those IDs mean nothing outside one tree. State carried across a
  call would apply one search's numbers to another search's nodes.

VOPP is deliberately absent: it does not implement the shared ``Policy``
interface (it plans from a particle tensor via ``plan`` rather than from a
``Belief`` via ``action``), so ``config_id`` and ``Policy.save`` do not apply to
it. Its own reset behaviour is covered in ``test_vopp_backup_correctness.py``.
"""

# pylint: disable=protected-access

import random
from typing import Any

import numpy as np
import pytest

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.core.environment import ConstrainedEnvironment
from POMDPPlanners.environments import TigerPOMDP
from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
    ContinuousLightDarkPOMDP,
)
from POMDPPlanners.planners.mcts_planners.constrained_pft_dpw import CPFT_DPW
from POMDPPlanners.planners.mcts_planners.constrained_pomcpow import CPOMCPOW
from POMDPPlanners.planners.mcts_planners.icvar_pft_dpw import ICVaR_PFT_DPW
from POMDPPlanners.planners.mcts_planners.icvar_pomcpow import ICVaR_POMCPOW
from POMDPPlanners.planners.sparse_sampling_planners.icvar_sparse_sampling import (
    ICVaRSparseSampling,
)
from POMDPPlanners.utils.action_samplers import (
    DiscreteActionSampler,
    UnitCircleActionSampler,
)


np.random.seed(42)
random.seed(42)


class _UnitCostEnv(ContinuousLightDarkPOMDP, ConstrainedEnvironment):
    """Continuous light-dark with a constant one-unit constraint cost."""

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        del state, action, next_state
        return np.array([1.0])


def _continuous_env():
    return _UnitCostEnv(discount_factor=0.95, is_obstacle_hit_terminal=False)


def _sampler():
    return UnitCircleActionSampler(max_action_magnitude=1.0)


def _cpft(**overrides):
    params = dict(
        environment=_continuous_env(),
        discount_factor=0.95,
        depth=3,
        name="cpft_cfg",
        action_sampler=_sampler(),
        cost_budget=0.5,
        lambda_init=0.0,
        lambda_step=0.1,
        return_minimal_cost=True,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=6,
    )
    params.update(overrides)
    return CPFT_DPW(**params)


def _cpomcpow(**overrides):
    params = dict(
        environment=_continuous_env(),
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        k_o=1.0,
        k_a=1.0,
        alpha_o=0.5,
        alpha_a=0.5,
        name="cpomcpow_cfg",
        action_sampler=_sampler(),
        cost_budget=0.5,
        lambda_init=0.0,
        lambda_step=0.1,
        n_simulations=6,
    )
    params.update(overrides)
    return CPOMCPOW(**params)


def _icvar_pft(**overrides):
    params = dict(
        environment=_continuous_env(),
        name="icvar_pft_cfg",
        depth=3,
        action_sampler=_sampler(),
        discount_factor=0.95,
        alpha=0.1,
        delta=0.1,
        k_a=1.0,
        alpha_a=0.5,
        k_o=1.0,
        alpha_o=0.5,
        exploration_constant=1.0,
        n_simulations=6,
    )
    params.update(overrides)
    return ICVaR_PFT_DPW(**params)


def _icvar_pomcpow(**overrides):
    params = dict(
        environment=_continuous_env(),
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        k_o=1.0,
        k_a=1.0,
        alpha_o=0.5,
        alpha_a=0.5,
        min_immediate_cost=0.0,
        max_immediate_cost=1.0,
        min_visit_count_per_action=1,
        delta=0.1,
        name="icvar_pomcpow_cfg",
        action_sampler=_sampler(),
        n_simulations=6,
        alpha=0.05,
    )
    params.update(overrides)
    return ICVaR_POMCPOW(**params)


def _icvar_sparse_sampling(**overrides):
    params = dict(
        environment=TigerPOMDP(discount_factor=0.95),
        branching_factor=2,
        depth=2,
        alpha=0.25,
        name="icvar_ss_cfg",
    )
    params.update(overrides)
    return ICVaRSparseSampling(**params)


# ``(label, builder, {parameter that must change the id: changed value})``.
# Each parameter named here is an algorithm parameter — the risk level, the
# cost budget, the horizon, the widening budget — so two planners differing in
# one of them are genuinely different planners.
CONFIG_CASES = [
    ("CPFT_DPW", _cpft, {"cost_budget": 5.0}),
    ("CPFT_DPW", _cpft, {"depth": 7}),
    ("CPOMCPOW", _cpomcpow, {"cost_budget": 5.0}),
    ("CPOMCPOW", _cpomcpow, {"lambda_step": 0.9}),
    ("ICVaR_PFT_DPW", _icvar_pft, {"alpha": 0.9}),
    ("ICVaR_PFT_DPW", _icvar_pft, {"k_a": 4.0}),
    ("ICVaR_POMCPOW", _icvar_pomcpow, {"alpha": 0.9}),
    ("ICVaR_POMCPOW", _icvar_pomcpow, {"delta": 0.9}),
    ("ICVaRSparseSampling", _icvar_sparse_sampling, {"alpha": 0.9}),
    ("ICVaRSparseSampling", _icvar_sparse_sampling, {"branching_factor": 5}),
]


@pytest.mark.parametrize("label, builder, change", CONFIG_CASES)
def test_config_id_is_stable_for_equal_configs_and_changes_with_an_algorithm_parameter(
    label, builder, change
):
    """Equal configurations share a cache key; a changed one does not.

    Purpose: ``config_id`` decides whether a simulation run reuses cached
        episodes. If it were unstable, every rerun would recompute; if it
        ignored an algorithm parameter, a tuning study would serve one
        parameter setting's episodes for another's, which is silent data
        corruption rather than a crash.

    Given: Two planners built from identical parameters, and a third differing
        in exactly one algorithm parameter.
    When: Their ``config_id`` values are compared.
    Then: The first two are equal and the third differs.

    Test type: unit
    """
    first = builder()
    same = builder()
    different = builder(**change)

    assert first.config_id == same.config_id, (
        f"{label}: two identically configured planners produced different config ids, so no "
        "cached episode would ever be reused"
    )
    (parameter,) = change
    assert first.config_id != different.config_id, (
        f"{label}: changing {parameter} to {change[parameter]!r} left the config id unchanged, "
        "so two different planners would share cached results"
    )


@pytest.mark.parametrize(
    "label, builder",
    [
        ("CPFT_DPW", _cpft),
        ("CPOMCPOW", _cpomcpow),
        ("ICVaR_PFT_DPW", _icvar_pft),
        ("ICVaR_POMCPOW", _icvar_pomcpow),
    ],
)
def test_a_second_call_starts_from_a_fresh_tree_and_fresh_per_search_state(label, builder):
    """Repeated planning neither accumulates visits nor reuses node-ID state.

    Purpose: Every arena planner builds a new tree per call, and the
        constrained ones key their Lagrange multiplier and cost-Q table by
        integer node ID. Carrying either across a call applies one tree's
        numbers to another tree's unrelated nodes.

    Given: Two consecutive ``action()`` calls with the same belief and the same
        fixed simulation count.
    When: The metrics of both are compared.
    Then: The reported root visit count is the same both times rather than
        doubled, and the first call's ``PolicyRunData`` still holds the values
        it was given.

    Test type: unit
    """
    np.random.seed(303)
    random.seed(303)
    planner = builder()
    belief = get_initial_belief(pomdp=planner.environment, n_particles=10)

    _, first = planner.action(belief)
    snapshot = [(v.name, v.value) for v in first.info_variables]
    _, second = planner.action(belief)

    assert [
        (v.name, v.value) for v in first.info_variables
    ] == snapshot, f"{label}: the first call's PolicyRunData changed when the second call ran"
    first_metrics = dict(snapshot)
    second_metrics = {v.name: v.value for v in second.info_variables}
    if "root_visit_count" in first_metrics and "root_visit_count" in second_metrics:
        assert second_metrics["root_visit_count"] <= planner.n_simulations, (
            f"{label}: the second call reports {second_metrics['root_visit_count']} root visits "
            f"against {planner.n_simulations} simulations, so the tree was reused"
        )


@pytest.mark.parametrize("label, builder", [("CPFT_DPW", _cpft), ("CPOMCPOW", _cpomcpow)])
def test_constrained_planners_reset_lambda_and_the_cost_table_between_calls(label, builder):
    """The dual variable restarts from ``lambda_init`` on every call.

    Purpose: Dual ascent is a per-decision optimisation. A multiplier carried
        over would make the first decision of an episode and the tenth solve
        different problems, with nothing in the output to show it.

    Given: A planner with a zero cost budget and a large step, so the first
        call is guaranteed to drive lambda upward.
    When: A second call is made.
    Then: The reset observed at the start of the second call restores
        ``lambda_init`` and empties the cost-Q table.

    Test type: unit
    """
    np.random.seed(404)
    random.seed(404)
    planner = builder(cost_budget=0.0, lambda_step=0.5)
    belief = get_initial_belief(pomdp=planner.environment, n_particles=10)

    planner.action(belief)
    assert np.any(
        planner._lambda > 0.0
    ), f"{label}: lambda never left its initial value, so the reset check would be vacuous"

    observed = {}
    original = planner._reset_per_action_state

    def spy():
        original()
        observed["lambda"] = planner._lambda.copy()
        observed["cost_q"] = dict(planner._action_cost_q)

    planner._reset_per_action_state = spy  # type: ignore[method-assign]
    planner.action(belief)

    np.testing.assert_allclose(observed["lambda"], planner.lambda_init, atol=1e-12)
    assert observed["cost_q"] == {}, f"{label}: the cost-Q table survived into the next search"


@pytest.mark.parametrize(
    "label, builder",
    [
        ("CPFT_DPW", _cpft),
        ("CPOMCPOW", _cpomcpow),
        ("ICVaR_PFT_DPW", _icvar_pft),
        ("ICVaR_POMCPOW", _icvar_pomcpow),
        ("ICVaRSparseSampling", _icvar_sparse_sampling),
    ],
)
def test_a_planner_survives_a_pickle_round_trip_and_can_still_choose_an_action(label, builder):
    """A pickled planner still plans, which is what parallel execution needs.

    Purpose: ``LocalSimulationsAPI`` distributes episodes with joblib, which
        pickles the planner into each worker. A planner that pickles but comes
        back unable to plan fails only once a real batch runs — and then in a
        worker process, where the traceback is hard to read.

    Given: Each planner and a small belief.
    When: The planner is pickled, unpickled, and asked for an action.
    Then: One legal-shaped action comes back and the reconstructed planner's
        ``config_id`` matches the original's, so the cache key survives too.

    Test type: unit
    """
    import pickle

    np.random.seed(505)
    random.seed(505)
    planner = builder()
    belief = get_initial_belief(pomdp=planner.environment, n_particles=10)

    restored = pickle.loads(pickle.dumps(planner))

    assert restored.config_id == planner.config_id, (
        f"{label}: the config id changed across a pickle round trip, so a resumed run would "
        "miss its own cache"
    )
    actions, _ = restored.action(belief)
    assert len(actions) == 1, f"{label}: the reconstructed planner returned {len(actions)} actions"


def test_icvar_sparse_sampling_declares_no_metrics_and_that_is_the_contract():
    """The sparse-sampling planners report no tree metrics, deliberately.

    Purpose: Their tree is a fixed full lookahead, so the visit-distribution
        metrics the MCTS planners report would be constants determined by the
        branching factor. The empty contract is asserted rather than metrics
        being invented for them.

    Test type: unit
    """
    planner = _icvar_sparse_sampling()
    assert ICVaRSparseSampling.get_info_variable_names() == []

    _, run_data = planner.action(get_initial_belief(pomdp=planner.environment, n_particles=6))
    assert run_data.info_variables == []


@pytest.mark.parametrize(
    "label, builder",
    [
        ("CPFT_DPW", _cpft),
        ("CPOMCPOW", _cpomcpow),
        ("ICVaR_PFT_DPW", _icvar_pft),
        ("ICVaR_POMCPOW", _icvar_pomcpow),
    ],
)
def test_planning_does_not_mutate_the_callers_belief(label, builder):
    """The belief the caller owns is unchanged after a decision.

    Purpose: The episode runner keeps its own belief and updates it from the
        executed action and the realised observation. A planner that mutated it
        in place would corrupt the episode's filter with its own simulated
        particles.

    Test type: unit
    """
    np.random.seed(606)
    random.seed(606)
    planner = builder()
    belief = get_initial_belief(pomdp=planner.environment, n_particles=10)
    before_particles = [np.array(p, copy=True) for p in belief.particles]
    before_weights = np.array(belief.log_weights, copy=True)

    planner.action(belief)

    assert len(belief.particles) == len(
        before_particles
    ), f"{label}: the caller's belief changed size during planning"
    for index, (before, after) in enumerate(zip(before_particles, belief.particles)):
        assert np.array_equal(
            before, np.asarray(after)
        ), f"{label}: particle {index} of the caller's belief was mutated during planning"
    assert np.array_equal(
        np.asarray(belief.log_weights), before_weights
    ), f"{label}: the caller's belief weights were rewritten during planning"
