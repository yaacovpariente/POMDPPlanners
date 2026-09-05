# SPDX-License-Identifier: MIT

"""Planner metrics must survive being recorded, stored and reloaded.

Every planner reports tree metrics through ``PolicyRunData``. Those metrics are
what a tuning study reads and what a results table is built from, so losing
them is a silent data loss rather than a crash. The existing round-trip test in
``test_core/test_simulation.py`` uses ``policy_run_data=[]``, which cannot
expose that loss at all.

There are two storage paths and both are covered here: the live one, where
``LocalSimulationsAPI`` runs episodes and caches them, and the dictionary
round trip on :class:`History`.
"""

import numpy as np

from POMDPPlanners.core.belief import WeightedParticleBelief, get_initial_belief
from POMDPPlanners.core.policy import PolicyInfoVariable, PolicyRunData
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.core.simulation.history import History, StepData
from POMDPPlanners.environments.tiger_pomdp import TigerPOMDP
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import (
    LocalSimulationsAPI,
)
from POMDPPlanners.utils.tree_statistics import TreeMetrics


def _history_with_metrics(**overrides) -> History:
    env = TigerPOMDP(discount_factor=0.95)
    belief = WeightedParticleBelief(env.states, np.array([0.0, -0.1]))
    step = StepData("tiger_left", "listen", "tiger_left", "tiger_left", -1.0, belief)
    defaults = dict(
        history=[step],
        discount_factor=0.95,
        average_state_sampling_time=0.001,
        average_action_time=0.01,
        average_observation_time=0.002,
        average_belief_update_time=0.005,
        average_reward_time=0.001,
        actual_num_steps=1,
        reach_terminal_state=False,
        policy_run_data=[
            PolicyRunData(
                info_variables=[
                    PolicyInfoVariable(name=TreeMetrics.ROOT_VISIT_COUNT.value, value=7),
                    PolicyInfoVariable(
                        name=TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value, value=0.8112781244591328
                    ),
                ]
            )
        ],
    )
    defaults.update(overrides)
    return History(**defaults)


def test_history_round_trip_preserves_planner_metrics_exactly():
    """``to_dict`` then ``from_dict`` keeps every metric's name, value and type.

    Purpose: ``from_dict`` has always read ``policy_run_data``; ``to_dict`` did
        not write it, so a round trip returned a history with an empty metric
        list and no error anywhere. This test fails against that behaviour.

    Given: A history carrying one ``PolicyRunData`` with ``root_visit_count = 7``
        (an int) and an entropy of 0.8112781244591328 (a float).
    When: The history is converted to a dict and back.
    Then: The reloaded history carries the same single run-data entry with both
        names, both exact values, and the same Python types — an int that came
        back as a float would corrupt a downstream count.

    Test type: unit
    """
    original = _history_with_metrics()

    restored = History.from_dict(original.to_dict())

    assert len(restored.policy_run_data) == 1, (
        f"round trip produced {len(restored.policy_run_data)} run-data entries, expected 1; "
        "to_dict must write the policy_run_data key that from_dict reads"
    )
    restored_variables = {v.name: v.value for v in restored.policy_run_data[0].info_variables}
    assert restored_variables == {
        TreeMetrics.ROOT_VISIT_COUNT.value: 7,
        TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value: 0.8112781244591328,
    }
    assert isinstance(restored_variables[TreeMetrics.ROOT_VISIT_COUNT.value], int)
    assert isinstance(restored_variables[TreeMetrics.ACTIONS_VISIT_COUNT_ENTROPY.value], float)


def test_history_round_trip_preserves_one_entry_per_decision():
    """A multi-step episode keeps one run-data entry per decision, in order.

    Purpose: The statistics layer averages each metric across the decisions of
        an episode, so collapsing three decisions into one would change every
        reported average.

    Given: A history with three decisions whose root visit counts are 1, 2, 3.
    When: The history round-trips.
    Then: Three entries come back with those values in that order.

    Test type: unit
    """
    original = _history_with_metrics(
        policy_run_data=[
            PolicyRunData(
                info_variables=[
                    PolicyInfoVariable(name=TreeMetrics.ROOT_VISIT_COUNT.value, value=count)
                ]
            )
            for count in (1, 2, 3)
        ]
    )

    restored = History.from_dict(original.to_dict())

    assert [run.info_variables[0].value for run in restored.policy_run_data] == [1, 2, 3]


def test_history_round_trip_still_accepts_the_legacy_single_dict_shape():
    """A payload written before the fix, holding one dict, still loads.

    Purpose: ``from_dict`` accepted a single mapping; cached payloads written
        that way must keep loading rather than raising.

    Given: A dict payload whose ``policy_run_data`` is one mapping, not a list.
    When: ``from_dict`` runs.
    Then: It becomes a one-element list with the metric intact.

    Test type: unit
    """
    payload = _history_with_metrics().to_dict()
    payload["policy_run_data"] = {
        "info_variables": [{"name": TreeMetrics.ROOT_VISIT_COUNT.value, "value": 7}]
    }

    restored = History.from_dict(payload)

    assert len(restored.policy_run_data) == 1
    assert restored.policy_run_data[0].info_variables[0].value == 7


def test_history_round_trip_of_an_empty_metric_list_stays_empty():
    """A history with no recorded metrics round-trips to an empty list.

    Purpose: Guards the fix against the opposite error — inventing an entry
        where the episode recorded none.

    Test type: unit
    """
    restored = History.from_dict(_history_with_metrics(policy_run_data=[]).to_dict())

    assert restored.policy_run_data == []


def test_real_run_records_nonempty_planner_metrics_that_survive_the_cache(tmp_path):
    """A real POMCP run through the simulations API records metrics, and a
    second, cache-served call returns exactly the same ones.

    Purpose: This is the path that actually carries planner metrics in
        production. It uses ``LocalSimulationsAPI`` and ``EnvironmentRunParams``
        rather than a hand-written episode loop, so the cache, the per-episode
        records and the metric plumbing are all exercised together.

    Given: Tiger, a POMCP with a fixed simulation count, two episodes of one
        step, and a cache directory.
    When: The run executes, and then the identical run executes again against
        the same cache directory.
    Then: Every returned history carries a non-empty ``policy_run_data`` whose
        metric names are exactly the planner's declared set; and the second
        call's metric names and values equal the first call's, episode by
        episode and decision by decision.

    Test type: integration
    """
    env = TigerPOMDP(discount_factor=0.95)
    planner = POMCP(
        environment=env,
        discount_factor=0.95,
        depth=3,
        exploration_constant=1.0,
        name="MetricsPersistencePOMCP",
        n_simulations=4,
    )
    params = [
        EnvironmentRunParams(
            environment=env,
            belief=get_initial_belief(env, n_particles=10),
            policies=[planner],
            num_episodes=2,
            num_steps=1,
        )
    ]
    cache_dir = tmp_path / "metrics_cache"
    api = LocalSimulationsAPI()

    def run():
        results, _ = api.run_multiple_environments_and_policies(
            environment_run_params=params,
            alpha=0.05,
            confidence_interval_level=0.95,
            n_jobs=1,
            cache_dir_path=cache_dir,
        )
        return results

    first = run()
    second = run()

    declared = set(POMCP.get_info_variable_names())
    first_metrics = _extract(first)
    second_metrics = _extract(second)

    assert first_metrics, "the run returned no histories at all"
    for episode_index, decisions in enumerate(first_metrics):
        assert decisions, (
            f"episode {episode_index} recorded no policy_run_data; every decision must carry "
            "the planner's tree metrics"
        )
        for decision_index, variables in enumerate(decisions):
            assert set(variables) == declared, (
                f"episode {episode_index} decision {decision_index} reported "
                f"{sorted(variables)}, expected {sorted(declared)}"
            )
    assert second_metrics == first_metrics, (
        "the second, cache-served call returned different planner metrics from the first; "
        "stored run data must reproduce the metrics the planner reported"
    )


def _extract(results):
    """``results -> [ [ {metric name: value} per decision ] per episode ]``."""
    extracted = []
    for policies in results.values():
        for histories in policies.values():
            for history in histories:
                extracted.append(
                    [
                        {variable.name: variable.value for variable in run_data.info_variables}
                        for run_data in history.policy_run_data
                    ]
                )
    return extracted
