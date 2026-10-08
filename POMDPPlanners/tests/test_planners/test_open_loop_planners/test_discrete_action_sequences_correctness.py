# SPDX-License-Identifier: MIT

"""DiscreteActionSequences correctness: exact returns, selection, action shape.

The existing suite checks that the return is finite and that ``actions[0]`` is
legal. That leaves the three things that matter unchecked: whether the
enumerated returns are the right numbers, whether the argmax picks the right
sequence, and whether the public return is one action or the whole plan.

``ActionRewardEnv`` makes the reward a function of the *action* — the chain
fixture's reward depends only on the state, which cannot tell one sequence from
another — and its state advances to an absorbing zero-reward terminal state
after a fixed horizon, so terminal behaviour is checkable too.
"""

# pylint: disable=protected-access

import itertools
import random

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.planners.open_loop_planners.discrete_action_sequences_planner import (
    DiscreteActionSequencesPlanner,
)
from POMDPPlanners.tests.test_planners.planner_fixtures import ActionRewardEnv


np.random.seed(42)
random.seed(42)

DISCOUNT = 0.5
TOL = 1e-12


def _belief(state=0):
    return WeightedParticleBelief(particles=[state], log_weights=np.array([-1.0]))


def _planner(env, *, depth=2, n_return_samples=1, name="das"):
    return DiscreteActionSequencesPlanner(
        environment=env,
        discount_factor=env.discount_factor,
        depth=depth,
        name=name,
        n_return_samples=n_return_samples,
    )


def _hand_computed_return(sequence, discount=DISCOUNT, horizon=2):
    """The discounted return of one sequence, written out step by step.

    Deliberately a plain loop over the fixture's own reward table. It reads
    nothing from the planner, so it is an independent expectation rather than a
    restatement of the code under test.
    """
    total = 0.0
    state = 0
    for index, action in enumerate(sequence):
        reward = 0.0 if state >= horizon else ActionRewardEnv.ACTION_REWARDS[action]
        total += reward * discount**index
        state = min(state + 1, horizon)
    return total


# ---------------------------------------------------------------------------
# Exact enumerated returns
# ---------------------------------------------------------------------------


def test_every_depth_two_sequence_return_matches_the_hand_computed_value():
    """All four sequences at depth 2 score exactly what the arithmetic says.

    Purpose: This is the planner's whole computation. Written out, with
        ``r(a) = 1``, ``r(b) = 3`` and discount 0.5:

        * ``(a, a)`` -> ``1 + 0.5*1 = 1.5``
        * ``(a, b)`` -> ``1 + 0.5*3 = 2.5``
        * ``(b, a)`` -> ``3 + 0.5*1 = 3.5``
        * ``(b, b)`` -> ``3 + 0.5*3 = 4.5``

        A missing discount makes the middle two equal at 4; discounting the
        first step as well scales everything by 0.5; reversing the discount
        powers swaps ``(a, b)`` and ``(b, a)``. All three differ from the list
        above.

    Given: The deterministic action-reward environment and one return sample,
        which is exact because the environment has no randomness.
    When: ``estimate_return`` is called on each sequence.
    Then: Each equals its hand-computed value.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    planner = _planner(env)
    expected = {
        ("a", "a"): 1.5,
        ("a", "b"): 2.5,
        ("b", "a"): 3.5,
        ("b", "b"): 4.5,
    }

    for sequence, want in expected.items():
        assert _hand_computed_return(sequence) == pytest.approx(want, abs=TOL)
        got = planner.estimate_return(action_sequence=list(sequence), belief=_belief())
        assert got == pytest.approx(
            want, abs=TOL
        ), f"sequence {sequence} scored {got}, expected {want}"


def test_search_returns_the_best_sequence_in_full():
    """The best sequence is returned whole, in order.

    Purpose: The public action shape is the contract that makes this an
        open-loop planner: ``EpisodeRunner._execute_policy_step`` iterates over
        every element of the returned list and executes each in turn, subject to
        the remaining step budget. Returning only the first action would turn
        the planner into a closed-loop one that replans every step, which is a
        different algorithm. The module docstring used to say "the first
        action"; the docstring was corrected, not the behaviour.

    Given: Depth 2 over the reward table above, where ``(b, b)`` scores highest.
    When: ``search`` runs.
    Then: The result is the list ``["b", "b"]`` — both actions, in order.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    planner = _planner(env)

    result = planner.search(_belief())

    assert result == ["b", "b"], f"search returned {result}, expected the full best sequence"


def test_action_returns_the_whole_sequence_and_no_metrics():
    """``action()`` hands back ``depth`` actions and an empty metric list.

    Purpose: Both halves are contracts the episode runner and the statistics
        layer depend on.

    Given: Depth 3.
    When: ``action()`` is called.
    Then: Three legal actions come back, in the same order ``search`` produced,
        and ``info_variables`` is empty — this planner declares no metrics.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=3)
    planner = _planner(env, depth=3)

    actions, run_data = planner.action(_belief())

    assert len(actions) == 3, (
        f"action() returned {len(actions)} actions for depth 3; the open-loop contract is the "
        "whole sequence, which the episode runner executes in order"
    )
    assert all(action in env.get_actions() for action in actions)
    assert run_data.info_variables == []
    assert DiscreteActionSequencesPlanner.get_info_variable_names() == []


def test_the_tie_rule_is_the_first_sequence_in_enumeration_order():
    """On a tie the earliest sequence in ``itertools.product`` order wins.

    Purpose: ``np.argmax`` returns the first maximal index, and the sequences
        are enumerated by ``product(actions, repeat=depth)``. That makes the
        planner deterministic on ties, which is worth pinning: a change to
        either the enumeration order or the argmax would silently change which
        plan a tied decision executes.

    Given: An environment where both actions earn the same reward, so all four
        depth-2 sequences tie.
    When: ``search`` runs.
    Then: The first sequence in enumeration order, ``("a", "a")``, is returned.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    env.ACTION_REWARDS = {"a": 2.0, "b": 2.0}
    planner = _planner(env)

    first_enumerated = list(itertools.product(env.get_actions(), repeat=2))[0]
    assert first_enumerated == ("a", "a")

    assert planner.search(_belief()) == ["a", "a"]


def test_a_reward_table_favouring_the_other_action_flips_the_answer():
    """The selection follows the rewards rather than the enumeration order.

    Purpose: Guards the tie test above from proving only that "a" always wins.

    Given: The reward table reversed, so ``a`` earns 3 and ``b`` earns 1.
    When: ``search`` runs.
    Then: ``["a", "a"]`` wins on merit, and its score is 4.5.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    env.ACTION_REWARDS = {"a": 3.0, "b": 1.0}
    planner = _planner(env)

    assert planner.search(_belief()) == ["a", "a"]
    assert planner.estimate_return(["a", "a"], _belief()) == pytest.approx(4.5, abs=TOL)


# ---------------------------------------------------------------------------
# Terminal behaviour and discount boundaries
# ---------------------------------------------------------------------------


def test_steps_after_the_terminal_state_contribute_nothing():
    """An absorbing zero-reward terminal state ends the accumulation.

    Purpose: This planner has no terminal check — it always simulates ``depth``
        steps. Its correctness therefore rests on the environment's terminal
        state being absorbing and unrewarded, and that is what is asserted:
        the reward is counted once, at the transition into the terminal state,
        and nothing accrues afterwards.

    Given: A horizon of 2, so the state is terminal from step 2 onward, and
        sequences of length 2 and 4 sharing the same prefix.
    When: Both are scored.
    Then: They score the same, and that score is the depth-2 value 4.5.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    planner = _planner(env, depth=4)

    short = planner.estimate_return(["b", "b"], _belief())
    long = planner.estimate_return(["b", "b", "b", "b"], _belief())

    assert short == pytest.approx(4.5, abs=TOL)
    assert long == pytest.approx(short, abs=TOL), (
        f"a four-step sequence scored {long} against the two-step {short}; the two extra steps "
        "happen in an absorbing terminal state and must earn nothing"
    )


def test_a_belief_already_in_the_terminal_state_scores_zero():
    """Planning from a terminal belief gives a return of zero for every sequence.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    planner = _planner(env)

    for sequence in itertools.product(env.get_actions(), repeat=2):
        assert planner.estimate_return(list(sequence), _belief(state=2)) == pytest.approx(
            0.0, abs=TOL
        )


def test_discount_zero_keeps_only_the_first_reward():
    """With gamma = 0 the return is the first step's reward alone.

    Purpose: A boundary that pins the discount's exponent: ``0**0 = 1`` weights
        the first step fully and every later one at zero.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=0.0, horizon=2)
    planner = _planner(env)

    assert planner.estimate_return(["b", "a"], _belief()) == pytest.approx(3.0, abs=TOL)
    assert planner.estimate_return(["a", "b"], _belief()) == pytest.approx(1.0, abs=TOL)


def test_discount_one_preserves_the_undiscounted_sum():
    """With gamma = 1 over a finite horizon the return is the plain sum.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=1.0, horizon=2)
    planner = _planner(env)

    assert planner.estimate_return(["a", "b"], _belief()) == pytest.approx(4.0, abs=TOL)
    assert planner.estimate_return(["b", "a"], _belief()) == pytest.approx(4.0, abs=TOL)


# ---------------------------------------------------------------------------
# Sampling, construction, caller belief
# ---------------------------------------------------------------------------


def test_the_estimate_averages_over_the_requested_number_of_samples():
    """``n_return_samples`` draws are averaged, not summed.

    Purpose: A missing division would scale every score by the sample count.
        The environment is deterministic, so the average of any number of
        identical samples is the single-sample value; a sum would be that value
        times the count.

    Given: The same sequence estimated with 1 and with 8 samples.
    When: Both estimates are taken.
    Then: They are equal.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    one = _planner(env, n_return_samples=1)
    many = _planner(env, n_return_samples=8)

    assert one.estimate_return(["b", "b"], _belief()) == pytest.approx(
        many.estimate_return(["b", "b"], _belief()), abs=TOL
    )


def test_planning_does_not_mutate_the_callers_belief():
    """The belief handed in is unchanged after a decision.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    planner = _planner(env, n_return_samples=4)
    belief = _belief()
    before_particles = list(belief.particles)
    before_weights = np.array(belief.log_weights, copy=True)

    planner.action(belief)

    assert belief.particles == before_particles
    assert np.array_equal(np.asarray(belief.log_weights), before_weights)


def test_invalid_construction_parameters_are_rejected():
    """Depth, sample count and discount all have stated valid ranges.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=2)
    with pytest.raises(ValueError, match="depth"):
        _planner(env, depth=0)
    with pytest.raises(ValueError, match="n_return_samples"):
        _planner(env, n_return_samples=0)
    with pytest.raises(ValueError, match="discount_factor"):
        DiscreteActionSequencesPlanner(
            environment=env, discount_factor=1.5, depth=2, name="bad", n_return_samples=1
        )


def test_the_number_of_enumerated_sequences_is_actions_to_the_depth():
    """Enumeration is exhaustive: ``|A|**depth`` sequences, no widening.

    Purpose: Progressive widening does not apply to this planner. Exhaustive
        enumeration is the contract that replaces it, and the cost of a deeper
        plan follows from it.

    Test type: unit
    """
    env = ActionRewardEnv(discount_factor=DISCOUNT, horizon=3)
    planner = _planner(env, depth=3)
    scored = []
    original = planner.estimate_return

    def spy(action_sequence, belief):
        scored.append(tuple(action_sequence))
        return original(action_sequence=action_sequence, belief=belief)

    planner.estimate_return = spy  # type: ignore[method-assign]
    planner.search(_belief())

    assert len(scored) == len(env.get_actions()) ** 3 == 8
    assert len(set(scored)) == 8, "every sequence must be scored exactly once"
