# SPDX-License-Identifier: MIT

"""Deterministic fixtures shared by the planner correctness tests.

Every planner backup rule in this repository is an equation over an immediate
reward (or cost), a discount, and a future value. None of the shipped
environments makes those three quantities separable by eye, so a test written
against Tiger or Light-Dark can only check types and ranges. This module
supplies the missing ingredient: a chain environment whose reward is a function
of the source state alone, whose transition is a function, and which records
every state it was asked to transition from.

The canonical fixture is :class:`ChainEnv`::

    root --(any action)--> next --(any action)--> end (terminal)

with ``reward(root) = 2``, ``reward(next) = 4``, ``reward(end) = 0``. At
discount ``0.5`` the return of the whole chain from ``root`` is
``2 + 0.5 * 4 = 4`` and from ``next`` is ``4``. Those two numbers appear as
expected values throughout the planner tests, always derived here by hand and
never by calling the code under test.

The observation is the state that was reached, so an observation identifies a
belief child uniquely and observation widening is exercisable without any
sampling noise.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import numpy as np

from POMDPPlanners.core.belief import (
    UnweightedParticleBeliefStateUpdate,
    WeightedParticleBelief,
)
from POMDPPlanners.core.distributions import Distribution
from POMDPPlanners.core.environment import (
    ConstrainedEnvironment,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.planners.planners_utils.dpw import ActionSampler


ROOT = "root"
NEXT = "next"
END = "end"

#: Reward earned for leaving each state. The successor and the action do not
#: change it, which is what makes ``belief_expectation_reward`` on a
#: single-particle belief exactly equal to this table's entry.
CHAIN_REWARDS: Dict[str, float] = {ROOT: 2.0, NEXT: 4.0, END: 0.0}

#: Deterministic successor of each state.
CHAIN_SUCCESSOR: Dict[str, str] = {ROOT: NEXT, NEXT: END, END: END}

#: Log-likelihood assigned to an observation that does not match the state it
#: was supposedly emitted from. Finite (``WeightedParticleBelief`` rejects
#: non-finite log weights) but small enough that a mismatched particle is
#: numerically dead after normalisation.
MISMATCH_LOG_PROB = -50.0


class ChainEnv(DiscreteActionsEnvironment):
    """Deterministic three-state chain with a per-source-state reward.

    Args:
        discount_factor: Discount used by planners built on this env. The
            environment itself only stores it.
        actions: The discrete action set. Two actions by default so that
            "the branch the simulation did not enter" is a meaningful check.
        terminal_states: States for which :meth:`is_terminal` returns ``True``.

    Attributes:
        transition_calls: ``(state, action)`` recorded for every
            :meth:`sample_next_state` call, in call order. This is how the
            tests establish which state a planner actually recursed from,
            rather than which state it claimed to.
        reward_calls: ``(state, action, next_state)`` for every
            :meth:`reward` call, in call order.
        terminal_calls: Every state passed to :meth:`is_terminal`.
    """

    def __init__(
        self,
        discount_factor: float = 0.5,
        actions: Optional[List[Any]] = None,
        terminal_states: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            discount_factor=discount_factor,
            name="ChainEnv",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
            reward_range=(0.0, 4.0),
        )
        self._actions: List[Any] = list(actions) if actions is not None else ["a", "b"]
        self._terminal_states: Tuple[str, ...] = (
            terminal_states if terminal_states is not None else (END,)
        )
        self.transition_calls: List[Tuple[Any, Any]] = []
        self.reward_calls: List[Tuple[Any, Any, Any]] = []
        self.terminal_calls: List[Any] = []

    # --- discrete action space ---

    def get_actions(self) -> List[Any]:
        return list(self._actions)

    # --- dynamics ---

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        self.transition_calls.append((state, action))
        successor = CHAIN_SUCCESSOR[state]
        return successor if n_samples == 1 else [successor] * n_samples

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        return next_state if n_samples == 1 else [next_state] * n_samples

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        self.reward_calls.append((state, action, next_state))
        return CHAIN_REWARDS[state]

    def is_terminal(self, state: Any) -> bool:
        self.terminal_calls.append(state)
        return state in self._terminal_states

    # --- likelihoods ---

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del action
        expected = CHAIN_SUCCESSOR[state]
        return np.array(
            [0.0 if ns == expected else MISMATCH_LOG_PROB for ns in next_states],
            dtype=np.float64,
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        return np.array(
            [0.0 if obs == next_state else MISMATCH_LOG_PROB for obs in observations],
            dtype=np.float64,
        )

    # --- identity / hashing ---

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def hash_observation(self, observation: Any) -> Any:
        return observation

    def hash_action(self, action: Any) -> Any:
        return action

    # --- initial distributions (unused by the planner tests) ---

    def initial_state_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[ROOT])
        return dist

    def initial_observation_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[ROOT])
        return dist

    # --- test conveniences ---

    def transitioned_states(self) -> List[Any]:
        """The states this environment was asked to transition from, in order."""
        return [state for state, _ in self.transition_calls]

    def reset_call_log(self) -> None:
        self.transition_calls.clear()
        self.reward_calls.clear()
        self.terminal_calls.clear()


class ConstrainedChainEnv(ChainEnv, ConstrainedEnvironment):
    """:class:`ChainEnv` with a vector constraint cost keyed on the source state.

    The default table gives ``[1, 3]`` for leaving ``root`` and ``[2, 0]`` for
    leaving ``next``. Those two vectors were chosen so that the discounted sum
    ``[1, 3] + 0.5 * [2, 0] = [2, 3]`` has two *different* entries whose order
    also differs from the immediate cost's order — a planner that swapped the
    two channels, or that dropped the future term, produces a visibly different
    vector rather than a coincidentally equal one.
    """

    def __init__(
        self,
        discount_factor: float = 0.5,
        actions: Optional[List[Any]] = None,
        costs: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        ChainEnv.__init__(self, discount_factor=discount_factor, actions=actions)
        self._costs: Dict[str, np.ndarray] = {
            state: np.asarray(value, dtype=np.float64)
            for state, value in (
                costs
                if costs is not None
                else {ROOT: [1.0, 3.0], NEXT: [2.0, 0.0], END: [0.0, 0.0]}
            ).items()
        }
        self.constraint_cost_calls: List[Tuple[Any, Any, Any]] = []

    def constraint_cost(self, state: Any, action: Any, next_state: Any) -> np.ndarray:
        self.constraint_cost_calls.append((state, action, next_state))
        return self._costs[state].copy()

    @property
    def n_constraints(self) -> int:
        return int(next(iter(self._costs.values())).shape[0])


class FixedActionSampler(ActionSampler):
    """Cycles through a fixed action list so widening is fully deterministic.

    ``ActionSampler.sample`` is the only source of new actions in every
    progressive-widening planner. Making it a deterministic cycle instead of a
    draw is what lets a widening-boundary test say which action is added on
    which call.
    """

    def __init__(self, actions: List[Any]) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        self._actions = list(actions)
        self._index = 0
        self.sample_calls = 0

    def sample(self, belief_node: Any = None) -> Any:
        del belief_node
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        self.sample_calls += 1
        return action

    def get_space(self) -> List[Any]:
        return list(self._actions)


class SingleActionSampler(FixedActionSampler):
    """Always returns the same action; no widening branching at all."""

    def __init__(self, action: Any = "a") -> None:
        super().__init__([action])


def chain_belief(state: str = ROOT) -> WeightedParticleBelief:
    """A one-particle weighted belief concentrated on ``state``.

    ``WeightedParticleBelief`` rejects an all-zero ``log_weights`` vector, so a
    single particle is given an arbitrary non-zero log weight; it normalises to
    1.0 either way.
    """
    return WeightedParticleBelief(particles=[state], log_weights=np.array([-1.0]))


def chain_state_update_belief(state: str = ROOT) -> UnweightedParticleBeliefStateUpdate:
    """A one-particle state-update belief concentrated on ``state``.

    This is the belief type POMCP *allocates* for its own non-root nodes, so it
    is the right input for tests that call ``_simulate_state_path`` or
    ``_learn_tree`` directly. It is **not** accepted by the public ``action()``
    path: ``ArenaPathSimulationPolicy._is_terminal_belief`` only recognises the
    weighted particle and Gaussian families and raises ``ValueError`` otherwise.
    Use :func:`chain_belief` for anything that goes through ``action()``.
    """
    return UnweightedParticleBeliefStateUpdate(particles=[state])


def two_state_belief(first: str, second: str, first_weight: float = 0.5) -> WeightedParticleBelief:
    """A two-particle belief with an explicit probability on ``first``.

    Used by the belief-conditioning checks, where the posterior has to be
    computed by hand from the prior and the likelihoods.
    """
    if not 0.0 < first_weight < 1.0:
        raise ValueError("first_weight must be strictly between 0 and 1")
    return WeightedParticleBelief(
        particles=[first, second],
        log_weights=np.array([math.log(first_weight), math.log(1.0 - first_weight)]),
    )


def discounted_return(rewards: List[float], discount: float) -> float:
    """Hand-checkable ``sum(r_t * discount**t)``.

    Deliberately a plain loop over the test's own reward list; it never reads
    anything from a planner, a tree, or an environment.
    """
    return float(sum(reward * discount**index for index, reward in enumerate(rewards)))


# ---------------------------------------------------------------------------
# State-threading fixture
# ---------------------------------------------------------------------------

#: Two non-terminal self-looping states whose rewards are far apart. Used by the
#: state-threading tests: because each state is its own successor, a one-step
#: search's return is exactly the source state's reward, so "which state did the
#: planner transition from" is readable straight off the returned number.
THREAD_A = "thread_a"
THREAD_B = "thread_b"

THREAD_REWARDS: Dict[str, float] = {THREAD_A: 1.0, THREAD_B: 10.0}


class SelfLoopEnv(DiscreteActionsEnvironment):
    """Two absorbing-but-nonterminal states with very different rewards.

    ``sample_next_state(s, a) == s`` for both states, so a depth-0 search's
    return is ``reward(s)`` and nothing else. That makes the return a direct
    readout of which state the planner fed to the generative model: 1.0 for
    ``THREAD_A``, 10.0 for ``THREAD_B``. A planner that discards a threaded
    state and redraws one from the node's belief therefore produces a
    different number rather than a coincidentally equal one.
    """

    def __init__(self, discount_factor: float = 0.5, terminal_states: Tuple[str, ...] = ()) -> None:
        super().__init__(
            discount_factor=discount_factor,
            name="SelfLoopEnv",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
            reward_range=(0.0, 10.0),
        )
        self._terminal_states = terminal_states
        self.transition_calls: List[Tuple[Any, Any]] = []
        self.terminal_calls: List[Any] = []

    def get_actions(self) -> List[Any]:
        return ["a", "b"]

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        self.transition_calls.append((state, action))
        return state if n_samples == 1 else [state] * n_samples

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        return next_state if n_samples == 1 else [next_state] * n_samples

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del action, next_state
        return THREAD_REWARDS[state]

    def is_terminal(self, state: Any) -> bool:
        self.terminal_calls.append(state)
        return state in self._terminal_states

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del action
        return np.array(
            [0.0 if ns == state else MISMATCH_LOG_PROB for ns in next_states],
            dtype=np.float64,
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        return np.array(
            [0.0 if obs == next_state else MISMATCH_LOG_PROB for obs in observations],
            dtype=np.float64,
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def hash_observation(self, observation: Any) -> Any:
        return observation

    def hash_action(self, action: Any) -> Any:
        return action

    def initial_state_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[THREAD_A])
        return dist

    def initial_observation_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[THREAD_A])
        return dist

    def transitioned_states(self) -> List[Any]:
        return [state for state, _ in self.transition_calls]


class ActionRewardEnv(DiscreteActionsEnvironment):
    """Deterministic chain whose reward depends on the *action*, not the state.

    The open-loop planner enumerates action sequences, so a fixture whose
    reward ignores the action cannot distinguish one sequence from another.
    Here ``reward(s, a) = ACTION_REWARDS[a]`` while the state advances along
    ``s0 -> s1 -> s2(terminal, reward 0 for every action)``, which makes each
    sequence's discounted return a short sum the test writes out by hand.
    """

    #: Per-action reward while the state is non-terminal.
    ACTION_REWARDS: Dict[str, float] = {"a": 1.0, "b": 3.0}

    def __init__(self, discount_factor: float = 0.5, horizon: int = 2) -> None:
        super().__init__(
            discount_factor=discount_factor,
            name="ActionRewardEnv",
            space_info=SpaceInfo(
                action_space=SpaceType.DISCRETE,
                observation_space=SpaceType.DISCRETE,
            ),
            reward_range=(0.0, 3.0),
        )
        # States are ints counting how many steps have been taken; anything at
        # or beyond ``horizon`` is the absorbing zero-reward terminal state.
        self.horizon = horizon

    def get_actions(self) -> List[Any]:
        return ["a", "b"]

    def sample_next_state(self, state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        successor = min(int(state) + 1, self.horizon)
        return successor if n_samples == 1 else [successor] * n_samples

    def sample_observation(self, next_state: Any, action: Any, n_samples: int = 1) -> Any:
        del action
        return next_state if n_samples == 1 else [next_state] * n_samples

    def reward(self, state: Any, action: Any, next_state: Any = None) -> float:
        del next_state
        if int(state) >= self.horizon:
            return 0.0
        return self.ACTION_REWARDS[action]

    def is_terminal(self, state: Any) -> bool:
        return int(state) >= self.horizon

    def transition_log_probability(self, state: Any, action: Any, next_states: Any) -> np.ndarray:
        del action
        expected = min(int(state) + 1, self.horizon)
        return np.array(
            [0.0 if ns == expected else MISMATCH_LOG_PROB for ns in next_states],
            dtype=np.float64,
        )

    def observation_log_probability(
        self, next_state: Any, action: Any, observations: Any
    ) -> np.ndarray:
        del action
        return np.array(
            [0.0 if obs == next_state else MISMATCH_LOG_PROB for obs in observations],
            dtype=np.float64,
        )

    def is_equal_observation(self, observation1: Any, observation2: Any) -> bool:
        return observation1 == observation2

    def hash_observation(self, observation: Any) -> Any:
        return observation

    def hash_action(self, action: Any) -> Any:
        return action

    def initial_state_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[0])
        return dist

    def initial_observation_dist(self) -> Distribution:
        dist = Mock(spec=Distribution)
        dist.sample = Mock(return_value=[0])
        return dist
