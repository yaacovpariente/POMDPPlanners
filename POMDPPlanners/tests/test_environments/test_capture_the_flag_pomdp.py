# SPDX-License-Identifier: MIT

"""Tests for the CaptureTheFlag POMDP environment.

Covers what is specific to this environment rather than the shared contracts:
the ordered transition stages, the tag-and-respawn rules, the factored
observation model, and the reward bound in the configurations that stress it.
The cross-environment contracts are covered once, in
``test_env_api_conformance.py``.
"""

import itertools
from typing import Dict, List, Tuple, cast

import numpy as np
import pytest

from POMDPPlanners.core.distributions import DiscreteDistribution

from POMDPPlanners.environments.capture_the_flag_pomdp import (
    CaptureTheFlagMetrics,
    CaptureTheFlagPOMDP,
    CaptureTheFlagStepChannel,
    decode_joint_action,
    encode_joint_action,
)
from POMDPPlanners.environments.capture_the_flag_pomdp.capture_the_flag_pomdp_utils import (
    ACTION_EAST,
    ACTION_NORTH,
    ACTION_SCAN,
    ACTION_STAY,
)


@pytest.fixture(name="env")
def env_fixture() -> CaptureTheFlagPOMDP:
    """Return the reference 9x7 two-a-side environment."""
    return CaptureTheFlagPOMDP()


def build_state(
    env: CaptureTheFlagPOMDP,
    blue: List[Tuple[int, int]],
    red: List[Tuple[int, int]],
    flag: Tuple[int, int],
    **fields: float,
) -> np.ndarray:
    """Assemble a state vector from readable parts.

    Args:
        env: The environment whose layout is used.
        blue: Blue player cells.
        red: Red player cells.
        flag: The red flag's home cell; must be a candidate.
        **fields: Scalar layout fields to override, e.g. ``carrier_red_flag=1``
            or indexed counters as ``freeze_blue_0=3``.

    Returns:
        The state vector.
    """
    layout = env.layout
    state = np.zeros(layout.size, dtype=np.float64)
    layout.write_blue_cells(state, blue)
    layout.write_red_cells(state, red)
    state[layout.flag_cell] = env.red_flag_candidates.index(flag)
    for name, value in fields.items():
        if name[-2] == "_" and name[-1].isdigit():
            base, index = name[:-2], int(name[-1])
            state[getattr(layout, base) + index] = float(value)
        else:
            state[getattr(layout, name)] = float(value)
    return state


def opening_states(env: CaptureTheFlagPOMDP) -> List[np.ndarray]:
    """Return the opening states, one per red flag candidate.

    Args:
        env: The environment whose initial distribution is read.

    Returns:
        The support of the initial state distribution.
    """
    return list(cast(DiscreteDistribution, env.initial_state_dist()).values)


def successors(env: CaptureTheFlagPOMDP, state: np.ndarray, action: int) -> Dict[bytes, float]:
    """Return the successor distribution keyed by state bytes."""
    # pylint: disable=protected-access
    states, probabilities = env._successor_distribution(
        state, action
    )  # pylint: disable=protected-access
    return {s.tobytes(): float(p) for s, p in zip(states, probabilities)}


class TestJointActions:
    """The blue team's joint action encoding."""

    def test_decode_inverts_encode_over_the_whole_space(self, env: CaptureTheFlagPOMDP) -> None:
        """Every joint action id round-trips through its per-player digits.

        Purpose: The joint action is a base-6 odometer over the team; an
            off-by-one in the digit order would silently permute which player
            receives which action.

        Given: Every joint action id in the reference two-player space
        When: Each is decoded to per-player actions and re-encoded
        Then: The original id comes back, and the space has 6**n_blue members

        Test type: unit
        """
        assert len(env.get_actions()) == 6**env.n_blue
        for action in env.get_actions():
            assert encode_joint_action(decode_joint_action(action, env.n_blue)) == action

    def test_first_digit_belongs_to_player_zero(self) -> None:
        """Player 0's action is the least significant digit.

        Purpose: Pins the convention the transition and the visualizer both
            assume, so a future reordering fails here rather than silently
            swapping the squad's orders.

        Given: A joint action built from distinct per-player actions
        When: It is decoded
        Then: The digits come back in player order

        Test type: unit
        """
        action = encode_joint_action([ACTION_SCAN, ACTION_NORTH])
        assert decode_joint_action(action, 2) == (ACTION_SCAN, ACTION_NORTH)


class TestTransitionModel:  # pylint: disable=protected-access
    """The ordered transition stages."""

    def test_successor_probabilities_sum_to_one(self, env: CaptureTheFlagPOMDP) -> None:
        """Every enumerated successor distribution normalises.

        Purpose: The distribution is a product over per-player move outcomes
            merged by resulting state; a merge that dropped or double-counted a
            branch would show up as mass that is not one.

        Given: A spread of reachable states and joint actions
        When: The successor distribution is enumerated for each pair
        Then: Each sums to one

        Test type: unit
        """
        rng = np.random.default_rng(0)
        np.random.seed(0)
        state = opening_states(env)[1]
        for _ in range(12):
            for action in (0, 7, 14, 21, 35):
                _, probabilities = env._successor_distribution(  # pylint: disable=protected-access
                    state, action
                )
                assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
            state = env.sample_next_state(state, int(rng.integers(0, 36)))
            if env.is_terminal(state):
                state = opening_states(env)[1]

    def test_sampling_matches_the_enumerated_distribution(self, env: CaptureTheFlagPOMDP) -> None:
        """Sampling and enumeration describe the same transition.

        Purpose: ``sample_next_state`` draws per player while
            ``transition_log_probability`` enumerates joint outcomes. The two
            are separate code paths over one model, which is exactly where
            drift hides.

        Given: One state and action
        When: Successors are drawn many times and compared to the exact weights
        Then: Every cell agrees within Monte-Carlo error

        Test type: unit
        """
        state = build_state(env, [(3, 3), (5, 5)], [(7, 4), (7, 5)], (6, 3))
        exact = successors(env, state, 11)
        np.random.seed(7)
        counts: Dict[bytes, int] = {}
        draws = 20000
        for _ in range(draws):
            key = env.sample_next_state(state, 11).tobytes()
            counts[key] = counts.get(key, 0) + 1
        assert set(counts) <= set(exact)
        for key, probability in exact.items():
            assert counts.get(key, 0) / draws == pytest.approx(probability, abs=0.01)

    def test_a_blocked_move_leaves_the_player_in_place(self) -> None:
        """Walking into a tree costs the step but not the position.

        Given: A blue player standing west of the tree at (3, 3)
        When: It is ordered east with slip disabled
        Then: It has not moved

        Test type: unit
        """
        quiet = CaptureTheFlagPOMDP(slip_probability=0.0, red_pursuit_probability=0.0)
        state = build_state(quiet, [(2, 3), (0, 3)], [(8, 3), (8, 3)], (6, 3))
        action = encode_joint_action([ACTION_EAST, ACTION_STAY])
        for successor, probability in zip(
            *quiet._successor_distribution(state, action)
        ):  # pylint: disable=protected-access
            if probability > 0:
                assert quiet.layout.blue_cells(successor)[0] == (2, 3)

    def test_a_frozen_player_does_not_move(self, env: CaptureTheFlagPOMDP) -> None:
        """A respawn freeze pins the player for its duration.

        Given: A blue player with a freeze counter running
        When: It is ordered to move
        Then: Every successor leaves it where it stood, with the counter one lower

        Test type: unit
        """
        state = build_state(env, [(1, 3), (0, 3)], [(8, 3), (8, 3)], (6, 3), freeze_blue_0=3)
        action = encode_joint_action([ACTION_EAST, ACTION_STAY])
        for successor, probability in zip(
            *env._successor_distribution(state, action)
        ):  # pylint: disable=protected-access
            if probability > 0:
                assert env.layout.blue_cells(successor)[0] == (1, 3)
                assert successor[env.layout.freeze_blue] == 2.0


class TestTaggingAndRespawn:
    """Tagging is a setback, not an elimination."""

    def _still(self, **overrides: object) -> CaptureTheFlagPOMDP:
        """Return an environment whose movement is deterministic.

        ``red_pursuit_probability=1.0``, not 0.0: at zero the red policy is
        *uniform* over its neighbours, which is the least still it can be. At
        one it steps straight at its target, so placing a red player on its own
        target is what pins it in place. Teams are cut to one red player so no
        second target is in play.
        """
        settings: Dict[str, object] = {
            "slip_probability": 0.0,
            "red_pursuit_probability": 1.0,
            "n_red": 1,
            "n_red_defenders": 1,
        }
        settings.update(overrides)
        return CaptureTheFlagPOMDP(**settings)  # type: ignore[arg-type]

    def test_a_tagged_player_respawns_frozen_at_its_own_base(self) -> None:
        """A blue player caught in the red half goes home and freezes.

        Purpose: This is the rule the whole episode structure rests on -- being
            caught must cost time, not the episode.

        Given: A blue player sharing a cell with a red player in the red half
        When: Both hold still
        Then: The blue player is at its base with a full freeze counter, and
            the red tagger is on cooldown

        Test type: unit
        """
        env = self._still()
        state = build_state(env, [(7, 4), (0, 3)], [(7, 4)], (6, 3))
        action = encode_joint_action([ACTION_STAY, ACTION_STAY])
        successor = env.sample_next_state(state, action)
        layout = env.layout
        assert layout.blue_cells(successor)[0] == env.blue_base
        assert successor[layout.freeze_blue] == env.freeze_steps
        assert successor[layout.cooldown_red] == env.tagger_cooldown_steps

    def test_no_tagging_outside_the_enemy_half(self) -> None:
        """Sharing a cell in your own half is not a tag.

        Purpose: Territory is what stops a defender camping the attacker's
            spawn; without it the environment degenerates into pursuit.

        Given: A blue and a red attacker sharing the blue flag cell, in the
            blue half
        When: Both hold still
        Then: The blue player is untouched, and the red one -- standing in the
            enemy half -- is the one tagged and sent home

        Test type: unit
        """
        env = self._still(n_red_defenders=0)
        state = build_state(env, [(1, 3), (0, 3)], [(1, 3)], (6, 3))
        successor = env.sample_next_state(state, encode_joint_action([ACTION_STAY, ACTION_STAY]))
        layout = env.layout
        assert layout.blue_cells(successor)[0] == (1, 3)
        assert successor[layout.freeze_blue] == 0.0
        assert layout.red_cells(successor)[0] == env.red_base
        assert successor[layout.freeze_red] == env.freeze_steps
        # It grabbed the flag on the way in, and dropped it when tagged.
        assert successor[layout.carrier_blue_flag] == 0.0

    def test_a_tagger_on_cooldown_cannot_tag(self) -> None:
        """A cooldown keeps a defender from tagging twice in a row.

        Given: A red player on cooldown sharing a cell with a blue player in
            the red half
        When: Both hold still
        Then: The blue player is untouched and the cooldown ticks down

        Test type: unit
        """
        env = self._still()
        state = build_state(env, [(7, 4), (0, 3)], [(7, 4)], (6, 3), cooldown_red_0=2)
        successor = env.sample_next_state(state, encode_joint_action([ACTION_STAY, ACTION_STAY]))
        layout = env.layout
        assert layout.blue_cells(successor)[0] == (7, 4)
        assert successor[layout.freeze_blue] == 0.0
        assert successor[layout.cooldown_red] == 1.0

    def test_a_tagged_carrier_drops_the_flag_home(self) -> None:
        """Pick-up resolves before tagging, so a carrier caught on the flag drops it.

        Purpose: The stage order is semantics. Tagging first would leave a
            player tagged on the flag cell holding nothing, and the flag
            unpicked -- a different game.

        Given: A blue player on the red flag cell, co-located with a red player
        When: Both hold still
        Then: The flag is home again, not carried, and the player respawned

        Test type: unit
        """
        env = self._still()
        state = build_state(env, [(6, 3), (0, 3)], [(6, 3)], (6, 3))
        successor = env.sample_next_state(state, encode_joint_action([ACTION_STAY, ACTION_STAY]))
        layout = env.layout
        assert successor[layout.carrier_red_flag] == 0.0
        assert layout.blue_cells(successor)[0] == env.blue_base
        assert successor[layout.freeze_blue] == env.freeze_steps


class TestScoring:
    """What it takes to score, and what it rules out."""

    def test_scoring_requires_your_own_flag_to_be_home(self) -> None:
        """A carrier cannot score while the enemy holds its flag.

        Purpose: This rule is what makes defending matter; without it the
            optimal policy ignores its own half entirely.

        Given: A blue carrier standing on its base, once with the blue flag
            home and once with it stolen
        When: The step resolves
        Then: Only the first scores

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(
            slip_probability=0.0, red_pursuit_probability=1.0, n_red=1, n_red_defenders=1
        )
        layout = env.layout
        scoring = build_state(env, [(0, 3), (0, 3)], [(6, 3)], (6, 3), carrier_red_flag=1)
        blocked = build_state(
            env,
            [(0, 3), (0, 3)],
            [(6, 3)],
            (6, 3),
            carrier_red_flag=1,
            carrier_blue_flag=1,
        )
        action = encode_joint_action([ACTION_STAY, ACTION_STAY])
        assert env.sample_next_state(scoring, action)[layout.score_blue] == 1.0
        assert env.sample_next_state(blocked, action)[layout.score_blue] == 0.0

    def test_both_sides_cannot_score_on_one_step(self) -> None:
        """Blue scoring and red scoring are mutually exclusive by construction.

        Purpose: Each requires the other side's flag to be home, so the two
            conditions cannot both hold. Evaluating them in sequence rather
            than against the same carrier ids would let whichever ran first
            enable the other, and the reward bound assumes it cannot.

        Given: Every reachable state of a long random rollout
        When: The score deltas are inspected
        Then: No step increments both scores

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(score_to_win=99)
        layout = env.layout
        np.random.seed(3)
        state = opening_states(env)[0]
        for _ in range(600):
            action = int(np.random.randint(0, len(env.get_actions())))
            successor = env.sample_next_state(state, action)
            grew_blue = successor[layout.score_blue] > state[layout.score_blue]
            grew_red = successor[layout.score_red] > state[layout.score_red]
            assert not (grew_blue and grew_red)
            state = successor

    def test_terminal_once_either_side_reaches_the_winning_score(
        self, env: CaptureTheFlagPOMDP
    ) -> None:
        """The episode ends on the winning capture, either way.

        Given: States at, below and above the winning score for each side
        When: ``is_terminal`` is asked
        Then: Only the states at or above the threshold are terminal

        Test type: unit
        """
        base = build_state(env, [(0, 3), (0, 3)], [(8, 3), (8, 3)], (6, 3))
        assert not env.is_terminal(base)
        for field in ("score_blue", "score_red"):
            scored = base.copy()
            scored[getattr(env.layout, field)] = env.score_to_win
            assert env.is_terminal(scored)


class TestObservationModel:  # pylint: disable=protected-access
    """The factored range and flag-detector likelihood."""

    def test_likelihood_mass_sums_to_one(self, env: CaptureTheFlagPOMDP) -> None:
        """The observation distribution normalises over its whole support.

        Purpose: The range readings are clipped at the field's extremes, and
            the mass that falls outside is folded back onto the end points. A
            fold that lost mass would leave a likelihood that quietly sums to
            less than one, which biases every belief update.

        Given: A state whose ranges sit at the clipping boundary
        When: Every reachable observation is enumerated and scored
        Then: The probabilities sum to one

        Test type: unit
        """
        state = build_state(env, [(0, 0), (8, 6)], [(8, 6), (0, 0)], (7, 1))
        action = encode_joint_action([ACTION_SCAN, ACTION_NORTH])
        ranges, _ = env._true_distances(state)  # pylint: disable=protected-access
        tables = [
            sorted(env._range_probabilities(ranges[i][j]))  # pylint: disable=protected-access
            for i in range(env.n_blue)
            for j in range(env.n_red)
        ]
        prefix = env._observed_prefix(state)  # pylint: disable=protected-access
        suffix = env._observed_suffix(state)  # pylint: disable=protected-access
        observations = [
            tuple(prefix) + tuple(float(value) for value in combo) + detector + tuple(suffix)
            for combo in itertools.product(*tables)
            for detector in itertools.product([0.0, 1.0], repeat=env.n_blue)
        ]
        mass = float(np.exp(env.observation_log_probability(state, action, observations)).sum())
        assert mass == pytest.approx(1.0, abs=1e-12)

    def test_scanning_sharpens_only_the_scanning_player(self, env: CaptureTheFlagPOMDP) -> None:
        """A scan widens that player's detector and nobody else's.

        Given: Two blue players equidistant from the flag, one scanning
        When: Their detection probabilities are compared
        Then: The scanner's is higher, and the other's matches the move value

        Test type: unit
        """
        distance = 4
        scanning = env._detection_probability(
            distance, ACTION_SCAN
        )  # pylint: disable=protected-access
        moving = env._detection_probability(
            distance, ACTION_NORTH
        )  # pylint: disable=protected-access
        assert scanning > moving
        assert moving == pytest.approx(
            0.5 * (1.0 + 2.0 ** (-distance / env.detector_half_distance_move))
        )

    def test_exactly_observed_components_act_as_a_delta(self, env: CaptureTheFlagPOMDP) -> None:
        """An observation disagreeing with the known parts is impossible.

        Purpose: Blue sees its own positions and score exactly. Scoring such a
            mismatch as merely unlikely would let a particle filter keep
            particles the observation has already ruled out.

        Given: A valid observation with one blue coordinate altered
        When: It is scored against the state that produced it
        Then: The likelihood is zero

        Test type: unit
        """
        state = build_state(env, [(3, 3), (5, 5)], [(7, 4), (7, 5)], (6, 3))
        action = encode_joint_action([ACTION_SCAN, ACTION_NORTH])
        np.random.seed(0)
        observation = list(env.sample_observation(state, action))
        assert np.isfinite(env.observation_log_probability(state, action, [observation])[0])
        observation[0] += 1.0
        assert env.observation_log_probability(state, action, [observation])[0] == -np.inf

    def test_reproduces_the_specified_three_state_likelihoods(
        self, env: CaptureTheFlagPOMDP
    ) -> None:
        """The worked example from the environment's specification.

        Purpose: Pins the observation model against numbers computed by hand,
            independently of this implementation. Every later change to the
            range noise, the detector decay or the factorisation has to explain
            itself here.

        Given: The three states of the specification's table, sharing blue
            positions, and the observation it scores them against
        When: Each is scored
        Then: The likelihoods match the published values

        Test type: unit
        """
        action = encode_joint_action([ACTION_SCAN, ACTION_NORTH])
        observation = tuple([4.0, 3.0, 5.0, 5.0] + [4.0, 5.0, 3.0, 2.0] + [1.0, 0.0] + [0.0] * 6)
        hypotheses = [
            (((7, 4), (7, 5)), (6, 3), 0.13111),
            (((7, 3), (7, 5)), (7, 1), 0.00213),
            (((5, 3), (7, 5)), (6, 3), 0.0),
        ]
        for red, flag, expected in hypotheses:
            state = build_state(env, [(4, 3), (5, 5)], list(red), flag)
            likelihood = float(
                np.exp(env.observation_log_probability(state, action, [observation])[0])
            )
            assert likelihood == pytest.approx(expected, abs=5e-6)


class TestRewardAndMetrics:
    """The reward bound and the reported channels."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"trees": [], "freeze_steps": 1, "tagger_cooldown_steps": 0},
            # Negative bonuses and costs so large the terminal zero is the
            # extreme: both broke a bound that assumed the signs.
            {"pickup_reward": -200.0, "n_blue": 1, "n_red": 1, "n_red_defenders": 1},
            {"move_cost": 100.0, "scan_cost": 100.0},
            # A candidate on the blue base lets pick-up and scoring land on one
            # step, which is the configuration the declared maximum assumes.
            {
                "red_flag_candidates": [(5, 3)],
                "midline": 4,
                "score_to_win": 5,
            },
            {"n_blue": 1, "n_red": 3, "n_red_defenders": 3, "scan_cost": 0.5},
        ],
    )
    def test_realised_rewards_stay_inside_the_declared_range(
        self, kwargs: Dict[str, object]
    ) -> None:
        """No rollout produces a reward outside the declared bound.

        Purpose: A wrong reward range is the most-repeated bug in this
            repository, and it is configuration-dependent -- team size, the
            action costs and the candidate placement all move the bound.

        Given: Several configurations, including ones that stack penalties
        When: Long random rollouts are scored
        Then: Every reward lies inside ``reward_range``

        Test type: integration
        """
        env = CaptureTheFlagPOMDP(**kwargs)  # type: ignore[arg-type]
        assert env.reward_range is not None
        low, high = env.reward_range
        np.random.seed(11)
        state = env.initial_state_dist().sample()[0]
        for _ in range(500):
            action = int(np.random.randint(0, len(env.get_actions())))
            successor = env.sample_next_state(state, action)
            reward = env.reward(state, action, successor)
            assert low <= reward <= high
            state = successor
            if env.is_terminal(state):
                state = env.initial_state_dist().sample()[0]

    def test_scoring_and_conceding_hit_the_bound_terms(self) -> None:
        """The extreme reward terms are reachable, not hypothetical.

        Purpose: A bound nothing can reach is as much a bug as one that is too
            tight -- it tells a planner the wrong scale.

        Given: A state one step from a blue capture, and one from a concession
        When: Each is scored
        Then: The rewards carry the capture bonus and the concession penalty

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(
            slip_probability=0.0, red_pursuit_probability=1.0, n_red=1, n_red_defenders=1
        )
        action = encode_joint_action([ACTION_STAY, ACTION_STAY])
        capture = build_state(env, [(0, 3), (0, 3)], [(6, 3)], (6, 3), carrier_red_flag=1)
        assert env.reward(capture, action, env.sample_next_state(capture, action)) == pytest.approx(
            env.capture_reward - 2 * env.move_cost
        )
        # A carrier heads home whatever its role, so the defender sitting on
        # the red base scores on the spot.
        concede = build_state(env, [(0, 3), (0, 3)], [(8, 3)], (6, 3), carrier_blue_flag=1)
        assert env.reward(concede, action, env.sample_next_state(concede, action)) == pytest.approx(
            -env.concede_penalty - 2 * env.move_cost
        )

    def test_step_info_tolerates_the_terminal_bookkeeping_step(
        self, env: CaptureTheFlagPOMDP
    ) -> None:
        """The terminal step reports state channels and neutral transition ones.

        Purpose: The episode-end channels are reduced with ``LAST``, so they
            are read from exactly this call. Returning nothing here would drop
            the three end-reason metrics for every terminated episode.

        Given: A terminal state, with no action and no successor
        When: ``step_info`` is called
        Then: The goal channel is set and the transition channels are zero

        Test type: unit
        """
        state = build_state(
            env, [(0, 3), (0, 3)], [(8, 3), (8, 3)], (6, 3), score_blue=env.score_to_win
        )
        info = env.step_info(state, None, None)
        assert info[CaptureTheFlagStepChannel.ENDED_BY_GOAL.value] == 1.0
        assert info[CaptureTheFlagStepChannel.ENDED_BY_TIMEOUT.value] == 0.0
        assert info[CaptureTheFlagStepChannel.TAGS_SUFFERED.value] == 0.0
        assert info[CaptureTheFlagStepChannel.RECORDED_STEP.value] == 1.0

    def test_every_declared_channel_is_emitted(self, env: CaptureTheFlagPOMDP) -> None:
        """Each metric spec reads a channel ``step_info`` actually reports.

        Purpose: A declared channel that is never emitted yields a metric the
            aggregator silently drops -- no error, no number.

        Given: The declared specs and one ordinary transition
        When: The emitted channel names are compared with the declared ones
        Then: Every declared channel is present, and the metric names match the
            enum

        Test type: unit
        """
        state = opening_states(env)[0]
        np.random.seed(0)
        emitted = set(env.step_info(state, 0, env.sample_next_state(state, 0)))
        declared = {spec.channel for spec in env.get_metric_specs()}
        assert declared <= emitted
        assert [spec.name for spec in env.get_metric_specs()] == [
            metric.value for metric in CaptureTheFlagMetrics
        ]

    def test_step_info_consumes_no_randomness(self, env: CaptureTheFlagPOMDP) -> None:
        """Reporting a step does not disturb the random stream.

        Purpose: A single draw here would shift every later transition and
            observation, changing seeded trajectories throughout a run rather
            than only the metrics.

        Given: One transition
        When: ``step_info`` is called between two identically seeded draws
        Then: The draws agree

        Test type: unit
        """
        state = opening_states(env)[0]
        np.random.seed(5)
        expected = np.random.random(4)
        np.random.seed(5)
        env.step_info(state, 0, state)
        assert np.array_equal(np.random.random(4), expected)


class TestConfiguration:
    """Construction-time validation and identity."""

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"blue_base": (3, 3)}, "stands on a tree"),
            ({"red_base": (1, 3)}, "red base must lie in the red half"),
            ({"red_flag_candidates": [(6, 1)]}, "not a free cell"),
            ({"n_red_defenders": 5}, "n_red_defenders must be in"),
            ({"midline": 0}, "midline must leave a column"),
            ({"freeze_steps": 0}, "freeze_steps must be at least 1"),
            ({"slip_probability": 1.5}, "slip_probability must be in"),
            ({"red_flag_candidates": [(6, 3), (6, 3)]}, "must be distinct"),
        ],
    )
    def test_inconsistent_configurations_are_rejected(
        self, kwargs: Dict[str, object], message: str
    ) -> None:
        """Undefined field layouts raise rather than running.

        Purpose: Each of these would otherwise produce an environment that
            runs and measures something other than capture-the-flag.

        Given: A configuration violating one field invariant
        When: The environment is constructed
        Then: A ``ValueError`` naming the violation is raised

        Test type: unit
        """
        with pytest.raises(ValueError, match=message):
            CaptureTheFlagPOMDP(**kwargs)  # type: ignore[arg-type]

    def test_config_id_does_not_depend_on_tree_ordering(self) -> None:
        """Two environments over the same cells share one identity.

        Purpose: ``config_id`` is a cache key. Serializing an unordered
            collection through ``str()`` would make the id depend on iteration
            order, so the same environment built two ways would miss its own
            cached results.

        Given: The reference trees, and the same cells shuffled
        When: Both environments are built
        Then: They are equal and share a ``config_id``

        Test type: unit
        """
        reference = CaptureTheFlagPOMDP()
        shuffled = CaptureTheFlagPOMDP(
            trees=[(6, 5), (2, 1), (4, 6), (3, 3), (2, 5), (6, 1), (4, 0)]
        )
        assert shuffled == reference
        assert shuffled.config_id == reference.config_id

    def test_an_empty_tree_collection_means_an_open_field(self) -> None:
        """Passing no trees is not the same as passing ``None``.

        Purpose: ``if not trees`` would take the no-trees branch for an
            explicitly empty collection *and* for ``None``, quietly giving the
            default field to someone who asked for an open one.

        Given: An environment built with an empty tree collection
        When: Its free cells are counted
        Then: Every cell of the grid is free

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(trees=[])
        assert len(env.free_cells()) == env.grid_size[0] * env.grid_size[1]
        assert CaptureTheFlagPOMDP(trees=None).trees

    def test_state_layout_covers_the_vector_exactly(self, env: CaptureTheFlagPOMDP) -> None:
        """The layout's fields tile the state vector without gaps or overlap.

        Given: The reference layout
        When: Its field offsets are listed
        Then: They are strictly increasing and end at the vector's length

        Test type: unit
        """
        layout = env.layout
        offsets = [
            layout.blue_pos,
            layout.red_pos,
            layout.flag_cell,
            layout.carrier_red_flag,
            layout.carrier_blue_flag,
            layout.freeze_blue,
            layout.freeze_red,
            layout.cooldown_blue,
            layout.cooldown_red,
            layout.score_blue,
            layout.score_red,
        ]
        assert offsets == sorted(offsets)
        assert layout.size == 4 * env.n_blue + 4 * env.n_red + 5
        assert layout.score_red + 1 == layout.size


class TestReviewRegressions:
    """One test per defect found reviewing this environment.

    Each of these was a silent wrong number rather than a crash, which is why
    they get named tests rather than a line in a docstring.
    """

    def test_end_channels_describe_the_state_the_step_landed_in(self) -> None:
        """A capture on the final budgeted step is a goal, not a timeout.

        Purpose: The episode runner checks its step budget before it checks
            termination, so an episode that scores on its last action never
            gets the terminal bookkeeping record. Reading the end channels off
            ``state`` would file that capture as a timeout -- and scoring late
            is exactly when this environment scores, so the bias runs one way.

        Given: A transition that scores
        When: ``step_info`` is called with both states
        Then: The goal channel is set and the timeout channel is not

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(
            slip_probability=0.0, red_pursuit_probability=1.0, n_red=1, n_red_defenders=1
        )
        state = build_state(env, [(0, 3), (0, 3)], [(6, 3)], (6, 3), carrier_red_flag=1)
        action = encode_joint_action([ACTION_STAY, ACTION_STAY])
        next_state = env.sample_next_state(state, action)
        assert env.is_terminal(next_state)
        info = env.step_info(state, action, next_state)
        assert info[CaptureTheFlagStepChannel.ENDED_BY_GOAL.value] == 1.0
        assert info[CaptureTheFlagStepChannel.ENDED_BY_TIMEOUT.value] == 0.0
        assert info[CaptureTheFlagStepChannel.CAPTURED.value] == 1.0

    def test_a_player_tagged_this_step_does_not_also_inflict_one(self) -> None:
        """Being sent home is not a way to tag someone standing on your base.

        Purpose: The tag stage teleports a tagged player to its own base and
            then keeps iterating the same list. Guarding the tagger on the
            *incoming* freeze let a player that had just been jailed tag an
            opponent from a cell it never walked through, earning the reward.

        Given: A blue player about to be tagged in the red half, and a red
            player standing on the blue base it will respawn onto
        When: The step resolves
        Then: Blue suffers its tag and inflicts none

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(
            slip_probability=0.0,
            red_pursuit_probability=1.0,
            n_blue=1,
            n_red=2,
            n_red_defenders=1,
        )
        # Red 0 defends and is parked on the intruder; red 1 attacks and is
        # standing on the blue base, which is where the tagged player lands.
        state = build_state(env, [(6, 3)], [(6, 3), (0, 3)], (6, 3))
        next_state = env.sample_next_state(state, encode_joint_action([ACTION_STAY]))
        counts = env._transition_counts(state, next_state)  # pylint: disable=protected-access
        assert counts["tags_suffered"] == 1
        assert counts["tags_inflicted"] == 0

    def test_off_support_observations_are_impossible(self, env: CaptureTheFlagPOMDP) -> None:
        """A fractional range or a non-binary detector bit scores zero.

        Purpose: The observation space is declared continuous, so a caller can
            pass a value the sampler could never emit. Rounding it onto the
            nearest supported value would hand an impossible observation the
            likelihood of a real one, and a particle filter would keep the
            particles it should have killed.

        Given: A valid observation, perturbed off the support two ways
        When: Each is scored
        Then: Both are impossible, while the unperturbed one is not

        Test type: unit
        """
        state = build_state(env, [(3, 3), (5, 5)], [(7, 4), (7, 5)], (6, 3))
        action = encode_joint_action([ACTION_SCAN, ACTION_NORTH])
        np.random.seed(0)
        observation = list(env.sample_observation(state, action))
        assert np.isfinite(env.observation_log_probability(state, action, [observation])[0])
        fractional = list(observation)
        fractional[2 * env.n_blue] += 0.2
        assert env.observation_log_probability(state, action, [fractional])[0] == -np.inf
        non_binary = list(observation)
        non_binary[2 * env.n_blue + env.n_blue * env.n_red] = 10.0
        assert env.observation_log_probability(state, action, [non_binary])[0] == -np.inf

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pickup_reward": -200.0},
            {"tagged_penalty": -50.0},
            {"move_cost": 100.0, "scan_cost": 100.0},
        ],
    )
    def test_reward_range_survives_unusual_coefficient_signs(
        self, kwargs: Dict[str, object]
    ) -> None:
        """The bound folds every term in by sign and admits the terminal zero.

        Purpose: Nothing rejects a negative bonus or a huge action cost. A
            bound that assumed each term's sign excluded ordinary rewards --
            including the ``0.0`` every terminal state scores.

        Given: Configurations with a negative bonus, a negative penalty, and
            costs large enough to dominate
        When: Ordinary and terminal rewards are scored
        Then: All of them lie inside the declared range

        Test type: unit
        """
        env = CaptureTheFlagPOMDP(**kwargs)  # type: ignore[arg-type]
        assert env.reward_range is not None
        low, high = env.reward_range
        assert low <= 0.0 <= high
        np.random.seed(2)
        state = env.initial_state_dist().sample()[0]
        for _ in range(120):
            action = int(np.random.randint(0, len(env.get_actions())))
            successor = env.sample_next_state(state, action)
            assert low <= env.reward(state, action, successor) <= high
            state = successor
            if env.is_terminal(state):
                assert low <= env.reward(state, action, state) <= high
                state = env.initial_state_dist().sample()[0]
