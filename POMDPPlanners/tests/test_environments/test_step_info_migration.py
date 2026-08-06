# SPDX-License-Identifier: MIT

"""Contract tests for environments that report metrics through ``step_info``.

The golden snapshot in :mod:`POMDPPlanners.tests.test_metric_golden_snapshot`
pins *what* each migrated environment computes. These tests pin the properties
that make that computation trustworthy, and which a frozen value comparison
cannot see:

- a declared spec whose channel is never emitted yields a metric the aggregator
  silently drops, so declared and produced names must be checked directly;
- ``step_info`` runs inside the episode loop, so it must be pure and must not
  consume randomness, or it would shift every later transition;
- it is called once per terminated episode with ``action`` and ``next_state``
  both ``None``, which must not raise and must not invent a measurement;
- its values ride back to the parent process inside a pickled ``History``.
"""

import dataclasses
import pickle
import random
from typing import Dict, List

import numpy as np
import pytest

from POMDPPlanners.core.environment import Environment
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_lab_pomdp import IsaacLabPOMDP
from POMDPPlanners.tests.test_utils.golden_metric_snapshot import (
    append_terminal_step,
    attach_step_info,
    build_registry,
    load_snapshot_histories,
)

# The environments this suite covers: those that declare per-step metric specs.
_SPEC_DRIVEN_SLUGS = sorted(
    slug for slug, factory in build_registry().items() if factory().get_metric_specs()
)


@pytest.fixture(name="frozen_histories", scope="module")
def _frozen_histories() -> Dict[str, List[History]]:
    """Load the committed history fixture once for the whole module."""
    return load_snapshot_histories()


def _measured_steps(environment: Environment, histories: List[History]) -> List[Dict[str, float]]:
    return [
        step.info or {}
        for history in attach_step_info(environment, histories)
        for step in history.history
    ]


# PushPOMDP pairs its per-episode collision counts against the full history list
# while excluding zero-step episodes from the counts, so a stepless episode beside
# a real one divides by zero. That defect predates this migration and was carried
# over verbatim rather than quietly fixed, so these tests expect it.
_ZERO_STEP_DIVIDES_BY_ZERO = {"push"}

# Channels that describe the *transition* rather than the state, per environment.
# The terminal bookkeeping step took no action and produced no successor, so
# these must read as zero there or a SUM would count an event that never
# happened. Listed explicitly rather than derived: which channels are
# transition-derived is a property of the measurement, not something the spec or
# the mapping's shape reveals.
_TRANSITION_DERIVED_CHANNELS = {
    "tiger": ("correct_door_opened", "listened"),
    "laser_tag": ("obstacle_collision",),
    "rock_sample": ("sampled_rock",),
}


def _empty_history() -> History:
    """A history with no recorded steps, which the episode loop cannot produce."""
    return History(
        history=[],
        discount_factor=0.95,
        average_state_sampling_time=0.0,
        average_action_time=0.0,
        average_observation_time=0.0,
        average_belief_update_time=0.0,
        average_reward_time=0.0,
        actual_num_steps=0,
        reach_terminal_state=False,
        policy_run_data=[],
    )


class TestDeclaredChannelsAreReported:
    """Every declared spec must correspond to a channel the environment emits."""

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_every_declared_spec_channel_is_emitted(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that no spec declares a channel step_info never reports.

        Purpose: Validates the invariant the shared aggregator cannot enforce --
            it omits a metric whose channel no episode reported, so a typo in a
            channel name would silently delete a metric rather than fail

        Given: A migrated environment and the frozen histories
        When: step_info is replayed over every step and the emitted channel names
            are collected
        Then: Every channel named by a declared spec was emitted

        Test type: unit
        """
        environment = build_registry()[slug]()
        emitted = set()
        for step_info in _measured_steps(environment, frozen_histories[slug]):
            emitted.update(step_info)

        declared = {spec.channel for spec in environment.get_metric_specs()}
        assert declared <= emitted, (
            f"{slug}: specs declare channels that step_info never emits: "
            f"{sorted(declared - emitted)}. Such a metric is dropped silently."
        )

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_declared_names_equal_produced_names(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that compute_metrics produces exactly the declared metric names.

        Purpose: Validates the contract that tuning objectives and saved runs
            depend on -- a metric that is declared but not produced breaks a
            lookup by name

        Given: A migrated environment and the measured frozen histories
        When: compute_metrics and get_metric_names are compared
        Then: They list the same names in the same order

        Test type: unit
        """
        environment = build_registry()[slug]()
        measured = attach_step_info(environment, frozen_histories[slug])
        produced = [metric.name for metric in environment.compute_metrics(measured)]

        assert produced == list(environment.get_metric_names())

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_degenerate_history_never_invents_or_reorders_a_name(self, slug: str) -> None:
        """Test that a stepless episode yields declared names in declared order.

        Purpose: Validates the contract on the degenerate input. An episode with
            no recorded steps reports no channels, so each metric is either
            omitted by the aggregator or filled by the environment's declared
            name list -- but the result must never contain a name that was not
            declared, and must never reorder the ones it does contain

        Given: A migrated environment and one history containing no steps
        When: compute_metrics is called
        Then: The produced names are a subsequence of the declared names

        Test type: unit
        """
        environment = build_registry()[slug]()
        empty_history = attach_step_info(environment, [_empty_history()])
        produced = [metric.name for metric in environment.compute_metrics(empty_history)]
        declared = list(environment.get_metric_names())

        remaining = iter(declared)
        assert all(name in remaining for name in produced), (
            f"{slug}: a zero-step episode produced {produced}, which is not a "
            f"subsequence of the declared names {declared}"
        )


class TestDegenerateHistoryShapes:
    """Inputs the frozen fixture and a live episode cannot produce."""

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    @pytest.mark.parametrize("shape", ["no_histories", "one_empty", "two_empty", "empty_plus_real"])
    def test_degenerate_shapes_do_not_raise_or_invent_names(
        self, slug: str, shape: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that degenerate history shapes stay well-formed.

        Purpose: Validates the edge of the contract. Several pre-migration
            implementations raised ValueError from confidence_interval on an
            empty list here; the shared aggregator must instead return a
            well-formed result whose names are a subsequence of the declared
            ones, never an invented or reordered name

        Given: A migrated environment and a degenerate history list -- none, one
            or two episodes with no steps, or a stepless episode beside a real one
        When: compute_metrics is called
        Then: It returns without raising, and every produced name is declared,
            in declared order

        Test type: unit
        """
        environment = build_registry()[slug]()
        shapes = {
            "no_histories": [],
            "one_empty": [_empty_history()],
            "two_empty": [_empty_history(), _empty_history()],
            "empty_plus_real": [_empty_history(), frozen_histories[slug][0]],
        }
        measured = attach_step_info(environment, list(shapes[shape]))

        if slug in _ZERO_STEP_DIVIDES_BY_ZERO and shape == "empty_plus_real":
            with pytest.raises(ZeroDivisionError):
                environment.compute_metrics(measured)
            return

        produced = [metric.name for metric in environment.compute_metrics(measured)]
        declared = list(environment.get_metric_names())

        remaining = iter(declared)
        assert all(
            name in remaining for name in produced
        ), f"{slug}/{shape}: produced {produced}, not a subsequence of {declared}"

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_a_stepless_episode_does_not_shift_a_real_one(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that a stepless episode leaves a real episode's metrics alone.

        Purpose: Validates that an episode reporting no channel at all is
            dropped from a spec's average rather than contributing a stand-in
            value. What an unmeasured episode would have measured is not knowable
            from the metric's declaration, so inventing a number for it would
            move the mean on the strength of a guess

        Given: One real episode, and the same episode paired with a stepless one
        When: compute_metrics is run on both lists
        Then: Every spec-derived metric reports the same value in both

        Test type: unit
        """
        if slug in _ZERO_STEP_DIVIDES_BY_ZERO:
            pytest.skip(f"{slug} raises on a stepless episode, preserved from before the migration")
        environment = build_registry()[slug]()
        real = frozen_histories[slug][:1]
        # Only the spec-derived names. An environment keeping bespoke metrics
        # divides them by the episode count itself, so a stepless episode does
        # move those -- that is their pre-existing behaviour, not this one's.
        spec_driven = {spec.name for spec in environment.get_metric_specs()}

        alone = {
            m.name: m.value
            for m in environment.compute_metrics(attach_step_info(environment, list(real)))
        }
        with_empty = {
            m.name: m.value
            for m in environment.compute_metrics(
                attach_step_info(environment, [_empty_history()] + list(real))
            )
        }

        compared = spec_driven & set(alone)
        assert compared, f"{slug}: no spec-derived metric was produced, nothing would be asserted"

        for name in compared:
            assert with_empty[name] == pytest.approx(alone[name]), (
                f"{slug}.{name}: a stepless episode must not shift the mean, but it "
                f"moved from {alone[name]} to {with_empty[name]}"
            )


class TestStepInfoContract:
    """Purity, terminal-step tolerance and transportability of step_info."""

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_step_info_is_pure(self, slug: str, frozen_histories: Dict[str, List[History]]) -> None:
        """Test that repeated calls with the same transition agree.

        Purpose: Validates that step_info is a function of its arguments. An
            implementation that consumed randomness or mutated state would both
            report unstable metrics and shift the seeded transition stream for
            every later step of the episode

        Given: A migrated environment and the frozen histories
        When: step_info is called twice for each recorded transition
        Then: The two mappings are equal

        Test type: unit
        """
        environment = build_registry()[slug]()
        for history in frozen_histories[slug]:
            for step in history.history:
                first = environment.step_info(step.state, step.action, step.next_state)
                second = environment.step_info(step.state, step.action, step.next_state)
                assert first == second, f"{slug}: step_info is not a pure function"

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_step_info_consumes_no_randomness(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that measuring a step leaves the global RNG untouched.

        Purpose: Validates the invariant that makes the hook safe to call from
            inside the episode loop. A single np.random draw here would advance
            the stream and silently change every subsequent transition and
            observation, breaking seeded reproducibility far beyond metrics

        Given: A migrated environment and seeded numpy and stdlib RNGs
        When: step_info is called for every recorded transition
        Then: Both RNG states are byte-identical to before

        Test type: unit
        """
        environment = build_registry()[slug]()
        np.random.seed(4242)
        random.seed(4242)
        before = np.random.get_state(legacy=True)
        # Covered separately from numpy: an implementation reaching for
        # ``random.random()`` would leave the numpy stream untouched and still
        # break reproducibility for anything seeded through the stdlib.
        before_stdlib = random.getstate()

        for history in frozen_histories[slug]:
            for step in history.history:
                environment.step_info(step.state, step.action, step.next_state)

        after = np.random.get_state(legacy=True)
        assert random.getstate() == before_stdlib, f"{slug}: step_info advanced the stdlib RNG"
        assert isinstance(before, tuple) and isinstance(after, tuple)
        # The Mersenne Twister key is an ndarray, so the tuples cannot be
        # compared directly; the position counter is what a stray draw moves.
        assert np.array_equal(before[1], after[1]), f"{slug}: step_info advanced the global RNG"
        assert before[2:] == after[2:], f"{slug}: step_info advanced the global RNG"

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_terminal_step_reports_no_transition_channel(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that the terminal bookkeeping call is tolerated and neutral.

        Purpose: Validates the terminal-step contract. That step records the
            final state and has no action or successor, so a transition-derived
            channel must report a neutral value there rather than raising or
            inventing a measurement

        Given: A migrated environment and the final state of each frozen episode
        When: step_info is called the way _add_terminal_step calls it
        Then: Every channel reported for a real transition is reported here too,
            so no metric loses the final state, and every transition-derived
            channel reports its neutral value, so none gains a phantom event

        Test type: unit
        """
        environment = build_registry()[slug]()
        transition_channels = set()
        for step_info in _measured_steps(environment, frozen_histories[slug]):
            transition_channels.update(step_info)

        for history in frozen_histories[slug]:
            final_state = history.history[-1].next_state
            terminal_info = environment.step_info(final_state, None, None)

            missing = transition_channels - set(terminal_info)
            assert not missing, (
                f"{slug}: the terminal step drops channels {sorted(missing)}. A SUM "
                "or MEAN over them would then exclude the final state."
            )
            for channel in _TRANSITION_DERIVED_CHANNELS.get(slug, ()):
                assert terminal_info[channel] == 0.0, (
                    f"{slug}: the terminal step reports {channel}="
                    f"{terminal_info[channel]}, inventing an event for a step where "
                    "no action was taken and no transition happened."
                )

    def test_isaac_lab_step_info_tolerates_the_terminal_call(self) -> None:
        """Test that a live-simulator environment survives the terminal call.

        Purpose: Validates that the terminal-step call added to the episode loop
            cannot mis-attribute a cached measurement, for the one shipped
            environment that serves values from a live simulator rather than
            computing them from the arguments

        Given: An IsaacLabPOMDP instance that has taken no step
        When: step_info is called with action and next_state both None
        Then: An empty mapping is returned rather than an exception

        Test type: unit
        """
        world = IsaacLabPOMDP.__new__(IsaacLabPOMDP)
        world._pending = None  # pylint: disable=protected-access
        assert not IsaacLabPOMDP.step_info(world, np.zeros(2), None, None)

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_reported_values_survive_pickling(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that every reported value is a plain picklable scalar.

        Purpose: Validates the transport requirement. The measurements ride back
            to the parent process inside a pickled History under every
            multiprocess task manager, so a numpy scalar or an unpicklable object
            here would fail far from its cause

        Given: A migrated environment and the frozen histories
        When: Each reported mapping is round-tripped through pickle
        Then: It compares equal, and every value is a float

        Test type: unit
        """
        environment = build_registry()[slug]()
        for step_info in _measured_steps(environment, frozen_histories[slug]):
            assert pickle.loads(pickle.dumps(step_info)) == step_info
            for channel, value in step_info.items():
                assert isinstance(
                    value, float
                ), f"{slug}.{channel} reported {type(value).__name__}, not a plain float"


class TestTerminalStepIsCounted:
    """The terminal bookkeeping step must reach the metrics."""

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_terminal_step_contributes_measurements(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that the terminal step carries measurements of the final state.

        Purpose: Validates the reason the episode loop measures the terminal step
            at all. Metrics that count every visited state need the final one,
            and it is recorded on no other step

        Given: A migrated environment and the terminated-episode shape of the
            frozen histories
        When: step_info is replayed over them
        Then: The appended terminal step carries a non-empty mapping

        Test type: unit
        """
        environment = build_registry()[slug]()
        extended = append_terminal_step(frozen_histories[slug])
        for history in attach_step_info(environment, extended):
            terminal_step = history.history[-1]
            assert terminal_step.action is None
            assert terminal_step.info, (
                f"{slug}: the terminal step carries no measurements, so any metric "
                f"counting every visited state silently loses the final state"
            )


class TestUnmeasuredHistoriesAreRejected:
    """Histories carrying no measurements must fail loudly, not score as zero."""

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_unmeasured_histories_raise_instead_of_reporting_zeros(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that histories with no per-step info are refused.

        Purpose: Validates the guard on the most dangerous failure mode of
            deriving metrics from a per-step channel. A history recorded before
            the channel existed, or by a runner that never calls step_info,
            carries no measurements at all -- and an absent channel reads as a
            rate of 0.0, which is indistinguishable from a planner that always
            failed

        Given: A migrated environment and the frozen histories exactly as
            recorded, with no info attached
        When: compute_metrics is called on them
        Then: A ValueError naming the environment is raised, rather than a full
            set of zero-valued metrics

        Test type: unit
        """
        environment = build_registry()[slug]()

        with pytest.raises(ValueError, match="per-step measurement channel"):
            environment.compute_metrics(frozen_histories[slug])

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_stepless_episodes_are_not_mistaken_for_unmeasured_ones(self, slug: str) -> None:
        """Test that an episode with no recorded steps still scores.

        Purpose: Validates that the guard keys on "steps were recorded and none
            carries a measurement", not on "no measurement is present". An
            episode with no steps legitimately measured nothing, and every
            migrated environment has always scored it

        Given: A migrated environment and a history with no recorded steps
        When: compute_metrics is called on it
        Then: It returns without raising

        Test type: unit
        """
        environment = build_registry()[slug]()

        assert isinstance(environment.compute_metrics([_empty_history()]), list)

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_one_measured_episode_does_not_vouch_for_an_unmeasured_one(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that a measured episode beside an unmeasured one still raises.

        Purpose: Validates that the guard is per episode rather than over the
            batch. A partially warm cache produces exactly this mix, and an
            unmeasured episode that slipped through would not raise and not
            report -- it would silently drop out of every average, shifting each
            metric toward whichever episodes happened to be re-run

        Given: A migrated environment, one measured episode and one identical
            but unmeasured episode
        When: compute_metrics is called on both together
        Then: A ValueError is raised, naming the unmeasured episode's position

        Test type: unit
        """
        environment = build_registry()[slug]()
        measured = attach_step_info(environment, frozen_histories[slug])[0]
        unmeasured = frozen_histories[slug][0]

        with pytest.raises(ValueError, match="episode 1"):
            environment.compute_metrics([measured, unmeasured])

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_unrelated_info_does_not_vouch_for_a_declared_channel(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that bookkeeping in info does not satisfy the guard.

        Purpose: Validates that the guard keys on the declared channels rather
            than on info being non-empty. A future runner stamping unrelated
            per-step bookkeeping into the same mapping would otherwise vouch for
            measurements that were never taken

        Given: A migrated environment and histories whose steps carry an info
            mapping containing only a channel no spec declares
        When: compute_metrics is called on them
        Then: A ValueError is still raised

        Test type: unit
        """
        environment = build_registry()[slug]()
        decoys = [
            dataclasses.replace(
                history,
                history=[
                    step._replace(info={"unrelated_bookkeeping": 1.0}) for step in history.history
                ],
            )
            for history in frozen_histories[slug]
        ]

        with pytest.raises(ValueError, match="per-step measurement channel"):
            environment.compute_metrics(decoys)

    @pytest.mark.parametrize("slug", _SPEC_DRIVEN_SLUGS)
    def test_terminal_only_episode_is_exempt(
        self, slug: str, frozen_histories: Dict[str, List[History]]
    ) -> None:
        """Test that an episode of only a terminal step is not rejected.

        Purpose: Validates the exemption the guard needs for an environment
            serving values cached during its own step. A live simulator has
            nothing to report for the terminal bookkeeping call, so an episode
            consisting solely of that step legitimately carries no measurement
            and must not be mistaken for an uninstrumented one

        Given: A migrated environment and a history whose only step is a
            terminal bookkeeping step carrying no info
        When: compute_metrics is called on it
        Then: It returns without raising

        Test type: unit
        """
        environment = build_registry()[slug]()
        final_state = frozen_histories[slug][0].history[-1].next_state
        terminal_only = dataclasses.replace(
            _empty_history(),
            history=[
                StepData(
                    state=final_state,
                    action=None,
                    next_state=None,
                    observation=None,
                    reward=None,
                    belief=frozen_histories[slug][0].history[-1].belief,
                    info=None,
                )
            ],
        )

        assert isinstance(environment.compute_metrics([terminal_only]), list)
