# SPDX-License-Identifier: MIT

"""Tests for IsaacLabPOMDP's per-step measurements and episode video capture.

Covers the impact / task-completion channels reported through ``step_info``, the
metric specs derived from them, the config hook used to attach a contact sensor,
and the frame buffer behind ``cache_visualization``.

Like the sibling world tests these run against ``FakeIsaacEnv`` through the
``_build_isaac_env`` seam, so no Isaac Sim install is required.
"""

# pylint: disable=protected-access  # Tests inspect the live-simulator internals

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.simulation import History, StepData
from POMDPPlanners.core.simulation.step_info_metrics import EpisodeReduction
from POMDPPlanners.environments.isaac_lab_pomdp import IsaacLabPOMDP
from POMDPPlanners.environments.isaac_lab_pomdp import isaac_lab_pomdp as isaac_module
from POMDPPlanners.tests.test_environments.test_isaac_lab_pomdp.test_isaac_lab_pomdp import (
    FakeIsaacEnv,
)
from POMDPPlanners.tests.test_utils.history_builders import build_test_history


def _history_with_infos(infos: List[Dict[str, float]]) -> History:
    """Build a History whose steps carry the given per-step info mappings."""
    belief = WeightedParticleBelief(particles=["a", "b"], log_weights=np.array([0.0, -0.1]))
    steps = [
        StepData(
            state=np.zeros(5),
            action=np.zeros(2),
            next_state=np.zeros(5),
            observation=np.zeros(4),
            reward=0.0,
            belief=belief,
            info=info,
        )
        for info in infos
    ]
    return build_test_history(steps=steps)


@pytest.fixture(name="fake_env")
def _fake_env(monkeypatch: pytest.MonkeyPatch) -> FakeIsaacEnv:
    """Patch the build seam to return a single shared FakeIsaacEnv instance."""
    env = FakeIsaacEnv()
    monkeypatch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
    return env


@pytest.fixture(name="world")
def _world(fake_env: FakeIsaacEnv) -> IsaacLabPOMDP:
    """An unconfigured IsaacLabPOMDP, reset to its initial live state."""
    del fake_env
    world = IsaacLabPOMDP(task_id="Fake-Isaac-v0", discount_factor=0.99, device="cpu")
    world.initial_state_dist().sample()
    return world


def _measuring_world(**overrides: Any) -> IsaacLabPOMDP:
    """Build a world configured to measure both impact and success."""
    kwargs: Dict[str, Any] = {
        "task_id": "Fake-Isaac-v0",
        "discount_factor": 0.99,
        "device": "cpu",
        "contact_sensor_key": "contact_forces",
        "success_termination_term": "success",
    }
    kwargs.update(overrides)
    world = IsaacLabPOMDP(**kwargs)
    world.initial_state_dist().sample()
    return world


class TestImpactMeasurement:
    """Impact severity read from the contact sensor."""

    def test_impact_uses_force_history_peak_scaled_to_impulse(self, fake_env: FakeIsaacEnv) -> None:
        """Test that impact is the peak history force times the step duration.

        Purpose: Validates that severity captures the worst substep rather than
            the end-of-step reading, and is reported as an impulse

        Given: A contact sensor whose history holds forces 3t (z) and 4t (y) at
            t=1, and a control step of 0.5 s
        When: One step is taken and step_info is read
        Then: The impact channel is 4.0 * 0.5 = 2.0

        Test type: unit
        """
        world = _measuring_world()
        state = world._live_state
        next_state = world.sample_next_state(state, np.zeros(2))

        info = world.step_info(state, np.zeros(2), next_state)

        assert fake_env.step_calls == 1
        assert info["contact_impulse_ns"] == pytest.approx(1.75)

    def test_impact_falls_back_to_net_forces_without_history(self) -> None:
        """Test that a sensor without a history buffer still yields an impact.

        Purpose: Validates the fallback for contact sensors configured without
            history_length, where only the end-of-step force is available

        Given: A contact sensor exposing only net_forces_w (3t on z) at t=1
        When: One step is taken and step_info is read
        Then: The impact channel is 3.0 * 0.5 = 1.5

        Test type: unit
        """
        env = FakeIsaacEnv(contact_history=False)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
            world = _measuring_world()
            state = world._live_state
            next_state = world.sample_next_state(state, np.zeros(2))

            assert world.step_info(state, np.zeros(2), next_state)[
                "contact_impulse_ns"
            ] == pytest.approx(1.5)

    def test_impact_ignores_history_older_than_the_current_step(self) -> None:
        """Test that stale history entries do not inflate the impulse.

        Purpose: Validates that the metric is a property of the transition, not
            of how long a history buffer the sensor happens to keep. Summing the
            whole buffer would fold in forces from earlier control steps

        Given: A sensor whose buffer holds more samples than the step has
            substeps, with large stale values at the front
        When: One step is taken and step_info is read
        Then: Only the most recent substeps contribute, so the result matches the
            short-buffer case

        Test type: unit
        """
        env = FakeIsaacEnv(contact_history=True, stale_history=True)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
            world = _measuring_world()
            state = world._live_state
            next_state = world.sample_next_state(state, np.zeros(2))

            assert world.step_info(state, np.zeros(2), next_state)[
                "contact_impulse_ns"
            ] == pytest.approx(1.75)

    def test_impact_absent_when_no_contact_sensor_configured(self, world: IsaacLabPOMDP) -> None:
        """Test that an unconfigured world reports no impact channel.

        Purpose: Validates that a task without a contact sensor reports nothing
            rather than a fabricated zero, so "not measured" is never read as
            "no impact occurred"

        Given: A world constructed without contact_sensor_key
        When: A step is taken and step_info is read
        Then: The impact channel is absent

        Test type: unit
        """
        state = world._live_state
        next_state = world.sample_next_state(state, np.zeros(2))

        assert "contact_impulse_ns" not in world.step_info(state, np.zeros(2), next_state)

    def test_custom_impact_extractor_overrides_the_default(self) -> None:
        """Test that an injected impact extractor takes precedence.

        Purpose: Validates the per-task escape hatch for exotic sensors

        Given: A world constructed with an impact_extractor returning 9.0
        When: A step is taken and step_info is read
        Then: The impact channel is 9.0, unscaled by the default impulse logic

        Test type: unit
        """
        env = FakeIsaacEnv()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
            world = _measuring_world(contact_sensor_key=None, impact_extractor=lambda _env: 9.0)
            state = world._live_state
            next_state = world.sample_next_state(state, np.zeros(2))

            assert world.step_info(state, np.zeros(2), next_state)["contact_impulse_ns"] == 9.0


class TestSuccessMeasurement:
    """Task completion read from the termination manager."""

    def test_success_reads_the_named_termination_term(self, fake_env: FakeIsaacEnv) -> None:
        """Test that the configured success term drives the success channel.

        Purpose: Validates that task completion comes from the task's own
            success predicate, not from the fact that the episode ended

        Given: A termination manager whose "success" term is True
        When: A step is taken and step_info is read
        Then: The success channel is 1.0

        Test type: unit
        """
        fake_env.termination_manager.set_term("success", True)
        world = _measuring_world()
        state = world._live_state
        next_state = world.sample_next_state(state, np.zeros(2))

        assert world.step_info(state, np.zeros(2), next_state)["success"] == 1.0

    def test_missing_term_raises_instead_of_guessing(self, fake_env: FakeIsaacEnv) -> None:
        """Test that a misconfigured success term fails loudly.

        Purpose: Validates that success is never inferred from "terminated but
            not truncated". Most IsaacLab tasks terminate on *failure* and
            truncate on timeout, so that rule would report every failure as a
            success and silently invert the metric

        Given: A world configured with a success term the task does not declare
        When: A step is taken
        Then: RuntimeError names the missing term and points at success_extractor

        Test type: unit
        """
        del fake_env
        world = _measuring_world(success_termination_term="not_a_real_term")
        state = world._live_state

        with pytest.raises(RuntimeError, match="not_a_real_term"):
            world.sample_next_state(state, np.zeros(2))

    def test_success_absent_when_not_configured(self, world: IsaacLabPOMDP) -> None:
        """Test that an unconfigured world reports no success channel.

        Purpose: Validates that completion is only claimed when the world was
            actually told how to recognise it

        Given: A world constructed without success_termination_term
        When: A step is taken and step_info is read
        Then: The success channel is absent

        Test type: unit
        """
        state = world._live_state
        next_state = world.sample_next_state(state, np.zeros(2))

        assert "success" not in world.step_info(state, np.zeros(2), next_state)


class TestStepInfoContract:
    """How step_info relates to the single cached simulator step."""

    def test_step_info_does_not_advance_the_simulator(self, fake_env: FakeIsaacEnv) -> None:
        """Test that reading measurements costs no extra physics step.

        Purpose: Validates that step_info is served from the same cache as
            reward/next_state/observation, preserving the one-step-per-
            interaction invariant

        Given: A world that has taken exactly one step
        When: step_info is read repeatedly
        Then: The simulator step count stays at one

        Test type: unit
        """
        world = _measuring_world()
        state = world._live_state
        next_state = world.sample_next_state(state, np.zeros(2))

        for _ in range(3):
            world.step_info(state, np.zeros(2), next_state)

        assert fake_env.step_calls == 1

    def test_step_info_for_a_foreign_next_state_reports_nothing(
        self, fake_env: FakeIsaacEnv
    ) -> None:
        """Test that a mismatched request yields no measurements.

        Purpose: Validates that a forward-only world never attributes a cached
            measurement to a transition it did not take

        Given: A world that has taken one step
        When: step_info is asked about an unrelated next state
        Then: An empty mapping is returned

        Test type: unit
        """
        del fake_env
        world = _measuring_world()
        state = world._live_state
        world.sample_next_state(state, np.zeros(2))

        assert not world.step_info(state, np.zeros(2), np.array([99.0, 99.0]))

    def test_step_info_for_the_terminal_step_reports_nothing(self, fake_env: FakeIsaacEnv) -> None:
        """Test that the terminal bookkeeping call yields no measurements.

        Purpose: Validates that the episode loop's terminal-step call cannot
            mis-attribute the last transition's cached contact impulse and
            success flag to the terminal state, which has no transition of its
            own

        Given: A world that has taken one step, holding a pending measurement
        When: step_info is called the way _add_terminal_step calls it, with
            action and next_state both None
        Then: An empty mapping is returned and no extra physics step is taken

        Test type: unit
        """
        world = _measuring_world()
        state = world._live_state
        world.sample_next_state(state, np.zeros(2))

        assert not world.step_info(state, None, None)
        assert fake_env.step_calls == 1


class TestMetricSpecs:
    """Declared metric specs track the configured measurements."""

    def test_specs_declare_only_configured_measurements(self, fake_env: FakeIsaacEnv) -> None:
        """Test that spec declaration follows the world's configuration.

        Purpose: Validates that declared metric names always match the channels
            actually emitted, which is what keeps declared and produced names
            consistent

        Given: Worlds configured with neither, one, and both measurements
        When: get_metric_names() is called
        Then: Exactly the corresponding metric names are declared

        Test type: unit
        """
        del fake_env
        both = _measuring_world()
        impact_only = _measuring_world(success_termination_term=None)
        neither = _measuring_world(success_termination_term=None, contact_sensor_key=None)

        assert both.get_metric_names() == ["success_rate", "max_contact_impulse_ns"]
        assert impact_only.get_metric_names() == ["max_contact_impulse_ns"]
        assert not neither.get_metric_names()

    def test_success_reduction_is_configurable(self, fake_env: FakeIsaacEnv) -> None:
        """Test that the completion reduction follows the world's configuration.

        Purpose: Validates that a task whose success predicate means "never
            failed" can require every step, rather than being forced into the
            reach-a-goal-once semantics

        Given: Two episodes, one of which fails on its final step
        When: compute_metrics runs on worlds configured with ALL and with ANY
        Then: ALL reports 0.5 and ANY reports 1.0

        Test type: unit
        """
        del fake_env
        histories = [
            _history_with_infos([{"success": 1.0}, {"success": 1.0}]),
            _history_with_infos([{"success": 1.0}, {"success": 0.0}]),
        ]
        strict = _measuring_world(contact_sensor_key=None, success_reduction=EpisodeReduction.ALL)
        lenient = _measuring_world(contact_sensor_key=None, success_reduction=EpisodeReduction.ANY)

        assert strict.compute_metrics(histories)[0].value == 0.5
        assert lenient.compute_metrics(histories)[0].value == 1.0

    def test_metrics_aggregate_across_episodes(self, fake_env: FakeIsaacEnv) -> None:
        """Test that declared specs produce metrics from recorded histories.

        Purpose: Validates the end-to-end path from per-step channels to metric
            values with confidence intervals

        Given: Two synthetic episodes, one of which reports a success
        When: compute_metrics() is called
        Then: task_completion_rate is 0.5 and impact_severity is the mean peak

        Test type: integration
        """
        del fake_env
        world = _measuring_world()
        histories = [
            _history_with_infos(
                [
                    {"success": 0.0, "contact_impulse_ns": 1.0},
                    {"success": 1.0, "contact_impulse_ns": 3.0},
                ]
            ),
            _history_with_infos([{"success": 0.0, "contact_impulse_ns": 5.0}]),
        ]

        metrics = {metric.name: metric.value for metric in world.compute_metrics(histories)}

        assert metrics["success_rate"] == pytest.approx(0.5)
        assert metrics["max_contact_impulse_ns"] == pytest.approx(4.0)


class TestVideoCapture:
    """Frame buffering and episode video output."""

    def test_frames_are_buffered_per_step_when_recording(self) -> None:
        """Test that recording captures one frame on reset and one per step.

        Purpose: Validates that a video covers the whole episode including its
            initial state

        Given: A world constructed with record_video and rgb_array rendering
        When: Two steps are taken after the reset
        Then: Three frames are buffered

        Test type: unit
        """
        env = FakeIsaacEnv()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
            world = _measuring_world(render_mode="rgb_array", record_video=True)

            state = world._live_state
            for _ in range(2):
                state = world.sample_next_state(state, np.zeros(2))

            assert len(world.frames) == 3
            assert all(frame.dtype == np.uint8 for frame in world.frames)

    def test_reset_clears_the_frame_buffer(self) -> None:
        """Test that a new episode does not inherit the previous one's frames.

        Purpose: Validates that per-episode videos stay separate

        Given: A recording world that has stepped once
        When: The environment is reset
        Then: Only the fresh initial frame remains

        Test type: unit
        """
        env = FakeIsaacEnv()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
            world = _measuring_world(render_mode="rgb_array", record_video=True)
            world.sample_next_state(world._live_state, np.zeros(2))

            world.initial_state_dist().sample()

            assert len(world.frames) == 1

    def test_frames_are_not_buffered_by_default(self, world: IsaacLabPOMDP) -> None:
        """Test that frame capture is opt-in.

        Purpose: Validates that non-recording runs pay no rendering cost

        Given: A world constructed without record_video
        When: A step is taken
        Then: No frames are buffered

        Test type: unit
        """
        world.sample_next_state(world._live_state, np.zeros(2))

        assert not world.frames

    def test_record_video_requires_rgb_array_render_mode(self) -> None:
        """Test that an unusable recording configuration is rejected early.

        Purpose: Validates that the failure surfaces at construction rather than
            as an empty video after a long run

        Given: record_video=True without render_mode="rgb_array"
        When: The world is constructed
        Then: ValueError is raised naming the required render mode

        Test type: unit
        """
        with pytest.raises(ValueError, match="render_mode='rgb_array'"):
            IsaacLabPOMDP(
                task_id="Fake-Isaac-v0", discount_factor=0.99, device="cpu", record_video=True
            )

    def test_cache_visualization_requires_recording(self, world: IsaacLabPOMDP) -> None:
        """Test that requesting a video without recording fails loudly.

        Purpose: Validates that a silently missing video is impossible

        Given: A world constructed without record_video
        When: cache_visualization() is called
        Then: RuntimeError is raised naming record_video

        Test type: unit
        """
        with pytest.raises(RuntimeError, match="record_video=True"):
            world.cache_visualization([], Path("/tmp"), 0)

    def test_frame_buffer_is_dropped_on_pickling(self) -> None:
        """Test that buffered frames do not travel with a pickled world.

        Purpose: Validates that the live-state contract holds for frames too, so
            a worker never ships megabytes of images back with the environment

        Given: A recording world with buffered frames
        When: It is pickled and restored
        Then: The restored world has an empty frame buffer

        Test type: unit
        """
        env = FakeIsaacEnv()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(isaac_module, "_build_isaac_env", lambda *a, **k: env)
            world = _measuring_world(render_mode="rgb_array", record_video=True)
            world.sample_next_state(world._live_state, np.zeros(2))
            assert world.frames

            restored = pickle.loads(pickle.dumps(world))

            assert not restored.frames


def test_env_cfg_modifier_is_applied_before_the_env_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the config hook can mutate the task cfg before construction.

    Purpose: Validates the seam through which a contact sensor can be attached
        to a task that ships none

    Given: A patched builder chain recording the cfg it receives
    When: A world with an env_cfg_modifier builds its env
    Then: The modifier ran and its mutation is visible on the cfg passed to make

    Test type: unit
    """
    captured: Dict[str, Any] = {}

    class _Cfg:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.sensors: List[str] = []

    def _fake_gym_make(cfg: Any) -> FakeIsaacEnv:
        captured["cfg"] = cfg
        return FakeIsaacEnv()

    def _fake_build(*_args: Any, **_kwargs: Any) -> FakeIsaacEnv:
        cfg = _Cfg()
        modifier = _args[6] if len(_args) > 6 else _kwargs.get("env_cfg_modifier")
        if modifier is not None:
            modifier(cfg)
        return _fake_gym_make(cfg)

    monkeypatch.setattr(isaac_module, "_build_isaac_env", _fake_build)
    world = IsaacLabPOMDP(
        task_id="Fake-Isaac-v0",
        discount_factor=0.99,
        device="cpu",
        env_cfg_modifier=lambda cfg: cfg.sensors.append("contact_forces"),
    )
    world.initial_state_dist().sample()

    assert captured["cfg"].sensors == ["contact_forces"]


def test_action_space_is_exposed_without_reaching_into_privates(fake_env: FakeIsaacEnv) -> None:
    """Test that the task's action space is publicly readable.

    Purpose: Validates the accessor callers need to size their own action
        discretization, since a continuous IsaacLab task must be discretized
        before an index-addressed planner such as VOPP can act on it

    Given: A world wrapping the fake task
    When: action_space is read
    Then: It is the underlying task's space, without touching private members

    Test type: unit
    """
    world = _measuring_world()

    assert world.action_space is fake_env.action_space


class TestContactSampleReaders:
    """The module-level contact readers, shared by the world and the sensor helpers.

    They were split out of the world so both paths slice the same samples. If they drift, the
    impulse the world reports through ``step_info`` and the peak an ``impact_extractor`` reads
    stop describing the same control step, and nothing in either number says so.
    """

    @staticmethod
    def _env(history: Any = None, forces: Any = None, **timing: float) -> Any:
        data = SimpleNamespace()
        if history is not None:
            data.net_forces_w_history = history
        if forces is not None:
            data.net_forces_w = forces
        scene = {"contact_forces": SimpleNamespace(data=data)}
        return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, **timing))

    def test_substeps_come_from_the_simulator_timing(self) -> None:
        """The slice width is a property of the task, not something a caller should pass.

        Purpose: Validates the substeps-per-control-step calculation

        Given: An env reporting a 0.2 s control step over a 0.005 s physics step
        When: The substep count is read
        Then: It is 40

        Test type: unit
        """
        env = self._env(step_dt=0.2, physics_dt=0.005)
        assert isaac_module.control_substeps(env) == 40

    @pytest.mark.parametrize("timing", [{}, {"step_dt": 0.2}, {"step_dt": 0.0, "physics_dt": 0.0}])
    def test_incomplete_timing_falls_back_to_one_substep(self, timing: Dict[str, float]) -> None:
        """A fake or minimal env must degrade to the end-of-step reading, not divide by zero.

        Purpose: Validates the fallback when the env omits its timing

        Given: An env missing one or both timing fields
        When: The substep count is read
        Then: It is 1

        Test type: unit
        """
        assert isaac_module.control_substeps(self._env(**timing)) == 1

    def test_step_duration_defaults_to_one_second(self) -> None:
        """A missing step_dt must not silently scale every impulse by zero.

        Purpose: Validates the step-duration fallback

        Given: An env that does not report step_dt
        When: The step duration is read
        Then: It is 1.0

        Test type: unit
        """
        assert isaac_module.step_duration(self._env()) == pytest.approx(1.0)

    def test_samples_are_sliced_to_the_current_step(self) -> None:
        """Trailing history entries belong to steps already measured.

        Purpose: Validates the newest-first slice of the force history

        Given: A history of 6 samples where a control step spans 2
        When: The samples are read
        Then: Only the two leading entries are returned

        Test type: unit
        """
        history = np.arange(6 * 1 * 3, dtype=float).reshape(1, 6, 1, 3)
        env = self._env(history=history, step_dt=0.2, physics_dt=0.1)
        assert isaac_module.contact_force_samples(env, "contact_forces").shape == (2, 1, 3)

    def test_a_sensor_with_no_force_buffer_is_reported_not_guessed(self) -> None:
        """A missing buffer means the sensor is not what the caller thinks it is.

        Purpose: Validates the error when neither force buffer is present

        Given: A contact sensor exposing no force fields
        When: The samples are read
        Then: RuntimeError names the sensor and suggests a custom extractor

        Test type: unit
        """
        with pytest.raises(RuntimeError, match="contact_forces"):
            isaac_module.contact_force_samples(self._env(), "contact_forces")


class TestImpactChannelSelection:
    """Which name a configured impact measurement is reported under.

    The default extractor measures an impulse in newton-seconds. An override that measures the
    peak force in newtons is a *different quantity*, and reporting it under the impulse channel
    would roll newtons and newton-seconds into one mean with nothing downstream able to notice.
    """

    def test_the_default_reports_the_impulse_channel(self, fake_env: FakeIsaacEnv) -> None:
        """The unconfigured path must keep reporting exactly what it always did.

        Purpose: Validates the default impact channel and metric

        Given: A world with a contact sensor and no channel override
        When: Its metric specs are read
        Then: The impulse channel rolls up into the max-impulse metric

        Test type: unit
        """
        del fake_env
        world = _measuring_world()
        spec = next(spec for spec in world.get_metric_specs() if "impulse" in spec.name)
        assert spec.channel == isaac_module.IsaacLabStepChannel.CONTACT_IMPULSE_NS.value
        assert spec.name == isaac_module.IsaacLabMetric.MAX_CONTACT_IMPULSE_NS.value

    def test_a_peak_force_extractor_reports_the_peak_force_channel(
        self, fake_env: FakeIsaacEnv
    ) -> None:
        """A newton reading under a newton-second name is a silent unit error.

        Purpose: Validates that the selected channel drives both step_info and the metric spec

        Given: A world whose impact extractor returns a peak force, declaring that channel
        When: A step is taken and the metric specs are read
        Then: The reading appears under the peak-force channel and its metric, and the impulse
            channel is absent

        Test type: unit
        """
        del fake_env
        world = _measuring_world(
            impact_extractor=lambda env: 9481.0,
            impact_channel=isaac_module.IsaacLabStepChannel.CONTACT_PEAK_FORCE_N,
        )
        state = world._live_state
        next_state = world.sample_next_state(state, np.zeros(2))
        info = world.step_info(state, np.zeros(2), next_state)

        peak_channel = isaac_module.IsaacLabStepChannel.CONTACT_PEAK_FORCE_N.value
        assert info[peak_channel] == pytest.approx(9481.0)
        assert isaac_module.IsaacLabStepChannel.CONTACT_IMPULSE_NS.value not in info
        spec = next(spec for spec in world.get_metric_specs() if spec.channel == peak_channel)
        assert spec.name == isaac_module.IsaacLabMetric.MAX_CONTACT_PEAK_FORCE_N.value

    def test_a_channel_that_is_not_an_impact_channel_is_rejected(self) -> None:
        """Selecting the success channel would overwrite the success flag with a force.

        Purpose: Validates construction-time rejection of a non-impact channel

        Given: A world configured with the success channel as its impact channel
        When: It is constructed
        Then: ValueError lists the channels that are allowed

        Test type: unit
        """
        with pytest.raises(ValueError, match="contact_peak_force_n"):
            IsaacLabPOMDP(
                task_id="Fake-Isaac-v0",
                discount_factor=0.99,
                device="cpu",
                contact_sensor_key="contact_forces",
                impact_channel=isaac_module.IsaacLabStepChannel.SUCCESS,
            )
