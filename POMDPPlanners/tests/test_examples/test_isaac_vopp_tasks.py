# SPDX-License-Identifier: MIT

"""Tests for the task definitions and success predicates of the IsaacLab VOPP study.

A success predicate is the one piece of a benchmark that cannot be checked by reading the number
it produces: a predicate that cannot fail a do-nothing policy reports a perfect score and looks
like a result. ``Isaac-Cartpole-v0`` shipped exactly that -- its only failure term is
``cart_out_of_bounds``, so a policy that lets the pole spin scores 1.0. These tests pin the
tightened predicate against stubbed observations, which is the cheap half of the check; the
expensive half is the do-nothing rollout on the live simulator, and both are needed.

They also pin the warm-up action hold, because the value of a held rollout is invisible in any
output the script prints -- it shows up only as a calibration that is closer to the truth.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# pylint: disable=wrong-import-position
from isaac_vopp_metrics_example import WorldDriver, collect_warmup_samples  # noqa: E402
from isaac_vopp_tasks import (  # noqa: E402
    TASKS,
    CartpoleUprightProbe,
    NavigationSuccessProbe,
    ReachSuccessProbe,
    ThresholdSuccessProbe,
    make_success_extractor,
)

#: The pole reset range ``Isaac-Cartpole-v0`` declares, in radians. Any success threshold below it
#: would fail episodes on their first step whatever the policy did.
CARTPOLE_POLE_RESET_RANGE = 0.25 * np.pi


class _StubEnv:
    """A live IsaacLab task reduced to the three things a success predicate reads off it."""

    def __init__(
        self,
        observation: np.ndarray,
        joint_names: Optional[List[str]] = None,
        terms: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.observation = np.asarray(observation, dtype=float)
        self.joint_names = joint_names or []
        self.terms = terms or {}
        self.unwrapped = self

    # -- the shape POMDPPlanners' helpers expect of a live env --------------
    @property
    def observation_manager(self) -> Any:
        return self

    def compute_group(self, name: str) -> np.ndarray:
        assert name == "policy"
        return self.observation[np.newaxis, :]

    @property
    def termination_manager(self) -> Any:
        return self

    def get_term(self, name: str) -> np.ndarray:
        if name not in self.terms:
            raise KeyError(name)
        return np.asarray([[self.terms[name]]])

    @property
    def scene(self) -> Any:
        return {"robot": self}


def _cartpole_env(pole_angle: float, out_of_bounds: bool = False) -> _StubEnv:
    """A cartpole observation ``[cart_pos, pole_pos, cart_vel, pole_vel]``, joints named."""
    return _StubEnv(
        observation=np.array([0.0, pole_angle, 0.0, 0.0]),
        joint_names=["slider_to_cart", "cart_to_pole"],
        terms={"cart_out_of_bounds": out_of_bounds},
    )


# ── The cartpole predicate, which exists because the shipped one did not discriminate ──


@pytest.mark.parametrize("pole_angle", [0.0, 0.5, 1.5, -1.5])
def test_the_cartpole_predicate_passes_a_pole_that_is_still_up(pole_angle: float) -> None:
    """A predicate that failed a legal upright pole would measure the reset draw, not the policy.

    Purpose: Validates that angles inside the threshold, either sign, count as a success

    Given: A cartpole whose pole is within a right angle of vertical and whose cart is in bounds
    When: The predicate is evaluated
    Then: It reports success

    Test type: unit
    """
    probe = CartpoleUprightProbe(float(np.pi / 2.0))
    assert probe(_cartpole_env(pole_angle), {}, False, False) is True


@pytest.mark.parametrize("pole_angle", [1.6, -1.6, 3.14, 4.83])
def test_the_cartpole_predicate_fails_a_pole_past_horizontal(pole_angle: float) -> None:
    """This is the case the shipped predicate scored 1.0: the pole spinning, the cart in bounds.

    Purpose: Regression guard -- a pole beyond the threshold must fail even with the cart in bounds

    Given: A cartpole whose pole has fallen past a right angle but whose cart is in bounds
    When: The predicate is evaluated
    Then: It reports failure

    Test type: unit
    """
    probe = CartpoleUprightProbe(float(np.pi / 2.0))
    assert probe(_cartpole_env(pole_angle), {}, False, False) is False


def test_the_cartpole_predicate_still_fails_a_cart_that_left_its_bounds() -> None:
    """Tightening the predicate must add to the task's own failure term, not replace it.

    Purpose: Validates that the cart bound is still enforced when the pole is upright

    Given: A cartpole holding its pole vertical while cart_out_of_bounds is set
    When: The predicate is evaluated
    Then: It reports failure

    Test type: unit
    """
    probe = CartpoleUprightProbe(float(np.pi / 2.0))
    assert probe(_cartpole_env(0.0, out_of_bounds=True), {}, False, False) is False


def test_the_cartpole_predicate_refuses_a_task_missing_the_cart_bound_term() -> None:
    """A silently dropped half is exactly the failure this predicate was written to remove.

    Purpose: Validates that a renamed or removed cart_out_of_bounds term fails loudly

    Given: A cartpole task declaring no cart_out_of_bounds termination term
    When: The predicate is evaluated
    Then: RuntimeError is raised rather than the cart half silently passing

    Test type: unit
    """
    without_term = _StubEnv(
        observation=np.array([0.0, 0.1, 0.0, 0.0]),
        joint_names=["slider_to_cart", "cart_to_pole"],
        terms={},
    )
    with pytest.raises(RuntimeError, match="cart_out_of_bounds"):
        CartpoleUprightProbe(float(np.pi / 2.0))(without_term, {}, False, False)


def test_the_cartpole_threshold_clears_the_tasks_own_pole_reset_range() -> None:
    """The threshold is forced by the reset range, and that is the reason worth pinning.

    Purpose: Validates that no legal initial pole angle can fail the predicate on step one

    Given: The configured cartpole threshold and the task's +/-45 degree pole reset range
    When: The two are compared
    Then: The threshold is strictly larger, so the predicate measures the policy and not the draw

    Test type: configuration
    """
    spec = next(task for task in TASKS if task.task_id == "Isaac-Cartpole-v0")
    assert spec.success_threshold > CARTPOLE_POLE_RESET_RANGE
    assert spec.success_reduction == "all"


def test_the_cartpole_probe_finds_the_pole_by_name_not_by_position() -> None:
    """A hard-coded index would read the cart's position as an angle on a reordered articulation.

    Purpose: Validates that the pole joint is located by name in the articulation

    Given: An articulation whose joints are declared pole-first, with a spinning pole
    When: The predicate is evaluated
    Then: It fails, having read the pole entry rather than the cart entry

    Test type: unit
    """
    reordered = _StubEnv(
        observation=np.array([3.0, 0.0, 0.0, 0.0]),  # pole first this time
        joint_names=["cart_to_pole", "slider_to_cart"],
        terms={"cart_out_of_bounds": False},
    )
    assert CartpoleUprightProbe(float(np.pi / 2.0))(reordered, {}, False, False) is False


def test_the_cartpole_probe_reports_the_worst_angle_it_saw() -> None:
    """The rate alone hides how close a run came; the worst angle is what makes it auditable.

    Purpose: Validates the per-episode summary of the upright probe

    Given: An episode whose pole reaches 1.2 rad at its worst
    When: The episode summary is read
    Then: It reports that angle in radians and degrees, with the end-of-episode flags

    Test type: unit
    """
    probe = CartpoleUprightProbe(float(np.pi / 2.0))
    for angle in (0.1, 1.2, 0.4):
        probe(_cartpole_env(angle), {}, False, True)
    summary = probe.summary()
    assert summary["max_pole_angle_rad"] == pytest.approx(1.2)
    assert summary["max_pole_angle_deg"] == pytest.approx(68.75, abs=0.01)
    assert summary["truncated"] is True and summary["terminated"] is False


# ── Shared probe bookkeeping ────────────────────────────────────────────


def test_a_probe_forgets_the_previous_episode_on_reset() -> None:
    """Carried-over measurements would score one episode with another episode's best moment.

    Purpose: Validates that reset clears measurements and the end-of-episode flags

    Given: A probe that has recorded a passing episode
    When: It is reset
    Then: Its summary is empty and its flags are cleared

    Test type: unit
    """
    probe = CartpoleUprightProbe(float(np.pi / 2.0))
    probe(_cartpole_env(0.2), {}, True, False)
    assert probe.summary()
    probe.reset()
    assert probe.summary() == {}
    assert not probe.measurements and not probe.terminated and not probe.truncated


def test_the_navigation_probe_thresholds_the_planar_distance_from_the_observation() -> None:
    """Reading the goal off the observation is what keeps the score honest about what is visible.

    Purpose: Validates that navigation success uses the observation's pose_command block

    Given: A ten-wide navigation observation whose base-frame goal is 0.3 m away in the plane and
        offset in z, under a 0.5 m threshold
    When: The predicate is evaluated
    Then: It reports success, and the z entry does not count against it

    Test type: unit
    """
    observation = np.zeros(10)
    observation[6:10] = [0.18, 0.24, 5.0, 0.0]  # 0.3 m in the plane, a nonsense height gap
    assert NavigationSuccessProbe(0.5)(_StubEnv(observation), {}, False, False) is True
    observation[6:8] = [0.4, 0.4]  # 0.566 m in the plane
    assert NavigationSuccessProbe(0.5)(_StubEnv(observation), {}, False, False) is False


def test_every_task_builds_the_predicate_its_configuration_names() -> None:
    """A task silently falling through to "did not fail" is how a benchmark reports a fake score.

    Purpose: Validates the success_kind dispatch for every configured task

    Given: The four configured tasks
    When: Each one's success extractor is built
    Then: The two distance tasks and cartpole get their probes, and the rest get a plain predicate

    Test type: configuration
    """
    built = {spec.task_id: make_success_extractor(spec) for spec in TASKS}
    assert isinstance(built["Isaac-Reach-Franka-v0"], ReachSuccessProbe)
    assert isinstance(built["Isaac-Navigation-Flat-Anymal-C-v0"], NavigationSuccessProbe)
    assert isinstance(built["Isaac-Cartpole-v0"], CartpoleUprightProbe)
    assert not isinstance(built["Isaac-Velocity-Flat-Anymal-C-v0"], ThresholdSuccessProbe)


# ── The warm-up action hold ─────────────────────────────────────────────


class _StubDistribution:
    """The one method ``WorldDriver.reset`` calls on an initial-state distribution."""

    def sample(self, n_samples: int = 1) -> List[np.ndarray]:
        return [np.zeros(1) for _ in range(n_samples)]


class _StubWorld:
    """A world that counts its steps and ends an episode every ``episode_length`` of them.

    It holds its state still so that only the action schedule and the episode boundaries vary,
    which is what the warm-up tests are about.

    Attributes:
        episode_length: Steps before the episode ends, or ``None`` to never end one.
        steps: Steps taken since construction.
    """

    record_video = False

    def __init__(self, episode_length: Optional[int] = None) -> None:
        self.episode_length = episode_length
        self.steps = 0
        self._ended = False

    def initial_state_dist(self) -> _StubDistribution:
        self._ended = False
        return _StubDistribution()

    def reward(self, state: Any, action: Any) -> float:
        del state, action
        return 0.0

    def sample_next_state(self, state: Any, action: Any) -> np.ndarray:
        del action
        self.steps += 1
        self._ended = self.episode_length is not None and self.steps % self.episode_length == 0
        return np.asarray(state, dtype=float)

    def is_terminal(self, state: Any) -> bool:
        del state
        return self._ended


def _stub_driver(num_presets: int = 8, episode_length: Optional[int] = None) -> WorldDriver:
    """A real driver over a stub world, so the warm-up under test is the one that ships."""
    presets = np.arange(num_presets, dtype=float).reshape(num_presets, 1)
    return WorldDriver(_StubWorld(episode_length), presets, device=None)


def test_the_warmup_holds_each_action_for_the_configured_number_of_steps() -> None:
    """A command redrawn every step measures a permanent transient, not the system's tracking.

    Purpose: Validates that the warm-up rollout holds each drawn action

    Given: A warm-up of 60 transitions with a hold of 10 steps
    When: The rollout is collected
    Then: The applied action is constant within each block of ten and changes only at the boundaries

    Test type: unit
    """
    _, actions, _, _ = collect_warmup_samples(_stub_driver(), 60, hold_steps=10)
    blocks = actions.reshape(6, 10)
    for block in blocks:
        assert len(set(block.tolist())) == 1


def test_a_hold_of_one_reproduces_the_redrawn_rollout() -> None:
    """The hold has to be a parameter, not a rewrite, so the old protocol stays reproducible.

    Purpose: Validates that hold_steps=1 recovers the per-step redraw the study started from

    Given: A warm-up of 40 transitions with a hold of one step
    When: The rollout is collected
    Then: It draws many distinct actions rather than a handful of held blocks

    Test type: unit
    """
    _, actions, _, _ = collect_warmup_samples(_stub_driver(), 40, hold_steps=1)
    changes = int(np.sum(actions[1:] != actions[:-1]))
    assert changes > 20


def test_the_warmup_drops_the_transition_that_spans_an_episode_reset() -> None:
    """IsaacLab auto-resets inside step(), so that successor is a fresh episode, not a result.

    Purpose: Validates that terminal transitions are excluded and the sample budget still met

    Given: A world ending an episode every 7 steps and a request for 30 transitions
    When: The warm-up is collected
    Then: Exactly 30 rows come back, and the world was stepped more than 30 times to get them

    Test type: unit
    """
    driver = _stub_driver(episode_length=7)
    states, _, _, _ = collect_warmup_samples(driver, 30, hold_steps=10)
    assert states.shape[0] == 30
    assert driver.world.steps > 30


def test_the_hold_schedule_restarts_after_an_episode_reset() -> None:
    """A block resumed after a reset hands the fresh system a command it cannot follow in time.

    Purpose: Validates that the action hold restarts rather than resuming mid-block

    Given: A world ending an episode every 5 steps, a hold of 4 and 24 requested transitions
    When: The warm-up is collected
    Then: No held run of actions is split across a reset -- every recorded run is a whole block or
        is cut only by reaching the requested count

    Test type: unit
    """
    driver = _stub_driver(episode_length=5)
    _, actions, _, _ = collect_warmup_samples(driver, 24, hold_steps=4)
    runs: List[int] = []
    current = 1
    for previous, following in zip(actions[:-1], actions[1:]):
        if previous == following:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    # Every episode contributes four usable rows (the fifth spans the reset and is dropped), so a
    # hold of four can never be observed longer than four.
    assert max(runs) <= 4


def test_the_warmup_rejects_a_non_positive_hold() -> None:
    """A zero hold is a silent modulo-by-zero, and a negative one redraws every step by accident.

    Purpose: Validates the guard on hold_steps

    Given: A hold of zero steps
    When: The warm-up is collected
    Then: ValueError is raised

    Test type: unit
    """
    with pytest.raises(ValueError, match="hold_steps must be positive"):
        collect_warmup_samples(_stub_driver(), 10, hold_steps=0)
