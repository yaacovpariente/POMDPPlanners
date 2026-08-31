# SPDX-License-Identifier: MIT

"""Unit and QA tests for the hazard/severity Isaac generative models.

Pure numpy throughout — no Isaac Sim. Covers the per-environment behaviour the shared
conformance suite cannot: hazard tracking under the SE(2) frame update, terminal contact
bookkeeping, reward-range conformance under adversarial states, signal informativeness inside
versus outside the zone, batch-versus-scalar parity, belief updates, the constrained twins, and
the env-qa cheapest gate (a scripted straight line reaches the goal under zero noise; a random
policy pays measurably more collision cost than a scripted detour).
"""

import numpy as np
import pytest

from POMDPPlanners.core.belief import WeightedParticleBelief
from POMDPPlanners.core.environment.environment import Environment
from POMDPPlanners.environments.isaac_lab_pomdp.isaac_generative_models.isaac_hazard_models import (
    ConstrainedHazardNavigationIsaacModel,
    ConstrainedHazardReachIsaacModel,
    EndEffectorPresenceSignalObservationModel,
    EPISODE_DONE_CHANNEL,
    HAZARD_SIGNAL_CHANNEL,
    HAZARD_TYPE_CHANNEL,
    HAZARD_XY_CHANNEL,
    HazardNavigationIsaacModel,
    HazardReachIsaacModel,
    OBSTACLE_PRESENCE_CHANNEL,
    RelativeHazardSignalObservationModel,
)
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler

QUIET = 1e-9  # noise stds must be strictly positive; this is "zero noise" in practice

STRAIGHT, SLOW, ARC_LEFT, ARC_RIGHT, STOP = range(5)

#: Scripted detour around the (2.0, 0.0, 0.5) hazard, found by simulation: arc left, run
#: straight, arc back right, correct left, then drive at the goal. Clears the hazard by >0.2 m
#: under zero noise.
DETOUR_PLAN = [ARC_LEFT] * 4 + [STRAIGHT] * 6 + [ARC_RIGHT] * 6 + [ARC_LEFT] * 1


def _quiet_navigation(**overrides):
    kwargs = {
        "hazards": [(2.0, 0.0, 0.5)],
        "p_bad": 1.0,
        "velocity_noise_std": QUIET,
        "position_noise_std": QUIET,
        "heading_noise_std": QUIET,
    }
    kwargs.update(overrides)
    return HazardNavigationIsaacModel(**kwargs)


def _rollout(model, plan, horizon=45):
    """Run a fixed action plan (padded with STRAIGHT), returning per-step states."""
    actions = model.get_actions()
    state = model.initial_state_dist().sample(1)[0]
    states = [state]
    sequence = list(plan) + [STRAIGHT] * horizon
    for index in sequence[:horizon]:
        if model.is_terminal(state):
            break
        state = model.sample_next_state(state, actions[index])
        states.append(state)
    return states


# ── Serialization and identity ─────────────────────────────────────────


@pytest.mark.parametrize(
    "builder",
    [
        lambda: HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.4)], p_bad=0.7),
        lambda: HazardReachIsaacModel(p_present=0.3, collision_penalty=30.0),
        lambda: ConstrainedHazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.4)], p_bad=0.7),
        lambda: ConstrainedHazardReachIsaacModel(p_present=0.3),
    ],
    ids=["navigation", "reach", "constrained-navigation", "constrained-reach"],
)
def test_serialization_round_trips(builder):
    """to_dict / from_dict rebuilds an identically configured environment.

    Purpose: Validates the env-implementation serialization contract, including the twins whose
    constructors take (*args, **kwargs) and publish the parent signature.

    Given: Each hazard model with non-default parameters.
    When: The environment is serialized and rebuilt via Environment.from_dict.
    Then: The rebuilt environment has the same config_id and class.

    Test type: unit
    """
    original = builder()
    rebuilt = Environment.from_dict(original.to_dict())
    assert type(rebuilt) is type(original)
    assert rebuilt.config_id == original.config_id


def test_config_id_stable_after_use():
    """Using the model (sampling, densities, belief math) does not change config_id.

    Purpose: config_id is a cache key; lazily memoized observation internals must not leak into
    it (the trap that motivates the constructor's cache warm-up).

    Given: A fresh navigation model and its config_id.
    When: The model samples transitions, observations and densities.
    Then: config_id is unchanged and equals a second fresh instance's.

    Test type: unit
    """
    np.random.seed(0)
    model = _quiet_navigation()
    fresh = model.config_id
    state = model.initial_state_dist().sample(1)[0]
    action = model.get_actions()[STRAIGHT]
    next_state = model.sample_next_state(state, action)
    observation = model.sample_observation(next_state, action)
    model.observation_log_probability(next_state, action, observation)
    model.reward(state, action, next_state)
    assert model.config_id == fresh
    assert _quiet_navigation().config_id == fresh


def test_state_schema_round_trip():
    """pack / split round-trips the hazard schema blocks.

    Purpose: Validates the augmented schema carves the flat vector where the model reads it.

    Given: A navigation and a reach model.
    When: A packed state is split back into blocks.
    Then: Every block matches what was packed, and widths match the hazard count.

    Test type: unit
    """
    model = HazardNavigationIsaacModel(hazards=[(1.0, 0.5, 0.3), (3.0, -1.0, 0.4)])
    assert model.state_schema.width(HAZARD_XY_CHANNEL) == 4
    assert model.state_schema.width(HAZARD_TYPE_CHANNEL) == 2
    assert model.state_schema.width(EPISODE_DONE_CHANNEL) == 1
    state = model.initial_state_dist().sample(1)[0]
    blocks = model.state_schema.split(state)
    assert np.array_equal(blocks[HAZARD_XY_CHANNEL], [1.0, 0.5, 3.0, -1.0])
    assert np.array_equal(model.state_schema.pack(blocks), state)

    reach = HazardReachIsaacModel()
    assert reach.state_schema.width(OBSTACLE_PRESENCE_CHANNEL) == 1
    assert reach.state_schema.total_dim == 9 + 9 + 7 + 7 + 1 + 1


# ── Hazard tracking under the SE(2) update ─────────────────────────────


def test_hazard_on_the_path_is_hit_by_driving_straight():
    """Driving straight under zero noise closes on a hazard placed on the path.

    Purpose: Validates the SE(2) inverse update moves hazard centres with the floor — the core
    correctness property of the goal-relative hazard representation.

    Given: A hazard at (2.0, 0.0) with radius 0.5, zero-noise dynamics, a bad type.
    When: The robot drives (1.0, 0, 0) at 0.2 s per step.
    Then: The hazard centre approaches the origin by 0.2 m per step and contact occurs on the
        step where its distance falls to 0.4 m (< radius), setting the terminal slot.

    Test type: unit
    """
    np.random.seed(0)
    model = _quiet_navigation()
    action = model.get_actions()[STRAIGHT]
    state = model.initial_state_dist().sample(1)[0]
    for step in range(1, 9):
        state = model.sample_next_state(state, action)
        center = model.state_schema.block(state, HAZARD_XY_CHANNEL)
        assert np.allclose(center, [2.0 - 0.2 * step, 0.0], atol=1e-6)
    assert float(model.state_schema.block(state, EPISODE_DONE_CHANNEL)[0]) == 1.0
    assert model.is_terminal(state)


def test_hazard_rotates_with_the_goal_under_turning():
    """A hazard co-located with the goal stays co-located through arbitrary turning.

    Purpose: The goal and a hazard centre are both floor-fixed points in the base frame, so the
    same SE(2) update must move them identically; divergence would mean the hazard detached
    from the floor.

    Given: A hazard centred exactly on the goal's planar position, zero noise.
    When: The robot follows a turning action sequence.
    Then: After every step the hazard centre equals the goal's (x, y) to numerical precision.

    Test type: unit
    """
    np.random.seed(0)
    model = HazardNavigationIsaacModel(
        hazards=[(4.0, 0.0, 0.1)],
        initial_goal=(4.0, 0.0, 0.0, 0.0),
        p_bad=0.0,
        velocity_noise_std=QUIET,
        position_noise_std=QUIET,
        heading_noise_std=QUIET,
    )
    actions = model.get_actions()
    state = model.initial_state_dist().sample(1)[0]
    for index in [ARC_LEFT, ARC_LEFT, STRAIGHT, ARC_RIGHT, SLOW, ARC_RIGHT, STRAIGHT]:
        state = model.sample_next_state(state, actions[index])
        goal_xy = model.state_schema.block(state, "pose_command")[:2]
        center = model.state_schema.block(state, HAZARD_XY_CHANNEL)
        assert np.allclose(center, goal_xy, atol=1e-6)


# ── Terminal bookkeeping ───────────────────────────────────────────────


def test_benign_contact_is_not_terminal():
    """Contact with a benign hazard neither terminates nor sets the slot.

    Purpose: Only the latent *bad* type is dangerous; a planner must be able to drive through a
    benign disc.

    Given: The path hazard with p_bad = 0.

    When: The robot drives straight through it to the goal.
    Then: No terminal slot is set and the episode ends by reaching the goal.

    Test type: unit
    """
    np.random.seed(0)
    model = _quiet_navigation(p_bad=0.0)
    states = _rollout(model, [STRAIGHT] * 45)
    final = states[-1]
    assert float(model.state_schema.block(final, EPISODE_DONE_CHANNEL)[0]) == 0.0
    assert model.hazard_reward.planar_goal_distance(final) < model.success_radius
    assert model.is_terminal(final)


def test_non_terminal_contact_still_pays_the_penalty_every_step():
    """With is_bad_contact_terminal=False the episode continues and keeps paying.

    Purpose: Validates the flag changes episode structure, not the reward.

    Given: A bad hazard, is_bad_contact_terminal=False, zero noise.
    When: The robot drives straight through the hazard.
    Then: No state is terminal before the goal, and every in-contact transition's reward
        carries the full-speed penalty.

    Test type: unit
    """
    np.random.seed(0)
    model = _quiet_navigation(is_bad_contact_terminal=False)
    action = model.get_actions()[STRAIGHT]
    state = model.initial_state_dist().sample(1)[0]
    penalized = 0
    for _ in range(12):
        assert not model.is_terminal(state)
        next_state = model.sample_next_state(state, action)
        if model.hazard_reward.bad_contact(next_state):
            reward = model.reward(state, action, next_state)
            no_hazard = model.hazard_reward
            expected_penalty = -no_hazard.collision_penalty  # full speed -> saturated severity
            base = reward - no_hazard.collision_term(next_state)
            assert reward == pytest.approx(base + expected_penalty)
            penalized += 1
        state = next_state
    assert penalized >= 2  # the disc is 1.0 m across at 0.2 m per step


def test_terminal_slot_is_sticky():
    """Once set, the terminal slot survives further transitions.

    Purpose: Process noise must not be able to un-terminate an episode.

    Given: A terminal state produced by a bad contact.
    When: Another transition is sampled from it.
    Then: The slot is still set and the state terminal.

    Test type: unit
    """
    np.random.seed(0)
    model = _quiet_navigation()
    states = _rollout(model, [STRAIGHT] * 45)
    terminal = states[-1]
    assert model.is_terminal(terminal)
    after = model.sample_next_state(terminal, model.get_actions()[STOP])
    assert float(model.state_schema.block(after, EPISODE_DONE_CHANNEL)[0]) == 1.0


# ── Reward range conformance ───────────────────────────────────────────


def _random_navigation_states(model, count, rng):
    dim = model.state_schema.total_dim
    rows = rng.normal(0.0, 5.0, size=(count, dim))
    # exercise extreme speeds, in-contact geometry and both latent types
    rows[:, : 2] = rng.normal(0.0, 10.0, size=(count, 2))
    type_slice = model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)
    rows[:, type_slice] = (rng.random((count, model.num_hazards)) < 0.5).astype(float)
    xy_slice = model.state_schema.slice_of(HAZARD_XY_CHANNEL)
    rows[: count // 2, xy_slice] = rng.normal(0.0, 0.2, size=(count // 2, 2 * model.num_hazards))
    done_slice = model.state_schema.slice_of(EPISODE_DONE_CHANNEL)
    rows[:, done_slice] = 0.0
    return rows


def test_navigation_reward_range_bounds_every_reward():
    """Every reachable-and-unreachable state's reward lies in the declared range.

    Purpose: The declared range is the repository's most-repeated bug; this drives adversarial
    states (unbounded speeds included — the clip is what keeps the bound true) through it.

    Given: Models across the flag grid (penalty on/off, p_bad 0/1, terminal on/off).
    When: 400 adversarial states are scored against each model.
    Then: All rewards lie within the declared range, and a worst-case constructed state
        approaches the declared minimum.

    Test type: unit
    """
    rng = np.random.default_rng(3)
    action = None
    for collision_penalty in (0.0, 50.0):
        for p_bad in (0.0, 1.0):
            for terminal in (True, False):
                model = HazardNavigationIsaacModel(
                    hazards=[(2.0, 0.0, 0.5)],
                    p_bad=p_bad,
                    collision_penalty=collision_penalty,
                    is_bad_contact_terminal=terminal,
                )
                assert model.reward_range is not None
                low, high = model.reward_range
                rows = _random_navigation_states(model, 400, rng)
                if p_bad == 0.0:
                    # states a p_bad = 0 model can actually visit carry benign types only
                    rows[:, model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)] = 0.0
                rewards = model.reward_batch(rows, action, rows)
                assert float(rewards.min()) >= low - 1e-12
                assert float(rewards.max()) <= high + 1e-12

    # worst case: saturated-speed bad contact with a pi heading error, far from the goal
    model = HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.5)], p_bad=1.0)
    worst = np.zeros(model.state_schema.total_dim)
    worst[0] = 99.0  # clipped to v_max
    worst[model.state_schema.slice_of("pose_command")] = [1e6, 0.0, 0.0, np.pi]
    worst[model.state_schema.slice_of(HAZARD_XY_CHANNEL)] = [0.0, 0.0]
    worst[model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)] = 1.0
    reward = model.reward(worst, None, worst)
    assert model.reward_range is not None
    low, _ = model.reward_range
    assert reward >= low - 1e-12
    assert reward == pytest.approx(low, abs=1e-9)


def test_reach_reward_range_bounds_every_reward():
    """Reach rewards, contact penalties included, lie in the declared range.

    Purpose: Same enumeration discipline for the reach model, whose distance bound comes from
    the kinematic reach of the chain.

    Given: Models across the flag grid (penalty on/off, p_present 0/1).
    When: 200 random joint states (huge joint angles included) are scored, with and without a
        moving predecessor state.
    Then: All rewards lie within the declared range.

    Test type: unit
    """
    rng = np.random.default_rng(4)
    for collision_penalty in (0.0, 25.0):
        for p_present in (0.0, 1.0):
            model = HazardReachIsaacModel(
                p_present=p_present, collision_penalty=collision_penalty
            )
            assert model.reward_range is not None
            low, high = model.reward_range
            dim = model.state_schema.total_dim
            rows = rng.normal(0.0, 3.0, size=(200, dim))
            presence_slice = model.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)
            command_slice = model.state_schema.slice_of("command")
            rows[:, presence_slice] = float(p_present)
            # The command block is carried with no noise, so every reachable state holds the
            # constructor's goal; the declared range is a claim about reachable states.
            rows[:, command_slice] = np.asarray(model.goal_command)
            previous = rng.normal(0.0, 3.0, size=(200, dim))
            previous[:, presence_slice] = float(p_present)
            previous[:, command_slice] = np.asarray(model.goal_command)
            rewards = model.reward_batch(previous, None, rows)
            assert float(rewards.min()) >= low - 1e-12
            assert float(rewards.max()) <= high + 1e-12
            resting = model.reward_batch(rows, None, None)
            assert float(resting.min()) >= low - 1e-12
            assert float(resting.max()) <= high + 1e-12


# ── Signal informativeness ─────────────────────────────────────────────


def test_navigation_signal_is_informative_only_inside_the_radius():
    """The hazard signal separates types in range and is a coin flip out of range.

    Purpose: Validates the reveal rule the whole construction depends on — the belief can learn
    the type only by approaching.

    Given: States with the hazard inside and outside signal_radius.
    When: Signal likelihoods are scored for agreeing and disagreeing readings.
    Then: In range the likelihoods differ by log(0.9 / 0.1); out of range they are equal, and
        a Bayes update leaves the prior untouched.

    Test type: unit
    """
    model = HazardNavigationIsaacModel(
        hazards=[(2.0, 0.0, 0.4)], signal_radius=1.5, signal_accuracy=0.9
    )
    assert model.observation_models is not None
    signal_model = model.observation_models[HAZARD_SIGNAL_CHANNEL]
    assert isinstance(signal_model, RelativeHazardSignalObservationModel)
    inside = {"hazard_type": np.array([1.0]), HAZARD_XY_CHANNEL: np.array([1.0, 0.0])}
    outside = {"hazard_type": np.array([1.0]), HAZARD_XY_CHANNEL: np.array([3.0, 0.0])}

    agree_in = signal_model.log_probability(inside, np.array([1.0]))
    disagree_in = signal_model.log_probability(inside, np.array([0.0]))
    assert agree_in - disagree_in == pytest.approx(np.log(0.9 / 0.1))

    agree_out = signal_model.log_probability(outside, np.array([1.0]))
    disagree_out = signal_model.log_probability(outside, np.array([0.0]))
    assert agree_out == pytest.approx(disagree_out)

    posterior = signal_model.posterior_after_signal([0.5], outside, [1.0])
    assert posterior[0] == pytest.approx(0.5)
    posterior_in = signal_model.posterior_after_signal([0.5], inside, [1.0])
    assert posterior_in[0] == pytest.approx(0.9)


def test_reach_signal_gated_on_end_effector_proximity():
    """The presence signal is informative only with the hand near the obstacle.

    Purpose: Same reveal rule for the reach model, through forward kinematics.

    Given: The default model, whose initial hand position is ~0.21 m from the obstacle with
        signal_radius 0.15.
    When: Accuracy is evaluated at the initial pose and at the +0.4 preset's settled pose.
    Then: Far pose: accuracy 0.5. Near pose: accuracy 0.9.

    Test type: unit
    """
    model = HazardReachIsaacModel(signal_accuracy=0.9)
    assert model.observation_models is not None
    signal_model = model.observation_models[HAZARD_SIGNAL_CHANNEL]
    assert isinstance(signal_model, EndEffectorPresenceSignalObservationModel)
    far = model.state_schema.split(model.initial_state_dist().sample(1)[0])
    assert signal_model.accuracy_at(far).tolist() == [0.5]
    # the +0.4 preset settles the commanded joints at 0.4 * action_scale = 0.2 rad offsets,
    # which puts the hand next to the default obstacle
    near = dict(far)
    near["joint_pos"] = np.concatenate([np.full(7, 0.2), np.zeros(2)])
    hand = signal_model.end_effector_position(near)
    assert np.linalg.norm(hand - signal_model.obstacle_center) <= model.signal_radius
    assert signal_model.accuracy_at(near).tolist() == [0.9]


# ── Batch versus scalar parity ─────────────────────────────────────────


def test_navigation_batch_paths_match_scalar_paths():
    """reward_batch and sample_next_state_batch agree with the scalar methods.

    Purpose: PFT-DPW belief updates run on the batch paths; disagreement (dtype included) would
    silently skew every particle weight.

    Given: A batch of sampled navigation states.
    When: Batch and per-state scalar results are computed under the same seeds.
    Then: Rewards agree exactly; batched samples equal looped scalar samples draw for draw;
        dtypes are float64.

    Test type: unit
    """
    np.random.seed(5)
    model = HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.5)], p_bad=1.0)
    action = model.get_actions()[STRAIGHT]
    rows = np.vstack(
        [model.sample_next_state(s, action) for s in model.initial_state_dist().sample(8)]
    )
    batch_rewards = model.reward_batch(rows, action, rows)
    scalar_rewards = np.array([model.reward(rows[i], action, rows[i]) for i in range(8)])
    assert batch_rewards.dtype == np.float64
    np.testing.assert_allclose(batch_rewards, scalar_rewards, rtol=0, atol=0)

    np.random.seed(11)
    batched = model.sample_next_state_batch(rows, action)
    np.random.seed(11)
    looped = np.vstack([model.sample_next_state(rows[i], action) for i in range(8)])
    assert batched.dtype == np.float64
    np.testing.assert_allclose(batched, looped, atol=1e-12)


def test_reach_batch_paths_match_scalar_paths():
    """Same parity check for the reach model, hand-speed severity included.

    Purpose: As above, for the manipulator.

    Given: A batch of perturbed reach states with the obstacle present.
    When: Batch and scalar rewards / samples are computed under the same seeds.
    Then: They agree exactly, dtype float64.

    Test type: unit
    """
    np.random.seed(6)
    model = HazardReachIsaacModel(p_present=1.0)
    action = model.get_actions()[1]
    starts = np.vstack(model.initial_state_dist().sample(6))
    nexts = np.vstack([model.sample_next_state(starts[i], action) for i in range(6)])
    batch_rewards = model.reward_batch(starts, action, nexts)
    scalar_rewards = np.array([model.reward(starts[i], action, nexts[i]) for i in range(6)])
    assert batch_rewards.dtype == np.float64
    np.testing.assert_allclose(batch_rewards, scalar_rewards, rtol=0, atol=0)

    np.random.seed(13)
    batched = model.sample_next_state_batch(starts, action)
    np.random.seed(13)
    looped = np.vstack([model.sample_next_state(starts[i], action) for i in range(6)])
    assert batched.dtype == np.float64
    np.testing.assert_allclose(batched, looped, atol=1e-12)


# ── Transition density semantics ───────────────────────────────────────


def test_transition_density_scores_latents_and_terminal_slot():
    """Latent flips are impossible; legitimate terminal-slot flips are not.

    Purpose: The factored point-mass rule must hold for the types while the deterministic
    contact rule governs the slot — the override this model carries exists exactly for this.

    Given: A pre-contact state one step from the hazard.
    When: Candidates with (a) the correct slot, (b) a flipped latent type, (c) a wrongly
        cleared slot are scored.
    Then: (a) is finite, (b) and (c) are -inf.

    Test type: unit
    """
    np.random.seed(0)
    model = _quiet_navigation()
    action = model.get_actions()[STRAIGHT]
    state = model.initial_state_dist().sample(1)[0]
    for _ in range(7):  # stop just outside the disc (0.6 m away)
        state = model.sample_next_state(state, action)
    assert not model.is_terminal(state)

    contact = model.sample_next_state(state, action)  # enters the disc, slot set
    assert float(model.state_schema.block(contact, EPISODE_DONE_CHANNEL)[0]) == 1.0
    finite = model.transition_log_probability(state, action, contact[np.newaxis, :])
    assert np.isfinite(finite).all()

    flipped_type = contact.copy()
    flipped_type[model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)] = 0.0
    cleared_slot = contact.copy()
    cleared_slot[model.state_schema.slice_of(EPISODE_DONE_CHANNEL)] = 0.0
    tilted_gravity = contact.copy()
    tilted_gravity[model.state_schema.slice_of("projected_gravity")] = [0.5, 0.0, -0.5]
    scores = model.transition_log_probability(
        state, action, np.vstack([flipped_type, cleared_slot, tilted_gravity])
    )
    assert np.all(np.isinf(scores) & (scores < 0))


# ── Belief updates and planning ────────────────────────────────────────


def test_belief_update_learns_the_hazard_type_near_the_hazard():
    """A particle belief over the latent type sharpens on in-zone signals.

    Purpose: Validates the model is a drop-in for WeightedParticleBelief — the pattern the old
    one-space model's POMCPOW test establishes — and that the signal actually teaches it.

    Given: A 50/50 type belief with the robot stopped inside the signal zone of a bad hazard.
    When: The belief is updated on several observations sampled from the true (bad) state.
    Then: The posterior probability of the bad type exceeds 0.9.

    Test type: integration
    """
    np.random.seed(2)
    model = HazardNavigationIsaacModel(
        hazards=[(1.0, 0.0, 0.4)],
        p_bad=0.5,
        signal_radius=1.5,
        signal_accuracy=0.9,
    )
    action = model.get_actions()[STOP]
    template = model.initial_state_dist().sample(1)[0]
    true_state = template.copy()
    true_state[model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)] = 1.0

    particles = []
    for index in range(60):
        particle = template.copy()
        particle[model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)] = float(index % 2)
        particles.append(particle)
    belief = WeightedParticleBelief(
        particles=particles, log_weights=np.log(np.ones(60) / 60), resampling=False
    )
    for _ in range(6):
        true_state = model.sample_next_state(true_state, action)
        observation = model.sample_observation(true_state, action)
        belief = belief.update(
            action=action, observation=observation, pomdp=model, state=true_state
        )
    type_slice = model.state_schema.slice_of(HAZARD_TYPE_CHANNEL)
    posterior_bad = float(
        sum(
            weight
            for particle, weight in zip(belief.particles, belief.normalized_weights)
            if particle[type_slice][0] > 0.5
        )
    )
    assert posterior_bad > 0.9


def test_pomcpow_plans_on_the_navigation_model():
    """POMCPOW selects one of the model's presets from a particle belief.

    Purpose: End-to-end drop-in check mirroring the one-space model's POMCPOW test.

    Given: A hazard navigation model and a belief drawn from its initial distribution.
    When: POMCPOW plans for one second.
    Then: The selected action is one of the presets.

    Test type: integration
    """
    np.random.seed(3)
    model = HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.5)], p_bad=0.5)
    particles = model.initial_state_dist().sample(50)
    belief = WeightedParticleBelief(
        particles=particles, log_weights=np.log(np.ones(50) / 50), resampling=True
    )
    actions = model.get_actions()
    planner = POMCPOW(
        environment=model,
        discount_factor=0.99,
        depth=8,
        exploration_constant=10.0,
        k_o=4.0,
        alpha_o=0.1,
        k_a=float(len(actions)),
        alpha_a=0.0,
        name="POMCPOW-hazard-test",
        action_sampler=DiscreteActionSampler(actions),
        time_out_in_seconds=1,
    )
    selected, _ = planner.action(belief)
    action = np.asarray(selected[0], dtype=float).reshape(-1)
    assert any(np.array_equal(action, preset) for preset in actions)


def test_reach_belief_update_runs_on_the_model():
    """The reach model supports the same particle belief update.

    Purpose: The manipulator twin of the drop-in check.

    Given: A belief over the latent presence bit.
    When: One update runs on a sampled observation.
    Then: A new belief is returned with finite weights.

    Test type: integration
    """
    np.random.seed(4)
    model = HazardReachIsaacModel(p_present=0.5)
    particles = model.initial_state_dist().sample(40)
    belief = WeightedParticleBelief(
        particles=particles, log_weights=np.log(np.ones(40) / 40), resampling=True
    )
    action = model.get_actions()[1]
    state = particles[0]
    next_state = model.sample_next_state(state, action)
    observation = model.sample_observation(next_state, action)
    updated = belief.update(action=action, observation=observation, pomdp=model, state=next_state)
    assert updated is not belief
    assert np.all(np.isfinite(updated.normalized_weights))


# ── Reach contact and severity ─────────────────────────────────────────


def test_reach_contact_penalty_scales_with_hand_speed():
    """Contact costs nothing at rest and the full severity when the hand sweeps in fast.

    Purpose: Validates the severity is the hand's displacement rate, computed where both states
    exist, and gated on the latent presence.

    Given: An obstacle centred on the hand's default-pose position, present.
    When: Rewards are computed for a resting contact, a fast sweep into contact, and the same
        sweep with the obstacle absent.
    Then: Resting adds no penalty; the sweep is penalized; absence removes the penalty.

    Test type: unit
    """
    model = HazardReachIsaacModel(p_present=1.0, collision_penalty=25.0, ee_speed_max=1.0)
    hand_home = model.hazard_reward.end_effector_position_of(
        model.initial_state_dist().sample(1)[0]
    )
    model = HazardReachIsaacModel(
        obstacle_center=tuple(hand_home), obstacle_radius=0.05, p_present=1.0,
        collision_penalty=25.0, ee_speed_max=1.0,
    )
    resting = model.initial_state_dist().sample(1)[0]
    resting[model.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)] = 1.0
    assert model.hazard_reward.contact(resting) == 1.0

    no_penalty = model.reward(resting, None, resting)  # zero displacement -> zero severity
    away = resting.copy()
    away[model.state_schema.slice_of("joint_pos")] = np.concatenate(
        [np.full(7, 0.4), np.zeros(2)]
    )
    swept = model.reward(away, None, resting)
    speed = model.hazard_reward.end_effector_speed(away, resting)
    assert speed > 0.0
    assert no_penalty - swept == pytest.approx(
        25.0 * (min(speed, 1.0)) ** 2
    )

    absent = resting.copy()
    absent[model.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)] = 0.0
    away_absent = away.copy()
    away_absent[model.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)] = 0.0
    assert model.reward(away_absent, None, absent) == pytest.approx(
        model.hazard_reward.distance_weight
        * model.hazard_reward.end_effector_distance(absent)
        + model.hazard_reward.shaping_weight
        * (
            1.0
            - np.tanh(
                model.hazard_reward.end_effector_distance(absent)
                / model.hazard_reward.shaping_std
            )
        )
    )


def test_reach_contact_terminal_flag():
    """is_contact_terminal=True makes a present-obstacle contact terminal.

    Purpose: Validates the optional terminal slot on the reach model.

    Given: An obstacle on the hand's default position, present, terminal contact enabled.
    When: A transition is sampled from a contact state.
    Then: The successor carries the slot and is terminal; with the default flag it is not.

    Test type: unit
    """
    np.random.seed(7)
    base = HazardReachIsaacModel(p_present=1.0)
    hand_home = base.hazard_reward.end_effector_position_of(
        base.initial_state_dist().sample(1)[0]
    )
    for terminal_flag in (True, False):
        model = HazardReachIsaacModel(
            obstacle_center=tuple(hand_home),
            obstacle_radius=0.08,
            p_present=1.0,
            is_contact_terminal=terminal_flag,
        )
        state = model.initial_state_dist().sample(1)[0]
        state[model.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)] = 1.0
        next_state = model.sample_next_state(state, model.get_actions()[0])
        assert model.is_terminal(next_state) is terminal_flag


# ── Metrics ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "builder",
    [
        lambda: HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.5)]),
        HazardReachIsaacModel,
    ],
    ids=["navigation", "reach"],
)
def test_step_info_contract(builder):
    """step_info emits every declared channel, draws no randomness, tolerates the terminal call.

    Purpose: The env-metrics contract in one place.

    Given: Each hazard model, a sampled transition.
    When: step_info runs on the transition and on the terminal bookkeeping call.
    Then: Declared metric channels are all emitted on a real transition; the RNG stream is
        untouched; the (state, None, None) call reports the state-derived channels.

    Test type: unit
    """
    np.random.seed(8)
    model = builder()
    state = model.initial_state_dist().sample(1)[0]
    action = model.get_actions()[0]
    next_state = model.sample_next_state(state, action)

    rng_before = np.random.get_state()
    info = model.step_info(state, action, next_state)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    assert np.array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]

    declared = {spec.channel for spec in model.get_metric_specs()}
    assert declared <= set(info)
    assert info["recorded_step"] == 1.0

    terminal_info = model.step_info(next_state, None, None)
    assert "recorded_step" in terminal_info
    assert "goal_reached" in terminal_info


def test_metric_names_include_the_success_metric():
    """Both models expose goal_reaching_rate and the episode-length metric.

    Purpose: Downstream tuning selects objectives by these names.

    Given: Both models.
    When: get_metric_names is read.
    Then: goal_reaching_rate, average_episode_length and the three episode-end rates appear.

    Test type: unit
    """
    for model in (
        HazardNavigationIsaacModel(hazards=[(2.0, 0.0, 0.5)]),
        HazardReachIsaacModel(),
    ):
        names = model.get_metric_names()
        for required in (
            "goal_reaching_rate",
            "average_episode_length",
            "ended_by_goal_rate",
            "ended_by_failure_rate",
            "ended_by_timeout_rate",
        ):
            assert required in names


# ── Constrained twins ──────────────────────────────────────────────────


def test_constrained_navigation_twin_moves_the_penalty_to_the_constraint():
    """The twin's reward carries no danger term; the constraint carries the indicator.

    Purpose: Encoding the same event in both channels would double-count it; on this model the
    reward would also shrink the range the constrained planner reads.

    Given: The twin and a bad-contact transition.
    When: Reward, reward_range and constraint_cost are compared against the unconstrained model.
    Then: The twin's reward equals the base navigation reward (no penalty), its range excludes
        the penalty, and constraint_cost is 1 exactly on bad contact.

    Test type: unit
    """
    np.random.seed(9)
    twin = ConstrainedHazardNavigationIsaacModel(
        hazards=[(2.0, 0.0, 0.5)], p_bad=1.0,
        velocity_noise_std=QUIET, position_noise_std=QUIET, heading_noise_std=QUIET,
    )
    assert twin.hazard_reward.collision_penalty == 0.0
    assert twin.reward_range[0] == pytest.approx(-twin.hazard_reward.heading_weight * np.pi)

    action = twin.get_actions()[STRAIGHT]
    state = twin.initial_state_dist().sample(1)[0]
    for _ in range(8):
        previous, state = state, twin.sample_next_state(state, action)
    assert twin.hazard_reward.bad_contact(state) == 1.0
    assert twin.constraint_cost(previous, action, state).tolist() == [1.0]
    assert twin.constraint_cost(state, action, previous).tolist() == [0.0]

    # reward at the contact state carries no penalty term
    nav_only = twin.hazard_reward.reward_rows(state[np.newaxis, :])[0]
    assert twin.reward(previous, action, state) == pytest.approx(nav_only)

    batch = twin.constraint_cost_batch(
        np.vstack([previous, previous]), action, np.vstack([state, previous])
    )
    assert batch.shape == (2, 1)
    assert batch[:, 0].tolist() == [1.0, 0.0]


def test_constrained_reach_twin_constraint_matches_contact():
    """The reach twin's constraint fires exactly on present-obstacle contact.

    Purpose: Same single-encoding rule for the manipulator.

    Given: The twin with the obstacle on the hand's start position.
    When: constraint_cost is evaluated for present and absent obstacles.
    Then: 1.0 while present and in contact, 0.0 otherwise, and the reward carries no penalty.

    Test type: unit
    """
    base = HazardReachIsaacModel(p_present=1.0)
    hand_home = base.hazard_reward.end_effector_position_of(
        base.initial_state_dist().sample(1)[0]
    )
    twin = ConstrainedHazardReachIsaacModel(
        obstacle_center=tuple(hand_home), obstacle_radius=0.05, p_present=1.0
    )
    assert twin.hazard_reward.collision_penalty == 0.0
    contact_state = twin.initial_state_dist().sample(1)[0]
    contact_state[twin.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)] = 1.0
    absent_state = contact_state.copy()
    absent_state[twin.state_schema.slice_of(OBSTACLE_PRESENCE_CHANNEL)] = 0.0
    assert twin.constraint_cost(contact_state, None, contact_state).tolist() == [1.0]
    assert twin.constraint_cost(contact_state, None, absent_state).tolist() == [0.0]
    batch = twin.constraint_cost_batch(
        np.vstack([contact_state, contact_state]),
        None,
        np.vstack([contact_state, absent_state]),
    )
    assert batch[:, 0].tolist() == [1.0, 0.0]


# ── env-qa cheapest gate ───────────────────────────────────────────────


def test_straight_line_policy_reaches_the_goal_under_zero_noise():
    """Driving straight at the goal completes the task when the hazard is benign.

    Purpose: env-qa's cheapest solvability check — a task nothing can complete is
    indistinguishable from a broken one.

    Given: Zero-noise dynamics, a benign hazard on the path.
    When: The scripted straight-line policy runs.
    Then: The goal is reached (planar distance < success_radius) within 20 steps.

    Test type: integration
    """
    np.random.seed(10)
    model = _quiet_navigation(p_bad=0.0)
    states = _rollout(model, [STRAIGHT] * 20, horizon=20)
    assert model.hazard_reward.planar_goal_distance(states[-1]) < model.success_radius
    assert len(states) - 1 <= 20


def test_random_policy_pays_more_collision_cost_than_scripted_detour():
    """A random policy is measurably worse than the scripted detour on collision cost.

    Purpose: env-qa's degeneracy check — if random does as well as an informed policy, the
    hazard is not actually shaping the task.

    Given: A certain-bad hazard on the path, non-terminal contact so cost accumulates, zero
        process noise, 10 episodes of 30 steps per policy on fixed seeds.
    When: Bad-contact steps are counted for the detour plan and for uniform random actions.
    Then: The detour pays zero; the random policy pays a clearly positive cost.

    Test type: integration
    """
    def total_contact_steps(policy_rng=None):
        total = 0.0
        for episode in range(10):
            np.random.seed(200 + episode)
            model = _quiet_navigation(is_bad_contact_terminal=False)
            actions = model.get_actions()
            state = model.initial_state_dist().sample(1)[0]
            plan = list(DETOUR_PLAN) + [STRAIGHT] * 30
            for step in range(30):
                if policy_rng is None:
                    index = plan[step]
                else:
                    index = int(policy_rng.integers(len(actions)))
                state = model.sample_next_state(state, actions[index])
                total += model.hazard_reward.bad_contact(state)
                if model.hazard_reward.planar_goal_distance(state) < model.success_radius:
                    break
        return total

    detour_cost = total_contact_steps()
    random_cost = total_contact_steps(policy_rng=np.random.default_rng(42))
    assert detour_cost == 0.0
    assert random_cost >= 5.0
