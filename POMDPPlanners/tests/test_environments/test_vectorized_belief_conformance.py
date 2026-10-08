# SPDX-License-Identifier: MIT

"""Vectorized beliefs agree with the scalar environment they duplicate.

A vectorized particle belief re-implements its environment's transition and
observation model over a whole particle array. That duplicate is only correct
if it describes the same model as the scalar
:class:`~POMDPPlanners.core.environment.Environment` -- and nothing about the
two being written separately keeps them in step. Each environment has its own
parity test, but each picks its own states, its own tolerances and its own idea
of which parts to check, and a new vectorized belief gets no check until
someone writes one.

This file checks every vectorized belief in
:data:`~POMDPPlanners.tests.test_environments._vectorized_registry.BELIEF_SPECS`
the same way, through public interfaces only:

* the updater's ``batch_observation_log_likelihood`` against the scalar
  ``observation_log_probability`` on the same particles -- exact, because the
  likelihood is a deterministic function;
* the updater's ``batch_transition`` against a loop over the scalar
  ``sample_next_state`` -- by distribution, because the two paths rarely draw
  their randomness in the same order. Where the registry says they do, the
  values are compared exactly under a shared seed as well;
* a full ``update`` of the vectorized belief against ``WeightedParticleBelief``
  -- exactly under a shared seed where the belief's update is plain
  reweighting, and on the posterior distribution for every belief.

States are reached by rolling the scalar environment forward under random
actions, so the checks see states past the initial distribution without
knowing anything about what a state means. The only per-environment knowledge
is in the registry: how an observation is handed to the updater, and which
paths share an RNG stream.

Rollouts stop at the first terminal state, so the checks above never hand the
updater a finished particle. A belief does: once some of its particles have
finished, a planner that tests one sampled particle for termination keeps
updating the rest. The last group of checks repeats the kernel comparisons on
terminal states, alone and mixed into a batch of live ones, and compares the
belief-level terminal test across the two belief types. The terminal states
come from ``_terminal_states``.

The registry's coverage tests at the bottom fail when a vectorized updater or
belief class exists that no entry exercises.
"""

from typing import Any, List, Tuple

import numpy as np
import pytest

from POMDPPlanners.core.belief.belief_utils import is_terminal_belief
from POMDPPlanners.core.belief.particle_beliefs import WeightedParticleBelief
from POMDPPlanners.core.belief.vectorized_weighted_particle_belief import (
    VectorizedWeightedParticleBelief,
)
from POMDPPlanners.core.environment import DiscreteActionsEnvironment, Environment
from POMDPPlanners.tests.test_core.test_belief.belief_equivalence_utils import (
    assert_update_particles_match,
    assert_update_weights_match,
)
from POMDPPlanners.tests.test_core.test_belief.vectorized_updater_test_utils import (
    assert_batch_transition_matches_loop,
)
from POMDPPlanners.tests.test_environments._env_config_variants import base_id_of
from POMDPPlanners.tests.test_environments._sample_distribution_checks import (
    assert_same_distribution,
)
from POMDPPlanners.tests.test_environments._terminal_states import (
    ENVS_WITHOUT_TERMINAL_STATES,
    copy_state,
    terminal_states,
)
from POMDPPlanners.tests.test_environments._vectorized_registry import (
    BELIEF_CLASS_EXCLUSIONS,
    BELIEF_SPECS,
    BELIEF_UPDATE_DISAGREEMENTS,
    ENVS_WITHOUT_VECTORIZED_BELIEF,
    OBSERVATION_HANDLING_DISAGREEMENTS,
    OBSERVATION_LIKELIHOOD_DISAGREEMENTS,
    POSTERIOR_DISTRIBUTION_DISAGREEMENTS,
    TERMINAL_BELIEF_DISAGREEMENTS,
    TERMINAL_OBSERVATION_LIKELIHOOD_DISAGREEMENTS,
    TERMINAL_TRANSITION_DISAGREEMENTS,
    TRANSITION_DISTRIBUTION_DISAGREEMENTS,
    TRANSITION_SHARED_SEED_DISAGREEMENTS,
    BeliefSpec,
    discover_vectorized_classes,
)
from POMDPPlanners.tests.test_environments.test_env_api_conformance import (
    ENV_BUILDERS,
    _random_action,
    _seed_all,
    _trajectory_key,
)
from POMDPPlanners.utils.belief_factory import BeliefType, create_environment_belief

# Rollouts that supply source states. Short, so a terminating env still yields
# a few; several seeds, so the states differ.
_ROLLOUT_SEEDS = (0, 1, 2)
_ROLLOUT_STEPS = 6

# Draws per (state, action) in the transition distribution check. Large enough
# that a misplaced probability of a few percent is detected, small enough that
# the scalar loop stays well under a second per environment.
_TRANSITION_DRAWS = 2000

# A log-likelihood below this is "impossible" for the comparison. Implementations
# floor impossible readings differently (-inf, log(1e-300) = -690.8, -1e18); all
# of them mean the same thing to a particle filter. The cut sits above the
# log(1e-300) floor and far below any real likelihood seen here: a reading of a
# hundred noisy cells legitimately scores around -40.
_NEGLIGIBLE_LOG_LIKELIHOOD = -500.0

# Particles in the posterior check. The posterior of a particle filter is
# itself random, and on rare branches (a fire spreading to one cell) far more
# random than its effective sample size suggests -- with 3000 particles one
# cell's posterior mean ranged over 0.015-0.064 between reruns. So the filters
# are large, and the two posteriors are compared on only a tenth of the
# reference's effective sample size in draws.
_POSTERIOR_PARTICLES = 10000

# Discrete action sets up to this size are checked action by action; larger
# joint action sets (Firefighting's 25 or 125) get this many, drawn once.
_MAX_ACTIONS_CHECKED = 8
_CONTINUOUS_ACTIONS_CHECKED = 3
_POSTERIOR_DRAWS_PER_ESS = 0.1
_MIN_POSTERIOR_DRAWS = 30

# Actions and draws for each terminal state ``terminal_states`` supplies. Fewer
# draws than the live-state transition check: most environments hold a terminal
# state still, and a constant column is compared exactly.
_TERMINAL_ACTIONS_CHECKED = 4
_TERMINAL_DRAWS = 500


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def _params(disagreements: dict, only: Any = None) -> List[Any]:
    """One param per spec, ``xfail(strict=True)`` where a disagreement is recorded."""
    params = []
    for spec in BELIEF_SPECS:
        if only is not None and not only(spec):
            continue
        marks = []
        if spec.env_id in disagreements:
            reason, raises = disagreements[spec.env_id]
            marks.append(pytest.mark.xfail(strict=True, reason=reason, raises=raises))
        params.append(pytest.param(spec, id=spec.env_id, marks=marks))
    return params


# ---------------------------------------------------------------------------
# Helpers -- interface-only
# ---------------------------------------------------------------------------


def _weighted_particle_belief(particles: List[Any], log_weights: np.ndarray) -> Any:
    return WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=False)


def _vectorized_belief(env: Environment, n_particles: int) -> VectorizedWeightedParticleBelief:
    belief = create_environment_belief(
        env, belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=n_particles
    )
    assert isinstance(belief, VectorizedWeightedParticleBelief)
    return belief


def _reachable_states(env: Environment) -> List[Any]:
    """Non-terminal states from the initial distribution and short random rollouts."""
    states: List[Any] = []
    for seed in _ROLLOUT_SEEDS:
        _seed_all(seed)
        action_rng = np.random.default_rng(seed)
        state = env.initial_state_dist().sample()[0]
        for _ in range(_ROLLOUT_STEPS):
            if env.is_terminal(state):
                break
            states.append(state)
            state = env.sample_next_state(state=state, action=_random_action(env, action_rng))
    assert states, f"{type(env).__name__}: every rollout started in a terminal state"
    return states


def _as_row(state: Any, dtype: Any) -> np.ndarray:
    return np.asarray(state, dtype=dtype).ravel()


def _actions_to_check(env: Environment, rng: np.random.Generator) -> List[Any]:
    """Every discrete action when there are few; otherwise a fixed random subset.

    A transition bug often lives in one action's branch (one direction's wall
    clip, one tool's effect), so a small action set is covered completely.
    """
    if isinstance(env, DiscreteActionsEnvironment):
        actions = list(env.get_actions())
        if len(actions) <= _MAX_ACTIONS_CHECKED:
            return actions
        picks = rng.choice(len(actions), size=_MAX_ACTIONS_CHECKED, replace=False)
        return [actions[int(i)] for i in sorted(picks)]
    return [_random_action(env, rng) for _ in range(_CONTINUOUS_ACTIONS_CHECKED)]


def _source_pairs(env: Environment, states: List[Any], n_states: int) -> List[Tuple[Any, Any]]:
    """(state, action) pairs: ``n_states`` reachable states, each with every checked action."""
    rng = np.random.default_rng(1234)
    picks = np.linspace(0, len(states) - 1, num=n_states).round().astype(int)
    actions = _actions_to_check(env, rng)
    return [(states[i], action) for i in picks for action in actions]


def _scalar_log_likelihoods(
    env: Environment, next_states: List[Any], action: Any, observation: Any
) -> np.ndarray:
    # The per-state scalar path, not observation_log_probability_per_state:
    # several envs route that batch hook through the vectorized updater itself,
    # which would make the comparison circular.
    return np.array(
        [
            float(
                env.observation_log_probability(
                    next_state=s, action=action, observations=[observation]
                )[0]
            )
            for s in next_states
        ]
    )


def _assert_log_likelihoods_agree(vectorized: np.ndarray, scalar: np.ndarray, context: str) -> None:
    vectorized = np.asarray(vectorized, dtype=np.float64)
    assert (
        vectorized.shape == scalar.shape
    ), f"{context}: batch log-likelihood shape {vectorized.shape} != {scalar.shape}"
    assert not np.any(np.isnan(vectorized)), f"{context}: batch log-likelihood has NaN"
    relevant = (vectorized > _NEGLIGIBLE_LOG_LIKELIHOOD) | (scalar > _NEGLIGIBLE_LOG_LIKELIHOOD)
    assert np.any(relevant), f"{context}: the observation is impossible under every particle"
    np.testing.assert_allclose(
        vectorized[relevant],
        scalar[relevant],
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"{context}: batch_observation_log_likelihood disagrees with the scalar env",
    )


def _weighted_draws(rows: np.ndarray, weights: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rows[rng.choice(len(rows), size=n_draws, p=weights)]


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(1.0 / np.sum(weights**2))


# ---------------------------------------------------------------------------
# Updater kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", _params(OBSERVATION_LIKELIHOOD_DISAGREEMENTS))
def test_batch_observation_log_likelihood_matches_scalar_env(spec: BeliefSpec) -> None:
    """The updater scores a reading exactly as the scalar environment does.

    Purpose: The particle weights are the belief. If the batch likelihood
        differs from the environment's observation model, the vectorized
        belief converges to a different posterior than the one the planner is
        modelling, and nothing at runtime shows it.

    Given: Reachable states, an action at each, a set of candidate next states
        drawn from the scalar env, and an observation drawn from the scalar env
        for one of them (so at least one candidate explains it).
    When: The updater's ``batch_observation_log_likelihood`` and a loop over the
        scalar ``observation_log_probability`` score that observation against
        every candidate.
    Then: The two agree to 1e-6 wherever either is above
        ``_NEGLIGIBLE_LOG_LIKELIHOOD``; below it, both only need to call the
        reading impossible, since implementations floor that differently.

    Test type: integration
    """
    env = spec.build_env()
    belief = _vectorized_belief(env, n_particles=8)
    updater = belief.updater
    dtype = belief.particles.dtype
    states = _reachable_states(env)

    for state, action in _source_pairs(env, states, n_states=3):
        _seed_all(7)
        candidates = [env.sample_next_state(state=s, action=action) for s in states[:24]]
        true_next = env.sample_next_state(state=state, action=action)
        observation = env.sample_observation(next_state=true_next, action=action)
        candidates.insert(0, true_next)

        rows = np.stack([_as_row(s, dtype) for s in candidates])
        vectorized = updater.batch_observation_log_likelihood(
            rows, action, spec.observation_for_updater(observation)
        )
        scalar = _scalar_log_likelihoods(env, candidates, action, observation)
        _assert_log_likelihoods_agree(
            vectorized, scalar, f"{spec.env_id} action={action!r} observation={observation!r}"
        )


@pytest.mark.parametrize("spec", _params(TRANSITION_DISTRIBUTION_DISAGREEMENTS))
def test_batch_transition_distribution_matches_scalar_env(spec: BeliefSpec) -> None:
    """``batch_transition`` samples the scalar environment's transition law.

    Purpose: The batch path moves every particle on every belief update. If it
        samples a different law than ``sample_next_state`` -- a missing noise
        term, a wrong clip, a different slip rate -- the belief tracks a world
        the environment does not simulate.

    Given: Several reachable states, each with one action, each state
        replicated ``_TRANSITION_DRAWS`` times.
    When: The replicated block goes through ``batch_transition`` once, and
        through the scalar ``sample_next_state`` once per row, from
        independent seeds.
    Then: The two sample matrices have the same distribution: components that
        are deterministic agree exactly, the rest pass a chi-square or
        Kolmogorov-Smirnov test at a fixed family-wise level.

    Test type: integration
    """
    env = spec.build_env()
    belief = _vectorized_belief(env, n_particles=8)
    updater = belief.updater
    dtype = belief.particles.dtype
    states = _reachable_states(env)

    for state, action in _source_pairs(env, states, n_states=2):
        block = np.repeat(_as_row(state, dtype)[None, :], _TRANSITION_DRAWS, axis=0)
        _seed_all(11)
        vectorized = np.asarray(updater.batch_transition(block.copy(), action), dtype=np.float64)
        _seed_all(12)
        scalar = np.stack(
            [
                _as_row(
                    env.sample_next_state(state=np.array(state, copy=True), action=action),
                    np.float64,
                )
                for _ in range(_TRANSITION_DRAWS)
            ]
        )
        try:
            assert_same_distribution(vectorized, scalar)
        except AssertionError as error:
            raise AssertionError(f"{spec.env_id} action={action!r}: {error}") from None


@pytest.mark.parametrize(
    "spec",
    _params(TRANSITION_SHARED_SEED_DISAGREEMENTS, only=lambda spec: spec.transition_shares_rng),
)
def test_batch_transition_matches_scalar_env_under_shared_seed(spec: BeliefSpec) -> None:
    """Where both paths draw the same random numbers, they produce the same states.

    Purpose: A distribution test cannot see a small, systematic error -- a
        per-particle offset of 1e-3 passes any two-sample test at these sizes.
        The registry marks the envs whose batch path consumes the seeded RNGs
        in the same order as the scalar loop; for those, the two must agree
        value for value.

    Given: A block of distinct reachable states and one action per check, with
        Python's, numpy's and every native RNG reseeded before each path.
    When: ``batch_transition`` runs on the block and ``sample_next_state`` runs
        on each row in order.
    Then: The results agree to 1e-8.

    Test type: integration
    """
    env = spec.build_env()
    belief = _vectorized_belief(env, n_particles=8)
    updater = belief.updater
    dtype = belief.particles.dtype
    states = _reachable_states(env)
    rows = np.stack([_as_row(s, dtype) for s in states])

    for _, action in _source_pairs(env, states, n_states=2):
        assert_batch_transition_matches_loop(
            updater,
            rows.copy(),
            action,
            lambda row, a: _as_row(
                env.sample_next_state(state=np.array(row, copy=True), action=a), dtype
            ),
            atol=1e-8,
            seed=21,
            seed_fn=_seed_all,
            err_msg=f"{spec.env_id} action={action!r}",
        )


# ---------------------------------------------------------------------------
# Full belief update
# ---------------------------------------------------------------------------


def _aligned_pair(
    spec: BeliefSpec, env: Environment, prior: VectorizedWeightedParticleBelief
) -> Tuple[Any, VectorizedWeightedParticleBelief]:
    """The reference belief and a vectorized belief over the same particles.

    The vectorized side is the factory's own class, rebuilt without
    resampling so a seeded update can be compared particle by particle.
    """
    states = _reachable_states(env)
    rng = np.random.default_rng(5)
    picks = rng.integers(0, len(states), size=48)
    rows = np.stack([_as_row(states[i], prior.particles.dtype) for i in picks])
    log_weights = np.log(rng.dirichlet(np.ones(len(rows))))
    vectorized = type(prior)(
        particles=rows.copy(),
        log_weights=log_weights.copy(),
        updater=prior.updater,
        resampling=False,
    )
    scalar = (spec.reference_belief or _weighted_particle_belief)(
        [states[i] for i in picks], log_weights.copy()
    )
    return scalar, vectorized


@pytest.mark.parametrize(
    "spec",
    _params(
        BELIEF_UPDATE_DISAGREEMENTS,
        only=lambda spec: spec.update_matches_reference and spec.transition_shares_rng,
    ),
)
def test_vectorized_belief_update_matches_weighted_particle_belief(spec: BeliefSpec) -> None:
    """A seeded vectorized update reproduces the reference belief's update.

    Purpose: The kernels can each be right and the belief still wrong -- the
        observation handed to the updater in a different encoding, the action
        converted twice, weights accumulated in a different space. This runs
        the whole ``update`` through the public interface on both sides.

    Given: The same 48 reachable states and the same random weights in the
        reference belief (``WeightedParticleBelief`` unless the registry names
        the environment's own scalar filter) and in the environment's
        vectorized belief, resampling off, and an observation the scalar env
        produced.
    When: Both are updated with that action and observation under the same
        seed (Python, numpy and every native RNG).
    Then: The next particles agree and the normalized weights agree to 1e-6.

    Test type: integration
    """
    env = spec.build_env()
    prior = _vectorized_belief(env, n_particles=8)
    for step_seed in (0, 1):
        scalar, vectorized = _aligned_pair(spec, env, prior)
        rng = np.random.default_rng(100 + step_seed)
        action = _random_action(env, rng)
        _seed_all(step_seed)
        true_state = scalar.particles[int(rng.integers(len(scalar.particles)))]
        true_next = env.sample_next_state(state=true_state, action=action)
        observation = env.sample_observation(next_state=true_next, action=action)

        assert_update_particles_match(
            scalar,
            vectorized,
            action,
            observation,
            env,
            atol=1e-8,
            seed=31,
            seed_fn=_seed_all,
            particle_to_array=lambda p: _as_row(p, prior.particles.dtype),
        )
        assert_update_weights_match(
            scalar, vectorized, action, observation, env, atol=1e-6, seed=31, seed_fn=_seed_all
        )


def _is_none_reading(observation: Any) -> bool:
    """Whether ``observation`` is the light-dark models' "None" (no reading) label."""
    return isinstance(observation, str) and observation == "None"


def _bayes_filter_posterior(
    env: Environment, particles: np.ndarray, log_weights: np.ndarray, action: Any, observation: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """One step of the textbook particle filter, through the scalar interface only.

    This is ``WeightedParticleBelief.update`` without its ``1e-10`` likelihood
    floor. The floor is harmless for low-dimensional readings, but a reading
    of a hundred noisy cells has a real likelihood near ``e^-40``, far below
    it -- the floor then outweighs the evidence and the "posterior" is close
    to the prediction. Writing the filter out keeps the reference exact.

    Returns:
        ``(next_particles, normalized_weights)``.
    """
    next_states = [
        env.sample_next_state(state=np.array(p, copy=True), action=action) for p in particles
    ]
    log_likelihoods = _scalar_log_likelihoods(env, next_states, action, observation)
    posterior = np.asarray(log_weights, dtype=np.float64) + log_likelihoods
    assert np.any(np.isfinite(posterior)), "no particle explains the reading"
    weights = np.exp(posterior - np.max(posterior))
    rows = np.stack([np.asarray(s, dtype=np.float64).ravel() for s in next_states])
    return rows, weights / weights.sum()


@pytest.mark.parametrize("spec", _params(POSTERIOR_DISTRIBUTION_DISAGREEMENTS))
def test_vectorized_belief_posterior_matches_scalar_bayes_filter(spec: BeliefSpec) -> None:
    """The vectorized belief reaches the posterior the scalar environment implies.

    Purpose: Several vectorized beliefs deliberately do more than reweight --
        they redraw from an exact posterior, condition on the noiseless part of
        a reading, stratify resampling over a static hidden variable, or
        reinvigorate. Those are valid only if they still estimate the Bayesian
        posterior that the scalar environment's transition and observation
        model define. This covers every registered belief, including the ones
        whose update cannot be compared draw for draw.

    Given: The factory's vectorized prior with ``_POSTERIOR_PARTICLES``
        particles, and an action and observation the scalar env produced from
        a state the prior holds. The reference is a particle filter over the
        same particles and weights, written out from the scalar interface
        (see ``_bayes_filter_posterior``), or the environment's own scalar
        belief where the registry names one.
    When: The vectorized belief and the reference are updated with that
        action and observation.
    Then: Draws from the two posteriors have the same distribution. The draw
        count is a tenth of the reference's effective sample size (capped at
        1000, raised to ``_MIN_POSTERIOR_DRAWS`` while that is at most half the
        ESS), so the test asks no more precision of a particle approximation
        than it has. With fewer draws than that the check could not fail, so
        it skips and says why rather than passing.

    Test type: integration
    """
    env = spec.build_env()
    _seed_all(41)
    prior = _vectorized_belief(env, n_particles=_POSTERIOR_PARTICLES)
    rng = np.random.default_rng(43)
    action = _random_action(env, rng)
    true_state = np.array(prior.particles[int(rng.integers(prior.n_particles))], copy=True)
    if spec.prior_holds_true_state:
        # See BeliefSpec.prior_holds_true_state: both filters get the same
        # prior, so neither implementation is favoured.
        prior.particles[: _POSTERIOR_PARTICLES // 2] = true_state
    prior_particles = np.array(prior.particles, copy=True)
    prior_log_weights = np.array(prior.log_weights, dtype=np.float64, copy=True)

    _seed_all(44)
    true_next = env.sample_next_state(state=true_state, action=action)
    observation = env.sample_observation(next_state=true_next, action=action)

    if spec.env_id in OBSERVATION_HANDLING_DISAGREEMENTS and _is_none_reading(observation):
        # Known crash; it is pinned by the strict xfail on the observation
        # handling test. The draw is platform-dependent, so it cannot be a
        # static strict xfail here. Assert the crash still happens, so a fix
        # turns this into a failure rather than a silent xfail.
        reason, raises = OBSERVATION_HANDLING_DISAGREEMENTS[spec.env_id]
        with pytest.raises(raises):
            prior.update(action=action, observation=observation, pomdp=env)
        pytest.xfail(reason)

    _seed_all(45)
    vectorized_posterior = prior.update(action=action, observation=observation, pomdp=env)
    _seed_all(46)
    if spec.reference_belief is None:
        reference_label = "scalar Bayes filter"
        reference_rows, reference_weights = _bayes_filter_posterior(
            env, prior_particles, prior_log_weights, action, observation
        )
    else:
        reference_posterior: Any = spec.reference_belief(
            [row for row in prior_particles], prior_log_weights
        ).update(action=action, observation=observation, pomdp=env)
        reference_label = type(reference_posterior).__name__
        reference_rows = np.stack(
            [np.asarray(p, dtype=np.float64).ravel() for p in reference_posterior.particles]
        )
        reference_weights = np.asarray(reference_posterior.normalized_weights, dtype=np.float64)

    reference_ess = _effective_sample_size(reference_weights)
    n_draws = int(
        min(
            1000,
            max(
                _POSTERIOR_DRAWS_PER_ESS * reference_ess,
                min(_MIN_POSTERIOR_DRAWS, reference_ess / 2),
            ),
        )
    )
    # Below this many draws a two-sample test over many columns cannot reach
    # alpha whatever the samples are, so the check would pass by construction.
    # Say so instead of passing. With the pinned seeds no registered entry
    # lands here; a skip after a seed or config change means the entry needs
    # prior_holds_true_state, or the reading is too sharp for this check.
    if n_draws < _MIN_POSTERIOR_DRAWS:
        pytest.skip(
            f"{spec.env_id}: reference posterior ESS {reference_ess:.1f} supports only "
            f"{n_draws} draws, too few for the check to be able to fail"
        )
    try:
        assert_same_distribution(
            _weighted_draws(
                np.stack(
                    [
                        np.asarray(p, dtype=np.float64).ravel()
                        for p in vectorized_posterior.particles
                    ]
                ),
                np.asarray(vectorized_posterior.normalized_weights, dtype=np.float64),
                n_draws,
                seed=47,
            ),
            _weighted_draws(reference_rows, reference_weights, n_draws, seed=48),
            label_b=reference_label,
        )
    except AssertionError as error:
        raise AssertionError(
            f"{spec.env_id} action={action!r} observation={observation!r} "
            f"({n_draws} draws): {error}"
        ) from None


@pytest.mark.parametrize("spec", _params(OBSERVATION_HANDLING_DISAGREEMENTS))
def test_vectorized_belief_accepts_every_observation_the_env_emits(spec: BeliefSpec) -> None:
    """The vectorized belief can be updated with any reading the environment produces.

    Purpose: A simulation hands the belief whatever ``sample_observation``
        returned. If the belief's ``update`` cannot convert one kind of reading
        -- a label, a sentinel, a variable-length tuple -- the episode crashes
        the first time that reading comes up, which a test using one
        hand-picked observation never sees.

    Given: The factory's vectorized belief, and every action and observation
        along several random rollouts of the scalar env.
    When: The belief is updated with each pair in turn.
    Then: Every update returns a belief whose normalized weights are finite
        and sum to one.

    Test type: integration
    """
    env = spec.build_env()
    for seed in _ROLLOUT_SEEDS + (3, 4, 5):
        _seed_all(seed)
        action_rng = np.random.default_rng(seed)
        belief = _vectorized_belief(env, n_particles=64)
        state = env.initial_state_dist().sample()[0]
        for _ in range(2 * _ROLLOUT_STEPS):
            if env.is_terminal(state):
                break
            action = _random_action(env, action_rng)
            state, observation, _ = env.sample_next_step(state=state, action=action)
            try:
                belief = belief.update(action=action, observation=observation, pomdp=env)
            except Exception as error:  # pylint: disable=broad-exception-caught
                # Re-raise the same type with the reading that caused it, so an
                # xfail(raises=...) entry still matches.
                context = (
                    f"{spec.env_id}: update(action={action!r}, observation={observation!r}) "
                    f"raised {type(error).__name__}: {error}"
                )
                try:
                    wrapped: Exception = type(error)(context)
                except Exception:  # pylint: disable=broad-exception-caught
                    wrapped = AssertionError(context)
                raise wrapped from error
            weights = np.asarray(belief.normalized_weights, dtype=np.float64)
            assert np.all(np.isfinite(weights)) and np.isclose(
                weights.sum(), 1.0
            ), f"{spec.env_id}: weights not a distribution after observation {observation!r}"


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


def _terminal_states_of(spec: BeliefSpec, env: Environment) -> List[Any]:
    """Terminal states of ``env``; skips an environment declared to have none."""
    states = terminal_states(spec.env_id, env)
    if not states:
        # test_env_terminal_conformance.py fails for any environment that
        # supplies no terminal state without being declared.
        pytest.skip(
            f"{spec.env_id} has no terminal state: "
            f"{ENVS_WITHOUT_TERMINAL_STATES.get(base_id_of(spec.env_id), 'none found')}"
        )
    return states


def _terminal_actions(env: Environment) -> List[Any]:
    return _actions_to_check(env, np.random.default_rng(1234))[:_TERMINAL_ACTIONS_CHECKED]


@pytest.mark.parametrize("spec", _params(TERMINAL_TRANSITION_DISAGREEMENTS))
def test_batch_transition_matches_scalar_env_on_terminal_states(spec: BeliefSpec) -> None:
    """``batch_transition`` does to a finished particle what the scalar env does.

    Purpose: Every belief update moves every particle, finished ones included.
        Whatever the scalar environment does to a terminal state -- hold it,
        move it, clear its flag -- the updater must do the same, or the
        finished part of the belief drifts from the model the planner searches
        on. A terminal rule applied to the whole array rather than row by row
        gets a mixed batch wrong, so the batch here mixes terminal states with
        a live one.

    Given: Terminal states of the scalar env and one reachable live state,
        interleaved into one block with each state repeated
        ``_TERMINAL_DRAWS`` times.
    When: The block goes through ``batch_transition`` once and through the
        scalar ``sample_next_state`` row by row, from independent seeds.
    Then: For every source state, terminal or live, the two samples have the
        same distribution.

    Test type: integration
    """
    env = spec.build_env()
    belief = _vectorized_belief(env, n_particles=8)
    updater = belief.updater
    dtype = belief.particles.dtype
    sources = _terminal_states_of(spec, env) + _reachable_states(env)[:1]
    rows = np.stack([_as_row(state, dtype) for state in sources])
    block = np.tile(rows, (_TERMINAL_DRAWS, 1))
    repeated = [sources[i % len(sources)] for i in range(len(block))]

    for action in _terminal_actions(env):
        _seed_all(11)
        vectorized = np.asarray(updater.batch_transition(block.copy(), action), dtype=np.float64)
        _seed_all(12)
        scalar = np.stack(
            [
                _as_row(env.sample_next_state(state=copy_state(state), action=action), np.float64)
                for state in repeated
            ]
        )
        for index, source in enumerate(sources):
            try:
                assert_same_distribution(
                    vectorized[index :: len(sources)], scalar[index :: len(sources)]
                )
            except AssertionError as error:
                kind = "terminal" if env.is_terminal(source) else "live"
                raise AssertionError(
                    f"{spec.env_id} action={action!r} {kind} state={source!r}: {error}"
                ) from None


@pytest.mark.parametrize("spec", _params(TERMINAL_OBSERVATION_LIKELIHOOD_DISAGREEMENTS))
def test_batch_observation_log_likelihood_matches_scalar_env_on_terminal_states(
    spec: BeliefSpec,
) -> None:
    """The updater scores a reading against a finished particle as the scalar env does.

    Purpose: After the transition every particle is reweighted, finished ones
        included. An updater that gives a terminal particle another likelihood
        than the environment's observation model -- zero where the model
        allows the reading, or the reverse -- shifts the weight between the
        finished and the live part of the belief.

    Given: Terminal states and reachable live states as candidate next
        states, an action, and an observation the scalar env emits from a
        terminal state.
    When: ``batch_observation_log_likelihood`` and a loop over the scalar
        ``observation_log_probability`` score that observation against every
        candidate.
    Then: The two agree to 1e-6 wherever either is above
        ``_NEGLIGIBLE_LOG_LIKELIHOOD``.

    Test type: integration
    """
    env = spec.build_env()
    belief = _vectorized_belief(env, n_particles=8)
    updater = belief.updater
    dtype = belief.particles.dtype
    terminal = _terminal_states_of(spec, env)
    candidates = terminal + _reachable_states(env)[:8]
    rows = np.stack([_as_row(state, dtype) for state in candidates])

    for action in _terminal_actions(env):
        for observed_from in terminal:
            _seed_all(7)
            observation = env.sample_observation(
                next_state=copy_state(observed_from), action=action
            )
            vectorized = updater.batch_observation_log_likelihood(
                rows.copy(), action, spec.observation_for_updater(observation)
            )
            scalar = _scalar_log_likelihoods(env, candidates, action, observation)
            _assert_log_likelihoods_agree(
                vectorized,
                scalar,
                f"{spec.env_id} action={action!r} observation={observation!r} "
                f"emitted from terminal state {observed_from!r}",
            )


@pytest.mark.parametrize("spec", _params(TERMINAL_BELIEF_DISAGREEMENTS))
def test_vectorized_belief_is_terminal_exactly_when_the_scalar_belief_is(spec: BeliefSpec) -> None:
    """A vectorized belief and a scalar one over the same particles end together.

    Purpose: The episode runner and the planners stop on ``is_terminal_belief``.
        It reads a vectorized belief's particles out of its array and a scalar
        belief's from its list, so a particle that changes on the way into the
        array -- a float truncated into an integer array, a flag lost in a
        cast -- can be terminal in one belief and live in the other. The same
        episode would then end or not depending on the belief type.

    Given: Three particle sets -- all terminal, terminal and live mixed, all
        live -- each held by the factory's vectorized belief class and by a
        ``WeightedParticleBelief``.
    When: ``is_terminal_belief`` is asked about each belief.
    Then: The two belief types agree, and the answer is true only for the
        all-terminal set.

    Test type: integration
    """
    env = spec.build_env()
    prior = _vectorized_belief(env, n_particles=8)
    terminal = _terminal_states_of(spec, env)
    live = _reachable_states(env)[:2]

    for label, particles, expected in (
        # Each set twice over: a belief of one particle has a single log-weight
        # of zero, which the belief constructors reject.
        ("all terminal", 2 * terminal, True),
        ("mixed", terminal + live, False),
        ("all live", 2 * live, False),
    ):
        log_weights = np.log(np.full(len(particles), 1.0 / len(particles)))
        vectorized = type(prior)(
            particles=np.stack([_as_row(state, prior.particles.dtype) for state in particles]),
            log_weights=log_weights.copy(),
            updater=prior.updater,
            resampling=False,
        )
        scalar = _weighted_particle_belief(list(particles), log_weights.copy())
        vectorized_answer = is_terminal_belief(vectorized, env)
        scalar_answer = is_terminal_belief(scalar, env)
        assert vectorized_answer == scalar_answer == expected, (
            f"{spec.env_id} ({label} particles): is_terminal_belief is {vectorized_answer} "
            f"for the vectorized belief and {scalar_answer} for the scalar one; "
            f"expected {expected}"
        )


# ---------------------------------------------------------------------------
# Belief factory contract, for every environment
# ---------------------------------------------------------------------------

# Particles requested from the factory in the contract checks below.
_FACTORY_PARTICLES = 16

_BELIEF_CLASS_BY_ENV = {spec.env_id: spec.belief_class for spec in BELIEF_SPECS}


def _all_env_params() -> List[Any]:
    return [pytest.param(env_id, builder, id=env_id) for env_id, builder in ENV_BUILDERS]


def _assert_particle_belief_shape(env_id: str, env: Environment, belief: Any) -> None:
    """The belief holds the requested particles, normalised weights and env-shaped samples."""
    assert (
        len(belief.particles) == _FACTORY_PARTICLES
    ), f"{env_id}: asked for {_FACTORY_PARTICLES} particles, got {len(belief.particles)}"
    weights = np.asarray(belief.normalized_weights, dtype=np.float64)
    assert weights.shape == (_FACTORY_PARTICLES,)
    assert np.all(np.isfinite(weights)) and np.isclose(weights.sum(), 1.0)
    sample = belief.sample()
    reference = env.initial_state_dist().sample()[0]
    assert np.shape(sample) == np.shape(reference), (
        f"{env_id}: a belief sample has shape {np.shape(sample)}, an initial state "
        f"{np.shape(reference)}"
    )


@pytest.mark.parametrize("env_id,env_builder", _all_env_params())
def test_default_belief_is_the_registered_one(env_id: str, env_builder: Any) -> None:
    """With no belief type, the factory builds the environment's registered belief.

    Purpose: Simulations call ``create_environment_belief(env)`` without a
        type, so the default is what actually runs. The checks above always
        ask for ``VECTORIZED_PARTICLE`` explicitly; a factory default that
        drifted back to the slow scalar filter would pass them all. Each
        environment's own factory test checked its default by hand.

    Given: Every registered environment configuration.
    When: The factory builds a belief with no ``belief_type``.
    Then: It is the registered vectorized belief class for an environment in
        ``BELIEF_SPECS``, and a ``WeightedParticleBelief`` for one declared
        without a vectorized belief, with the requested particle count.

    Test type: unit
    """
    env = env_builder()
    belief = create_environment_belief(env, n_particles=_FACTORY_PARTICLES)
    expected = _BELIEF_CLASS_BY_ENV.get(env_id, WeightedParticleBelief.__name__)
    assert type(belief).__name__ == expected
    _assert_particle_belief_shape(env_id, env, belief)


@pytest.mark.parametrize("env_id,env_builder", _all_env_params())
def test_particle_belief_type_builds_a_scalar_particle_belief(
    env_id: str, env_builder: Any
) -> None:
    """``BeliefType.PARTICLE`` is available for every environment.

    Purpose: The scalar ``WeightedParticleBelief`` is the reference every
        vectorized belief is compared against, and the fallback when a
        vectorized one cannot run a configuration. Each environment's factory
        test checked that it can still be requested.

    Given: Every registered environment configuration.
    When: The factory builds a ``PARTICLE`` belief.
    Then: It is a ``WeightedParticleBelief`` -- not a vectorized one -- with
        the requested particle count, normalised weights, and samples shaped
        like an initial state; its particles are initial states.

    Test type: unit
    """
    env = env_builder()
    belief = create_environment_belief(
        env, belief_type=BeliefType.PARTICLE, n_particles=_FACTORY_PARTICLES
    )
    assert isinstance(belief, WeightedParticleBelief)
    assert not isinstance(belief, VectorizedWeightedParticleBelief)
    _assert_particle_belief_shape(env_id, env, belief)
    reference_type = type(_trajectory_key(env.initial_state_dist().sample()[0]))
    assert all(isinstance(_trajectory_key(p), reference_type) for p in belief.particles)


@pytest.mark.parametrize("env_id,env_builder", _all_env_params())
def test_unsupported_belief_type_is_rejected(env_id: str, env_builder: Any) -> None:
    """Asking for a belief type an environment does not support raises ``ValueError``.

    Purpose: A factory that silently returns some other belief for a type it
        does not support makes an experiment labelled "Gaussian mixture" run
        a particle filter. No environment supports ``GAUSSIAN_MIXTURE``
        today, so it is the probe; each factory test checked its own env.

    Given: Every registered environment configuration.
    When: The factory is asked for ``BeliefType.GAUSSIAN_MIXTURE``.
    Then: It raises ``ValueError``.

    Test type: unit
    """
    with pytest.raises(ValueError):
        create_environment_belief(
            env_builder(), belief_type=BeliefType.GAUSSIAN_MIXTURE, n_particles=4
        )


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [pytest.param(s, id=s.env_id) for s in BELIEF_SPECS])
def test_belief_factory_returns_the_registered_classes(spec: BeliefSpec) -> None:
    """The registry names the classes a simulation actually uses.

    Purpose: Every check above runs whatever the belief factory returns. The
        registry pins the class names so the coverage test can tell which
        classes are exercised; if the factory starts returning another class,
        that coverage claim is false and must fail here.

    Given: The registry entry's environment.
    When: ``create_environment_belief`` builds its default vectorized belief.
    Then: The belief and its updater are exactly the registered classes.

    Test type: unit
    """
    belief = _vectorized_belief(spec.build_env(), n_particles=4)
    assert type(belief).__name__ == spec.belief_class
    assert type(belief.updater).__name__ == spec.updater_class


@pytest.mark.parametrize("spec", [pytest.param(s, id=s.env_id) for s in BELIEF_SPECS])
def test_updater_config_id_is_stable_across_identical_constructions(spec: BeliefSpec) -> None:
    """Two updaters built from one configuration share one ``config_id``.

    Purpose: The updater's ``config_id`` is hashed into the belief's, and the
        belief's into the cache key of every simulation that uses it. If it
        picks up anything instance-specific -- an object id, a cached array's
        address -- every run gets a fresh key and the cache never hits.

        The converse is not checked here: a variant need not change the
        updater's id, because a switch that only selects a reward model leaves
        the updater's two kernels as they were.

    Given: Two environments built from the registry entry's builder.
    When: Each one's default vectorized belief is built and its updater's
        ``config_id`` read.
    Then: The two ids are equal strings.

    Test type: unit
    """
    first = _vectorized_belief(spec.build_env(), n_particles=4).updater.config_id
    second = _vectorized_belief(spec.build_env(), n_particles=4).updater.config_id
    assert isinstance(first, str) and first == second, (
        f"{spec.env_id}: the updater's config_id is not a pure function of the "
        f"configuration: {first!r} != {second!r}"
    )


def test_every_vectorized_belief_class_is_registered_or_excluded() -> None:
    """No vectorized updater or belief class escapes this suite silently.

    Purpose: A new vectorized belief gets no parity coverage until someone
        registers it, and nothing fails when they forget. The source tree is
        scanned instead, so a new class fails here until it is either
        registered or excluded with a stated reason.

    Given: Every updater and vectorized belief class defined under
        ``POMDPPlanners/environments``, found by parsing source.
    When: Each is looked up in the registry and the exclusion list.
    Then: Each is in exactly one, and no exclusion names a class that no
        longer exists.

    Test type: unit
    """
    discovered = discover_vectorized_classes()
    found = discovered.updaters | discovered.beliefs
    registered = {spec.updater_class for spec in BELIEF_SPECS} | {
        spec.belief_class for spec in BELIEF_SPECS
    }
    registered.discard("VectorizedWeightedParticleBelief")

    unregistered = sorted(found - registered - set(BELIEF_CLASS_EXCLUSIONS))
    assert not unregistered, (
        f"Vectorized belief classes with no conformance coverage: {unregistered}. "
        "Add a BeliefSpec in _vectorized_registry.py, or add the class to "
        "BELIEF_CLASS_EXCLUSIONS with the reason it needs none."
    )
    stale = sorted(set(BELIEF_CLASS_EXCLUSIONS) - found)
    assert not stale, f"BELIEF_CLASS_EXCLUSIONS names classes that no longer exist: {stale}"
    both = sorted(set(BELIEF_CLASS_EXCLUSIONS) & registered)
    assert not both, f"Classes both registered and excluded: {both}"


def test_every_environment_is_registered_or_declared_without_vectorized_belief() -> None:
    """Every environment in the API conformance suite is accounted for here.

    Purpose: An environment that gains a vectorized belief but no registry
        entry is invisible to this file. Cross-checking against
        ``ENV_BUILDERS`` and the belief factory catches it.

    Given: Every environment in ``ENV_BUILDERS``.
    When: Each is looked up in ``BELIEF_SPECS`` and in
        ``ENVS_WITHOUT_VECTORIZED_BELIEF``.
    Then: Each is in exactly one, and an environment declared without a
        vectorized belief really has none in the factory.

    Test type: unit
    """
    registered = {spec.env_id for spec in BELIEF_SPECS}
    for env_id, builder in ENV_BUILDERS:
        in_registry = env_id in registered
        # A variant is the same class as its pinned configuration, so the
        # declaration on the pinned id covers it.
        declared_without = base_id_of(env_id) in ENVS_WITHOUT_VECTORIZED_BELIEF
        assert in_registry != declared_without, (
            f"{env_id} must be in exactly one of BELIEF_SPECS and " "ENVS_WITHOUT_VECTORIZED_BELIEF"
        )
        if declared_without:
            with pytest.raises(ValueError):
                create_environment_belief(
                    builder(), belief_type=BeliefType.VECTORIZED_PARTICLE, n_particles=4
                )
