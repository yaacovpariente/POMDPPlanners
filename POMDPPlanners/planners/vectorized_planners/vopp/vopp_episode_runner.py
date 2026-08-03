# SPDX-License-Identifier: MIT

"""Closed-loop episode driver for the vectorized planner (VOPP / PORPP).

:class:`VOPPPlanner` only answers a single question -- ``plan(root_particles)``
returns one greedy action index -- so on its own it cannot run a POMDP
*episode*: there is no ground-truth world stepping forward, no belief filter
threading information between steps, and no record of the trajectory. This
module adds exactly that thin layer.

:class:`VOPPEpisodeRunner` drives a full closed loop entirely on-device:

#. plan an action from the current particle belief with the wrapped
   :class:`VOPPPlanner` (timed with a CUDA sync so the wall clock is real);
#. step a ground-truth world state forward and draw an observation;
#. run a sequential-importance-resampling (SIR) particle filter -- propagate
   the belief particles through the model transition, weight them by the
   model observation likelihood, and resample -- to obtain the next belief.

The world is, by default, the same :class:`VectorizedGenerativeModel` the
planner searches (a faithful "model-is-world" rollout, justified for
Continuous Light-Dark by its native-parity test). A caller with a *different*
ground-truth simulator -- a real CARLA or Isaac world -- injects it through the
optional ``world_transition`` / ``world_observation`` hooks while the belief
filter keeps using the vectorized model.

Every quantity the definition-of-done analysis needs is recorded per step:
the true state, the belief particle cloud, the chosen action, the immediate
reward, the planning wall-clock time, and the planner's tree metrics.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from POMDPPlanners.core.belief.batched_particle_belief import BatchedParticleBelief
from POMDPPlanners.core.environment.vectorized_generative_model import (
    VectorizedGenerativeModel,
)
from POMDPPlanners.planners.vectorized_planners.vopp.vopp import VOPPPlanner

WorldTransition = Callable[[Tensor, Tensor], Tensor]
WorldObservation = Callable[[Tensor, Tensor], Tensor]
WorldStepInfo = Callable[[], Dict[str, float]]
WorldTerminal = Callable[[], bool]


@dataclass
class VOPPEpisodeResult:
    """Recorded trajectory and per-step statistics of one VOPP episode.

    Attributes:
        states: ``[ds]`` true states, one per visited step plus the final state.
        beliefs: Belief particle clouds ``[num_particles, ds]``, one per step.
        action_indices: Greedy action index chosen at each step.
        rewards: Immediate reward collected at each step.
        plan_times: Wall-clock seconds spent inside each :meth:`plan` call.
        root_visit_counts: Planner root visit count (forward-search particle
            simulations) backing each step's action.
        reached_terminal_state: Whether the episode ended because a terminal
            state was reached, rather than by exhausting ``max_steps``. This is
            *generic* termination, not success: with a ``world_terminal`` hook it
            is whatever the world calls terminal, and IsaacLab tasks typically
            terminate on **failure** (a robot falling, a cart leaving its bounds)
            or timeout. Read task success from the ``success`` channel in
            :attr:`step_infos`, never from this flag.
        num_steps: Number of executed actions.
        step_infos: Per-step auxiliary measurements reported by the world, one
            mapping per executed action. Mirrors
            :attr:`~POMDPPlanners.core.simulation.history.StepData.info` so the
            same metric aggregator serves both this runner and the standard
            episode loop. Empty when no ``world_step_info`` hook was supplied.
    """

    states: List[Tensor] = field(default_factory=list)
    beliefs: List[Tensor] = field(default_factory=list)
    action_indices: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    plan_times: List[float] = field(default_factory=list)
    root_visit_counts: List[int] = field(default_factory=list)
    step_infos: List[Dict[str, float]] = field(default_factory=list)
    reached_terminal_state: bool = False
    num_steps: int = 0

    @property
    def reached_goal(self) -> bool:
        """Deprecated attribute alias for :attr:`reached_terminal_state`.

        The old name claimed more than the flag delivers: a terminal state is
        usually a *failure* in IsaacLab tasks, not a goal. Reading and writing
        this attribute still works, but it is **not** accepted as a constructor
        keyword — ``VOPPEpisodeResult(reached_goal=...)`` is a breaking change and
        must become ``VOPPEpisodeResult(reached_terminal_state=...)``. The only
        construction site in this repository is the no-argument one in
        :meth:`VOPPEpisodeRunner.run_episode`.
        """
        return self.reached_terminal_state

    @reached_goal.setter
    def reached_goal(self, value: bool) -> None:
        self.reached_terminal_state = bool(value)

    @property
    def total_plan_time(self) -> float:
        """Total wall-clock seconds spent planning across the episode."""
        return float(sum(self.plan_times))

    @property
    def total_root_visits(self) -> int:
        """Total forward-search particle simulations across the episode."""
        return int(sum(self.root_visit_counts))


class VOPPEpisodeRunner:
    """Runs closed-loop POMDP episodes with a :class:`VOPPPlanner`.

    The runner owns the interaction loop but no planning policy of its own: the
    planner decides actions, the vectorized model supplies the dynamics and the
    belief filter, and an optional pair of world hooks overrides the
    ground-truth transition / observation when a real simulator is the world.

    Attributes:
        num_belief_particles: Size of the particle belief carried between steps.
        max_steps: Maximum number of actions per episode.

    Example:
        >>> import torch
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_pomdp import (
        ...     ContinuousLightDarkPOMDP,
        ... )
        >>> from POMDPPlanners.environments.light_dark_pomdp.continuous_light_dark_vectorized_model import (
        ...     ContinuousLightDarkVectorizedModel,
        ... )
        >>> from POMDPPlanners.planners.vectorized_planners import (
        ...     VOPPEpisodeRunner,
        ...     VOPPPlanner,
        ... )
        >>> _ = torch.manual_seed(0)
        >>> env = ContinuousLightDarkPOMDP(discount_factor=0.95, is_obstacle_hit_terminal=False)
        >>> model = ContinuousLightDarkVectorizedModel(env, device=torch.device("cpu"))
        >>> planner = VOPPPlanner(
        ...     model, num_actions=model.num_actions, num_particles=128,
        ...     max_depth=6, num_planning_iterations=8,
        ... )
        >>> runner = VOPPEpisodeRunner(planner, model, num_belief_particles=256, max_steps=20)
        >>> initial = torch.tensor([[0.0, 5.0]])
        >>> result = runner.run_episode(initial)
        >>> result.num_steps >= 1
        True
    """

    def __init__(
        self,
        planner: VOPPPlanner,
        model: VectorizedGenerativeModel,
        *,
        num_belief_particles: int = 1000,
        max_steps: int = 50,
        world_transition: Optional[WorldTransition] = None,
        world_observation: Optional[WorldObservation] = None,
        world_step_info: Optional[WorldStepInfo] = None,
        world_terminal: Optional[WorldTerminal] = None,
    ) -> None:
        """Initialise the runner.

        Args:
            planner: The vectorized planner queried once per step.
            model: Vectorized generative model providing the belief filter and
                the default ground-truth dynamics.
            num_belief_particles: Number of particles in the carried belief.
            max_steps: Maximum actions taken before the episode is cut off.
            world_transition: Optional ``(state, action) -> next_state`` hook
                overriding the ground-truth transition (e.g. a real simulator).
            world_observation: Optional ``(next_state, action) -> observation``
                hook overriding the ground-truth observation model.
            world_step_info: Optional ``() -> {channel: value}`` hook reporting
                the world's auxiliary measurements for the step just taken,
                recorded into :attr:`VOPPEpisodeResult.step_infos`. Called after
                the world transition, so it can serve values the world cached
                during its own step.
            world_terminal: Optional ``() -> bool`` hook deciding episode
                termination from the *world* rather than the model. Supply it
                whenever the surrogate model cannot represent termination — with
                a model whose ``terminal_mask`` is constantly false, episodes
                otherwise always run the full ``max_steps`` and never report
                having reached the goal.

        Raises:
            ValueError: If ``num_belief_particles`` or ``max_steps`` is not
                positive.
        """
        if num_belief_particles <= 0:
            raise ValueError("num_belief_particles must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._planner = planner
        self._model = model
        self.num_belief_particles = num_belief_particles
        self.max_steps = max_steps
        self.device = model.device
        self._world_transition = world_transition or model.sample_next_states
        self._world_observation = world_observation or model.sample_observations
        self._world_step_info = world_step_info
        self._world_terminal = world_terminal

    def run_episode(
        self, initial_state: Tensor, initial_particles: Optional[Tensor] = None
    ) -> VOPPEpisodeResult:
        """Run one closed-loop episode and return its recorded trajectory.

        Args:
            initial_state: ``[1, ds]`` (or ``[ds]``) ground-truth start state.
            initial_particles: Optional ``[num_particles, ds]`` initial belief;
                defaults to the start state replicated across the particle set.

        Returns:
            A :class:`VOPPEpisodeResult` holding the states, beliefs, actions,
            rewards, planning times, and root visit counts of the episode.

        Raises:
            ValueError: If ``initial_state`` does not describe a single state.
        """
        state = self._validate_initial_state(initial_state)
        particles = self._initial_particles(state, initial_particles)
        result = VOPPEpisodeResult()
        for _ in range(self.max_steps):
            action_index, plan_time = self._timed_plan(particles)
            next_state, reward, observation = self._step_world(state, action_index)
            self._record_step(result, state, particles, action_index, reward, plan_time)
            state = next_state
            if self._is_terminal(state):
                result.reached_terminal_state = True
                break
            particles = self._filter_belief(particles, action_index, observation)
        result.states.append(state.squeeze(0))
        result.num_steps = len(result.action_indices)
        return result

    def _validate_initial_state(self, initial_state: Tensor) -> Tensor:
        state = initial_state.reshape(1, -1) if initial_state.dim() == 1 else initial_state
        if state.dim() != 2 or state.shape[0] != 1:
            raise ValueError("initial_state must describe a single [1, ds] state")
        return state.to(self.device)

    def _initial_particles(self, state: Tensor, initial_particles: Optional[Tensor]) -> Tensor:
        if initial_particles is not None:
            return initial_particles.to(self.device)
        return state.repeat(self.num_belief_particles, 1)

    def _timed_plan(self, particles: Tensor) -> tuple[int, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        action_index = self._planner.plan(particles)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return action_index, time.perf_counter() - start

    def _step_world(self, state: Tensor, action_index: int) -> tuple[Tensor, float, Tensor]:
        action = torch.tensor([action_index], dtype=torch.int64, device=self.device)
        next_state = self._world_transition(state, action)
        reward = float(self._model.rewards(state, action, next_state).item())
        observation = self._world_observation(next_state, action)
        return next_state, reward, observation

    def _is_terminal(self, state: Tensor) -> bool:
        # The world's own verdict wins when available: a surrogate model fitted
        # from rollouts generally cannot represent termination at all.
        if self._world_terminal is not None:
            return bool(self._world_terminal())
        return bool(self._model.terminal_mask(state).any())

    def _record_step(
        self,
        result: VOPPEpisodeResult,
        state: Tensor,
        particles: Tensor,
        action_index: int,
        reward: float,
        plan_time: float,
    ) -> None:
        result.states.append(state.squeeze(0))
        result.beliefs.append(particles.clone())
        result.action_indices.append(action_index)
        result.rewards.append(reward)
        result.plan_times.append(plan_time)
        result.root_visit_counts.append(self._root_visit_count())
        if self._world_step_info is not None:
            result.step_infos.append(dict(self._world_step_info()))

    def _root_visit_count(self) -> int:
        for variable in self._planner.tree_metrics():
            if variable.name == "root_visit_count":
                return int(variable.value)
        return 0

    def _filter_belief(self, particles: Tensor, action_index: int, observation: Tensor) -> Tensor:
        """Propagate, weight, and resample the belief (a SIR particle filter).

        Delegates to a batch-of-one :class:`BatchedParticleBelief`: propagate
        through the model transition, reweight by the observation likelihood
        (with the all-``-inf`` degeneracy falling back to uniform weights),
        and resample back to an unweighted ``[num_particles, ds]`` cloud.
        """
        belief = BatchedParticleBelief(
            particles=particles.unsqueeze(0),
            log_weights=torch.zeros(1, particles.shape[0], device=self.device),
            model=self._model,
        )
        actions = torch.tensor([action_index], dtype=torch.int64, device=self.device)
        posterior = belief.propagate(actions).reweight(actions, observation.reshape(1, -1))
        return posterior.sample_states(particles.shape[0]).squeeze(0)
