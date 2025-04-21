from typing import Any, NamedTuple, Union


class StepData(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    observation: Any
    reward: float


class History(NamedTuple):
    history: list[StepData]
    discount_factor: float
    average_state_sampling_time: float
    average_action_time: float
    average_observation_time: float
    average_belief_update_time: float
    average_reward_time: float


class CategoricalHyperParameter(NamedTuple):
    choices: list[Any]
    name: str


class NumericalHyperParameter(NamedTuple):
    low: Union[int, float]
    high: Union[int, float]
    name: str


class MetricValue(NamedTuple):
    name: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float


# class Simulation(ABC):
#     def __init__(
#         self,
#         environment: Environment,
#         policy: Policy,
#         initial_belief: Belief,
#         discount_factor: float
#     ):
#         assert isinstance(initial_belief, Belief)
#         assert isinstance(environment, Environment)
#         assert isinstance(policy, Policy)

#         self.environment = environment
#         self.policy = policy
#         self.initial_belief = initial_belief
#         self.discount_factor = discount_factor

#         self.average_state_sampling_time = 0.
#         self.average_action_time = 0.
#         self.average_observation_time = 0.
#         self.average_belief_update_time = 0.
#         self.terminal_states_counter = 0

#         self.step_counter = 0
#         self.return_statistics = dict()

#     def run(self, num_episodes: int, num_steps: int) -> float:
#         assert num_episodes > 0
#         assert num_steps > 0

#         returns = []
#         histories = []
#         for _ in range(num_episodes):
#             return_, history = self.run_episode(num_steps)
#             returns.append(return_)
#             histories.append(history)

#         self.return_statistics = self._compute_return_statistics(returns)
#         self.terminal_states_average = self.terminal_states_counter / num_episodes
#         self._history_analysis(histories)

#         return returns

#     @abstractmethod
#     def _compute_return_statistics(self, returns: list[float]) -> dict:
#         pass

#     def _history_analysis(self, histories: list):
#         pass

#     def run_episode(self, num_steps: int):
#         assert num_steps > 0

#         state = self.initial_belief.sample()
#         return_ = 0
#         belief = self.initial_belief

#         history = []

#         for i in range(num_steps):
#             self.step_counter += 1

#             if self.environment.is_terminal(state):
#                 self.terminal_states_counter += 1
#                 break

#             action = self._get_action(belief)
#             next_state = self._get_next_state(state, action)
#             next_observation = self._get_observation(next_state, action)
#             belief = self._update_belief(belief, action, next_observation)
#             state = next_state

#             reward = self.environment.reward(state, action)
#             return_ += self.discount_factor ** i * reward

#             history.append((state, action, next_state, next_observation, reward))

#         return return_, history

#     def _get_action(self, belief: Belief):
#         start_time = time()
#         action = self.policy.action(belief)
#         action_time = time() - start_time
#         self.average_action_time = (self.average_action_time * (self.step_counter - 1) + action_time) / self.step_counter

#         return action

#     def _get_next_state(self, state: Any, action: Any):
#         start_time = time()
#         next_state = self.environment.state_transition_model(state=state, action=action).sample()
#         state_sampling_time = time() - start_time
#         self.average_state_sampling_time = (self.average_state_sampling_time * (self.step_counter - 1) + state_sampling_time) / self.step_counter

#         return next_state

#     def _get_observation(self, next_state: Any, action: Any):
#         start_time = time()
#         observation = self.environment.observation_model(state=next_state, action=action).sample()
#         observation_time = time() - start_time
#         self.average_observation_time = (self.average_observation_time * (self.step_counter - 1) + observation_time) / self.step_counter

#         return observation

#     def _update_belief(self, belief: Belief, action: Any, observation: Any):
#         start_time = time()
#         belief = belief.update(action=action, observation=observation, pomdp=self.environment)
#         belief_update_time = time() - start_time
#         self.average_belief_update_time = (self.average_belief_update_time * (self.step_counter - 1) + belief_update_time) / self.step_counter

#         return belief


# class SafetySimulation(Simulation):
#     def __init__(
#         self,
#         environment: Environment,
#         policy: Policy,
#         initial_belief: Belief,
#         alpha: float,
#         discount_factor: float
#     ):
#         super().__init__(
#             environment=environment,
#             policy=policy,
#             initial_belief=initial_belief,
#             discount_factor=discount_factor
#         )

#         assert isinstance(alpha, float)
#         assert 1 >= alpha >= 0.

#         self.alpha = alpha

#     def _compute_return_statistics(self, returns: list[float]) -> dict:
#         return {
#             "average_return": sum(returns) / len(returns),
#             "returns_cvar": cvar_estimator(returns, self.alpha)
#         }


# class DiscreteLightDarkSimulation(SafetySimulation):
#     def __init__(
#         self,
#         environment: Environment,
#         policy: Policy,
#         initial_belief: Belief,
#         alpha: float,
#         discount_factor: float
#     ):
#         super().__init__(
#             environment=environment,
#             policy=policy,
#             initial_belief=initial_belief,
#             alpha=alpha,
#             discount_factor=discount_factor
#         )

#     def _history_analysis(self, histories: list):
#         pass
