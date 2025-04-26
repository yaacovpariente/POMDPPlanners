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
    actual_num_steps: int
    reach_terminal_state: bool


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

