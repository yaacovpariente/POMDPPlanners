from typing import Any, NamedTuple, Union

class CategoricalHyperParameter(NamedTuple):
    choices: list[Any]
    name: str


class NumericalHyperParameter(NamedTuple):
    low: Union[int, float]
    high: Union[int, float]
    name: str


HyperParameterFeatures = Union[CategoricalHyperParameter, NumericalHyperParameter]

# TODO: add hyper parameter tuning simulation interface.
