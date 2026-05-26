# Copyright 2025 Yaacov Pariente
# SPDX-License-Identifier: MIT

from typing import NamedTuple


class MetricValue(NamedTuple):
    name: str
    value: float
    lower_confidence_bound: float
    upper_confidence_bound: float
