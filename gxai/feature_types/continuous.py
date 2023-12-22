#!/usr/bin/env python3

from statistics import mean, stdev
from typing import List, Tuple, Any

from .interface import FeatureType


class ContinuousFeature(FeatureType):
    """
    Class to represent continuous (numeric) features in a dataset.

    This class extends from the FeatureType interface and provides implementation
    specific to continuous features. It stores the title of the feature, its minimum,
    maximum, mean, and standard deviation values.

    Methods:
        set_min: Sets the minimum value of the feature.
        set_max: Sets the maximum value of the feature.
        __str__: Returns a formatted string representation of the ContinuousFeature.
        __eq__: Overloaded equality operator for comparing instances of ContinuousFeature with types.
    """
    def __init__(self, title: str, values: List[float]) -> None:
        self._title = title
        self._min = min(values)
        self._max = max(values)
        self._mean = mean(values)
        self._stdev = stdev(values)

    @property
    def title(self) -> str:
        return self._title

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    def set_min(self, value: float) -> "ContinuousFeature":
        self._min = value
        return self

    def set_max(self, value: float) -> "ContinuousFeature":
        self._max = value
        return self

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def stdev(self) -> float:
        return self._stdev

    def __str__(self) -> str:
        return f"""
            ContinuousFeature '{self.title}':
                range: [{self.min}, {self.max}]
                mean: {self.mean}
                stdev: {self.stdev}"""

    def __eq__(self, comparison_value: type) -> bool:
        return comparison_value == type(self)
