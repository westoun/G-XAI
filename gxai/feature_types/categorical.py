#!/usr/bin/env python3

from typing import List, Tuple, Any

from .interface import FeatureType


class CategoricalFeature(FeatureType):
    """
    Class to represent categorical features in a dataset.

    This class extends from the FeatureType interface and provides implementation
    specific to categorical features. It stores the title of the feature and a list of 
    its unique values.

    Methods:
        __str__: Returns a formatted string representation of the CategoricalFeature.
        __eq__: Overloaded equality operator for comparing instances of CategoricalFeature with types.
    """
    def __init__(self, title: str, values: List[Any]) -> None:
        self._title = title
        self._unique_values = list(set(values))

    @property
    def title(self) -> str:
        return self._title

    @property
    def unique_values(self) -> List:
        return self._unique_values

    def __str__(self) -> str:
        return f"""
            CategoricalFeature '{self.title}':
                unique values: {self._unique_values}"""

    def __eq__(self, comparison_value: type) -> bool:
        return comparison_value == type(self)
