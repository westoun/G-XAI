#!/usr/bin/env python3

from typing import List, Tuple, Any


class FeatureType:
    """
    Base class (Interface) for feature types.

    This class serves as an interface for specific types of features such as categorical 
    and continuous. It contains methods that must be implemented by subclasses.
    
    Methods:
        __init__: Initialises a new instance of FeatureType.
        __str__: Returns a string representation of the FeatureType.
        __eq__: Overloaded equality operator for comparing instances of FeatureType with types.
    """
    def __init__(self, title: str, values: List[Any]) -> None:
        raise NotImplementedError()

    @property
    def title(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def __eq__(self, comparison_value: type) -> bool:
        raise NotImplementedError()
