#!/usr/bin/env python3

import math
import numpy as np
from statistics import mean
from typing import List, Tuple

from gxai.feature_types import FeatureType, CategoricalFeature, ContinuousFeature


def compute_gini_importance(feature_values: List, feature_type: FeatureType) -> float:
    if feature_type == CategoricalFeature:
        divergence_from_randomness = compute_categorical_gini(
            feature_values, feature_type
        )
        return divergence_from_randomness
    elif feature_type == ContinuousFeature:
        divergence_from_randomness = compute_continuous_gini(
            feature_values, feature_type
        )
        return divergence_from_randomness
    else:
        raise NotImplementedError()


def compute_continuous_gini(
    feature_values: List, feature_type: ContinuousFeature
) -> float:
    return compute_gini(feature_values)


def compute_categorical_gini(
    feature_values: List, feature_type: CategoricalFeature
) -> float:
    unique_values = feature_type.unique_values
    probabilities = [
        feature_values.count(unique_value) / len(feature_values)
        for unique_value in unique_values
    ]
    return compute_gini(probabilities)


# def gini(x):
#     # Taken from https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
#     # (Warning: This is a concise implementation, but it is O(n**2)
#     # in time and memory, where n = len(x).  *Don't* pass in huge
#     # samples!)

#     # Mean absolute difference
#     mad = np.abs(np.subtract.outer(x, x)).mean()
#     # Relative mean absolute difference
#     rmad = mad/np.mean(x)
#     # Gini coefficient
#     g = 0.5 * rmad
#     return g


def compute_gini(x: List[float], w=None):
    # Taken from https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (
            cumxw[-1] * cumw[-1]
        )
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
