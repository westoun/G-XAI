#!/usr/bin/env python3

import math
import numpy as np
from scipy.stats import wasserstein_distance
from statistics import mean
from typing import List, Tuple

from gxai.feature_types import FeatureType, CategoricalFeature, ContinuousFeature


def compute_contrast_wasserstein_importance(
    feature_values: List, feature_type: FeatureType
) -> float:
    if feature_type == CategoricalFeature:
        divergence_from_randomness = compute_categorical_contrast_wasserstein(
            feature_values, feature_type
        )
        return divergence_from_randomness
    elif feature_type == ContinuousFeature:
        divergence_from_randomness = compute_continuous_contrast_wasserstein(
            feature_values, feature_type
        )
        return divergence_from_randomness
    else:
        raise NotImplementedError()


def compute_continuous_contrast_wasserstein(
    feature_values: List, feature_type: ContinuousFeature
) -> float:
    equal_value_distance = (feature_type.max - feature_type.min) / len(feature_values)
    equally_distributed_values = [
        feature_type.min + i * equal_value_distance for i in range(len(feature_values))
    ]

    # Theoretically, it should not matter whether feature_type.max or
    # feature_type.min is used here.
    extreme_feature_values = [feature_type.max for i in range(len(feature_values))]

    actual_distance = wasserstein_distance(feature_values, equally_distributed_values)
    extreme_case_distance = wasserstein_distance(
        extreme_feature_values, equally_distributed_values
    )

    return actual_distance / extreme_case_distance


def compute_categorical_contrast_wasserstein(
    feature_values: List, feature_type: CategoricalFeature
) -> float:
    unique_values = feature_type.unique_values

    actual_probabilities = [
        feature_values.count(unique_value) / len(feature_values)
        for unique_value in unique_values
    ]
    uniform_probability = 1 / len(unique_values)

    actual_distance = 0
    for probability in actual_probabilities:
        if probability > uniform_probability:
            actual_distance += probability - uniform_probability

    # Case where only one unique value is encountered
    extreme_case_distance = 1 - uniform_probability
    return actual_distance / extreme_case_distance
