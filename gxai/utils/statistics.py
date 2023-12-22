#!/usr/bin/env python3

import math
import numpy as np
from statistics import mean
from typing import List, Tuple

from gxai.feature_types import FeatureType, CategoricalFeature, ContinuousFeature
from .matrix import transpose_values



def compute_randomness(
    population: List[Tuple], feature_types: List[FeatureType]
) -> float:
    """
    Compute the randomness of the entire population.

    Args:
        population: A list of tuples representing the entire population.
        feature_types: A list of FeatureType instances corresponding to the features of the population.

    Returns:
        The computed randomness of the population as a float.

    Raises:
        NotImplementedError: If an unsupported feature type is encountered.
    """
    transposed_population = transpose_values(population)

    feature_randomnesses: List[float] = []
    for feature_values, feature_type in zip(transposed_population, feature_types):
        if feature_type == CategoricalFeature:
            feature_randomness = compute_categorical_randomness(
                feature_values, feature_type
            )
            feature_randomnesses.append(feature_randomness)

        elif feature_type == ContinuousFeature:
            feature_randomness = compute_continuous_randomness(
                feature_values, feature_type
            )
            feature_randomnesses.append(feature_randomness)

        else:
            raise NotImplementedError()

    population_randomness = mean(feature_randomnesses)
    return population_randomness


def compute_continuous_randomness(
    feature_values: List, feature_type: ContinuousFeature
) -> float:
    """
    Compute the randomness of a continuous feature.

    Args:
        feature_values: A list of feature values.
        feature_type: An instance of ContinuousFeature representing the feature.

    Returns:
        The computed randomness of the continuous feature as a float.
    """
    bucket_count = len(feature_values)
    min_value, max_value = feature_type.min, feature_type.max
    bucket_width = (max_value - min_value) / bucket_count

    if bucket_width == 0:
        return 0

    uniform_probabilities = [1 / bucket_count] * bucket_count
    uniform_entropy = compute_entropy(uniform_probabilities)

    feature_values = sorted(feature_values)

    actual_probabilities = []
    j = 0
    for i in range(bucket_count):
        actual_probabilities.append(0)

        bucket_min = min_value + bucket_width * i
        if i == 0:  # account for edge case with point at left boundary
            epsilon = 0.0000001
            bucket_min = bucket_min - epsilon

        bucket_max = min_value + bucket_width * (i + 1)

        for feature_value in feature_values[j:]:
            if feature_value > bucket_min and feature_value <= bucket_max:
                actual_probabilities[i] += 1 / len(feature_values)
                j += 1
            else:
                break

    actual_entropy = compute_entropy(actual_probabilities)

    return actual_entropy / uniform_entropy


def compute_categorical_randomness(
    feature_values: List, feature_type: CategoricalFeature
) -> float:
    """
    Compute the randomness of a categorical feature.

    Args:
        feature_values: A list of feature values.
        feature_type: An instance of CategoricalFeature representing the feature.

    Returns:
        The computed randomness of the categorical feature as a float.
    """
    unique_values = feature_type.unique_values

    if len(unique_values) == 1:
        return 0

    uniform_probabilities = [1 / len(unique_values)] * len(unique_values)
    uniform_entropy = compute_entropy(uniform_probabilities)

    actual_probabilities = [
        feature_values.count(unique_value) / len(feature_values)
        for unique_value in unique_values
    ]
    actual_entropy = compute_entropy(actual_probabilities)

    return actual_entropy / uniform_entropy


def compute_entropy(probabilities: List[float]) -> float:
    """
    Compute the entropy given a list of probabilities.

    Args:
        probabilities: A list of probabilities.

    Returns:
        The computed entropy as a float.
    """
    entropy = 0

    for probability in probabilities:
        if probability == 0:
            continue

        entropy += -probability * math.log(probability)

    return entropy
