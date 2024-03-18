#!/usr/bin/env python3

from scipy.spatial import distance
from statistics import mean
from typing import List, Tuple

from gxai.feature_types import FeatureType, CategoricalFeature, ContinuousFeature


def compute_contrast_crowding_importance(
    feature_values: List, feature_type: FeatureType
) -> float:
    if feature_type == CategoricalFeature:
        divergence_from_randomness = compute_categorical_contrast_crowding(
            feature_values, feature_type
        )
        return divergence_from_randomness
    elif feature_type == ContinuousFeature:
        divergence_from_randomness = compute_continuous_contrast_crowding(
            feature_values, feature_type
        )
        return divergence_from_randomness
    else:
        raise NotImplementedError()


def compute_continuous_contrast_crowding(
    feature_values: List, feature_type: ContinuousFeature
) -> float:
    feature_values.sort()

    equal_distance = (feature_type.max - feature_type.min) / (len(feature_values) + 1)

    crowding_distances = []
    for i, current_value in enumerate(feature_values):
        # edge cases
        if i == 0:
            crowding_distances.append(
                min(
                    current_value - feature_type.min,
                    feature_values[i + 1] - current_value,
                )
            )
        elif i == len(feature_values) - 1:
            crowding_distances.append(
                min(
                    current_value - feature_values[i - 1],
                    feature_type.max - current_value,
                )
            )
        else:
            crowding_distances.append(
                min(
                    current_value - feature_values[i - 1],
                    feature_values[i + 1] - current_value,
                )
            )

    mean_crowding_distance = mean(crowding_distances)
    return 1 - mean_crowding_distance / equal_distance


def compute_categorical_contrast_crowding(
    feature_values: List, feature_type: CategoricalFeature
) -> float:
    unique_values = feature_type.unique_values

    actual_probabilities = [
        feature_values.count(unique_value) / len(feature_values)
        for unique_value in unique_values
    ]
    uniform_probability = 1 / len(unique_values)

    uniform_probabilities = [
        uniform_probability for _ in unique_values
    ]

    extreme_probabilities = [0 for _ in unique_values]
    extreme_probabilities[0] = 1

    actual_distance = distance.euclidean(actual_probabilities, uniform_probabilities)
    extreme_distance = distance.euclidean(extreme_probabilities, uniform_probabilities)

    return actual_distance / extreme_distance
