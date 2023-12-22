#!/usr/bin/env python3

from typing import List, Tuple, Any

from .feature_types import FeatureType, ContinuousFeature, CategoricalFeature
from .utils.matrix import transpose_values
from .utils.statistics import (
    compute_categorical_randomness,
    compute_continuous_randomness,
)


def compute_feature_importance_scores(
    population: List[Tuple], feature_types: List[FeatureType]
) -> List[float]:
    population = population.copy().values
    transposed_population = transpose_values(population)

    importances: List[float] = []
    for feature_values, feature_type in zip(transposed_population, feature_types):
        if feature_type == CategoricalFeature:
            divergence_from_randomness = 1 - compute_categorical_randomness(
                feature_values, feature_type
            )
            importances.append(divergence_from_randomness)

        elif feature_type == ContinuousFeature:
            divergence_from_randomness = 1 - compute_continuous_randomness(
                feature_values, feature_type
            )
            importances.append(divergence_from_randomness)

    importance_sum = sum(importances)
    for i in range(len(importances)):
        importances[i] = importances[i] / importance_sum

    return importances
