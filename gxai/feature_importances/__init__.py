#!/usr/bin/env python3

from typing import List, Tuple, Any, Callable

from gxai.feature_types import FeatureType, ContinuousFeature, CategoricalFeature
from gxai.utils.matrix import transpose_values
from .contrast_entropy import compute_contrast_entropy_importance
from .wasserstein import compute_contrast_wasserstein_importance
from .gini import compute_gini_importance


def compute_feature_importance_scores(
    population: List[Tuple],
    feature_types: List[FeatureType],
    compute_importance: Callable[[List[float], FeatureType], float],
) -> List[float]:
    population = population.copy().values
    transposed_population = transpose_values(population)

    importances: List[float] = []
    for feature_values, feature_type in zip(transposed_population, feature_types):
        feature_importance = compute_importance(feature_values, feature_type)
        importances.append(feature_importance)

    importance_sum = sum(importances)
    for i in range(len(importances)):
        importances[i] = importances[i] / importance_sum

    return importances
