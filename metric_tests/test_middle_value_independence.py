#!/usr/bin/env python3

"""
Evaluate whether a feature importance scoring approach has 
any bias towards the middle value of a continuously distributed 
variable.
"""

from random import uniform
import scipy
from typing import Callable, List

from gxai import FeatureType, CategoricalFeature, ContinuousFeature


def run_middle_value_independence_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):

    VALUE_RANGE = 1000
    MIN_MIDDLE_VALUE = -100
    MAX_MIDDLE_VALUE = 100

    DISTRIBUTION_COUNT = 1000
    SAMPLE_SIZE = 1000

    middle_values = []
    importance_scores = []

    for i in range(DISTRIBUTION_COUNT):
        middle_value = uniform(MIN_MIDDLE_VALUE, MAX_MIDDLE_VALUE)

        min_value = middle_value - 0.5 * VALUE_RANGE
        max_value = middle_value + 0.5 * VALUE_RANGE

        distribution = []
        for _ in range(SAMPLE_SIZE):
            distribution.append(uniform(min_value, max_value))

        feature_type = ContinuousFeature(title=f"feature_{i}", values=distribution)
        feature_type.set_min(min_value)
        feature_type.set_max(max_value)

        importance_score = compute_importance(distribution, feature_type)

        importance_scores.append(importance_score)
        middle_values.append(middle_value)

    correlation = scipy.stats.pearsonr(middle_values, importance_scores)
    correlation = correlation.statistic

    print(
        "Correlation coefficient between the middle value and the "
        "importance score of randomly generated continuous variable: ",
        correlation,
    )
