#!/usr/bin/env python3

"""
Evaluate whether a feature importance scoring approach has 
any bias towards the range of a continuously distributed variable
"""

from random import uniform
import scipy
from typing import Callable, List

from gxai import FeatureType, CategoricalFeature, ContinuousFeature


def run_range_independence_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):

    MAX_RANGE = 1000
    DISTRIBUTION_COUNT = 1000
    SAMPLE_SIZE = 1000

    ranges = []
    importance_scores = []

    for i in range(DISTRIBUTION_COUNT):
        value_range = uniform(0, MAX_RANGE)

        min_value = -0.5 * value_range
        max_value = 0.5 * value_range

        distribution = []
        for _ in range(SAMPLE_SIZE):
            distribution.append(uniform(min_value, max_value))

        feature_type = ContinuousFeature(title=f"feature_{i}", values=distribution)
        feature_type.set_min(min_value)
        feature_type.set_max(max_value)

        importance_score = compute_importance(distribution, feature_type)

        importance_scores.append(importance_score)
        ranges.append(value_range)

    correlation = scipy.stats.pearsonr(ranges, importance_scores)
    correlation = correlation.statistic

    print(
        "Correlation coefficient between the range of values and the "
        "importance score of randomly generated continuous variable: ",
        correlation,
    )
