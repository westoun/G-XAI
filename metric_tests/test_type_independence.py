#!/usr/bin/env python3

"""
Evaluate whether a feature importance scoring approach has 
any bias towards the type of variable it is evaluating. 
"""

from random import uniform, random, choices, randint
import scipy
from typing import Callable, List

from gxai import FeatureType, CategoricalFeature, ContinuousFeature


def run_type_independence_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):

    MAX_CATEGORIES = 20

    MAX_RANGE = 1000
    MIN_MIDDLE_VALUE = -100
    MAX_MIDDLE_VALUE = 100

    DISTRIBUTION_COUNT = 1000
    SAMPLE_SIZE = 1000

    type_indicators = []
    importance_scores = []

    for i in range(DISTRIBUTION_COUNT):

        if random() < 0.5:
            middle_value = uniform(MIN_MIDDLE_VALUE, MAX_MIDDLE_VALUE)
            value_range = uniform(0, MAX_RANGE)

            min_value = middle_value - 0.5 * value_range
            max_value = middle_value + 0.5 * value_range

            distribution = []
            for _ in range(SAMPLE_SIZE):
                distribution.append(uniform(min_value, max_value))

            feature_type = ContinuousFeature(title=f"feature_{i}", values=distribution)
            feature_type.set_min(min_value)
            feature_type.set_max(max_value)

            importance_score = compute_importance(distribution, feature_type)

            importance_scores.append(importance_score)
            type_indicators.append(0)

        else:
            category_count = randint(2, MAX_CATEGORIES)

            categories = [j for j in range(category_count)]

            distribution = choices(categories, k=SAMPLE_SIZE)
            feature_type = CategoricalFeature(title=f"feature_{i}", values=distribution)

            importance_score = compute_importance(distribution, feature_type)

            importance_scores.append(importance_score)
            type_indicators.append(1)

    correlation = scipy.stats.pearsonr(type_indicators, importance_scores)
    correlation = correlation.statistic

    print(
        "Correlation coefficient between the type of statistical variable "
        "and its importance score: ",
        correlation,
    )
