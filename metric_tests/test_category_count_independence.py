#!/usr/bin/env python3

"""
Evaluate whether a feature importance scoring approach has 
any bias towards the amount of categories of a categorical 
variable.
"""

from random import choices, randint
import scipy
from typing import Callable, List

from gxai import FeatureType, CategoricalFeature, ContinuousFeature


def run_category_count_independence_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):
    MAX_CATEGORIES = 20
    DISTRIBUTION_COUNT = 1000
    SAMPLE_SIZE = 1000

    category_counts = []
    importance_scores = []

    for i in range(DISTRIBUTION_COUNT):
        category_count = randint(2, MAX_CATEGORIES)

        categories = [j for j in range(category_count)]

        distribution = choices(categories, k=SAMPLE_SIZE)
        feature_type = CategoricalFeature(title=f"feature_{i}", values=distribution)

        importance_score = compute_importance(distribution, feature_type)

        importance_scores.append(importance_score)
        category_counts.append(category_count)

    correlation = scipy.stats.pearsonr(category_counts, importance_scores)
    correlation = correlation.statistic

    print(
        "Correlation coefficient between the amount of categories and the "
        "importance score of randomly generated categorical variable: ",
        correlation,
    )
