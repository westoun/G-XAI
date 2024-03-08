#!/usr/bin/env python3

"""
Evaluate whether both variable types behave as expected
in the edge case situations of perfect uniform distribution
and extreme one-sidedness.
"""

from random import choices, randint, uniform
from statistics import mean
from typing import Callable, List

from gxai import FeatureType, CategoricalFeature, ContinuousFeature


def run_edge_case_behavior_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):
    run_categorical_one_sided_case_behavior_test(compute_importance)
    run_categorical_perfectly_uniform_case_behavior_test(compute_importance)
    run_continuous_one_sided_case_behavior_test(compute_importance)
    run_continuous_perfectly_uniform_case_behavior_test(compute_importance)


def run_categorical_perfectly_uniform_case_behavior_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):
    MAX_CATEGORIES = 20
    DISTRIBUTION_COUNT = 100
    SAMPLE_SIZE = 1000

    importance_scores = []

    for i in range(DISTRIBUTION_COUNT):
        category_count = randint(2, MAX_CATEGORIES)

        categories = [c for c in range(category_count)]

        entries_per_category = SAMPLE_SIZE // category_count

        distribution = []
        for _ in range(entries_per_category):
            for category in categories:
                distribution.append(category)

        feature_type = CategoricalFeature(title=f"feature_{i}", values=distribution)

        importance_score = compute_importance(distribution, feature_type)
        importance_scores.append(importance_score)

    mean_importance = mean(importance_scores)

    print(
        "Mean importance score for a categorical variable with only one "
        "value in its distribution: ",
        mean_importance,
        "(should be 0).",
    )


def run_categorical_one_sided_case_behavior_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):
    MAX_CATEGORIES = 20
    DISTRIBUTION_COUNT = 100
    SAMPLE_SIZE = 1000

    importance_scores = []

    for i in range(DISTRIBUTION_COUNT):
        category_count = randint(2, MAX_CATEGORIES)

        categories = [c for c in range(category_count)]
        category_i = randint(0, category_count - 1)

        distribution = [categories[category_i] for _ in range(SAMPLE_SIZE)]

        feature_type = CategoricalFeature(title=f"feature_{i}", values=distribution)
        feature_type.set_unique_values(categories)

        importance_score = compute_importance(distribution, feature_type)
        importance_scores.append(importance_score)

    mean_importance = mean(importance_scores)

    print(
        "Mean importance score for a categorical variable with only one "
        "value in its distribution: ",
        mean_importance,
        "(should be 1).",
    )


def run_continuous_one_sided_case_behavior_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):
    MAX_RANGE = 1000
    MIN_MIDDLE_VALUE = -100
    MAX_MIDDLE_VALUE = 100

    DISTRIBUTION_COUNT = 100
    SAMPLE_SIZE = 1000

    importance_scores = []
    for i in range(DISTRIBUTION_COUNT):
        middle_value = uniform(MIN_MIDDLE_VALUE, MAX_MIDDLE_VALUE)
        value_range = uniform(0, MAX_RANGE)

        min_value = middle_value - 0.5 * value_range
        max_value = middle_value + 0.5 * value_range

        unique_value = uniform(min_value, max_value)

        distribution = [unique_value for _ in range(SAMPLE_SIZE)]

        feature_type = ContinuousFeature(title=f"feature_{i}", values=distribution)
        feature_type.set_min(min_value)
        feature_type.set_max(max_value)

        importance_score = compute_importance(distribution, feature_type)
        importance_scores.append(importance_score)

    mean_importance = mean(importance_scores)

    print(
        "Mean importance score for a continuous variable with only one "
        "value in its distribution: ",
        mean_importance,
        "(should be 1).",
    )


def run_continuous_perfectly_uniform_case_behavior_test(
    compute_importance: Callable[[List[float], FeatureType], float]
):
    MAX_RANGE = 1000
    MIN_MIDDLE_VALUE = -100
    MAX_MIDDLE_VALUE = 100

    DISTRIBUTION_COUNT = 100
    SAMPLE_SIZE = 1000

    importance_scores = []
    for i in range(DISTRIBUTION_COUNT):
        middle_value = uniform(MIN_MIDDLE_VALUE, MAX_MIDDLE_VALUE)
        value_range = uniform(0, MAX_RANGE)

        min_value = middle_value - 0.5 * value_range
        max_value = middle_value + 0.5 * value_range

        equal_value_distance = value_range / SAMPLE_SIZE
        distribution = [
            min_value + i * equal_value_distance for i in range(SAMPLE_SIZE)
        ]

        feature_type = ContinuousFeature(title=f"feature_{i}", values=distribution)
        feature_type.set_min(min_value)
        feature_type.set_max(max_value)

        importance_score = compute_importance(distribution, feature_type)
        importance_scores.append(importance_score)

    mean_importance = mean(importance_scores)

    print(
        "Mean importance score for a continuous variable with perfectly "
        "evenly spaced values in its distribution: ",
        mean_importance,
        "(should be 0).",
    )
