#!/usr/bin/env python3

from functools import partial
import numpy as np
import pandas as pd
from random import choice, uniform
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from typing import List, Tuple

from gxai import (
    FeatureType,
    CategoricalFeature,
    ContinuousFeature,
    run_genetic_algorithm,
    compute_feature_importance_scores,
    generate_comparison_charts,
    GAParams,
)
from gxai.utils.matrix import transpose_values
from gxai.feature_importances import (
    compute_contrast_entropy_importance,
    compute_contrast_wasserstein_importance,
)


def generate_dataset(
    dataset_size: int, categorical_count: int, continuous_count: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float], List[FeatureType]]:
    X = []
    y = []

    absolute_weights = [
        uniform(0, 1) for _ in range(categorical_count + continuous_count)
    ]

    relative_weights = [weight / sum(absolute_weights) for weight in absolute_weights]

    for i in range(dataset_size):
        X.append([])

        for _ in range(categorical_count):
            X[i].append(choice([0, 1]))

        for _ in range(continuous_count):
            X[i].append(uniform(0, 1))

        score = sum(X[i][j] * relative_weights[j] for j in range(len(X[i])))
        if score > 0.5:
            y.append([0, 1])
        else:
            y.append([1, 0])

    feature_types = []

    X_t = transpose_values(X)
    for i in range(categorical_count):
        feature_types.append(CategoricalFeature(title=f"feature_{i}", values=X_t[i]))

    for j in range(continuous_count):
        feature_types.append(
            ContinuousFeature(
                title=f"feature_{categorical_count + j}",
                values=X_t[categorical_count + j],
            )
        )

    feature_names = [
        f"feature_{k}" for k in range(categorical_count + continuous_count)
    ]

    return (
        pd.DataFrame(X, columns=feature_names),
        pd.DataFrame(y),
        relative_weights,
        feature_types,
    )


def evaluation_function(classifier, population: List[Tuple]) -> List[float]:
    prediction = classifier.predict_proba(population)
    fitness = [(result[1],) for result in prediction]
    return fitness


def train_classifier(
    train: pd.DataFrame, target: pd.DataFrame, evaluate_score: bool = False
):
    train = train.copy()

    clf = MLPClassifier(random_state=1, max_iter=300).fit(train, target)

    if evaluate_score:
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

        score = cross_val_score(
            clf, train, target, cv=k_fold, n_jobs=1, scoring="accuracy"
        )

        average_score = round(np.mean(score) * 100, 2)
        print(
            f"\nThe model achieved an average accuracy of {average_score} on the training data."
        )

    clf.fit(train, target)
    return clf


def compute_pertubation_feature_importances(
    model, X: pd.DataFrame, y: pd.DataFrame, relative: bool = True
) -> List[float]:
    result = permutation_importance(
        model, X.values, y.values, n_repeats=20, random_state=0
    )
    importances = result.importances_mean

    if relative:
        importance_sum = sum(importances)
        for i in range(len(importances)):
            importances[i] = importances[i] / importance_sum

    return importances


def run_controlled_synthetic_data_check():
    X, y, weights, feature_types = generate_dataset(
        dataset_size=1000, categorical_count=5, continuous_count=5
    )

    classifier = train_classifier(X.values, y.values, evaluate_score=True)

    evaluate = partial(evaluation_function, classifier)

    params = GAParams(population_size=10000, max_generations=70)
    original_population, improved_population = run_genetic_algorithm(
        feature_types, column_names=X.columns, evaluate=evaluate, params=params
    )

    importance_scores = compute_feature_importance_scores(
        improved_population,
        feature_types,
        compute_importance=compute_contrast_wasserstein_importance,
    )
    for feature_name, importance, weight in zip(X.columns, importance_scores, weights):
        print(f"{feature_name}: {importance} (vs. weight: {weight})")

    print("\nPertubation Feature Importances:")
    pertubation_feature_importances = compute_pertubation_feature_importances(
        classifier, X=X, y=y
    )
    for feature_name, importance in zip(X.columns, pertubation_feature_importances):
        print(f"{feature_name}: {importance}")

    generate_comparison_charts(
        data1=original_population,
        data2=improved_population,
        feature_types=feature_types,
        target_dir="tmp/synthetic_data_check",
    )
