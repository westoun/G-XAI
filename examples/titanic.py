#!/usr/bin/env python3

from functools import partial
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from typing import List, Tuple

from gxai import (
    FeatureType,
    CategoricalFeature,
    ContinuousFeature,
    run_genetic_algorithm,
    compute_feature_importance_scores,
    generate_comparison_charts,
)
from gxai.feature_importances import compute_contrast_entropy_importance, \
    compute_contrast_wasserstein_importance


def train_classifier(
    train: pd.DataFrame, target: pd.DataFrame, evaluate_score: bool = False
):
    clf = RandomForestClassifier(n_estimators=10)

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


def load_titanic(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(path)

    # Drop unnecessary columns
    drop_columns = ["PassengerId", "Name", "Cabin", "Ticket", "Parch"]
    X.drop(drop_columns, axis=1, inplace=True)

    # Fill missing values to avoid mean filling age
    X["Embarked"].fillna("N/A", inplace=True)

    # Drop rows with missing values
    X = X.dropna()

    # Map categorical columns to numeric
    embarked_mapping = {"S": 0, "C": 1, "Q": 2, "N/A": 3}
    sex_mapping = {"male": 0, "female": 1}

    X["Embarked"] = X["Embarked"].map(embarked_mapping)
    X["Sex"] = X["Sex"].map(sex_mapping)

    # Separate target variable
    y = X["Survived"]
    X = X.drop("Survived", axis=1)

    return X, pd.Series(y)


def create_titanic_feature_types(data: pd.DataFrame) -> List[FeatureType]:
    feature_types: List[FeatureType] = [
        CategoricalFeature(title="Pclass", values=data["Pclass"].tolist()),
        CategoricalFeature(title="Sex", values=data["Sex"].tolist()),
        ContinuousFeature(title="Age", values=data["Age"].tolist())
        .set_min(0)
        .set_max(100),
        ContinuousFeature(title="SibSp", values=data["SibSp"].tolist()).set_min(0),
        ContinuousFeature(title="Fare", values=data["Fare"].tolist()).set_min(0),
        CategoricalFeature(title="Embarked", values=data["Embarked"].tolist()),
    ]

    return feature_types


def reverse_map_titanic(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function is responsible for reversing the mapping of the 'Sex' and 'Embarked' columns.
    It's mainly used to translate the numeric representation back into the original categorical values.
    """
    data = data.copy()

    reverse_sex_mapping = {0: "male", 1: "female"}
    data["Sex"] = data["Sex"].map(reverse_sex_mapping)

    reverse_embarked_mapping = {0: "S", 1: "C", 2: "Q", 3: "N/A"}
    data["Embarked"] = data["Embarked"].map(reverse_embarked_mapping)

    return data


def evaluation_function(classifier, population: List[Tuple]) -> List[float]:
    prediction = classifier.predict_proba(population)
    fitness = [(result[1],) for result in prediction]
    return fitness


def run_titanic_example():
    train_csv_path = "./data/titanic/train.csv"

    X, y = load_titanic(path=train_csv_path)
    feature_types = create_titanic_feature_types(X)

    classifier = train_classifier(X.values, y.values, evaluate_score=True)

    evaluate = partial(evaluation_function, classifier)
    original_population, improved_population = run_genetic_algorithm(
        feature_types, column_names=X.columns, evaluate=evaluate
    )

    importance_scores = compute_feature_importance_scores(
        improved_population,
        feature_types,
        compute_importance=compute_contrast_wasserstein_importance,
    )
    for feature_name, importance in zip(X.columns, importance_scores):
        print(f"{feature_name}: {importance}")

    original_population = reverse_map_titanic(original_population)
    improved_population = reverse_map_titanic(improved_population)
    generate_comparison_charts(
        data1=original_population,
        data2=improved_population,
        feature_types=feature_types,
    )
