#!/usr/bin/env python3

from functools import partial
import numpy as np
import pandas as pd
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


def load_adult(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dtypes = [
        ("Age", "float32"),
        ("Workclass", "category"),
        ("fnlwgt", "float32"),
        ("Education", "category"),
        ("Education-Num", "float32"),
        ("Marital Status", "category"),
        ("Occupation", "category"),
        ("Relationship", "category"),
        ("Race", "category"),
        ("Sex", "category"),
        ("Capital Gain", "float32"),
        ("Capital Loss", "float32"),
        ("Hours per week", "float32"),
        ("Country", "category"),
        ("Target", "category"),
    ]

    raw_data = pd.read_csv(
        path,
        names=[d[0] for d in dtypes],
        na_values=" ?",
        dtype=dict(dtypes),
    )

    data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
    data.dropna(inplace=True)  # Drop missing values
    filt_dtypes = list(filter(lambda x: x[0] not in ["Target", "Education"], dtypes))
    data["Target"] = (data["Target"] == " >50K").astype(int)  # convert boolean to int
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5,
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    return data.drop(["Target", "fnlwgt"], axis=1), data["Target"]


def create_adult_feature_types(data: pd.DataFrame) -> List[FeatureType]:
    feature_types: List[FeatureType] = []
    continuous_features = ["Age", "Capital Gain", "Capital Loss", "Hours per week"]
    custom_bounds = {"Age": (0, 100), "Hours per week": (0, 168)}

    for column in data.columns:
        if column in continuous_features:
            feature = ContinuousFeature(title=column, values=data[column].tolist())

            if column in custom_bounds:
                min_bound, max_bound = custom_bounds[column]
                feature.set_min(min_bound)
                feature.set_max(max_bound)
            else:
                min_bound = data[column].min()
                max_bound = data[column].max()

                feature.set_min(min_bound)
                feature.set_max(max_bound)

            feature_types.append(feature)
        else:
            feature_types.append(
                CategoricalFeature(title=column, values=data[column].tolist())
            )

    return feature_types


def train_classifier(
    train: pd.DataFrame, target: pd.DataFrame, evaluate_score: bool = False
):
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


def reverse_map_adult(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    mappings = {
        "Education-Num": {
            1.0: " Preschool",
            2.0: " 1st-4th",
            3.0: " 5th-6th",
            4.0: " 7th-8th",
            5.0: " 9th",
            6.0: " 10th",
            7.0: " 11th",
            8.0: " 12th",
            9.0: " HS-grad",
            10.0: " Some-college",
            11.0: " Assoc-voc",
            12.0: " Assoc-acdm",
            13.0: " Bachelors",
            14.0: " Masters",
            15.0: " Prof-school",
            16.0: " Doctorate",
        },
        "Workclass": {
            0: " State-gov",
            1: " Self-emp-not-inc",
            2: " Private",
            3: " Federal-gov",
            4: " Local-gov",
            5: " Self-emp-inc",
            6: " Without-pay",
        },
        "Marital Status": {
            0: " Never-married",
            1: " Married-civ-spouse",
            2: " Divorced",
            3: " Married-spouse-absent",
            4: " Separated",
            5: " Married-AF-spouse",
            6: " Widowed",
        },
        "Occupation": {
            0: " Adm-clerical",
            1: " Exec-managerial",
            2: " Handlers-cleaners",
            3: " Prof-specialty",
            4: " Other-service",
            5: " Sales",
            6: " Transport-moving",
            7: " Farming-fishing",
            8: " Machine-op-inspct",
            9: " Tech-support",
            10: " Craft-repair",
            11: " Protective-serv",
            12: " Armed-Forces",
            13: " Priv-house-serv",
        },
        "Relationship": {
            0: "Not-in-family",
            1: "Unmarried",
            2: "Other-relative",
            3: "Own-child",
            4: "Husband",
            5: "Wife",
        },
        "Race": {
            0: " White",
            1: " Black",
            2: " Asian-Pac-Islander",
            3: " Amer-Indian-Eskimo",
            4: " Other",
        },
        "Sex": {0: " Male", 1: " Female"},
        "Country": {
            0: " United-States",
            1: " Cuba",
            2: " Jamaica",
            3: " India",
            4: " Mexico",
            5: " Puerto-Rico",
            6: " Honduras",
            7: " England",
            8: " Canada",
            9: " Germany",
            10: " Iran",
            11: " Philippines",
            12: " Poland",
            13: " Columbia",
            14: " Cambodia",
            15: " Thailand",
            16: " Ecuador",
            17: " Laos",
            18: " Taiwan",
            19: " Haiti",
            20: " Portugal",
            21: " Dominican-Republic",
            22: " El-Salvador",
            23: " France",
            24: " Guatemala",
            25: " Italy",
            26: " China",
            27: " South",
            28: " Japan",
            29: " Yugoslavia",
            30: " Peru",
            31: " Outlying-US(Guam-USVI-etc)",
            32: " Scotland",
            33: " Trinadad&Tobago",
            34: " Greece",
            35: " Nicaragua",
            36: " Vietnam",
            37: " Hong",
            38: " Ireland",
            39: " Hungary",
            40: " Holand-Netherlands",
        },
    }

    for column, mapping in mappings.items():
        data[column] = data[column].map(mapping)

    return data


def evaluation_function(classifier, population: List[Tuple]) -> List[float]:
    prediction = classifier.predict_proba(population)
    fitness = [(result[1],) for result in prediction]
    return fitness


def run_adult_example():
    train_csv_path = "./data/adult/adult.data"

    X, y = load_adult(path=train_csv_path)
    feature_types = create_adult_feature_types(X)

    classifier = train_classifier(X.values, y.values, evaluate_score=True)

    evaluate = partial(evaluation_function, classifier)

    params = GAParams(population_size=10000, max_generations=100)
    original_population, improved_population = run_genetic_algorithm(
        feature_types, column_names=X.columns, evaluate=evaluate, params=params
    )

    importance_scores = compute_feature_importance_scores(
        improved_population, feature_types
    )
    for feature_name, importance in zip(X.columns, importance_scores):
        print(f"{feature_name}: {importance}")

    original_population = reverse_map_adult(original_population)
    improved_population = reverse_map_adult(improved_population)
    generate_comparison_charts(
        data1=original_population,
        data2=improved_population,
        feature_types=feature_types,
    )
