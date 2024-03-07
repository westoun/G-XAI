#!/usr/bin/env python3

from deap import creator, base, tools
import random
from typing import List, Any, Tuple, Dict, Callable

from gxai.feature_types import (
    CategoricalFeature,
    ContinuousFeature,
    FeatureType,
)
from .random import randfloat


def init_toolbox(
    evaluate: Callable[[], float], feature_types: List[FeatureType]
) -> Any:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", create_individual, creator.Individual, feature_types=feature_types
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate", mixed_data_mutation, feature_types=feature_types, indpb=0.1
    )
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def create_individual(container: Callable, feature_types: List[FeatureType]):
    x = []
    for feature_type in feature_types:
        if feature_type == CategoricalFeature:
            x_i = random.choice(feature_type.unique_values)
            x.append(x_i)

        elif feature_type == ContinuousFeature:
            x_i = randfloat(feature_type.min, feature_type.max)
            x.append(x_i)
        else:
            raise NotImplementedError()

    return container(x)


def mixed_data_mutation(
    individual: List,
    feature_types: List[FeatureType],
    indpb: float,
) -> Tuple[List]:
    for i in range(len(individual)):
        if random.random() > indpb:
            continue

        feature_type = feature_types[i]

        if feature_type == CategoricalFeature:
            individual[i] = random.choice(feature_type.unique_values)

        elif feature_type == ContinuousFeature:
            individual[i] = randfloat(feature_type.min, feature_type.max)

        else:
            raise NotImplementedError()

    return (individual,)
