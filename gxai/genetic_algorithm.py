#!/usr/bin/env python3

from dataclasses import dataclass
import pandas as pd
import random
from typing import Tuple, Callable, List

from .feature_types import FeatureType
from .utils.genetic_algorithm import init_toolbox


@dataclass
class GAParams:
    population_size: int = 1000
    crossover_prob: float = 0.4
    mutation_prob: float = 0.2
    max_generations: int = 50


def run_genetic_algorithm(
    feature_types: List[FeatureType],
    column_names: List[str],
    evaluate: Callable[[List[Tuple]], List[float]],
    params: GAParams = GAParams(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    toolbox = init_toolbox(evaluate, feature_types)

    population = toolbox.population(n=params.population_size)

    original_population = pd.DataFrame(population, columns=column_names)

    for generation in range(1, params.max_generations + 1):
        offspring = [toolbox.clone(ind) for ind in population]

        for i in range(1, len(offspring), 2):
            if random.random() < params.crossover_prob:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i]
                )
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < params.mutation_prob:
                (offspring[i],) = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        fits = toolbox.evaluate(offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

        if generation % 5 == 0:
            print(f"Finished population {generation}")

    improved_population = pd.DataFrame(population, columns=column_names)
    return original_population, improved_population
