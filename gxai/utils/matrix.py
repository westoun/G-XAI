#!/usr/bin/env python3

from functools import lru_cache
from typing import List


def _matrix_to_tuple(function):
    """
    A decorator function to convert lists into tuples so they can be used as keys for caching.
    """
    # Internally, lru_cache stores parameters as dictionary
    # keys and the return values as corresponding values.
    # As such, the parameters of the cached function have to
    # be hashable. Lists are not hashable. Tuples, however,
    # are. Inspired by https://stackoverflow.com/a/60980685.
    def wrapper(data: List[List]):
        tuple_list = [tuple(item) for item in data]
        result = function(tuple(tuple_list))
        return result

    return wrapper


@_matrix_to_tuple
@lru_cache(maxsize=1)
# Set max_size=1 to avoid calling the same transposition
# logic multiple times within the same round of the algorithm.
# Since the returned list is likely to be large and sure to be
# outdated after each iteration, we do not want the cache to
# grow beyond necessary.
def transpose_values(data: List[List]) -> List[List]:
    """
    Transpose a list of lists. Uses memoization to avoid unnecessary computation.
    """
    transposed_values: List[List] = []

    for j, item in enumerate(data):
        for i, xi in enumerate(item):
            if j == 0:
                transposed_values.append([])

            transposed_values[i].append(xi)

    return transposed_values


@_matrix_to_tuple
@lru_cache(maxsize=1)
# Set max_size=1 to avoid calling the same transposition
# logic multiple times within the same round of the algorithm.
# Since the returned list is likely to be large and sure to be
# outdated after each iteration, we do not want the cache to
# grow beyond necessary.
def extract_unique_values(data: List[List]) -> List[List]:
    """
    Extract all unique values from each column of a list of lists.
    """
    transposed_values = transpose_values(data)

    unique_values: List[List] = []
    for i in range(len(transposed_values)):
        unique_values.append(list(set(transposed_values[i])))

    return unique_values
