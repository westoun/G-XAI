#!/usr/bin/env python3

import math
from typing import List


def compute_entropy(probabilities: List[float]) -> float:
    """
    Compute the entropy given a list of probabilities.

    Args:
        probabilities: A list of probabilities.

    Returns:
        The computed entropy as a float.
    """
    entropy = 0

    for probability in probabilities:
        if probability == 0:
            continue

        entropy += -probability * math.log(probability)

    return entropy
