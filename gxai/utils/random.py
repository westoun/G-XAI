#!/usr/bin/env ptyhon3

import random


def randfloat(min: float = - 100.0, max: float = 100.0) -> float:
    return random.random() * (max - min) + min
