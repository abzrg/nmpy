"""
Provides the `forward_euler` function which computes
an approximation of the first order derivative
"""

import math
from typing import Callable

import numpy as np

from ..types.type_aliases import ArrayOrFloat


def forward_euler(
    f: Callable[[ArrayOrFloat], ArrayOrFloat], x: ArrayOrFloat, h: ArrayOrFloat
) -> ArrayOrFloat:
    """
    @param f: function that we want to calculate its derivative
    @param x: position at which the derivative will be calculated
    @param h: an array of step sizes
    """
    return (f(x + h) - f(x)) / h
