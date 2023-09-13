import math
from typing import Callable

import numpy as np

from nm.typing import NDArrayOrFloat


def relative_error(
    curr_approx: NDArrayOrFloat, prev_approx: NDArrayOrFloat
) -> NDArrayOrFloat:
    """Given current and previous iteration/approximation value returns the
    relative error (does not return percentage)."""
    return abs((curr_approx - prev_approx) / curr_approx)


def forward_euler(
    f: Callable[[NDArrayOrFloat], NDArrayOrFloat],
    x: NDArrayOrFloat,
    h: NDArrayOrFloat,
) -> NDArrayOrFloat:
    """Calculates the forward Euler approximation for a derivative

    Positional arguments:
        f -- Function that we want to calculate its derivative
        x -- Position at which the derivative will be calculated
        h -- An array of step sizes
    """
    return (f(x + h) - f(x)) / h


def taylor_series_approx_n(
    formula: Callable[[NDArrayOrFloat, int], NDArrayOrFloat],
    x: NDArrayOrFloat,
    num_terms: int,
) -> NDArrayOrFloat:
    """Computes Taylor Series Approximation of a function up to n terms, given
    it's series forumla."""
    approx: NDArrayOrFloat = 0.0

    for n in range(1, num_terms + 1):
        approx += formula(x, n)

    return approx


def taylor_series_approx_tol(
    formula: Callable[[NDArrayOrFloat, int], NDArrayOrFloat],
    x: NDArrayOrFloat,
    tolerance: NDArrayOrFloat,
) -> NDArrayOrFloat:
    """Computes the approximation of a continuous function at the vicinity of a = 0.

    Positional arguments:
        forumla -- A func that represents the specific Taylor series formula of the function
        x       -- Position at which we want the approximate value of the function
        tol     -- At what tolerance we should stop iteration (<1)

    Returns:
        Approximate value of a function at x
    """
    nth_term: int = 0
    rel_err: NDArrayOrFloat = 1.0
    curr: NDArrayOrFloat = 0.0
    prev: NDArrayOrFloat = 0.0

    while rel_err > tolerance:
        curr += formula(x, nth_term)

        rel_err = relative_error(curr, prev) * 100

        nth_term += 1
        prev = curr

    return curr
