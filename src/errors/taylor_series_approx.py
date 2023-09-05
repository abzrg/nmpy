"""
Approximating value of functions using
Taylor Series down to an error (a certain tolerance)
"""

import math
from typing import Callable

from ..types.type_aliases import ArrayOrFloat
from .relative_error import relative_error


def taylor_series_approx(
    formula: Callable[[ArrayOrFloat, int], ArrayOrFloat],
    x: ArrayOrFloat,
    num_terms: int,
) -> ArrayOrFloat:
    """Computes Taylor Series Approximation of a function up to n terms, given it's series forumla"""
    approx: ArrayOrFloat = 0.0

    for n in range(1, num_terms + 1):
        approx += formula(x, n)

    return approx


def taylor_series_approx_tol(
    formula: Callable[[ArrayOrFloat, int], ArrayOrFloat],
    x: ArrayOrFloat,
    tolerance: ArrayOrFloat,
) -> ArrayOrFloat:
    """
    Computes the approximation of a continuous function at the vicinity of a = 0

    @param forumla: a func that represents the specific Taylor series formula of the function
    @param x: position at which we want the approximate value of the function
    @param tolerance: at what tolerance we should stop iteration (<1)

    @return: approximate value of a function at x
    """
    nth_term: int = 0
    rel_err: ArrayOrFloat = 1.0
    curr: ArrayOrFloat = 0.0
    prev: ArrayOrFloat = 0.0

    while rel_err > tolerance:
        curr += formula(x, nth_term)

        rel_err = relative_error(curr, prev) * 100

        nth_term += 1
        prev = curr

    return curr


# ---<test>--------------------------------------------------------------------


def exp_formula(x: float, nth_term: int) -> float:
    """
    Taylor Series forumla for exp function

    @param x: position
    @param nth_term: nth term in the Taylor Series
    """
    return (1.0 / math.factorial(nth_term)) * (x**nth_term)


def main() -> int:
    print(f"{taylor_series_approx_tol(exp_formula, 2, 1e-6)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
