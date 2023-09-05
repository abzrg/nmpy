"""
Visualizing truncation error of a the Taylor Series of a function as h goes to zero
"""

import matplotlib.pyplot as plt
import numpy as np

from ..types.type_aliases import ArrayOrFloat
from .forward_euler import forward_euler


def func(x: ArrayOrFloat) -> ArrayOrFloat:
    return x**2 - 3 * x - 1


def main() -> int:
    x = 0.5

    # True value of the derivative at x
    true_val = 2 * x - 3

    steps = np.array([1e-1, 1e-2, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])

    approxs: ArrayOrFloat = forward_euler(func, x, steps)

    trunc_err = abs(true_val - approxs)  # pyright: ignore

    plt.loglog(steps, trunc_err, 'o-')

    plt.xlabel("Steps (h)")
    plt.ylabel("Total Error")

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
