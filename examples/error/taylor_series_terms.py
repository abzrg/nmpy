"""
Visualizing Taylor Series of exp for various terms
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from nm.typing import NDArray, NDArrayOrFloat
from nm.linalg import taylor_series_approx_n


def exp_formula(x: NDArrayOrFloat, nth_term: int) -> NDArrayOrFloat:
    """Taylor Series formula for e^x around a=1"""
    assert nth_term > 0, "Number of terms should be at least 1."
    return (
        (1.0 / math.factorial(nth_term - 1)) * np.exp(1) * ((x - 1) ** (nth_term - 1))
    )


def main() -> int:
    grid_size: int = 50

    domain_start: float = -2.0
    domain_end: float = 3.0
    domain: NDArray = np.linspace(domain_start, domain_end, num=grid_size)

    # Number of terms
    num_terms: int = 4

    # Approximation plot
    n: int
    for n in range(1, num_terms + 1):
        approxs = taylor_series_approx_n(exp_formula, domain, n)
        plt.plot(domain, approxs, label=f"{n} terms")

    # Plot of the "exact" graph
    plt.plot(domain, np.exp(domain), label="exact")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
