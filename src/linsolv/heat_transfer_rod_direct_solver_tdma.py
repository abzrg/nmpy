"""
Using the TDMA direct solver algorithm to solve the one dimensional steady state
heat transfer in a rod with heat source applied to it.

Equation:

    d^T / dx^2  =  -1/k S(x)


    - T   : temperature
    - x   : position
    - S(x): source term
    - k   : ?

NOTE: This code was adapted from Tony Saad's Numerical Method course with some
modifications to meet my taste
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import diag

from .tdma import thomas


def source(x):
    """Source term: S(x) == Normal Distribution"""
    # Magnitude of the pulse
    alpha = 0.1

    # Standard deviation: controls how wide the distribution is
    sigma = 0.1

    # Length of the domain (kinda repeating myself, but it's ok for now)
    L = 1

    # Location of the maximum (in this case two-thirds from the left side of the domain)
    icenter = 2.0 * L / 3.0

    return alpha * np.exp(-(((x - icenter) / sigma) ** 2))


def main() -> int:
    # Number of grid points along the rod
    # (chose a value no more than 1000, or you'll regret ;)
    grid_size: int = 10000  # [1]

    # Physical parameters
    k = 1e-5  # Heat transfer coefficient (how fast heat is being transferred)

    # Domain parameters
    domain_length = 1.0  # [m]
    dx = domain_length / (grid_size - 1)  # [m]
    x = np.linspace(0, domain_length, grid_size)  # [m]

    # Boundary conditions
    T_left = 300  # K (Kelvin)
    T_right = 350  # K

    #
    # -- Building Solution Matrices
    #

    # Main diagonal matrix
    d = -2.0 * np.ones(grid_size)
    d[0] = 1.0
    d[-1] = 1.0

    # First upper diagonal matrix
    u = np.ones(grid_size - 1)
    u[0] = 0.0

    # First lower diagonal matrix
    l = np.ones(grid_size - 1)
    l[-1] = 0.0

    # Coefficient matrix
    A = diag(l, -1) + diag(d, 0) + diag(u, 1)

    # Right hand side vector
    b = -dx * dx * source(x) / k
    b[0] = T_left
    b[-1] = T_right

    # Time the solution process done by Numpy's `solve` function
    tic = time.time()
    T = thomas(A, b)  # solve into `T` field
    toc = time.time()
    print(f"It took {toc - tic}s to solve the system with {grid_size} equations")

    # # Plot the temperature
    # T = np.reshape(T, [1, grid_size])
    # plt.imshow(T, cmap="inferno", aspect="auto")
    # plt.colorbar()
    # plt.show()

    # plt.plot(x, T, label="T")
    # plt.legend()
    # plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
