"""
Implementation of TDMA (Tri Diagonal Matrix Algorithm) a.k.a. Thomas algorithm
"""

import numpy as np


# def thomas(l, d, u, b):
def thomas(A, b):
    """
    Given:
        - 1st lower diagonal (l),
        - main diagonal (d),
        - 1st upper diagnoal (d),
        - right hand side vector (b),
    returns the solution to the problem
    """

    u = np.diag(A, +1).copy()
    d = np.diag(A, 0).copy()
    l = np.diag(A, -1).copy()

    # # Creating a copy of input arrays to not overwrite
    # # the original arrays
    # l = l.copy()
    # d = d.copy()
    # u = u.copy()
    # b = b.copy()

    grid_size = len(d)

    # Define solution field
    field = np.zeros(grid_size)

    # Step 1: Forward elimination
    for i in range(1, grid_size):
        d[i] = d[i] - u[i - 1] * l[i - 1] / d[i - 1]
        b[i] = b[i] - b[i - 1] * l[i - 1] / d[i - 1]

    # Step 2: Backward substitution
    field[grid_size - 1] = b[grid_size - 1]
    for i in range(grid_size - 2, -1, -1):
        field[i] = (b[i] - u[i] * field[i + 1]) / d[i]

    return field
