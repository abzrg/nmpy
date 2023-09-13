import numpy as np

from nm.typing import NDArray


def tdma(A: NDArray, b: NDArray):
    """TDMA (Tri-Diagonal Matrix Algorithm) for a special type of linear system
    of equations.

    Positional arguments:
        A -- Coefficient matrix
        b -- Right-hand-side vector

    Returns:
        Solution vector of the linear system of equations
    """

    u = np.diag(A, +1).copy()
    d = np.diag(A, 0).copy()
    l = np.diag(A, -1).copy()

    grid_size = len(d)

    # Define solution field
    x = np.zeros(grid_size)

    # Step 1: Forward elimination
    for i in range(1, grid_size):
        d[i] = d[i] - u[i - 1] * l[i - 1] / d[i - 1]
        b[i] = b[i] - b[i - 1] * l[i - 1] / d[i - 1]

    # Step 2: Backward substitution
    x[grid_size - 1] = b[grid_size - 1]
    for i in range(grid_size - 2, -1, -1):
        x[i] = (b[i] - u[i] * x[i + 1]) / d[i]

    return x


def jacobi(
    A: NDArray, x_0: NDArray, b: NDArray, tol: float, max_iter: int = 50
) -> NDArray:
    """Jacobi iterative method for linear system of equations

    Positional arguments:
        A   -- matrix coefficient
        x_0 -- initial guess
        b   -- right hand side vector
        tol -- absolute tolerance

    Keyword arguments:
        max_iter -- maximum number of iterations

    Returns:
        Solution to the linear system of equations
    """
    # Make sure A is a Numpy array
    A = np.array(A)
    num_row, num_col = A.shape

    if num_row != num_col:
        raise ValueError(f"A[{num_row} x {num_col}] is not a a square matrix.")

    if num_col != len(b):
        raise ValueError(f"Incompatible dimension: A[{A.shape}], b[{b.shape}]")

    if not max_iter > 0:
        raise ValueError(
            f"Maximum number of iteration must be greater than 0: {max_iter = }"
        )

    # Main diagonal vector of matrix A
    d: NDArray = np.diag(A, 0)

    # Initialize solution vectors
    x_curr: NDArray = np.zeros(num_row)  # Current iteration (Solution)
    x_prev: NDArray = x_0  # Previous iteration

    # Initialize error (absolute approximate error)
    residual = b - A @ x_0  # Residual vector
    error: float = np.linalg.norm(residual, 2)

    k: int = 0  # Iteration counter
    while error > tol and k < max_iter:
        x_curr = x_prev + (1.0 / d) * (b - A @ x_prev)

        # Compute the (absolute) approximate error
        error = np.linalg.norm(x_curr - x_prev, 2)

        # Update the guess
        x_prev = x_curr.copy()

        k += 1

    return x_curr


def gs(
    A: NDArray,
    x_0: NDArray,
    b: NDArray,
    tol: float,
    omega: float = 1.0,
    max_iter: int = 50,
) -> NDArray:
    """Gauss-Seidel iterative method for linear system of equations.

    Positional arguments:
        A   -- Coefficient matrix
        x_0 -- Initial guess
        b   -- Right hand side vector
        tol -- Absolute tolerance

    Keyword arguments:
        max_iter -- Maximum number of iterations
        omega    -- Relaxation parameter (sor method) 0 < omega < 2

    Returns:
        Solution vector of the linear system of equations
    """
    # Make sure A is a Numpy array
    A = np.array(A)
    num_row, num_col = A.shape

    if num_row != num_col:
        raise ValueError(f"A[{num_row} x {num_col}] is not a a square matrix.")

    if num_col != len(b):
        raise ValueError(f"Incompatible dimension: A[{A.shape}], b[{b.shape}]")

    if not 0 < omega < 2:
        raise ValueError(f"Invalid {omega = }. 0 < omega < 2")

    if not max_iter > 0:
        raise ValueError(
            f"Maximum number of iteration must be greater than 0: {max_iter = }"
        )

    # Main diagonal vector of matrix A
    d: NDArray = np.diag(A, 0)

    # Initialize solution vectors
    x_curr: NDArray = np.zeros(num_row)  # Current iteration (Solution)
    x_prev: NDArray = x_0  # Previous iteration

    # Initialize error (absolute approximate error)
    residual = b - A @ x_0  # Residual vector
    error: float = np.linalg.norm(residual, 2)

    k: int = 0  # Iteration counter
    while error > tol and k < max_iter:
        x_curr = x_prev + (omega / d) * (b - A @ x_curr)

        # Compute the (absolute) approximate error
        error = np.linalg.norm(x_curr - x_prev, 2)

        # Update the guess
        x_prev = x_curr.copy()

        k += 1

    return x_curr
