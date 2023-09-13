import numpy as np

# from src.errors.relative_error import relative_error


def jacobi(
    A: np.ndarray, x_0: np.ndarray, b: np.ndarray, tol: float, max_iter: int = 50
) -> np.ndarray:
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
        raise ValueError(f"Maximum number of iteration must be greater than 0: {max_iter = }")

    # Main diagonal vector of matrix A
    d: np.ndarray = np.diag(A, 0)

    # Initialize solution vectors
    x_curr: np.ndarray = np.zeros(num_row)  # Current iteration (Solution)
    x_prev: np.ndarray = x_0  # Previous iteration

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
