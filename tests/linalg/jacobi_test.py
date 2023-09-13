import pytest

import numpy as np

from nm.linalg import jacobi


A = np.array([[5, 1, 1], [2, 3, 0], [3, 0, 4]])
b = np.array([10, 11, 12])
x_0 = np.array([1, 1, 1])
max_iter = 100
atol = 1e-3


def test_solution():
    """Tests if jacobi computes the right solution."""
    x_jacobi = jacobi(A, x_0, b, atol, max_iter=max_iter)
    x_solve = np.linalg.solve(A, b)
    assert any(np.isclose(x_jacobi, x_solve, atol=atol)) is True


def test_invalid_coeff_matrix():
    """Tests if it raises an exception when A is not a square matrix."""
    A_invalid = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        jacobi(A_invalid, x_0, b, atol)


def test_invalid_rhs_vector():
    """Tests if it raises an exception when b has a incompatible dimension with A."""
    b_invalid = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        jacobi(A, x_0, b_invalid, atol)


def test_max_iter():
    """Tests if it raises an exception when max_iter is not greater than 0"""
    with pytest.raises(ValueError):
        jacobi(A, x_0, b, atol, max_iter=0)
