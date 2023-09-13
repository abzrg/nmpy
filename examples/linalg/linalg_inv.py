"""
Using Numpy's matrix inverse function to inverse a matrix, to ultimately be used
in the solution of a system of linear equations
"""

from numpy.linalg import inv

from nm.typing import NDArray


def main() -> int:
    # Matrix of coefficients
    A: list[list[float]] = [[3.0, -1.5, 0.0], [0.0, 4.5, -1.3], [-3.0, 0.0, 3.3]]

    # Right-Hand-Side vector
    b: list[float] = [50.0, 0.0, 0.0]

    Ainv: NDArray = inv(A)

    x: NDArray = Ainv @ b

    print(f"{x = }")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
