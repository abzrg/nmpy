"""
Experimenting with Approximate Errors
"""

# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false

import numpy as np

from nm.typing import NDArray, NDArrayOrFloat
from nm.error import forward_euler
from nm.error import relative_error


def func(x: NDArrayOrFloat) -> NDArrayOrFloat:
    return 2 * np.exp(0.5 * x)


def main() -> int:
    max_iter: int = 21

    steps: NDArray = np.array([8.0 / (2.0 ** float(i)) for i in range(max_iter)])

    derivs = forward_euler(func, 3.0, steps)

    prev = derivs[0:-1]
    curr = derivs[1:]

    rel_errs: NDArrayOrFloat = relative_error(curr, prev)

    print(rel_errs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
