"""
Experimenting with Approximate Errors
"""

# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false

import numpy as np

from ..types.type_aliases import ArrayOrFloat
from .forward_euler import forward_euler
from .relative_error import relative_error


def func(x: ArrayOrFloat) -> ArrayOrFloat:
    return 2 * np.exp(0.5 * x)


def main() -> int:
    max_iter: int = 21

    steps: np.ndarray = np.array([8.0 / (2.0 ** float(i)) for i in range(max_iter)])

    derivs = forward_euler(func, 3.0, steps)

    prev = derivs[0:-1]
    curr = derivs[1:]

    rel_errs: ArrayOrFloat = relative_error(curr, prev)

    print(rel_errs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
