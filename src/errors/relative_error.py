"""
Provides a function to compute the relative error
"""

from ..types.type_aliases import ArrayOrFloat


def relative_error(
    curr_approx: ArrayOrFloat, prev_approx: ArrayOrFloat
) -> ArrayOrFloat:
    """
    Given current and previous iteration/approximation value returns the
    relative error (does not return percentage)
    """
    return abs((curr_approx - prev_approx) / curr_approx)
