import math

from nm.error import taylor_series_approx_tol

def exp_formula(x: float, nth_term: int) -> float:
    """Taylor Series forumla for exp function

    Positional arguments:
        x        -- position
        nth_term -- nth term in the Taylor Series

    Returns:
        Nth term in the taylor series of the exp
    """
    return (1.0 / math.factorial(nth_term)) * (x**nth_term)


def main() -> int:
    print(f"{taylor_series_approx_tol(exp_formula, 2, 1e-6)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
