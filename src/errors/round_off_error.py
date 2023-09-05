"""
Summing the number 10e-5, 10e5 times

Using different integer types for storage and calculate the error
"""

import numpy as np


def main() -> int:
    number: float = 1e-5

    # Assign this number to storages of various size
    number_word: np.float16 = np.float16(number)
    number_dword: np.float32 = np.float32(number)
    number_qword: np.float64 = np.float64(number)

    sum_word: float = 0.0
    sum_dword: float = 0.0
    sum_qword: float = 0.0

    # Sum
    for _ in range(100_000):
        sum_word += float(number_word)
        sum_dword += float(number_dword)
        sum_qword += float(number_qword)

    # Compute relative error
    print(f"word  (16 bit) error:\t{abs(1.0 - sum_word) / 1.0 * 100:1.3e}%")
    print(f"dword (32 bit) error:\t{abs(1.0 - sum_dword)/ 1.0 * 100:1.3e}%")
    print(f"qword (64 bit) error:\t{abs(1.0 - sum_qword)/ 1.0 * 100:1.3e}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
