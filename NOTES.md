# NOTES

Here are some notes on tips and tricks, function and classes that I learn

## `diag`

`diag(v, k)`: Extract a diagonal or construct a diagonal array
 - If `v` is a 1D array, return a 2D array with `v` on the `k`th diagonal.
 - If `v` is a 2D array, return a copy of its `k`th diagonal.
 - Use `k > 0` for diagonals above the main diagonal
 - Use `k < 0` for diagonals below the main diagonal
 - Returns a ndarray
