"""Numerical conventions shared by the eigen-analyses of an information matrix.

The Laplace covariance of a calibration and the identifiability report both
decide which eigen-directions of a Hessian / Fisher matrix are exactly null and
which parameters load on them; they must agree, so the thresholds live here.
"""

# Eigenvalues at or below this fraction of the largest are treated as exactly
# zero (a structurally non-identifiable direction), so the parameters loading on
# them get an infinite standard error rather than a round-off-sized finite one.
_NULL_RTOL = 1e-10
# Default smallest loading on a null direction that counts as involvement (see
# ``identifiability``'s ``null_loading_tol``); calibrate's Laplace covariance
# uses the same value.
_NULL_LOADING_TOL = 1e-2
