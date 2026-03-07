"""
Q1 Part 4 — Engle-Manganelli (2004) Dynamic Quantile (DQ) Test.

The DQ test regresses the demeaned hit sequence H_t = I_t - p onto a constant
and its own K lagged values.  Under correct conditional coverage H_t is not
predictable by its lags and the Wald-type statistic has an asymptotic chi^2
distribution with (K + 1) degrees of freedom.

H0: Cov(H_t, H_{t-k}) = 0  for all k = 1, ..., K,  and  E[H_t] = 0
HA: At least one autoregressive coefficient is non-zero

Test statistic
--------------
Define the T × (K+1) regressor matrix  X = [1, H_{t-1}, ..., H_{t-K}].
Under H0:
    DQ = (H' X (X'X)^{-1} X' H) / (p * (1-p))  ~  chi^2(K+1)

A value larger than the chi^2 critical value indicates that violations
cluster or have serial correlation — i.e. the model mis-specifies the tail.

Reference
---------
Engle, R.F. & Manganelli, S. (2004). CAViaR: Conditional Autoregressive
Value at Risk by Regression Quantiles. *Journal of Business & Economic
Statistics*, 22(4), 367-381.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats


def dq_test(violations, p0, n_lags: int = 4):
    """
    Engle-Manganelli (2004) Dynamic Quantile test.

    Parameters
    ----------
    violations : array-like of int (0/1) — daily violation flags
    p0         : float — nominal tail probability (e.g. 0.01 for 99% VaR)
    n_lags     : int   — number of lagged hit terms to include (default 4)

    Returns
    -------
    dq_stat : float — DQ Wald statistic (chi^2(K+1) under H0)
    p_dq    : float — p-value (right-tail)
    reject  : bool  — True if p_dq < 0.05
    """
    v = np.asarray(violations, dtype=float)
    v = v[~np.isnan(v)]

    T = len(v)
    if T < n_lags + 20:      # insufficient observations
        return np.nan, np.nan, np.nan

    H = v - p0               # demeaned hit sequence

    # Build regressor matrix: [constant, H_{t-1}, ..., H_{t-K}]
    n_obs = T - n_lags
    X     = np.ones((n_obs, n_lags + 1), dtype=float)
    for k in range(1, n_lags + 1):
        X[:, k] = H[n_lags - k : T - k]

    y = H[n_lags:]           # dependent variable H_t  (t = K+1, ..., T)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # DQ = (H' X (X'X)^{-1} X' H) / (p0 * (1-p0))
    dq_stat = float((y @ X @ XtX_inv @ X.T @ y) / (p0 * (1.0 - p0)))
    dq_stat = max(dq_stat, 0.0)

    df    = n_lags + 1
    p_dq  = float(stats.chi2.sf(dq_stat, df=df))
    return dq_stat, p_dq, p_dq < 0.05
