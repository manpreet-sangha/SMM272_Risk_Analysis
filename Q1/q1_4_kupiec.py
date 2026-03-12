"""
Q1 Part 4 — Kupiec (1995) Proportion of Failures (POF) test.
"""

import numpy as np
from scipy import stats


def kupiec_pof(k, n, p0):
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    Tests whether the observed VaR violation rate is consistent with the
    nominal violation probability under the null of correct unconditional
    coverage.

    Parameters
    ----------
    k  : int   — observed number of violations
    n  : int   — total number of observations
    p0 : float — nominal violation probability (= 1 - confidence)

    Returns
    -------
    lr_stat : float  — likelihood ratio statistic (chi^2(1) under H0)
    p_value : float  — right-tail p-value
    reject  : bool   — True if H0 rejected at 5% significance
    """
    if k == 0:
        # log(0) guard — treat as near-zero observed rate
        p_hat = 1e-10
    else:
        p_hat = k / n

    try:
        ll_null = k * np.log(p0) + (n - k) * np.log(1.0 - p0)
        ll_alt  = k * np.log(p_hat) + (n - k) * np.log(1.0 - p_hat)
        lr_stat = -2.0 * (ll_null - ll_alt)
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan, np.nan

    lr_stat = max(lr_stat, 0.0)   # numerical clip
    p_value = float(stats.chi2.sf(lr_stat, df=1))
    reject  = p_value < 0.05
    return float(lr_stat), p_value, reject
