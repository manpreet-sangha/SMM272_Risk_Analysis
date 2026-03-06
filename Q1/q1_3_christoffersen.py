"""
Q1 Part 3 — Christoffersen (1998) independence test for VaR violations.
"""

import numpy as np
from scipy import stats


def christoffersen_independence(violations):
    """
    Christoffersen (1998) independence test.

    Tests whether VaR violations are serially independent (i.i.d. Bernoulli)
    using a first-order Markov transition matrix.  Under H0 the probability
    of a violation tomorrow is the same regardless of whether there was a
    violation today.

    Parameters
    ----------
    violations : array-like of int
        Daily 0/1 flags where 1 indicates a VaR violation.

    Returns
    -------
    lr_ind : float — independence LR statistic (chi^2(1) under H0)
    p_ind  : float — right-tail p-value
    """
    v = np.asarray(violations, dtype=int)
    n = len(v)

    # First-order transition counts
    t00 = int(((v[:-1] == 0) & (v[1:] == 0)).sum())
    t01 = int(((v[:-1] == 0) & (v[1:] == 1)).sum())
    t10 = int(((v[:-1] == 1) & (v[1:] == 0)).sum())
    t11 = int(((v[:-1] == 1) & (v[1:] == 1)).sum())

    # Estimated transition probabilities
    pi0 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0.0
    pi1 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0.0
    pi  = (t01 + t11) / n   if n > 0 else 0.0

    # LR under independence (constant pi) vs dependence (pi0, pi1 free)
    try:
        ll_ind = (t01 * np.log(pi  + 1e-15) + t00 * np.log(1 - pi  + 1e-15)
               +  t11 * np.log(pi  + 1e-15) + t10 * np.log(1 - pi  + 1e-15))
        ll_dep = (t01 * np.log(pi0 + 1e-15) + t00 * np.log(1 - pi0 + 1e-15)
               +  t11 * np.log(pi1 + 1e-15) + t10 * np.log(1 - pi1 + 1e-15))
        lr_ind = max(-2.0 * (ll_ind - ll_dep), 0.0)
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan

    p_ind = float(stats.chi2.sf(lr_ind, df=1))
    return float(lr_ind), p_ind
