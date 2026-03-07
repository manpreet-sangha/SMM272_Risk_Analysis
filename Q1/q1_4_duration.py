"""
Q1 Part 4 — Christoffersen-Pelletier (2004) Weibull Duration Test.

Under correct conditional coverage the times between VaR violations
follow a Geometric(p) distribution, well approximated for small p by an
Exponential(p) distribution (no duration dependence, i.e. no clustering).

Under the alternative, durations follow a Weibull(a, b) distribution whose
hazard function h(d) = (b/a)(d/a)^(b-1) is non-constant when b != 1:
  b > 1  → increasing hazard  (violations cluster into short bursts)
  b < 1  → decreasing hazard  (violations are more evenly spaced than chance)

H0: durations ~ Exp(p)        [0 free parameters; p fixed by nominal tail]
HA: durations ~ Weibull(a, b) [2 free parameters]

LR_dur = -2 * (LL_0 - LL_A) ~ chi^2(2)

Reference
---------
Christoffersen, P. & Pelletier, D. (2004). Backtesting Value-at-Risk:
A Duration-Based Approach. *Journal of Financial Econometrics*, 2(1), 84-108.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def duration_test(violations, p0):
    """
    Christoffersen-Pelletier Weibull duration test.

    Parameters
    ----------
    violations : array-like of int (0/1) — daily violation flags
    p0         : float — nominal tail probability (e.g. 0.01 for 99% VaR)

    Returns
    -------
    lr_dur  : float  — likelihood-ratio statistic (chi^2(2) under H0)
    p_dur   : float  — p-value (right-tail)
    reject  : bool   — True if p_dur < 0.05
    """
    v = np.asarray(violations, dtype=float)
    v = v[~np.isnan(v)].astype(int)
    viol_idx = np.where(v == 1)[0]

    if len(viol_idx) < 3:            # need at least 2 complete durations
        return np.nan, np.nan, np.nan

    # Complete inter-violation durations (days between successive violations)
    durations = np.diff(viol_idx).astype(float)
    if len(durations) < 2:
        return np.nan, np.nan, np.nan

    # ── Null log-likelihood: Exp(p0) ──────────────────────────────────────
    # LL_0 = n * log(p0) - p0 * sum(d)
    n_dur   = len(durations)
    ll_null = n_dur * np.log(p0) - p0 * durations.sum()

    # ── Alternative log-likelihood: Weibull MLE ──────────────────────────
    # Weibull(a, b): f(d) = (b/a)(d/a)^(b-1) exp(-(d/a)^b)
    # LL_A = sum[ log(b/a) + (b-1)*log(d/a) - (d/a)^b ]
    def neg_ll_weibull(params):
        a, b = params
        if a <= 0.0 or b <= 0.0:
            return 1.0e10
        ll = np.sum(np.log(b / a) + (b - 1.0) * np.log(durations / a) -
                    (durations / a) ** b)
        return -ll

    # Initialise at exponential (b=1, a=mean duration = 1/p̂)
    a0 = durations.mean()
    b0 = 1.0
    res = minimize(
        neg_ll_weibull,
        x0=[a0, b0],
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 10_000},
    )
    ll_alt = -res.fun

    # ── LR statistic ─────────────────────────────────────────────────────
    lr_dur = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    p_dur  = float(stats.chi2.sf(lr_dur, df=2))
    return lr_dur, p_dur, p_dur < 0.05
