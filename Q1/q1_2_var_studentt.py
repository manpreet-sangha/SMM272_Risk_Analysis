"""
Q1 Part 2 — Method 3: Parametric Student-t VaR and ES.

The Student-t distribution has heavier tails than the Normal and provides a
better empirical fit for equity returns (confirmed by the distribution-fitting
module of Q1 Part 1, which estimated ν ≈ 3.30).

A location-scale Student-t,  X = μ + σ Z  where  Z ~ t(ν), is fitted to the
rolling window returns by Maximum Likelihood Estimation (MLE).

VaR formula
-----------
VaR(99%)  =  t_{ν}⁻¹(0.01; μ, σ)

where t_{ν}⁻¹(α; μ, σ)  is the α-quantile of the location-scale Student-t.

Expected Shortfall formula  (closed form)
-----------------------------------------
For the standardised Student-t with ν degrees of freedom, the ES of Z ≡ (X−μ)/σ is:

    E[Z | Z ≤ z_α]  =  −f_ν(z_α) / α  ×  (ν + z_α²) / (ν − 1)

where  f_ν  is the standard t PDF and  z_α = t_ν⁻¹(α).

Scaling back to the original return units:

    ES(99%)  =  μ  +  σ × [−f_ν(z_α) / α × (ν + z_α²) / (ν − 1)]

Valid for ν > 1.  Falls back to the Normal formula if ν ≤ 1 or MLE fails.

Both VaR and ES are returned as negative numbers (losses).
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats
from config import VAR_CONFIDENCE_LEVEL
from q1_2_var_normal import compute_normal_var_es   # used as fallback


def compute_studentt_var_es(window_returns, confidence=None):
    """
    Parametric Student-t VaR and ES for a single rolling window.

    Parameters
    ----------
    window_returns : array-like
        Daily log returns for the estimation window.
    confidence : float, optional
        Confidence level (e.g. 0.99).  Defaults to VAR_CONFIDENCE_LEVEL.

    Returns
    -------
    var : float
        VaR estimate (negative — a loss).
    es : float
        ES / CVaR estimate (negative — a loss, worse than VaR).
    """
    if confidence is None:
        confidence = VAR_CONFIDENCE_LEVEL

    alpha = 1.0 - confidence
    arr   = np.asarray(window_returns, dtype=float)

    try:
        # MLE fit: returns (df, loc, scale)
        nu, loc, scale = stats.t.fit(arr)

        if nu <= 1.0:          # ES undefined for df ≤ 1; fall back
            return compute_normal_var_es(arr, confidence)
        if scale <= 0:
            return compute_normal_var_es(arr, confidence)

        # VaR: quantile of fitted location-scale t
        var = float(stats.t.ppf(alpha, df=nu, loc=loc, scale=scale))

        # ES: closed-form formula using standardised quantile
        z_alpha = stats.t.ppf(alpha, df=nu)           # standardised quantile
        f_z     = stats.t.pdf(z_alpha, df=nu)         # standard t PDF at z_alpha
        es      = float(loc + scale * (-f_z / alpha * (nu + z_alpha ** 2) / (nu - 1)))

        return var, es

    except Exception:
        return compute_normal_var_es(arr, confidence)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logger import setup_run_logger, get_logger
    from q1_1_build_portfolio import build_portfolio
    from q1_2_rolling_window import generate_rolling_windows

    setup_run_logger("smm272_q1_2_var_studentt")
    log = get_logger("q1_2_var_studentt")

    _, _, port_ret = build_portfolio()
    _, first_window = next(generate_rolling_windows(port_ret))
    var, es = compute_studentt_var_es(first_window)
    log.info(f"First-window Student-t  VaR(99%) = {var*100:.4f}%")
    log.info(f"First-window Student-t  ES(99%)  = {es*100:.4f}%")
