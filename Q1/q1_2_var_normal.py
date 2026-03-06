"""
Q1 Part 2 — Method 2: Parametric Normal VaR and ES.

The Parametric Normal approach assumes that daily log returns for the next
period follow a Normal distribution whose parameters (μ, σ) are estimated
from the rolling window by their sample analogues (maximum likelihood
estimators for the Normal family).

VaR formula
-----------
VaR(99%)  =  μ  +  Φ⁻¹(0.01) × σ

where Φ⁻¹ is the standard Normal quantile function (≈ −2.326 at 1 %).

Expected Shortfall formula  (closed form from truncated Normal)
--------------------
ES(99%)   =  μ  −  σ × φ(Φ⁻¹(0.01)) / 0.01

where φ is the standard Normal PDF.

Derivation:
  E[X | X ≤ VaR]  =  μ + σ × E[Z | Z ≤ z_α]

  For Z ~ N(0,1):  E[Z | Z ≤ z_α]  =  −φ(z_α) / Φ(z_α)  =  −φ(z_α) / α

Both VaR and ES are returned as negative numbers (losses).
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats
from config import VAR_CONFIDENCE_LEVEL


def compute_normal_var_es(window_returns, confidence=None):
    """
    Parametric Normal VaR and ES for a single rolling window.

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

    mu    = arr.mean()
    sigma = arr.std(ddof=1)

    z_alpha = stats.norm.ppf(alpha)                       # ≈ −2.326
    var     = mu + z_alpha * sigma
    es      = mu - sigma * stats.norm.pdf(z_alpha) / alpha

    return float(var), float(es)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logger import setup_run_logger, get_logger
    from q1_1_build_portfolio import build_portfolio
    from q1_2_rolling_window import generate_rolling_windows

    setup_run_logger("smm272_q1_2_var_normal")
    log = get_logger("q1_2_var_normal")

    _, _, port_ret = build_portfolio()
    _, first_window = next(generate_rolling_windows(port_ret))
    var, es = compute_normal_var_es(first_window)
    log.info(f"First-window Normal  VaR(99%) = {var*100:.4f}%")
    log.info(f"First-window Normal  ES(99%)  = {es*100:.4f}%")
