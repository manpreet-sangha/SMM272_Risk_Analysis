"""
Q1 Part 2 — Method 1: Historical Simulation VaR and ES.

Historical Simulation (HS) is a fully non-parametric approach.  It makes no
assumption about the shape of the return distribution; instead it treats the
empirical return distribution of the rolling estimation window as the true
distribution for the next day.

VaR definition
--------------
VaR_{α}  =  inf{ x : P(r ≤ x) ≥ α }   (left-tail quantile)

For a 99 % confidence level (α = 1 %):
    VaR(99%)  =  1st percentile of the rolling window returns

Expected Shortfall (CVaR) definition
-------------------------------------
ES_{α}  =  E[ r | r ≤ VaR_{α} ]

For a 99 % confidence level:
    ES(99%)  =  mean of returns BELOW the VaR threshold

Both VaR and ES are returned as negative numbers (they represent losses).
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from config import VAR_CONFIDENCE_LEVEL


def compute_historical_var_es(window_returns, confidence=None):
    """
    Compute Historical Simulation VaR and ES for a single rolling window.

    Parameters
    ----------
    window_returns : array-like
        Daily log returns for the estimation window.
    confidence : float, optional
        Confidence level (e.g. 0.99).  Defaults to VAR_CONFIDENCE_LEVEL.

    Returns
    -------
    var : float
        VaR estimate (negative value — a loss).
    es : float
        ES / CVaR estimate (negative value, worse than VaR).
    """
    if confidence is None:
        confidence = VAR_CONFIDENCE_LEVEL

    alpha = 1.0 - confidence          # e.g. 0.01
    arr   = np.asarray(window_returns, dtype=float)

    var = np.percentile(arr, alpha * 100)   # 1st percentile

    tail = arr[arr <= var]
    es   = float(tail.mean()) if len(tail) > 0 else var

    return float(var), float(es)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logger import setup_run_logger, get_logger
    from q1_1_build_portfolio import build_portfolio
    from q1_2_rolling_window import generate_rolling_windows

    setup_run_logger("smm272_q1_2_var_historical")
    log = get_logger("q1_2_var_historical")

    _, _, port_ret = build_portfolio()
    _, first_window = next(generate_rolling_windows(port_ret))
    var, es = compute_historical_var_es(first_window)
    log.info(f"First-window HS  VaR(99%) = {var*100:.4f}%")
    log.info(f"First-window HS  ES(99%)  = {es*100:.4f}%")
