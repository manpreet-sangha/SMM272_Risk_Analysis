"""
Q1 Part 2 — Method 4: GARCH(1,1) Dynamic VaR and ES.

The three parametric methods above assume constant variance within each
rolling window.  The GARCH(1,1) model (Bollerslev, 1986) relaxes this
assumption by allowing the conditional variance to evolve over time via
the recursion:

    σ²_t  =  ω  +  α ε²_{t-1}  +  β σ²_{t-1}

where ε_t = r_t − μ are the mean-zero return innovations.

Procedure
---------
1.  Fit GARCH(1,1) with constant mean and Normal innovations to the rolling
    estimation window (via the `arch` library's maximum likelihood routine).
2.  Obtain the one-step-ahead conditional variance forecast σ²_{T+1|T} and
    the estimated constant mean μ.
3.  Compute VaR and ES under the assumption that the next-day standardised
    residual follows a standard Normal distribution:

    VaR(99%)  =  μ  +  Φ⁻¹(0.01) × σ_{T+1|T}
    ES (99%)  =  μ  −  σ_{T+1|T} × φ(Φ⁻¹(0.01)) / 0.01

Fallback
--------
If the `arch` library is unavailable or the GARCH optimisation fails to
converge, the function silently falls back to the Parametric Normal estimate
so that the overall pipeline continues uninterrupted.

Both VaR and ES are returned as negative numbers (losses).

Reference
---------
Bollerslev, T. (1986). Generalised autoregressive conditional
heteroscedasticity. Journal of Econometrics, 31(3), 307–327.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats
from config import VAR_CONFIDENCE_LEVEL
from q1_2_var_normal import compute_normal_var_es   # fallback

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def compute_garch_var_es(window_returns, confidence=None):
    """
    GARCH(1,1) one-step-ahead VaR and ES for a single rolling window.

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

    if not ARCH_AVAILABLE:
        return compute_normal_var_es(arr, confidence)

    try:
        # Scale to percentage returns for numerical stability during optimisation
        arr_pct = arr * 100.0

        model = arch_model(
            arr_pct,
            mean="Constant",
            vol="Garch",
            p=1, q=1,
            dist="Normal",
            rescale=False,
        )
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 500})

        # One-step-ahead conditional variance (in pct² units)
        forecast   = res.forecast(horizon=1, reindex=False)
        cond_var   = float(forecast.variance.iloc[-1, 0])
        cond_std   = np.sqrt(max(cond_var, 1e-12)) / 100.0   # back to decimal

        mu_garch   = float(res.params["mu"]) / 100.0

        z_alpha    = stats.norm.ppf(alpha)
        var        = mu_garch + z_alpha * cond_std
        es         = mu_garch - cond_std * stats.norm.pdf(z_alpha) / alpha

        return float(var), float(es)

    except Exception:
        return compute_normal_var_es(arr, confidence)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logger import setup_run_logger, get_logger
    from q1_1_build_portfolio import build_portfolio
    from q1_2_rolling_window import generate_rolling_windows

    setup_run_logger("smm272_q1_2_var_garch")
    log = get_logger("q1_2_var_garch")
    log.info(f"arch package available: {ARCH_AVAILABLE}")

    _, _, port_ret = build_portfolio()
    _, first_window = next(generate_rolling_windows(port_ret))
    var, es = compute_garch_var_es(first_window)
    log.info(f"First-window GARCH  VaR(99%) = {var*100:.4f}%")
    log.info(f"First-window GARCH  ES(99%)  = {es*100:.4f}%")
