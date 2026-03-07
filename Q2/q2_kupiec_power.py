"""
Q2 — Core Monte Carlo engine: Power of the Kupiec (1995) POF test.

Procedure
---------
Given fitted Gaussian (H0) and GARCH(1,1) (H1) parameters:

1.  For each confidence level c ∈ {0.90, 0.95, 0.99}:

    a.  Compute the fixed Gaussian VaR threshold:
            VaR_c = μ_gauss + Φ⁻¹(1−c) × σ_gauss   (a negative number)

        This is the VaR a risk manager would use if she assumed H0.

    b.  SIZE SIMULATION (under H0):
        Simulate B paths of length T from N(μ_gauss, σ²_gauss).
        For each path, count violations (r_t < VaR_c), compute LR_uc
        and record rejection at 5%.  Empirical rejection rate ≈ 5% confirms
        the simulation is calibrated correctly.

    c.  POWER SIMULATION (under H1):
        Simulate B paths of length T from fitted GARCH(1,1).
        For each path, count violations using the SAME fixed Gaussian VaR
        (i.e. the risk manager still uses the Gaussian model), compute LR_uc
        and record rejection.  Empirical rejection rate = estimated power.

    d.  Store full LR_uc distribution under H0 and H1 for plotting.

References
----------
Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives, 3(2), 73–84.
Christoffersen, P. F. (1998). Evaluating interval forecasts.
    International Economic Review, 39(4), 841–862.
Campbell, S. D. (2005). A review of backtesting and backtesting procedures.
    Finance and Economics Discussion Series, Federal Reserve Board.
Candelon, B., Colletaz, G., Hurlin, C., & Tokpavi, S. (2011). Backtesting
    Value-at-Risk: A GMM duration-based test. Journal of Financial
    Econometrics, 9(2), 314–343.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

from config import Q2_CONFIDENCE_LEVELS, Q2_MC_REPS, Q2_RANDOM_SEED, Q2_OUTPUT_DIR
from logger import get_logger
from q2_fit_models import gaussian_var, simulate_gaussian, simulate_garch

logger = get_logger("q2_kupiec_power")

# ── Kupiec LR_uc statistic ────────────────────────────────────────────────────

def _kupiec_lr(k, T, alpha):
    """
    Return LR_uc = −2 ln(L_null / L_alt).

    k     : observed violations
    T     : total observations
    alpha : nominal tail probability (= 1 − confidence)
    """
    if k == 0:
        p_hat = 1e-10
    elif k == T:
        p_hat = 1.0 - 1e-10
    else:
        p_hat = k / T

    try:
        ll_null = k * np.log(alpha)    + (T - k) * np.log(1.0 - alpha)
        ll_alt  = k * np.log(p_hat)   + (T - k) * np.log(1.0 - p_hat)
        lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    except (ValueError, ZeroDivisionError):
        lr = np.nan
    return lr


# ── Vectorised rejection over many paths ─────────────────────────────────────

def _compute_rejections(paths, var_threshold, alpha, crit_value):
    """
    For each simulated path (row), count violations, compute LR_uc, and
    determine whether H0 is rejected at the given critical value.

    Parameters
    ----------
    paths       : ndarray (n_paths, T)
    var_threshold : float — VaR as a negative threshold (r < threshold → violation)
    alpha       : float  — nominal tail probability
    crit_value  : float  — chi^2(1) critical value (default 3.841 at 5%)

    Returns
    -------
    lr_stats   : ndarray (n_paths,)
    rejections : ndarray bool (n_paths,)
    """
    n_paths, T = paths.shape
    violations = (paths < var_threshold).sum(axis=1)   # (n_paths,)

    lr_stats = np.array([_kupiec_lr(int(k), T, alpha) for k in violations])
    rejections = lr_stats > crit_value
    return lr_stats, rejections


# ── Main power estimation ─────────────────────────────────────────────────────

def estimate_power(gauss_params, garch_params,
                   T=None,
                   confidence_levels=None,
                   n_reps=None,
                   seed=None):
    """
    Estimate the size (under H0) and power (under H1) of the Kupiec test
    for each confidence level.

    Parameters
    ----------
    gauss_params : dict from fit_gaussian()
    garch_params : dict from fit_garch()
    T            : int   — path length; defaults to Q1 evaluation period (2893)
    confidence_levels : list of floats; defaults to Q2_CONFIDENCE_LEVELS
    n_reps       : int   — MC replications; defaults to Q2_MC_REPS
    seed         : int   — RNG seed

    Returns
    -------
    results : list of dicts, one per confidence level, containing:
        'confidence', 'alpha', 'T',
        'var_threshold',
        'size'  (empirical type-I error under H0),
        'power' (empirical rejection rate under H1),
        'lr_h0' (ndarray of LR_uc stats under H0),
        'lr_h1' (ndarray of LR_uc stats under H1),
        'viol_h0' (ndarray of violation counts under H0),
        'viol_h1' (ndarray of violation counts under H1),
        'expected_violations',
    """
    if T is None:
        T = 2893
    if confidence_levels is None:
        confidence_levels = Q2_CONFIDENCE_LEVELS
    if n_reps is None:
        n_reps = Q2_MC_REPS
    if seed is None:
        seed = Q2_RANDOM_SEED

    # Chi-squared critical values
    crit_05 = chi2.ppf(0.95, df=1)   # 3.8415
    crit_01 = chi2.ppf(0.99, df=1)   # 6.6349

    # Simulate ONCE, re-use across confidence levels to save time
    logger.info(f"Simulating {n_reps:,} Gaussian paths (T={T}) …")
    paths_h0 = simulate_gaussian(gauss_params, T, n_reps, seed=seed)

    logger.info(f"Simulating {n_reps:,} GARCH paths (T={T}) …")
    paths_h1 = simulate_garch(garch_params, T, n_reps, seed=seed + 1)

    results = []
    for conf in confidence_levels:
        alpha = 1.0 - conf

        var_thr = gaussian_var(gauss_params["sigma"], gauss_params["mu"],
                               alpha=alpha, confidence=conf)

        lr_h0, rej_h0 = _compute_rejections(paths_h0, var_thr, alpha, crit_05)
        lr_h1, rej_h1 = _compute_rejections(paths_h1, var_thr, alpha, crit_05)

        viol_h0 = (paths_h0 < var_thr).sum(axis=1)
        viol_h1 = (paths_h1 < var_thr).sum(axis=1)

        size  = rej_h0.mean()
        power = rej_h1.mean()

        logger.info(
            f"  CI={conf*100:.0f}%: VaR={var_thr:.4f}, "
            f"size={size:.3f}, power={power:.3f}, "
            f"E[viol_H1]={viol_h1.mean():.1f} (expected {alpha*T:.1f})"
        )

        results.append({
            "confidence":          conf,
            "alpha":               alpha,
            "T":                   T,
            "var_threshold":       var_thr,
            "size":                size,
            "power":               power,
            "lr_h0":               lr_h0,
            "lr_h1":               lr_h1,
            "viol_h0":             viol_h0,
            "viol_h1":             viol_h1,
            "expected_violations": alpha * T,
        })

    return results


# ── Save summary CSV ──────────────────────────────────────────────────────────

def save_power_summary(results, filename="power_summary.csv"):
    """Save the scalar power/size summary to a CSV."""
    rows = []
    for r in results:
        rows.append({
            "Confidence Level (%)": f"{r['confidence']*100:.0f}%",
            "Tail Probability (α)": r["alpha"],
            "Sample Size (T)":      r["T"],
            "VaR Threshold":        r["var_threshold"],
            "Expected Violations":  r["expected_violations"],
            "Mean Violations H0":   r["viol_h0"].mean(),
            "Mean Violations H1":   r["viol_h1"].mean(),
            "Empirical Size":       r["size"],
            "Estimated Power":      r["power"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(Q2_OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"Saved power summary → {path}")
    return df


# ── Stand-alone run ────────────────────────────────────────────────────────────

def run_kupiec_power(gauss_params=None, garch_params=None):
    """Estimate power, save CSV, return results list."""
    from logger import log_start, log_end
    log_start(logger, "q2_kupiec_power.py")

    if gauss_params is None or garch_params is None:
        from q2_fit_models import run_fit_models
        gauss_params, garch_params, _ = run_fit_models()

    results = estimate_power(gauss_params, garch_params)
    save_power_summary(results)

    log_end(logger, "q2_kupiec_power.py")
    return results


if __name__ == "__main__":
    from logger import setup_run_logger
    setup_run_logger("q2_kupiec_power")
    run_kupiec_power()
