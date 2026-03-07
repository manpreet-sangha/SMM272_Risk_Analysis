"""
Q2 — Power of the Kupiec test as a function of sample size T.

The key weakness of the Kupiec (1995) POF test is its low power at
short evaluation horizons (T ≈ 250, i.e. one trading year), which is
the standard regulatory backtesting window under Basel II/III.

This module repeats the core Monte Carlo simulation across a grid of
sample sizes T ∈ Q2_T_GRID and records:
  - Estimated power at each T for both H1 (GARCH) and, as a reference,
    the theoretical power under a fixed alternative violation rate p1.

The result illustrates the well-known finding of Kupiec (1995) that the
test has insufficient power to distinguish a mis-specified model from a
correct one over a single year of daily data.

References
----------
Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives, 3(2), 73–84.
Christoffersen, P. F. (1998). Evaluating interval forecasts.
    International Economic Review, 39(4), 841–862.
Berkowitz, J., & O'Brien, J. (2002). How accurate are Value-at-Risk
    models at commercial banks? Journal of Finance, 57(3), 1093–1111.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import chi2, binom

from config import (Q2_CONFIDENCE_LEVELS, Q2_MC_REPS, Q2_RANDOM_SEED,
                    Q2_T_GRID, Q2_OUTPUT_DIR)
from logger import get_logger
from q2_fit_models import gaussian_var, simulate_gaussian, simulate_garch
from q2_kupiec_power import _compute_rejections

logger = get_logger("q2_power_vs_T")

CRIT_05 = chi2.ppf(0.95, df=1)   # 3.8415


def theoretical_power(p0, p1, T, crit=CRIT_05):
    """
    Approximate analytical power of the Kupiec test against a fixed
    alternative where the true violation rate is p1 (> p0).

    Under H1 with true rate p1, the number of violations k ~ Binomial(T, p1).
    The test rejects when LR_uc(k, T, p0) > crit.
    Power = Σ_k P(k | T, p1) × 1[LR_uc(k,T,p0) > crit]

    This is exact for i.i.d. Bernoulli violations but approximate for
    GARCH (where violation clustering means the effective variance of k
    is higher than Binomial).
    """
    k_vals = np.arange(0, T + 1)
    probs  = binom.pmf(k_vals, T, p1)

    # Vectorised LR_uc
    p_hat = np.where(k_vals == 0, 1e-10, k_vals / T)
    p_hat = np.where(k_vals == T, 1 - 1e-10, p_hat)
    ll_null = k_vals * np.log(p0) + (T - k_vals) * np.log(1 - p0)
    ll_alt  = k_vals * np.log(p_hat) + (T - k_vals) * np.log(1 - p_hat)
    lr      = np.maximum(-2.0 * (ll_null - ll_alt), 0.0)

    power = (probs * (lr > crit)).sum()
    return float(power)


def power_vs_T_simulation(gauss_params, garch_params,
                           T_grid=None,
                           confidence_levels=None,
                           n_reps=None,
                           seed=None):
    """
    Estimate simulated power for each (T, confidence level) combination.

    For each T:
      - Simulate n_reps Gaussian paths (H0) and GARCH paths (H1) of length T.
      - Apply fixed Gaussian VaR thresholds.
      - Record rejection rates (size and power).
      - Also compute exact Binomial analytical power for a reference p1
        equal to the mean empirical violation rate under GARCH at full T.

    Returns
    -------
    df : pd.DataFrame with columns
        ['T', 'confidence', 'alpha', 'size', 'power_sim',
         'power_binomial', 'mean_viol_h1', 'expected_viol']
    """
    if T_grid is None:
        T_grid = Q2_T_GRID
    if confidence_levels is None:
        confidence_levels = Q2_CONFIDENCE_LEVELS
    if n_reps is None:
        n_reps = Q2_MC_REPS
    if seed is None:
        seed = Q2_RANDOM_SEED

    rows = []
    for idx_T, T in enumerate(T_grid):
        logger.info(f"  T = {T} — simulating {n_reps:,} paths …")

        paths_h0 = simulate_gaussian(gauss_params, T, n_reps, seed=seed + idx_T * 100)
        paths_h1 = simulate_garch(garch_params, T, n_reps, seed=seed + idx_T * 100 + 1)

        for conf in confidence_levels:
            alpha = 1.0 - conf
            var_thr = gaussian_var(gauss_params["sigma"], gauss_params["mu"],
                                   alpha=alpha, confidence=conf)

            _, rej_h0 = _compute_rejections(paths_h0, var_thr, alpha, CRIT_05)
            _, rej_h1 = _compute_rejections(paths_h1, var_thr, alpha, CRIT_05)

            viol_h1      = (paths_h1 < var_thr).sum(axis=1)
            mean_viol_h1 = viol_h1.mean()
            p1_empirical = mean_viol_h1 / T

            power_binom = theoretical_power(alpha, p1_empirical, T)

            rows.append({
                "T":               T,
                "confidence":      conf,
                "alpha":           alpha,
                "size":            rej_h0.mean(),
                "power_sim":       rej_h1.mean(),
                "power_binomial":  power_binom,
                "mean_viol_h1":    mean_viol_h1,
                "expected_viol":   alpha * T,
                "p1_empirical":    p1_empirical,
            })

    df = pd.DataFrame(rows)
    return df


def save_power_vs_T(df, filename="power_vs_T.csv"):
    path = os.path.join(Q2_OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"Saved power-vs-T table → {path}")
    return path


def run_power_vs_T(gauss_params=None, garch_params=None):
    """Run power-vs-T analysis, save CSV, return DataFrame."""
    from logger import log_start, log_end
    log_start(logger, "q2_power_vs_T.py")

    if gauss_params is None or garch_params is None:
        from q2_fit_models import run_fit_models
        gauss_params, garch_params, _ = run_fit_models()

    df = power_vs_T_simulation(gauss_params, garch_params)
    save_power_vs_T(df)

    log_end(logger, "q2_power_vs_T.py")
    return df


if __name__ == "__main__":
    from logger import setup_run_logger
    setup_run_logger("q2_power_vs_T")
    run_power_vs_T()
