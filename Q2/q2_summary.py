"""
Q2 — Summary: print and save a consolidated report of all Q2 results.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from config import Q2_OUTPUT_DIR
from logger import get_logger, log_start, log_end

logger = get_logger("q2_summary")

BANNER = "=" * 62


def print_fitted_params(gauss_params, garch_params):
    logger.info(BANNER)
    logger.info("FITTED MODEL PARAMETERS")
    logger.info(BANNER)
    logger.info("Gaussian H0 (constant volatility):")
    logger.info(f"  μ     = {gauss_params['mu']:.6f}")
    logger.info(f"  σ     = {gauss_params['sigma']:.6f}  "
                f"(annualised ≈ {gauss_params['sigma']*np.sqrt(252)*100:.2f}%)")
    logger.info("")
    logger.info("GARCH(1,1) H1:")
    logger.info(f"  μ     = {garch_params['mu']:.6f}")
    logger.info(f"  ω     = {garch_params['omega']:.2e}")
    logger.info(f"  α     = {garch_params['alpha']:.4f}")
    logger.info(f"  β     = {garch_params['beta']:.4f}")
    logger.info(f"  α+β   = {garch_params['persistence']:.4f}  (persistence)")
    logger.info(f"  σ_LR  = {garch_params['sigma_uncond']:.6f}  "
                f"(long-run std, annualised ≈ "
                f"{garch_params['sigma_uncond']*np.sqrt(252)*100:.2f}%)")
    logger.info(f"  LogL  = {garch_params['loglik']:.2f}")
    logger.info(f"  AIC   = {garch_params['aic']:.2f}")
    logger.info(f"  BIC   = {garch_params['bic']:.2f}")


def print_power_summary(power_results):
    logger.info(BANNER)
    logger.info("KUPIEC POWER SUMMARY  (T = 2893, B = 10,000 replications)")
    logger.info(BANNER)
    logger.info(f"{'CI':>6s} | {'α':>6s} | {'VaR thr':>10s} | "
                f"{'E[viol]':>8s} | {'E[viol|H1]':>11s} | "
                f"{'Size':>6s} | {'Power':>6s}")
    logger.info("-" * 62)
    for r in power_results:
        logger.info(
            f"{r['confidence']*100:>5.0f}% | "
            f"{r['alpha']:>6.4f} | "
            f"{r['var_threshold']:>10.4f} | "
            f"{r['expected_violations']:>8.1f} | "
            f"{r['viol_h1'].mean():>11.1f} | "
            f"{r['size']:>6.3f} | "
            f"{r['power']:>6.3f}"
        )


def print_power_vs_T(df_T):
    logger.info(BANNER)
    logger.info("POWER vs. SAMPLE SIZE T  (99% CI)")
    logger.info(BANNER)
    sub = df_T[df_T["confidence"] == 0.99].sort_values("T")
    logger.info(f"{'T':>6s} | {'Size':>6s} | {'Power (sim)':>12s} | {'Power (Binom)':>14s}")
    logger.info("-" * 45)
    for _, row in sub.iterrows():
        logger.info(
            f"{int(row['T']):>6d} | "
            f"{row['size']:>6.3f} | "
            f"{row['power_sim']:>12.3f} | "
            f"{row['power_binomial']:>14.3f}"
        )


def save_full_summary_csv(power_results, df_T, df_persist):
    """Save a single consolidated CSV with all scalar results."""
    rows = []
    # Main power results
    for r in power_results:
        rows.append({
            "Table":    "Power Summary (T=2893)",
            "CI":       f"{r['confidence']*100:.0f}%",
            "T":        r["T"],
            "alpha":    r["alpha"],
            "VaR_threshold": r["var_threshold"],
            "Expected_violations": r["expected_violations"],
            "Mean_viol_H1": r["viol_h1"].mean(),
            "Empirical_size": r["size"],
            "Estimated_power": r["power"],
        })

    df_summary = pd.DataFrame(rows)
    path = os.path.join(Q2_OUTPUT_DIR, "q2_full_summary.csv")
    df_summary.to_csv(path, index=False)
    logger.info(f"Saved full summary → {path}")
    return df_summary


def run_summary(gauss_params, garch_params, power_results, df_T, df_persist):
    log_start(logger, "q2_summary.py")
    print_fitted_params(gauss_params, garch_params)
    print_power_summary(power_results)
    print_power_vs_T(df_T)
    save_full_summary_csv(power_results, df_T, df_persist)
    log_end(logger, "q2_summary.py")
