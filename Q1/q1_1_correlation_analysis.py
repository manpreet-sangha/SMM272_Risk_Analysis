"""
Q1 : Covariance / correlation analysis.

Covers: full-sample pairwise correlation matrix, crisis-period correlation
(COVID-19, 2020) versus non-crisis, and average pairwise correlation comparison.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from config import Q1_1_OUTPUT_DIR, CRISIS_START, CRISIS_END
from q1_1_build_portfolio import build_portfolio
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_correlation_analysis")


def _avg_offdiag(corr_mat):
    """Return mean of all off-diagonal pairwise correlations."""
    vals = corr_mat.values
    mask = ~np.eye(len(vals), dtype=bool)
    return vals[mask].mean()


def run_correlation_analysis(log_returns=None):
    """
    Compute and log correlation matrices for the full sample, COVID-19 crisis
    period (2020), and the non-crisis remainder.

    Returns
    -------
    corr_matrix : pd.DataFrame
        Full-sample correlation matrix.
    crisis_corr : pd.DataFrame
        Correlation matrix during the crisis window.
    non_crisis_corr : pd.DataFrame
        Correlation matrix outside the crisis window.
    """
    if log_returns is None:
        _, log_returns, _ = build_portfolio()

    log_start(logger, "q1_1_correlation_analysis.py")

    # ── Full-sample correlation ───────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("CORRELATION MATRIX — FULL SAMPLE")
    logger.info("-" * 70)
    corr_matrix = log_returns.corr()
    logger.info(corr_matrix.to_string())
    corr_matrix.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "correlation_matrix.csv"))

    # ── Crisis period (COVID-19 2020) ─────────────────────────────────────
    logger.info("-" * 70)
    logger.info(f"CORRELATION MATRIX — CRISIS PERIOD  ({CRISIS_START} → {CRISIS_END})")
    logger.info("-" * 70)
    crisis_mask    = ((log_returns.index >= CRISIS_START) &
                      (log_returns.index <= CRISIS_END))
    crisis_returns = log_returns[crisis_mask]
    crisis_corr    = crisis_returns.corr()
    logger.info(f"  Observations in crisis window: {crisis_mask.sum()}")
    logger.info(crisis_corr.to_string())
    crisis_corr.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "correlation_matrix_crisis.csv"))

    # ── Non-crisis period ─────────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("CORRELATION MATRIX — NON-CRISIS PERIOD")
    logger.info("-" * 70)
    non_crisis_returns = log_returns[~crisis_mask]
    non_crisis_corr    = non_crisis_returns.corr()
    logger.info(f"  Observations outside crisis window: {(~crisis_mask).sum()}")
    logger.info(non_crisis_corr.to_string())
    non_crisis_corr.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "correlation_matrix_non_crisis.csv"))

    # ── Average pairwise correlation comparison ───────────────────────────
    avg_full       = _avg_offdiag(corr_matrix)
    avg_crisis     = _avg_offdiag(crisis_corr)
    avg_non_crisis = _avg_offdiag(non_crisis_corr)

    logger.info("-" * 70)
    logger.info("AVERAGE PAIRWISE CORRELATION — PERIOD COMPARISON")
    logger.info("-" * 70)
    logger.info(f"  Full sample avg correlation  : {avg_full:.4f}")
    logger.info(f"  Crisis period avg correlation: {avg_crisis:.4f}")
    logger.info(f"  Non-crisis avg correlation   : {avg_non_crisis:.4f}")
    diff = avg_crisis - avg_non_crisis
    logger.info(f"  Δ (crisis − non-crisis)      : {diff:+.4f}")
    if diff > 0:
        logger.info("  → Correlations rise during the crisis, reducing diversification "
                    "benefit precisely when it is most needed (correlation breakdown).")
    else:
        logger.info("  → Correlations do not significantly increase during the crisis.")

    log_end(logger, "q1_1_correlation_analysis.py")
    return corr_matrix, crisis_corr, non_crisis_corr


if __name__ == "__main__":
    setup_run_logger("smm272_q1_correlation_analysis")
    run_correlation_analysis()
