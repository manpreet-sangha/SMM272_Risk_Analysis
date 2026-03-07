"""
Q1 Part 4 — Logging helpers for all backtest results.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from logger import get_logger

logger = get_logger("q1_4_logging")

_YN  = lambda v: ("YES ***" if v else "No")        if isinstance(v, bool)  else "n/a"
_FMT = lambda x, d=4: f"{x:.{d}f}"                if not np.isnan(float(x) if x is not None else float("nan")) else "n/a"


def _safe(val, fmt=".4f"):
    """Format a numeric value or return 'n/a' if NaN / None."""
    try:
        f = float(val)
        return f"nan" if np.isnan(f) else f"{f:{fmt}}"
    except (TypeError, ValueError):
        return "n/a"


def log_backtest_results(results_df):
    """
    Log all four sets of backtest results as formatted tables.

    Parameters
    ----------
    results_df : pd.DataFrame — combined output of run_all_backtests()
    """

    # ── 1. Kupiec POF ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("KUPIEC POF TEST  (H0: unconditional coverage is correct)")
    logger.info("  LR_uc ~ chi^2(1)   |  critical value at 5%: 3.841")
    logger.info("=" * 80)
    logger.info(
        f"  {'Method':<22} {'CI%':>5} {'k':>5} {'N':>6} "
        f"{'LR_uc':>8} {'p-val':>8} {'Reject':>8}"
    )
    logger.info("  " + "-" * 65)
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['Method']:<22} {row['Confidence (%)']:>5.0f}"
            f" {row['Violations (k)']:>5,d} {row['N']:>6,d}"
            f" {_safe(row['LR_uc']):>8} {_safe(row['p_uc']):>8}"
            f" {_YN(row['Reject_uc']):>8}"
        )

    # ── 2. Christoffersen Independence ───────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("CHRISTOFFERSEN INDEPENDENCE TEST  (H0: violations are independent)")
    logger.info("  LR_ind ~ chi^2(1)  |  critical value at 5%: 3.841")
    logger.info("=" * 80)
    logger.info(
        f"  {'Method':<22} {'CI%':>5} {'LR_ind':>8} {'p-val':>8} {'Reject':>8}"
    )
    logger.info("  " + "-" * 55)
    for _, row in results_df.iterrows():
        rej = row.get("Reject_ind", np.nan)
        logger.info(
            f"  {row['Method']:<22} {row['Confidence (%)']:>5.0f}"
            f" {_safe(row['LR_ind']):>8} {_safe(row['p_ind']):>8}"
            f" {_YN(rej):>8}"
        )

    # ── 3. Conditional Coverage (CC) ─────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("CHRISTOFFERSEN CONDITIONAL COVERAGE (CC) TEST")
    logger.info("  LR_cc = LR_uc + LR_ind ~ chi^2(2)  |  critical value at 5%: 5.991")
    logger.info("=" * 80)
    logger.info(
        f"  {'Method':<22} {'CI%':>5} {'LR_cc':>8} {'p-val':>8} {'Reject':>8}"
    )
    logger.info("  " + "-" * 55)
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['Method']:<22} {row['Confidence (%)']:>5.0f}"
            f" {_safe(row['LR_cc']):>8} {_safe(row['p_cc']):>8}"
            f" {_YN(row['Reject_cc']):>8}"
        )

    # ── 4. Duration Test ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("CHRISTOFFERSEN-PELLETIER DURATION TEST  (Weibull vs Exponential)")
    logger.info("  LR_dur ~ chi^2(2)  |  critical value at 5%: 5.991")
    logger.info("  H0: durations i.i.d. Exp(p)  |  HA: Weibull(a, b) with b != 1")
    logger.info("=" * 80)
    logger.info(
        f"  {'Method':<22} {'CI%':>5} {'LR_dur':>8} {'p-val':>8} {'Reject':>8}"
    )
    logger.info("  " + "-" * 55)
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['Method']:<22} {row['Confidence (%)']:>5.0f}"
            f" {_safe(row['LR_dur']):>8} {_safe(row['p_dur']):>8}"
            f" {_YN(row['Reject_dur']):>8}"
        )

    # ── 5. DQ Test ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("ENGLE-MANGANELLI DYNAMIC QUANTILE (DQ) TEST")
    logger.info("  DQ ~ chi^2(K+1), K=4 lags  |  critical value at 5%: 11.070")
    logger.info("  H0: demeaned hit sequence H_t is not predictable by its lags")
    logger.info("=" * 80)
    logger.info(
        f"  {'Method':<22} {'CI%':>5} {'DQ_stat':>8} {'p-val':>8} {'Reject':>8}"
    )
    logger.info("  " + "-" * 55)
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['Method']:<22} {row['Confidence (%)']:>5.0f}"
            f" {_safe(row['DQ_stat']):>8} {_safe(row['p_dq']):>8}"
            f" {_YN(row['Reject_dq']):>8}"
        )
