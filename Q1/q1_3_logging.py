"""
Q1 Part 3 — Logging helpers for violation and backtest summaries.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from logger import get_logger

logger = get_logger("q1_3_logging")


def log_violations(violations_df):
    """
    Log the violations summary table (counts and rates only).

    Parameters
    ----------
    violations_df : pd.DataFrame — output of count_violations()
    """
    logger.info("\n" + "=" * 80)
    logger.info("VaR VIOLATIONS SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"  {'Method':<22} {'CI%':>5} {'N':>6} {'k':>6} {'Exp':>6}"
        f" {'Rate%':>8} {'Nom%':>6} {'Excess pp':>10}"
    )
    logger.info("  " + "-" * 75)
    for _, row in violations_df.iterrows():
        logger.info(
            f"  {row['Method']:<22} {row['Confidence (%)']:>5.0f}"
            f" {row['N']:>6,d} {row['Violations (k)']:>6,d}"
            f" {row['Expected (\u2248)']:>6.1f} {row['Observed Rate (%)']:>8.3f}%"
            f" {row['Nominal Rate (%)']:>6.1f}% {row['Excess (pp)']:>+9.3f}pp"
        )
