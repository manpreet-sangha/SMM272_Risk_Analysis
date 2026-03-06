"""
Q1 : Descriptive statistics for portfolio and individual stocks.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config import TICKERS, TRADING_DAYS, Q1_1_OUTPUT_DIR
from q1_1_build_portfolio import build_portfolio
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_descriptive_stats")


def compute_descriptive_stats(series):
    """
    Compute pure descriptive / distributional statistics for a return series.
    """
    return {
        "Observations":          len(series),
        "Daily Mean Return":     series.mean(),
        "Daily Median Return":   series.median(),
        "Skewness":              series.skew(),
        "Excess Kurtosis":       series.kurtosis(),   # excess over Normal
        "Minimum Daily Return":  series.min(),
        "Maximum Daily Return":  series.max(),
        "Q1 (25th percentile)": series.quantile(0.25),
        "Q3 (75th percentile)": series.quantile(0.75),
        "Cumulative Return (%)": (np.exp(series.sum()) - 1) * 100,
    }


def run_descriptive_stats(log_returns=None, portfolio_returns=None):
    """Compute and print descriptive statistics for portfolio & individual stocks."""
    if log_returns is None or portfolio_returns is None:
        _, log_returns, portfolio_returns = build_portfolio()

    log_start(logger, "q1_1_descriptive_stats.py")
    # Portfolio statistics
    port_stats = compute_descriptive_stats(portfolio_returns)

    logger.info(">>> Portfolio Descriptive Statistics")
    for key, val in port_stats.items():
        if isinstance(val, int):
            logger.info(f"  {key:<40s}: {val}")
        else:
            logger.info(f"  {key:<40s}: {val:>12.6f}")

    # Individual stock statistics
    logger.info(">>> Individual Stock Descriptive Statistics")
    individual_stats = {}
    for ticker in TICKERS:
        individual_stats[ticker] = compute_descriptive_stats(log_returns[ticker])

    stats_df = pd.DataFrame(individual_stats)
    stats_df["EW_Portfolio"] = pd.Series(port_stats)
    logger.info(stats_df.to_string())
    stats_df.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "descriptive_statistics.csv"))
    log_end(logger, "q1_1_descriptive_stats.py")

    return port_stats, stats_df


if __name__ == "__main__":
    setup_run_logger("smm272_q1_descriptive_stats")
    run_descriptive_stats()
