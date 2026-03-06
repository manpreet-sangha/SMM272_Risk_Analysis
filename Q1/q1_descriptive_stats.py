"""
Q1 – Step 3: Descriptive statistics for portfolio and individual stocks.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config import TICKERS, TRADING_DAYS, Q1_OUTPUT_DIR
from q1_build_portfolio import build_portfolio


def compute_descriptive_stats(series, annualisation_factor=TRADING_DAYS):
    """Compute comprehensive descriptive statistics for a return series."""
    n = len(series)
    mean_daily = series.mean()
    std_daily = series.std()
    skew = series.skew()
    kurt = series.kurtosis()          # excess kurtosis
    min_ret = series.min()
    max_ret = series.max()
    median_ret = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    var_95 = series.quantile(0.05)    # Historical VaR at 95%
    var_99 = series.quantile(0.01)    # Historical VaR at 99%
    cvar_95 = series[series <= var_95].mean()
    cvar_99 = series[series <= var_99].mean()

    # Annualised figures
    ann_mean = mean_daily * annualisation_factor
    ann_std = std_daily * np.sqrt(annualisation_factor)
    sharpe = ann_mean / ann_std       # assuming Rf ≈ 0

    # Cumulative return
    cum_ret = (np.exp(series.sum()) - 1) * 100  # percentage

    return {
        "Observations": n,
        "Daily Mean Return": mean_daily,
        "Daily Median Return": median_ret,
        "Daily Std Deviation": std_daily,
        "Annualised Mean Return": ann_mean,
        "Annualised Std Deviation": ann_std,
        "Annualised Sharpe Ratio (Rf=0)": sharpe,
        "Skewness": skew,
        "Excess Kurtosis": kurt,
        "Minimum Daily Return": min_ret,
        "Maximum Daily Return": max_ret,
        "Q1 (25th percentile)": q1,
        "Q3 (75th percentile)": q3,
        "Historical VaR (95%)": var_95,
        "Historical VaR (99%)": var_99,
        "Historical CVaR / ES (95%)": cvar_95,
        "Historical CVaR / ES (99%)": cvar_99,
        "Cumulative Return (%)": cum_ret,
    }


def run_descriptive_stats(log_returns=None, portfolio_returns=None):
    """Compute and print descriptive statistics for portfolio & individual stocks."""
    if log_returns is None or portfolio_returns is None:
        _, log_returns, portfolio_returns = build_portfolio()

    # Portfolio statistics
    port_stats = compute_descriptive_stats(portfolio_returns)

    print("\n>>> Portfolio Descriptive Statistics\n")
    for key, val in port_stats.items():
        if isinstance(val, int):
            print(f"  {key:<40s}: {val}")
        else:
            print(f"  {key:<40s}: {val:>12.6f}")

    # Individual stock statistics
    print("\n>>> Individual Stock Descriptive Statistics\n")
    individual_stats = {}
    for ticker in TICKERS:
        individual_stats[ticker] = compute_descriptive_stats(log_returns[ticker])

    stats_df = pd.DataFrame(individual_stats)
    stats_df["EW_Portfolio"] = pd.Series(port_stats)
    print(stats_df.to_string())
    stats_df.to_csv(os.path.join(Q1_OUTPUT_DIR, "descriptive_statistics.csv"))

    return port_stats, stats_df


if __name__ == "__main__":
    run_descriptive_stats()
