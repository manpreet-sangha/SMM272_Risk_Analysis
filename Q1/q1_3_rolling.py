"""
Q1 Part 3 — Rolling VaR estimation across all confidence levels.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from config import ROLLING_START_DATE, ROLLING_WINDOW_MONTHS
from logger import get_logger
from q1_2_rolling_window import generate_rolling_windows

logger = get_logger("q1_3_rolling") 


def run_rolling_all_levels(portfolio_returns, confidence_levels, methods):
    """
    Single-pass rolling window loop that computes VaR for every method ×
    every confidence level simultaneously.

    Parameters
    ----------
    portfolio_returns : pd.Series
    confidence_levels : list of float  — e.g. [0.90, 0.95, 0.99]
    methods : list of (tag, label, fn, colour)

    Returns
    -------
    pd.DataFrame
        Index = forecast date.
        Columns = Actual_Return, <tag>_VaR_<ci_pct> for each (tag, ci).
    """
    windows = list(generate_rolling_windows(
        portfolio_returns,
        start_date=ROLLING_START_DATE,
        window_months=ROLLING_WINDOW_MONTHS,
    ))
    total = len(windows)
    logger.info(f"  Total rolling windows  : {total:,}")
    logger.info(f"  First forecast date    : {windows[0][0].date()}")
    logger.info(f"  Last  forecast date    : {windows[-1][0].date()}")
    logger.info(f"  Confidence levels      : {[f'{c*100:.0f}%' for c in confidence_levels]}")

    records = []
    for i, (date, window) in enumerate(windows):
        if i % 500 == 0 or i == total - 1:
            logger.info(f"  [{i+1:>5}/{total}] {date.date()}")

        row = {"Date": date}
        row["Actual_Return"] = (
            float(portfolio_returns.loc[date])
            if date in portfolio_returns.index else np.nan
        )

        for tag, _, fn, _ in methods:
            for ci in confidence_levels:
                var, _ = fn(window, confidence=ci)
                ci_pct = int(round(ci * 100))
                row[f"{tag}_VaR_{ci_pct}"] = var

        records.append(row)

    df = pd.DataFrame(records).set_index("Date")
    return df
