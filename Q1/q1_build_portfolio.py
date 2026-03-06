"""
Q1 – Step 2: Build an equally weighted portfolio from log returns.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config import TICKERS, Q1_OUTPUT_DIR
from q1_download_prices import download_prices


def load_prices():
    """Load prices from CSV if available, otherwise download."""
    csv_path = os.path.join(Q1_OUTPUT_DIR, "adjusted_close_prices.csv")
    if os.path.exists(csv_path):
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"Loaded prices from cache ({prices.shape[0]} days)\n")
        return prices
    return download_prices()


def build_portfolio(prices=None):
    """Compute individual log returns and equally weighted portfolio returns."""
    if prices is None:
        prices = load_prices()

    print("-" * 70)
    print("EQUALLY WEIGHTED PORTFOLIO (1/6 each)")
    print("-" * 70)

    # Individual log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Equal weights
    n_assets = len(TICKERS)
    weights = np.array([1 / n_assets] * n_assets)
    portfolio_returns = log_returns.dot(weights)
    portfolio_returns.name = "EW_Portfolio"

    # Save for reuse
    log_returns.to_csv(os.path.join(Q1_OUTPUT_DIR, "log_returns.csv"))
    portfolio_returns.to_csv(os.path.join(Q1_OUTPUT_DIR, "portfolio_returns.csv"),
                             header=True)

    print(f"  Log returns computed: {log_returns.shape[0]} observations")
    print(f"  Portfolio returns saved to output/\n")

    return prices, log_returns, portfolio_returns


if __name__ == "__main__":
    build_portfolio()
