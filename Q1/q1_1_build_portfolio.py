"""
Q1 : Build an equally weighted portfolio from log returns.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config import TICKERS, Q1_1_OUTPUT_DIR
from q1_1_download_prices import download_prices
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_build_portfolio")


def load_prices():
    """Load prices from CSV if available, otherwise download."""
    csv_path = os.path.join(Q1_1_OUTPUT_DIR, "adjusted_close_prices.csv")
    if os.path.exists(csv_path):
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded prices from cache ({prices.shape[0]} days)")
        return prices
    return download_prices()


def build_portfolio(prices=None):
    """Compute individual log returns and equally weighted portfolio returns."""
    log_start(logger, "q1_1_build_portfolio.py")
    if prices is None:
        prices = load_prices()

    logger.info("-" * 70)
    logger.info("EQUALLY WEIGHTED PORTFOLIO (1/6 each)")
    logger.info("-" * 70)

    # Individual log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Equal weights
    n_assets = len(TICKERS)
    weights = np.array([1 / n_assets] * n_assets)
    portfolio_returns = log_returns.dot(weights)
    portfolio_returns.name = "EW_Portfolio"

    # Save for reuse
    log_returns.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "log_returns.csv"))
    portfolio_returns.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "portfolio_returns.csv"),
                             header=True)

    logger.info(f"  Log returns computed: {log_returns.shape[0]} observations")
    logger.info("  Portfolio returns saved to output/")
    log_end(logger, "q1_1_build_portfolio.py")

    return prices, log_returns, portfolio_returns


if __name__ == "__main__":
    setup_run_logger("smm272_q1_build_portfolio")
    build_portfolio()
