"""
Q1 : Download adjusted closing prices from Yahoo Finance.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf
from config import TICKERS, START_DATE, END_DATE, Q1_1_OUTPUT_DIR
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_download_prices")


def download_prices():
    """Download adjusted closing prices for the configured tickers and period."""
    log_start(logger, "q1_1_download_prices.py")
    logger.info("=" * 70)
    logger.info("SMM272 Risk Analysis Coursework — Q1: Download Prices")
    logger.info("=" * 70)
    logger.info(f"Downloading adjusted closing prices for {TICKERS}")
    logger.info(f"Period: {START_DATE} to {END_DATE}")

    prices = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    prices = prices[TICKERS]  # ensure consistent column order
    prices.dropna(inplace=True)

    logger.info(f"Data downloaded: {prices.shape[0]} trading days, {prices.shape[1]} stocks")

    # Save to CSV for reuse
    out_path = os.path.join(Q1_1_OUTPUT_DIR, "adjusted_close_prices.csv")
    prices.to_csv(out_path)
    logger.info(f"Saved to: {out_path}")
    log_end(logger, "q1_1_download_prices.py")

    return prices


if __name__ == "__main__":
    setup_run_logger("smm272_q1_download_prices")
    download_prices()
