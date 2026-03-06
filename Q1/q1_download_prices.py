"""
Q1 – Step 1: Download adjusted closing prices from Yahoo Finance.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf
from config import TICKERS, START_DATE, END_DATE, Q1_OUTPUT_DIR


def download_prices():
    """Download adjusted closing prices for the configured tickers and period."""
    print("=" * 70)
    print("SMM272 Risk Analysis Coursework — Q1: Download Prices")
    print("=" * 70)
    print(f"\nDownloading adjusted closing prices for {TICKERS}")
    print(f"Period: {START_DATE} to {END_DATE}\n")

    prices = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    prices = prices[TICKERS]  # ensure consistent column order
    prices.dropna(inplace=True)

    print(f"Data downloaded: {prices.shape[0]} trading days, {prices.shape[1]} stocks")

    # Save to CSV for reuse
    out_path = os.path.join(Q1_OUTPUT_DIR, "adjusted_close_prices.csv")
    prices.to_csv(out_path)
    print(f"Saved to: {out_path}\n")

    return prices


if __name__ == "__main__":
    download_prices()
