"""
SMM272 Risk Analysis Coursework 2025-2026 — Question 1
Master script: runs every Q1 step in sequence.

Modules
-------
config.py                → shared constants (tickers, dates, paths)
q1_download_prices.py    → Step 1  download adjusted closing prices
q1_build_portfolio.py    → Step 2  build equally weighted portfolio
q1_descriptive_stats.py  → Step 3  descriptive statistics
q1_normality_tests.py    → Step 4  normality tests
q1_autocorrelation.py    → Step 5  autocorrelation & correlation matrix
q1_visualisations.py     → Step 6  all 9 figures
q1_summary.py            → Step 7  narrative summary of findings
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from q1_download_prices import download_prices
from q1_build_portfolio import build_portfolio
from q1_descriptive_stats import run_descriptive_stats
from q1_normality_tests import run_normality_tests
from q1_autocorrelation import run_autocorrelation_analysis
from q1_visualisations import generate_visualisations
from q1_summary import print_summary
from config import Q1_OUTPUT_DIR


def main():
    print("=" * 70)
    print("SMM272 Risk Analysis Coursework — Q1: Statistical Analysis")
    print("=" * 70)

    # Step 1 – Download prices
    prices = download_prices()

    # Step 2 – Build portfolio
    prices, log_returns, portfolio_returns = build_portfolio(prices)

    # Step 3 – Descriptive statistics
    port_stats, stats_df = run_descriptive_stats(log_returns, portfolio_returns)

    # Step 4 – Normality tests
    normality_results = run_normality_tests(portfolio_returns)

    # Step 5 – Autocorrelation analysis & correlation matrix
    lb_results, lb_sq_results, corr_matrix = run_autocorrelation_analysis(
        log_returns, portfolio_returns)

    # Step 6 – Visualisations
    generate_visualisations(prices, log_returns, portfolio_returns, corr_matrix)

    # Step 7 – Summary
    print_summary(port_stats, lb_results, corr_matrix)

    print(f"\nAll outputs saved to: {Q1_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
