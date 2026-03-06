"""
SMM272 Risk Analysis Coursework 2025-2026 — Question 1
Master script: runs every Q1 module in sequence.

Modular structure
-----------------
config.py                        → shared constants (tickers, dates, paths)
q1_1_download_prices.py            → Step 1   download adjusted closing prices
q1_1_build_portfolio.py            → Step 2   build equally weighted portfolio
q1_1_descriptive_stats.py          → Module 1 mean, median, skewness, kurtosis
q1_1_risk_metrics.py               → Module 2 std, VaR, CVaR, Sharpe, Sortino
q1_1_normality_tests.py            → Module 3 normality tests + distribution fitting
q1_1_timeseries_diagnostics.py     → Module 4 Ljung-Box, ADF, KPSS stationarity
q1_1_correlation_analysis.py       → Module 5 correlation matrix + crisis split
q1_1_visualisations.py             →          all 9 figures
q1_1_portfolio_conclusions.py      → Module 6 risk-return, diversification benefits
q1_1_summary.py                    →          narrative summary of all findings
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger import setup_run_logger, get_logger, log_start, log_end
from q1_1_download_prices import download_prices
from q1_1_build_portfolio import build_portfolio
from q1_1_descriptive_stats import run_descriptive_stats
from q1_1_risk_metrics import run_risk_metrics
from q1_1_normality_tests import run_normality_tests
from q1_1_timeseries_diagnostics import run_timeseries_diagnostics
from q1_1_correlation_analysis import run_correlation_analysis
from q1_1_visualisations import generate_visualisations
from q1_1_portfolio_conclusions import run_portfolio_conclusions
from q1_1_summary import print_summary
from config import Q1_1_OUTPUT_DIR

logger = get_logger("q1_1_statistical_analysis")


def main():
    log_path = setup_run_logger("smm272_q1")
    log_start(logger, "q1_1_statistical_analysis.py")
    logger.info("=" * 70)
    logger.info("SMM272 Risk Analysis Coursework — Q1: Statistical Analysis")
    logger.info("=" * 70)
    logger.info(f"Run log: {log_path}")

    # Step 1 – Download prices
    prices = download_prices()

    # Step 2 – Build portfolio
    prices, log_returns, portfolio_returns = build_portfolio(prices)

    # Module 1 – Descriptive statistics (mean, median, skewness, kurtosis)
    port_stats, stats_df = run_descriptive_stats(log_returns, portfolio_returns)

    # Module 2 – Risk metrics (std, VaR, CVaR, Sharpe, Sortino)
    port_risk, risk_df = run_risk_metrics(log_returns, portfolio_returns)

    # Module 3 – Normality tests + distribution fitting
    normality_results = run_normality_tests(portfolio_returns)

    # Module 4 – Time-series diagnostics (Ljung-Box, ADF, KPSS)
    lb_results, lb_sq_results, adf_result, kpss_result = run_timeseries_diagnostics(
        portfolio_returns)

    # Module 5 – Correlation analysis (full sample + crisis/non-crisis split)
    corr_matrix, crisis_corr, non_crisis_corr = run_correlation_analysis(log_returns)

    # Visualisations
    generate_visualisations(prices, log_returns, portfolio_returns, corr_matrix)

    # Module 6 – Portfolio-level conclusions (risk-return, diversification)
    run_portfolio_conclusions(log_returns, portfolio_returns, port_risk)

    # Narrative summary
    print_summary(port_stats, risk_stats=port_risk, lb_results=lb_results,
                  corr_matrix=corr_matrix)

    logger.info(f"All outputs saved to: {Q1_1_OUTPUT_DIR}")
    log_end(logger, "q1_1_statistical_analysis.py")


if __name__ == "__main__":
    main()
