"""
Q1 – Step 5: Autocorrelation analysis (Ljung-Box) & correlation matrix.

NOTE: This module is retained for backward compatibility.
New code should use the dedicated modules directly:
  - q1_1_timeseries_diagnostics.run_timeseries_diagnostics()  (Ljung-Box + ADF + KPSS)
  - q1_1_correlation_analysis.run_correlation_analysis()       (correlation matrix + crisis split)
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Q1_1_OUTPUT_DIR
from q1_1_build_portfolio import build_portfolio
from q1_1_timeseries_diagnostics import run_timeseries_diagnostics
from q1_1_correlation_analysis import run_correlation_analysis
from logger import setup_run_logger, get_logger

logger = get_logger("q1_1_autocorrelation")


def run_autocorrelation_analysis(log_returns=None, portfolio_returns=None):
    """
    Backward-compatible wrapper.
    Delegates to q1_1_timeseries_diagnostics and q1_1_correlation_analysis.
    Returns (lb_results, lb_sq_results, corr_matrix) as before.
    """
    if log_returns is None or portfolio_returns is None:
        _, log_returns, portfolio_returns = build_portfolio()

    lb_results, lb_sq_results, _, _ = run_timeseries_diagnostics(portfolio_returns)
    corr_matrix, _, _ = run_correlation_analysis(log_returns)

    return lb_results, lb_sq_results, corr_matrix


if __name__ == "__main__":
    setup_run_logger("smm272_q1_autocorrelation")
    run_autocorrelation_analysis()
