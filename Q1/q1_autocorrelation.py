"""
Q1 – Step 5: Autocorrelation analysis (Ljung-Box) & correlation matrix.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from statsmodels.stats.diagnostic import acorr_ljungbox
from config import Q1_OUTPUT_DIR
from q1_build_portfolio import build_portfolio


def run_autocorrelation_analysis(log_returns=None, portfolio_returns=None):
    """Ljung-Box tests on returns & squared returns, plus correlation matrix."""
    if log_returns is None or portfolio_returns is None:
        _, log_returns, portfolio_returns = build_portfolio()

    # ── Ljung-Box on raw returns ───────────────────────────────────────────
    print("\n" + "-" * 70)
    print("AUTOCORRELATION ANALYSIS")
    print("-" * 70)

    lb_results = acorr_ljungbox(portfolio_returns, lags=[5, 10, 15, 20],
                                return_df=True)
    print("\n  Ljung-Box test on portfolio returns:")
    print(lb_results.to_string())

    # ── Ljung-Box on squared returns (volatility clustering) ───────────────
    lb_sq_results = acorr_ljungbox(portfolio_returns ** 2, lags=[5, 10, 15, 20],
                                   return_df=True)
    print("\n  Ljung-Box test on squared returns (volatility clustering):")
    print(lb_sq_results.to_string())

    # ── Correlation matrix ─────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("CORRELATION MATRIX (LOG RETURNS)")
    print("-" * 70)
    corr_matrix = log_returns.corr()
    print(f"\n{corr_matrix.to_string()}")
    corr_matrix.to_csv(os.path.join(Q1_OUTPUT_DIR, "correlation_matrix.csv"))

    return lb_results, lb_sq_results, corr_matrix


if __name__ == "__main__":
    run_autocorrelation_analysis()
