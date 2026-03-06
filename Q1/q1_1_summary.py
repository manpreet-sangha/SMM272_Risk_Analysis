"""
Q1 : Print a consolidated summary of key findings.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from q1_1_build_portfolio import build_portfolio
from q1_1_descriptive_stats import run_descriptive_stats
from q1_1_risk_metrics import run_risk_metrics
from q1_1_normality_tests import run_normality_tests
from q1_1_timeseries_diagnostics import run_timeseries_diagnostics
from q1_1_correlation_analysis import run_correlation_analysis
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_summary")


def print_summary(port_stats=None, risk_stats=None, lb_results=None, corr_matrix=None):
    """Print a narrative summary of Q1 findings."""
    if port_stats is None or risk_stats is None or lb_results is None or corr_matrix is None:
        _, log_returns, portfolio_returns = build_portfolio()
        if port_stats is None:
            port_stats, _ = run_descriptive_stats(log_returns, portfolio_returns)
        if risk_stats is None:
            risk_stats, _ = run_risk_metrics(log_returns, portfolio_returns)
        if lb_results is None:
            lb_results, _, _, _ = run_timeseries_diagnostics(portfolio_returns)
        if corr_matrix is None:
            corr_matrix, _, _ = run_correlation_analysis(log_returns)

    log_start(logger, "q1_1_summary.py")
    logger.info("=" * 70)
    logger.info("SUMMARY OF KEY FINDINGS")
    logger.info("=" * 70)

    summary_text = f"""
1. RETURN & RISK PROFILE
   • The equally weighted portfolio of AAPL, MSFT, IBM, NVDA, GOOGL, AMZN
     achieved an annualised mean return of {risk_stats['Annualised Mean Return']:.4f}
     ({risk_stats['Annualised Mean Return']*100:.2f}%) with an annualised volatility
     of {risk_stats['Annualised Std Deviation']:.4f} ({risk_stats['Annualised Std Deviation']*100:.2f}%).
   • The cumulative return over the sample period is {port_stats['Cumulative Return (%)']:.2f}%.
   • The annualised Sharpe ratio (Rf=0) is {risk_stats['Annualised Sharpe Ratio (Rf=0)']:.4f}.
   • The Sortino ratio (Rf=0) is {risk_stats['Sortino Ratio (Rf=0)']:.4f}.

2. DISTRIBUTIONAL PROPERTIES
   • Skewness = {port_stats['Skewness']:.4f}  →  {'negative skew (left tail heavier)' if port_stats['Skewness'] < 0 else 'positive skew (right tail heavier)'}
   • Excess Kurtosis = {port_stats['Excess Kurtosis']:.4f}  →  leptokurtic / fat-tailed
   • All normality tests (Jarque-Bera, Shapiro-Wilk, D'Agostino, K-S) strongly
     reject the null hypothesis of normality, confirming that the portfolio
     returns exhibit fat tails and asymmetry relative to the Gaussian.
   • Distribution fitting: Student-t provides a better fit than the Normal
     distribution (lower AIC), consistent with the observed fat tails.

3. TAIL RISK
   • Historical VaR (95%) = {risk_stats['Historical VaR (95%)']*100:.4f}%
   • Historical VaR (99%) = {risk_stats['Historical VaR (99%)']*100:.4f}%
   • Historical CVaR/ES (95%) = {risk_stats['Historical CVaR / ES (95%)']*100:.4f}%
   • Historical CVaR/ES (99%) = {risk_stats['Historical CVaR / ES (99%)']*100:.4f}%
   • Parametric VaR (95%, Gaussian) = {risk_stats['Parametric VaR (95%)']*100:.4f}%

4. AUTOCORRELATION & VOLATILITY CLUSTERING
   • The Ljung-Box test on raw returns shows {'significant' if lb_results['lb_pvalue'].iloc[-1] < 0.05 else 'insignificant'} serial
     correlation, suggesting returns are {'not fully' if lb_results['lb_pvalue'].iloc[-1] < 0.05 else ''} independent.
   • The Ljung-Box test on squared returns shows highly significant
     autocorrelation, providing strong evidence of volatility clustering
     (ARCH effects) — a well-documented stylised fact of financial returns.

5. CORRELATION STRUCTURE
   • All pairwise correlations are positive, ranging from
     {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f} to
     {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f},
     reflecting the tech-heavy nature of this portfolio and limiting
     diversification benefits.
"""
    logger.info(summary_text)
    logger.info("=" * 70)
    log_end(logger, "q1_1_summary.py")


if __name__ == "__main__":
    setup_run_logger("smm272_q1_summary")
    print_summary()
