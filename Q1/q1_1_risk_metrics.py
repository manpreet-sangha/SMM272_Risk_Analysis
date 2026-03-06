"""
Q1 : Risk metrics.

Covers: daily/annualised standard deviation, Sharpe ratio, Sortino ratio,
historical VaR/CVaR (95 % & 99 %), and parametric (Gaussian) VaR.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import TICKERS, TRADING_DAYS, Q1_1_OUTPUT_DIR
from q1_1_build_portfolio import build_portfolio
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_risk_metrics")


def compute_risk_metrics(series, annualisation_factor=TRADING_DAYS):
    """
    Compute risk metrics for a return series.

    Parameters
    ----------
    series : pd.Series
        Daily log-return series.
    annualisation_factor : int
        Number of trading days per year (default 252).

    Returns
    -------
    dict
        Risk metrics keyed by descriptive name.
    """
    std_daily  = series.std()
    ann_mean   = series.mean() * annualisation_factor
    ann_std    = std_daily * np.sqrt(annualisation_factor)

    # Sharpe ratio (Rf = 0)
    sharpe = ann_mean / ann_std if ann_std != 0 else np.nan

    # Sortino ratio: annualised return divided by annualised downside deviation
    downside_returns    = series[series < 0]
    ann_downside_std    = downside_returns.std() * np.sqrt(annualisation_factor)
    sortino = ann_mean / ann_downside_std if ann_downside_std != 0 else np.nan

    # Historical VaR and CVaR (Expected Shortfall)
    var_95  = series.quantile(0.05)
    var_99  = series.quantile(0.01)
    cvar_95 = series[series <= var_95].mean()
    cvar_99 = series[series <= var_99].mean()

    # Parametric (Gaussian) VaR — assumes normally distributed returns
    mu = series.mean()
    param_var_95 = mu + norm.ppf(0.05) * std_daily   # = mu - 1.645σ
    param_var_99 = mu + norm.ppf(0.01) * std_daily   # = mu - 2.326σ

    return {
        "Daily Std Deviation":           std_daily,
        "Annualised Mean Return":        ann_mean,
        "Annualised Std Deviation":      ann_std,
        "Annualised Sharpe Ratio (Rf=0)": sharpe,
        "Sortino Ratio (Rf=0)":          sortino,
        "Historical VaR (95%)":          var_95,
        "Historical VaR (99%)":          var_99,
        "Historical CVaR / ES (95%)":    cvar_95,
        "Historical CVaR / ES (99%)":    cvar_99,
        "Parametric VaR (95%)":          param_var_95,
        "Parametric VaR (99%)":          param_var_99,
    }


def run_risk_metrics(log_returns=None, portfolio_returns=None):
    """
    Compute and log risk metrics for the EW portfolio and each constituent.

    Returns
    -------
    port_risk : dict
        Risk metrics for the EW portfolio.
    risk_df : pd.DataFrame
        Risk metrics for all assets + EW portfolio (columns = assets).
    """
    if log_returns is None or portfolio_returns is None:
        _, log_returns, portfolio_returns = build_portfolio()

    log_start(logger, "q1_1_risk_metrics.py")

    port_risk = compute_risk_metrics(portfolio_returns)

    logger.info("-" * 70)
    logger.info("RISK METRICS — EW PORTFOLIO")
    logger.info("-" * 70)
    for key, val in port_risk.items():
        logger.info(f"  {key:<40s}: {val:>12.6f}")

    logger.info("-" * 70)
    logger.info("RISK METRICS — INDIVIDUAL STOCKS")
    logger.info("-" * 70)
    individual_risk = {ticker: compute_risk_metrics(log_returns[ticker])
                       for ticker in TICKERS}
    risk_df = pd.DataFrame(individual_risk)
    risk_df["EW_Portfolio"] = pd.Series(port_risk)
    logger.info(risk_df.to_string())

    risk_df.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "risk_metrics.csv"))
    logger.info(f"  Risk metrics saved to: {Q1_1_OUTPUT_DIR}/risk_metrics.csv")

    log_end(logger, "q1_1_risk_metrics.py")
    return port_risk, risk_df


if __name__ == "__main__":
    setup_run_logger("smm272_q1_risk_metrics")
    run_risk_metrics()
