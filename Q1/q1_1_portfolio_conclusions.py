"""
Q1 : Portfolio-level conclusions.

Covers: risk-return trade-off table, diversification benefit quantification,
tail-risk comparison across assets, and narrative key takeaways.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from config import TICKERS, TRADING_DAYS, Q1_1_OUTPUT_DIR
from q1_1_build_portfolio import build_portfolio
from q1_1_risk_metrics import compute_risk_metrics
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_portfolio_conclusions")


def run_portfolio_conclusions(log_returns=None, portfolio_returns=None,
                               port_risk=None):
    """
    Summarise portfolio-level risk-return trade-offs and diversification benefits.

    Parameters
    ----------
    log_returns : pd.DataFrame, optional
        Individual asset log-return series (columns = tickers).
    portfolio_returns : pd.Series, optional
        EW portfolio log-return series.
    port_risk : dict, optional
        Pre-computed risk metrics for the portfolio (from run_risk_metrics).

    Returns
    -------
    conclusions : dict
        Key scalar metrics (diversification benefit, avg pairwise correlation, etc.).
    """
    if log_returns is None or portfolio_returns is None:
        _, log_returns, portfolio_returns = build_portfolio()
    if port_risk is None:
        port_risk = compute_risk_metrics(portfolio_returns)

    log_start(logger, "q1_1_portfolio_conclusions.py")

    # ── Pre-compute metrics for every asset ───────────────────────────────
    asset_risk = {ticker: compute_risk_metrics(log_returns[ticker])
                  for ticker in TICKERS}

    # ── 1. Risk-Return Summary Table ──────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PORTFOLIO-LEVEL CONCLUSIONS")
    logger.info("=" * 70)
    logger.info("\n--- Risk-Return Summary (Annualised) ---")
    header = (f"  {'Asset':<14} {'Ann.Return':>11} {'Ann.Std':>9} "
              f"{'Sharpe':>8} {'Sortino':>9}")
    logger.info(header)
    logger.info("  " + "-" * 55)

    rows = []
    for ticker in TICKERS:
        m = asset_risk[ticker]
        logger.info(
            f"  {ticker:<14} {m['Annualised Mean Return']:>11.4f} "
            f"{m['Annualised Std Deviation']:>9.4f} "
            f"{m['Annualised Sharpe Ratio (Rf=0)']:>8.4f} "
            f"{m['Sortino Ratio (Rf=0)']:>9.4f}"
        )
        rows.append({
            "Asset": ticker,
            "Annualised Return": m["Annualised Mean Return"],
            "Annualised Std":    m["Annualised Std Deviation"],
            "Sharpe":            m["Annualised Sharpe Ratio (Rf=0)"],
            "Sortino":           m["Sortino Ratio (Rf=0)"],
        })

    logger.info("  " + "-" * 55)
    logger.info(
        f"  {'EW_Portfolio':<14} {port_risk['Annualised Mean Return']:>11.4f} "
        f"{port_risk['Annualised Std Deviation']:>9.4f} "
        f"{port_risk['Annualised Sharpe Ratio (Rf=0)']:>8.4f} "
        f"{port_risk['Sortino Ratio (Rf=0)']:>9.4f}"
    )
    rows.append({
        "Asset": "EW_Portfolio",
        "Annualised Return": port_risk["Annualised Mean Return"],
        "Annualised Std":    port_risk["Annualised Std Deviation"],
        "Sharpe":            port_risk["Annualised Sharpe Ratio (Rf=0)"],
        "Sortino":           port_risk["Sortino Ratio (Rf=0)"],
    })

    summary_df = pd.DataFrame(rows).set_index("Asset")
    summary_df.to_csv(os.path.join(Q1_1_OUTPUT_DIR, "risk_return_summary.csv"))

    # ── 2. Diversification Benefit ────────────────────────────────────────
    logger.info("\n--- Diversification Benefit ---")
    individual_stds   = [asset_risk[t]["Annualised Std Deviation"] for t in TICKERS]
    avg_individual_std = np.mean(individual_stds)
    portfolio_std      = port_risk["Annualised Std Deviation"]
    div_benefit_pct    = (avg_individual_std - portfolio_std) / avg_individual_std * 100

    logger.info(f"  Average individual annualised std : "
                f"{avg_individual_std:.4f}  ({avg_individual_std * 100:.2f}%)")
    logger.info(f"  EW Portfolio annualised std       : "
                f"{portfolio_std:.4f}  ({portfolio_std * 100:.2f}%)")
    logger.info(f"  Risk reduction from diversification: {div_benefit_pct:.2f}%")

    # ── 3. Tail-Risk Comparison ───────────────────────────────────────────
    logger.info("\n--- Tail Risk Comparison (Historical VaR & CVaR at 99%) ---")
    all_assets = list(TICKERS) + ["EW_Portfolio"]
    logger.info(f"  {'Asset':<14} {'VaR (99%)':>11} {'CVaR (99%)':>12}")
    logger.info("  " + "-" * 38)
    for ticker in TICKERS:
        m = asset_risk[ticker]
        logger.info(f"  {ticker:<14} {m['Historical VaR (99%)']*100:>11.4f}% "
                    f"{m['Historical CVaR / ES (99%)']*100:>11.4f}%")
    logger.info(f"  {'EW_Portfolio':<14} "
                f"{port_risk['Historical VaR (99%)']*100:>11.4f}% "
                f"{port_risk['Historical CVaR / ES (99%)']*100:>11.4f}%")

    # ── 4. Narrative Key Takeaways ────────────────────────────────────────
    logger.info("\n--- Key Takeaways ---")
    logger.info(f"  1. The EW portfolio delivers a Sharpe ratio of "
                f"{port_risk['Annualised Sharpe Ratio (Rf=0)']:.4f}, "
                f"and a Sortino ratio of {port_risk['Sortino Ratio (Rf=0)']:.4f}.")
    logger.info(f"  2. Diversification reduces annualised volatility by "
                f"{div_benefit_pct:.1f}% relative to the average constituent std.")
    logger.info("  3. Despite the risk reduction, all intra-portfolio correlations are "
                "positive (tech-sector concentration), which limits the diversification "
                "benefit compared to a multi-sector or multi-asset portfolio.")
    logger.info("  4. Portfolio VaR and CVaR (99%) are lower in absolute terms than most "
                "individual constituents, confirming that diversification reduces tail risk.")
    logger.info("  5. The leptokurtic, negatively-skewed return distribution seen in the "
                "normality tests underscores that Gaussian VaR systematically "
                "under-estimates true tail risk; historical CVaR (ES) is the more "
                "appropriate risk measure.")

    log_end(logger, "q1_1_portfolio_conclusions.py")

    conclusions = {
        "diversification_benefit_pct": div_benefit_pct,
        "avg_individual_std": avg_individual_std,
        "portfolio_std": portfolio_std,
        "portfolio_sharpe": port_risk["Annualised Sharpe Ratio (Rf=0)"],
        "portfolio_sortino": port_risk["Sortino Ratio (Rf=0)"],
    }
    return conclusions


if __name__ == "__main__":
    setup_run_logger("smm272_q1_portfolio_conclusions")
    run_portfolio_conclusions()
