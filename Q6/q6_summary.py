"""
Q6 — Summary Output: Console Tables and CSV Files
===================================================
Prints formatted results to stdout and saves four CSV files to Q6_OUTPUT_DIR.

CSV files written
-----------------
  q6_portfolio_summary.csv     position details, initial prices, Greeks
  q6_risk_metrics.csv          VaR, ES, Component, Marginal VaR/ES
  q6_covariance_matrix.csv     EWMA daily covariance matrix
  q6_pnl_statistics.csv        descriptive stats of portfolio P&L distribution
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

SEP  = "=" * 72
SEP2 = "-" * 72


def _hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _usd(x: float) -> str:
    return f"${x:>12,.2f}"


# ── 1. Market data snapshot ────────────────────────────────────────────────────

def print_market_snapshot(
    spot_prices:  pd.Series,
    hist_vols:    pd.Series,
    log_returns:  pd.DataFrame,
    reference_date: str = config.Q6_REFERENCE_DATE,
) -> None:
    _hdr("Q6  MARKET DATA SNAPSHOT")
    print(f"  Reference date  : {reference_date}")
    print(f"  History window  : 2 years  ({config.Q6_HIST_START} → {reference_date})")
    print(f"  Risk-free rate  : {config.Q6_RISK_FREE_RATE*100:.1f}% p.a.")
    print(f"  Observations    : {len(log_returns)} daily log-returns per ticker\n")
    print(f"  {'Ticker':<7} {'Spot ($)':>11} {'HVol (%)':>10}")
    print(f"  {'-'*7} {'-'*11} {'-'*10}")
    for t in config.Q6_TICKERS:
        print(f"  {t:<7} {spot_prices[t]:>11.2f} {hist_vols[t]*100:>10.2f}")


# ── 2. Portfolio summary ───────────────────────────────────────────────────────

def print_portfolio_summary(portfolio: list[dict]) -> None:
    _hdr("Q6  PORTFOLIO CONSTRUCTION  (Feb 12, 2026)")
    hdr = (f"  {'Ticker':<5} {'Type':<5} {'Q':>4} {'Spot':>8} "
           f"{'Strike':>8} {'TTM(mo)':>7} "
           f"{'HVol%':>7} {'Opt($)':>9} {'PosVal($)':>12}")
    print(hdr)
    print(f"  {'-'*5} {'-'*5} {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*12}")
    total_val = 0.0
    for leg in portfolio:
        sign_str = "Long" if leg["quantity"] > 0 else "Short"
        print(
            f"  {leg['ticker']:<5} "
            f"{leg['option_type'][:4]:<5} "
            f"{leg['quantity']:>+4d} "
            f"{leg['spot']:>8.2f} "
            f"{leg['strike']:>8.2f} "
            f"{leg['ttm_months']:>7d} "
            f"{leg['vol']*100:>7.2f} "
            f"{leg['price_per_share']:>9.4f} "
            f"{leg['position_value']:>12,.2f}"
        )
        total_val += leg["position_value"]
    print(f"  {'-'*72}")
    print(f"  {'TOTAL NET POSITION VALUE':>54} {total_val:>12,.2f}")
    print(f"\n  Multiplier: {config.Q6_MULTIPLIER} shares per contract (standard US equity option)")


def print_greeks(portfolio: list[dict]) -> None:
    _hdr("Q6  OPTION GREEKS  (per share, at inception)")
    print(f"  {'Ticker':<7} {'Delta':>8} {'Gamma':>9} {'Vega':>9} {'Theta/day':>10}")
    print(f"  {'-'*7} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")
    for leg in portfolio:
        print(
            f"  {leg['ticker']:<7} "
            f"{leg['delta']:>8.4f} "
            f"{leg['gamma']:>9.6f} "
            f"{leg['vega']:>9.4f} "
            f"{leg['theta']:>10.4f}"
        )


# ── 3. Simulation summary ──────────────────────────────────────────────────────

def print_simulation_summary(
    total_pnl: np.ndarray,
    n_sims:    int = config.Q6_N_SIMS,
    horizon:   int = config.Q6_HORIZON_DAYS,
) -> None:
    _hdr("Q6  MONTE CARLO SIMULATION SUMMARY")
    print(f"  Simulations  : {n_sims:,}")
    print(f"  Horizon      : {horizon} trading days")
    print(f"  Covariance   : EWMA (λ={config.Q6_EWMA_LAMBDA})")
    print(f"  Repricing    : Full revaluation (Black-Scholes, frozen historical vol)\n")
    print(f"  Portfolio P&L statistics:")
    print(f"    Mean        : {_usd(np.mean(total_pnl))}")
    print(f"    Std Dev     : {_usd(np.std(total_pnl))}")
    print(f"    Min (worst) : {_usd(np.min(total_pnl))}")
    print(f"    5th pctile  : {_usd(np.percentile(total_pnl, 5))}")
    print(f"    1st pctile  : {_usd(np.percentile(total_pnl, 1))}")
    print(f"    Median      : {_usd(np.median(total_pnl))}")
    print(f"    Max (best)  : {_usd(np.max(total_pnl))}")


# ── 4. VaR / ES results ───────────────────────────────────────────────────────

def print_var_es(var: float, es: float, confidence: float = 0.99) -> None:
    _hdr(f"Q6  VaR and ES  ({confidence*100:.0f}% confidence, 10-day horizon)")
    print(f"  VaR_{confidence*100:.0f}%  = {_usd(var)}")
    print(f"  ES_{confidence*100:.0f}%   = {_usd(es)}")
    print(f"\n  Interpretation:")
    print(f"    There is a {(1-confidence)*100:.0f}% probability that the portfolio")
    print(f"    loses more than ${var:,.2f} over the next 10 trading days.")
    print(f"    The expected loss conditional on exceeding VaR is ${es:,.2f}.")


# ── 5. Risk decomposition ──────────────────────────────────────────────────────

def print_risk_decomposition(metrics: dict) -> None:
    tickers    = metrics["tickers"]
    quantities = metrics["quantities"]
    comp_var   = metrics["comp_var"]
    comp_es    = metrics["comp_es"]
    marg_var   = metrics["marg_var"]
    marg_es    = metrics["marg_es"]
    var        = metrics["var"]
    es         = metrics["es"]
    n_tail     = metrics["tail_count"]

    _hdr("Q6  MARGINAL VaR/ES  (non-Gaussian, per additional long contract)")
    print(f"  {'Ticker':<7} {'Q':>5} {'MargVaR($)':>13} {'MargES($)':>13}")
    print(f"  {'-'*7} {'-'*5} {'-'*13} {'-'*13}")
    for t, q, mv, me in zip(tickers, quantities, marg_var, marg_es):
        print(f"  {t:<7} {int(q):>+5d} {mv:>13,.2f} {me:>13,.2f}")

    _hdr("Q6  COMPONENT VaR/ES  (Euler allocation, non-Gaussian)")
    print(f"  {'Ticker':<7} {'Q':>5} {'CompVaR($)':>13} {'CompES($)':>13}  {'%VaR':>7}  {'%ES':>7}")
    print(f"  {'-'*7} {'-'*5} {'-'*13} {'-'*13}  {'-'*7}  {'-'*7}")
    for t, q, cv, ce in zip(tickers, quantities, comp_var, comp_es):
        pct_v = cv / var * 100 if var != 0 else 0
        pct_e = ce / es  * 100 if es  != 0 else 0
        print(f"  {t:<7} {int(q):>+5d} {cv:>13,.2f} {ce:>13,.2f}  {pct_v:>6.1f}%  {pct_e:>6.1f}%")

    print(f"  {'-'*68}")
    cv_sum = comp_var.sum()
    ce_sum = comp_es.sum()
    print(f"  {'SUM':>13} {cv_sum:>18,.2f} {ce_sum:>13,.2f}")
    print(f"  {'Portfolio':>13} {var:>18,.2f} {es:>13,.2f}")
    print(f"\n  Tail observations used for ES decomposition : {n_tail:,}")
    print(f"  Component ES sum check   : ${ce_sum:,.2f} vs ES ${es:,.2f}"
          f"  (Δ = ${abs(ce_sum - es):,.2f})")
    print(f"  Component VaR sum check  : ${cv_sum:,.2f} vs VaR ${var:,.2f}"
          f"  (Δ = ${abs(cv_sum - var):,.2f}) [FD approximation]")


# ── 6. CSV output ─────────────────────────────────────────────────────────────

def save_csvs(
    portfolio:     list[dict],
    metrics:       dict,
    cov_daily:     pd.DataFrame,
    total_pnl:     np.ndarray,
) -> list[str]:
    out  = config.Q6_OUTPUT_DIR
    tickers = metrics["tickers"]

    paths = []

    # Portfolio summary
    df_port = pd.DataFrame(portfolio).set_index("ticker")
    cols = ["description", "option_type", "quantity", "spot", "strike",
            "strike_pct", "ttm_months", "ttm", "vol",
            "price_per_share", "price_per_contract", "position_value",
            "delta", "gamma", "vega", "theta", "moneyness"]
    p1 = os.path.join(out, "q6_portfolio_summary.csv")
    df_port[cols].to_csv(p1, float_format="%.6f")
    paths.append(p1)

    # Risk metrics
    risk_rows = []
    for i, t in enumerate(tickers):
        risk_rows.append({
            "ticker":       t,
            "quantity":     metrics["quantities"][i],
            "comp_var":     metrics["comp_var"][i],
            "marg_var":     metrics["marg_var"][i],
            "comp_es":      metrics["comp_es"][i],
            "marg_es":      metrics["marg_es"][i],
            "pct_comp_var": metrics["comp_var"][i] / metrics["var"] * 100,
            "pct_comp_es":  metrics["comp_es"][i]  / metrics["es"]  * 100,
        })
    df_risk = pd.DataFrame(risk_rows).set_index("ticker")
    # Append portfolio totals
    totals = pd.Series({
        "ticker":       "PORTFOLIO",
        "quantity":     float("nan"),
        "comp_var":     metrics["var"],
        "marg_var":     float("nan"),
        "comp_es":      metrics["es"],
        "marg_es":      float("nan"),
        "pct_comp_var": 100.0,
        "pct_comp_es":  100.0,
    }, name="PORTFOLIO")
    df_risk = pd.concat([df_risk, totals.to_frame().T.set_index("ticker")])
    p2 = os.path.join(out, "q6_risk_metrics.csv")
    df_risk.to_csv(p2, float_format="%.6f")
    paths.append(p2)

    # Covariance matrix
    p3 = os.path.join(out, "q6_covariance_matrix.csv")
    cov_daily.to_csv(p3, float_format="%.8f")
    paths.append(p3)

    # P&L distribution statistics
    percentiles = [0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5]
    pnl_stats = {
        "mean":   np.mean(total_pnl),
        "std":    np.std(total_pnl),
        "skew":   float(pd.Series(total_pnl).skew()),
        "kurt":   float(pd.Series(total_pnl).kurt()),
        "min":    np.min(total_pnl),
        "max":    np.max(total_pnl),
        "var_99": metrics["var"],
        "es_99":  metrics["es"],
    }
    for p in percentiles:
        pnl_stats[f"pct_{p:.1f}"] = float(np.percentile(total_pnl, p))
    df_stats = pd.DataFrame.from_dict(pnl_stats, orient="index", columns=["value"])
    p4 = os.path.join(out, "q6_pnl_statistics.csv")
    df_stats.to_csv(p4, float_format="%.6f")
    paths.append(p4)

    return paths
