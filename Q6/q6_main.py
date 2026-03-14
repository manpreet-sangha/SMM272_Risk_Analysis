"""
SMM272 Risk Analysis Coursework 2025-2026 — Question 6
Q6 Main Orchestrator: VaR of a Portfolio with Options

Procedure
---------
  1  market_data   Download 2 years of prices; compute log-returns, vols, spot prices
  2  covariance    Estimate EWMA daily covariance matrix (λ=0.94, RiskMetrics)
  3  portfolio     Build portfolio; compute initial Black-Scholes prices and Greeks
  4  simulation    Run 10,000 correlated log-normal simulations over 10 trading days
  5  risk          Compute VaR₉₉, ES₉₉ (non-Gaussian, simulation-based)
  6  decompose     Compute Marginal VaR/ES and Component VaR/ES (Euler allocation)
  7  figures       Generate 6 publication-quality figures in output_q6/
  8  summary       Print tables and save 4 CSV files to output_q6/

Usage
-----
  python Q6/q6_main.py
  python Q6/q6_main.py --parts simulation risk figures
  python Q6/q6_main.py --dry-run
"""

import argparse
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger import setup_run_logger, get_logger, log_start, log_end
import config

logger = get_logger("q6_main")

BANNER = "*" * 60

STEPS = [
    ("market_data",  "Download prices, compute returns, vols, spots"),
    ("covariance",   "Estimate EWMA daily covariance matrix"),
    ("portfolio",    "Build portfolio and compute initial BS prices"),
    ("simulation",   "Run Monte Carlo simulation (10,000 paths × 10 days)"),
    ("risk",         "Compute VaR₉₉ and ES₉₉ (non-Gaussian)"),
    ("decompose",    "Marginal VaR/ES and Component VaR/ES (Euler)"),
    ("figures",      "Generate 6 figures in output_q6/"),
    ("summary",      "Print tables and save 4 CSV files"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q6: VaR of a Portfolio with Options"
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=[s[0] for s in STEPS] + ["all"],
        default=["all"],
        help="Which steps to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List steps without running them",
    )
    return parser.parse_args()


def main():
    args    = parse_args()
    log_path = setup_run_logger("q6_main")
    log_start(logger, "q6_main.py")

    requested = set(args.parts)
    if "all" in requested:
        requested = {s[0] for s in STEPS}

    if args.dry_run:
        logger.info("DRY RUN — steps that would execute:")
        for key, desc in STEPS:
            marker = "→" if key in requested else "  (skip)"
            logger.info(f"  {marker}  {key:<14} {desc}")
        log_end(logger, "q6_main.py")
        return

    print(f"\n{BANNER}")
    print(f"  SMM272 Risk Analysis — Q6: VaR of a Portfolio with Options")
    print(f"  Reference date   : {config.Q6_REFERENCE_DATE}")
    print(f"  Simulations      : {config.Q6_N_SIMS:,}")
    print(f"  Horizon          : {config.Q6_HORIZON_DAYS} trading days")
    print(f"  Confidence level : {config.Q6_CONFIDENCE*100:.0f}%")
    print(BANNER)

    t0 = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Market data
    # ──────────────────────────────────────────────────────────────────────────
    prices = log_returns = hist_vols = spot_prices = None
    if "market_data" in requested:
        logger.info("STEP 1 — Downloading market data …")
        from q6_market_data import load_all, describe_data
        prices, log_returns, hist_vols, spot_prices = load_all()
        describe_data(prices, log_returns, hist_vols, spot_prices)
        logger.info(
            f"  Downloaded {len(prices)} daily observations "
            f"for {config.Q6_TICKERS} | ref date {config.Q6_REFERENCE_DATE}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Covariance estimation
    # ──────────────────────────────────────────────────────────────────────────
    cov_daily = sample_cov_df = None
    if "covariance" in requested:
        logger.info("STEP 2 — Estimating EWMA covariance matrix …")
        if log_returns is None:
            from q6_market_data import get_log_returns
            log_returns = get_log_returns()
        from q6_covariance import ewma_cov, sample_cov
        cov_daily    = ewma_cov(log_returns, lam=config.Q6_EWMA_LAMBDA)
        sample_cov_df = sample_cov(log_returns)
        logger.info(f"  EWMA λ={config.Q6_EWMA_LAMBDA} | daily variances: "
                    + "  ".join(f"{t}={cov_daily.loc[t,t]*252*100:.1f}%Y ann"
                                for t in config.Q6_TICKERS))

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — Portfolio construction
    # ──────────────────────────────────────────────────────────────────────────
    portfolio = None
    if "portfolio" in requested:
        logger.info("STEP 3 — Building portfolio …")
        if spot_prices is None:
            from q6_market_data import get_spot_prices
            spot_prices = get_spot_prices()
        if hist_vols is None:
            from q6_market_data import get_hist_vols
            hist_vols = get_hist_vols()
        from q6_portfolio import build_portfolio
        portfolio = build_portfolio(spot_prices, hist_vols, r=config.Q6_RISK_FREE_RATE)
        from q6_summary import print_portfolio_summary, print_greeks
        print_portfolio_summary(portfolio)
        print_greeks(portfolio)
        logger.info(f"  Portfolio net value = ${sum(l['position_value'] for l in portfolio):,.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Monte Carlo simulation
    # ──────────────────────────────────────────────────────────────────────────
    S_T = log_returns_sim = None
    if "simulation" in requested:
        logger.info(f"STEP 4 — Running {config.Q6_N_SIMS:,} Monte Carlo simulations …")
        if spot_prices is None:
            from q6_market_data import get_spot_prices
            spot_prices = get_spot_prices()
        if cov_daily is None:
            from q6_market_data import get_log_returns
            from q6_covariance import ewma_cov
            cov_daily = ewma_cov(get_log_returns(), lam=config.Q6_EWMA_LAMBDA)
        from q6_simulation import simulate_prices, simulate_log_returns
        import pandas as pd
        # Align spot_prices with cov_daily column order
        aligned_spots = pd.Series(
            {t: float(spot_prices[t]) for t in cov_daily.columns}
        )
        S_T = simulate_prices(
            aligned_spots, cov_daily,
            n_sims=config.Q6_N_SIMS,
            horizon_days=config.Q6_HORIZON_DAYS,
            seed=config.Q6_RANDOM_SEED,
        )
        log_returns_sim = simulate_log_returns(
            aligned_spots, cov_daily,
            n_sims=config.Q6_N_SIMS,
            horizon_days=config.Q6_HORIZON_DAYS,
            seed=config.Q6_RANDOM_SEED,
        )
        logger.info(f"  Simulation complete | S_T shape: {S_T.shape}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5 + 6 — VaR/ES and decomposition
    # ──────────────────────────────────────────────────────────────────────────
    metrics = None
    if "risk" in requested or "decompose" in requested:
        logger.info("STEP 5-6 — Computing VaR, ES, and risk decomposition …")
        if S_T is None:
            raise RuntimeError(
                "Cannot compute risk metrics without simulation data. "
                "Include 'simulation' in --parts."
            )
        if portfolio is None:
            raise RuntimeError(
                "Cannot compute risk metrics without portfolio. "
                "Include 'portfolio' in --parts."
            )
        from q6_var_es import compute_all_metrics
        metrics = compute_all_metrics(
            S_T, portfolio,
            r=config.Q6_RISK_FREE_RATE,
            horizon_days=config.Q6_HORIZON_DAYS,
            confidence=config.Q6_CONFIDENCE,
        )
        from q6_summary import print_simulation_summary, print_var_es, print_risk_decomposition
        print_simulation_summary(metrics["total_pnl"])
        print_var_es(metrics["var"], metrics["es"], config.Q6_CONFIDENCE)
        print_risk_decomposition(metrics)
        logger.info(f"  VaR₉₉ = ${metrics['var']:,.2f}  |  ES₉₉ = ${metrics['es']:,.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7 — Figures
    # ──────────────────────────────────────────────────────────────────────────
    if "figures" in requested:
        logger.info("STEP 7 — Generating figures …")
        if metrics is None or portfolio is None:
            raise RuntimeError("Run 'simulation', 'portfolio', and 'risk' before 'figures'.")
        if log_returns is None:
            from q6_market_data import get_log_returns
            log_returns = get_log_returns()
        if sample_cov_df is None:
            from q6_covariance import sample_cov
            sample_cov_df = sample_cov(log_returns)
        from q6_visualisations import generate_all_figures
        fig_paths = generate_all_figures(
            portfolio        = portfolio,
            total_pnl        = metrics["total_pnl"],
            inst_pnl         = metrics["inst_pnl"],
            log_returns_sim  = log_returns_sim,
            metrics          = metrics,
            cov_daily        = cov_daily,
            sample_cov_df    = sample_cov_df,
            hist_log_returns = log_returns,
            r                = config.Q6_RISK_FREE_RATE,
        )
        for fp in fig_paths:
            logger.info(f"  Saved: {os.path.basename(fp)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8 — CSV output
    # ──────────────────────────────────────────────────────────────────────────
    if "summary" in requested:
        logger.info("STEP 8 — Saving CSV files …")
        if metrics is None or portfolio is None:
            raise RuntimeError("Run 'portfolio' and 'risk' before 'summary'.")
        from q6_summary import save_csvs
        csv_paths = save_csvs(
            portfolio, metrics, cov_daily, metrics["total_pnl"]
        )
        for cp in csv_paths:
            logger.info(f"  Saved: {os.path.basename(cp)}")

    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print(f"\n{BANNER}")
    print(f"  Q6 pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs: {os.path.basename(config.Q6_OUTPUT_DIR)}")
    print(BANNER)
    log_end(logger, "q6_main.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"Q6 pipeline failed: {exc}")
        raise
