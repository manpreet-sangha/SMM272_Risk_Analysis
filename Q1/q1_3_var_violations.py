"""
Q1 Part 3 — VaR Violations Analysis (orchestrator).

For each of the four rolling VaR models (Historical Simulation, Parametric
Normal, Parametric Student-t, GARCH(1,1)) and each of three confidence
levels (90%, 95%, 99%), this module:

  1.  Re-runs the rolling 6-month window estimation to obtain VaR forecasts
      at each confidence level.         → q1_3_rolling.py
  2.  Counts VaR violations (days where realised return < VaR estimate).
  3.  Applies the Kupiec (1995) POF test for unconditional coverage.    → q1_3_kupiec.py
  4.  Applies the Christoffersen (1998) independence test.              → q1_3_christoffersen.py
  5.  Computes the joint conditional-coverage (CC) test statistic.      → q1_3_backtests.py
  6.  Logs summary tables.                                              → q1_3_logging.py
  7.  Generates three diagnostic figures.                               → q1_3_plots.py

Outputs (saved to Q1/output_q1_3/)
------------------------------------
  rolling_var_all_levels.csv      — raw VaR series at every confidence level
  violations_summary.csv          — violations count and rate per (method, CI)
  kupiec_results.csv              — Kupiec LR, p-value, decision per (method, CI)
  christoffersen_results.csv      — independence LR, CC LR, p-values per (method, CI)
  violation_series.csv            — daily violation flags for every (method, CI) pair
  fig_violations_heatmap.png      — observed vs expected violations heatmap
  fig_violations_barchart.png     — grouped bar chart of breach rates
  fig_violation_timeseries.png    — cumulative violation counts over time
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config import (
    ROLLING_WINDOW_MONTHS,
    ROLLING_START_DATE,
    Q1_3_CONFIDENCE_LEVELS,
    Q1_3_OUTPUT_DIR,
    Q1_1_OUTPUT_DIR,
)
from logger import setup_run_logger, get_logger, log_start, log_end
from q1_3_rolling             import run_rolling_all_levels
from q1_3_count_violations    import count_violations
from q1_3_logging             import log_violations
from q1_3_plots               import generate_plots
from q1_methods               import METHODS

logger = get_logger("q1_3_var_violations")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = setup_run_logger("smm272_q1_3")
    log_start(logger, "q1_3_var_violations.py")

    logger.info("=" * 80)
    logger.info("SMM272 Q1 Part 3 — VaR Violations at Multiple Confidence Levels")
    logger.info("=" * 80)
    ci_strs = [f"{c*100:.0f}%" for c in Q1_3_CONFIDENCE_LEVELS]
    logger.info(f"  Confidence levels : {ci_strs}")
    logger.info(f"  Rolling window    : {ROLLING_WINDOW_MONTHS} months")
    logger.info(f"  Start date        : {ROLLING_START_DATE}")
    logger.info(f"  Run log           : {log_path}")

    # Load portfolio returns from pre-computed CSV (avoids re-running q1_1)
    _portfolio_csv = os.path.join(Q1_1_OUTPUT_DIR, "portfolio_returns.csv")
    if not os.path.exists(_portfolio_csv):
        from q1_1_build_portfolio import build_portfolio
        _, _, portfolio_returns = build_portfolio()
    else:
        portfolio_returns = pd.read_csv(
            _portfolio_csv, index_col=0, parse_dates=True
        ).squeeze()
        portfolio_returns.name = "EW_Portfolio"
        logger.info(f"  Portfolio returns loaded from cache ({len(portfolio_returns)} obs)")

    # ── Rolling VaR estimation ────────────────────────────────────────────
    logger.info("\n--- Running rolling windows (all confidence levels) ---")
    df = run_rolling_all_levels(portfolio_returns, Q1_3_CONFIDENCE_LEVELS, METHODS)

    out_csv = os.path.join(Q1_3_OUTPUT_DIR, "rolling_var_all_levels.csv")
    df.to_csv(out_csv)
    logger.info(f"\n  Raw VaR series saved to : {out_csv}  ({df.shape})")

    # ── Count violations ──────────────────────────────────────────────────
    logger.info("\n--- Counting VaR violations ---")
    violations_df, violation_flags = count_violations(
        df, Q1_3_CONFIDENCE_LEVELS, METHODS
    )

    log_violations(violations_df)

    violations_df.to_csv(
        os.path.join(Q1_3_OUTPUT_DIR, "violations_summary.csv"), index=False
    )
    violation_flags.to_csv(
        os.path.join(Q1_3_OUTPUT_DIR, "violation_series.csv")
    )
    logger.info(f"\n  CSVs saved to: {Q1_3_OUTPUT_DIR}")

    # ── Visualisations ────────────────────────────────────────────────────
    logger.info("\n--- Generating figures ---")
    generate_plots(df, violations_df, violation_flags, Q1_3_CONFIDENCE_LEVELS, METHODS)

    logger.info(f"\n  All outputs saved to: {Q1_3_OUTPUT_DIR}")
    log_end(logger, "q1_3_var_violations.py")
    return violations_df, violation_flags


if __name__ == "__main__":
    main()
