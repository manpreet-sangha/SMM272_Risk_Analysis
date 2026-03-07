"""
Q1 Part 4 — Backtesting Orchestrator.

Applies four statistical backtests to the VaR violation series produced in
Part 3:

  1. Kupiec (1995) Proportion of Failures (POF) test
     — tests whether the observed violation rate matches the nominal tail
       probability (unconditional coverage).

  2. Christoffersen (1998) Independence test
     — tests whether consecutive violations are independent (violations
       should not cluster in time).

  3. Christoffersen (1998) Conditional Coverage (CC) test
     — joint test: LR_cc = LR_uc + LR_ind ~ chi^2(2).

  4. Christoffersen-Pelletier (2004) Weibull Duration test
     — tests whether the time between violations follows an Exponential
       distribution (no duration dependence / no clustering).

  5. Engle-Manganelli (2004) Dynamic Quantile (DQ) test
     — tests whether the demeaned hit sequence is predictable from its
       own lags (a mis-specified model would show serial dependence).

Outputs (saved to Q1/output_q1_4/)
------------------------------------
  backtest_results.csv            — all four test statistics, p-values, decisions
  fig_backtest_pvalues_heatmap.png — p-value heat-map per test × model × CI
  fig_backtest_lr_stats.png        — LR statistics vs 5 % critical values

Dependencies
------------
This module loads the violation summary and flag series produced by Part 3
(violations_summary.csv and violation_series.csv) from Q1/output_q1_3/.
If those files are missing it runs Part 3 first.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config import (
    Q1_3_CONFIDENCE_LEVELS,
    Q1_3_OUTPUT_DIR,
    Q1_4_OUTPUT_DIR,
)
from logger import setup_run_logger, get_logger, log_start, log_end
from q1_4_backtests  import run_all_backtests
from q1_4_logging    import log_backtest_results
from q1_4_plots      import generate_plots
from q1_methods      import METHODS

logger = get_logger("q1_4_backtesting")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_q1_3_outputs():
    """
    Load Part 3 violation outputs.  Runs Part 3 first if CSVs are absent.

    Returns
    -------
    violations_df   : pd.DataFrame
    violation_flags : pd.DataFrame (indexed by date)
    """
    summary_csv = os.path.join(Q1_3_OUTPUT_DIR, "violations_summary.csv")
    series_csv  = os.path.join(Q1_3_OUTPUT_DIR, "violation_series.csv")

    if not os.path.exists(summary_csv) or not os.path.exists(series_csv):
        logger.warning(
            "Part 3 outputs not found — running q1_3_var_violations.main() first."
        )
        import q1_3_var_violations
        q1_3_var_violations.main()

    violations_df   = pd.read_csv(summary_csv)
    violation_flags = pd.read_csv(series_csv, index_col=0, parse_dates=True)
    logger.info(
        f"  Loaded violations_summary.csv   ({violations_df.shape[0]} rows)"
    )
    logger.info(
        f"  Loaded violation_series.csv     ({violation_flags.shape})"
    )
    return violations_df, violation_flags


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = setup_run_logger("smm272_q1_4")
    log_start(logger, "q1_4_backtesting.py")

    logger.info("=" * 80)
    logger.info("SMM272 Q1 Part 4 — VaR Backtesting")
    logger.info("=" * 80)
    ci_strs = [f"{c*100:.0f}%" for c in Q1_3_CONFIDENCE_LEVELS]
    logger.info(f"  Confidence levels : {ci_strs}")
    logger.info(f"  Run log           : {log_path}")

    # ── Load Part 3 violation data ────────────────────────────────────────
    logger.info("\n--- Loading Part 3 violation data ---")
    violations_df, violation_flags = _load_q1_3_outputs()

    # ── Run all backtests ─────────────────────────────────────────────────
    logger.info("\n--- Running backtests (Kupiec, CC, Duration, DQ) ---")
    results_df = run_all_backtests(
        violations_df,
        violation_flags,
        Q1_3_CONFIDENCE_LEVELS,
        METHODS,
        n_lags=4,
    )

    # ── Save results ──────────────────────────────────────────────────────
    out_csv = os.path.join(Q1_4_OUTPUT_DIR, "backtest_results.csv")
    results_df.to_csv(out_csv, index=False)
    logger.info(f"\n  Results saved to: {out_csv}")

    # ── Log tables ────────────────────────────────────────────────────────
    logger.info("\n--- Backtest Summary Tables ---")
    log_backtest_results(results_df)

    # ── Figures ───────────────────────────────────────────────────────────
    logger.info("\n--- Generating figures ---")
    generate_plots(results_df, Q1_3_CONFIDENCE_LEVELS, METHODS)

    logger.info(f"\n  All outputs saved to: {Q1_4_OUTPUT_DIR}")
    log_end(logger, "q1_4_backtesting.py")
    return results_df


if __name__ == "__main__":
    main()
