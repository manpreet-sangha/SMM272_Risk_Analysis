"""
SMM272 Risk Analysis Coursework 2025-2026 — Question 5
Q5 Main Orchestrator: CME SPAN Margining for COMEX Copper Options

Steps (all run by default)
--------------------------
  1  scenarios   Build 16-scenario SPAN risk array
  2  margins     Compute no-netting and SPAN-netting margins for 4 pairs
  3  figures     Generate all 6 figures
  4  summary     Print tables and save CSVs

Usage
-----
  # Run all steps:
  python Q5/q5_main.py

  # Run only margin computation:
  python Q5/q5_main.py --parts margins

  # figures + summary only:
  python Q5/q5_main.py --parts figures summary

  # Dry run:
  python Q5/q5_main.py --dry-run
"""

import argparse
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q5_main")

BANNER = "*" * 54

STEPS = [
    ("scenarios", "Build SPAN 16-scenario risk arrays for all 4 pairs"),
    ("margins",   "Compute no-netting and SPAN-netting margins"),
    ("figures",   "Generate all 6 figures"),
    ("summary",   "Print tables and save CSVs to output_q5/"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Q5: CME SPAN Margining for COMEX Copper Options"
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
        help="Show what would run without executing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_path = setup_run_logger("q5_main")
    log_start(logger, "q5_main.py")

    requested = set(args.parts)
    if "all" in requested:
        requested = {s[0] for s in STEPS}

    if args.dry_run:
        logger.info("DRY RUN — steps that would execute:")
        for key, desc in STEPS:
            marker = "  [x]" if key in requested else "  [ ]"
            logger.info(f"{marker} {key}: {desc}")
        return

    # ── Step 1: Scenarios ─────────────────────────────────────────────────────
    if "scenarios" in requested:
        logger.info(BANNER)
        logger.info("STEP 1 — Build SPAN 16-scenario risk arrays")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q5_span_scenarios import build_scenario_table, scenario_breakdown_df
            from q5_positions import ALL_PAIRS
            scenario_df = build_scenario_table()
            logger.info(f"  Scenario table built: {len(scenario_df)} scenarios")
            for pair in ALL_PAIRS:
                bd = scenario_breakdown_df(pair["legs"], scenario_df)
                worst_loss = bd["weighted_loss"].max()
                logger.info(f"  {pair['short']} worst weighted loss: ${worst_loss:,.2f}")
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())
            sys.exit(1)

    # ── Step 2: Margins ───────────────────────────────────────────────────────
    if "margins" in requested:
        logger.info(BANNER)
        logger.info("STEP 2 — No-netting and SPAN-netting margins")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q5_margin_calculator import run_all_margins
            df = run_all_margins()
            logger.info(f"  {'Pair':<36} {'No-Net':>9} {'SPAN-Net':>9} {'Benefit%':>9}")
            logger.info(f"  {'-'*65}")
            for _, row in df.iterrows():
                logger.info(
                    f"  {row['pair_name']:<36}"
                    f"  ${row['no_net_total']:>8,.0f}"
                    f"  ${row['span_net_margin']:>8,.0f}"
                    f"  {row['netting_pct']:>8.1f}%"
                )
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    # ── Step 3: Figures ───────────────────────────────────────────────────────
    if "figures" in requested:
        logger.info(BANNER)
        logger.info("STEP 3 — Generate figures")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q5_visualisations import run_all_figures
            paths = run_all_figures()
            for p in paths:
                logger.info(f"  Saved: {p}")
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    if "summary" in requested:
        logger.info(BANNER)
        logger.info("STEP 4 — Summary tables and CSV output")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q5_summary import run_all_summary
            run_all_summary()
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    log_end(logger, "q5_main.py")


if __name__ == "__main__":
    main()
