"""
SMM272 Risk Analysis Coursework 2025-2026 — Question 2
Q2 Main Orchestrator: Power of the Kupiec Test

Steps (all run by default)
--------------------------
  1  Fit Gaussian H0 and GARCH(1,1) H1 to Q1 EW portfolio returns.
  2  Monte Carlo power estimation at T=2893, three confidence levels.
  3  Power as a function of sample size T.
  4  Power as a function of GARCH persistence ρ = α + β.
  5  Generate all figures.
  6  Print and save summary.

Usage
-----
  # Run all steps:
  python q2_main.py

  # Run only model fitting:
  python q2_main.py --parts fit

  # Run fitting + power estimation only:
  python q2_main.py --parts fit power

  # Dry run (show steps without executing):
  python q2_main.py --dry-run
"""

import argparse
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q2_main")

BANNER = "*" * 54

STEPS = [
    ("fit",         "Fit Gaussian (H0) and GARCH(1,1) (H1) models"),
    ("power",       "Monte Carlo power estimation (T=2893)"),
    ("power_vs_T",  "Power vs. sample size T"),
    ("persistence", "Power vs. GARCH persistence ρ"),
    ("figures",     "Generate all figures"),
    ("summary",     "Print and save summary"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Q2: Power of the Kupiec Test")
    parser.add_argument("--parts", nargs="+",
                        choices=[s[0] for s in STEPS] + ["all"],
                        default=["all"],
                        help="Which steps to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    return parser.parse_args()


def main():
    args = parse_args()

    log_path = setup_run_logger("q2_main")
    log_start(logger, "q2_main.py")

    requested = set(args.parts)
    if "all" in requested:
        requested = {s[0] for s in STEPS}

    if args.dry_run:
        logger.info("DRY RUN — steps that would execute:")
        for key, desc in STEPS:
            marker = "  [x]" if key in requested else "  [ ]"
            logger.info(f"{marker} {key}: {desc}")
        return

    # ── State shared between steps ────────────────────────────────────────────
    gauss_params  = None
    garch_params  = None
    returns       = None
    power_results = None
    df_T          = None
    df_persist    = None

    # ── Step 1: Fit models ────────────────────────────────────────────────────
    if "fit" in requested:
        logger.info(BANNER)
        logger.info("STEP 1 — Fit H0 (Gaussian) and H1 (GARCH(1,1))")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q2_fit_models import run_fit_models
            gauss_params, garch_params, returns = run_fit_models()
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())
            sys.exit(1)
    else:
        
        # Still need params for downstream steps
        from q2_fit_models import run_fit_models
        gauss_params, garch_params, returns = run_fit_models()

    # ── Step 2: Power estimation ──────────────────────────────────────────────
    if "power" in requested:
        logger.info(BANNER)
        logger.info("STEP 2 — Monte Carlo Kupiec power (T=2893)")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q2_kupiec_power import run_kupiec_power
            power_results = run_kupiec_power(gauss_params, garch_params)
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    # ── Step 3: Power vs. T ───────────────────────────────────────────────────
    if "power_vs_T" in requested:
        logger.info(BANNER)
        logger.info("STEP 3 — Power vs. sample size T")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q2_power_vs_T import run_power_vs_T
            df_T = run_power_vs_T(gauss_params, garch_params)
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    # ── Step 4: Power vs. persistence ────────────────────────────────────────
    if "persistence" in requested:
        logger.info(BANNER)
        logger.info("STEP 4 — Power vs. GARCH persistence")
        logger.info(BANNER)
        t0 = time.time()
        try:
            from q2_power_vs_persistence import run_power_vs_persistence
            df_persist = run_power_vs_persistence(gauss_params, garch_params)
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    # ── Step 5: Visualisations ────────────────────────────────────────────────
    if "figures" in requested:
        logger.info(BANNER)
        logger.info("STEP 5 — Generate figures")
        logger.info(BANNER)
        t0 = time.time()
        try:
            # Load CSVs if in-memory objects are missing
            import pandas as pd
            from config import Q2_OUTPUT_DIR
            if df_T is None:
                df_T = pd.read_csv(os.path.join(Q2_OUTPUT_DIR, "power_vs_T.csv"))
            if df_persist is None:
                df_persist = pd.read_csv(
                    os.path.join(Q2_OUTPUT_DIR, "power_vs_persistence.csv"))

            if power_results is None:
                logger.warning("Power results not in memory; running step 2 first.")
                from q2_kupiec_power import run_kupiec_power
                power_results = run_kupiec_power(gauss_params, garch_params)

            from q2_visualisations import run_visualisations
            run_visualisations(power_results, df_T, df_persist,
                               gauss_params, garch_params, returns)
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    if "summary" in requested:
        logger.info(BANNER)
        logger.info("STEP 6 — Summary")
        logger.info(BANNER)
        t0 = time.time()
        try:
            import pandas as pd
            from config import Q2_OUTPUT_DIR
            if df_T is None:
                df_T = pd.read_csv(os.path.join(Q2_OUTPUT_DIR, "power_vs_T.csv"))
            if df_persist is None:
                df_persist = pd.read_csv(
                    os.path.join(Q2_OUTPUT_DIR, "power_vs_persistence.csv"))
            if power_results is None:
                from q2_kupiec_power import run_kupiec_power
                power_results = run_kupiec_power(gauss_params, garch_params)

            from q2_summary import run_summary
            run_summary(gauss_params, garch_params, power_results, df_T, df_persist)
            logger.info(f"  Completed in {time.time()-t0:.1f}s")
        except Exception:
            logger.error(traceback.format_exc())

    log_end(logger, "q2_main.py")
    logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
