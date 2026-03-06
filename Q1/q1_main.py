"""
SMM272 Risk Analysis Coursework 2025-2026 — Question 1
Q1 Main Orchestrator

Runs any combination of Q1 sub-parts in sequence.

Sub-parts available
-------------------
  q1_1  Statistical analysis of the EW portfolio (descriptive stats, risk
         metrics, normality tests, time-series diagnostics, correlation,
         visualisations, conclusions, summary).
  q1_2  Rolling 6-month VaR and ES at 99% — four methods (Historical
         Simulation, Normal, Student-t, GARCH(1,1)).
  q1_3  (placeholder — add when implemented)
  q1_4  (placeholder — add when implemented)

Usage examples
--------------
  # Run all available sub-parts in order:
  python q1_main.py

  # Run only q1_1:
  python q1_main.py --parts q1_1

  # Run only q1_2:
  python q1_main.py --parts q1_2

  # Run q1_1 and q1_2:
  python q1_main.py --parts q1_1 q1_2

  # Run a range from q1_1 through q1_3 (inclusive):
  python q1_main.py --from q1_1 --to q1_3

  # Run from q1_2 to the end of all registered parts:
  python q1_main.py --from q1_2

  # Run up to q1_2 (start from q1_1 by default):
  python q1_main.py --to q1_2

  # Dry-run: show what would run without executing:
  python q1_main.py --dry-run
  python q1_main.py --parts q1_2 --dry-run
"""

import argparse
import importlib
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger import setup_run_logger, get_logger

logger = get_logger("q1_main")

# ── Registry of all Q1 sub-parts ─────────────────────────────────────────────
# Each entry: (part_id, module_name, description)
# Add new parts here as they are implemented.

REGISTRY: list[tuple[str, str, str]] = [
    (
        "q1_1",
        "q1_1_statistical_analysis",
        "Q1 Part 1 — EW portfolio statistical analysis "
        "(descriptive stats, risk metrics, normality, time-series, correlation, "
        "visualisations, conclusions, summary)",
    ),
    (
        "q1_2",
        "q1_2_statistical_analysis",
        "Q1 Part 2 — Rolling 6-month VaR and ES at 99% "
        "(Historical Simulation, Normal, Student-t, GARCH(1,1))",
    ),
    (
        "q1_3",
        "q1_3_var_violations",
        "Q1 Part 3 — VaR violations at 90 %, 95 %, and 99 % confidence levels "
        "(Kupiec POF test, Christoffersen independence and joint tests)",
    ),
    # (
    #     "q1_4",
    #     "q1_4_statistical_analysis",
    #     "Q1 Part 4 — (not yet implemented)",
    # ),
]

ALL_PART_IDS = [r[0] for r in REGISTRY]


# ── Selection helpers ─────────────────────────────────────────────────────────

def _registry_index(part_id: str) -> int:
    for i, (pid, *_) in enumerate(REGISTRY):
        if pid == part_id:
            return i
    raise ValueError(
        f"Unknown part '{part_id}'. Available: {ALL_PART_IDS}"
    )


def resolve_parts(
    parts: list[str] | None,
    from_part: str | None,
    to_part: str | None,
) -> list[tuple[str, str, str]]:
    """
    Return the ordered list of (part_id, module_name, description) tuples
    to execute based on the CLI arguments.

    Priority:
    1. --parts  (explicit list; --from/--to are ignored)
    2. --from / --to  (range, inclusive on both ends)
    3. All registered parts  (default)
    """
    if parts:
        selected = []
        for pid in parts:
            idx = _registry_index(pid)
            selected.append(REGISTRY[idx])
        return selected

    start_idx = _registry_index(from_part) if from_part else 0
    end_idx   = _registry_index(to_part)   if to_part   else len(REGISTRY) - 1
    if start_idx > end_idx:
        raise ValueError(
            f"--from {from_part} comes after --to {to_part} in the registry."
        )
    return REGISTRY[start_idx : end_idx + 1]


# ── Execution ─────────────────────────────────────────────────────────────────

def run_parts(selected: list[tuple[str, str, str]], dry_run: bool = False) -> None:
    total   = len(selected)
    passed  = []
    failed  = []

    logger.info("=" * 70)
    logger.info(f"Q1 MAIN ORCHESTRATOR — {total} sub-part(s) selected")
    logger.info("=" * 70)

    for i, (part_id, module_name, description) in enumerate(selected, start=1):
        banner = f"[{i}/{total}]  {part_id.upper()}  —  {description}"
        logger.info("")
        logger.info("─" * 70)
        logger.info(banner)
        logger.info("─" * 70)

        if dry_run:
            logger.info(f"  DRY-RUN: would import '{module_name}' and call main()")
            passed.append(part_id)
            continue

        t0 = time.perf_counter()
        try:
            mod  = importlib.import_module(module_name)
            if not hasattr(mod, "main"):
                raise AttributeError(
                    f"Module '{module_name}' has no 'main()' function."
                )
            mod.main()
            elapsed = time.perf_counter() - t0
            logger.info(f"  ✓  {part_id} completed in {elapsed:.1f}s")
            passed.append(part_id)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(f"  ✗  {part_id} FAILED after {elapsed:.1f}s: {exc}")
            logger.debug(traceback.format_exc())
            failed.append((part_id, exc))

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("ORCHESTRATION COMPLETE")
    logger.info(f"  Passed : {len(passed)}  —  {', '.join(passed) if passed else '—'}")
    if failed:
        logger.info(
            f"  Failed : {len(failed)}  —  "
            + ", ".join(f"{pid} ({exc})" for pid, exc in failed)
        )
        sys.exit(1)
    else:
        logger.info("  Failed : 0")
    logger.info("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="q1_main.py",
        description="SMM272 Q1 Orchestrator — run one or more Q1 sub-parts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [
                "Registered sub-parts:",
                *[f"  {pid:<8}  {desc}" for pid, _, desc in REGISTRY],
                "",
                "Examples:",
                "  python q1_main.py                        # run all parts",
                "  python q1_main.py --parts q1_1           # only part 1",
                "  python q1_main.py --parts q1_1 q1_2      # parts 1 and 2",
                "  python q1_main.py --from q1_1 --to q1_2  # range q1_1→q1_2",
                "  python q1_main.py --from q1_2            # from q1_2 to end",
                "  python q1_main.py --to q1_1              # only up to q1_1",
                "  python q1_main.py --dry-run              # preview only",
            ]
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--parts",
        nargs="+",
        metavar="PART",
        choices=ALL_PART_IDS,
        help="Explicit list of sub-parts to run (e.g. --parts q1_1 q1_2).",
    )

    parser.add_argument(
        "--from",
        dest="from_part",
        metavar="PART",
        choices=ALL_PART_IDS,
        help="Start of the execution range (inclusive).",
    )
    parser.add_argument(
        "--to",
        dest="to_part",
        metavar="PART",
        choices=ALL_PART_IDS,
        help="End of the execution range (inclusive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything.",
    )
    return parser


def main() -> None:
    setup_run_logger("smm272_q1_main")
    parser  = _build_parser()
    args    = parser.parse_args()

    # --parts and --from/--to are partially mutually exclusive;
    # enforce that --parts cannot be combined with --from/--to.
    if args.parts and (args.from_part or args.to_part):
        parser.error("--parts cannot be combined with --from / --to.")

    selected = resolve_parts(
        parts     = args.parts,
        from_part = args.from_part,
        to_part   = args.to_part,
    )

    if not selected:
        logger.warning("No sub-parts matched the selection — nothing to run.")
        return

    run_parts(selected, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
