"""
Q1 Part 2 — Master script: Rolling-window VaR and ES (all four methods).

Modular structure
-----------------
q1_2_rolling_window.py   → sliding 6-month window generator
q1_2_var_historical.py   → Method 1 : Historical Simulation
q1_2_var_normal.py       → Method 2 : Parametric Normal
q1_2_var_studentt.py     → Method 3 : Parametric Student-t (MLE)
q1_2_var_garch.py        → Method 4 : GARCH(1,1) dynamic conditional VaR/ES

Outputs (saved to Q1/output_q1_2/)
------------------------------------
rolling_var_es_results.csv          — full time series of all estimates
backtest_summary.csv                — exceedance rates per method
fig_rolling_var_es_combined.png     — VaR(99%) and ES(99%) side by side vs actual returns
fig_var_es_breach_panels.png        — per-method panel with breach scatter
fig_exceedance_rates.png            — bar chart: actual vs expected exceedance
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    VAR_CONFIDENCE_LEVEL,
    ROLLING_START_DATE,
    ROLLING_WINDOW_MONTHS,
    Q1_2_OUTPUT_DIR,
)
from logger import setup_run_logger, get_logger, log_start, log_end
from q1_1_build_portfolio import build_portfolio
from q1_2_rolling_window import generate_rolling_windows
from q1_2_var_garch      import ARCH_AVAILABLE
from q1_methods          import METHODS

logger = get_logger("q1_2_statistical_analysis")


# ── Core rolling loop ─────────────────────────────────────────────────────────

def run_rolling_var_es(portfolio_returns):
    """
    Iterate through rolling windows and compute VaR/ES by all four methods.

    Returns
    -------
    pd.DataFrame
        Columns: Actual_Return, <Method>_VaR, <Method>_ES for each method.
        Indexed by forecast date.
    """
    windows = list(generate_rolling_windows(
        portfolio_returns,
        start_date=ROLLING_START_DATE,
        window_months=ROLLING_WINDOW_MONTHS,
    ))
    total = len(windows)
    logger.info(f"  Total rolling windows : {total:,}")
    logger.info(f"  First forecast date   : {windows[0][0].date()}")
    logger.info(f"  Last  forecast date   : {windows[-1][0].date()}")

    records = []
    for i, (date, window) in enumerate(windows):
        if i % 500 == 0 or i == total - 1:
            logger.info(f"  [{i+1:>5}/{total}] {date.date()}")

        row = {"Date": date}
        row["Actual_Return"] = (
            float(portfolio_returns.loc[date])
            if date in portfolio_returns.index else np.nan
        )

        for tag, _, fn, _ in METHODS:
            var, es      = fn(window)
            row[f"{tag}_VaR"] = var
            row[f"{tag}_ES"]  = es

        records.append(row)

    df = pd.DataFrame(records).set_index("Date")
    return df


# ── Summary / backtesting ─────────────────────────────────────────────────────

def log_and_return_summary(df):
    """Log exceedance-rate summary and return a summary DataFrame."""
    expected_rate = 1.0 - VAR_CONFIDENCE_LEVEL   # 0.01

    logger.info("\n" + "=" * 70)
    logger.info(f"ROLLING VaR / ES — BACKTEST SUMMARY"
                f"  (confidence {VAR_CONFIDENCE_LEVEL*100:.0f}%,"
                f"  expected exceedance {expected_rate*100:.1f}%)")
    logger.info("=" * 70)
    logger.info(
        f"  {'Method':<14} {'Avg VaR%':>9} {'Avg ES%':>9}"
        f" {'Exceed':>8} {'Exceed%':>9} {'vs Expected':>12}"
    )
    logger.info("  " + "-" * 65)

    summary_rows = []
    for tag, label, _, _ in METHODS:
        var_col = f"{tag}_VaR"
        es_col  = f"{tag}_ES"
        valid   = df[[var_col, "Actual_Return"]].dropna()
        n       = len(valid)
        exceed  = int((valid["Actual_Return"] < valid[var_col]).sum())
        rate    = exceed / n if n > 0 else np.nan
        avg_var = df[var_col].mean()
        avg_es  = df[es_col].mean()

        flag = ""
        if rate is not np.nan:
            if rate > expected_rate * 1.5:
                flag = " ← OVER (under-conservative)"
            elif rate < expected_rate * 0.5:
                flag = " ← UNDER (over-conservative)"

        logger.info(
            f"  {label:<14} {avg_var*100:>9.4f} {avg_es*100:>9.4f}"
            f" {exceed:>8,d} {rate*100:>8.3f}%"
            f" {(rate-expected_rate)*100:>+10.3f}%{flag}"
        )
        summary_rows.append({
            "Method":            label,
            "Avg VaR (%)":      avg_var * 100,
            "Avg ES (%)":       avg_es * 100,
            "Exceedances":       exceed,
            "Exceedance Rate (%)": rate * 100,
        })

    return pd.DataFrame(summary_rows)


# ── Visualisations ────────────────────────────────────────────────────────────

def _fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def generate_plots(df):
    """Generate four diagnostic figures and save to Q1_2_OUTPUT_DIR."""
    colours = {tag: col for tag, _, _, col in METHODS}
    labels  = {tag: lbl for tag, lbl, _, _ in METHODS}

    # ── Figure 1+2: VaR and ES side by side ─────────────────────────────
    fig, (ax_var, ax_es) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

    # Left panel — VaR
    ax_var.plot(df.index, df["Actual_Return"] * 100,
                color="#aaaaaa", alpha=0.5, linewidth=0.5, label="Daily Return", zorder=1)
    for tag, _, _, col in METHODS:
        ax_var.plot(df.index, df[f"{tag}_VaR"] * 100,
                    color=colours[tag], linewidth=1.1,
                    label=f"{labels[tag]} VaR(99%)", zorder=2)
    ax_var.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax_var.set_title("Rolling 6-Month VaR (99%)", fontsize=12)
    ax_var.set_xlabel("Date")
    ax_var.set_ylabel("Daily Return (%)")
    ax_var.legend(ncol=2, fontsize=8, loc="lower left")
    _fmt_xaxis(ax_var)

    # Right panel — ES
    ax_es.plot(df.index, df["Actual_Return"] * 100,
               color="#aaaaaa", alpha=0.5, linewidth=0.5, label="Daily Return", zorder=1)
    for tag, _, _, col in METHODS:
        ax_es.plot(df.index, df[f"{tag}_ES"] * 100,
                   color=colours[tag], linewidth=1.1,
                   label=f"{labels[tag]} ES(99%)", zorder=2)
    ax_es.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax_es.set_title("Rolling 6-Month Expected Shortfall (99%)", fontsize=12)
    ax_es.set_xlabel("Date")
    ax_es.legend(ncol=2, fontsize=8, loc="lower left")
    _fmt_xaxis(ax_es)

    fig.tight_layout()
    fig.savefig(os.path.join(Q1_2_OUTPUT_DIR, "fig_rolling_var_es_combined.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved: fig_rolling_var_es_combined.png")

    # ── Figure 3: Per-method panels with breach scatter ───────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for i, (ax, (tag, lbl, _, col)) in enumerate(zip(axes, METHODS)):
        ax.plot(df.index, df["Actual_Return"] * 100,
                color="#aaaaaa", alpha=0.4, linewidth=0.5, label="Daily Return")
        ax.plot(df.index, df[f"{tag}_VaR"] * 100,
                color=col, linewidth=1.2, label="VaR(99%)")
        ax.fill_between(df.index,
                        df[f"{tag}_VaR"] * 100,
                        df[f"{tag}_ES"]  * 100,
                        alpha=0.18, color=col, label="VaR→ES band")
        breaches = df[df["Actual_Return"] < df[f"{tag}_VaR"]]
        ax.scatter(breaches.index, breaches["Actual_Return"] * 100,
                   color="red", s=8, zorder=5, label=f"Breach ({len(breaches)})")
        if i == 0:
            ax.set_ylabel("Return (%)", fontsize=10)
        ax.set_title(lbl, fontsize=10)
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.axhline(0, color="black", linewidth=0.3, linestyle="--")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(Q1_2_OUTPUT_DIR, "fig_var_es_breach_panels.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: fig_var_es_breach_panels.png")

    # ── Figure 4: Exceedance-rate comparison bar chart ────────────────────
    expected_rate = (1.0 - VAR_CONFIDENCE_LEVEL) * 100
    tags  = [t for t, *_ in METHODS]
    lbls  = [lbl for _, lbl, *_ in METHODS]
    cols  = [col for *_, col in METHODS]
    rates = []
    for tag in tags:
        valid  = df[[f"{tag}_VaR", "Actual_Return"]].dropna()
        exceed = (valid["Actual_Return"] < valid[f"{tag}_VaR"]).sum()
        rates.append(exceed / len(valid) * 100 if len(valid) > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lbls, rates, color=cols, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(expected_rate, color="black", linestyle="--", linewidth=1.5,
               label=f"Expected {expected_rate:.1f}%")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"VaR(99%) Exceedance Rates — Rolling 6-Month Window", fontsize=12)
    ax.set_ylabel("Exceedance Rate (%)")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.legend(fontsize=10)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(Q1_2_OUTPUT_DIR, "fig_exceedance_rates.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved: fig_exceedance_rates.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = setup_run_logger("smm272_q1_2")
    log_start(logger, "q1_2_statistical_analysis.py")

    logger.info("=" * 70)
    logger.info("SMM272 Q1 Part 2 — Rolling VaR and ES (4 Methods)")
    logger.info("=" * 70)
    logger.info(f"  Confidence level  : {VAR_CONFIDENCE_LEVEL*100:.0f}%")
    logger.info(f"  Rolling window    : {ROLLING_WINDOW_MONTHS} months")
    logger.info(f"  Start date        : {ROLLING_START_DATE}")
    logger.info(f"  GARCH (arch pkg)  : {'available' if ARCH_AVAILABLE else 'NOT available — Normal fallback'}")
    logger.info(f"  Run log           : {log_path}")

    # Load portfolio returns (uses cached CSV if available)
    _, _, portfolio_returns = build_portfolio()

    # ── Rolling estimation loop ───────────────────────────────────────────
    logger.info("\n--- Running rolling windows ---")
    results = run_rolling_var_es(portfolio_returns)

    # ── Save full results ─────────────────────────────────────────────────
    out_csv = os.path.join(Q1_2_OUTPUT_DIR, "rolling_var_es_results.csv")
    results.to_csv(out_csv)
    logger.info(f"\n  Results saved to : {out_csv}")
    logger.info(f"  Shape            : {results.shape}")

    # ── Backtest summary ──────────────────────────────────────────────────
    summary_df = log_and_return_summary(results)
    summary_df.to_csv(os.path.join(Q1_2_OUTPUT_DIR, "backtest_summary.csv"), index=False)

    # ── Visualisations ────────────────────────────────────────────────────
    logger.info("\n--- Generating figures ---")
    generate_plots(results)

    logger.info(f"\n  All outputs saved to: {Q1_2_OUTPUT_DIR}")
    log_end(logger, "q1_2_statistical_analysis.py")
    return results


if __name__ == "__main__":
    main()
