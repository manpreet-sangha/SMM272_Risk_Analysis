"""
Q1 Part 3 — Visualisations for VaR violation analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import Q1_3_OUTPUT_DIR
from logger import get_logger

logger = get_logger("q1_3_plots")


def _fmt_xaxis(ax):
    """Apply year-level date formatting to a matplotlib x-axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def generate_plots(df, violations_df, violation_flags, confidence_levels, methods):
    """
    Generate three diagnostic figures and save to Q1_3_OUTPUT_DIR.

    Parameters
    ----------
    df                : pd.DataFrame — rolling VaR results (from run_rolling_all_levels)
    violations_df     : pd.DataFrame — violation counts / rates
    violation_flags   : pd.DataFrame — daily 0/1 flags per (method, CI)
    confidence_levels : list of float — e.g. [0.90, 0.95, 0.99]
    methods           : list of (tag, label, fn, colour)
    """
    colours = {tag: col for tag, _, _, col in methods}
    labels  = {tag: lbl for tag, lbl, _, _ in methods}
    ci_pcts = [int(round(c * 100)) for c in confidence_levels]
    ci_strs = [f"{p}%" for p in ci_pcts]

    # ── Figure 1: Grouped bar chart — observed vs expected breach rates ───
    tags      = [t for t, *_ in methods]
    n_ci      = len(ci_pcts)
    x         = np.arange(len(tags))
    width     = 0.22
    fig, ax   = plt.subplots(figsize=(11, 5))

    for j, (ci, ci_pct) in enumerate(zip(confidence_levels, ci_pcts)):
        nominal = (1.0 - ci) * 100
        rates = []
        for tag in tags:
            row = violations_df[
                (violations_df["Tag"] == tag) &
                (violations_df["Confidence (%)"] == ci * 100)
            ]
            rates.append(float(row["Observed Rate (%)"].values[0]))

        ax.bar(
            x + (j - (n_ci - 1) / 2) * width, rates,
            width=width * 0.9,
            color=[colours[t] for t in tags],
            alpha=0.75 + j * 0.08,
            label=f"Observed ({ci_pct}% CI)",
            edgecolor="black", linewidth=0.4,
        )
        ax.axhline(nominal, color=f"C{j}", linestyle="--", linewidth=1.2,
                   label=f"Expected {ci_pct}% CI = {nominal:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in tags], rotation=12, ha="right")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title(
        "VaR Violation Rates vs Expected — All Methods and Confidence Levels",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(Q1_3_OUTPUT_DIR, "fig_violations_barchart.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved: fig_violations_barchart.png")

    # ── Figure 2: Heatmap — observed vs expected violation counts ────────
    tags_all   = [t for t, *_ in methods]
    lbl_all    = [lbl for _, lbl, *_ in methods]
    matrix     = np.zeros((len(tags_all), len(ci_pcts)))
    exp_matrix = np.zeros_like(matrix)

    for i, tag in enumerate(tags_all):
        for j, (ci, ci_pct) in enumerate(zip(confidence_levels, ci_pcts)):
            row = violations_df[
                (violations_df["Tag"] == tag) &
                (violations_df["Confidence (%)"] == ci * 100)
            ]
            matrix[i, j]     = float(row["Violations (k)"].values[0])
            exp_matrix[i, j] = float(row["Expected (\u2248)"].values[0])

    # Professional colormaps: warm (YlOrRd) for observed, cool (GnBu) for expected
    _CMAPS = ["YlOrRd", "GnBu"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title, cmap_name in zip(
        axes,
        [matrix, exp_matrix],
        ["Observed Violations (k)", "Expected Violations (\u2248)"],
        _CMAPS,
    ):
        cmap = plt.get_cmap(cmap_name)
        vmin, vmax = data.min(), data.max()
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(ci_strs)))
        ax.set_xticklabels(ci_strs, fontsize=10)
        ax.set_yticks(range(len(lbl_all)))
        ax.set_yticklabels(lbl_all, fontsize=9)
        ax.set_xlabel("Confidence Level", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.85)
        # Draw thin white separator lines between cells
        n_rows, n_cols = data.shape
        for x_line in range(1, n_cols):
            ax.axvline(x_line - 0.5, color="white", linewidth=1.2, zorder=3)
        for y_line in range(1, n_rows):
            ax.axhline(y_line - 0.5, color="white", linewidth=1.2, zorder=3)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)

        # Auto-contrast text: white on dark cells, dark navy on light cells
        for i in range(len(tags_all)):
            for j in range(len(ci_pcts)):
                norm_val = (data[i, j] - vmin) / max(vmax - vmin, 1e-9)
                r, g, b, _ = cmap(norm_val)
                luminance  = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color  = "white" if luminance < 0.50 else "#1a1a2e"
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_color)

    fig.suptitle(
        "VaR Violation Counts — All Methods and Confidence Levels", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(os.path.join(Q1_3_OUTPUT_DIR, "fig_violations_heatmap.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved: fig_violations_heatmap.png")

    # ── Figure 3: Cumulative violations over time (one panel per CI level) ─
    fig, axes = plt.subplots(
        len(ci_pcts), 1,
        figsize=(14, 4 * len(ci_pcts)),
        sharex=True,
    )
    if len(ci_pcts) == 1:
        axes = [axes]

    for ax, (ci, ci_pct) in zip(axes, zip(confidence_levels, ci_pcts)):
        nominal_daily = 1.0 - ci
        for tag, lbl, _, col in methods:
            col_flag = f"{tag}_{ci_pct}"
            if col_flag in violation_flags.columns:
                series = violation_flags[col_flag].fillna(0).astype(float)
                ax.plot(violation_flags.index, series.cumsum(),
                        color=col, linewidth=1.2, label=lbl)

        # Expected cumulative violations (straight line)
        valid_flags = violation_flags.dropna(how="all")
        n_obs = len(valid_flags)
        expected_cum = np.arange(1, n_obs + 1) * nominal_daily
        ax.plot(valid_flags.index, expected_cum,
                color="black", linewidth=1.5, linestyle="--",
                label=f"Expected ({(1 - ci) * 100:.0f}% tail)")

        ax.set_ylabel("Cumulative Violations")
        ax.set_title(
            f"Cumulative VaR Violations — {ci_pct}% Confidence Level", fontsize=10
        )
        ax.legend(fontsize=8, ncol=3, loc="upper left")

    _fmt_xaxis(axes[-1])
    axes[-1].set_xlabel("Date")
    fig.suptitle("Cumulative VaR Violations Over Time", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(Q1_3_OUTPUT_DIR, "fig_violation_timeseries.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    logger.info("  Saved: fig_violation_timeseries.png")
