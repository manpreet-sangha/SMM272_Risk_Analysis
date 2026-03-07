"""
Q1 Part 4 — Visualisations for VaR backtesting results.

Generates two figures:

  fig_backtest_pvalues_heatmap.png
      Single compact heatmap — rows = (model, confidence level),
      columns = backtests.  Deep teal = high p-value (pass); amber =
      low p-value (reject).  No red is used anywhere.

  fig_backtest_lr_stats.png
      Grouped bar chart comparing observed LR / DQ test statistics against
      the corresponding chi^2 critical values at 5 % significance.
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
import matplotlib.colors as mcolors

from config import Q1_4_OUTPUT_DIR
from logger import get_logger
from scipy import stats

logger = get_logger("q1_4_plots")

# Critical values at 5%
_CRIT = {
    "Kupiec (UC)":   stats.chi2.ppf(0.95, df=1),       # ~3.841
    "Independence":  stats.chi2.ppf(0.95, df=1),        # ~3.841
    "CC":            stats.chi2.ppf(0.95, df=2),        # ~5.991
    "Duration":      stats.chi2.ppf(0.95, df=2),        # ~5.991
    "DQ (K=4)":      stats.chi2.ppf(0.95, df=5),        # ~11.070
}

_PVAL_COLS  = ["p_uc",    "p_ind",  "p_cc",  "p_dur",  "p_dq"]
_STAT_COLS  = ["LR_uc",   "LR_ind", "LR_cc", "LR_dur", "DQ_stat"]
_TEST_NAMES = ["Kupiec (UC)", "Independence", "CC", "Duration", "DQ (K=4)"]


def generate_plots(results_df, confidence_levels, methods):
    """
    Generate backtest result figures and save to Q1_4_OUTPUT_DIR.

    Parameters
    ----------
    results_df        : pd.DataFrame — output of run_all_backtests()
    confidence_levels : list of float — e.g. [0.90, 0.95, 0.99]
    methods           : list of (tag, label, fn, colour)
    """
    tags    = [t for t, *_ in methods]
    labels  = {t: lbl for t, lbl, _, _ in methods}
    colours = {t: col for t, _, _, col in methods}
    ci_pcts = [int(round(c * 100)) for c in confidence_levels]

    _plot_pvalue_heatmap(results_df, tags, labels, ci_pcts)
    _plot_lr_barchart(results_df, tags, labels, colours, ci_pcts)


# ── Figure 1: P-value heatmap ─────────────────────────────────────────────────

def _plot_pvalue_heatmap(results_df, tags, labels, ci_pcts):
    """
    2×3 grid of sub-panels (5 tests + 1 colorbar panel).
      Each panel: rows = 4 models, columns = CI levels.
    Colormap: deep teal (pass) → amber (fail). No red.
    Rejected cells (p < 0.05) are marked with asterisks.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec

    _TEAL_AMBER = LinearSegmentedColormap.from_list(
        "teal_amber",
        ["#F5A623", "#F8D89A", "#F5F5F0", "#A8D5D0", "#1A7A75"],
    )
    norm     = mcolors.Normalize(vmin=0, vmax=1)
    n_rows_m = len(tags)
    n_cols_ci = len(ci_pcts)
    ci_labels = [f"{p}%" for p in ci_pcts]
    row_labels = [labels[t] for t in tags]

    # 2×3 grid: first 5 cells = tests, last cell = colorbar
    fig = plt.figure(figsize=(11, 7))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.55)
    panel_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    test_axes  = panel_axes[:5]
    cbar_ax    = panel_axes[5]

    # Column index for each of the 5 test panels: 0,1,2,0,1
    _panel_col = [0, 1, 2, 0, 1]

    for panel_idx, (ax, pval_col, test_name) in enumerate(
        zip(test_axes, _PVAL_COLS, _TEST_NAMES)
    ):
        matrix = np.full((n_rows_m, n_cols_ci), np.nan)
        for i, tag in enumerate(tags):
            for j, ci_pct in enumerate(ci_pcts):
                mask = (
                    (results_df["Tag"]            == tag) &
                    (results_df["Confidence (%)"] == ci_pct)
                )
                if mask.sum() > 0:
                    try:
                        matrix[i, j] = float(results_df.loc[mask, pval_col].values[0])
                    except (TypeError, ValueError):
                        pass

        ax.imshow(matrix, cmap=_TEAL_AMBER, norm=norm,
                  aspect="auto", interpolation="nearest")

        # Cell separators
        for x_line in range(1, n_cols_ci):
            ax.axvline(x_line - 0.5, color="white", linewidth=1.0, zorder=3)
        for y_line in range(1, n_rows_m):
            ax.axhline(y_line - 0.5, color="white", linewidth=1.0, zorder=3)

        # Annotations
        for i in range(n_rows_m):
            for j in range(n_cols_ci):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=7, color="#888888", zorder=4)
                else:
                    rgba      = _TEAL_AMBER(norm(val))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    txt_color = "white" if luminance < 0.50 else "#1a1a2e"
                    reject    = val < 0.05
                    txt       = f"{'*' if reject else ''}{val:.3f}{'*' if reject else ''}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8, fontweight="bold" if reject else "normal",
                            color=txt_color, zorder=4)

        ax.set_title(test_name, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks(range(n_cols_ci))
        ax.set_xticklabels(ci_labels, fontsize=8)
        ax.set_yticks(range(n_rows_m))
        # Only leftmost column of each row shows model name labels
        if _panel_col[panel_idx] == 0:
            ax.set_yticklabels(row_labels, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)

    # Colorbar in the 6th cell
    cbar_ax.set_visible(False)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=_TEAL_AMBER, norm=norm),
        ax=cbar_ax, fraction=0.9, shrink=0.85,
    )
    cbar_ax.set_visible(True)
    cbar_ax.axis("off")
    cbar.set_label("p-value", fontsize=9)
    cbar.set_ticks([0, 0.05, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.axhline(0.05, color="#1a1a2e", linewidth=1.5, linestyle="--")

    fig.suptitle(
        "VaR Backtesting — p-values  (* p < 0.05: reject H\u2080 | amber = fail, teal = pass)",
        fontsize=10, y=1.01,
    )

    out = os.path.join(Q1_4_OUTPUT_DIR, "fig_backtest_pvalues_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Figure 2: LR / DQ statistics vs critical values ─────────────────────────

def _plot_lr_barchart(results_df, tags, labels, colours, ci_pcts):
    """
    One sub-panel per CI level.  Grouped bars show the observed test
    statistic for each method; dashed horizontal lines mark the 5% critical
    values.  Legend is placed below the figure to avoid overlapping bars.
    """
    n_ci   = len(ci_pcts)
    n_tags = len(tags)
    # Taller figure so text has room; extra bottom margin for shared legend
    fig, axes = plt.subplots(1, n_ci, figsize=(6 * n_ci, 6.5), sharey=False)
    if n_ci == 1:
        axes = [axes]
    fig.suptitle(
        "Q1 Part 4 — Backtest statistics vs 5% critical values",
        fontsize=15, fontweight="bold",
    )

    n_tests = len(_TEST_NAMES)
    x       = np.arange(n_tests)
    width   = 0.8 / n_tags

    legend_handles = []

    for ax, ci_pct in zip(axes, ci_pcts):
        for j, tag in enumerate(tags):
            mask = (
                (results_df["Tag"]            == tag) &
                (results_df["Confidence (%)"] == ci_pct)
            )
            if mask.sum() == 0:
                continue
            row    = results_df[mask].iloc[0]
            values = [_safe_float(row.get(c)) for c in _STAT_COLS]
            offsets = x + (j - n_tags / 2.0 + 0.5) * width
            bar = ax.bar(
                offsets, values,
                width=width,
                color=colours[tag],
                alpha=0.85,
                label=labels[tag],
                edgecolor="white", linewidth=0.5,
            )
            # Value labels on top of each bar
            for rect, val in zip(bar, values):
                if val > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.05,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#333333",
                    )
            # Collect handles once (from first CI panel)
            if ci_pct == ci_pcts[0]:
                legend_handles.append(bar)

        # Critical value lines with labels
        for test_name, crit in _CRIT.items():
            idx = _TEST_NAMES.index(test_name)
            ax.hlines(
                crit,
                xmin=idx - 0.45,
                xmax=idx + 0.45,
                colors="#2c5f8a",
                linewidths=2.2,
                linestyles="--",
                zorder=5,
            )

        ax.set_title(f"CI = {ci_pct}%", fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(_TEST_NAMES, rotation=25, ha="right", fontsize=11)
        ax.set_ylabel("Test statistic", fontsize=11)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Add critical value line to legend
    from matplotlib.lines import Line2D
    crit_handle = Line2D([0], [0], color="#2c5f8a", linewidth=2.2,
                         linestyle="--", label="5% critical value")

    all_handles = [b[0] for b in legend_handles] + [crit_handle]
    all_labels  = [labels[t] for t in tags] + ["5% critical value"]

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        ncol=len(all_handles),
        fontsize=10.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(Q1_4_OUTPUT_DIR, "fig_backtest_lr_stats.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


def _safe_float(val):
    """Return float value or 0.0 if NaN/None."""
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0
