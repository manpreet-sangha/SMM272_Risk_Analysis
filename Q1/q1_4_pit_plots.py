"""
Q1 Part 4 — PIT diagnostic visualisations.

Generates the following figures in Q1/output_q1_4/:

  fig_pit_qq.png
      One Q-Q plot per model comparing the PIT series z_t against the
      Uniform(0, 1) theoretical quantiles.  Points on the 45° line = perfect
      calibration; systematic deviations indicate distributional mis-fit.

  fig_pit_histograms.png
      PIT histograms with a Uniform(0, 1) reference line.  A well-specified
      density produces a flat histogram.

  fig_pit_autocorrelation.png
      ACF of the normal-probability transform u_t = Φ⁻¹(z_t) and its
      square u_t².  Under correct specification both ACFs should be flat
      (inside Bartlett confidence bands).

  fig_pit_pvalues_heatmap.png
      Compact heatmap of p-values for all four PIT tests ×  four models.
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
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from config import Q1_4_OUTPUT_DIR
from logger import get_logger

logger = get_logger("q1_4_pit_plots")

# Colour map matching q1_4_plots.py style
_TEAL_AMBER = LinearSegmentedColormap.from_list(
    "teal_amber",
    ["#F5A623", "#F8D89A", "#F5F5F0", "#A8D5D0", "#1A7A75"],
)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pit_plots(pit_series: dict, results_df: pd.DataFrame, methods):
    """
    Generate all PIT diagnostic figures.

    Parameters
    ----------
    pit_series : dict
        ``{tag: pd.Series}`` of PIT values z_t, as returned by
        ``run_pit_tests()``.
    results_df : pd.DataFrame
        PIT test results table (one row per model), as returned by
        ``run_pit_tests()``.
    methods : list of (tag, label, compute_fn, colour)
        Model registry.
    """
    labels  = {t: lbl for t, lbl, _, _ in methods}
    colours = {t: col for t, _, _, col in methods}
    tags    = [t for t, *_ in methods]

    _plot_qq(pit_series, tags, labels, colours)
    _plot_histograms(pit_series, tags, labels, colours)
    _plot_acf(pit_series, tags, labels, colours)
    _plot_pvalue_heatmap(results_df, tags, labels)


# ── Figure 1: Q-Q plots ──────────────────────────────────────────────────────

def _plot_qq(pit_series, tags, labels, colours):
    """
    Uniform Q-Q plot for each model.

    The theoretical quantiles are U(0,1) order statistics:
        q_i = (i − 0.5) / n  for i = 1, …, n

    Points lying on the 45° line indicate perfect calibration.
    """
    n_models = len(tags)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5),
                             squeeze=False)
    axes = axes.ravel()

    for ax, tag in zip(axes, tags):
        z = np.sort(np.asarray(pit_series[tag], dtype=float))
        z = z[~np.isnan(z)]
        n = len(z)
        if n == 0:
            ax.set_title(f"{labels[tag]} — no data")
            continue

        # Theoretical Uniform(0,1) quantiles (Hazen plotting positions)
        theoretical = (np.arange(1, n + 1) - 0.5) / n

        ax.scatter(theoretical, z, s=4, alpha=0.4, color=colours[tag],
                   edgecolors="none", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#1a1a2e", linewidth=1.2,
                linestyle="--", label="45° line")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Theoretical U(0,1) quantiles", fontsize=9)
        ax.set_ylabel("Empirical PIT quantiles", fontsize=9)
        ax.set_title(labels[tag], fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "PIT Q-Q Plots — Uniform(0, 1) Reference",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(Q1_4_OUTPUT_DIR, "fig_pit_qq.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Figure 2: PIT histograms ─────────────────────────────────────────────────

def _plot_histograms(pit_series, tags, labels, colours):
    """
    PIT histograms with 20 bins.  Dashed line = expected Uniform frequency.
    """
    n_bins   = 20
    n_models = len(tags)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4),
                             squeeze=False)
    axes = axes.ravel()

    for ax, tag in zip(axes, tags):
        z = np.asarray(pit_series[tag], dtype=float)
        z = z[~np.isnan(z)]
        n = len(z)
        if n == 0:
            ax.set_title(f"{labels[tag]} — no data")
            continue

        ax.hist(z, bins=n_bins, range=(0, 1), density=True,
                color=colours[tag], alpha=0.7, edgecolor="white",
                linewidth=0.5)
        ax.axhline(1.0, color="#1a1a2e", linewidth=1.2, linestyle="--",
                   label="Uniform density")

        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT value $z_t$", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(labels[tag], fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        "PIT Histograms — Uniform(0, 1) Reference",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(Q1_4_OUTPUT_DIR, "fig_pit_histograms.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Figure 3: ACF of u_t and u_t² ────────────────────────────────────────────

def _plot_acf(pit_series, tags, labels, colours):
    """
    Two-row panel per model:
      Row 1: ACF of u_t  = Φ⁻¹(z_t)         — first-moment dependence
      Row 2: ACF of u_t² = [Φ⁻¹(z_t)]²     — second-moment / ARCH effects
    """
    n_models = len(tags)
    n_lags   = 20
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 6),
                             squeeze=False)

    for j, tag in enumerate(tags):
        z = np.asarray(pit_series[tag], dtype=float)
        z = z[~np.isnan(z)]

        if len(z) <= n_lags + 1:
            for row in range(2):
                axes[row, j].set_title(f"{labels[tag]} — insufficient data")
            continue

        # Clip to open (0, 1) to avoid ±inf from ppf
        z = np.clip(z, 1e-10, 1 - 1e-10)
        u  = stats.norm.ppf(z)
        u2 = u ** 2

        for row, series, series_label in [
            (0, u,  "$u_t = \\Phi^{-1}(z_t)$"),
            (1, u2, "$u_t^2 = [\\Phi^{-1}(z_t)]^2$"),
        ]:
            ax = axes[row, j]
            plot_acf(series, ax=ax, lags=n_lags, alpha=0.05,
                     color=colours[tag], vlines_kwargs={"color": colours[tag]},
                     title="")
            ax.set_title(
                f"{labels[tag]}\n{series_label}",
                fontsize=9, fontweight="bold",
            )
            ax.set_xlabel("Lag", fontsize=8)
            ax.set_ylabel("ACF", fontsize=8)

    fig.suptitle(
        "PIT Independence — ACF of $u_t$ and $u_t^2$",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(Q1_4_OUTPUT_DIR, "fig_pit_autocorrelation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Figure 4: p-value heatmap ─────────────────────────────────────────────────

def _plot_pvalue_heatmap(results_df, tags, labels):
    """
    Single heatmap: rows = models, columns = PIT tests.
    Same teal-amber palette as the backtest heatmap for visual consistency.
    """
    _PVAL_COLS  = ["p_KS", "p_Chi2", "p_LB_u", "p_LB_u2"]
    _TEST_NAMES = ["KS Uniformity", "χ² Uniformity",
                   "LB($u_t$)", "LB($u_t^2$)"]

    n_rows = len(tags)
    n_cols = len(_PVAL_COLS)
    norm   = mcolors.Normalize(vmin=0, vmax=1)

    row_labels = [labels[t] for t in tags]

    matrix = np.full((n_rows, n_cols), np.nan)
    for i, tag in enumerate(tags):
        mask = results_df["Tag"] == tag
        if mask.sum() == 0:
            continue
        row = results_df[mask].iloc[0]
        for j, col in enumerate(_PVAL_COLS):
            try:
                matrix[i, j] = float(row[col])
            except (TypeError, ValueError):
                pass

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(matrix, cmap=_TEAL_AMBER, norm=norm,
                   aspect="auto", interpolation="nearest")

    # Grid lines
    for x in range(1, n_cols):
        ax.axvline(x - 0.5, color="white", linewidth=1.0, zorder=3)
    for y in range(1, n_rows):
        ax.axhline(y - 0.5, color="white", linewidth=1.0, zorder=3)

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="#888888", zorder=4)
            else:
                rgba      = _TEAL_AMBER(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if luminance < 0.50 else "#1a1a2e"
                reject    = val < 0.05
                txt       = f"{'*' if reject else ''}{val:.3f}{'*' if reject else ''}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight="bold" if reject else "normal",
                        color=txt_color, zorder=4)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_TEST_NAMES, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("p-value", fontsize=9)
    cbar.set_ticks([0, 0.05, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.axhline(0.05, color="#1a1a2e", linewidth=1.5, linestyle="--")

    ax.set_title(
        "PIT Tests — p-values  (* p < 0.05: reject H₀ | amber = fail, teal = pass)",
        fontsize=10, fontweight="bold", pad=10,
    )

    fig.tight_layout()
    out = os.path.join(Q1_4_OUTPUT_DIR, "fig_pit_pvalues_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")
