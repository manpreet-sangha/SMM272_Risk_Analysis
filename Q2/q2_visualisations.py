"""
Q2 — All visualisations for the Kupiec power analysis.

Figures produced
----------------
fig_q2_1_lr_distributions.png
    Empirical distribution of LR_uc under H0 (should match χ²(1)) and H1
    (GARCH), for each of the three confidence levels.  Demonstrates how
    well-separated the two distributions are, and therefore how much
    statistical power is available.

fig_q2_2_violation_distributions.png
    Histogram of violation counts under H0 (Binomial) and H1 (GARCH) for
    each confidence level, with the Binomial PMF overlay and the observed
    nominal rate marked.  Visualises the excess violations that drive
    Kupiec rejections.

fig_q2_3_power_vs_T.png
    Power curves as a function of sample size T for each confidence level,
    comparing simulated power (GARCH H1) vs. analytical Binomial power.
    Highlights the well-known low power of Kupiec at T = 250.

fig_q2_4_power_vs_persistence.png
    Kupiec power as a function of GARCH persistence ρ = α + β for each
    confidence level at T = 2893.  Shows how clustering amplifies
    violation frequency but can also reduce power via over-dispersion.

fig_q2_5_size_vs_T.png
    Empirical size (type-I error) as a function of T — should stay near
    the nominal 5% level, confirming test calibration.
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
import matplotlib.ticker as mticker
from scipy.stats import chi2 as chi2_dist, norm

from config import Q2_OUTPUT_DIR, Q2_CONFIDENCE_LEVELS
from logger import get_logger

logger = get_logger("q2_visualisations")

# ── Style constants ────────────────────────────────────────────────────────────
CONF_COLOURS = {0.90: "#1f77b4", 0.95: "#ff7f0e", 0.99: "#d62728"}
CONF_LABELS  = {0.90: "90% CI", 0.95: "95% CI", 0.99: "99% CI"}
DPI = 150
FIGSIZE_WIDE  = (12, 4.5)
FIGSIZE_TALL  = (10, 6)
FIGSIZE_PANEL = (14, 5)


def _save(fig, name):
    path = os.path.join(Q2_OUTPUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure → {path}")
    return path


# ── Figure 1: LR_uc distributions under H0 and H1 ────────────────────────────

def plot_lr_distributions(power_results):
    """
    One panel per confidence level.
    Left: histogram of LR_uc under H0 with χ²(1) overlay.
    Right: histogram of LR_uc under H1 with critical value marked.
    Combined into a single two-row × 3-col figure.
    """
    confs = [r["confidence"] for r in power_results]
    n     = len(confs)

    fig, axes = plt.subplots(2, n, figsize=(14, 7))
    fig.suptitle("Empirical Distribution of Kupiec $LR_{uc}$ Statistic\n"
                 "Top row: under H$_0$ (Gaussian); "
                 "Bottom row: under H$_1$ (GARCH(1,1))",
                 fontsize=12, fontweight="bold")

    # χ²(1) PDF overlay
    x_chi = np.linspace(0, 20, 300)
    y_chi = chi2_dist.pdf(x_chi, df=1)
    crit  = chi2_dist.ppf(0.95, df=1)

    for col, r in enumerate(power_results):
        conf  = r["confidence"]
        col_c = CONF_COLOURS[conf]
        lbl   = CONF_LABELS[conf]
        lr_h0 = r["lr_h0"]
        lr_h1 = r["lr_h1"]

        # Row 0: H0
        ax0 = axes[0, col]
        ax0.hist(lr_h0, bins=80, density=True, color=col_c, alpha=0.55,
                 label=f"H$_0$ (n={len(lr_h0):,})")
        ax0.plot(x_chi, y_chi, "k-", lw=1.5, label="$\\chi^2(1)$ PDF")
        ax0.axvline(crit, color="red", ls="--", lw=1.2,
                    label=f"5% crit = {crit:.2f}")
        ax0.set_title(f"{lbl} — H$_0$ (Gaussian)", fontsize=10)
        ax0.set_xlabel("$LR_{uc}$")
        ax0.set_ylabel("Density")
        ax0.set_xlim(0, 20)
        ax0.legend(fontsize=7)
        ax0.text(0.97, 0.97,
                 f"Size = {r['size']*100:.1f}%",
                 transform=ax0.transAxes, ha="right", va="top",
                 fontsize=9, color="red",
                 bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))

        # Row 1: H1
        ax1 = axes[1, col]
        ax1.hist(lr_h1, bins=80, density=True, color=col_c, alpha=0.55,
                 label=f"H$_1$ (GARCH, n={len(lr_h1):,})")
        ax1.axvline(crit, color="red", ls="--", lw=1.2,
                    label=f"5% crit = {crit:.2f}")
        ax1.set_title(f"{lbl} — H$_1$ (GARCH(1,1))", fontsize=10)
        ax1.set_xlabel("$LR_{uc}$")
        ax1.set_ylabel("Density")
        ax1.legend(fontsize=7)
        ax1.text(0.97, 0.97,
                 f"Power = {r['power']*100:.1f}%",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=9, color="darkgreen",
                 bbox=dict(boxstyle="round,pad=0.2", fc="lightgreen", alpha=0.5))

    plt.tight_layout()
    return _save(fig, "fig_q2_1_lr_distributions.png")


# ── Figure 2: Violation count distributions ───────────────────────────────────

def plot_violation_distributions(power_results):
    """
    Histogram of violation counts under H0 and H1 per confidence level.
    Shows Binomial(T, α) PMF as reference.
    """
    from scipy.stats import binom as binom_dist
    n = len(power_results)
    fig, axes = plt.subplots(1, n, figsize=FIGSIZE_PANEL)
    fig.suptitle("Violation Count Distributions: H$_0$ vs H$_1$",
                 fontsize=12, fontweight="bold")

    for ax, r in zip(axes, power_results):
        conf   = r["confidence"]
        alpha  = r["alpha"]
        T      = r["T"]
        col_c  = CONF_COLOURS[conf]
        lbl    = CONF_LABELS[conf]
        v_h0   = r["viol_h0"]
        v_h1   = r["viol_h1"]
        exp_v  = r["expected_violations"]

        # Shared bins
        all_v  = np.concatenate([v_h0, v_h1])
        bins   = np.arange(max(0, all_v.min() - 1), all_v.max() + 3)

        ax.hist(v_h0, bins=bins, density=True, color="steelblue", alpha=0.5,
                label="H$_0$ (Gaussian)")
        ax.hist(v_h1, bins=bins, density=True, color=col_c, alpha=0.5,
                label="H$_1$ (GARCH)")

        # Binomial PMF
        k_range = np.arange(bins[0], bins[-1] + 1)
        binom_pmf = binom_dist.pmf(k_range, T, alpha)
        ax.step(k_range, binom_pmf, "k-", lw=1.2, where="mid",
                label=f"Binom({T},{alpha:.2f})")

        ax.axvline(exp_v, color="black", ls="--", lw=1.2,
                   label=f"E[k]={exp_v:.0f}")

        ax.set_title(f"{lbl}  (T={T})", fontsize=10)
        ax.set_xlabel("Violation count $k$")
        ax.set_ylabel("Density / PMF")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    return _save(fig, "fig_q2_2_violation_distributions.png")


# ── Figure 3: Power vs. T ──────────────────────────────────────────────────────

def plot_power_vs_T(df_power_T):
    """
    Power curves as a function of sample size T.
    One line per confidence level for simulated power.
    Dashed: analytical Binomial power.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)

    for conf in Q2_CONFIDENCE_LEVELS:
        sub  = df_power_T[df_power_T["confidence"] == conf].sort_values("T")
        col  = CONF_COLOURS[conf]
        lbl  = CONF_LABELS[conf]

        ax.plot(sub["T"], sub["power_sim"] * 100,
                color=col, lw=2, marker="o", markersize=5, label=f"{lbl} (simulated)")
        ax.plot(sub["T"], sub["power_binomial"] * 100,
                color=col, lw=1.2, ls="--",
                label=f"{lbl} (Binomial approx.)")

    ax.axhline(5, color="gray", ls=":", lw=1.2, label="Nominal size (5%)")
    ax.axvline(250, color="red", ls="--", lw=1, alpha=0.7,
               label="T = 250 (1 trading year)")

    ax.set_xlabel("Sample size T (trading days)", fontsize=11)
    ax.set_ylabel("Power (%)", fontsize=11)
    ax.set_title("Power of the Kupiec Test vs. Sample Size\n"
                 "(H$_1$: GARCH(1,1); Gaussian VaR used for violations)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "fig_q2_3_power_vs_T.png")


# ── Figure 4: Power vs. GARCH persistence ─────────────────────────────────────

def plot_power_vs_persistence(df_persist):
    """
    Kupiec power vs. GARCH persistence ρ = α + β  for all three CIs.
    Vertical dashed line marks the fitted persistence from data.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)

    for conf in Q2_CONFIDENCE_LEVELS:
        sub = df_persist[df_persist["confidence"] == conf].sort_values("persistence")
        ax.plot(sub["persistence"], sub["power"] * 100,
                color=CONF_COLOURS[conf], lw=2, marker="o", markersize=5,
                label=CONF_LABELS[conf])

    ax.axhline(5, color="gray", ls=":", lw=1.2, label="Nominal size (5%)")
    ax.set_xlabel("GARCH persistence  ρ = α + β", fontsize=11)
    ax.set_ylabel("Power (%)", fontsize=11)
    ax.set_title("Kupiec Power vs. GARCH Persistence\n"
                 "$T = 2893$; unconditional variance held constant",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(df_persist["persistence"].min() - 0.02,
                df_persist["persistence"].max() + 0.01)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "fig_q2_4_power_vs_persistence.png")


# ── Figure 5: Size (empirical type-I error) vs. T ─────────────────────────────

def plot_size_vs_T(df_power_T):
    """
    Empirical size of the Kupiec test as a function of T.
    Should be close to 5% for all T and all confidence levels.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)

    for conf in Q2_CONFIDENCE_LEVELS:
        sub = df_power_T[df_power_T["confidence"] == conf].sort_values("T")
        ax.plot(sub["T"], sub["size"] * 100,
                color=CONF_COLOURS[conf], lw=2, marker="s", markersize=5,
                label=CONF_LABELS[conf])

    ax.axhline(5, color="black", ls="--", lw=1.2, label="Nominal 5% level")
    ax.fill_between([df_power_T["T"].min(), df_power_T["T"].max()],
                    3.5, 6.5, color="gray", alpha=0.15, label="±1.5 pp band")

    ax.set_xlabel("Sample size T (trading days)", fontsize=11)
    ax.set_ylabel("Empirical size (%)", fontsize=11)
    ax.set_title("Empirical Size of the Kupiec Test vs. Sample Size\n"
                 "(H$_0$: Gaussian; should ≈ 5%)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save(fig, "fig_q2_5_size_vs_T.png")


# ── Figure 6: Power heatmap (T × confidence) ──────────────────────────────────

def plot_power_heatmap(df_power_T):
    """
    Heatmap of power(%) across (T, confidence level) combinations.
    """
    pivot = df_power_T.pivot(index="T", columns="confidence", values="power_sim") * 100
    pivot.columns = [f"{c*100:.0f}%" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=100, origin="lower")
    fig.colorbar(cax, ax=ax, label="Power (%)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=9)
    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("Sample Size T", fontsize=11)
    ax.set_title("Kupiec Test Power (%) — GARCH(1,1) H$_1$\n"
                 "Heatmap: T × Confidence Level",
                 fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color="black" if 20 < val < 80 else "white")

    plt.tight_layout()
    return _save(fig, "fig_q2_6_power_heatmap.png")


# ── Figure 7: Return histogram H0 vs H1 (representative path) ────────────────

def plot_return_diagnostics(gauss_params, garch_params, returns):
    """
    Overlay actual portfolio returns, Gaussian fit, and GARCH unconditional
    density to show why H0 is mis-specified.
    """
    arr = returns.values
    x   = np.linspace(arr.min() - 0.005, arr.max() + 0.005, 400)

    mu_g  = gauss_params["mu"]
    sig_g = gauss_params["sigma"]

    mu_gc  = garch_params["mu"]
    sig_gc = garch_params["sigma_uncond"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: return histogram
    ax = axes[0]
    ax.hist(arr, bins=100, density=True, color="steelblue", alpha=0.5,
            label="Observed returns")
    ax.plot(x, norm.pdf(x, mu_g, sig_g), "r-", lw=2,
            label=f"Gaussian H$_0$  (σ={sig_g:.4f})")
    ax.plot(x, norm.pdf(x, mu_gc, sig_gc), "g--", lw=2,
            label=f"GARCH LR density  (σ_LR={sig_gc:.4f})")
    ax.set_xlabel("Daily log return", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("EW Portfolio Return Histogram\nwith H$_0$ and H$_1$ Gaussian Fits",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(arr.min() - 0.005, arr.max() + 0.005)

    # Right: QQ plot of standardised residuals vs. Normal
    from scipy.stats import probplot
    ax2 = axes[1]
    standardised = (arr - mu_g) / sig_g
    (osm, osr), (slope, intercept, r) = probplot(standardised, dist="norm")
    ax2.scatter(osm, osr, s=4, color="steelblue", alpha=0.4, label="Data")
    ax2.plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5,
             label="Normal reference")
    ax2.set_xlabel("Theoretical quantiles", fontsize=10)
    ax2.set_ylabel("Sample quantiles", fontsize=10)
    ax2.set_title("Normal Q–Q Plot of Standardised Returns\n"
                  "(Departure confirms H$_0$ mis-specification)",
                  fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, "fig_q2_7_return_diagnostics.png")


# ── Master function ───────────────────────────────────────────────────────────

def run_visualisations(power_results, df_power_T, df_persist,
                       gauss_params=None, garch_params=None, returns=None):
    """Generate all Q2 figures."""
    from logger import log_start, log_end
    log_start(logger, "q2_visualisations.py")

    plot_lr_distributions(power_results)
    plot_violation_distributions(power_results)
    plot_power_vs_T(df_power_T)
    plot_power_vs_persistence(df_persist)
    plot_size_vs_T(df_power_T)
    plot_power_heatmap(df_power_T)

    if gauss_params is not None and garch_params is not None and returns is not None:
        plot_return_diagnostics(gauss_params, garch_params, returns)

    log_end(logger, "q2_visualisations.py")


if __name__ == "__main__":
    from logger import setup_run_logger
    setup_run_logger("q2_visualisations")
    from q2_fit_models import run_fit_models
    from q2_kupiec_power import run_kupiec_power
    from q2_power_vs_T import run_power_vs_T
    from q2_power_vs_persistence import run_power_vs_persistence

    gauss_params, garch_params, returns = run_fit_models()
    power_results = run_kupiec_power(gauss_params, garch_params)
    df_T          = run_power_vs_T(gauss_params, garch_params)
    df_persist    = run_power_vs_persistence(gauss_params, garch_params)
    run_visualisations(power_results, df_T, df_persist,
                       gauss_params, garch_params, returns)
