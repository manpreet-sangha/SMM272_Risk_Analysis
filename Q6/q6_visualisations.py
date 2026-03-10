"""
Q6 — Visualisations
====================
Six publication-quality figures saved to Q6_OUTPUT_DIR.

Figure  File                                 Description
------  -----------------------------------  ---------------------------------
1       fig_q6_1_pnl_distribution.png        Portfolio P&L histogram + VaR/ES
2       fig_q6_2_risk_decomposition.png      Component & Marginal VaR/ES bars
3       fig_q6_3_correlation_heatmap.png     Stock correlation matrix
4       fig_q6_4_option_profiles.png         Option P&L vs stock price move
5       fig_q6_5_simulated_returns.png       Simulated 10-day return densities
6       fig_q6_6_instrument_pnl.png          Per-instrument P&L distributions
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from q6_option_pricing import bs_price
from q6_covariance import cov_to_corr

# ── Style constants ────────────────────────────────────────────────────────────
PALETTE = {
    "INTC": "#1f77b4",   # blue
    "JPM":  "#ff7f0e",   # orange
    "AA":   "#2ca02c",   # green
    "PG":   "#d62728",   # red
}
C_VAR  = "#d62728"
C_ES   = "#7b2d8b"
C_LONG = "#2ca02c"
C_SHORT= "#d62728"

TICKER_LABELS = {
    "INTC": "INTC (Short 3 Call)",
    "JPM":  "JPM  (Long  6 Put)",
    "AA":   "AA   (Long  6 Call)",
    "PG":   "PG   (Short 2 Put)",
}


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(config.Q6_OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 1 — Portfolio P&L Distribution ────────────────────────────────────

def fig_q6_1_pnl_distribution(
    total_pnl: np.ndarray,
    var: float,
    es: float,
    confidence: float = 0.99,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Histogram
    ax.hist(total_pnl, bins=120, color="#4393c3", alpha=0.72, edgecolor="none",
            density=True, label="Simulated P&L")

    # KDE overlay
    kde = gaussian_kde(total_pnl, bw_method="scott")
    x   = np.linspace(total_pnl.min(), total_pnl.max(), 600)
    ax.plot(x, kde(x), color="#1f4e79", linewidth=1.6, label="KDE")

    # VaR and ES lines
    ax.axvline(-var, color=C_VAR, linewidth=2.0, linestyle="--",
               label=f"VaR$_{{99}}$ = −${var:,.0f}")
    ax.axvline(-es,  color=C_ES,  linewidth=2.0, linestyle=":",
               label=f"ES$_{{99}}$   = −${es:,.0f}")

    # Shade tail region
    x_tail = np.linspace(total_pnl.min(), -var, 200)
    ax.fill_between(x_tail, 0, kde(x_tail), color=C_VAR, alpha=0.18,
                    label=f"Tail (worst {100*(1-confidence):.0f}%)")

    ax.set_xlabel("10-Day Portfolio P&L  (USD)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Portfolio P&L Distribution — 10,000 Monte Carlo Simulations\n"
                 "(10-day horizon, 99% confidence)", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, "fig_q6_1_pnl_distribution.png")


# ── Figure 2 — Risk Decomposition ────────────────────────────────────────────

def fig_q6_2_risk_decomposition(
    tickers:  list[str],
    comp_var: np.ndarray,
    comp_es:  np.ndarray,
    marg_var: np.ndarray,
    marg_es:  np.ndarray,
    var:      float,
    es:       float,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

    x     = np.arange(len(tickers))
    width = 0.36
    colors = [C_LONG if v >= 0 else C_SHORT for v in comp_var]

    # Left: Component VaR vs Component ES
    ax = axes[0]
    ax.bar(x - width/2, comp_var, width, color=colors,   alpha=0.85, label="Component VaR")
    ax.bar(x + width/2, comp_es,  width, color=[C_ES]*4, alpha=0.75, label="Component ES",
           edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(var, color=C_VAR, linewidth=1.0, linestyle="--", alpha=0.6,
               label=f"Portfolio VaR = ${var:,.0f}")
    ax.axhline(es,  color=C_ES,  linewidth=1.0, linestyle=":",  alpha=0.6,
               label=f"Portfolio ES  = ${es:,.0f}")
    ax.set_xticks(x)
    ax.set_xticklabels([TICKER_LABELS[t] for t in tickers], fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("USD", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Component VaR and ES (Euler allocation)", fontsize=8)
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Right: Marginal VaR vs Marginal ES
    ax2 = axes[1]
    m_colors = [C_LONG if v >= 0 else "#2d8b4f" for v in marg_var]
    ax2.bar(x - width/2, marg_var, width, color=m_colors, alpha=0.85,
            label="Marginal VaR (per contract)")
    ax2.bar(x + width/2, marg_es,  width, color=[C_ES]*4, alpha=0.75,
            label="Marginal ES  (per contract)")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([TICKER_LABELS[t] for t in tickers], fontsize=7.5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.set_ylabel("USD per contract", fontsize=7.5)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.set_title("Marginal VaR and ES (per additional contract)", fontsize=8)
    ax2.legend(fontsize=6.5, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=0.6)
    out = Path(config.Q6_OUTPUT_DIR) / "fig_q6_2_risk_decomposition.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ── Figure 3 — Correlation Heatmap ───────────────────────────────────────────

def fig_q6_3_correlation_heatmap(
    cov_daily:  pd.DataFrame,
    sample_cov: pd.DataFrame,
) -> str:
    corr_ewma = cov_to_corr(cov_daily)
    corr_hist = cov_to_corr(sample_cov)
    tickers   = list(cov_daily.columns)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))

    for ax, corr, title in zip(
        axes,
        [corr_ewma, corr_hist],
        ["EWMA Correlation (λ=0.94)", "Historical Correlation (2-yr)"],
    ):
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, fontsize=8)
        ax.set_yticklabels(tickers, fontsize=8)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black" if abs(val) < 0.7 else "white")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.80)

    fig.tight_layout(pad=0.6)
    path = os.path.join(config.Q6_OUTPUT_DIR, "fig_q6_3_correlation_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 4 — Option P&L Profiles ───────────────────────────────────────────

def fig_q6_4_option_profiles(
    portfolio: list[dict],
    r:         float,
) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    horizon_years = config.Q6_HORIZON_DAYS / config.TRADING_DAYS

    for j, (leg, ax) in enumerate(zip(portfolio, axes)):
        S0    = leg["spot"]
        K     = leg["strike"]
        T0    = leg["ttm"]
        T_rem = max(T0 - horizon_years, 1e-6)
        sigma = leg["vol"]
        q     = leg["quantity"]
        mult  = leg["multiplier"]
        ot    = leg["option_type"]
        v0    = leg["price_per_share"]

        # Range: ±30% around spot
        pct_moves = np.linspace(-0.30, 0.30, 300)
        S_range   = S0 * (1 + pct_moves)
        vT        = bs_price(S_range, K, r, T_rem, sigma, ot)
        pnl_pos   = q * mult * (vT - v0)   # signed position P&L

        color = C_SHORT if q < 0 else C_LONG
        ax.plot(pct_moves * 100, pnl_pos, color=color, linewidth=1.6, label="Position P&L")
        ax.axhline(0,  color="black",    linewidth=0.7, linestyle="-")
        ax.axvline(0,  color="grey",     linewidth=0.7, linestyle="--", alpha=0.6)
        ax.axvline((K/S0 - 1) * 100, color="black", linewidth=1.0,
                   linestyle=":", label=f"K=${K:.0f}")

        # Shade profit / loss regions
        ax.fill_between(pct_moves * 100, pnl_pos, 0,
                        where=(pnl_pos >= 0), alpha=0.18, color="green")
        ax.fill_between(pct_moves * 100, pnl_pos, 0,
                        where=(pnl_pos < 0),  alpha=0.18, color="red")

        ax.set_title(f"{leg['ticker']} — {leg['description']}", fontsize=7)
        ax.set_xlabel("Underlying Price Change (%)", fontsize=6.5)
        ax.set_ylabel("Position P&L (USD)" if j == 0 else "", fontsize=6.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, framealpha=0.9)
        ax.grid(alpha=0.3)

    fig.tight_layout(pad=0.6)
    path = os.path.join(config.Q6_OUTPUT_DIR, "fig_q6_4_option_profiles.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 5 — Simulated Stock Return Distributions ──────────────────────────

def fig_q6_5_simulated_returns(
    log_returns_sim: np.ndarray,
    portfolio:       list[dict],
    hist_log_returns: pd.DataFrame,
    horizon_days:    int = config.Q6_HORIZON_DAYS,
) -> str:
    tickers = [leg["ticker"] for leg in portfolio]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    for j, (ticker, ax) in enumerate(zip(tickers, axes)):
        sim_r = log_returns_sim[:, j]
        color = list(PALETTE.values())[j]

        ax.hist(sim_r, bins=80, density=True, color=color, alpha=0.55,
                label="Simulated")

        mu, std = np.mean(sim_r), np.std(sim_r)
        x_range = np.linspace(sim_r.min(), sim_r.max(), 300)
        ax.plot(x_range, norm.pdf(x_range, mu, std), color="black",
                linewidth=1.2, linestyle="--", label="Normal fit")

        kde = gaussian_kde(sim_r, bw_method="scott")
        ax.plot(x_range, kde(x_range), color=color, linewidth=1.4,
                label="KDE")

        ax.axvline(np.percentile(sim_r, 1), color=C_VAR, linewidth=1.2,
                   linestyle=":", label="VaR 1%")

        ax.set_title(TICKER_LABELS[ticker], fontsize=8)
        ax.set_xlabel("10-day log-return", fontsize=7)
        if j == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6, framealpha=0.9)
        ax.grid(alpha=0.3)

    fig.tight_layout(pad=0.6)
    out = Path(config.Q6_OUTPUT_DIR) / "fig_q6_5_simulated_returns.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ── Figure 6 — Per-Instrument P&L Distributions ───────────────────────────────

def fig_q6_6_instrument_pnl(
    inst_pnl:  np.ndarray,
    portfolio: list[dict],
    total_pnl: np.ndarray,
    var:       float,
) -> str:
    tickers   = [leg["ticker"] for leg in portfolio]
    n_assets  = len(tickers)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    tail_mask = total_pnl <= -var

    for j, (ticker, ax) in enumerate(zip(tickers, axes_flat)):
        pnl_j = inst_pnl[:, j]
        color = list(PALETTE.values())[j]

        ax.hist(pnl_j, bins=80, density=True, color=color, alpha=0.60,
                label="All simulations")

        # Highlight tail obs
        pnl_tail = pnl_j[tail_mask]
        if len(pnl_tail) > 5:
            ax.hist(pnl_tail, bins=40, density=True, color=C_VAR, alpha=0.50,
                    label="Portfolio tail scenarios")

        ax.axvline(0, color="black", linewidth=0.9, linestyle="-")
        ax.axvline(pnl_j.mean(), color=color, linewidth=1.5, linestyle="--",
                   label=f"Mean = ${pnl_j.mean():,.0f}")

        ax.set_title(f"{TICKER_LABELS[ticker]}", fontsize=9.5)
        ax.set_xlabel("Position P&L (USD)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-Instrument P&L Distributions\n"
                 "(Red shading = portfolio tail scenarios driving VaR/ES)",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, "fig_q6_6_instrument_pnl.png")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_all_figures(
    portfolio:        list[dict],
    total_pnl:        np.ndarray,
    inst_pnl:         np.ndarray,
    log_returns_sim:  np.ndarray,
    metrics:          dict,
    cov_daily:        pd.DataFrame,
    sample_cov_df:    pd.DataFrame,
    hist_log_returns: pd.DataFrame,
    r:                float = config.Q6_RISK_FREE_RATE,
) -> list[str]:
    """Generate all 6 figures and return list of saved paths."""
    paths = []
    tickers    = metrics["tickers"]
    quantities = metrics["quantities"]
    var        = metrics["var"]
    es         = metrics["es"]

    paths.append(fig_q6_1_pnl_distribution(total_pnl, var, es))
    paths.append(fig_q6_2_risk_decomposition(
        tickers, metrics["comp_var"], metrics["comp_es"],
        metrics["marg_var"], metrics["marg_es"], var, es,
    ))
    paths.append(fig_q6_3_correlation_heatmap(cov_daily, sample_cov_df))
    paths.append(fig_q6_4_option_profiles(portfolio, r))
    paths.append(fig_q6_5_simulated_returns(
        log_returns_sim, portfolio, hist_log_returns,
    ))
    paths.append(fig_q6_6_instrument_pnl(inst_pnl, portfolio, total_pnl, var))
    return paths
