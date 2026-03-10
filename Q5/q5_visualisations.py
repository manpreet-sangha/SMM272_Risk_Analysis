"""
Q5 — Visualisations
====================
Six publication-quality figures saved to Q5_OUTPUT_DIR.

Figure  File                              Description
------  --------------------------------  ----------------------------------
1       fig_q5_1_risk_array_heatmap.png   16-scenario P&L heatmap (cell annotated)
2       fig_q5_2_scenario_profiles.png    Payoff vs price shift line chart
3       fig_q5_3_margin_comparison.png    Grouped bar: no-net vs SPAN margin
4       fig_q5_4_netting_benefit.png      Netting benefit (abs + %) ranked
5       fig_q5_5_option_payoffs.png       At-expiry P&L diagrams for 4 pairs
6       fig_q5_6_span_sensitivity.png     SPAN margin vs PSR for all 4 pairs
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from q5_market_data import (
    F, ATM_STRIKE, CONTRACT_SIZE,
    IV_CALL_ATM, IV_PUT_ATM,
    PSR, SCENARIO_LABELS,
)
from q5_positions import ALL_PAIRS
from q5_span_scenarios import (
    build_scenario_table,
    scenario_breakdown_df,
    instrument_pnl_array,
    weighted_loss_array,
)
from q5_span_engine import span_margin
from q5_margin_calculator import (
    run_all_margins,
    sensitivity_to_psr,
)
from q5_option_pricing import black76_price
import q5_market_data as _md

# ── Style ─────────────────────────────────────────────────────────────────────

COLOURS = {
    "no_net":  "#2c7bb6",
    "span":    "#d7191c",
    "benefit": "#1a9641",
    "futures": "#756bb1",
    "call":    "#2ca02c",
    "put":     "#d62728",
    "neutral": "#636363",
}

PAIR_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _save(fig, filename: str) -> str:
    path = os.path.join(config.Q5_OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 1 — Risk Array Heatmap ────────────────────────────────────────────

def fig_q5_1_risk_array_heatmap():
    """
    Heatmap: 16 SPAN scenarios (rows) × position legs + portfolio (cols).
    Cell colour encodes P&L (green=gain, red=loss).
    """
    scenario_df = build_scenario_table()
    pair = ALL_PAIRS[3]          # P4: Long Futures + Long Put (most instructive)
    legs = pair["legs"]

    df = scenario_breakdown_df(legs, scenario_df)

    leg_cols     = [pos.get("label", pos["type"]) for pos in legs]
    display_cols = leg_cols + ["portfolio_pnl"]

    matrix = df[display_cols].to_numpy(dtype=float)
    labels = df["label"].tolist()

    # Build a 4-pair combined matrix: show all pairs' portfolio P&L
    all_pf = {}
    for p in ALL_PAIRS:
        bd = scenario_breakdown_df(p["legs"], scenario_df)
        all_pf[p["short"]] = bd["portfolio_pnl"].to_numpy()

    pf_matrix = np.column_stack([all_pf[p["short"]] for p in ALL_PAIRS])
    col_names  = [p["short"] for p in ALL_PAIRS]

    vmax = np.abs(pf_matrix).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pf_matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, fontsize=8, fontweight="bold")
    ax.set_yticks(range(16))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Position Pair", fontsize=8)
    ax.set_ylabel("SPAN Scenario", fontsize=8)

    # Annotate cells
    for r in range(16):
        for c in range(len(col_names)):
            val = pf_matrix[r, c]
            ax.text(c, r, f"${val:,.0f}", ha="center", va="center",
                    fontsize=5.5,
                    color="black" if abs(val) < 0.6 * vmax else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Portfolio P&L (USD)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()
    path = _save(fig, "fig_q5_1_risk_array_heatmap.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 2 — Scenario P&L Profiles ─────────────────────────────────────────

def fig_q5_2_scenario_profiles():
    """
    Line charts: P&L vs price move (ΔF) for the 8 symmetric price scenarios
    holding vol unchanged (scenarios 1-8 + baseline).
    Shows convexity difference between call, put, futures.
    """
    scenario_df = build_scenario_table()

    # Only scenarios with vol_frac=0 and the extreme tails (weight=0.35 at vol=0)
    price_only = scenario_df[
        (scenario_df["vol_frac"] == 0)
    ].sort_values("delta_F").reset_index(drop=True)

    delta_F_vals = price_only["delta_F"].tolist()

    from q5_option_pricing import scenario_pnl_per_lb, futures_scenario_pnl

    instruments = [
        ("Long Call",    "call",    +1, ATM_STRIKE, IV_CALL_ATM, COLOURS["call"]),
        ("Long Put",     "put",     +1, ATM_STRIKE, IV_PUT_ATM,  COLOURS["put"]),
        ("Short Call",   "call",    -1, ATM_STRIKE, IV_CALL_ATM, "#74c476"),
        ("Short Put",    "put",     -1, ATM_STRIKE, IV_PUT_ATM,  "#fc8d59"),
        ("Long Futures", "futures", +1, None,        None,        COLOURS["futures"]),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for lbl, itype, qty, K, iv, col in instruments:
        pnl_vals = []
        for dF in delta_F_vals:
            if itype == "futures":
                pnl_lb = futures_scenario_pnl(dF)
            else:
                pnl_lb = scenario_pnl_per_lb(_md.F, K, _md.RISK_FREE_RATE, _md.T, iv, itype, dF, 0.0)
            pnl_vals.append(qty * pnl_lb * CONTRACT_SIZE)

        ax.plot(delta_F_vals, pnl_vals, marker="o", linewidth=1.5, label=lbl,
                color=col, markersize=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Price Move ΔF (USD/lb)", fontsize=8)
    ax.set_ylabel("P&L per contract (USD)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = _save(fig, "fig_q5_2_scenario_profiles.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 3 — Margin Comparison ─────────────────────────────────────────────

def fig_q5_3_margin_comparison():
    """
    Grouped bar chart: no-netting total vs SPAN-netting margin for 4 pairs.
    """
    df = run_all_margins()

    x      = np.arange(len(df))
    width  = 0.35
    labels = df["pair_name"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, df["no_net_total"],    width,
                color=COLOURS["no_net"], label="No-Netting (Additive)", alpha=0.85)
    b2 = ax.bar(x + width / 2, df["span_net_margin"], width,
                color=COLOURS["span"],   label="SPAN Netting",           alpha=0.85)

    # Value labels
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7.5)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split(":")[0] for lbl in labels], fontsize=10)
    ax.set_ylabel("Initial Margin  (USD per portfolio)", fontsize=10)
    ax.set_title(
        "Figure 3 — SPAN Margin: No-Netting vs Portfolio Netting\n"
        "COMEX Copper HG May 2026, 1 contract per leg", fontsize=11
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Pair names as secondary x-labels
    ax2 = ax.secondary_xaxis("bottom")
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl.split(": ")[1] if ": " in lbl else lbl for lbl in labels],
                        fontsize=8, color="dimgray")
    ax2.tick_params(length=0, pad=18)

    fig.tight_layout()
    path = _save(fig, "fig_q5_3_margin_comparison.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 4 — Netting Benefit ────────────────────────────────────────────────

def fig_q5_4_netting_benefit():
    """
    Side-by-side: absolute netting benefit (USD) and percentage benefit,
    pairs sorted by % benefit descending.
    """
    df = run_all_margins().sort_values("netting_pct", ascending=False).reset_index(drop=True)
    labels = [row["pair_name"].split(":")[0] for _, row in df.iterrows()]
    subtitles = [row["pair_name"].split(": ")[1] if ": " in row["pair_name"] else "" for _, row in df.iterrows()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars1 = ax1.bar(labels, df["netting_benefit"], color=COLOURS["benefit"], alpha=0.85)
    ax1.set_ylabel("Netting Benefit  (USD)", fontsize=10)
    ax1.set_title("Absolute Netting Benefit", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, sub in zip(bars1, subtitles):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(labels, df["netting_pct"], color=COLOURS["benefit"], alpha=0.70)
    ax2.set_ylabel("Netting Benefit  (%)", fontsize=10)
    ax2.set_title("Percentage Netting Benefit", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.grid(True, axis="y", alpha=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xlabel("Position Pair", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Figure 4 — SPAN Netting Benefits by Position Pair\n"
        "Sorted by % benefit (highest first)", fontsize=11, y=1.02
    )
    fig.tight_layout()
    path = _save(fig, "fig_q5_4_netting_benefit.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 3+4 combined — Margin Comparison & Netting Benefit ─────────────────

def fig_q5_34_margin_netting():
    """
    Three-panel horizontal figure:
      Left   – grouped bar: no-netting vs SPAN margin
      Centre – absolute netting benefit (USD)
      Right  – percentage netting benefit
    """
    df_raw = run_all_margins()
    df_sorted = df_raw.sort_values("netting_pct", ascending=False).reset_index(drop=True)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 5.5))

    # ── Left: margin comparison (original pair order) ──
    x = np.arange(len(df_raw))
    w = 0.35
    labels_raw = [r["pair_name"].split(":")[0] for _, r in df_raw.iterrows()]

    b1 = ax0.bar(x - w / 2, df_raw["no_net_total"], w,
                 color=COLOURS["no_net"], label="No-Netting", alpha=0.85)
    b2 = ax0.bar(x + w / 2, df_raw["span_net_margin"], w,
                 color=COLOURS["span"], label="SPAN Netting", alpha=0.85)
    for bar in b1:
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels_raw, fontsize=9)
    ax0.set_ylabel("Initial Margin (USD)", fontsize=10)
    ax0.set_title("Margin: No-Netting vs SPAN", fontsize=10, fontweight="bold")
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax0.legend(fontsize=8)
    ax0.grid(True, axis="y", alpha=0.3)

    # ── Centre: absolute netting benefit (sorted) ──
    labels_s = [r["pair_name"].split(":")[0] for _, r in df_sorted.iterrows()]
    bars1 = ax1.bar(labels_s, df_sorted["netting_benefit"],
                    color=COLOURS["benefit"], alpha=0.85)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20,
                 f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_ylabel("Netting Benefit (USD)", fontsize=10)
    ax1.set_title("Absolute Benefit", fontsize=10, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Right: percentage netting benefit (sorted) ──
    bars2 = ax2.bar(labels_s, df_sorted["netting_pct"],
                    color=COLOURS["benefit"], alpha=0.70)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylabel("Netting Benefit (%)", fontsize=10)
    ax2.set_title("Percentage Benefit", fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(True, axis="y", alpha=0.3)

    for ax in (ax0, ax1, ax2):
        ax.set_xlabel("Position Pair", fontsize=9)

    fig.tight_layout()
    path = _save(fig, "fig_q5_34_margin_netting.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 5 — At-Expiry Payoff Diagrams ─────────────────────────────────────

def fig_q5_5_option_payoffs():
    """
    At-expiry P&L for all 4 pairs across a range of final copper prices.
    Computed analytically at T=0 (intrinsic value only).
    """
    K       = ATM_STRIKE
    F_range = np.linspace(3.80, 5.50, 300)   # USD/lb at expiry

    def call_payoff(F_T, strike, qty):
        return qty * max(F_T - strike, 0) * CONTRACT_SIZE

    def put_payoff(F_T, strike, qty):
        return qty * max(strike - F_T, 0) * CONTRACT_SIZE

    def futures_payoff(F_T, entry, qty):
        return qty * (F_T - entry) * CONTRACT_SIZE

    premiums = {
        "call_long":  +black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_CALL_ATM, "call") * CONTRACT_SIZE,
        "call_short": -black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_CALL_ATM, "call") * CONTRACT_SIZE,
        "put_long":   +black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_PUT_ATM,  "put")  * CONTRACT_SIZE,
        "put_short":  -black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_PUT_ATM,  "put")  * CONTRACT_SIZE,
    }

    pairs_payoff_fns = [
        # (name, fn_list)  — list of callables f(F_T) -> leg P&L at expiry
        ("P1: Long Call + Short Put",
         [lambda ft: call_payoff(ft, K, +1) - premiums["call_long"],
          lambda ft: put_payoff(ft, K, -1) + premiums["put_short"]]),
        ("P2: Short Call + Long Put",
         [lambda ft: call_payoff(ft, K, -1) + premiums["call_short"],
          lambda ft: put_payoff(ft, K, +1) - premiums["put_long"]]),
        ("P3: Long Call + Long Futures",
         [lambda ft: call_payoff(ft, K, +1) - premiums["call_long"],
          lambda ft: futures_payoff(ft, _md.F, +1)]),
        ("P4: Long Futures + Long Put",
         [lambda ft: futures_payoff(ft, _md.F, +1),
          lambda ft: put_payoff(ft, K, +1) - premiums["put_long"]]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat  = axes.flatten()

    for ax, (name, fns), col in zip(axes_flat, pairs_payoff_fns, PAIR_COLOURS):
        total = np.array([sum(f(ft) for f in fns) for ft in F_range])
        ax.plot(F_range, total, color=col, linewidth=2.0, label="Combined P&L")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(K, color="grey", linewidth=0.8, linestyle=":", label=f"K = {K}")
        ax.axvline(_md.F, color="orange", linewidth=0.8, linestyle=":", label=f"F₀ = {_md.F}")
        ax.fill_between(F_range, total, 0,
                        where=(total >= 0), alpha=0.12, color="green")
        ax.fill_between(F_range, total, 0,
                        where=(total < 0),  alpha=0.12, color="red")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylabel("P&L at expiry (USD)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for ax in axes[1]:
        ax.set_xlabel("Copper futures price at expiry  (USD/lb)", fontsize=9)

    fig.suptitle(
        "Figure 5 — At-Expiry P&L for 4 Position Pairs\n"
        "K = 4.65, F₀ = 4.6490, ATM implieds, 1 contract per leg", fontsize=11
    )
    fig.tight_layout()
    path = _save(fig, "fig_q5_5_option_payoffs.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 5 horizontal — At-Expiry Payoff Diagrams (1×4) ────────────────────

def fig_q5_5_option_payoffs_horizontal():
    """
    At-expiry P&L for all 4 pairs in a 2×2 grid. Compact, high-resolution.
    """
    K       = ATM_STRIKE
    F_range = np.linspace(3.80, 5.50, 300)

    def call_payoff(F_T, strike, qty):
        return qty * max(F_T - strike, 0) * CONTRACT_SIZE

    def put_payoff(F_T, strike, qty):
        return qty * max(strike - F_T, 0) * CONTRACT_SIZE

    def futures_payoff(F_T, entry, qty):
        return qty * (F_T - entry) * CONTRACT_SIZE

    premiums = {
        "call_long":  +black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_CALL_ATM, "call") * CONTRACT_SIZE,
        "call_short": -black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_CALL_ATM, "call") * CONTRACT_SIZE,
        "put_long":   +black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_PUT_ATM,  "put")  * CONTRACT_SIZE,
        "put_short":  -black76_price(_md.F, K, _md.RISK_FREE_RATE, _md.T, IV_PUT_ATM,  "put")  * CONTRACT_SIZE,
    }

    pairs_payoff_fns = [
        ("P1: Long Call + Short Put",
         [lambda ft: call_payoff(ft, K, +1) - premiums["call_long"],
          lambda ft: put_payoff(ft, K, -1) + premiums["put_short"]]),
        ("P2: Short Call + Long Put",
         [lambda ft: call_payoff(ft, K, -1) + premiums["call_short"],
          lambda ft: put_payoff(ft, K, +1) - premiums["put_long"]]),
        ("P3: Long Call + Long Futures",
         [lambda ft: call_payoff(ft, K, +1) - premiums["call_long"],
          lambda ft: futures_payoff(ft, _md.F, +1)]),
        ("P4: Long Futures + Long Put",
         [lambda ft: futures_payoff(ft, _md.F, +1),
          lambda ft: put_payoff(ft, K, +1) - premiums["put_long"]]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6, 5.5), sharey=True)
    axes = axes.ravel()

    for ax, (name, fns), col in zip(axes, pairs_payoff_fns, PAIR_COLOURS):
        total = np.array([sum(f(ft) for f in fns) for ft in F_range])
        ax.plot(F_range, total, color=col, linewidth=1.4, label="Combined P&L")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(K, color="grey", linewidth=0.7, linestyle=":", label=f"K = {K}")
        ax.axvline(_md.F, color="orange", linewidth=0.7, linestyle=":", label=f"F\u2080 = {_md.F}")
        ax.fill_between(F_range, total, 0,
                        where=(total >= 0), alpha=0.12, color="green")
        ax.fill_between(F_range, total, 0,
                        where=(total < 0),  alpha=0.12, color="red")
        ax.set_title(name, fontsize=7, fontweight="bold")
        ax.set_xlabel("Copper price at expiry (USD/lb)", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=5, framealpha=0.6)

    axes[0].set_ylabel("P&L at expiry (USD)", fontsize=6)
    axes[2].set_ylabel("P&L at expiry (USD)", fontsize=6)

    fig.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.5)
    path = _save(fig, "fig_q5_5_option_payoffs_horizontal.png")
    print(f"  Saved: {path}")
    return path


# ── Figure 6 — SPAN Sensitivity to PSR ───────────────────────────────────────

def fig_q5_6_span_sensitivity():
    """
    SPAN margin (portfolio-netting) vs PSR (price scanning range) for all 4 pairs.
    Compact square, high-resolution.
    """
    psr_grid = np.linspace(0.05, 0.60, 60)

    fig, ax = plt.subplots(figsize=(6, 6))

    for pair, col in zip(ALL_PAIRS, PAIR_COLOURS):
        df_sens = sensitivity_to_psr(pair, psr_grid)
        ax.plot(df_sens["psr_usdlb"], df_sens["span_net_margin"],
                linewidth=1.8, label=pair["short"], color=col)
        ax.plot(df_sens["psr_usdlb"], df_sens["no_net_total"],
                linewidth=0.9, linestyle="--", color=col, alpha=0.5)

    # Mark current PSR
    ax.axvline(PSR, color="black", linewidth=0.9, linestyle=":",
               label=f"Current PSR = {PSR:.2f}")

    ax.set_xlabel("Price Scanning Range  PSR (USD/lb)", fontsize=8)
    ax.set_ylabel("Margin  (USD)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = _save(fig, "fig_q5_6_span_sensitivity.png")
    print(f"  Saved: {path}")
    return path


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all_figures():
    print("Generating Q5 figures...")
    paths = []
    paths.append(fig_q5_1_risk_array_heatmap())
    paths.append(fig_q5_2_scenario_profiles())
    paths.append(fig_q5_3_margin_comparison())
    paths.append(fig_q5_4_netting_benefit())
    paths.append(fig_q5_34_margin_netting())
    paths.append(fig_q5_5_option_payoffs())
    paths.append(fig_q5_5_option_payoffs_horizontal())
    paths.append(fig_q5_6_span_sensitivity())
    print(f"All {len(paths)} figures saved.")
    return paths


if __name__ == "__main__":
    run_all_figures()
