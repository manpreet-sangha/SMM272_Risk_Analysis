"""
Q5 — SPAN Margin Engine
=======================
Full CME SPAN margin calculation for a list of positions.

SPAN margin = max(0, active_risk - NOV_credit)

where:
    scanning_risk = max weighted loss across 16 scenarios
    active_risk   = max(scanning_risk, short_option_minimum)
    NOV_credit    = min(active_risk,  net_long_option_value)

No intra-commodity spread charge is applied (single contract month).
"""

import numpy as np

from q5_market_data import (
    F, RISK_FREE_RATE as r, T, CONTRACT_SIZE, PSR, VSR, SOM_RATE, SCENARIOS,
)
from q5_option_pricing import black76_price
from q5_span_scenarios import (
    build_scenario_table,
    portfolio_weighted_loss,
    instrument_pnl_array,
    weighted_loss_array,
)


# ── SPAN components ───────────────────────────────────────────────────────────

def scanning_risk(weighted_loss_arr: np.ndarray) -> float:
    """
    Scanning risk = maximum of the 16 weighted-loss values.

    A positive result means the worst scenario produce a net loss.
    Result is in USD (per-portfolio, per set of contracts analysed).
    """
    return float(np.max(weighted_loss_arr))


def short_option_minimum(positions: list) -> float:
    """
    Short Option Minimum (SOM) charge.

    For each net-short option leg (quantity < 0), apply SOM_RATE per contract.
    SOM = Σ |qty| × SOM_RATE  (only summed over short options).

    Futures legs are ignored.
    """
    total = 0.0
    for pos in positions:
        if pos["type"] in ("call", "put") and pos["quantity"] < 0:
            total += abs(pos["quantity"]) * SOM_RATE
    return total


def net_option_value(positions: list) -> float:
    """
    Net Option Value (NOV) — current market value of all LONG option positions.

    Only long options (quantity > 0) contribute positive value.
    Short options owe premium (negative), but SPAN only credits long value;
    short option liability does not create additional margin reduction.

    Returns the sum in USD (positive if long options have value).
    """
    total = 0.0
    for pos in positions:
        if pos["type"] in ("call", "put") and pos["quantity"] > 0:
            price_lb = black76_price(
                F,
                pos["strike"],
                r,
                T,
                pos["iv"],
                pos["type"],
            )
            total += pos["quantity"] * price_lb * CONTRACT_SIZE
    return total


# ── Full SPAN margin ──────────────────────────────────────────────────────────

def span_margin(positions: list) -> dict:
    """
    Compute SPAN margin for a portfolio of positions (NETTING applied).

    The portfolio P&L is computed as the sum of all legs for every scenario
    before the worst-loss scenario is identified.

    Returns
    -------
    dict with keys:
        scanning_risk   float  — maximum weighted scenario loss (USD)
        som             float  — short-option minimum (USD)
        active_risk     float  — max(scanning_risk, som)
        nov_credit      float  — min(active_risk, net_long_option_value)
        span_margin     float  — max(0, active_risk - nov_credit)
        worst_scenario  int    — 1-based index of worst SPAN scenario
        pnl_array       ndarray shape (16,) — unweighted portfolio P&L
        wloss_array     ndarray shape (16,) — weighted loss per scenario
    """
    scenario_df  = build_scenario_table()

    # Combined portfolio P&L and weighted losses
    wloss = portfolio_weighted_loss(positions, scenario_df)

    # Reconstruct unweighted pnl for reference
    pnl_total = np.zeros(16)
    for pos in positions:
        pnl_total += instrument_pnl_array(pos, scenario_df)

    scan_risk   = scanning_risk(wloss)
    som         = short_option_minimum(positions)
    active      = max(scan_risk, som)
    nov         = net_option_value(positions)
    nov_credit  = min(active, nov)
    margin      = max(0.0, active - nov_credit)
    worst_idx   = int(np.argmax(wloss)) + 1  # 1-based

    return {
        "scanning_risk":  scan_risk,
        "som":            som,
        "active_risk":    active,
        "nov_credit":     nov_credit,
        "span_margin":    margin,
        "worst_scenario": worst_idx,
        "pnl_array":      pnl_total,
        "wloss_array":    wloss,
    }


# ── No-netting margin ─────────────────────────────────────────────────────────

def no_netting_margin(positions: list) -> dict:
    """
    Compute the additive (no-netting) margin for a list of positions.

    Each leg is treated as a standalone portfolio; SPAN is computed
    independently for each leg, and the results are summed.

    Returns
    -------
    dict with keys:
        per_leg         list of dicts — span_margin() result for each leg
        total_margin    float — sum of per-leg span_margin values
        total_scanning  float — sum of per-leg scanning_risk values
        total_som       float — sum of per-leg som values
        total_nov       float — sum of per-leg nov_credit values
    """
    per_leg = [span_margin([pos]) for pos in positions]

    return {
        "per_leg":       per_leg,
        "total_margin":  sum(r["span_margin"]   for r in per_leg),
        "total_scanning": sum(r["scanning_risk"] for r in per_leg),
        "total_som":     sum(r["som"]            for r in per_leg),
        "total_nov":     sum(r["nov_credit"]     for r in per_leg),
    }
