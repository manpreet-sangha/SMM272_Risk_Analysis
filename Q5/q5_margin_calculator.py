"""
Q5 — Margin Calculator
=======================
Runs no-netting and SPAN-netting margin calculations for all four
position pairs and produces comparison DataFrames.

Main public API
---------------
run_all_margins()         -> pd.DataFrame   main comparison table
decompose_margin(pair)    -> dict            scanning / SOM / NOV breakdown
sensitivity_to_psr(pair)  -> pd.DataFrame   SPAN margin vs PSR grid
"""

import numpy as np
import pandas as pd

from q5_market_data import (
    F, RISK_FREE_RATE as r, T, CONTRACT_SIZE, PSR, VSR, SOM_RATE,
)
from q5_positions import ALL_PAIRS, single_positions
from q5_span_engine import span_margin, no_netting_margin
from q5_span_scenarios import build_scenario_table


# ── Main comparison table ─────────────────────────────────────────────────────

def run_all_margins() -> pd.DataFrame:
    """
    Compute no-netting and SPAN-netting margins for all 4 pairs.

    Returns
    -------
    pd.DataFrame with columns:
        pair_name       str
        leg1_margin     float  no-net margin for leg 1 alone
        leg2_margin     float  no-net margin for leg 2 alone
        no_net_total    float  sum of per-leg standalone margins
        span_net_margin float  portfolio SPAN margin (with netting)
        netting_benefit float  no_net_total - span_net_margin
        netting_pct     float  netting_benefit / no_net_total × 100
    """
    rows = []
    for pair in ALL_PAIRS:
        legs          = pair["legs"]
        solo_lists    = single_positions(pair)

        r1            = span_margin(solo_lists[0])
        r2            = span_margin(solo_lists[1])
        nn            = no_netting_margin(legs)
        sn            = span_margin(legs)

        no_net_tot    = nn["total_margin"]
        span_net      = sn["span_margin"]
        benefit       = no_net_tot - span_net
        pct           = (benefit / no_net_tot * 100) if no_net_tot > 0 else 0.0

        rows.append({
            "pair_name":        pair["name"],
            "leg1_margin":      r1["span_margin"],
            "leg2_margin":      r2["span_margin"],
            "no_net_total":     no_net_tot,
            "span_net_margin":  span_net,
            "netting_benefit":  benefit,
            "netting_pct":      pct,
        })

    return pd.DataFrame(rows)


# ── Margin decomposition ──────────────────────────────────────────────────────

def decompose_margin(pair: dict) -> dict:
    """
    Return a detailed SPAN margin decomposition for a pair under both
    netting assumptions.

    Keys returned:
        pair_name, no_net, span_net
        — each sub-dict has: scanning_risk, som, active_risk, nov_credit,
          span_margin, worst_scenario
    """
    legs       = pair["legs"]
    nn_result  = no_netting_margin(legs)
    sn_result  = span_margin(legs)

    def _summarise(r):
        return {
            "scanning_risk": r["scanning_risk"],
            "som":           r["som"],
            "active_risk":   r["active_risk"],
            "nov_credit":    r["nov_credit"],
            "span_margin":   r["span_margin"],
            "worst_scenario": r["worst_scenario"],
        }

    no_net_summary = {
        "scanning_risk": nn_result["total_scanning"],
        "som":           nn_result["total_som"],
        "active_risk":   nn_result["total_scanning"] + max(0, nn_result["total_som"] - nn_result["total_scanning"]),
        "nov_credit":    nn_result["total_nov"],
        "span_margin":   nn_result["total_margin"],
        "worst_scenario": "multiple",
    }

    return {
        "pair_name": pair["name"],
        "no_net":    no_net_summary,
        "span_net":  _summarise(sn_result),
    }


def decomposition_df() -> pd.DataFrame:
    """
    Build a multi-row DataFrame with full decomposition for all 4 pairs
    under both no-netting and SPAN-netting.

    Columns: pair_name, method, scanning_risk, som, active_risk,
             nov_credit, span_margin
    """
    rows = []
    for pair in ALL_PAIRS:
        d = decompose_margin(pair)
        for method, vals in [("No-Net", d["no_net"]), ("SPAN-Net", d["span_net"])]:
            rows.append({
                "pair_name":     d["pair_name"],
                "method":        method,
                "scanning_risk": vals["scanning_risk"],
                "som":           vals["som"],
                "active_risk":   vals["active_risk"],
                "nov_credit":    vals["nov_credit"],
                "span_margin":   vals["span_margin"],
            })
    return pd.DataFrame(rows)


# ── PSR sensitivity ───────────────────────────────────────────────────────────

def _span_margin_with_psr(positions: list, psr_val: float) -> float:
    """
    Compute SPAN margin for *positions* using a custom PSR value.
    All other market params (VSR, F, r, T, IV) are held constant.
    """
    import q5_market_data as _md
    from q5_option_pricing import black76_price
    from q5_span_scenarios import weighted_loss_array, instrument_pnl_array

    # Build custom scenario df for this psr_val
    scen_rows = []
    for i, (pf, vf, w) in enumerate(_md.SCENARIOS, start=1):
        scen_rows.append({
            "scenario_no": i,
            "label":       _md.SCENARIO_LABELS[i - 1],
            "price_frac":  pf,
            "vol_frac":    vf,
            "delta_F":     pf * psr_val,
            "delta_sigma": vf * _md.VSR,
            "weight":      w,
        })
    sdf = pd.DataFrame(scen_rows)

    # Portfolio P&L
    total_pnl = np.zeros(16)
    for pos in positions:
        total_pnl += instrument_pnl_array(pos, sdf)
    weights = sdf["weight"].to_numpy()
    wloss = -weights * total_pnl

    scan  = float(np.max(wloss))
    som   = sum(abs(p["quantity"]) * _md.SOM_RATE
                for p in positions if p["type"] in ("call", "put") and p["quantity"] < 0)
    active = max(scan, som)

    nov = sum(
        p["quantity"] * black76_price(_md.F, p["strike"], _md.RISK_FREE_RATE, _md.T, p["iv"], p["type"]) * _md.CONTRACT_SIZE
        for p in positions if p["type"] in ("call", "put") and p["quantity"] > 0
    )
    nov_credit = min(active, nov)
    return max(0.0, active - nov_credit)


def sensitivity_to_psr(pair: dict, psr_grid: np.ndarray = None) -> pd.DataFrame:
    """
    Compute the SPAN-netting margin for a pair across a range of PSR values.

    Parameters
    ----------
    pair     : one element of ALL_PAIRS
    psr_grid : 1-D array of PSR values to sweep (USD/lb); defaults to
               np.linspace(0.05, 0.60, 50)

    Returns
    -------
    pd.DataFrame  columns: psr_usdlb, psr_contract, span_net_margin, no_net_total
    """
    import q5_market_data as _md

    if psr_grid is None:
        psr_grid = np.linspace(0.05, 0.60, 50)

    legs = pair["legs"]
    rows_out = []
    for psr_val in psr_grid:
        span_m   = _span_margin_with_psr(legs, psr_val)
        nn_total = sum(_span_margin_with_psr([leg], psr_val) for leg in legs)
        rows_out.append({
            "psr_usdlb":       psr_val,
            "psr_contract":    psr_val * _md.CONTRACT_SIZE,
            "span_net_margin": span_m,
            "no_net_total":    nn_total,
        })

    return pd.DataFrame(rows_out)
