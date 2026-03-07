"""
Q5 — SPAN Scenario Engine
=========================
Builds the CME SPAN 16-scenario risk array for any instrument or combined
portfolio.  A "position" is a plain dict with the following keys:

  {
    "type"    : "futures" | "call" | "put",
    "quantity": int,        # +1 = long, -1 = short (contracts)
    "strike"  : float,      # USD/lb; None or omitted for futures
    "iv"      : float,      # implied vol used for Black-76 pricing (decimal)
    "label"   : str,        # human-readable e.g. "Long Call K=4.65"
  }

All monetary values in USD (full contract, not per-pound) unless
named with the _per_lb suffix.
"""

import numpy as np
import pandas as pd

from q5_market_data import (
    F, CONTRACT_SIZE, RISK_FREE_RATE as r, T,
    PSR, VSR, SCENARIOS, SCENARIO_LABELS,
)
from q5_option_pricing import (
    black76_price,
    scenario_pnl_per_lb,
    futures_scenario_pnl,
)


# ── Scenario table builder ────────────────────────────────────────────────────

def build_scenario_table() -> pd.DataFrame:
    """
    Return a DataFrame with one row per SPAN scenario.

    Columns:
        scenario_no   : 1..16
        label         : human-readable description
        delta_F       : absolute price move (USD/lb) = price_frac × PSR
        delta_sigma   : absolute vol move (decimal)  = vol_frac   × VSR
        weight        : scenario weight (1.0 or 0.35 for extreme tails)
        price_frac    : fraction of PSR
        vol_frac      : fraction of VSR
    """
    rows = []
    for i, (pf, vf, w) in enumerate(SCENARIOS, start=1):
        rows.append({
            "scenario_no":  i,
            "label":        SCENARIO_LABELS[i - 1],
            "price_frac":   pf,
            "vol_frac":     vf,
            "delta_F":      pf * PSR,
            "delta_sigma":  vf * VSR,
            "weight":       w,
        })
    return pd.DataFrame(rows)


# ── Per-instrument P&L array ──────────────────────────────────────────────────

def instrument_pnl_array(position: dict, scenario_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the per-contract P&L (USD) for each of the 16 SPAN scenarios
    for a single position.

    Positive values indicate a gain; negative = loss.
    Weights are NOT applied here — call weighted_loss_array() afterwards.

    Parameters
    ----------
    position    : dict  (see module docstring)
    scenario_df : output of build_scenario_table()

    Returns
    -------
    np.ndarray  shape (16,)  — unweighted P&L in USD per contract
    """
    qty   = position["quantity"]
    ptype = position["type"]
    pnl   = np.zeros(16)

    for idx, row in scenario_df.iterrows():
        dF    = row["delta_F"]
        dsig  = row["delta_sigma"]
        s     = row["scenario_no"] - 1  # 0-based index

        if ptype == "futures":
            # P&L = quantity × ΔF × contract_size
            pnl[s] = qty * futures_scenario_pnl(dF) * CONTRACT_SIZE

        elif ptype in ("call", "put"):
            K     = position["strike"]
            sigma = position["iv"]
            pnl_lb = scenario_pnl_per_lb(
                F, K, r, T, sigma, ptype, dF, dsig
            )
            pnl[s] = qty * pnl_lb * CONTRACT_SIZE

        else:
            raise ValueError(f"Unknown position type: '{ptype}'")

    return pnl


def weighted_loss_array(pnl_array: np.ndarray, scenario_df: pd.DataFrame) -> np.ndarray:
    """
    Convert the unweighted P&L array into a weighted loss array.

    weighted_loss[s] = -weight[s] × pnl[s]

    Positive weighted_loss = loss from the portfolio's perspective.

    Returns
    -------
    np.ndarray shape (16,)
    """
    weights = scenario_df["weight"].to_numpy()
    return -weights * pnl_array


# ── Portfolio P&L array ───────────────────────────────────────────────────────

def portfolio_pnl_array(positions: list, scenario_df: pd.DataFrame) -> np.ndarray:
    """
    Sum the unweighted P&L arrays across all positions in the portfolio.

    Returns
    -------
    np.ndarray shape (16,)  — total unweighted portfolio P&L per scenario
    """
    total = np.zeros(16)
    for pos in positions:
        total += instrument_pnl_array(pos, scenario_df)
    return total


def portfolio_weighted_loss(positions: list, scenario_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the weighted loss array for the combined portfolio.

    Returns
    -------
    np.ndarray shape (16,)
    """
    pnl = portfolio_pnl_array(positions, scenario_df)
    return weighted_loss_array(pnl, scenario_df)


# ── Diagnostic DataFrame ─────────────────────────────────────────────────────

def scenario_breakdown_df(positions: list, scenario_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a diagnostic DataFrame with P&L per instrument and combined
    portfolio across all 16 scenarios.

    Columns: scenario_no, label, delta_F, delta_sigma, weight,
             <leg labels...>, portfolio_pnl, weighted_loss
    """
    df = scenario_df[["scenario_no", "label", "delta_F", "delta_sigma", "weight"]].copy()

    portfolio_pnl = np.zeros(16)

    for pos in positions:
        arr = instrument_pnl_array(pos, scenario_df)
        lbl = pos.get("label", pos["type"])
        df[lbl] = arr
        portfolio_pnl += arr

    df["portfolio_pnl"]   = portfolio_pnl
    weights               = scenario_df["weight"].to_numpy()
    df["weighted_loss"]   = -weights * portfolio_pnl

    return df
