"""
Q6 — Portfolio Definition and Initial Valuation
================================================
Defines the four options positions and computes initial market values,
Greeks, and option prices as of the reference date (February 12th, 2026).

Portfolio specification
-----------------------
  INTC : Short  3 call options | K =  90 % of spot | TTM = 9 months
  JPM  : Long   6 put  options | K = 100 % of spot | TTM = 6 months  (ATM)
  AA   : Long   6 call options | K = 105 % of spot | TTM = 12 months
  PG   : Short  2 put  options | K = 110 % of spot | TTM = 9 months

Conventions
-----------
  quantity    : signed number of CONTRACTS
                positive = long, negative = short
  multiplier  : 100 shares per standard US equity option contract
  strike_pct  : K as a fraction of the spot price at the reference date
  ttm         : time to expiry in years (months / 12)

Sign convention for P&L
-----------------------
  Per-contract unit P&L = multiplier × (V_T − V_0)   [positive = gain]
  Position P&L_i        = quantity_i × unit P&L_i
  Portfolio P&L         = Σ Position P&L_i
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from q6_option_pricing import (
    bs_price, bs_delta, bs_gamma, bs_vega, bs_theta,
)

# ── Position definitions ──────────────────────────────────────────────────────
# These are fixed at the time of portfolio construction (Feb 12, 2026).
# Strike prices are expressed as a percentage of the spot price and fixed once
# spot prices are known; see build_portfolio() below.

POSITION_SPECS: dict[str, dict] = {
    "INTC": {
        "option_type": "call",
        "quantity":    -3,          # SHORT 3 contracts
        "strike_pct":  0.90,        # ITM call  (K = 90 % × S)
        "ttm_months":  9,
        "ttm":         9 / 12,      # 0.75 years
        "multiplier":  config.Q6_MULTIPLIER,
        "description": "Short 3 ITM call options (K = 90% × spot)",
    },
    "JPM": {
        "option_type": "put",
        "quantity":    +6,          # LONG 6 contracts
        "strike_pct":  1.00,        # ATM put
        "ttm_months":  6,
        "ttm":         6 / 12,      # 0.50 years
        "multiplier":  config.Q6_MULTIPLIER,
        "description": "Long 6 ATM put options (K = 100% × spot)",
    },
    "AA": {
        "option_type": "call",
        "quantity":    +6,          # LONG 6 contracts
        "strike_pct":  1.05,        # OTM call  (K = 105 % × S)
        "ttm_months":  12,
        "ttm":         12 / 12,     # 1.00 year
        "multiplier":  config.Q6_MULTIPLIER,
        "description": "Long 6 OTM call options (K = 105% × spot)",
    },
    "PG": {
        "option_type": "put",
        "quantity":    -2,          # SHORT 2 contracts
        "strike_pct":  1.10,        # ITM put   (K = 110 % × S)
        "ttm_months":  9,
        "ttm":         9 / 12,      # 0.75 years
        "multiplier":  config.Q6_MULTIPLIER,
        "description": "Short 2 ITM put options (K = 110% × spot)",
    },
}

TICKERS = list(POSITION_SPECS.keys())   # ["INTC", "JPM", "AA", "PG"]


# ── Portfolio construction ────────────────────────────────────────────────────

def build_portfolio(
    spot_prices: pd.Series,
    hist_vols:   pd.Series,
    r:           float = config.Q6_RISK_FREE_RATE,
) -> list[dict]:
    """
    Compute initial option prices, Greeks, and notional values for each leg.

    Parameters
    ----------
    spot_prices : pd.Series   closing prices as of the reference date
    hist_vols   : pd.Series   annualised historical volatility per ticker
    r           : float       annualised risk-free rate

    Returns
    -------
    list of dicts, one per ticker, with fields:
      ticker, option_type, quantity, strike, ttm, spot, vol, mult,
      price_per_share, price_per_contract, position_value,
      delta, gamma, vega, theta, moneyness
    """
    portfolio = []
    for ticker in TICKERS:
        spec  = POSITION_SPECS[ticker]
        S     = float(spot_prices[ticker])
        sigma = float(hist_vols[ticker])
        K     = round(S * spec["strike_pct"], 4)
        T     = spec["ttm"]
        q     = spec["quantity"]
        mult  = spec["multiplier"]
        ot    = spec["option_type"]

        # Option price (per share)
        v0 = bs_price(S, K, r, T, sigma, ot)

        # Greeks
        delta = bs_delta(S, K, r, T, sigma, ot)
        gamma = bs_gamma(S, K, r, T, sigma)
        vega  = bs_vega(S, K, r, T, sigma)
        theta = bs_theta(S, K, r, T, sigma, ot)

        # Moneyness: S/K for calls, K/S for puts (> 1 = in-the-money)
        moneyness = S / K if ot == "call" else K / S

        portfolio.append({
            "ticker":               ticker,
            "description":          spec["description"],
            "option_type":          ot,
            "quantity":             q,
            "strike":               K,
            "strike_pct":           spec["strike_pct"],
            "ttm":                  T,
            "ttm_months":           spec["ttm_months"],
            "spot":                 S,
            "vol":                  sigma,
            "multiplier":           mult,
            "price_per_share":      v0,
            "price_per_contract":   v0 * mult,
            "position_value":       q * v0 * mult,    # signed (negative = short liability)
            "delta":                delta,
            "gamma":                gamma,
            "vega":                 vega,
            "theta":                theta,
            "moneyness":            moneyness,
        })

    return portfolio


def portfolio_to_df(portfolio: list[dict]) -> pd.DataFrame:
    """Convert a portfolio list (from build_portfolio) to a tidy DataFrame."""
    return pd.DataFrame(portfolio).set_index("ticker")


def get_quantities(portfolio: list[dict]) -> np.ndarray:
    """Return signed contract quantities as a (4,) array in TICKERS order."""
    return np.array([leg["quantity"] for leg in portfolio], dtype=float)


def get_initial_values(portfolio: list[dict]) -> np.ndarray:
    """Return per-share initial option prices as a (4,) array in TICKERS order."""
    return np.array([leg["price_per_share"] for leg in portfolio], dtype=float)
