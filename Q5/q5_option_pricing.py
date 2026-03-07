"""
Q5 — Black-76 Option Pricing Model for Commodity Futures Options
================================================================
Implements the Black (1976) model for pricing European calls and puts
on futures contracts, plus the key Greeks (delta, gamma, vega, theta).

Black-76 formula
----------------
  C = e^{-rT} [ F·N(d1) - K·N(d2) ]
  P = e^{-rT} [ K·N(-d2) - F·N(-d1) ]

  d1 = [ln(F/K) + 0.5·σ²·T] / (σ·√T)
  d2 = d1 - σ·√T

Returns are in USD per pound (per unit of underlying).
Multiply by CONTRACT_SIZE to get per-contract values.

Notes on American-style approximation
--------------------------------------
COMEX Copper options are American-style (early exercise permitted).
For options with significant time to expiry (> ~20 days) that are ATM or OTM,
the early exercise premium is negligible for call options on non-dividend-paying
futures (the optimal early exercise boundary for calls on futures is well above
current price for typical parameters).  For put options the early exercise
premium is also small when interest rates are low relative to option time value.
Black-76 is therefore used as an accurate analytical approximation consistent
with CME's own SPAN scenario re-pricing methodology (which also uses a
European analytical model for scenario generation).
"""

import warnings
import numpy as np
from scipy.stats import norm


# ── Core Black-76 functions ───────────────────────────────────────────────────

def _d1d2(F: float, K: float, T: float, sigma: float):
    """Compute d1 and d2 for Black-76."""
    if T <= 0:
        return np.inf, np.inf
    if sigma <= 0:
        return np.inf, np.inf
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black76_price(F: float, K: float, r: float, T: float,
                  sigma: float, option_type: str) -> float:
    """
    Black-76 option price in USD per pound.

    Parameters
    ----------
    F           : futures price (USD/lb)
    K           : strike price  (USD/lb)
    r           : risk-free rate (decimal p.a.)
    T           : time to expiry (years)
    sigma       : implied volatility (decimal, e.g. 0.21 for 21 %)
    option_type : 'call' or 'put'

    Returns
    -------
    float : option price in USD per pound
    """
    if T <= 0:
        # At expiry — return intrinsic value
        intrinsic = max(F - K, 0.0) if option_type == "call" else max(K - F, 0.0)
        return intrinsic

    d1, d2 = _d1d2(F, K, T, sigma)
    discount = np.exp(-r * T)

    if option_type == "call":
        price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return float(max(price, 0.0))


def black76_delta(F: float, K: float, r: float, T: float,
                  sigma: float, option_type: str) -> float:
    """
    Black-76 delta: ∂V/∂F.

    Call delta ∈ (0, e^{-rT}).
    Put  delta ∈ (-e^{-rT}, 0).
    """
    if T <= 0:
        if option_type == "call":
            return np.exp(-r * T) if F > K else 0.0
        else:
            return -np.exp(-r * T) if F < K else 0.0

    d1, _ = _d1d2(F, K, T, sigma)
    discount = np.exp(-r * T)

    if option_type == "call":
        return float(discount * norm.cdf(d1))
    else:
        return float(-discount * norm.cdf(-d1))


def black76_gamma(F: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Black-76 gamma: ∂²V/∂F² (same for calls and puts).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    discount = np.exp(-r * T)
    return float(discount * norm.pdf(d1) / (F * sigma * np.sqrt(T)))


def black76_vega(F: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Black-76 vega: ∂V/∂σ (per unit change in σ, i.e. per 100% vol change).
    To get per 1 vol-point (per 0.01 change in σ), divide by 100.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    discount = np.exp(-r * T)
    return float(discount * F * norm.pdf(d1) * np.sqrt(T))


def black76_theta(F: float, K: float, r: float, T: float,
                  sigma: float, option_type: str) -> float:
    """
    Black-76 theta: ∂V/∂T (per year; divide by 365 for per-calendar-day).
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(F, K, T, sigma)
    discount = np.exp(-r * T)
    common = -discount * F * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

    if option_type == "call":
        return float(common + r * discount * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    else:
        return float(common + r * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


# ── Convenience wrappers ──────────────────────────────────────────────────────

def contract_value(price_per_lb: float, contract_size: int) -> float:
    """Convert a per-pound option price to a per-contract USD value."""
    return price_per_lb * contract_size


def scenario_pnl_per_lb(F: float, K: float, r: float, T: float,
                         sigma_base: float, option_type: str,
                         delta_F: float, delta_sigma: float) -> float:
    """
    Compute the P&L per pound for one SPAN scenario.

    P&L = repriced_value - current_value
    (positive = gain; SPAN later negates to get loss)

    Parameters
    ----------
    delta_F     : absolute price move = price_fraction × PSR
    delta_sigma : absolute vol move   = vol_fraction   × VSR
    """
    current = black76_price(F, K, r, T, sigma_base, option_type)
    scenario = black76_price(F + delta_F, K, r, T, sigma_base + delta_sigma, option_type)
    return scenario - current


def futures_scenario_pnl(delta_F: float) -> float:
    """
    P&L per pound for a long futures position when price moves by delta_F.
    (For short futures, multiply by -1 at the calling level.)
    """
    return delta_F


# ── Greeks summary helper ─────────────────────────────────────────────────────

def option_greeks(F: float, K: float, r: float, T: float,
                  sigma: float, option_type: str) -> dict:
    """Return a dict of all Black-76 greeks and the current price."""
    price  = black76_price(F, K, r, T, sigma, option_type)
    delta  = black76_delta(F, K, r, T, sigma, option_type)
    gamma  = black76_gamma(F, K, r, T, sigma)
    vega   = black76_vega(F, K, r, T, sigma)
    theta  = black76_theta(F, K, r, T, sigma, option_type)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
    }
