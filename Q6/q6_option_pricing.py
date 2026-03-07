"""
Q6 — Black-Scholes Option Pricing for Equity Options
=====================================================
Standard Black-Scholes-Merton model for European calls and puts on
non-dividend-paying stocks.

Model formulas
--------------
  C = S·N(d₁) - K·e^{-rT}·N(d₂)
  P = K·e^{-rT}·N(-d₂) - S·N(-d₁)

  d₁ = [ln(S/K) + (r + ½σ²)·T] / (σ·√T)
  d₂ = d₁ - σ·√T

All functions accept both scalar and NumPy array inputs for S (the
underlying stock price), enabling fully vectorised repricing across all
Monte Carlo simulation paths in a single call.
"""

import numpy as np
from scipy.stats import norm


# ── Core helpers ──────────────────────────────────────────────────────────────

def _d1d2(
    S: np.ndarray | float,
    K: float,
    r: float,
    T: float,
    sigma: float,
) -> tuple:
    """Compute d₁ and d₂ (scalar or array S accepted)."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ── Pricing ───────────────────────────────────────────────────────────────────

def bs_price(
    S: np.ndarray | float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str,
) -> np.ndarray | float:
    """
    Black-Scholes option price (scalar or vectorised over S).

    Parameters
    ----------
    S           : current stock price(s) — scalar or (N,) array
    K           : strike price
    r           : continuously compounded risk-free rate (p.a.)
    T           : time to expiry (years); clamped to ≥ 1e-6
    sigma       : annualised volatility (decimal)
    option_type : 'call' or 'put'

    Returns
    -------
    Price(s) in the same units as S and K.
    """
    T = max(float(T), 1e-6)
    S = np.asarray(S, dtype=float)
    scalar_in = S.ndim == 0
    S = np.atleast_1d(S)

    d1, d2 = _d1d2(S, K, r, T, sigma)
    disc = np.exp(-r * T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif option_type == "put":
        price = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return float(price[0]) if scalar_in else price


def bs_price_intrinsic(
    S: np.ndarray | float,
    K: float,
    option_type: str,
) -> np.ndarray | float:
    """Intrinsic (at-expiry) value of an option."""
    S = np.asarray(S, dtype=float)
    scalar_in = S.ndim == 0
    S = np.atleast_1d(S)

    if option_type == "call":
        val = np.maximum(S - K, 0.0)
    else:
        val = np.maximum(K - S, 0.0)

    return float(val[0]) if scalar_in else val


# ── Greeks ────────────────────────────────────────────────────────────────────

def bs_delta(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str,
) -> float:
    """Black-Scholes delta — scalar only."""
    T = max(float(T), 1e-6)
    d1, _ = _d1d2(float(S), K, r, T, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """Black-Scholes gamma (identical for calls and puts) — scalar only."""
    T = max(float(T), 1e-6)
    d1, _ = _d1d2(float(S), K, r, T, sigma)
    return float(norm.pdf(d1) / (float(S) * sigma * np.sqrt(T)))


def bs_vega(S: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Black-Scholes vega — scalar only.

    Returns the change in option price for a 1-unit (100 pp) increase in
    volatility.  Divide by 100 to get the per-percentage-point sensitivity.
    """
    T = max(float(T), 1e-6)
    d1, _ = _d1d2(float(S), K, r, T, sigma)
    return float(float(S) * norm.pdf(d1) * np.sqrt(T))


def bs_theta(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black-Scholes theta (per calendar day) — scalar only.

    Returns the change in option price for one day of time decay
    (i.e., ∂V/∂t × (1/365)).
    """
    T = max(float(T), 1e-6)
    S = float(S)
    d1, d2 = _d1d2(S, K, r, T, sigma)
    disc = np.exp(-r * T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == "call":
        theta_pa = term1 - r * K * disc * norm.cdf(d2)
    else:
        theta_pa = term1 + r * K * disc * norm.cdf(-d2)

    return float(theta_pa / 365.0)   # per calendar day
