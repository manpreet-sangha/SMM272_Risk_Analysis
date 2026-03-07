"""
Q6 — Covariance Matrix Estimation
===================================
Preferred approach: Exponentially Weighted Moving Average (EWMA)
with decay factor λ = 0.94 (RiskMetrics standard, JP Morgan 1994).

Why EWMA over the equal-weighted sample covariance for short-horizon VaR?
  1. Recent observations are given more weight, capturing current market
     volatility regimes more accurately than a 2-year equal-weight average.
  2. Equivalent memory ≈ 1/(1−λ) ≈ 17 trading days, appropriate for a
     10-day VaR horizon.
  3. Industry standard; implemented widely in risk systems and validated
     in the academic literature (RiskMetrics methodology, 1994).

Both EWMA and historical (equal-weight) covariance are provided for
transparency; the pipeline uses EWMA by default.

All covariance matrices are in DAILY units (variance of daily log-returns).
Multiply by 252 to annualise or by h to get the h-day covariance.
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── EWMA covariance ───────────────────────────────────────────────────────────

def ewma_cov(
    returns: pd.DataFrame,
    lam: float = config.Q6_EWMA_LAMBDA,
) -> pd.DataFrame:
    """
    Exponentially Weighted Moving Average (EWMA) daily covariance matrix.

    Update rule:   Σ_t = λ·Σ_{t-1} + (1-λ)·r_t·r_t^T

    Initialisation: sample covariance of the first 30 observations.
    The decay factor λ = 0.94 is the RiskMetrics standard for daily data.

    Parameters
    ----------
    returns : pd.DataFrame  shape (T, n)   daily log-returns
    lam     : float         EWMA decay factor ∈ (0, 1)

    Returns
    -------
    pd.DataFrame  (n × n)  EWMA daily covariance matrix
    """
    r = returns.values.astype(float)   # (T, n)
    T, n = r.shape
    init_window = min(30, T // 4)

    # Initialise with equal-weight sample covariance of first init_window days
    C = np.cov(r[:init_window].T, ddof=1)
    if C.ndim == 0:          # single asset edge-case
        C = np.array([[float(C)]])

    # Recursive EWMA update
    for t in range(init_window, T):
        rt = r[t].reshape(-1, 1)          # (n, 1)
        C  = lam * C + (1.0 - lam) * rt @ rt.T

    return pd.DataFrame(C, index=returns.columns, columns=returns.columns)


# ── Historical (equal-weight) covariance ──────────────────────────────────────

def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Historical sample covariance matrix (equal weights over the full window).

    Uses an unbiased estimator (ddof=1).  The 2-year window corresponds
    to approximately 504 trading days.
    """
    return returns.cov()


# ── Correlation matrix ────────────────────────────────────────────────────────

def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_psd(cov: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """
    Regularise a covariance matrix to ensure positive semi-definiteness.
    Adds eps·I if the smallest eigenvalue is below eps.
    """
    vals = np.linalg.eigvalsh(cov.values)
    if vals.min() < eps:
        cov_reg = cov.copy()
        cov_reg.values[:] += (eps - vals.min()) * np.eye(len(cov))
        return cov_reg
    return cov


def annualise_cov(
    cov: pd.DataFrame,
    trading_days: int = config.TRADING_DAYS,
) -> pd.DataFrame:
    """Annualise a daily covariance matrix by multiplying by trading_days."""
    return cov * trading_days


def scale_to_horizon(
    cov_daily: pd.DataFrame,
    horizon_days: int = config.Q6_HORIZON_DAYS,
) -> pd.DataFrame:
    """Scale a daily covariance matrix to a multi-day horizon (√T rule)."""
    return cov_daily * horizon_days
