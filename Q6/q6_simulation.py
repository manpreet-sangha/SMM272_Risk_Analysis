"""
Q6 — Monte Carlo Simulation Engine
====================================
Simulates correlated log-normal stock price paths over a 10-trading-day
horizon using the Cholesky decomposition of the EWMA daily covariance matrix.

Simulation method
-----------------
Single-step log-normal model (standard for short-horizon VaR):

  log(S_T / S_0) ~ N(−½·diag(Σ_h), Σ_h)

  where  Σ_h = h · Σ_daily   (h = 10 trading days)
         Σ_daily              = EWMA daily covariance matrix

Implementation
--------------
  1. L = cholesky(Σ_h)           [Cholesky factor, shape n×n]
  2. z ~ N(0, I_n)               [standard normal draw, shape N×n]
  3. ε = z @ L^T                 [correlated innovations, shape N×n]
  4. S_T,i = S_0,i · exp(−½Σ_h[i,i] + ε_i)

The drift term −½Σ_h[i,i] ensures the expected stock price equals the
current price under the physical measure with zero drift, which is the
standard conservative assumption for short-horizon risk measurement.

Returns
-------
simulate_prices() returns S_T: np.ndarray of shape (N_SIMS, N_ASSETS)
simulate_log_returns() returns log(S_T/S_0): np.ndarray of shape (N_SIMS, N_ASSETS)
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from q6_covariance import ensure_psd, scale_to_horizon


def simulate_prices(
    spot_prices:  pd.Series,
    cov_daily:    pd.DataFrame,
    n_sims:       int   = config.Q6_N_SIMS,
    horizon_days: int   = config.Q6_HORIZON_DAYS,
    seed:         int   = config.Q6_RANDOM_SEED,
) -> np.ndarray:
    """
    Simulate terminal stock prices over a multi-day horizon.

    Parameters
    ----------
    spot_prices  : pd.Series      current stock prices (aligned with cov columns)
    cov_daily    : pd.DataFrame   EWMA daily covariance matrix (n × n)
    n_sims       : int            number of Monte Carlo paths
    horizon_days : int            simulation horizon in trading days
    seed         : int            random seed for reproducibility

    Returns
    -------
    S_T : np.ndarray  shape (n_sims, n_assets)
        Simulated terminal stock prices.
    """
    S0 = np.asarray(spot_prices, dtype=float)   # (n,)
    n  = len(S0)

    # Scale covariance to the horizon
    cov_h = scale_to_horizon(cov_daily, horizon_days)  # daily × h
    cov_h = ensure_psd(cov_h)

    # Cholesky decomposition: Σ_h = L L^T
    try:
        L = np.linalg.cholesky(cov_h.values)
    except np.linalg.LinAlgError:
        # Fallback: add regularisation
        reg = 1e-8 * np.eye(n)
        L = np.linalg.cholesky(cov_h.values + reg)

    # Variance correction (drift-free log-normal)
    var_h = np.diag(cov_h.values)    # (n,)  = σ_i^2 × h

    # Draw standard normal innovations
    rng = np.random.default_rng(seed)
    z   = rng.standard_normal((n_sims, n))   # (N, n)

    # Correlated innovations
    eps = z @ L.T                            # (N, n)

    # Terminal prices: log(S_T/S_0) ~ N(-½·var_h, Σ_h)
    log_S_T = np.log(S0) - 0.5 * var_h + eps    # (N, n)
    S_T     = np.exp(log_S_T)

    return S_T   # (n_sims, n_assets)


def simulate_log_returns(
    spot_prices:  pd.Series,
    cov_daily:    pd.DataFrame,
    n_sims:       int   = config.Q6_N_SIMS,
    horizon_days: int   = config.Q6_HORIZON_DAYS,
    seed:         int   = config.Q6_RANDOM_SEED,
) -> np.ndarray:
    """
    Simulate log-returns log(S_T / S_0) over the horizon.

    Returns
    -------
    log_rets : np.ndarray  shape (n_sims, n_assets)
    """
    S0  = np.asarray(spot_prices, dtype=float)
    S_T = simulate_prices(spot_prices, cov_daily, n_sims, horizon_days, seed)
    return np.log(S_T / S0)
