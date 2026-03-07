"""
Q6 — VaR, ES, and Risk Decomposition (Non-Gaussian Case)
==========================================================
Implements portfolio P&L computation, VaR, ES, and the full Euler
risk decomposition for the non-Gaussian (simulation-based) case.

Definitions (loss convention: positive = loss)
----------------------------------------------
  VaR_α    = −quantile_{1−α}(ΔW)        α = 0.99 → worst 1 % tail
  ES_α     = −E[ΔW | ΔW ≤ −VaR_α]      average loss in the tail

Risk decomposition (non-Gaussian Euler allocation)
---------------------------------------------------
For a portfolio ΔW = Σ_i ΔP_i where ΔP_i = Q_i · mult_i · (V_i(S_T) − V_i(S_0)):

  Marginal ES_i   = ∂ES/∂Q_i = −E[unit_pnl_i | ΔW ≤ −VaR_α]
  Component ES_i  = Q_i · Marginal ES_i = −E[ΔP_i | ΔW ≤ −VaR_α]

  Property: Σ_i Component ES_i = Portfolio ES  (exact, by Euler's theorem)

  Marginal VaR_i  = ∂VaR/∂Q_i ≈ (VaR(Q + ε·eᵢ) − VaR(Q)) / ε
  Component VaR_i = Q_i · Marginal VaR_i

  Property: Σ_i Component VaR_i ≈ Portfolio VaR  (Euler, ε → 0)

"Non-Gaussian" here means that:
  • VaR and ES are computed directly from simulation quantiles without
    assuming a parametric distribution.
  • Marginal VaR uses numerical finite differences on the simulated P&L
    rather than the Gaussian covariance formula (δ·Σ·δ / σ_P).
  • Component ES exploits the conditional expectation in the tail, which
    captures fat tails, skewness, and non-linear option payoffs exactly.
"""

import sys
import os
import copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from q6_option_pricing import bs_price


# ── P&L computation ───────────────────────────────────────────────────────────

def compute_pnl(
    S_T:          np.ndarray,
    portfolio:    list[dict],
    r:            float = config.Q6_RISK_FREE_RATE,
    horizon_days: int   = config.Q6_HORIZON_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute portfolio P&L and per-instrument P&L for all simulations.

    Parameters
    ----------
    S_T       : np.ndarray  (N_sims, 4)  simulated terminal stock prices
    portfolio : list[dict]  from build_portfolio()
    r         : float       risk-free rate
    horizon_days : int      horizon in trading days

    Returns
    -------
    total_pnl : (N_sims,)      portfolio P&L per simulation
    inst_pnl  : (N_sims, 4)   per-instrument P&L (Q_i × unit P&L)
    unit_pnl  : (N_sims, 4)   per-CONTRACT P&L [mult × (V_T − V_0)], sign-free
    """
    n_sims      = S_T.shape[0]
    n_assets    = len(portfolio)
    inst_pnl    = np.zeros((n_sims, n_assets))
    unit_pnl    = np.zeros((n_sims, n_assets))
    T_years     = horizon_days / config.TRADING_DAYS   # 10/252

    for j, leg in enumerate(portfolio):
        S0       = leg["spot"]
        K        = leg["strike"]
        T0       = leg["ttm"]
        T_rem    = max(T0 - T_years, 1e-6)   # remaining TTM
        sigma    = leg["vol"]
        q        = leg["quantity"]
        mult     = leg["multiplier"]
        ot       = leg["option_type"]

        # Initial option value (per share, scalar)
        v0  = leg["price_per_share"]

        # Simulated terminal values (per share, vectorised over N_sims)
        S_sim = S_T[:, j]
        v_T   = bs_price(S_sim, K, r, T_rem, sigma, ot)   # (N_sims,)

        # P&L per contract (full revaluation)
        u_pnl       = mult * (v_T - v0)    # (N_sims,) — per long contract
        unit_pnl[:, j] = u_pnl
        inst_pnl[:, j] = q * u_pnl        # signed position P&L

    total_pnl = inst_pnl.sum(axis=1)
    return total_pnl, inst_pnl, unit_pnl


# ── VaR and ES ────────────────────────────────────────────────────────────────

def compute_var_es(
    total_pnl:  np.ndarray,
    confidence: float = config.Q6_CONFIDENCE,
) -> tuple[float, float]:
    """
    Simulation-based VaR and ES (non-parametric, loss convention).

    Parameters
    ----------
    total_pnl  : (N,) array of portfolio P&L values
    confidence : confidence level (e.g. 0.99 for 99 %)

    Returns
    -------
    var : float  VaR (positive = potential loss)
    es  : float  ES  (positive = expected loss in tail)
    """
    alpha    = 1.0 - confidence
    var      = float(-np.percentile(total_pnl, alpha * 100.0))
    tail_mask = total_pnl <= -var
    es        = float(-total_pnl[tail_mask].mean()) if tail_mask.sum() > 0 else var
    return var, es


# ── Component ES (non-Gaussian Euler allocation) ──────────────────────────────

def compute_component_es(
    total_pnl:  np.ndarray,
    inst_pnl:   np.ndarray,
    unit_pnl:   np.ndarray,
    quantities: np.ndarray,
    confidence: float = config.Q6_CONFIDENCE,
) -> dict[str, np.ndarray]:
    """
    Non-Gaussian Component ES and Marginal ES via Euler allocation.

    Component ES_i = −E[ΔP_i | ΔW ≤ −VaR_α]
    Marginal  ES_i = Component ES_i / Q_i  (per-contract sensitivity)

    The component ES values sum exactly to portfolio ES.

    Parameters
    ----------
    total_pnl  : (N,)    portfolio P&L
    inst_pnl   : (N, 4)  per-instrument P&L  (signed by Q_i)
    unit_pnl   : (N, 4)  per-contract P&L    (unsigned Q_i factored out)
    quantities : (4,)    signed contract quantities Q_i
    confidence : float   confidence level

    Returns
    -------
    dict with keys:
      "var"         : float    portfolio VaR
      "es"          : float    portfolio ES
      "comp_es"     : (4,)     Component ES per instrument
      "marg_es"     : (4,)     Marginal ES per additional contract
      "tail_count"  : int      number of tail observations
    """
    var, es   = compute_var_es(total_pnl, confidence)
    tail_mask = total_pnl <= -var
    n_tail    = int(tail_mask.sum())

    if n_tail == 0:
        comp_es = np.zeros(inst_pnl.shape[1])
        marg_es = np.zeros(unit_pnl.shape[1])
    else:
        # Component ES: conditional mean of inst_pnl in the tail (negated → loss)
        comp_es = -inst_pnl[tail_mask].mean(axis=0)   # (4,)

        # Marginal ES: conditional mean of unit_pnl in the tail (per additional contract)
        marg_es = -unit_pnl[tail_mask].mean(axis=0)   # (4,)

    return {
        "var":        var,
        "es":         es,
        "comp_es":    comp_es,
        "marg_es":    marg_es,
        "tail_count": n_tail,
    }


# ── Marginal VaR (numerical finite differences) ───────────────────────────────

def compute_marginal_var(
    total_pnl:  np.ndarray,
    unit_pnl:   np.ndarray,
    quantities: np.ndarray,
    confidence: float = config.Q6_CONFIDENCE,
    epsilon:    float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Numerical Marginal VaR via finite differences on the simulation.

    Since the portfolio P&L is linear in the position sizes around the
    current holdings, we can perturb Q_i by ε without re-running the
    simulation:

      ΔW'(k) = ΔW(k) + ε · unit_pnl_i(k)

    This exploits the linearity of P&L in contract size for a fixed
    set of simulated stock paths.

    Parameters
    ----------
    total_pnl  : (N,)    baseline portfolio P&L
    unit_pnl   : (N, 4)  per-contract P&L (positive = per long contract)
    quantities : (4,)    signed contract quantities Q_i
    confidence : float   confidence level
    epsilon    : float   perturbation size in contracts (default: 1 contract)

    Returns
    -------
    dict with keys:
      "var"      : float  baseline portfolio VaR
      "marg_var" : (4,)   Marginal VaR per additional contract
      "comp_var" : (4,)   Component VaR = Q_i × Marginal VaR_i
    """
    base_var, _ = compute_var_es(total_pnl, confidence)
    n_assets    = unit_pnl.shape[1]
    marg_var    = np.zeros(n_assets)

    for j in range(n_assets):
        # Perturbed portfolio P&L: add ε additional long contracts of instrument j
        pert_pnl   = total_pnl + epsilon * unit_pnl[:, j]
        pert_var, _ = compute_var_es(pert_pnl, confidence)
        marg_var[j] = (pert_var - base_var) / epsilon

    comp_var = quantities * marg_var   # Euler decomposition

    return {
        "var":      base_var,
        "marg_var": marg_var,
        "comp_var": comp_var,
    }


# ── Aggregate results ─────────────────────────────────────────────────────────

def compute_all_metrics(
    S_T:          np.ndarray,
    portfolio:    list[dict],
    r:            float = config.Q6_RISK_FREE_RATE,
    horizon_days: int   = config.Q6_HORIZON_DAYS,
    confidence:   float = config.Q6_CONFIDENCE,
) -> dict:
    """
    Run the full risk computation pipeline.

    Returns a single master results dict with:
      pnl_arrays  : dict  (total_pnl, inst_pnl, unit_pnl)
      var_es      : dict  (var, es)
      comp_es_res : dict  (comp_es, marg_es, tail_count)
      marg_var_res: dict  (marg_var, comp_var)
      tickers     : list
      quantities  : np.ndarray
    """
    tickers    = [leg["ticker"]   for leg in portfolio]
    quantities = np.array([leg["quantity"] for leg in portfolio], dtype=float)

    total_pnl, inst_pnl, unit_pnl = compute_pnl(
        S_T, portfolio, r, horizon_days
    )

    var, es = compute_var_es(total_pnl, confidence)

    comp_es_res  = compute_component_es(
        total_pnl, inst_pnl, unit_pnl, quantities, confidence
    )
    marg_var_res = compute_marginal_var(
        total_pnl, unit_pnl, quantities, confidence
    )

    return {
        "tickers":    tickers,
        "quantities": quantities,
        "confidence": confidence,
        "total_pnl":  total_pnl,
        "inst_pnl":   inst_pnl,
        "unit_pnl":   unit_pnl,
        "var":        var,
        "es":         es,
        "comp_es":    comp_es_res["comp_es"],
        "marg_es":    comp_es_res["marg_es"],
        "tail_count": comp_es_res["tail_count"],
        "marg_var":   marg_var_res["marg_var"],
        "comp_var":   marg_var_res["comp_var"],
    }
