"""
Q2 — Sensitivity of Kupiec power to GARCH persistence (α + β).

Varying the persistence parameter ρ = α + β while keeping the
unconditional variance constant allows us to isolate the effect of
volatility clustering on the power of the Kupiec test.

Parameterisation
----------------
Given target persistence ρ = α + β and the fitted unconditional variance
σ²_LR = ω / (1 − ρ), we keep σ²_LR fixed at the value estimated from Q1
data and decompose as:
    α = γ · ρ,   β = (1−γ) · ρ,   ω = σ²_LR · (1 − ρ)
where γ = α / (α + β) is kept equal to the fitted ratio, preserving the
relative contribution of the ARCH and GARCH terms.

For each ρ in Q2_PERSISTENCE_GRID, n_reps GARCH paths are simulated and
the Kupiec power is estimated, producing a power curve over [0, 1).

At ρ → 0 (no clustering), violation independence holds and Kupiec has its
highest power (equivalent to i.i.d. Binomial deviations).
At ρ → 1 (near unit-root), violations cluster severely, inflating the
variance of the violation count; the Binomial approximation breaks down
and Kupiec power can even decline.

References
----------
Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
    measurement models. Journal of Derivatives, 3(2), 73–84.
Candelon, B., Colletaz, G., Hurlin, C., & Tokpavi, S. (2011). Backtesting
    Value-at-Risk: A GMM duration-based test. Journal of Financial
    Econometrics, 9(2), 314–343.
McNeil, A. J., Frey, R., & Embrechts, P. (2015). Quantitative Risk
    Management (Revised ed.). Princeton University Press.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import chi2

from config import (Q2_CONFIDENCE_LEVELS, Q2_MC_REPS, Q2_RANDOM_SEED,
                    Q2_PERSISTENCE_GRID, Q2_OUTPUT_DIR)
from logger import get_logger
from q2_fit_models import gaussian_var, simulate_garch
from q2_kupiec_power import _compute_rejections

logger = get_logger("q2_power_vs_persistence")

CRIT_05 = chi2.ppf(0.95, df=1)


def _make_garch_params_for_persistence(base_params, rho):
    """
    Construct a modified GARCH(1,1) parameter dict with persistence = rho,
    holding the unconditional variance constant at its fitted value.

    gamma = α / (α + β) is preserved from the fitted model.
    """
    gamma = base_params["alpha"] / base_params["persistence"]
    sigma2_lr = base_params["sigma2_uncond"]

    omega_new = sigma2_lr * (1.0 - rho)
    alpha_new = gamma * rho
    beta_new  = (1.0 - gamma) * rho

    new_params = dict(base_params)
    new_params["omega"]       = omega_new
    new_params["alpha"]       = alpha_new
    new_params["beta"]        = beta_new
    new_params["persistence"] = rho
    new_params["sigma2_uncond"] = sigma2_lr   # unchanged by construction
    new_params["sigma_uncond"]  = np.sqrt(sigma2_lr)
    return new_params


def power_vs_persistence(gauss_params, garch_params,
                          persistence_grid=None,
                          confidence_levels=None,
                          T=2893,
                          n_reps=None,
                          seed=None):
    """
    Simulate Kupiec power for each (persistence, confidence level) pair.

    Returns
    -------
    df : pd.DataFrame with columns
        ['persistence', 'confidence', 'alpha', 'power', 'size',
         'mean_viol_h1', 'expected_viol']
    """
    if persistence_grid is None:
        persistence_grid = Q2_PERSISTENCE_GRID
    if confidence_levels is None:
        confidence_levels = Q2_CONFIDENCE_LEVELS
    if n_reps is None:
        n_reps = Q2_MC_REPS
    if seed is None:
        seed = Q2_RANDOM_SEED

    rows = []
    for idx_rho, rho in enumerate(persistence_grid):
        logger.info(f"  ρ = {rho:.2f} — simulating {n_reps:,} GARCH paths …")
        mod_params = _make_garch_params_for_persistence(garch_params, rho)
        paths_h1   = simulate_garch(mod_params, T, n_reps,
                                     seed=seed + idx_rho * 7)

        for conf in confidence_levels:
            alpha   = 1.0 - conf
            var_thr = gaussian_var(gauss_params["sigma"], gauss_params["mu"],
                                   alpha=alpha, confidence=conf)

            _, rej_h1 = _compute_rejections(paths_h1, var_thr, alpha, CRIT_05)
            viol_h1   = (paths_h1 < var_thr).sum(axis=1)

            rows.append({
                "persistence":   rho,
                "confidence":    conf,
                "alpha":         alpha,
                "power":         rej_h1.mean(),
                "mean_viol_h1":  viol_h1.mean(),
                "expected_viol": alpha * T,
            })

    df = pd.DataFrame(rows)
    return df


def save_power_vs_persistence(df, filename="power_vs_persistence.csv"):
    path = os.path.join(Q2_OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"Saved power-vs-persistence table → {path}")
    return path


def run_power_vs_persistence(gauss_params=None, garch_params=None):
    """Run sensitivity analysis, save CSV, return DataFrame."""
    from logger import log_start, log_end
    log_start(logger, "q2_power_vs_persistence.py")

    if gauss_params is None or garch_params is None:
        from q2_fit_models import run_fit_models
        gauss_params, garch_params, _ = run_fit_models()

    df = power_vs_persistence(gauss_params, garch_params)
    save_power_vs_persistence(df)

    log_end(logger, "q2_power_vs_persistence.py")
    return df


if __name__ == "__main__":
    from logger import setup_run_logger
    setup_run_logger("q2_power_vs_persistence")
    run_power_vs_persistence()
