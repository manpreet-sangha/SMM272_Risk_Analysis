"""
Q2 — Model Fitting: Gaussian (H0) and GARCH(1,1) (H1).

Under H0, the assumed DGP is i.i.d. N(0, σ²) with constant volatility
estimated by MLE (= sample standard deviation of de-meaned returns).

Under H1, the true DGP is a GARCH(1,1) process:
    r_t   = μ + ε_t,      ε_t = σ_t z_t,    z_t ~ i.i.d. N(0,1)
    σ²_t  = ω + α ε²_{t-1} + β σ²_{t-1}

Both models are fitted to the EW portfolio returns from Q1 Part 1.

References
----------
Bollerslev, T. (1986). Generalised autoregressive conditional
    heteroscedasticity. Journal of Econometrics, 31(3), 307–327.
Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with
    estimates of the variance of United Kingdom inflation. Econometrica,
    50(4), 987–1007.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from arch import arch_model

from config import Q1_1_OUTPUT_DIR, Q2_OUTPUT_DIR
from logger import get_logger

logger = get_logger("q2_fit_models")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_portfolio_returns():
    """Load EW portfolio daily log returns produced by Q1 Part 1."""
    path = os.path.join(Q1_1_OUTPUT_DIR, "portfolio_returns.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Column may be named 'EW_Portfolio' or 'portfolio_return'
    col = df.columns[0]
    returns = df[col].dropna()
    logger.info(f"Loaded portfolio returns: {len(returns)} observations "
                f"({returns.index[0].date()} to {returns.index[-1].date()})")
    return returns


# ── H0: Gaussian constant-volatility model ────────────────────────────────────

def fit_gaussian(returns):
    """
    Fit the Gaussian H0 model: r_t ~ i.i.d. N(μ, σ²).

    MLE for a Gaussian gives:
        μ̂  = T⁻¹ Σ r_t
        σ̂² = T⁻¹ Σ (r_t − μ̂)²

    Parameters
    ----------
    returns : pd.Series  — daily log returns

    Returns
    -------
    params : dict with keys 'mu', 'sigma'
    """
    arr  = returns.values.astype(float)
    mu   = arr.mean()
    sigma = arr.std(ddof=0)          # MLE (divide by T)
    params = {"mu": mu, "sigma": sigma}
    logger.info(f"Gaussian H0 fit:  μ = {mu:.6f},  σ = {sigma:.6f}")
    return params


def gaussian_var(sigma, mu, alpha, confidence):
    """
    Compute VaR under the Gaussian model.

    VaR (as a negative number, representing a loss) at confidence level c:
        VaR_c = μ + Φ⁻¹(1−c) × σ

    Parameters
    ----------
    sigma      : float — conditional std (constant under H0)
    mu         : float — mean (constant under H0)
    alpha      : float — tail probability  = 1 − confidence
    confidence : float — e.g. 0.99

    Returns
    -------
    var : float (negative — a loss)
    """
    from scipy.stats import norm
    z   = norm.ppf(alpha)           # e.g. Φ⁻¹(0.01) ≈ −2.326
    var = mu + z * sigma            # negative value: this IS a loss threshold
    return float(var)


# ── H1: GARCH(1,1) Normal model ───────────────────────────────────────────────

def fit_garch(returns):
    """
    Fit a GARCH(1,1) model with constant mean and Normal innovations via MLE.

    Model:
        r_t   = μ + ε_t
        ε_t   = σ_t z_t,   z_t ~ N(0,1)
        σ²_t  = ω + α ε²_{t-1} + β σ²_{t-1}

    Uses the `arch` library.  Returns are scaled to percentage points for
    numerical stability during optimisation (σ in pct units), then converted
    back to decimal.

    Parameters
    ----------
    returns : pd.Series

    Returns
    -------
    params : dict with keys
        'mu'     — unconditional mean (decimal)
        'omega'  — GARCH intercept (decimal²)
        'alpha'  — ARCH coefficient
        'beta'   — GARCH coefficient
        'sigma2_uncond' — long-run unconditional variance  ω/(1−α−β)
        'persistence'   — α + β
        'sigma_uncond'  — √(long-run variance)  (decimal)
        'loglik'        — log-likelihood at optimum
        'aic', 'bic'    — information criteria
    """
    arr_pct = returns.values * 100.0

    model = arch_model(
        arr_pct,
        mean="Constant",
        vol="Garch",
        p=1, q=1,
        dist="Normal",
        rescale=False,
    )
    res = model.fit(disp="off", show_warning=False, options={"maxiter": 1000})

    mu_pct    = float(res.params["mu"])
    omega_pct = float(res.params["omega"])   # in pct² units
    alpha1    = float(res.params["alpha[1]"])
    beta1     = float(res.params["beta[1]"])

    # Convert back to decimal units
    mu    = mu_pct / 100.0
    omega = omega_pct / 10_000.0             # pct² → decimal²

    persist     = alpha1 + beta1
    sigma2_uncond = omega / (1.0 - persist) if persist < 1.0 else np.nan
    sigma_uncond  = np.sqrt(sigma2_uncond) if not np.isnan(sigma2_uncond) else np.nan

    params = {
        "mu":           mu,
        "omega":        omega,
        "alpha":        alpha1,
        "beta":         beta1,
        "persistence":  persist,
        "sigma2_uncond": sigma2_uncond,
        "sigma_uncond":  sigma_uncond,
        "loglik":       float(res.loglikelihood),
        "aic":          float(res.aic),
        "bic":          float(res.bic),
        # store pct-unit params for simulation convenience
        "_mu_pct":    mu_pct,
        "_omega_pct": omega_pct,
    }

    logger.info(
        f"GARCH(1,1) H1 fit: μ={mu:.6f}, ω={omega:.2e}, "
        f"α={alpha1:.4f}, β={beta1:.4f}, α+β={persist:.4f}, "
        f"σ_LR={sigma_uncond:.6f}"
    )
    return params


# ── GARCH simulation ──────────────────────────────────────────────────────────

def simulate_garch(params, T, n_paths, seed=None):
    """
    Simulate n_paths independent realisations of a GARCH(1,1) process,
    each of length T.

    Parameters
    ----------
    params  : dict from fit_garch()
    T       : int   — path length (number of daily returns)
    n_paths : int   — number of Monte Carlo paths
    seed    : int or None — RNG seed

    Returns
    -------
    paths : ndarray, shape (n_paths, T) — simulated decimal returns
    """
    rng   = np.random.default_rng(seed)
    mu    = params["mu"]
    omega = params["omega"]
    alpha = params["alpha"]
    beta  = params["beta"]
    sigma2_uncond = params["sigma2_uncond"]

    # Initialise variance at the unconditional level
    sigma2_init = sigma2_uncond if np.isfinite(sigma2_uncond) else omega / (1 - alpha - beta + 1e-8)

    paths  = np.empty((n_paths, T), dtype=float)
    sigma2 = np.full(n_paths, sigma2_init, dtype=float)

    # Draw all innovations at once
    z = rng.standard_normal((n_paths, T))

    for t in range(T):
        eps          = np.sqrt(sigma2) * z[:, t]   # shape (n_paths,)
        paths[:, t]  = mu + eps
        sigma2       = omega + alpha * eps**2 + beta * sigma2

    return paths


def simulate_gaussian(params, T, n_paths, seed=None):
    """
    Simulate n_paths independent realisations of the Gaussian H0 DGP,
    each of length T.

    Parameters
    ----------
    params  : dict from fit_gaussian()
    T       : int
    n_paths : int
    seed    : int or None

    Returns
    -------
    paths : ndarray, shape (n_paths, T)
    """
    rng   = np.random.default_rng(seed)
    mu    = params["mu"]
    sigma = params["sigma"]
    return mu + sigma * rng.standard_normal((n_paths, T))


# ── Save fitted parameters ─────────────────────────────────────────────────────

def save_fitted_params(gauss_params, garch_params):
    """Write fitted model parameters to a CSV in output_q2/."""
    rows = [
        {"Model": "Gaussian (H0)", "Parameter": "mu",          "Value": gauss_params["mu"]},
        {"Model": "Gaussian (H0)", "Parameter": "sigma",        "Value": gauss_params["sigma"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "mu",         "Value": garch_params["mu"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "omega",      "Value": garch_params["omega"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "alpha",      "Value": garch_params["alpha"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "beta",       "Value": garch_params["beta"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "persistence","Value": garch_params["persistence"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "sigma_uncond","Value": garch_params["sigma_uncond"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "loglik",     "Value": garch_params["loglik"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "aic",        "Value": garch_params["aic"]},
        {"Model": "GARCH(1,1) (H1)", "Parameter": "bic",        "Value": garch_params["bic"]},
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(Q2_OUTPUT_DIR, "fitted_params.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved fitted parameters → {path}")
    return df


# ── Stand-alone run ────────────────────────────────────────────────────────────

def run_fit_models():
    """Fit both models, log results, save CSV. Returns (gauss_params, garch_params)."""
    from logger import log_start, log_end
    log_start(logger, "q2_fit_models.py")

    returns      = load_portfolio_returns()
    gauss_params = fit_gaussian(returns)
    garch_params = fit_garch(returns)
    save_fitted_params(gauss_params, garch_params)

    log_end(logger, "q2_fit_models.py")
    return gauss_params, garch_params, returns


if __name__ == "__main__":
    from logger import setup_run_logger
    setup_run_logger("q2_fit_models")
    run_fit_models()
