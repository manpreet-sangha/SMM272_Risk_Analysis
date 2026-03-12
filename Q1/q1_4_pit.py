"""
Q1 Part 4 — Probability Integral Transform (PIT) test.

Based on Christoffersen, P.F. (2003/2012) "Elements of Financial Risk
Management", Academic Press.

Theory
------
For a sequence of one-step-ahead conditional density forecasts F_t(·), the
*probability integral transforms* (PITs) are defined as

    z_t  =  F_t(r_t | Ω_{t-1})  =  ∫_{-∞}^{r_t} f_t(u; Ω_{t-1}) du

where r_t is the realized portfolio return on day t and Ω_{t-1} is
information available through day t−1 (represented here by the rolling
estimation window).

Diebold, Gunther & Tay (1998) show that if the sequence of forecasted
densities {f_t} is correctly specified, the PITs {z_t} should be i.i.d.
Uniform(0, 1).  Departures from this null arise from two sources:

  •  Mis-specification of the *marginal distribution* (distributional mis-fit)
     — tested via the KS and chi-squared uniformity tests.

  •  Failure of *serial independence* (temporal mis-fit: clustering,
     autocorrelation) — tested via Ljung–Box tests on the normal-probability
     transforms u_t = Φ⁻¹(z_t) and their squares u_t².

Tests applied
-------------
1. Kolmogorov–Smirnov uniformity test  (H0 : z_t ~ U(0,1))
2. Pearson chi-squared uniformity test  (10 equal-width histogram bins)
3. Ljung–Box test on u_t  = Φ⁻¹(z_t)         — first-moment independence
4. Ljung–Box test on u_t² = [Φ⁻¹(z_t)]²     — second-moment / ARCH effects

PIT computation per model
-------------------------
  Historical Simulation : rank-based empirical CDF (continuity-corrected)
  Parametric Normal     : Φ((r_t − μ_w) / σ_w)
  Parametric Student-t  : T_ν((r_t − loc) / scale)  with MLE-fitted ν
  GARCH(1,1)            : Φ((r_t − μ_G) / σ_{t|t-1})  with one-step forecast

References
----------
Christoffersen, P.F. (1998). Evaluating Interval Forecasts. *International
    Economic Review*, 39(4), 841–862.
Christoffersen, P.F. (2003/2012). *Elements of Financial Risk Management*.
    Academic Press.  §4.4 (Density Evaluation).
Diebold, F.X., Gunther, T.A. & Tay, A.S. (1998). Evaluating Density
    Forecasts with Applications to Financial Risk Management. *International
    Economic Review*, 39(4), 863–883.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import Q1_4_OUTPUT_DIR
from q1_2_rolling_window import generate_rolling_windows

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# ── Per-model PIT computation functions ───────────────────────────────────────

def _pit_normal(window: np.ndarray, r_next: float) -> float:
    """
    PIT for the Parametric Normal model.

    z_t = Φ((r_t − μ_w) / σ_w)

    where μ_w and σ_w are the sample mean and standard deviation of the
    rolling estimation window.
    """
    mu    = window.mean()
    sigma = window.std(ddof=1)
    if sigma <= 0.0:
        return np.nan
    return float(stats.norm.cdf((r_next - mu) / sigma))


def _pit_studentt(window: np.ndarray, r_next: float) -> float:
    """
    PIT for the Parametric Student-t model.

    z_t = T_ν(r_t; loc, scale)

    where (ν, loc, scale) are estimated from the rolling window by MLE.
    Falls back to the Normal PIT if MLE fails or scale ≤ 0.
    """
    try:
        nu, loc, scale = stats.t.fit(window)
        if scale <= 0.0 or nu <= 0.5:
            return _pit_normal(window, r_next)
        return float(stats.t.cdf(r_next, df=nu, loc=loc, scale=scale))
    except Exception:
        return _pit_normal(window, r_next)


def _pit_historical(window: np.ndarray, r_next: float) -> float:
    """
    PIT for Historical Simulation.

    Uses the continuity-corrected empirical CDF (Hazen formula):

        z_t = (#{r_w ≤ r_t} + 0.5) / (n_w + 1)

    This adjustment avoids exact 0 or 1 values that would be undefined under
    the normal-probability transform Φ⁻¹(·) used in the LB independence test.
    """
    n     = len(window)
    count = int(np.sum(window <= r_next))
    return float((count + 0.5) / (n + 1))


def _pit_garch(window: np.ndarray, r_next: float) -> float:
    """
    PIT for the GARCH(1,1)-Normal model.

    z_t = Φ((r_t − μ_G) / σ_{t|t-1})

    where μ_G is the GARCH constant mean and σ_{t|t-1} is the one-step-ahead
    conditional standard deviation from a GARCH(1,1) model fitted to the
    rolling window.  Falls back to the Normal PIT if arch is unavailable or
    estimation fails.
    """
    if not ARCH_AVAILABLE:
        return _pit_normal(window, r_next)
    try:
        arr_pct = window * 100.0          # scale to % for numerical stability
        model   = arch_model(
            arr_pct,
            mean="Constant",
            vol="Garch",
            p=1, q=1,
            dist="Normal",
            rescale=False,
        )
        res      = model.fit(disp="off", show_warning=False,
                             options={"maxiter": 500})
        forecast = res.forecast(horizon=1, reindex=False)
        cond_var = float(forecast.variance.iloc[-1, 0])
        cond_std = np.sqrt(max(cond_var, 1e-12)) / 100.0   # back to decimal
        mu_g     = float(res.params["mu"]) / 100.0
        return float(stats.norm.cdf((r_next - mu_g) / cond_std))
    except Exception:
        return _pit_normal(window, r_next)


_PIT_FN = {
    "HS":       _pit_historical,
    "Normal":   _pit_normal,
    "StudentT": _pit_studentt,
    "GARCH":    _pit_garch,
}


# ── Build PIT series ───────────────────────────────────────────────────────────

def compute_pit_series(port_ret: pd.Series, tag: str) -> pd.Series:
    """
    Compute the full out-of-sample PIT series for a given model.

    For each forecast date t the function:
      1. constructs the rolling estimation window [t − 6 m, t) from port_ret,
      2. calibrates the model on that window,
      3. evaluates the model CDF at the realized return r_t.

    Parameters
    ----------
    port_ret : pd.Series
        Daily portfolio log-return series indexed by pd.Timestamp.
    tag : str
        Model tag — one of ``'HS'``, ``'Normal'``, ``'StudentT'``,
        ``'GARCH'``.

    Returns
    -------
    z : pd.Series
        PIT values z_t ∈ (0, 1), indexed by forecast date.  NaN entries
        (rare numerical failures) are dropped.
    """
    pit_fn  = _PIT_FN[tag]
    results = {}

    for date, window in generate_rolling_windows(port_ret):
        if date not in port_ret.index:
            continue
        r_next = float(port_ret.loc[date])
        z      = pit_fn(window.values, r_next)
        if not np.isnan(z):
            # Clip to open (0, 1) — required for the Φ⁻¹ transform
            results[date] = float(np.clip(z, 1e-10, 1.0 - 1e-10))

    return pd.Series(results, name=f"PIT_{tag}")


# ── Test 1: KS uniformity ──────────────────────────────────────────────────────

def pit_ks_test(z) -> tuple:
    """
    Kolmogorov–Smirnov test for Uniform(0, 1).

    H0 : z_t ~ Uniform(0, 1)   (correctly specified density)
    HA : z_t is not uniformly distributed

    The two-sided KS statistic measures the maximum absolute deviation between
    the empirical CDF of {z_t} and the Uniform(0,1) CDF.

    Parameters
    ----------
    z : array-like
        PIT values.

    Returns
    -------
    ks_stat : float — KS test statistic
    p_value : float — asymptotic p-value (right-tail)
    reject  : bool  — True if H0 rejected at the 5 % level
    """
    arr = np.asarray(z, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    ks_stat, p_value = stats.kstest(arr, "uniform")
    return float(ks_stat), float(p_value), bool(p_value < 0.05)


# ── Test 2: Pearson chi-squared uniformity ─────────────────────────────────────

def pit_chisq_test(z, n_bins: int = 10) -> tuple:
    """
    Pearson chi-squared test for Uniform(0, 1).

    Partitions [0, 1] into *n_bins* equal-width bins and tests whether the
    observed counts are consistent with the expected uniform frequency n/K.

    Under H0, the test statistic Σ (O_k − E_k)² / E_k ~ χ²(K − 1).

    Parameters
    ----------
    z      : array-like — PIT values.
    n_bins : int        — number of histogram bins (default 10).

    Returns
    -------
    chi2_stat : float
    p_value   : float
    reject    : bool  — True if H0 rejected at the 5 % level
    """
    arr = np.asarray(z, dtype=float)
    arr = arr[~np.isnan(arr)]
    n   = len(arr)
    if n < n_bins:
        return np.nan, np.nan, np.nan
    observed, _ = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
    expected    = np.full(n_bins, n / n_bins, dtype=float)
    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
    return float(chi2_stat), float(p_value), bool(p_value < 0.05)


# ── Tests 3 & 4: Ljung–Box independence ───────────────────────────────────────

def pit_lb_test(z, n_lags: int = 10) -> tuple:
    """
    Ljung–Box independence tests on the normal-probability transforms.

    Under H0, u_t = Φ⁻¹(z_t) ~ i.i.d. N(0, 1).  Two LB statistics are
    computed at *n_lags* lags:

    • LB on u_t   — detects first-moment serial dependence (mean clustering).
    • LB on u_t²  — detects second-moment dependence / ARCH-type effects.

    Both statistics follow χ²(n_lags) under H0.

    Parameters
    ----------
    z      : array-like — PIT values in (0, 1).
    n_lags : int        — number of autocorrelation lags (default 10).

    Returns
    -------
    lb_stat_u  : float — LB statistic for u_t
    p_lb_u     : float — p-value for LB(u_t)
    reject_u   : bool  — True if H0 rejected at 5 % for u_t
    lb_stat_u2 : float — LB statistic for u_t²
    p_lb_u2    : float — p-value for LB(u_t²)
    reject_u2  : bool  — True if H0 rejected at 5 % for u_t²
    """
    arr = np.asarray(z, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) <= n_lags + 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    u  = stats.norm.ppf(arr)    # Φ⁻¹(z_t)  ~ i.i.d. N(0,1) under H0
    u2 = u ** 2

    lb_u  = acorr_ljungbox(u,  lags=[n_lags], return_df=True)
    lb_u2 = acorr_ljungbox(u2, lags=[n_lags], return_df=True)

    stat_u  = float(lb_u["lb_stat"].iloc[-1])
    p_u     = float(lb_u["lb_pvalue"].iloc[-1])
    stat_u2 = float(lb_u2["lb_stat"].iloc[-1])
    p_u2    = float(lb_u2["lb_pvalue"].iloc[-1])

    return stat_u, p_u, bool(p_u < 0.05), stat_u2, p_u2, bool(p_u2 < 0.05)


# ── Aggregator ─────────────────────────────────────────────────────────────────

def run_pit_tests(
    port_ret: pd.Series,
    methods,
    n_bins: int = 10,
    n_lags: int = 10,
) -> tuple:
    """
    Run the full PIT test battery for every model in *methods*.

    Computes the PIT series for each model, then applies the KS uniformity
    test, chi-squared uniformity test, and Ljung–Box independence tests.

    Parameters
    ----------
    port_ret : pd.Series
        Portfolio daily log-return series indexed by pd.Timestamp.
    methods  : list of (tag, label, compute_fn, colour)
        Model registry — typically imported from ``q1_methods.METHODS``.
    n_bins   : int — chi-squared histogram bins (default 10).
    n_lags   : int — Ljung–Box lags (default 10).

    Returns
    -------
    results_df : pd.DataFrame
        One row per model with all test statistics and reject flags.
    pit_series : dict
        Mapping ``{tag: pd.Series}`` of z_t values for each model.
    """
    rows       = []
    pit_series = {}

    for tag, label, _, _ in methods:
        z = compute_pit_series(port_ret, tag)
        pit_series[tag] = z

        ks_stat,   p_ks,   rej_ks   = pit_ks_test(z)
        chi2_stat, p_chi2, rej_chi2 = pit_chisq_test(z, n_bins=n_bins)
        stat_u, p_u, rej_u, stat_u2, p_u2, rej_u2 = pit_lb_test(z, n_lags=n_lags)

        def _r(x):
            return round(float(x), 4) if not (x is None or np.isnan(x)) else np.nan

        rows.append({
            "Method":          label,
            "Tag":             tag,
            "N":               len(z),
            # Test 1 — KS uniformity
            "KS_stat":         _r(ks_stat),
            "p_KS":            _r(p_ks),
            "Reject_KS":       rej_ks,
            # Test 2 — Chi-squared uniformity
            "Chi2_stat":       _r(chi2_stat),
            "p_Chi2":          _r(p_chi2),
            "Reject_Chi2":     rej_chi2,
            # Test 3 — LB on u_t (first moment)
            "LB_u_stat":       _r(stat_u),
            "p_LB_u":          _r(p_u),
            "Reject_LB_u":     rej_u,
            # Test 4 — LB on u_t² (second moment / ARCH)
            "LB_u2_stat":      _r(stat_u2),
            "p_LB_u2":         _r(p_u2),
            "Reject_LB_u2":    rej_u2,
        })

    return pd.DataFrame(rows), pit_series


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    """
    Standalone entry point.

    Builds the portfolio, runs the PIT test battery for all four models, and
    saves the results table to Q1/output_q1_4/pit_results.csv.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from logger import setup_run_logger, get_logger, log_start, log_end
    from q1_1_build_portfolio import build_portfolio
    from q1_methods import METHODS
    from q1_4_pit_plots import generate_pit_plots

    setup_run_logger("smm272_q1_4_pit")
    log = get_logger("q1_4_pit")

    log_start(log, "Q1 Part 4 — PIT Test")

    _, _, port_ret = build_portfolio()
    results_df, pit_series = run_pit_tests(port_ret, METHODS)

    out_csv = os.path.join(Q1_4_OUTPUT_DIR, "pit_results.csv")
    results_df.to_csv(out_csv, index=False)
    log.info(f"PIT results saved → {out_csv}")

    generate_pit_plots(pit_series, results_df, METHODS)

    header = (
        f"{'Method':<30}  {'N':>4}  "
        f"{'KS p':>6}  {'Chi2 p':>6}  "
        f"{'LB-u p':>6}  {'LB-u² p':>7}"
    )
    log.info(header)
    log.info("-" * len(header))
    for _, row in results_df.iterrows():
        def _fmt(p, rej):
            if p != p:           # NaN check
                return f"{'N/A':>6}  "
            flag = "*" if rej else " "
            return f"{p:>6.3f}{flag} "

        log.info(
            f"{row['Method']:<30}  {int(row['N']):>4}  "
            f"{_fmt(row['p_KS'],    row['Reject_KS'])}"
            f"{_fmt(row['p_Chi2'],  row['Reject_Chi2'])}"
            f"{_fmt(row['p_LB_u'], row['Reject_LB_u'])}"
            f"{_fmt(row['p_LB_u2'],row['Reject_LB_u2'])}"
        )

    log_end(log, "Q1 Part 4 — PIT Test")
    return results_df, pit_series


if __name__ == "__main__":
    main()
