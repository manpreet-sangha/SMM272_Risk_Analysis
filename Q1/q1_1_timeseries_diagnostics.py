"""
Q1 : Time-series diagnostics.

Covers: Ljung-Box autocorrelation tests on raw and squared returns,
Augmented Dickey-Fuller (ADF) stationarity test, and KPSS stationarity test.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from config import Q1_1_OUTPUT_DIR
from q1_1_build_portfolio import build_portfolio
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_timeseries_diagnostics")


def run_timeseries_diagnostics(portfolio_returns=None):
    """
    Run autocorrelation (Ljung-Box) and stationarity (ADF, KPSS) tests.

    Returns
    -------
    lb_results : pd.DataFrame or None
        Ljung-Box results on raw returns.
    lb_sq_results : pd.DataFrame or None
        Ljung-Box results on squared returns.
    adf_result : dict
        ADF test outputs (stat, pval, crit).
    kpss_result : dict
        KPSS test outputs (stat, pval, crit).
    """
    if portfolio_returns is None:
        _, _, portfolio_returns = build_portfolio()

    log_start(logger, "q1_1_timeseries_diagnostics.py")

    ret_arr = portfolio_returns.dropna().to_numpy()
    max_lag = min(20, len(ret_arr) // 2 - 1)
    lags    = [l for l in [5, 10, 15, 20] if l <= max_lag]

    # ── Ljung-Box on raw returns ──────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("AUTOCORRELATION — LJUNG-BOX TEST (RAW RETURNS)")
    logger.info("-" * 70)

    if max_lag < 1:
        logger.warning("Insufficient observations for Ljung-Box test – skipping.")
        lb_results = lb_sq_results = None
    else:
        lb_results = acorr_ljungbox(ret_arr, lags=lags, return_df=True)
        logger.info(lb_results.to_string())

        # ── Ljung-Box on squared returns (ARCH / volatility clustering) ───
        logger.info("-" * 70)
        logger.info("AUTOCORRELATION — LJUNG-BOX TEST (SQUARED RETURNS)")
        logger.info("-" * 70)
        lb_sq_results = acorr_ljungbox(ret_arr ** 2, lags=lags, return_df=True)
        logger.info(lb_sq_results.to_string())
        logger.info("  (Significant autocorrelation in squared returns → ARCH effects / "
                    "volatility clustering)")

    # ── Augmented Dickey-Fuller stationarity test ─────────────────────────
    logger.info("-" * 70)
    logger.info("STATIONARITY — AUGMENTED DICKEY-FULLER (ADF) TEST")
    logger.info("-" * 70)
    logger.info("  H₀: unit root present (non-stationary)  |  H₁: stationary")

    adf_stat, adf_pval, adf_lags_used, _, adf_crit, _ = adfuller(ret_arr, autolag="AIC")
    logger.info(f"  ADF test statistic : {adf_stat:.6f}")
    logger.info(f"  p-value            : {adf_pval:.6e}")
    logger.info(f"  Lags used (AIC)    : {adf_lags_used}")
    for level, cv in adf_crit.items():
        logger.info(f"  Critical value ({level}): {cv:.4f}")
    conclusion = "stationary" if adf_pval < 0.05 else "non-stationary"
    logger.info(f"  → {'Reject' if adf_pval < 0.05 else 'Fail to reject'} H₀ at 5% "
                f"— series is {conclusion}")

    # ── KPSS stationarity test ────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STATIONARITY — KPSS TEST")
    logger.info("-" * 70)
    logger.info("  H₀: stationary  |  H₁: unit root present (non-stationary)")

    kpss_stat, kpss_pval, kpss_lags_used, kpss_crit = kpss(
        ret_arr, regression="c", nlags="auto")
    logger.info(f"  KPSS test statistic: {kpss_stat:.6f}")
    logger.info(f"  p-value            : {kpss_pval:.6e}")
    logger.info(f"  Lags used          : {kpss_lags_used}")
    for level, cv in kpss_crit.items():
        logger.info(f"  Critical value ({level}): {cv:.4f}")
    kpss_conclusion = "non-stationary" if kpss_pval < 0.05 else "stationary"
    logger.info(f"  → {'Reject' if kpss_pval < 0.05 else 'Fail to reject'} H₀ at 5% "
                f"— series is {kpss_conclusion}")

    logger.info("-" * 70)
    logger.info("STATIONARITY INTERPRETATION")
    logger.info("-" * 70)
    if adf_pval < 0.05 and kpss_pval >= 0.05:
        logger.info("  Both ADF and KPSS agree: portfolio returns are STATIONARY.")
    elif adf_pval >= 0.05 and kpss_pval < 0.05:
        logger.info("  Both ADF and KPSS agree: portfolio returns are NON-STATIONARY.")
    else:
        logger.info("  ADF and KPSS results are conflicting — inconclusive.")

    log_end(logger, "q1_1_timeseries_diagnostics.py")

    adf_result  = {"stat": adf_stat,  "pval": adf_pval,  "crit": adf_crit}
    kpss_result = {"stat": kpss_stat, "pval": kpss_pval, "crit": kpss_crit}
    return lb_results, lb_sq_results, adf_result, kpss_result


if __name__ == "__main__":
    setup_run_logger("smm272_q1_timeseries_diagnostics")
    run_timeseries_diagnostics()
