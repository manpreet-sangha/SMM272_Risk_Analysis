"""
Q1 : Normality tests on portfolio returns.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy import stats
from q1_1_build_portfolio import build_portfolio
from logger import setup_run_logger, get_logger, log_start, log_end

logger = get_logger("q1_1_normality_tests")


def run_normality_tests(portfolio_returns=None):
    """Run four normality tests on the portfolio return series."""
    log_start(logger, "q1_1_normality_tests.py")
    if portfolio_returns is None:
        _, _, portfolio_returns = build_portfolio()

    logger.info("-" * 70)
    logger.info("NORMALITY TESTS")
    logger.info("-" * 70)

    results = {}

    # 1. Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(portfolio_returns)
    results["Jarque-Bera"] = (jb_stat, jb_pval)
    logger.info(f"  Jarque-Bera test statistic : {jb_stat:.4f}")
    logger.info(f"  Jarque-Bera p-value        : {jb_pval:.6e}")
    logger.info(f"  → {'Reject' if jb_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # 2. Shapiro-Wilk (subsample if > 5 000 observations)
    sample = (portfolio_returns if len(portfolio_returns) <= 5000
              else portfolio_returns.sample(5000, random_state=42))
    sw_stat, sw_pval = stats.shapiro(sample)
    results["Shapiro-Wilk"] = (sw_stat, sw_pval)
    logger.info(f"  Shapiro-Wilk test statistic: {sw_stat:.6f}")
    logger.info(f"  Shapiro-Wilk p-value       : {sw_pval:.6e}")
    logger.info(f"  → {'Reject' if sw_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # 3. D'Agostino-Pearson K²
    dp_stat, dp_pval = stats.normaltest(portfolio_returns)
    results["DAgostino-Pearson"] = (dp_stat, dp_pval)
    logger.info(f"  D'Agostino-Pearson K² stat : {dp_stat:.4f}")
    logger.info(f"  D'Agostino-Pearson p-value : {dp_pval:.6e}")
    logger.info(f"  → {'Reject' if dp_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # 4. Kolmogorov-Smirnov (against fitted normal)
    ks_stat, ks_pval = stats.kstest(
        portfolio_returns, "norm",
        args=(portfolio_returns.mean(), portfolio_returns.std())
    )
    results["Kolmogorov-Smirnov"] = (ks_stat, ks_pval)
    logger.info(f"  Kolmogorov-Smirnov stat    : {ks_stat:.6f}")
    logger.info(f"  Kolmogorov-Smirnov p-value : {ks_pval:.6e}")
    logger.info(f"  → {'Reject' if ks_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # ── Distribution fitting ──────────────────────────────────────────────
    import numpy as np
    logger.info("\n" + "-" * 70)
    logger.info("DISTRIBUTION FITTING")
    logger.info("-" * 70)

    ret_arr = portfolio_returns.to_numpy()
    n = len(ret_arr)

    # Fit Normal distribution
    mu_fit, sigma_fit = stats.norm.fit(ret_arr)
    logger.info(f"  Normal fit      : mu = {mu_fit:.6f},  sigma = {sigma_fit:.6f}")

    # Fit Student-t distribution (heavy-tailed alternative)
    nu_fit, loc_fit, scale_fit = stats.t.fit(ret_arr, floc=mu_fit)
    logger.info(f"  Student-t fit   : df = {nu_fit:.4f},  loc = {loc_fit:.6f},  "
                f"scale = {scale_fit:.6f}")
    logger.info(f"  (df < 10 → fat tails; estimated df = {nu_fit:.2f})")

    # AIC / BIC comparison (lower is better)
    ll_norm = stats.norm.logpdf(ret_arr, mu_fit, sigma_fit).sum()
    aic_norm = -2 * ll_norm + 2 * 2          # 2 parameters
    bic_norm = -2 * ll_norm + np.log(n) * 2

    ll_t    = stats.t.logpdf(ret_arr, nu_fit, loc_fit, scale_fit).sum()
    aic_t   = -2 * ll_t + 2 * 3             # 3 parameters
    bic_t   = -2 * ll_t + np.log(n) * 3

    logger.info(f"  Normal    : AIC = {aic_norm:.2f},  BIC = {bic_norm:.2f}")
    logger.info(f"  Student-t : AIC = {aic_t:.2f},  BIC = {bic_t:.2f}")
    preferred = "Student-t" if aic_t < aic_norm else "Normal"
    logger.info(f"  → {preferred} distribution preferred by AIC")

    results["Normal_fit"]     = {"mu": mu_fit, "sigma": sigma_fit,
                                  "AIC": aic_norm, "BIC": bic_norm}
    results["StudentT_fit"]   = {"df": nu_fit, "loc": loc_fit,
                                  "scale": scale_fit, "AIC": aic_t, "BIC": bic_t}

    log_end(logger, "q1_1_normality_tests.py")
    return results


if __name__ == "__main__":
    setup_run_logger("smm272_q1_normality_tests")
    run_normality_tests()
