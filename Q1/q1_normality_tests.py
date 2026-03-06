"""
Q1 – Step 4: Normality tests on portfolio returns.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy import stats
from q1_build_portfolio import build_portfolio


def run_normality_tests(portfolio_returns=None):
    """Run four normality tests on the portfolio return series."""
    if portfolio_returns is None:
        _, _, portfolio_returns = build_portfolio()

    print("\n" + "-" * 70)
    print("NORMALITY TESTS")
    print("-" * 70)

    results = {}

    # 1. Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(portfolio_returns)
    results["Jarque-Bera"] = (jb_stat, jb_pval)
    print(f"\n  Jarque-Bera test statistic : {jb_stat:.4f}")
    print(f"  Jarque-Bera p-value        : {jb_pval:.6e}")
    print(f"  → {'Reject' if jb_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # 2. Shapiro-Wilk (subsample if > 5 000 observations)
    sample = (portfolio_returns if len(portfolio_returns) <= 5000
              else portfolio_returns.sample(5000, random_state=42))
    sw_stat, sw_pval = stats.shapiro(sample)
    results["Shapiro-Wilk"] = (sw_stat, sw_pval)
    print(f"\n  Shapiro-Wilk test statistic: {sw_stat:.6f}")
    print(f"  Shapiro-Wilk p-value       : {sw_pval:.6e}")
    print(f"  → {'Reject' if sw_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # 3. D'Agostino-Pearson K²
    dp_stat, dp_pval = stats.normaltest(portfolio_returns)
    results["DAgostino-Pearson"] = (dp_stat, dp_pval)
    print(f"\n  D'Agostino-Pearson K² stat : {dp_stat:.4f}")
    print(f"  D'Agostino-Pearson p-value : {dp_pval:.6e}")
    print(f"  → {'Reject' if dp_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    # 4. Kolmogorov-Smirnov (against fitted normal)
    ks_stat, ks_pval = stats.kstest(
        portfolio_returns, "norm",
        args=(portfolio_returns.mean(), portfolio_returns.std())
    )
    results["Kolmogorov-Smirnov"] = (ks_stat, ks_pval)
    print(f"\n  Kolmogorov-Smirnov stat    : {ks_stat:.6f}")
    print(f"  Kolmogorov-Smirnov p-value : {ks_pval:.6e}")
    print(f"  → {'Reject' if ks_pval < 0.05 else 'Fail to reject'} normality at 5% level")

    return results


if __name__ == "__main__":
    run_normality_tests()
