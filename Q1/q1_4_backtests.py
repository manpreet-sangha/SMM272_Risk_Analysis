"""
Q1 Part 4 — Backtesting aggregator.

Runs all four statistical backtests for every (method, confidence level) pair:

  1. Kupiec (1995) POF test           — unconditional coverage
  2. Christoffersen (1998) CC test    — independence + joint conditional coverage
  3. Christoffersen-Pelletier (2004) duration test — Weibull-based clustering test
  4. Engle-Manganelli (2004) DQ test  — Dynamic Quantile regression test

The Christoffersen conditional coverage (CC) test combines the Kupiec
unconditional coverage test and the Christoffersen independence test into a
joint LR statistic: LR_cc = LR_uc + LR_ind ~ chi^2(2).
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from q1_4_kupiec          import kupiec_pof
from q1_4_christoffersen  import christoffersen_independence
from q1_4_duration        import duration_test
from q1_4_dq_test         import dq_test


def run_all_backtests(violations_df, violation_flags, confidence_levels, methods,
                      n_lags: int = 4):
    """
    Run all four backtests for every (method, confidence level) combination.

    Parameters
    ----------
    violations_df    : pd.DataFrame — from q1_3_count_violations (k, n, rate, ...)
    violation_flags  : pd.DataFrame — daily 0/1 flag series per (tag, ci_pct)
    confidence_levels: list of float
    methods          : list of (tag, label, fn, colour)
    n_lags           : int — lags for DQ test (default 4)

    Returns
    -------
    results_df : pd.DataFrame — one row per (method, CI), all test statistics
    """
    rows = []

    for tag, label, _, _ in methods:
        for ci in confidence_levels:
            ci_pct = int(round(ci * 100))
            p0     = 1.0 - ci

            # ── Pull violation counts from Part 3 summary ─────────────────
            mask = (
                (violations_df["Tag"]             == tag) &
                (violations_df["Confidence (%)"]  == ci * 100)
            )
            if mask.sum() == 0:
                continue
            row = violations_df[mask].iloc[0]
            k   = int(row["Violations (k)"])
            n   = int(row["N"])

            # ── Daily violation flag series ───────────────────────────────
            flag_col   = f"{tag}_{ci_pct}"
            viol_array = (
                violation_flags[flag_col].dropna().astype(int).values
                if flag_col in violation_flags.columns else np.array([])
            )

            # ── 1. Kupiec POF test ────────────────────────────────────────
            lr_uc, p_uc, rej_uc = kupiec_pof(k, n, p0)

            # ── 2. Christoffersen independence test ───────────────────────
            lr_ind, p_ind = christoffersen_independence(viol_array)
            rej_ind = (p_ind < 0.05) if not np.isnan(p_ind) else np.nan

            # Joint CC: LR_cc = LR_uc + LR_ind ~ chi^2(2)
            if not (np.isnan(lr_uc) or np.isnan(lr_ind)):
                lr_cc  = lr_uc + lr_ind
                p_cc   = float(stats.chi2.sf(lr_cc, df=2))
                rej_cc = p_cc < 0.05
            else:
                lr_cc, p_cc, rej_cc = np.nan, np.nan, np.nan

            # ── 3. Duration test (Christoffersen-Pelletier) ───────────────
            lr_dur, p_dur, rej_dur = duration_test(viol_array, p0)

            # ── 4. DQ test (Engle-Manganelli) ─────────────────────────────
            dq_stat, p_dq, rej_dq = dq_test(viol_array, p0, n_lags=n_lags)

            def _r(x, decimals=4):
                return round(float(x), decimals) if not np.isnan(x) else np.nan

            rows.append({
                "Method":             label,
                "Tag":                tag,
                "Confidence (%)":     ci * 100,
                "Nominal Tail (%)":   p0 * 100,
                "N":                  n,
                "Violations (k)":     k,
                # Kupiec POF
                "LR_uc":              _r(lr_uc),
                "p_uc":               _r(p_uc),
                "Reject_uc":          rej_uc,
                # Christoffersen independence
                "LR_ind":             _r(lr_ind),
                "p_ind":              _r(p_ind),
                "Reject_ind":         rej_ind,
                # Conditional coverage
                "LR_cc":              _r(lr_cc),
                "p_cc":               _r(p_cc),
                "Reject_cc":          rej_cc,
                # Duration test
                "LR_dur":             _r(lr_dur),
                "p_dur":              _r(p_dur),
                "Reject_dur":         rej_dur,
                # DQ test
                "DQ_stat":            _r(dq_stat),
                "p_dq":               _r(p_dq),
                "Reject_dq":          rej_dq,
            })

    return pd.DataFrame(rows)
