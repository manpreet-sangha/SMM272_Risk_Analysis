"""
Q1 Part 3 — Violations counting and statistical backtests.

Runs Kupiec POF and Christoffersen independence tests for every
(method, confidence level) combination.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from q1_3_kupiec           import kupiec_pof
from q1_3_christoffersen   import christoffersen_independence


def run_all_tests(df, confidence_levels, methods):
    """
    Compute violations, Kupiec POF, and Christoffersen independence tests
    for every (method, confidence level) combination.

    Parameters
    ----------
    df               : pd.DataFrame — must contain Actual_Return and
                       <tag>_VaR_<ci_pct> columns (output of run_rolling_all_levels)
    confidence_levels: list of float — e.g. [0.90, 0.95, 0.99]
    methods          : list of (tag, label, fn, colour)

    Returns
    -------
    violations_df   : pd.DataFrame — violation counts and rates
    kupiec_df       : pd.DataFrame — Kupiec POF test results
    christoff_df    : pd.DataFrame — Christoffersen independence and CC results
    violation_flags : pd.DataFrame — daily 0/1 flags for each (method, CI) pair
    """
    violations_rows = []
    kupiec_rows     = []
    christoff_rows  = []
    flag_cols       = {}

    for tag, label, _, _ in methods:
        for ci in confidence_levels:
            ci_pct = int(round(ci * 100))
            col    = f"{tag}_VaR_{ci_pct}"
            p0     = 1.0 - ci          # nominal violation probability

            valid = df[[col, "Actual_Return"]].dropna()
            n     = len(valid)
            viol  = (valid["Actual_Return"] < valid[col]).astype(int)
            k     = int(viol.sum())
            rate  = k / n if n > 0 else np.nan
            exp_k = p0 * n

            # Store violation flag series aligned to full index
            flag_cols[f"{tag}_{ci_pct}"] = viol.reindex(df.index, fill_value=np.nan)

            violations_rows.append({
                "Method":            label,
                "Tag":               tag,
                "Confidence (%)":    ci * 100,
                "Nominal Tail (%)":  p0 * 100,
                "N":                 n,
                "Violations (k)":    k,
                "Expected (\u2248)": round(exp_k, 1),
                "Observed Rate (%)": round(rate * 100, 4),
                "Nominal Rate (%)":  round(p0 * 100, 4),
                "Excess (pp)":       round((rate - p0) * 100, 4),
            })

            # Kupiec POF test
            lr_uc, p_uc, rej_uc = kupiec_pof(k, n, p0)
            kupiec_rows.append({
                "Method":         label,
                "Tag":            tag,
                "Confidence (%)": ci * 100,
                "Violations (k)": k,
                "N":              n,
                "LR_uc":          round(lr_uc, 4) if not np.isnan(lr_uc) else np.nan,
                "p-value":        round(p_uc,  4) if not np.isnan(p_uc)  else np.nan,
                "Reject H0 (5%)": rej_uc,
            })

            # Christoffersen independence test
            lr_ind, p_ind = christoffersen_independence(viol.values)

            # Joint conditional coverage (CC): LR_cc = LR_uc + LR_ind ~ chi^2(2)
            if not (np.isnan(lr_uc) or np.isnan(lr_ind)):
                lr_cc  = lr_uc + lr_ind
                p_cc   = float(stats.chi2.sf(lr_cc, df=2))
                rej_cc = p_cc < 0.05
            else:
                lr_cc, p_cc, rej_cc = np.nan, np.nan, np.nan

            christoff_rows.append({
                "Method":             label,
                "Tag":                tag,
                "Confidence (%)":     ci * 100,
                "LR_ind":             round(lr_ind, 4) if not np.isnan(lr_ind) else np.nan,
                "p-value (ind)":      round(p_ind,  4) if not np.isnan(p_ind)  else np.nan,
                "Reject Indep (5%)":  (p_ind < 0.05) if not np.isnan(p_ind) else np.nan,
                "LR_cc":              round(lr_cc, 4) if not np.isnan(lr_cc)  else np.nan,
                "p-value (CC)":       round(p_cc,  4) if not np.isnan(p_cc)   else np.nan,
                "Reject CC (5%)":     rej_cc,
            })

    violations_df   = pd.DataFrame(violations_rows)
    kupiec_df       = pd.DataFrame(kupiec_rows)
    christoff_df    = pd.DataFrame(christoff_rows)
    violation_flags = pd.DataFrame(flag_cols, index=df.index)

    return violations_df, kupiec_df, christoff_df, violation_flags
