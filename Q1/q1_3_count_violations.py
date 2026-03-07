"""
Q1 Part 3 — VaR violation counting.

For each (method, confidence level) pair, counts the number of days on which
the realised portfolio return fell below the rolling VaR forecast.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def count_violations(df, confidence_levels, methods):
    """
    Count VaR violations for every (method, confidence level) combination.

    Parameters
    ----------
    df                : pd.DataFrame — output of run_rolling_all_levels().
                        Must contain 'Actual_Return' and '<tag>_VaR_<ci_pct>' columns.
    confidence_levels : list of float — e.g. [0.90, 0.95, 0.99]
    methods           : list of (tag, label, fn, colour)

    Returns
    -------
    violations_df   : pd.DataFrame — one row per (method, CI) with counts and rates
    violation_flags : pd.DataFrame — daily 0/1 flags per (method, CI), indexed like df
    """
    violation_rows = []
    flag_cols      = {}

    for tag, label, _, _ in methods:
        for ci in confidence_levels:
            ci_pct = int(round(ci * 100))
            col    = f"{tag}_VaR_{ci_pct}"
            p0     = 1.0 - ci

            valid = df[[col, "Actual_Return"]].dropna()
            n     = len(valid)
            viol  = (valid["Actual_Return"] < valid[col]).astype(int)
            k     = int(viol.sum())
            rate  = k / n if n > 0 else np.nan
            exp_k = p0 * n

            flag_cols[f"{tag}_{ci_pct}"] = viol.reindex(df.index, fill_value=np.nan)

            violation_rows.append({
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

    violations_df   = pd.DataFrame(violation_rows)
    violation_flags = pd.DataFrame(flag_cols, index=df.index)
    return violations_df, violation_flags
