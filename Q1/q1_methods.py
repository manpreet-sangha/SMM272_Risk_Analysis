"""
Q1 — Canonical METHODS registry.

Combines the static metadata from config.VAR_METHODS_META with the
compute functions from each VaR module to produce the single authoritative
METHODS list used by q1_2_statistical_analysis and q1_3_var_violations.

Import pattern
--------------
    from q1_methods import METHODS
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import VAR_METHODS_META
from q1_2_var_historical import compute_historical_var_es
from q1_2_var_normal     import compute_normal_var_es
from q1_2_var_studentt   import compute_studentt_var_es
from q1_2_var_garch      import compute_garch_var_es

_FN_MAP = {
    "HS":       compute_historical_var_es,
    "Normal":   compute_normal_var_es,
    "StudentT": compute_studentt_var_es,
    "GARCH":    compute_garch_var_es,
}

# (tag, label, compute_fn, colour)
METHODS = [(tag, lbl, _FN_MAP[tag], col) for tag, lbl, col in VAR_METHODS_META]
