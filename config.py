"""
SMM272 Risk Analysis Coursework 2025-2026
Global configuration — shared constants and paths.
"""

import os

# ── Tickers & Date Range ───────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "IBM", "NVDA", "GOOGL", "AMZN"]
START_DATE = "2014-01-01"
END_DATE = "2025-12-31"

# ── Annualisation ──────────────────────────────────────────────────────────────
TRADING_DAYS = 252

# ── Crisis Window (COVID-19) ───────────────────────────────────────────────────
CRISIS_START = "2020-01-01"
CRISIS_END   = "2020-12-31"

# ── Q1 Part 2 — Rolling VaR / ES ──────────────────────────────────────────────
ROLLING_WINDOW_MONTHS = 6
ROLLING_START_DATE    = "2014-07-01"
VAR_CONFIDENCE_LEVEL  = 0.99      # 99 % confidence → 1 % tail

# ── Q1 Part 3 — VaR Violations at multiple confidence levels ──────────────────
Q1_3_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]   # 10 %, 5 %, 1 % nominal tails
# ── VaR method metadata (tag, label, colour) — functions added in q1_methods.py ──
# Defined here so any script can read tags/labels/colours without importing Q1 code.
VAR_METHODS_META = [
    ("HS",       "Historical Simulation", "#1f77b4"),
    ("Normal",   "Parametric Normal",     "#ff7f0e"),
    ("StudentT", "Parametric Student-t",  "#2ca02c"),
    ("GARCH",    "GARCH(1,1) Normal",     "#d62728"),
]
# ── Project Paths ──────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
Q1_DIR        = os.path.join(ROOT_DIR, "Q1")
Q1_1_OUTPUT_DIR = os.path.join(Q1_DIR, "output_q1_1")
Q1_2_OUTPUT_DIR = os.path.join(Q1_DIR, "output_q1_2")
Q1_3_OUTPUT_DIR = os.path.join(Q1_DIR, "output_q1_3")

# Create output directories if they don't exist
os.makedirs(Q1_1_OUTPUT_DIR, exist_ok=True)
os.makedirs(Q1_2_OUTPUT_DIR, exist_ok=True)
os.makedirs(Q1_3_OUTPUT_DIR, exist_ok=True)
