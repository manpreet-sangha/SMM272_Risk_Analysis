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
Q1_4_OUTPUT_DIR = os.path.join(Q1_DIR, "output_q1_4")

# Create output directories if they don't exist
os.makedirs(Q1_1_OUTPUT_DIR, exist_ok=True)
os.makedirs(Q1_2_OUTPUT_DIR, exist_ok=True)
os.makedirs(Q1_3_OUTPUT_DIR, exist_ok=True)
os.makedirs(Q1_4_OUTPUT_DIR, exist_ok=True)

# ── Q2 — Power of the Kupiec Test ─────────────────────────────────────────────
Q2_DIR        = os.path.join(ROOT_DIR, "Q2")
Q2_OUTPUT_DIR = os.path.join(Q2_DIR, "output_q2")
os.makedirs(Q2_OUTPUT_DIR, exist_ok=True)

# Confidence levels examined in Q2 (same as Q1 Part 3)
Q2_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Monte Carlo replications for power estimation
Q2_MC_REPS = 10_000

# Random seed for reproducibility
Q2_RANDOM_SEED = 42

# Sample sizes for the power-vs-T analysis
Q2_T_GRID = [125, 250, 500, 750, 1000, 1500, 2000, 2893]

# GARCH persistence grid for sensitivity analysis  (α + β values)
Q2_PERSISTENCE_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]

# ── Q5 — SPAN Margining System ────────────────────────────────────────────────
Q5_DIR        = os.path.join(ROOT_DIR, "Q5")
Q5_OUTPUT_DIR = os.path.join(Q5_DIR, "output_q5")
os.makedirs(Q5_OUTPUT_DIR, exist_ok=True)

# ── Q6 — VaR of a Portfolio with Options ──────────────────────────────────────
Q6_DIR        = os.path.join(ROOT_DIR, "Q6")
Q6_OUTPUT_DIR = os.path.join(Q6_DIR, "output_q6")
os.makedirs(Q6_OUTPUT_DIR, exist_ok=True)

Q6_TICKERS          = ["INTC", "JPM", "AA", "PG"]
Q6_REFERENCE_DATE   = "2026-02-12"   # portfolio valuation date
Q6_HIST_START       = "2024-02-12"   # 2 years of history
Q6_RISK_FREE_RATE   = 0.04           # 4.0 % p.a. (annualised)
Q6_N_SIMS           = 10_000         # Monte Carlo replications
Q6_HORIZON_DAYS     = 10             # 10-trading-day VaR horizon
Q6_CONFIDENCE       = 0.99           # 99 % confidence level
Q6_EWMA_LAMBDA      = 0.94           # RiskMetrics EWMA decay factor
Q6_RANDOM_SEED      = 42             # reproducibility seed
Q6_MULTIPLIER       = 100            # shares per standard equity option contract
