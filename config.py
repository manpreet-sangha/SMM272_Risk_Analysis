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

# ── Project Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
Q1_DIR = os.path.join(ROOT_DIR, "Q1")
Q1_OUTPUT_DIR = os.path.join(Q1_DIR, "output")

# Create output directories if they don't exist
os.makedirs(Q1_OUTPUT_DIR, exist_ok=True)
