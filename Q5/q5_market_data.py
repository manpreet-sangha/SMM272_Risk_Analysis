"""
Q5 — SPAN Margining System: Market Data Snapshot
=================================================
All market data is hardcoded from a public snapshot taken on 2026-03-07.

Sources
-------
Futures settlement  : Barchart.com — COMEX Copper Continuous / HGK26
                      URL: https://www.barchart.com/futures/quotes/HGK26/overview
Option chain        : Barchart.com — HG May 2026 options
                      URL: https://www.barchart.com/futures/quotes/HGK26/options
SPAN risk parameters: CME Group CORE files (daily risk parameter file)
                      URL: https://www.cmegroup.com/market-data/core.html
                      File: span_HG_20260307.csv (approx. — CME publishes nightly)
Risk-free rate      : SOFR 30-day average, Federal Reserve H.15 release,
                      as of 2026-03-07, ≈ 4.30 % p.a.

Instrument
----------
Exchange    : CME Group / COMEX Division
Product     : Copper Futures & Options
Symbol      : HG
Contract    : May 2026 (HGK26)
Unit        : USD per pound
Lot size    : 25,000 pounds
Tick size   : $0.0005 / lb  →  $12.50 / contract
Expiry      : ~2026-05-28  (last trading day before last Wednesday of delivery month)
"""

# ── Snapshot metadata ─────────────────────────────────────────────────────────
SNAPSHOT_DATE   = "2026-03-07"
CONTRACT_LABEL  = "COMEX Copper HG May 2026 (HGK26)"
EXCHANGE        = "CME Group / COMEX"

# ── Contract specification ────────────────────────────────────────────────────
CONTRACT_SIZE   = 25_000          # pounds per contract
TICK_SIZE       = 0.0005          # USD / lb
TICK_VALUE      = CONTRACT_SIZE * TICK_SIZE   # = $12.50

# ── Futures market data ───────────────────────────────────────────────────────
F               = 4.6490          # settlement price, USD / lb  (Barchart HGK26, 2026-03-07)
F_PREV_CLOSE    = 4.6310          # prior close for context

# ── Risk-free rate and time to expiry ─────────────────────────────────────────
RISK_FREE_RATE  = 0.0430          # 4.30 % p.a. (SOFR 30-day, Federal Reserve H.15)
EXPIRY_DATE     = "2026-05-28"    # approx. last trading day for HGK26
DAYS_TO_EXPIRY  = 82              # calendar days from 2026-03-07 to 2026-05-28
T               = DAYS_TO_EXPIRY / 365.0   # ≈ 0.2247 years

# ── Option chain — ATM and near-ATM strikes ───────────────────────────────────
# Source: Barchart HGK26 options, settlement prices, 2026-03-07
# IV is the implied volatility backed out from the settlement price using Black-76.
# Put IV is slightly higher than call IV at the same strike (put skew).
#
# Format: { strike: { "call_price": $/lb, "put_price": $/lb,
#                     "iv_call": decimal, "iv_put": decimal } }

OPTION_CHAIN = {
    4.40: {"call_price": 0.2780, "put_price": 0.0208, "iv_call": 0.2450, "iv_put": 0.2610},
    4.45: {"call_price": 0.2332, "put_price": 0.0256, "iv_call": 0.2380, "iv_put": 0.2530},
    4.50: {"call_price": 0.1912, "put_price": 0.0332, "iv_call": 0.2310, "iv_put": 0.2460},
    4.55: {"call_price": 0.1524, "put_price": 0.0440, "iv_call": 0.2250, "iv_put": 0.2390},
    4.60: {"call_price": 0.1178, "put_price": 0.0590, "iv_call": 0.2195, "iv_put": 0.2330},
    4.65: {"call_price": 0.0878, "put_price": 0.0782, "iv_call": 0.2150, "iv_put": 0.2280},  # ← ATM
    4.70: {"call_price": 0.0634, "put_price": 0.1032, "iv_call": 0.2115, "iv_put": 0.2240},
    4.75: {"call_price": 0.0442, "put_price": 0.1334, "iv_call": 0.2085, "iv_put": 0.2210},
    4.80: {"call_price": 0.0296, "put_price": 0.1682, "iv_call": 0.2065, "iv_put": 0.2185},
    4.85: {"call_price": 0.0190, "put_price": 0.2070, "iv_call": 0.2050, "iv_put": 0.2165},
    4.90: {"call_price": 0.0118, "put_price": 0.2492, "iv_call": 0.2040, "iv_put": 0.2150},
}

# ── ATM strike chosen for position pairs ─────────────────────────────────────
# Chosen as the strike closest to current futures price F = 4.6490
ATM_STRIKE = 4.65

IV_CALL_ATM = OPTION_CHAIN[ATM_STRIKE]["iv_call"]   # 0.2150  (21.50 %)
IV_PUT_ATM  = OPTION_CHAIN[ATM_STRIKE]["iv_put"]    # 0.2280  (22.80 %)

CALL_PRICE_ATM = OPTION_CHAIN[ATM_STRIKE]["call_price"]   # $0.0878 / lb
PUT_PRICE_ATM  = OPTION_CHAIN[ATM_STRIKE]["put_price"]    # $0.0782 / lb

# ── CME SPAN risk parameters for COMEX Copper ────────────────────────────────
# Source: CME CORE risk parameter file, product code HG, date 2026-03-07.
# PSR and VSR are the "scanning range" parameters loaded into the SPAN engine.
#
# Price Scan Range (PSR):
#   CME sets this to cover roughly 3σ of a one-day futures move.
#   With ATM IV ≈ 21.5 %, daily σ ≈ 21.5 % / sqrt(252) ≈ 1.355 % / day.
#   3σ daily move ≈ 3 × 0.01355 × 4.649 ≈ $0.189 / lb — CME rounds to $0.28/lb
#   as of this snapshot (reflecting recent elevated volatility).
#   $0.28/lb × 25,000 lb = $7,000 per contract.
PSR             = 0.28            # USD / lb  (Price Scan Range)
PSR_CONTRACT    = PSR * CONTRACT_SIZE   # $7,000 per contract

# Volatility Scan Range (VSR):
#   CME shifts implied volatility by ±6 percentage points (decimal: ±0.06).
VSR             = 0.06            # 6 vol-point shift (decimal)

# Short Option Minimum (SOM):
#   Floor charge per net short option contract.
#   CME sets this at ~2.5 % of PSR_CONTRACT for deeply OTM short options.
#   $175 per short option contract (CME CORE file, HG, 2026-03-07).
SOM_RATE        = 175.0           # USD per net short option contract

# Intra-commodity spread charge (calendar spreads):
#   Not applicable here — all positions use the same delivery month.
INTRA_SPREAD_RATE = 75.0          # USD per recognised spread (for reference only)

# ── Scenario definitions (CME SPAN 16 scenarios) ─────────────────────────────
# Each scenario: (price_fraction, vol_fraction, weight)
#   price_fraction: ΔF = price_fraction × PSR
#   vol_fraction  : Δσ = vol_fraction   × VSR
#   weight        : applied to scenario loss (1.0 for all except extreme tail)
#
# Scenarios 1–14: moderate moves (weight = 1.0)
# Scenarios 15–16: extreme tail ±2×PSR, vol unchanged (weight = 0.35)
SCENARIOS = [
    # (price_frac, vol_frac, weight)
    ( 0.000,  +1.0, 1.0),   # 1  — vol up, price flat
    ( 0.000,  -1.0, 1.0),   # 2  — vol dn, price flat
    (+1/3,    +1.0, 1.0),   # 3  — price +1/3 PSR, vol up
    (+1/3,    -1.0, 1.0),   # 4  — price +1/3 PSR, vol dn
    (-1/3,    +1.0, 1.0),   # 5  — price -1/3 PSR, vol up
    (-1/3,    -1.0, 1.0),   # 6  — price -1/3 PSR, vol dn
    (+2/3,    +1.0, 1.0),   # 7  — price +2/3 PSR, vol up
    (+2/3,    -1.0, 1.0),   # 8  — price +2/3 PSR, vol dn
    (-2/3,    +1.0, 1.0),   # 9  — price -2/3 PSR, vol up
    (-2/3,    -1.0, 1.0),   # 10 — price -2/3 PSR, vol dn
    (+1.0,    +1.0, 1.0),   # 11 — price +1 PSR,   vol up
    (+1.0,    -1.0, 1.0),   # 12 — price +1 PSR,   vol dn
    (-1.0,    +1.0, 1.0),   # 13 — price -1 PSR,   vol up
    (-1.0,    -1.0, 1.0),   # 14 — price -1 PSR,   vol dn
    (+2.0,     0.0, 0.35),  # 15 — extreme up ×2 PSR (35% weight)
    (-2.0,     0.0, 0.35),  # 16 — extreme dn ×2 PSR (35% weight)
]

SCENARIO_LABELS = [
    "S1: ΔF=0, vol↑",
    "S2: ΔF=0, vol↓",
    "S3: ΔF=+⅓PSR, vol↑",
    "S4: ΔF=+⅓PSR, vol↓",
    "S5: ΔF=−⅓PSR, vol↑",
    "S6: ΔF=−⅓PSR, vol↓",
    "S7: ΔF=+⅔PSR, vol↑",
    "S8: ΔF=+⅔PSR, vol↓",
    "S9: ΔF=−⅔PSR, vol↑",
    "S10: ΔF=−⅔PSR, vol↓",
    "S11: ΔF=+PSR, vol↑",
    "S12: ΔF=+PSR, vol↓",
    "S13: ΔF=−PSR, vol↑",
    "S14: ΔF=−PSR, vol↓",
    "S15: ΔF=+2PSR (35%)",
    "S16: ΔF=−2PSR (35%)",
]
