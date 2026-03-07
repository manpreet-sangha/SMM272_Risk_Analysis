"""
Q5 — Position Pair Definitions
===============================
Defines the four position pairs used for SPAN margin analysis.
Each pair is a list of position dicts (see q5_span_scenarios module
docstring for the dict schema).

All legs use the ATM strike K = 4.65 USD/lb with one contract per leg.
Implied vols are sourced from q5_market_data.

Pairs
-----
P1 : Long Call  + Short Put   (synthetic long  / risk reversal – bullish)
P2 : Short Call + Long Put    (synthetic short / risk reversal – bearish)
P3 : Long Call  + Long Futures (leveraged bullish)
P4 : Long Futures + Long Put  (protective put / hedged long)
"""

from q5_market_data import (
    ATM_STRIKE,
    IV_CALL_ATM,
    IV_PUT_ATM,
    F,
)


# ── Individual legs ───────────────────────────────────────────────────────────

LONG_CALL = {
    "type":     "call",
    "quantity": +1,
    "strike":   ATM_STRIKE,
    "iv":       IV_CALL_ATM,
    "label":    "Long Call (K=4.65)",
}

SHORT_PUT = {
    "type":     "put",
    "quantity": -1,
    "strike":   ATM_STRIKE,
    "iv":       IV_PUT_ATM,
    "label":    "Short Put (K=4.65)",
}

SHORT_CALL = {
    "type":     "call",
    "quantity": -1,
    "strike":   ATM_STRIKE,
    "iv":       IV_CALL_ATM,
    "label":    "Short Call (K=4.65)",
}

LONG_PUT = {
    "type":     "put",
    "quantity": +1,
    "strike":   ATM_STRIKE,
    "iv":       IV_PUT_ATM,
    "label":    "Long Put (K=4.65)",
}

LONG_FUTURES = {
    "type":     "futures",
    "quantity": +1,
    "strike":   None,
    "iv":       None,
    "label":    "Long Futures (HGK26)",
}


# ── Position pairs ────────────────────────────────────────────────────────────

PAIR_1 = {
    "name":  "P1: Long Call + Short Put",
    "short": "P1",
    "legs":  [LONG_CALL, SHORT_PUT],
}

PAIR_2 = {
    "name":  "P2: Short Call + Long Put",
    "short": "P2",
    "legs":  [SHORT_CALL, LONG_PUT],
}

PAIR_3 = {
    "name":  "P3: Long Call + Long Futures",
    "short": "P3",
    "legs":  [LONG_CALL, LONG_FUTURES],
}

PAIR_4 = {
    "name":  "P4: Long Futures + Long Put",
    "short": "P4",
    "legs":  [LONG_FUTURES, LONG_PUT],
}

ALL_PAIRS = [PAIR_1, PAIR_2, PAIR_3, PAIR_4]


# ── Helper: single-leg portfolios ──────────────────────────────────────────

def single_positions(pair: dict) -> list:
    """
    Return a list of one-element position lists — one per leg — so that
    each leg can be passed individually to span_margin() for the
    no-netting calculation.

    Parameters
    ----------
    pair : one element of ALL_PAIRS

    Returns
    -------
    list of lists  e.g. [[leg1_dict], [leg2_dict]]
    """
    return [[leg] for leg in pair["legs"]]
