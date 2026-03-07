"""
Q6 — Market Data Download & Feature Engineering
================================================
Downloads 2 years of daily adjusted-close prices for INTC, JPM, AA, PG
from Yahoo Finance using the reference date of February 12th, 2026.

Public API
----------
get_prices()        -> pd.DataFrame   adjusted-close prices (date × ticker)
get_log_returns()   -> pd.DataFrame   daily log-return matrix
get_hist_vols()     -> pd.Series      annualised historical volatility per ticker
get_spot_prices()   -> pd.Series      closing prices on the reference date

All functions cache results in module-level variables after the first call
to avoid redundant downloads during a single run.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Constants ─────────────────────────────────────────────────────────────────
TICKERS        = config.Q6_TICKERS          # ["INTC", "JPM", "AA", "PG"]
REFERENCE_DATE = config.Q6_REFERENCE_DATE   # "2026-02-12"
HIST_START     = config.Q6_HIST_START       # "2024-02-12"
HIST_END       = "2026-02-14"               # exclusive upper bound (yfinance)
TRADING_DAYS   = config.TRADING_DAYS        # 252

# ── Module-level cache ────────────────────────────────────────────────────────
_prices_cache:   pd.DataFrame | None = None
_returns_cache:  pd.DataFrame | None = None
_vols_cache:     pd.Series    | None = None
_spot_cache:     pd.Series    | None = None


# ── Data download ─────────────────────────────────────────────────────────────

def _download_raw() -> pd.DataFrame:
    """Download adjusted-close prices from Yahoo Finance and return a tidy DataFrame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            TICKERS,
            start=HIST_START,
            end=HIST_END,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][TICKERS]
    else:
        prices = raw[["Close"]].rename(columns={"Close": TICKERS[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(how="all")

    # Filter to at most reference date
    ref = pd.Timestamp(REFERENCE_DATE)
    prices = prices[prices.index <= ref]

    return prices


def get_prices() -> pd.DataFrame:
    """Return daily adjusted-close price DataFrame (date × ticker)."""
    global _prices_cache
    if _prices_cache is None:
        _prices_cache = _download_raw()
    return _prices_cache.copy()


def get_log_returns(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return daily log-return DataFrame (date × ticker)."""
    global _returns_cache
    if _returns_cache is None:
        p = prices if prices is not None else get_prices()
        _returns_cache = np.log(p / p.shift(1)).dropna()
    return _returns_cache.copy()


def get_hist_vols(returns: pd.DataFrame | None = None) -> pd.Series:
    """
    Annualised historical volatility for each ticker.

    σ_hist = std(daily log-return) × √252

    Uses the full 2-year window; this is passed to Black-Scholes as the
    input volatility for option pricing, per the question specification.
    """
    global _vols_cache
    if _vols_cache is None:
        r = returns if returns is not None else get_log_returns()
        _vols_cache = r.std() * np.sqrt(TRADING_DAYS)
        _vols_cache.name = "hist_vol_annualised"
    return _vols_cache.copy()


def get_spot_prices(prices: pd.DataFrame | None = None) -> pd.Series:
    """
    Return closing prices as of the reference date (Feb 12, 2026).

    Falls back to the last available date on or before the reference date
    (handles weekends / holidays).
    """
    global _spot_cache
    if _spot_cache is None:
        p = prices if prices is not None else get_prices()
        ref = pd.Timestamp(REFERENCE_DATE)
        avail = p.index[p.index <= ref]
        if len(avail) == 0:
            raise ValueError(
                f"No price data available on or before {REFERENCE_DATE}. "
                "Check HIST_END or internet connectivity."
            )
        actual_ref = avail[-1]
        _spot_cache = p.loc[actual_ref]
        _spot_cache.name = f"spot_{actual_ref.date()}"
    return _spot_cache.copy()


# ── Convenience ───────────────────────────────────────────────────────────────

def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Download and compute all market data in one call.

    Returns
    -------
    prices      : pd.DataFrame  (T × 4)
    log_returns : pd.DataFrame  (T-1 × 4)
    hist_vols   : pd.Series     (4,)   annualised
    spot_prices : pd.Series     (4,)   as of reference date
    """
    prices      = get_prices()
    log_returns = get_log_returns(prices)
    hist_vols   = get_hist_vols(log_returns)
    spot_prices = get_spot_prices(prices)
    return prices, log_returns, hist_vols, spot_prices


# ── Diagnostic ────────────────────────────────────────────────────────────────

def describe_data(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    hist_vols: pd.Series,
    spot_prices: pd.Series,
) -> None:
    """Print a compact data summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Q6 MARKET DATA SUMMARY")
    print(sep)
    print(f"  Reference date : {REFERENCE_DATE}")
    print(f"  History window : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Observations   : {len(prices)} closing prices per ticker")
    print(f"  Log-returns    : {len(log_returns)} daily observations\n")

    print(f"  {'Ticker':<8} {'Spot ($)':>10} {'HVol (%)':>10} {'Min ($)':>10} {'Max ($)':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for t in TICKERS:
        spot = spot_prices[t]
        vol  = hist_vols[t] * 100
        lo   = prices[t].min()
        hi   = prices[t].max()
        print(f"  {t:<8} {spot:>10.2f} {vol:>10.2f} {lo:>10.2f} {hi:>10.2f}")
    print(sep)
