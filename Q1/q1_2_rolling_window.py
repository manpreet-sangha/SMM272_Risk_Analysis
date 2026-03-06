"""
Q1 Part 2 — Rolling window generator.

For each trading day t >= ROLLING_START_DATE the generator yields:
    (t,  window_returns)
where window_returns contains all daily log returns in the half-open
interval  [t - ROLLING_WINDOW_MONTHS calendar months,  t).

Design notes
------------
* Calendar months (via dateutil.relativedelta) are used instead of a
  fixed number of trading days so that the window length is invariant
  to holidays.  A 6-month calendar window typically contains ~126
  trading days.
* A minimum-observations guard (MIN_OBS = 20) skips degenerate windows
  at the very start of the sample.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dateutil.relativedelta import relativedelta

from config import ROLLING_WINDOW_MONTHS, ROLLING_START_DATE

MIN_OBS = 20   # skip windows with fewer observations than this


def generate_rolling_windows(returns, start_date=None, window_months=None):
    """
    Yield (forecast_date, window_returns) for every trading day from
    *start_date* onwards.

    Parameters
    ----------
    returns : pd.Series
        Daily log-return series indexed by pd.Timestamp.
    start_date : str or pd.Timestamp, optional
        First forecast date.  Defaults to ROLLING_START_DATE from config.
    window_months : int, optional
        Estimation window length in calendar months.
        Defaults to ROLLING_WINDOW_MONTHS from config.

    Yields
    ------
    forecast_date : pd.Timestamp
        The date for which VaR / ES is being forecast.
    window_returns : pd.Series
        Historical returns in [forecast_date - window_months, forecast_date).
    """
    if start_date is None:
        start_date = ROLLING_START_DATE
    if window_months is None:
        window_months = ROLLING_WINDOW_MONTHS

    start = pd.Timestamp(start_date)
    forecast_dates = returns.index[returns.index >= start]

    for date in forecast_dates:
        window_start = date - relativedelta(months=window_months)
        mask = (returns.index >= window_start) & (returns.index < date)
        window = returns.loc[mask]
        if len(window) >= MIN_OBS:
            yield date, window


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from logger import setup_run_logger, get_logger
    from q1_1_build_portfolio import build_portfolio

    setup_run_logger("smm272_q1_2_rolling_window")
    log = get_logger("q1_2_rolling_window")

    _, _, port_ret = build_portfolio()
    windows = list(generate_rolling_windows(port_ret))
    log.info(f"Total forecast dates : {len(windows):,}")
    d0, w0 = windows[0]
    log.info(f"  First forecast date : {d0.date()}  |  window obs: {len(w0)}")
    dn, wn = windows[-1]
    log.info(f"  Last  forecast date : {dn.date()}  |  window obs: {len(wn)}")
