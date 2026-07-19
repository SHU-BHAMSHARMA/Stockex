"""
Multi-Timeframe Confluence  ·  companion module to order_blocks.py
──────────────────────────────────────────────────────────────────
Pulls a *higher* timeframe than the one currently being analysed (weekly
trend for a daily system, daily trend for an hourly system, etc.) and
computes its market-structure trend using the exact same swing/BOS/CHoCH
walk that order_blocks.py uses internally, so "trend" means the identical
thing on both timeframes.

This is deliberately a thin module: it only produces a directional bias
(up / down / undefined) plus a little context for transparency. It does
NOT re-run order-block tagging on the higher timeframe — only the
higher-timeframe trend is needed to filter/score the lower-timeframe
signals, per the rule "only take order blocks whose direction agrees with
the higher-timeframe trend."

Importable:
    from mtf_confluence import get_higher_timeframe_trend
    trend, info = get_higher_timeframe_trend("NABIL.NP", "1d", "1y")
"""
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# base interval -> (higher interval, higher period to request)
# the higher period is chosen generously so enough bars exist to confirm
# swing pivots on the higher timeframe, not just to match the base window.
_HTF_MAP = {
    "1m":  ("15m", "5d"),
    "2m":  ("30m", "1mo"),
    "5m":  ("30m", "1mo"),
    "15m": ("1h",  "3mo"),
    "30m": ("1h",  "6mo"),
    "60m": ("1d",  "2y"),
    "90m": ("1d",  "2y"),
    "1h":  ("1d",  "2y"),
    "1d":  ("1wk", "5y"),
    "5d":  ("1wk", "5y"),
    "1wk": ("1mo", "10y"),
    "1mo": ("3mo", "20y"),
}


def _higher_timeframe(interval):
    return _HTF_MAP.get(interval, ("1wk", "5y"))


def _structure_trend(o, h, l, c, pivot_left=3, pivot_right=3):
    """
    Minimal BOS/CHoCH structure walk (same logic as order_blocks.py's main
    loop, without the order-block tagging step) -> returns the final
    confirmed trend plus the event that produced it.
    """
    n = len(c)
    order = max(pivot_left, pivot_right)
    if n < order * 2 + 5:
        return None, None

    sh_idx = argrelextrema(h, np.greater, order=order)[0].tolist()
    sl_idx = argrelextrema(l, np.less, order=order)[0].tolist()
    swing_highs = sorted([(i, h[i]) for i in sh_idx], key=lambda x: x[0])
    swing_lows = sorted([(i, l[i]) for i in sl_idx], key=lambda x: x[0])
    confirm_lag = pivot_right

    cur_high = cur_low = None
    trend = None
    last_event = None
    sh_ptr = sl_ptr = 0

    for i in range(n):
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] + confirm_lag <= i:
            idx_p, price_p = swing_highs[sh_ptr]
            if cur_high is None or idx_p > cur_high[0]:
                cur_high = (idx_p, price_p)
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] + confirm_lag <= i:
            idx_p, price_p = swing_lows[sl_ptr]
            if cur_low is None or idx_p > cur_low[0]:
                cur_low = (idx_p, price_p)
            sl_ptr += 1

        if cur_high is not None and i > cur_high[0] and c[i] > cur_high[1]:
            event_type = "BOS" if trend == "up" else "CHoCH"
            trend = "up"
            last_event = {"type": event_type, "bar": i, "level": float(cur_high[1])}
            cur_high = None

        if cur_low is not None and i > cur_low[0] and c[i] < cur_low[1]:
            event_type = "BOS" if trend == "down" else "CHoCH"
            trend = "down"
            last_event = {"type": event_type, "bar": i, "level": float(cur_low[1])}
            cur_low = None

    return trend, last_event


def get_higher_timeframe_trend(ticker, base_interval, base_period):
    """
    Downloads the appropriate higher timeframe for `base_interval` and
    returns (trend, info):
        trend -> "up" / "down" / None (undefined / not enough data)
        info  -> dict with the higher timeframe used and last-event context,
                 safe to embed directly in an API JSON response.
    Never raises for "no data" conditions — callers get trend=None and an
    explanatory info dict instead, so a higher-timeframe outage degrades
    gracefully to "no filter applied" rather than crashing the endpoint.
    """
    htf_interval, htf_period = _higher_timeframe(base_interval)
    try:
        df = yf.download(ticker, period=htf_period, interval=htf_interval, progress=False)
    except Exception as e:
        return None, {"higher_timeframe": htf_interval, "higher_period": htf_period,
                       "error": f"download failed: {e}"}

    if df is None or len(df) == 0:
        return None, {"higher_timeframe": htf_interval, "higher_period": htf_period,
                       "error": "no data returned"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna().copy()
    df = df.reset_index()
    date_col = df.columns[0]

    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)

    trend, last_event = _structure_trend(o, h, l, c)

    info = {
        "higher_timeframe": htf_interval,
        "higher_period": htf_period,
        "bars_used": int(len(df)),
    }
    if last_event is not None:
        info["last_event"] = last_event["type"]
        info["last_event_date"] = str(df.loc[last_event["bar"], date_col])[:10]
        info["last_event_level"] = round(last_event["level"], 4)
    else:
        info["note"] = "Not enough higher-timeframe bars to confirm a structure trend."

    return trend, info
