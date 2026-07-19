"""
Data Integrity Checks  ·  standalone module usable by any indicator
─────────────────────────────────────────────────────────────────────
Bad OHLC input is the most common *silent* cause of a wrong signal — a
stale bar, an unadjusted split, a missing trading day — and it's invisible
unless something explicitly checks for it. This module runs two layers:

1. Internal consistency checks on the yfinance bars themselves:
     - OHLC sanity (high must be the max, low must be the min of the bar)
     - zero-volume bars that still show price movement (stale/bad print)
     - runs of 3+ bars with identical OHLC (duplicated/stale data)
     - abnormally large single-bar moves relative to the series' own
       volatility (a common symptom of an unadjusted stock split)
     - missing business-day bars for daily data (calendar gaps)

2. A best-effort cross-check against a second source (stooq's daily CSV
   endpoint) on overlapping dates. This only works for interval="1d" and
   only for tickers stooq actually lists (mostly US-listed symbols) —
   many exchange-specific tickers (e.g. NEPSE-style ".NP" symbols) simply
   won't have a stooq match. In that case cross_check_available is False
   rather than raising or reporting a false mismatch.

Importable:
    from data_integrity import check_data_integrity
    report = check_data_integrity("AAPL", period="1y", interval="1d")
"""
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
except ImportError:  # pragma: no cover - requests ships with yfinance's deps in practice
    requests = None


# ────────────────────────────────────────────────────────────────────
#  INTERNAL CONSISTENCY CHECKS
# ────────────────────────────────────────────────────────────────────
def _internal_consistency_checks(df, interval):
    issues = []
    n = len(df)
    if n == 0:
        return [{"type": "no_data", "detail": "No bars returned."}]

    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(n)
    dates = df["Date"]

    def _d(i):
        return str(dates.iloc[i])[:10]

    # 1. OHLC sanity: high must be >= open/close/low; low must be <= open/close/high
    bad_ohlc = np.where((h < l) | (h < o) | (h < c) | (l > o) | (l > c))[0]
    for i in bad_ohlc[:20]:
        issues.append({"type": "bad_ohlc", "bar": int(i), "date": _d(i),
                        "detail": f"O={o[i]:.4f} H={h[i]:.4f} L={l[i]:.4f} C={c[i]:.4f} "
                                  f"violates High>=all>=Low."})

    # 2. Zero volume on a bar that still shows price movement
    if v.sum() > 0:
        zero_vol = np.where((v == 0) & (np.abs(c - o) > 1e-9))[0]
        for i in zero_vol[:20]:
            issues.append({"type": "zero_volume_with_movement", "bar": int(i), "date": _d(i),
                            "detail": "Zero reported volume on a bar with nonzero price movement."})

    # 3. Stale/duplicated bars: 3+ consecutive bars with identical OHLC
    if n > 3:
        same = (np.abs(np.diff(c)) < 1e-9) & (np.abs(np.diff(o)) < 1e-9) & \
               (np.abs(np.diff(h)) < 1e-9) & (np.abs(np.diff(l)) < 1e-9)
        run = 0
        for i, s in enumerate(same):
            if s:
                run += 1
                if run == 2:  # the 3rd identical bar in a row, at index i+1
                    issues.append({"type": "stale_bars", "bar": int(i + 1), "date": _d(i + 1),
                                    "detail": "3+ consecutive bars with identical OHLC — "
                                              "possible stale or duplicated data."})
            else:
                run = 0

    # 4. Abnormally large single-bar move -> possible unadjusted split or bad print
    if n > 20:
        pct_change = np.abs(np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1]))
        pct_change = np.nan_to_num(pct_change, nan=0.0)
        median_move = np.median(pct_change[pct_change > 0]) if (pct_change > 0).any() else 0.0
        thresh = max(0.20, median_move * 15)  # heuristic: >=20% or 15x the typical daily move
        jumps = np.where(pct_change > thresh)[0]
        for i in jumps[:10]:
            ratio = c[i + 1] / c[i] if c[i] != 0 else None
            issues.append({"type": "large_single_bar_move", "bar": int(i + 1), "date": _d(i + 1),
                            "detail": f"{pct_change[i] * 100:.1f}% single-bar move "
                                      f"(close ratio {ratio:.3f} if ratio else 'n/a'); "
                                      f"verify this isn't an unadjusted stock split or a bad print."})

    # 5. Calendar gaps for daily data (long runs of missing business days)
    if interval in ("1d", "5d") and n > 5:
        d = pd.to_datetime(dates)
        business_days = pd.bdate_range(d.iloc[0], d.iloc[-1])
        missing = sorted(business_days.difference(d))
        if missing:
            run_start = missing[0]
            run_len = 1
            for i in range(1, len(missing)):
                if (missing[i] - missing[i - 1]).days == 1:
                    run_len += 1
                else:
                    if run_len >= 4:
                        issues.append({"type": "data_gap",
                                        "detail": f"Gap of {run_len} consecutive business days "
                                                  f"starting {run_start.date()} — check for missing bars "
                                                  f"(could also be an exchange closure/holiday run)."})
                    run_start, run_len = missing[i], 1
            if run_len >= 4:
                issues.append({"type": "data_gap",
                                "detail": f"Gap of {run_len} consecutive business days "
                                          f"starting {run_start.date()} — check for missing bars."})
    return issues


# ────────────────────────────────────────────────────────────────────
#  SECOND-SOURCE CROSS-CHECK (stooq)
# ────────────────────────────────────────────────────────────────────
def _stooq_symbol_guess(ticker):
    t = ticker.lower()
    return t if "." in t else f"{t}.us"


def _fetch_stooq_daily(ticker):
    if requests is None:
        raise RuntimeError("The 'requests' package is not available.")
    sym = _stooq_symbol_guess(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    text = r.text
    if not text or text.strip().startswith("<") or "Exceeded" in text:
        raise ValueError("Stooq returned no usable CSV for this symbol.")
    sdf = pd.read_csv(StringIO(text))
    if "Date" not in sdf.columns or len(sdf) == 0:
        raise ValueError("Stooq CSV was empty or malformed for this symbol.")
    sdf["Date"] = pd.to_datetime(sdf["Date"])
    return sdf


def cross_check_with_stooq(df, ticker, interval, tolerance_pct=1.0):
    """
    Best-effort cross-check of yfinance daily closes against stooq's daily
    closes on overlapping dates. Many exchange-specific tickers won't have
    a stooq match at all — that's reported as cross_check_available=False,
    not as a mismatch.
    """
    result = {"cross_check_available": False, "source": "stooq", "mismatches": []}
    if interval not in ("1d", "5d"):
        result["note"] = "Cross-check is only implemented for daily bars."
        return result
    try:
        sdf = _fetch_stooq_daily(ticker)
    except Exception as e:
        result["note"] = f"Could not fetch a secondary source for {ticker}: {e}"
        return result

    dates = pd.to_datetime(df["Date"])
    merged = pd.DataFrame({"Date": dates, "yf_close": df["Close"].values.astype(float)})
    merged = merged.merge(sdf[["Date", "Close"]].rename(columns={"Close": "stooq_close"}),
                           on="Date", how="inner")
    if len(merged) == 0:
        result["note"] = "No overlapping dates between yfinance and stooq data for this symbol."
        return result

    merged["pct_diff"] = (np.abs(merged["yf_close"] - merged["stooq_close"]) /
                           merged["stooq_close"].replace(0, np.nan)) * 100
    bad = merged[merged["pct_diff"] > tolerance_pct]
    result["cross_check_available"] = True
    result["overlapping_bars"] = int(len(merged))
    result["mismatch_count"] = int(len(bad))
    result["mismatches"] = [
        {"date": str(row.Date)[:10], "yfinance_close": round(float(row.yf_close), 4),
         "stooq_close": round(float(row.stooq_close), 4), "pct_diff": round(float(row.pct_diff), 3)}
        for row in bad.itertuples()
    ][:20]
    return result


# ────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ────────────────────────────────────────────────────────────────────
def check_data_integrity(ticker, period="1y", interval="1d", cross_check=True):
    """
    Runs internal OHLC sanity checks plus (optionally) a best-effort
    cross-check against stooq for daily data. Returns a JSON-safe dict
    suitable for a Flask endpoint.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or len(df) == 0:
        return {"ticker": ticker, "period": period, "interval": interval,
                "error": f"No data returned by yfinance for {ticker}."}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna().copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})

    issues = _internal_consistency_checks(df, interval)
    cc = cross_check_with_stooq(df, ticker, interval) if cross_check else \
        {"cross_check_available": False, "note": "skipped"}

    return {
        "ticker": ticker, "period": period, "interval": interval,
        "bars_checked": int(len(df)),
        "issues_found": len(issues),
        "issues": issues,
        "cross_check": cc,
        "healthy": (len(issues) == 0) and not cc.get("mismatch_count", 0),
    }


if __name__ == "__main__":
    tkr = input("Enter stock ticker: ").strip().upper()
    rep = check_data_integrity(tkr)
    print(f"Bars checked: {rep.get('bars_checked')}  Issues found: {rep.get('issues_found')}  "
          f"Healthy: {rep.get('healthy')}")
    for iss in rep.get("issues", []):
        print(" -", iss["type"], iss.get("date", ""), iss["detail"])
    print("Cross-check:", rep.get("cross_check"))
