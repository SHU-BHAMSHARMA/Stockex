"""
series_utils.py  ·  shared helper for indicator modules
─────────────────────────────────────────────────────────
Pure formatting helpers used to attach a JSON-safe "series" block (dates +
OHLC + the indicator's own already-computed arrays) to each indicator's
return dict, so the frontend can draw its own dynamic (pan/zoom) charts
instead of relying on the matplotlib PNGs.

This module contains NO analysis/decision logic of its own — it only
reshapes numbers that the indicator files already computed.
"""
import math
import numpy as np


def clean_list(arr, ndigits=4):
    """numpy/pandas array -> plain python list, NaN/inf -> None."""
    out = []
    for v in np.asarray(arr, dtype=float):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out


def dates_list(df, col="Date"):
    """Full ISO-ish datetime strings (not truncated) so intraday bars on the
    same calendar day still get distinct timestamps on the frontend chart."""
    return [str(d) for d in df[col]]


def ohlc_payload(df, o=None, h=None, l=None, c=None, v=None):
    """Build the shared OHLC(V) block every chart needs as its base layer."""
    payload = {
        "dates": dates_list(df),
        "open": clean_list(o if o is not None else df["Open"].values),
        "high": clean_list(h if h is not None else df["High"].values),
        "low": clean_list(l if l is not None else df["Low"].values),
        "close": clean_list(c if c is not None else df["Close"].values),
    }
    if v is not None or "Volume" in df.columns:
        payload["volume"] = clean_list(v if v is not None else df["Volume"].values, ndigits=0)
    return payload
