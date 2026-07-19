"""
Liquidity Sweeps + Fair Value Gaps (FVG)  ·  companion module to order_blocks.py
──────────────────────────────────────────────────────────────────────────────
Two independent, deterministic SMC-style detectors that plug straight into
the order-block engine as confluence/precondition filters:

1. Fair Value Gaps (imbalances)
   A 3-candle gap: candle[i]'s low is above candle[i-2]'s high (bullish FVG)
   or candle[i]'s high is below candle[i-2]'s low (bearish FVG). Price often
   returns to fill these before continuing — requiring an order block to sit
   inside/near an *unfilled* FVG of the same direction is a standard way to
   tighten which OBs are treated as high-quality entries.

2. Liquidity sweeps (stop hunts)
   Before a genuine reversal (CHoCH), price commonly wicks just beyond a
   prior swing high/low — grabbing resting stop-loss liquidity — and then
   closes back inside the range. Requiring this precondition before trusting
   a CHoCH filters out a lot of premature/false reversal calls that are just
   a clean break with no stop-hunt behind it.

Both functions operate on plain numpy arrays (no network, no pandas
requirement) so they are trivially unit-testable and reusable from the
backtester as well as the live engine.
"""
import numpy as np


# ────────────────────────────────────────────────────────────────────
#  FAIR VALUE GAPS
# ────────────────────────────────────────────────────────────────────
def detect_fvgs(o, h, l, c):
    """
    Scan the full series for 3-candle Fair Value Gaps.

    Bullish FVG at bar i: low[i] > high[i-2]   (gap up, unfilled space between
        the top of candle i-2's wick and the bottom of candle i's wick)
    Bearish FVG at bar i: high[i] < low[i-2]   (gap down)

    Returns a list of dicts, each JSON-safe:
        {"direction": "bullish"/"bearish", "formed_bar": i,
         "top": float, "bottom": float}
    `formed_bar` is the index of the 3rd candle (the one that confirms the
    gap exists) — this is also the earliest bar from which "has this been
    filled yet" can be evaluated.
    """
    n = len(c)
    fvgs = []
    for i in range(2, n):
        if l[i] > h[i - 2]:
            fvgs.append({
                "direction": "bullish",
                "formed_bar": i,
                "top": float(l[i]),
                "bottom": float(h[i - 2]),
            })
        if h[i] < l[i - 2]:
            fvgs.append({
                "direction": "bearish",
                "formed_bar": i,
                "top": float(l[i - 2]),
                "bottom": float(h[i]),
            })
    return fvgs


def mark_fvg_fill_status(fvgs, l, h):
    """
    For each FVG, vectorised-scan every bar after it formed for the first
    bar whose range overlaps the gap at all (a "fill" doesn't require a
    full close through it — any wick trading back into the zone counts as
    the market having revisited the imbalance). Mutates and returns fvgs
    with `filled` (bool) and `filled_bar` (int or None) added.
    """
    n = len(l)
    for fvg in fvgs:
        start = fvg["formed_bar"] + 1
        filled, filled_bar = False, None
        if start < n:
            seg_l = l[start:]
            seg_h = h[start:]
            touched = (seg_l <= fvg["top"]) & (seg_h >= fvg["bottom"])
            if touched.any():
                filled = True
                filled_bar = start + int(np.argmax(touched))
        fvg["filled"] = filled
        fvg["filled_bar"] = filled_bar
    return fvgs


# ────────────────────────────────────────────────────────────────────
#  LIQUIDITY SWEEPS (STOP HUNTS)
# ────────────────────────────────────────────────────────────────────
def check_liquidity_sweep(l, h, c, pivot_idx, pivot_price, break_bar, direction,
                           max_lookback=50):
    """
    Precondition check for trusting a CHoCH: was there a bar, between the
    reference swing pivot and the structure break, whose wick pushed
    *beyond* the pivot (grabbing the stop-loss liquidity resting there) but
    whose close came back *inside* the range? That pattern — sweep then
    reject — is the classic stop-hunt signature that precedes a real
    reversal, as opposed to a break that just runs cleanly through the
    level with no rejection at all.

    direction:
        "bullish_reversal" -> pivot_price is a prior swing LOW; looks for
            low[j] < pivot_price (wick below) AND close[j] >= pivot_price
            (closed back above) at any bar j in (pivot_idx, break_bar).
        "bearish_reversal" -> pivot_price is a prior swing HIGH; looks for
            high[j] > pivot_price (wick above) AND close[j] <= pivot_price
            (closed back below) at any bar j in (pivot_idx, break_bar).

    Returns True/False. Returns False (not None) when there's nothing to
    check, so callers can use the result directly in boolean filters.
    """
    if pivot_idx is None or break_bar <= pivot_idx + 1:
        return False
    start = max(pivot_idx + 1, break_bar - max_lookback)
    end = min(break_bar, len(c))
    for j in range(start, end):
        if direction == "bullish_reversal":
            if l[j] < pivot_price and c[j] >= pivot_price:
                return True
        elif direction == "bearish_reversal":
            if h[j] > pivot_price and c[j] <= pivot_price:
                return True
    return False
