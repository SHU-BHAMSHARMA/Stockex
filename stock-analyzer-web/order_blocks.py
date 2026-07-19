"""
Order Block + Market-Structure Reversal Detector  ·  NEPSE / any ticker
────────────────────────────────────────────────────────────────────────
Smart-Money-Concepts (SMC) style engine that:

  1. Finds swing highs / swing lows (fractal pivots).
  2. Walks the bars chronologically to build market structure and flags
     BOS  (Break of Structure   -> trend continuation)
     CHoCH(Change of Character  -> trend reversal)
  3. For every BOS/CHoCH, locates the Order Block: the last opposite-colour
     candle before the impulse leg that produced the break.
  4. Tracks every order block's life-cycle: untested -> tested -> mitigated
     / invalidated.
  5. Produces a deterministic BUY / SELL / WATCH / NEUTRAL verdict with an
     explicit entry price, stop-loss and take-profit, built only from the
     rules below (no discretionary/model guessing).
  6. Renders a full-history chart and a zoomed recent-window chart.

Every threshold used below is declared once in CONFIG and is the ONLY
thing that decides the outcome for a given bar of data — i.e. the same
OHLC input always reproduces the exact same signal.

Importable:
    from order_blocks import analyze_order_blocks
    result = analyze_order_blocks("NABIL.NP", period="1y", interval="1d")

API-ready: every field in the returned dict is a plain python
int/float/str/bool/list/dict (no numpy / pandas objects), so the result
can be passed straight to `json.dumps` / a Flask-Fast API jsonify call.
"""
import matplotlib
matplotlib.use('Agg')
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import os

from liquidity_fvg import detect_fvgs, mark_fvg_fill_status, check_liquidity_sweep
from mtf_confluence import get_higher_timeframe_trend
from series_utils import clean_list, ohlc_payload


# ────────────────────────────────────────────────────────────────────
#  CORE ANALYSIS (operates on a ready-made OHLCV DataFrame so it can be
#  unit-tested without a network call — the yfinance download only
#  happens in analyze_order_blocks()).
# ────────────────────────────────────────────────────────────────────
def _run_analysis(df, ticker, period, interval, save_chart=True,
                   htf_trend=None, htf_info=None,
                   require_liquidity_sweep=True,
                   require_fvg_confluence=False,
                   require_mtf_alignment=False):

    # ── CONFIG ──────────────────────────────────────────────────────
    PIVOT_LEFT   = 4          # bars required on each side to confirm a swing
    PIVOT_RIGHT  = 4
    PIVOT_ORDER  = max(PIVOT_LEFT, PIVOT_RIGHT)

    ATR_PERIOD          = 14
    OB_MAX_ATR_MULT      = 3.0    # an OB wider than this many ATRs is discarded (not a clean block)
    OB_LOOKBACK_CAP      = 30     # max bars to search backward for the OB candle
    INVALIDATION_BUFFER  = 0.0    # extra ATR buffer past the zone before calling it invalidated
    RECENT_BARS          = 25
    VOLUME_SPIKE_MULT    = 1.5    # breakout candle volume vs its own 20-bar average
    MAX_ACTIVE_OB_PER_SIDE = 5    # keep charts / output readable

    n = len(df)
    if n == 0:
        return {"error": f"No data for {ticker}"}
    if n < (PIVOT_ORDER * 2 + ATR_PERIOD + 5):
        return {"error": f"Not enough bars ({n}) for {ticker} to run structure analysis "
                          f"(need at least {PIVOT_ORDER*2+ATR_PERIOD+5})."}

    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float) if "Volume" in df.columns else np.zeros(n)
    dates = df["Date"] if "Date" in df.columns else df.index

    # ── ATR (Wilder) ────────────────────────────────────────────────
    prev_close = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().bfill().values
    atr = np.where(atr == 0, np.nanmean(tr) or 1e-6, atr)

    avg_vol20 = pd.Series(v).rolling(20, min_periods=5).mean().bfill().values

    # ── FAIR VALUE GAPS (imbalances) ─────────────────────────────────
    fvgs = detect_fvgs(o, h, l, c)
    fvgs = mark_fvg_fill_status(fvgs, l, h)

    # ── 1. SWING PIVOTS (fractals) ─────────────────────────────────
    sh_idx = argrelextrema(h, np.greater, order=PIVOT_ORDER)[0].tolist()
    sl_idx = argrelextrema(l, np.less,    order=PIVOT_ORDER)[0].tolist()
    # a pivot at index i is only *known* to the market CONFIRM_LAG bars later
    CONFIRM_LAG = PIVOT_RIGHT

    swing_highs = sorted([(i, h[i]) for i in sh_idx], key=lambda x: x[0])
    swing_lows  = sorted([(i, l[i]) for i in sl_idx], key=lambda x: x[0])

    # ── 2. MARKET STRUCTURE WALK: BOS / CHoCH detection ────────────
    # current_high_pivot / current_low_pivot = the most recently CONFIRMED
    # swing that has not yet been broken by a close.
    current_high_pivot = None   # (idx, price)
    current_low_pivot  = None
    trend = None                 # 'up' / 'down' / None (undefined)
    structure_events = []        # BOS / CHoCH log
    order_blocks = []            # every order block ever tagged
    sh_ptr, sl_ptr = 0, 0

    for i in range(n):
        # bring in any newly-confirmed swing highs / lows (idx+CONFIRM_LAG<=i)
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr][0] + CONFIRM_LAG <= i:
            idx_p, price_p = swing_highs[sh_ptr]
            if current_high_pivot is None or idx_p > current_high_pivot[0]:
                current_high_pivot = (idx_p, price_p)
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr][0] + CONFIRM_LAG <= i:
            idx_p, price_p = swing_lows[sl_ptr]
            if current_low_pivot is None or idx_p > current_low_pivot[0]:
                current_low_pivot = (idx_p, price_p)
            sl_ptr += 1

        # bullish break: close closes above the active swing high
        if current_high_pivot is not None and i > current_high_pivot[0] and c[i] > current_high_pivot[1]:
            event_type = "BOS" if trend == "up" else "CHoCH"
            liquidity_swept = None
            if event_type == "CHoCH" and current_low_pivot is not None:
                liquidity_swept = check_liquidity_sweep(
                    l, h, c, pivot_idx=current_low_pivot[0], pivot_price=current_low_pivot[1],
                    break_bar=i, direction="bullish_reversal")
            structure_events.append({
                "type": event_type, "direction": "bullish",
                "break_bar": i, "pivot_bar": current_high_pivot[0],
                "level": float(current_high_pivot[1]),
                "liquidity_swept": liquidity_swept,
            })
            trend = "up"
            broken_pivot = current_high_pivot
            current_high_pivot = None
            if event_type == "CHoCH" and require_liquidity_sweep and not liquidity_swept:
                pass  # reversal not confirmed by a stop-hunt sweep -> don't tag an OB for it
            else:
                _tag_order_block(df, o, h, l, c, v, avg_vol20, atr,
                                  leg_start=broken_pivot[0], break_bar=i,
                                  direction="bullish", ob_type=event_type,
                                  lookback_cap=OB_LOOKBACK_CAP,
                                  max_atr_mult=OB_MAX_ATR_MULT,
                                  vol_spike_mult=VOLUME_SPIKE_MULT,
                                  out_list=order_blocks,
                                  liquidity_swept=liquidity_swept)

        # bearish break: close closes below the active swing low
        if current_low_pivot is not None and i > current_low_pivot[0] and c[i] < current_low_pivot[1]:
            event_type = "BOS" if trend == "down" else "CHoCH"
            liquidity_swept = None
            if event_type == "CHoCH" and current_high_pivot is not None:
                liquidity_swept = check_liquidity_sweep(
                    l, h, c, pivot_idx=current_high_pivot[0], pivot_price=current_high_pivot[1],
                    break_bar=i, direction="bearish_reversal")
            structure_events.append({
                "type": event_type, "direction": "bearish",
                "break_bar": i, "pivot_bar": current_low_pivot[0],
                "level": float(current_low_pivot[1]),
                "liquidity_swept": liquidity_swept,
            })
            trend = "down"
            broken_pivot = current_low_pivot
            current_low_pivot = None
            if event_type == "CHoCH" and require_liquidity_sweep and not liquidity_swept:
                pass  # reversal not confirmed by a stop-hunt sweep -> don't tag an OB for it
            else:
                _tag_order_block(df, o, h, l, c, v, avg_vol20, atr,
                                  leg_start=broken_pivot[0], break_bar=i,
                                  direction="bearish", ob_type=event_type,
                                  lookback_cap=OB_LOOKBACK_CAP,
                                  max_atr_mult=OB_MAX_ATR_MULT,
                                  vol_spike_mult=VOLUME_SPIKE_MULT,
                                  out_list=order_blocks,
                                  liquidity_swept=liquidity_swept)

    # ── 3. MITIGATION / INVALIDATION LIFE-CYCLE (vectorised) ───────
    # For every OB, work on the numpy slice from formation onward instead of
    # a bar-by-bar Python loop -> same exact semantics, far fewer Python-level
    # iterations (matters once you run this on intraday data with thousands
    # of bars and dozens of order blocks).
    for ob in order_blocks:
        start = ob["formed_bar"] + 1
        if start >= n:
            ob.update(tests=0, mitigated=False, invalidated=False,
                      first_test_bar=None, invalidated_bar=None)
            continue

        seg_l, seg_h, seg_c, seg_atr = l[start:], h[start:], c[start:], atr[start:]
        if ob["direction"] == "bullish":
            touched_mask = seg_l <= ob["zone_high"]
            broken_mask = seg_c < (ob["zone_low"] - INVALIDATION_BUFFER * seg_atr)
        else:
            touched_mask = seg_h >= ob["zone_low"]
            broken_mask = seg_c > (ob["zone_high"] + INVALIDATION_BUFFER * seg_atr)

        if broken_mask.any():
            brk_rel = int(np.argmax(broken_mask))          # first True index
            window = touched_mask[:brk_rel + 1]              # only counts up to & incl. the invalidating bar
            invalidated, invalidated_bar = True, start + brk_rel
        else:
            window = touched_mask
            invalidated, invalidated_bar = False, None

        tests = int(window.sum())
        first_test_bar = start + int(np.argmax(window)) if window.any() else None

        ob.update(tests=tests, mitigated=tests > 0, invalidated=invalidated,
                  first_test_bar=first_test_bar, invalidated_bar=invalidated_bar)

    # ── 4. CURRENT MARKET CONTEXT ───────────────────────────────────
    cur_price = float(c[-1])
    cur_atr = float(atr[-1])

    # ── 5. DETERMINISTIC SCORE FOR EVERY OB ─────────────────────────
    def _has_fvg_confluence(ob):
        """Does an unfilled, same-direction FVG overlap this OB's zone?"""
        for f in fvgs:
            if f["direction"] != ob["direction"] or f["filled"]:
                continue
            lo = max(ob["zone_low"], f["bottom"])
            hi = min(ob["zone_high"], f["top"])
            if hi > lo:
                return True
        return False

    def score_ob(ob):
        score = 0
        aligned = (trend == "up" and ob["direction"] == "bullish") or \
                  (trend == "down" and ob["direction"] == "bearish")
        if aligned:
            score += 30
        if ob["tests"] == 0:
            score += 25
        elif ob["tests"] == 1:
            score += 12
        else:
            score += 3
        # zone tightness relative to ATR at formation -> tighter = cleaner institutional print
        zone_h = ob["zone_high"] - ob["zone_low"]
        atr_at_formation = atr[ob["formed_bar"]]
        if atr_at_formation > 0:
            ratio = zone_h / atr_at_formation
            if ratio <= 1.0:
                score += 20
            elif ratio <= 2.0:
                score += 10
        if ob.get("volume_confirmed"):
            score += 10
        if ob["type"] == "CHoCH":
            score += 10  # OB born from a reversal carries extra weight
            if ob.get("liquidity_swept"):
                score += 10  # reversal confirmed by a stop-hunt sweep, not just a bare break
        if ob["formed_bar"] >= n - RECENT_BARS:
            score += 5
        # proximity to current price, in ATR units -> closer = more immediately actionable
        if cur_atr > 0:
            dist_atr = abs(cur_price - _zone_mid(ob)) / cur_atr
            if dist_atr <= 2:
                score += 15
            elif dist_atr <= 4:
                score += 8
            elif dist_atr <= 8:
                score += 3
        # FVG confluence: the natural complement to an order block
        ob["fvg_confluence"] = _has_fvg_confluence(ob)
        if ob["fvg_confluence"]:
            score += 15
        # higher-timeframe alignment: the single biggest false-signal reducer
        if htf_trend in ("up", "down"):
            ob["htf_aligned"] = (htf_trend == "up" and ob["direction"] == "bullish") or \
                                 (htf_trend == "down" and ob["direction"] == "bearish")
            score += 20 if ob["htf_aligned"] else -20
        else:
            ob["htf_aligned"] = None
        return min(100, max(0, score))

    for ob in order_blocks:
        ob["score"] = score_ob(ob)

    # ── 6. FILTER TO ACTIVE SET, THEN MERGE OVERLAPPING/NESTED ZONES ─
    def _dedupe_overlapping(obs, overlap_thresh=0.5):
        """Greedily keep the highest-scored OB and drop any other OB whose
        zone overlaps it by more than `overlap_thresh` of its own height.
        Prevents reporting near-duplicate/nested zones (e.g. a later BOS
        inside an earlier CHoCH block) as separate tradeable levels."""
        kept = []
        for ob in sorted(obs, key=lambda b: -b["score"]):
            is_dup = False
            for k in kept:
                lo = max(ob["zone_low"], k["zone_low"])
                hi = min(ob["zone_high"], k["zone_high"])
                inter = max(0.0, hi - lo)
                smaller = min(ob["zone_high"] - ob["zone_low"], k["zone_high"] - k["zone_low"])
                if smaller > 0 and inter / smaller > overlap_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(ob)
        return kept

    bullish_obs = [b for b in order_blocks if b["direction"] == "bullish"]
    bearish_obs = [b for b in order_blocks if b["direction"] == "bearish"]
    active_bullish = _dedupe_overlapping([b for b in bullish_obs if not b["invalidated"]])
    active_bearish = _dedupe_overlapping([b for b in bearish_obs if not b["invalidated"]])
    active_bullish = sorted(active_bullish, key=lambda b: b["formed_bar"])[-MAX_ACTIVE_OB_PER_SIDE:]
    active_bearish = sorted(active_bearish, key=lambda b: b["formed_bar"])[-MAX_ACTIVE_OB_PER_SIDE:]

    last_choch = next((e for e in reversed(structure_events) if e["type"] == "CHoCH"), None)
    last_bos = next((e for e in reversed(structure_events) if e["type"] == "BOS"), None)

    # ── 7. ENTRY-CANDIDATE SELECTION ────────────────────────────────

    candidates = [b for b in (active_bullish if trend == "up" else active_bearish)
                  if not b["invalidated"]]
    if require_mtf_alignment and htf_trend in ("up", "down"):
        candidates = [b for b in candidates if b.get("htf_aligned")]
    if require_fvg_confluence:
        candidates = [b for b in candidates if b.get("fvg_confluence")]
    candidates = sorted(candidates, key=lambda b: (-b["score"], abs(cur_price - _zone_mid(b))))

    best_ob = candidates[0] if candidates else None

    # price currently sitting inside a zone?
    in_zone_now = False
    if best_ob is not None:
        in_zone_now = best_ob["zone_low"] <= cur_price <= best_ob["zone_high"]

    entry_plan = None
    if best_ob is not None:
        direction = "BUY" if best_ob["direction"] == "bullish" else "SELL"
        if direction == "BUY":
            entry_price = min(cur_price, best_ob["zone_high"])
            stop_loss = best_ob["zone_low"] - 0.25 * cur_atr
            risk = entry_price - stop_loss
            target = entry_price + 2.0 * risk if risk > 0 else None
        else:
            entry_price = max(cur_price, best_ob["zone_low"])
            stop_loss = best_ob["zone_high"] + 0.25 * cur_atr
            risk = stop_loss - entry_price
            target = entry_price - 2.0 * risk if risk > 0 else None
        rr = 2.0 if risk and risk > 0 else None
        entry_plan = {
            "direction": direction,
            "status": "ENTRY (price inside zone)" if in_zone_now else "WATCH (limit order at zone)",
            "order_block_bar": best_ob["formed_bar"],
            "order_block_date": str(dates.iloc[best_ob["formed_bar"]])[:10] if hasattr(dates, "iloc") else str(dates[best_ob["formed_bar"]])[:10],
            "zone_low": round(float(best_ob["zone_low"]), 4),
            "zone_high": round(float(best_ob["zone_high"]), 4),
            "entry_price": round(float(entry_price), 4),
            "stop_loss": round(float(stop_loss), 4),
            "take_profit": round(float(target), 4) if target is not None else None,
            "risk_reward": rr,
            "confidence_score": best_ob["score"],
            "tests_so_far": best_ob["tests"],
            "born_from": best_ob["type"],
        }

    # ── 8. OVERALL VERDICT ──────────────────────────────────────────
    if best_ob is None or trend is None:
        signal, confidence = "NEUTRAL", 0
        reason = "No aligned, unmitigated order block available in the current trend context."
    else:
        base_conf = best_ob["score"]
        if in_zone_now:
            signal = entry_plan["direction"]
            confidence = base_conf
            reason = (f"Price is trading inside a {best_ob['direction']} order block formed by a "
                      f"{best_ob['type']} on {entry_plan['order_block_date']}; structure trend is {trend}.")
        else:
            signal = "WATCH-" + entry_plan["direction"]
            confidence = max(0, base_conf - 15)
            test_note = "has not been tested yet" if best_ob["tests"] == 0 else \
                        f"has been tested {best_ob['tests']}x already and remains unmitigated"
            reason = (f"Nearest {best_ob['direction']} order block "
                      f"[{best_ob['zone_low']:.2f} - {best_ob['zone_high']:.2f}] {test_note}; "
                      f"price is not currently inside it; structure trend is {trend}.")

    # ── 9. TEXT REPORT ───────────────────────────────────────────────
    report_lines = []
    report_lines.append(f"{'='*70}")
    report_lines.append(f"ORDER BLOCK / STRUCTURE REPORT — {ticker}  ({period}, {interval})")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"Current price : {cur_price:.4f}")
    report_lines.append(f"ATR({ATR_PERIOD})     : {cur_atr:.4f}")
    report_lines.append(f"Structure trend: {trend.upper() if trend else 'UNDEFINED'}")
    if htf_trend in ("up", "down"):
        report_lines.append(f"Higher-TF trend: {htf_trend.upper()} "
                            f"({(htf_info or {}).get('higher_timeframe', '?')})  "
                            f"[mtf alignment required: {require_mtf_alignment}]")
    else:
        report_lines.append("Higher-TF trend: unavailable/undefined (no multi-timeframe filter applied)")
    unfilled_fvgs = [f for f in fvgs if not f["filled"]]
    report_lines.append(f"Unfilled FVGs   : {len(unfilled_fvgs)}  "
                        f"[fvg confluence required: {require_fvg_confluence}]")
    report_lines.append(f"Liquidity-sweep required for CHoCH OBs: {require_liquidity_sweep}")
    if last_choch:
        d = str(dates.iloc[last_choch['break_bar']])[:10] if hasattr(dates, "iloc") else str(dates[last_choch['break_bar']])[:10]
        report_lines.append(f"Last CHoCH     : {last_choch['direction'].upper()} on {d} @ {last_choch['level']:.4f}")
    if last_bos:
        d = str(dates.iloc[last_bos['break_bar']])[:10] if hasattr(dates, "iloc") else str(dates[last_bos['break_bar']])[:10]
        report_lines.append(f"Last BOS       : {last_bos['direction'].upper()} on {d} @ {last_bos['level']:.4f}")
    report_lines.append("-" * 70)
    report_lines.append(f"Active bullish order blocks : {len(active_bullish)}")
    for b in active_bullish:
        d = str(dates.iloc[b['formed_bar']])[:10] if hasattr(dates, "iloc") else str(dates[b['formed_bar']])[:10]
        report_lines.append(f"  [{d}] zone {b['zone_low']:.2f}-{b['zone_high']:.2f} "
                            f"| tests={b['tests']} | score={b['score']} | origin={b['type']}")
    report_lines.append(f"Active bearish order blocks : {len(active_bearish)}")
    for b in active_bearish:
        d = str(dates.iloc[b['formed_bar']])[:10] if hasattr(dates, "iloc") else str(dates[b['formed_bar']])[:10]
        report_lines.append(f"  [{d}] zone {b['zone_low']:.2f}-{b['zone_high']:.2f} "
                            f"| tests={b['tests']} | score={b['score']} | origin={b['type']}")
    report_lines.append("-" * 70)
    report_lines.append(f"VERDICT: {signal}  (confidence {confidence}/100)")
    report_lines.append(f"Reason: {reason}")
    if entry_plan:
        report_lines.append(f"Entry plan: {entry_plan['direction']} | status={entry_plan['status']}")
        report_lines.append(f"  Entry={entry_plan['entry_price']}  SL={entry_plan['stop_loss']}  "
                            f"TP={entry_plan['take_profit']}  R:R={entry_plan['risk_reward']}")
    report_lines.append("=" * 70)
    text_report = "\n".join(report_lines)

    # ── 10. CHARTS ────────────────────────────────────────────────────
    chart_paths = {"full": None, "recent": None}
    if save_chart:
        chart_paths = _draw_charts(df, ticker, period, interval, o, h, l, c, dates,
                                    swing_highs, swing_lows, structure_events,
                                    order_blocks, active_bullish, active_bearish,
                                    best_ob, entry_plan, trend, RECENT_BARS)

    # ── 11. JSON-SAFE RETURN ──────────────────────────────────────────
    def _clean_ob(b):
        d = str(dates.iloc[b['formed_bar']])[:10] if hasattr(dates, "iloc") else str(dates[b['formed_bar']])[:10]
        return {
            "direction": b["direction"], "origin": b["type"],
            "formed_bar": int(b["formed_bar"]), "date": d,
            "zone_low": round(float(b["zone_low"]), 4),
            "zone_high": round(float(b["zone_high"]), 4),
            "tests": int(b["tests"]), "mitigated": bool(b["mitigated"]),
            "invalidated": bool(b["invalidated"]), "score": int(b["score"]),
            "volume_confirmed": bool(b.get("volume_confirmed", False)),
            "liquidity_swept": b.get("liquidity_swept"),
            "fvg_confluence": bool(b.get("fvg_confluence", False)),
            "htf_aligned": b.get("htf_aligned"),
        }

    def _clean_event(e):
        d = str(dates.iloc[e['break_bar']])[:10] if hasattr(dates, "iloc") else str(dates[e['break_bar']])[:10]
        return {"type": e["type"], "direction": e["direction"], "date": d,
                "level": round(float(e["level"]), 4), "bar": int(e["break_bar"]),
                "liquidity_swept": e.get("liquidity_swept")}

    def _clean_fvg(f):
        d = str(dates.iloc[f['formed_bar']])[:10] if hasattr(dates, "iloc") else str(dates[f['formed_bar']])[:10]
        return {
            "direction": f["direction"], "formed_bar": int(f["formed_bar"]), "date": d,
            "top": round(float(f["top"]), 4), "bottom": round(float(f["bottom"]), 4),
            "filled": bool(f["filled"]),
        }

    return {
        "ticker": ticker, "period": period, "interval": interval,
        "current_price": cur_price, "atr": round(cur_atr, 4),
        "trend": trend if trend else "undefined",
        "htf_trend": htf_trend if htf_trend else "undefined",
        "htf_info": htf_info,
        "signal": signal, "confidence": int(confidence), "reason": reason,
        "entry_plan": entry_plan,
        "active_bullish_order_blocks": [_clean_ob(b) for b in active_bullish],
        "active_bearish_order_blocks": [_clean_ob(b) for b in active_bearish],
        "active_fvgs": [_clean_fvg(f) for f in fvgs if not f["filled"]][-20:],
        "recent_structure_events": [_clean_event(e) for e in structure_events[-10:]],
        "filters_applied": {
            "require_liquidity_sweep": require_liquidity_sweep,
            "require_fvg_confluence": require_fvg_confluence,
            "require_mtf_alignment": require_mtf_alignment,
        },
        "text_report": text_report,
        "chart_paths": chart_paths,
        "series": {
            **ohlc_payload(df, o, h, l, c, v),
            "swing_high_bars": [i for i, _ in swing_highs],
            "swing_low_bars": [i for i, _ in swing_lows],
        },
    }


def _zone_mid(ob):
    return (ob["zone_low"] + ob["zone_high"]) / 2.0


def _tag_order_block(df, o, h, l, c, v, avg_vol20, atr, leg_start, break_bar,
                      direction, ob_type, lookback_cap, max_atr_mult,
                      vol_spike_mult, out_list, liquidity_swept=None):
    """
    Locate the order-block candle: the last opposite-colour candle between
    the pivot that started the impulsive leg (leg_start) and the bar that
    broke structure (break_bar, exclusive), searching back at most
    lookback_cap bars from break_bar.
    Bullish OB -> last bearish (close < open) candle before the up-break.
    Bearish OB -> last bullish (close > open) candle before the down-break.
    """
    search_from = max(leg_start, break_bar - lookback_cap)
    ob_idx = None
    for j in range(break_bar - 1, search_from - 1, -1):
        is_bear = c[j] < o[j]
        is_bull = c[j] > o[j]
        if direction == "bullish" and is_bear:
            ob_idx = j
            break
        if direction == "bearish" and is_bull:
            ob_idx = j
            break
    if ob_idx is None:
        return  # no clean opposite candle found -> skip, do not guess

    zone_low = float(l[ob_idx])
    zone_high = float(h[ob_idx])
    zone_height = zone_high - zone_low
    if atr[ob_idx] > 0 and zone_height > max_atr_mult * atr[ob_idx]:
        return  # zone too wide to be a clean OB -> discard rather than guess

    vol_confirmed = bool(avg_vol20[break_bar] > 0 and v[break_bar] >= vol_spike_mult * avg_vol20[break_bar])

    out_list.append({
        "direction": direction, "type": ob_type,
        "formed_bar": ob_idx, "break_bar": break_bar,
        "zone_low": zone_low, "zone_high": zone_high,
        "volume_confirmed": vol_confirmed,
        "liquidity_swept": liquidity_swept,
    })


def _draw_charts(df, ticker, period, interval, o, h, l, c, dates,
                  swing_highs, swing_lows, structure_events, order_blocks,
                  active_bullish, active_bearish, best_ob, entry_plan, trend, RECENT_BARS):
    n = len(df)
    os.makedirs("static", exist_ok=True)
    base = f"static/{ticker}_OB"
    full_chart = f"{base}_full.png"
    recent_chart = f"{base}_recent.png"

    def _plot(ax, start, end, ob_list, sh, sl, events, wide):
        idxs = list(range(start, end))
        for i in idxs:
            x = i - start
            col = "#26a69a" if c[i] >= o[i] else "#ef5350"
            ax.plot([x, x], [o[i], c[i]], linewidth=6 if not wide else 3,
                    color=col, solid_capstyle="round", zorder=3)
            ax.plot([x, x], [l[i], h[i]], linewidth=1.5 if not wide else 1,
                    color=col, alpha=0.7, zorder=2)
        for i, price in sh:
            if start <= i < end:
                ax.scatter(i - start, price, marker="v", color="#ef5350", s=40, alpha=0.6, zorder=4)
        for i, price in sl:
            if start <= i < end:
                ax.scatter(i - start, price, marker="^", color="#26a69a", s=40, alpha=0.6, zorder=4)
        for ob in ob_list:
            if ob["formed_bar"] < start - 5 or ob["formed_bar"] >= end:
                continue
            x0 = max(0, ob["formed_bar"] - start)
            x1 = end - start
            col = "#26a69a" if ob["direction"] == "bullish" else "#ef5350"
            alpha = 0.28 if not ob["invalidated"] else 0.08
            rect = mpatches.Rectangle((x0, ob["zone_low"]), max(1, x1 - x0), ob["zone_high"] - ob["zone_low"],
                                       facecolor=col, edgecolor=col, linewidth=1.0,
                                       alpha=alpha, zorder=1,
                                       linestyle="-" if not ob["invalidated"] else "--")
            ax.add_patch(rect)
            label = f"{'OB+' if ob['direction']=='bullish' else 'OB-'}{'★' if not ob['mitigated'] else ''}"
            ax.text(x0 + 0.2, ob["zone_high"], label, fontsize=6.5, color=col,
                    va="bottom", fontweight="bold", zorder=5)
        for e in events:
            if start <= e["break_bar"] < end:
                col = "#42a5f5" if e["direction"] == "bullish" else "#ffa726"
                ax.axvline(e["break_bar"] - start, color=col, lw=0.9, ls=":", alpha=0.7, zorder=2)
                ax.text(e["break_bar"] - start, e["level"], f" {e['type']}", fontsize=6.5,
                        color=col, rotation=90, va="bottom", ha="center", zorder=5)

    # ---------- FULL CHART ----------
    fig = plt.figure(figsize=(20, 11), facecolor="#0d1117")
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#0d1117")
    _plot(ax, 0, n, order_blocks, swing_highs, swing_lows, structure_events, wide=True)

    cur = float(c[-1])
    ax.axhline(cur, color="white", lw=0.8, ls=":", alpha=0.8)
    ax.text(n + 0.5, cur, f" {cur:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
    ax.set_title(f"{ticker} · ({period}) · Order Blocks & Market Structure  ·  Trend: "
                 f"{(trend or 'undefined').upper()}", color="white", fontsize=13, pad=10)
    ax.set_ylabel("Price", color="#9e9e9e", fontsize=9)
    tick_step = max(1, n // 14)
    tick_pos = list(range(0, n, tick_step))
    tick_lbl = [str(df.loc[i, "Date"])[:10] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=35, ha="right", fontsize=7.5, color="#9e9e9e")
    ax.tick_params(colors="#9e9e9e", labelsize=8)
    ax.grid(axis="y", color="#1a1a1a", lw=0.5)
    ax.spines[:].set_color("#2a2a2a")
    ax.set_xlim(-1, n + 8)

    leg = [
        mpatches.Patch(color="#26a69a", label="Bullish Order Block"),
        mpatches.Patch(color="#ef5350", label="Bearish Order Block"),
        mpatches.Patch(color="#42a5f5", label="Bullish BOS/CHoCH"),
        mpatches.Patch(color="#ffa726", label="Bearish BOS/CHoCH"),
    ]
    ax.legend(handles=leg, loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
              labelcolor="white", fontsize=8)
    ax.text(0.5, -0.07, "\u26a0  For educational purposes only — not financial advice.",
            ha="center", va="top", fontsize=7.5, color="#555", transform=ax.transAxes, style="italic")
    plt.savefig(full_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)

    # ---------- RECENT (ZOOMED) CHART ----------
    recent_start = max(0, n - RECENT_BARS)
    fig2 = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    ax2 = fig2.add_subplot(111)
    ax2.set_facecolor("#0d1117")
    _plot(ax2, recent_start, n, order_blocks, swing_highs, swing_lows, structure_events, wide=False)

    if best_ob is not None and entry_plan is not None:
        x0 = max(0, best_ob["formed_bar"] - recent_start)
        col = "#ffd54f"
        ax2.axhline(entry_plan["entry_price"], color=col, lw=1.1, ls="--", alpha=0.9)
        ax2.text(n - recent_start + 0.3, entry_plan["entry_price"], f" ENTRY {entry_plan['entry_price']:.2f}",
                  color=col, fontsize=8, fontweight="bold", va="center")
        ax2.axhline(entry_plan["stop_loss"], color="#ef5350", lw=1.0, ls="--", alpha=0.8)
        ax2.text(n - recent_start + 0.3, entry_plan["stop_loss"], f" SL {entry_plan['stop_loss']:.2f}",
                  color="#ef5350", fontsize=7.5, va="center")
        if entry_plan["take_profit"] is not None:
            ax2.axhline(entry_plan["take_profit"], color="#26a69a", lw=1.0, ls="--", alpha=0.8)
            ax2.text(n - recent_start + 0.3, entry_plan["take_profit"], f" TP {entry_plan['take_profit']:.2f}",
                      color="#26a69a", fontsize=7.5, va="center")

    nr = n - recent_start
    cur2 = float(c[-1])
    ax2.axhline(cur2, color="white", lw=0.8, ls=":", alpha=0.8)
    ax2.text(nr - 0.5, cur2, f" {cur2:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
    ax2.set_title(f"{ticker} · Recent {nr} Candles · Order Block Entry Zoom", color="white", fontsize=12, pad=10)
    ax2.set_ylabel("Price", color="#9e9e9e", fontsize=9)
    tick_lbl2 = [str(df.loc[i, "Date"])[:10] for i in range(recent_start, n)]
    ax2.set_xticks(list(range(nr)))
    ax2.set_xticklabels(tick_lbl2, rotation=45, ha="right", fontsize=6.5, color="#9e9e9e")
    ax2.tick_params(colors="#9e9e9e", labelsize=8)
    ax2.grid(axis="y", color="#1a1a1a", lw=0.5)
    ax2.spines[:].set_color("#2a2a2a")
    ax2.set_xlim(-0.5, nr + 6)

    if entry_plan:
        title_line = f"{entry_plan['direction']} plan  |  {entry_plan['status']}  |  score {entry_plan['confidence_score']}/100"
        ax2.text(0.01, 1.03, title_line, transform=ax2.transAxes, fontsize=9, fontweight="bold",
                  color="#ffd54f", ha="left", va="bottom")

    plt.savefig(recent_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig2)

    return {"full": full_chart, "recent": recent_chart}


# ────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT (downloads data, then delegates to _run_analysis)
# ────────────────────────────────────────────────────────────────────
def analyze_order_blocks(ticker, period='1y', interval='1d', save_chart=True,
                          use_mtf_filter=True,
                          require_liquidity_sweep=True,
                          require_fvg_confluence=False,
                          require_mtf_alignment=False):
    """
    Run the order-block / market-structure analysis and return a
    JSON-serialisable dict. Safe to call from main.py or wrap in an API
    endpoint (e.g. Flask/FastAPI `return jsonify(analyze_order_blocks(...))`).

    New confluence toggles (all optional, all default to sensible values):
      use_mtf_filter          fetch a higher timeframe and score/tag each OB
                              by whether it agrees with that higher trend
                              (this alone does NOT exclude anything unless
                              require_mtf_alignment is also True)
      require_liquidity_sweep only tag an order block for a CHoCH if a
                              stop-hunt sweep of the opposing prior swing
                              preceded it (filters premature reversal calls)
      require_fvg_confluence hard-filter entry candidates to only those
                              overlapping an unfilled, same-direction FVG
      require_mtf_alignment  hard-filter entry candidates to only those
                              agreeing with the higher-timeframe trend
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})

    htf_trend, htf_info = None, None
    if use_mtf_filter:
        try:
            htf_trend, htf_info = get_higher_timeframe_trend(ticker, interval, period)
        except Exception as e:
            htf_trend, htf_info = None, {"error": str(e)}

    return _run_analysis(df, ticker, period, interval, save_chart,
                          htf_trend=htf_trend, htf_info=htf_info,
                          require_liquidity_sweep=require_liquidity_sweep,
                          require_fvg_confluence=require_fvg_confluence,
                          require_mtf_alignment=require_mtf_alignment)


if __name__ == "__main__":
    tkr = input("Enter stock ticker: ").strip().upper()
    result = analyze_order_blocks(tkr, save_chart=True)
    if "error" in result:
        print(result["error"])
    else:
        print(result["text_report"])
        print("Charts:", result["chart_paths"])