"""
ICHIMOKU CLOUD ANALYSIS  ·  NEPSE / any ticker
────────────────────────────────────────────────────────────────────
...
"""
import matplotlib; 
matplotlib.use('Agg')
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import os

from series_utils import clean_list, ohlc_payload

def analyze_ichimoku(ticker, period='2y', interval='1d', save_chart=True):
    """
    Run Ichimoku Cloud analysis and return structured results.
    """
    # ── CONFIG ──────────────────────────────────
    TENKAN_PERIOD  = 9
    KIJUN_PERIOD   = 26
    SENKOU_B_PERIOD = 52
    DISPLACEMENT   = 26
    PIVOT_ORDER  = 3
    RECENT_BARS  = 30
    THICK_CLOUD_PCT = 0.015

    # ── 1. DOWNLOAD ────────────────────────────
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df = df.reset_index()
    n = len(df)
    if n == 0:
        return {"error": f"No data for {ticker}"}
    if n < SENKOU_B_PERIOD + DISPLACEMENT + 10:
        return {"error": f"Insufficient data: need at least {SENKOU_B_PERIOD + DISPLACEMENT + 10} bars, got {n}"}

    # ── 2. ICHIMOKU CALCULATIONS ──────────────
    def donchian_mid(series_high, series_low, period):
        return (series_high.rolling(period).max() + series_low.rolling(period).min()) / 2

    close_s = df["Close"]
    high_s  = df["High"]
    low_s   = df["Low"]

    tenkan = donchian_mid(high_s, low_s, TENKAN_PERIOD)
    kijun  = donchian_mid(high_s, low_s, KIJUN_PERIOD)
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = donchian_mid(high_s, low_s, SENKOU_B_PERIOD)
    chikou = close_s.copy()

    df["Tenkan"]   = tenkan.values
    df["Kijun"]    = kijun.values
    df["SpanA_raw"]= span_a_raw.values
    df["SpanB_raw"]= span_b_raw.values
    df["Chikou"]   = chikou.values

    total_bars = n + DISPLACEMENT
    span_a_plot = np.full(total_bars, np.nan)
    span_b_plot = np.full(total_bars, np.nan)
    for i in range(n):
        target = i + DISPLACEMENT
        if target < total_bars:
            if not np.isnan(df["SpanA_raw"].iloc[i]):
                span_a_plot[target] = df["SpanA_raw"].iloc[i]
            if not np.isnan(df["SpanB_raw"].iloc[i]):
                span_b_plot[target] = df["SpanB_raw"].iloc[i]

    span_a_current = span_a_plot[:n]
    span_b_current = span_b_plot[:n]

    tenkan_v  = df["Tenkan"].values
    kijun_v   = df["Kijun"].values
    close_v   = df["Close"].values
    chikou_v  = df["Chikou"].values

    # ── 3. HELPERS ─────────────────────────────
    def cloud_position(price, sa, sb):
        if np.isnan(sa) or np.isnan(sb):
            return "unknown"
        top = max(sa, sb); bottom = min(sa, sb)
        if price > top: return "above"
        elif price < bottom: return "below"
        else: return "inside"

    def tk_strength(bar, sa_arr, sb_arr, price_arr):
        sa = sa_arr[bar]; sb = sb_arr[bar]; p = price_arr[bar]
        pos = cloud_position(p, sa, sb)
        if pos == "above": return "STRONG"
        elif pos == "inside": return "NEUTRAL"
        else: return "WEAK"

    def kumo_thick_strong(bar, sa_arr, sb_arr, price_arr):
        sa = sa_arr[bar]; sb = sb_arr[bar]; p = price_arr[bar]
        if np.isnan(sa) or np.isnan(sb) or p == 0: return False
        return abs(sa - sb) / p >= THICK_CLOUD_PCT

    # ── 4. SIGNAL DETECTION ────────────────────
    # A) TK CROSS
    def find_tk_crosses(tenkan_arr, kijun_arr, sa_arr, sb_arr, close_arr):
        crosses = []
        for i in range(1, len(tenkan_arr)):
            t0, t1 = tenkan_arr[i-1], tenkan_arr[i]
            k0, k1 = kijun_arr[i-1], kijun_arr[i]
            if any(np.isnan(v) for v in [t0, t1, k0, k1]):
                continue
            prev_diff = t0 - k0
            cur_diff  = t1 - k1
            if prev_diff <= 0 and cur_diff > 0:
                direction = "BUY"
            elif prev_diff >= 0 and cur_diff < 0:
                direction = "SELL"
            else:
                continue
            strength = tk_strength(i, sa_arr, sb_arr, close_arr)
            crosses.append({"bar": i, "direction": direction, "strength": strength,
                            "tenkan": t1, "kijun": k1})
        return crosses

    # B) CHIKOU CROSS
    def find_chikou_crosses(close_arr, displacement):
        crosses = []
        for i in range(displacement + 1, len(close_arr)):
            ch0 = close_arr[i - 1]; ch1 = close_arr[i]
            pr0 = close_arr[i - 1 - displacement]
            pr1 = close_arr[i     - displacement]
            prev_diff = ch0 - pr0
            cur_diff  = ch1 - pr1
            if prev_diff <= 0 and cur_diff > 0:
                direction = "BUY"
            elif prev_diff >= 0 and cur_diff < 0:
                direction = "SELL"
            else:
                continue
            crosses.append({"bar": i, "direction": direction,
                            "chikou": ch1, "price_under": pr1})
        return crosses

    # C) PRICE / KIJUN CROSS
    def find_kijun_crosses(close_arr, kijun_arr, sa_arr, sb_arr):
        crosses = []
        for i in range(1, len(close_arr)):
            c0, c1 = close_arr[i-1], close_arr[i]
            k0, k1 = kijun_arr[i-1], kijun_arr[i]
            if np.isnan(k0) or np.isnan(k1):
                continue
            prev_diff = c0 - k0
            cur_diff  = c1 - k1
            if prev_diff <= 0 and cur_diff > 0:
                direction = "BUY"
            elif prev_diff >= 0 and cur_diff < 0:
                direction = "SELL"
            else:
                continue
            strength = tk_strength(i, sa_arr, sb_arr, close_arr)
            crosses.append({"bar": i, "direction": direction, "strength": strength,
                            "close": c1, "kijun": k1})
        return crosses

    # D) KUMO BREAKOUT
    def find_kumo_breakouts(close_arr, sa_arr, sb_arr):
        breakouts = []
        prev_pos = None
        for i in range(len(close_arr)):
            if np.isnan(sa_arr[i]) or np.isnan(sb_arr[i]):
                prev_pos = None
                continue
            pos = cloud_position(close_arr[i], sa_arr[i], sb_arr[i])
            if prev_pos is not None and prev_pos != pos:
                if prev_pos in ("below", "inside") and pos == "above":
                    direction = "BUY"
                    strength = "STRONG" if kumo_thick_strong(i, sa_arr, sb_arr, close_arr) else "WEAK"
                    breakouts.append({"bar": i, "direction": direction, "strength": strength,
                                      "close": close_arr[i],
                                      "cloud_top": max(sa_arr[i], sb_arr[i]),
                                      "cloud_bot": min(sa_arr[i], sb_arr[i]),
                                      "prev_pos": prev_pos})
                elif prev_pos in ("above", "inside") and pos == "below":
                    direction = "SELL"
                    strength = "STRONG" if kumo_thick_strong(i, sa_arr, sb_arr, close_arr) else "WEAK"
                    breakouts.append({"bar": i, "direction": direction, "strength": strength,
                                      "close": close_arr[i],
                                      "cloud_top": max(sa_arr[i], sb_arr[i]),
                                      "cloud_bot": min(sa_arr[i], sb_arr[i]),
                                      "prev_pos": prev_pos})
            prev_pos = pos
        return breakouts

    # E) KUMO TWIST
    def find_kumo_twists(span_a_full, span_b_full, total_len):
        twists = []
        for i in range(1, total_len):
            a0, a1 = span_a_full[i-1], span_a_full[i]
            b0, b1 = span_b_full[i-1], span_b_full[i]
            if any(np.isnan(v) for v in [a0, a1, b0, b1]):
                continue
            prev_diff = a0 - b0
            cur_diff  = a1 - b1
            if prev_diff <= 0 and cur_diff > 0:
                direction = "BUY"
            elif prev_diff >= 0 and cur_diff < 0:
                direction = "SELL"
            else:
                continue
            twists.append({"bar": i, "direction": direction,
                           "span_a": a1, "span_b": b1,
                           "is_future": i >= n})
        return twists

    tk_crosses      = find_tk_crosses(tenkan_v, kijun_v, span_a_current, span_b_current, close_v)
    chikou_crosses  = find_chikou_crosses(close_v, DISPLACEMENT)
    kijun_crosses   = find_kijun_crosses(close_v, kijun_v, span_a_current, span_b_current)
    kumo_breakouts  = find_kumo_breakouts(close_v, span_a_current, span_b_current)
    kumo_twists     = find_kumo_twists(span_a_plot, span_b_plot, total_bars)

    # ── 5. CURRENT STATE ────────────────────────
    last = n - 1
    cur_price  = float(close_v[last])
    cur_tenkan = float(tenkan_v[last])
    cur_kijun  = float(kijun_v[last])
    cur_sa     = float(span_a_current[last]) if not np.isnan(span_a_current[last]) else float("nan")
    cur_sb     = float(span_b_current[last]) if not np.isnan(span_b_current[last]) else float("nan")
    cur_cloud_top = max(cur_sa, cur_sb) if not (np.isnan(cur_sa) or np.isnan(cur_sb)) else float("nan")
    cur_cloud_bottom = min(cur_sa, cur_sb) if not (np.isnan(cur_sa) or np.isnan(cur_sb)) else float("nan")
    cur_pos = cloud_position(cur_price, cur_sa, cur_sb)
    cur_cloud_bullish = (not np.isnan(cur_sa) and not np.isnan(cur_sb) and cur_sa > cur_sb)

    chikou_compare_bar = last - DISPLACEMENT
    chikou_vs_price = None
    if chikou_compare_bar >= 0:
        chikou_val = float(close_v[last])
        price_26ago = float(close_v[chikou_compare_bar])
        chikou_vs_price = "above" if chikou_val > price_26ago else ("below" if chikou_val < price_26ago else "at")

    # ── Consensus score ──────────────────────────
    score = 0
    reasons = []
    if cur_pos == "above":
        score += 1; reasons.append("Price ABOVE cloud (+1 bullish)")
    elif cur_pos == "below":
        score -= 1; reasons.append("Price BELOW cloud (−1 bearish)")
    else:
        reasons.append("Price INSIDE cloud (neutral)")

    if not (np.isnan(cur_tenkan) or np.isnan(cur_kijun)):
        if cur_tenkan > cur_kijun:
            score += 1; reasons.append("Tenkan > Kijun (+1 bullish)")
        elif cur_tenkan < cur_kijun:
            score -= 1; reasons.append("Tenkan < Kijun (−1 bearish)")
        else:
            reasons.append("Tenkan = Kijun (neutral)")

    if not (np.isnan(cur_sa) or np.isnan(cur_sb)):
        if cur_sa > cur_sb:
            score += 1; reasons.append("Cloud is GREEN / bullish (SpanA > SpanB, +1)")
        else:
            score -= 1; reasons.append("Cloud is RED / bearish (SpanA < SpanB, −1)")

    if chikou_vs_price == "above":
        score += 1; reasons.append("Chikou ABOVE price 26 bars ago (+1 bullish)")
    elif chikou_vs_price == "below":
        score -= 1; reasons.append("Chikou BELOW price 26 bars ago (−1 bearish)")

    if not np.isnan(cur_kijun):
        if cur_price > cur_kijun:
            score += 1; reasons.append("Price ABOVE Kijun (+1 bullish)")
        elif cur_price < cur_kijun:
            score -= 1; reasons.append("Price BELOW Kijun (−1 bearish)")

    if not np.isnan(cur_tenkan):
        if cur_price > cur_tenkan:
            score += 1; reasons.append("Price ABOVE Tenkan (+1 bullish)")
        elif cur_price < cur_tenkan:
            score -= 1; reasons.append("Price BELOW Tenkan (−1 bearish)")

    if score >= 4: consensus = "STRONGLY BULLISH"
    elif score >= 2: consensus = "BULLISH"
    elif score <= -4: consensus = "STRONGLY BEARISH"
    elif score <= -2: consensus = "BEARISH"
    else: consensus = "MIXED / NEUTRAL"

    # ── Recent window ────────────────────────────
    recent_start = max(0, n - RECENT_BARS)
    def recent(lst): return [x for x in lst if x["bar"] >= recent_start]
    recent_tk       = recent(tk_crosses)
    recent_chikou   = recent(chikou_crosses)
    recent_kijun_x  = recent(kijun_crosses)
    recent_breakout = recent(kumo_breakouts)
    recent_twists_past = [t for t in kumo_twists if not t["is_future"] and t["bar"] >= recent_start]

    # ── 6. BUILD VERDICT ────────────────────────
    def build_verdict():
        if cur_pos == "inside":
            verdict = "WAIT  —  Price inside the Kumo (no-trade zone)"
            action = "Do NOT enter long or short. The cloud is congestion. Wait for a decisive close outside."
            watch_levels = []
            if not np.isnan(cur_cloud_top):
                watch_levels.append(f"Bull breakout above  : {cur_cloud_top:.2f}  (Kumo top)   → then consider BUY")
                watch_levels.append(f"Bear breakdown below : {cur_cloud_bottom:.2f}  (Kumo bottom) → then consider SELL")
            reasons_v = ["Price is inside the cloud — Ichimoku treats this as a congestion / indecision zone.",
                         "Hosoda's rule: trade only when price is clearly above or below the Kumo."]
            upcoming_twist = next((t for t in kumo_twists if t["is_future"] and (t["bar"] - (n - 1)) <= 10), None)
            if upcoming_twist:
                d = "bullish" if upcoming_twist["direction"] == "BUY" else "bearish"
                bars_ahead = upcoming_twist["bar"] - (n - 1)
                reasons_v.append(f"Future Kumo twist ({d}) in ~{bars_ahead} bars — watch for directional resolution then.")
            return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                    "reasons": reasons_v, "direction": "NEUTRAL", "confidence": 0}

        if cur_pos == "above":
            recent_bull_signal = any(x["direction"] == "BUY" for x in recent_tk + recent_breakout + recent_chikou)
            if score >= 4 and recent_bull_signal:
                verdict = "BUY  —  Strong bullish confluence"
                action = "All Ichimoku conditions confirmed. Current price is a valid entry."
                watch_levels = []
                if not np.isnan(cur_tenkan):
                    watch_levels.append(f"Stop loss (tight)   : {cur_tenkan:.2f}  (below Tenkan-sen)")
                if not np.isnan(cur_kijun):
                    watch_levels.append(f"Stop loss (standard): {cur_kijun:.2f}  (below Kijun-sen / Base Line)")
                if not np.isnan(cur_cloud_top):
                    watch_levels.append(f"Stop loss (wide)    : {cur_cloud_top:.2f}  (below Kumo top)")
                reasons_v = [f"Score {score}/6 — all six Ichimoku conditions bullish."]
                if recent_chikou: reasons_v.append("Chikou confirmed above price.")
                if recent_breakout: reasons_v.append("Recent Kumo breakout in the window.")
                if recent_tk: reasons_v.append("Recent TK bullish cross in the window.")
                return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                        "reasons": reasons_v, "direction": "BUY", "confidence": 85}

            elif score >= 2:
                verdict = "WAIT  —  Bullish trend, wait for a pullback entry"
                action = "Trend is up but no fresh signal recently. Do NOT chase. Wait for price to dip to a support level below."
                watch_levels = []
                if not np.isnan(cur_tenkan):
                    watch_levels.append(f"Entry zone 1 (shallow pullback) : {cur_tenkan:.2f}  (Tenkan-sen / Conversion Line)")
                if not np.isnan(cur_kijun):
                    watch_levels.append(f"Entry zone 2 (deeper pullback)  : {cur_kijun:.2f}  (Kijun-sen / Base Line)  ← stronger level")
                if not np.isnan(cur_cloud_top):
                    watch_levels.append(f"Entry zone 3 (full retest)      : {cur_cloud_top:.2f}  (Kumo top — cloud re-test)")
                reasons_v = [f"Score {score}/6 — trend is bullish but no recent signal. Chasing is risky.",
                             "Kijun-sen is the primary mean-reversion level in Ichimoku — it acts like a magnet.",
                             "A pullback to Kijun with price holding above the cloud = ideal Hosoda entry."]
                return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                        "reasons": reasons_v, "direction": "BUY", "confidence": 60}

            else:
                verdict = "CAUTION  —  Above cloud but momentum weakening"
                action = "Price is above the cloud but internal conditions are conflicting. Reduce risk. Watch closely."
                watch_levels = []
                if not np.isnan(cur_kijun):
                    watch_levels.append(f"Key support : {cur_kijun:.2f}  (Kijun-sen — a close below = warning)")
                if not np.isnan(cur_cloud_top):
                    watch_levels.append(f"Cloud re-entry risk at : {cur_cloud_top:.2f}  (Kumo top)")
                reasons_v = [f"Score {score}/6 — above the cloud but signals are mixed."]
                if cur_tenkan < cur_kijun:
                    reasons_v.append("Tenkan < Kijun: short-term momentum has already turned bearish — a bearish TK cross.")
                if chikou_vs_price == "below":
                    reasons_v.append("Chikou is below past price — lack of lagging confirmation is a red flag.")
                return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                        "reasons": reasons_v, "direction": "NEUTRAL", "confidence": 30}

        if cur_pos == "below":
            recent_bear_signal = any(x["direction"] == "SELL" for x in recent_tk + recent_breakout + recent_chikou)
            if score <= -4 and recent_bear_signal:
                verdict = "SELL / AVOID  —  Strong bearish confluence"
                action = "All Ichimoku conditions bearish. Avoid longs. If shorting is possible, this is a valid setup."
                watch_levels = []
                if not np.isnan(cur_tenkan):
                    watch_levels.append(f"Short stop loss (tight)   : {cur_tenkan:.2f}  (above Tenkan-sen)")
                if not np.isnan(cur_kijun):
                    watch_levels.append(f"Short stop loss (standard): {cur_kijun:.2f}  (above Kijun-sen)")
                if not np.isnan(cur_cloud_bottom):
                    watch_levels.append(f"Short stop loss (wide)    : {cur_cloud_bottom:.2f}  (above Kumo bottom)")
                reasons_v = [f"Score {score}/6 — all six Ichimoku conditions bearish."]
                if recent_chikou: reasons_v.append("Chikou confirmed below price.")
                if recent_breakout: reasons_v.append("Recent Kumo breakdown in the window.")
                if recent_tk: reasons_v.append("Recent TK bearish cross in the window.")
                return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                        "reasons": reasons_v, "direction": "SELL", "confidence": 85}

            elif score <= -2:
                verdict = "WAIT  —  Downtrend, wait for a bounce to resistance before shorting"
                action = "Trend is down. For long positions: STAY OUT — do not try to catch a falling knife."
                watch_levels = []
                if not np.isnan(cur_tenkan):
                    watch_levels.append(f"Resistance 1 (shallow bounce): {cur_tenkan:.2f}  (Tenkan-sen)")
                if not np.isnan(cur_kijun):
                    watch_levels.append(f"Resistance 2 (deeper bounce) : {cur_kijun:.2f}  (Kijun-sen)")
                if not np.isnan(cur_cloud_bottom):
                    watch_levels.append(f"Resistance 3 (cloud re-test) : {cur_cloud_bottom:.2f}  (Kumo bottom — cloud underbelly)")
                reasons_v = [f"Score {score}/6 — trend is bearish. Kijun acts as overhead resistance in a downtrend.",
                             "Watch for a bounce to Kijun that FAILS to break above it → confirms downtrend continuation."]
                return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                        "reasons": reasons_v, "direction": "SELL", "confidence": 60}

            else:
                verdict = "CAUTION  —  Below cloud but signals conflicting"
                action = "Below the cloud is structurally bearish but conditions are not fully aligned. Sit out and observe."
                watch_levels = []
                if not np.isnan(cur_cloud_bottom):
                    watch_levels.append(f"Bearish confirmation if price stays below : {cur_cloud_bottom:.2f}  (Kumo bottom)")
                if not np.isnan(cur_kijun):
                    watch_levels.append(f"Potential recovery signal if price reclaims: {cur_kijun:.2f}  (Kijun-sen)")
                reasons_v = [f"Score {score}/6 — mixed despite being below the cloud. Not a clean setup."]
                return {"verdict": verdict, "action": action, "watch_levels": watch_levels,
                        "reasons": reasons_v, "direction": "NEUTRAL", "confidence": 30}

        return {"verdict": "INSUFFICIENT DATA", "action": "Not enough bars to calculate the Kumo. Use a longer period.",
                "watch_levels": [], "reasons": ["Span A or Span B is NaN at the current bar."],
                "direction": "NEUTRAL", "confidence": 0}

    verdict_dict = build_verdict()
    signal = verdict_dict["direction"]
    confidence = verdict_dict["confidence"]

    # ── 7. CHARTS ────────────────────────────────
    chart_paths = {"full": None, "recent": None}
    if save_chart:
        os.makedirs("static", exist_ok=True)
        base = f"static/{ticker}_Ichimoku"
        full_chart = f"{base}_full.png"
        recent_chart = f"{base}_recent.png"

        # ----- Full chart (adapted) -----
        BG = "#0d1117"
        BULL_CLR = "#26a69a"; BEAR_CLR = "#ef5350"
        TENKAN_C = "#e91e63"; KIJUN_C = "#2196f3"; CHIKOU_C = "#ab47bc"
        SPAN_A_C = "#26a69a"; SPAN_B_C = "#ef5350"
        KUMO_BULL = "#26a69a"; KUMO_BEAR = "#ef5350"
        FUTURE_ALPHA = 0.10; CLOUD_ALPHA = 0.20

        x_all = np.arange(total_bars)
        x_cur = np.arange(n)

        fig = plt.figure(figsize=(22, 13), facecolor=BG)
        gs  = gridspec.GridSpec(3, 1, height_ratios=[5, 1.5, 0.9], hspace=0.04)
        ax_p = fig.add_subplot(gs[0])
        ax_v = fig.add_subplot(gs[1], sharex=ax_p)
        ax_s = fig.add_subplot(gs[2])
        for ax in [ax_p, ax_v, ax_s]:
            ax.set_facecolor(BG)

        for i in range(n):
            o = float(df.loc[i, "Open"]); c = float(df.loc[i, "Close"])
            h = float(df.loc[i, "High"]); lo = float(df.loc[i, "Low"])
            clr = BULL_CLR if c >= o else BEAR_CLR
            ax_p.plot([i, i], [o, c],   linewidth=3.5, color=clr, solid_capstyle="round", zorder=4)
            ax_p.plot([i, i], [lo, h],  linewidth=0.9, color=clr, alpha=0.6, zorder=3)

        ax_p.plot(x_cur, tenkan_v, color=TENKAN_C, linewidth=1.3, label="Tenkan-sen (9)", zorder=5)
        ax_p.plot(x_cur, kijun_v,  color=KIJUN_C,  linewidth=1.5, label="Kijun-sen (26)",  zorder=5)
        chikou_x = x_cur[DISPLACEMENT:]
        chikou_val = close_v[DISPLACEMENT:]
        ax_p.plot(x_cur[:-DISPLACEMENT], chikou_val, color=CHIKOU_C, linewidth=1.1, ls="--", alpha=0.8,
                  label="Chikou Span (lagging)", zorder=5)

        # Past cloud
        for i in range(1, n):
            a0, a1 = span_a_plot[i-1], span_a_plot[i]
            b0, b1 = span_b_plot[i-1], span_b_plot[i]
            if any(np.isnan(v) for v in [a0, a1, b0, b1]):
                continue
            top0 = max(a0, b0); bot0 = min(a0, b0)
            top1 = max(a1, b1); bot1 = min(a1, b1)
            is_bull = (a0 + a1) > (b0 + b1)
            clr = KUMO_BULL if is_bull else KUMO_BEAR
            ax_p.fill_betweenx([bot0, top0], i-1, i, alpha=CLOUD_ALPHA, color=clr, zorder=1)

        # Future cloud
        for i in range(n, total_bars):
            a0, a1 = span_a_plot[i-1], span_a_plot[i] if i < total_bars else np.nan
            b0, b1 = span_b_plot[i-1], span_b_plot[i] if i < total_bars else np.nan
            if any(np.isnan(v) for v in [a0, a1, b0, b1]):
                continue
            top0 = max(a0, b0); bot0 = min(a0, b0)
            is_bull = (a0 + a1) > (b0 + b1)
            clr = KUMO_BULL if is_bull else KUMO_BEAR
            ax_p.fill_betweenx([bot0, top0], i-1, i, alpha=FUTURE_ALPHA, color=clr,
                                linestyle=":", zorder=1)

        ax_p.plot(x_all, span_a_plot, color=SPAN_A_C, linewidth=0.8, alpha=0.6, label="Senkou Span A", zorder=3)
        ax_p.plot(x_all, span_b_plot, color=SPAN_B_C, linewidth=0.8, alpha=0.6, label="Senkou Span B", zorder=3)
        ax_p.axvline(n - 1, color="#ffd54f", lw=1.2, ls=":", alpha=0.7, zorder=6)
        ax_p.text(n, ax_p.get_ylim()[0] if ax_p.get_ylim()[0] != 0 else 1,
                  " ◀ Future cloud", fontsize=7.5, color="#ffd54f", va="bottom", alpha=0.8)

        for x in tk_crosses:
            clr = BULL_CLR if x["direction"] == "BUY" else BEAR_CLR
            mk = "^" if x["direction"] == "BUY" else "v"
            sz = 100 if x["strength"] == "STRONG" else 55
            yp = x["kijun"]
            ax_p.scatter(x["bar"], yp, marker=mk, color=clr, s=sz,
                         edgecolors="white", linewidths=0.6, zorder=8)

        for x in kumo_breakouts:
            clr = BULL_CLR if x["direction"] == "BUY" else BEAR_CLR
            mk = "D" if x["strength"] == "STRONG" else "o"
            yp = x["cloud_top"] if x["direction"] == "BUY" else x["cloud_bot"]
            ax_p.scatter(x["bar"], yp, marker=mk, color=clr, s=60,
                         edgecolors="white", linewidths=0.5, zorder=8, alpha=0.9)

        for x in chikou_crosses:
            clr = BULL_CLR if x["direction"] == "BUY" else BEAR_CLR
            ax_p.scatter(x["bar"] - DISPLACEMENT, x["chikou"],
                         marker="x", color=clr, s=55, linewidths=1.2, zorder=8, alpha=0.8)

        for t in kumo_twists:
            clr = BULL_CLR if t["direction"] == "BUY" else BEAR_CLR
            ax_p.axvline(t["bar"], color=clr, lw=0.8, ls=":", alpha=0.4, zorder=2)

        # Volume
        vol = df["Volume"].values
        vol_colors = [BULL_CLR if float(df.loc[i,"Close"]) >= float(df.loc[i,"Open"])
                      else BEAR_CLR for i in range(n)]
        ax_v.bar(x_cur, vol, color=vol_colors, width=0.7, alpha=0.5)
        ax_v.set_ylabel("Volume", color="#9e9e9e", fontsize=8)
        ax_v.tick_params(colors="#9e9e9e", labelsize=7)
        ax_v.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x))))

        tick_step = max(1, n // 14)
        tick_pos  = list(range(0, n, tick_step))
        tick_lbl  = [str(df.loc[i, "Date"])[:10] for i in tick_pos]
        ax_v.set_xticks(tick_pos)
        ax_v.set_xticklabels(tick_lbl, rotation=35, ha="right", fontsize=7, color="#9e9e9e")
        ax_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
        ax_p.set_xlim(-1, total_bars + 2)

        ax_p.axhline(cur_price, color="white", lw=0.8, ls=":", alpha=0.8)
        ax_p.text(total_bars + 0.5, cur_price, f" {cur_price:.2f}",
                  va="center", fontsize=8, color="white", fontweight="bold")

        ax_p.set_title(f"{ticker} · Daily ({period}) · Ichimoku Cloud  "
                       f"({TENKAN_PERIOD},{KIJUN_PERIOD},{SENKOU_B_PERIOD},{DISPLACEMENT})",
                       color="white", fontsize=13, pad=10)
        ax_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
        for ax in [ax_p, ax_v]:
            ax.grid(axis="y", color="#1a1a1a", lw=0.5)
            ax.spines[:].set_color("#2a2a2a")

        ax_p.axvspan(recent_start, n - 0.5, color="#ffffff", alpha=0.025, zorder=0)
        ax_v.axvspan(recent_start, n - 0.5, color="#ffffff", alpha=0.025, zorder=0)
        ax_p.axvline(recent_start, color="#ffd54f", lw=0.9, ls=":", alpha=0.55)

        leg_handles = [
            mpatches.Patch(color=TENKAN_C,  label="Tenkan-sen (9)"),
            mpatches.Patch(color=KIJUN_C,   label="Kijun-sen (26)"),
            mpatches.Patch(color=CHIKOU_C,  label="Chikou Span"),
            mpatches.Patch(color=SPAN_A_C,  label="Senkou Span A"),
            mpatches.Patch(color=SPAN_B_C,  label="Senkou Span B"),
            mpatches.Patch(color=KUMO_BULL, label="Bullish Kumo",  alpha=0.5),
            mpatches.Patch(color=KUMO_BEAR, label="Bearish Kumo",  alpha=0.5),
            mpatches.Patch(color="#ffd54f", label=f"Recent {RECENT_BARS}-bar zone"),
        ]
        ax_p.legend(handles=leg_handles, loc="upper left",
                    facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=7.5,
                    ncol=2)

        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
        score_col = BULL_CLR if score > 0 else (BEAR_CLR if score < 0 else "#ffd54f")
        ax_s.text(0.01, 0.75,
                  f"Consensus: {consensus}  (score {score:+d}/6)   |   "
                  f"Price: {cur_pos.upper()} cloud   |   "
                  f"Cloud: {'GREEN (bullish)' if cur_cloud_bullish else 'RED (bearish)'}",
                  ha="left", va="center", fontsize=8.5, color=score_col, transform=ax_s.transAxes)
        lx_tk = tk_crosses[-1] if tk_crosses else None
        lx_bo = kumo_breakouts[-1] if kumo_breakouts else None
        tk_str = (f"TK cross: {lx_tk['direction']} ({lx_tk['strength']}) {str(df.loc[lx_tk['bar'], 'Date'])[:10]}"
                  if lx_tk else "TK cross: —")
        bo_str = (f"Breakout: {lx_bo['direction']} ({lx_bo['strength']}) {str(df.loc[lx_bo['bar'], 'Date'])[:10]}"
                  if lx_bo else "Breakout: —")
        ax_s.text(0.01, 0.30, f"Latest: {tk_str}   |   {bo_str}",
                  ha="left", va="center", fontsize=8.5, color="#cccccc", transform=ax_s.transAxes)
        ax_s.text(0.5, -0.15,
                  "⚠  Ichimoku is a trend-following system — works best in trending markets. Not financial advice.",
                  ha="center", va="top", fontsize=7, color="#555",
                  transform=ax_s.transAxes, style="italic")

        plt.savefig(full_chart, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        # ----- Recent chart (simplified; we reuse the same pattern but limit to recent window) -----
        r_start = recent_start
        r_end = n
        nr = r_end - r_start
        r_total = nr + DISPLACEMENT
        df_r = df.iloc[r_start:r_end].reset_index(drop=True)
        g_start = r_start
        g_end = min(r_start + r_total, total_bars)
        sa_r = span_a_plot[g_start:g_end]
        sb_r = span_b_plot[g_start:g_end]
        nr_total = len(sa_r)
        tenkan_r = tenkan_v[r_start:r_end]
        kijun_r  = kijun_v[r_start:r_end]
        close_r  = close_v[r_start:r_end]
        x_r = np.arange(nr)
        x_rt = np.arange(nr_total)

        fig2 = plt.figure(figsize=(18, 11), facecolor=BG)
        gs2 = gridspec.GridSpec(3, 1, height_ratios=[5, 1.5, 0.9], hspace=0.04)
        ax2_p = fig2.add_subplot(gs2[0])
        ax2_v = fig2.add_subplot(gs2[1], sharex=ax2_p)
        ax2_s = fig2.add_subplot(gs2[2])
        for ax in [ax2_p, ax2_v, ax2_s]:
            ax.set_facecolor(BG)

        for i in range(nr):
            o = float(df_r.loc[i, "Open"]); c_ = float(df_r.loc[i, "Close"])
            h = float(df_r.loc[i, "High"]); lo = float(df_r.loc[i, "Low"])
            clr = BULL_CLR if c_ >= o else BEAR_CLR
            ax2_p.plot([i, i], [o, c_],  linewidth=5.5, color=clr, solid_capstyle="round", zorder=4)
            ax2_p.plot([i, i], [lo, h],  linewidth=1.3, color=clr, alpha=0.7, zorder=3)

        ax2_p.plot(x_r, tenkan_r, color=TENKAN_C, linewidth=1.5, label="Tenkan-sen")
        ax2_p.plot(x_r, kijun_r,  color=KIJUN_C,  linewidth=1.7, label="Kijun-sen")
        chikou_plot_start = DISPLACEMENT
        if chikou_plot_start < nr:
            ch_x = x_r[:nr - chikou_plot_start]
            ch_val = close_r[chikou_plot_start:]
            ax2_p.plot(ch_x, ch_val, color=CHIKOU_C, linewidth=1.3, ls="--", alpha=0.85,
                       label="Chikou Span")

        for i in range(1, nr_total):
            a0, a1 = sa_r[i-1], sa_r[i]
            b0, b1 = sb_r[i-1], sb_r[i]
            if any(np.isnan(v) for v in [a0, a1, b0, b1]):
                continue
            top0 = max(a0, b0); bot0 = min(a0, b0)
            is_bull = (a0 + a1) > (b0 + b1)
            clr = KUMO_BULL if is_bull else KUMO_BEAR
            alph = CLOUD_ALPHA if i < nr else FUTURE_ALPHA
            ax2_p.fill_betweenx([bot0, top0], i-1, i, alpha=alph, color=clr, zorder=1)

        ax2_p.plot(x_rt, sa_r, color=SPAN_A_C, linewidth=0.9, alpha=0.65, label="Span A")
        ax2_p.plot(x_rt, sb_r, color=SPAN_B_C, linewidth=0.9, alpha=0.65, label="Span B")
        ax2_p.axvline(nr - 1, color="#ffd54f", lw=1.2, ls=":", alpha=0.7, zorder=6)
        ax2_p.text(nr, ax2_p.get_ylim()[0] if ax2_p.get_ylim()[0] != 0 else 1,
                   " ◀ Future", fontsize=7.5, color="#ffd54f", va="bottom", alpha=0.8)

        # Remap recent signals
        def to_local(bar_global):
            return bar_global - r_start

        for x in recent_tk:
            clr = BULL_CLR if x["direction"] == "BUY" else BEAR_CLR
            mk = "^" if x["direction"] == "BUY" else "v"
            sz = 130 if x["strength"] == "STRONG" else 80
            li = to_local(x["bar"])
            ax2_p.scatter(li, x["kijun"], marker=mk, color=clr, s=sz,
                          edgecolors="white", linewidths=0.8, zorder=8)
            lbl = f"TK {x['direction']}" + (" ★" if x["strength"] == "STRONG" else "")
            yoff = x["kijun"] * (0.972 if x["direction"] == "BUY" else 1.028)
            ax2_p.annotate(lbl, xy=(li, x["kijun"]), xytext=(li, yoff),
                           fontsize=7.5, fontweight="bold", color=clr,
                           ha="center", va="top" if x["direction"] == "BUY" else "bottom",
                           arrowprops=dict(arrowstyle="-", color=clr, lw=0.7, alpha=0.6),
                           zorder=9)

        for x in recent_breakout:
            clr = BULL_CLR if x["direction"] == "BUY" else BEAR_CLR
            mk = "D" if x["strength"] == "STRONG" else "o"
            li = to_local(x["bar"])
            yp = x["cloud_top"] if x["direction"] == "BUY" else x["cloud_bot"]
            ax2_p.scatter(li, yp, marker=mk, color=clr, s=100,
                          edgecolors="white", linewidths=0.7, zorder=8)
            lbl = f"BO {x['direction']}" + (" ★" if x["strength"] == "STRONG" else "")
            yoff = yp * (0.970 if x["direction"] == "BUY" else 1.030)
            ax2_p.annotate(lbl, xy=(li, yp), xytext=(li, yoff),
                           fontsize=7.5, fontweight="bold", color=clr,
                           ha="center", va="top" if x["direction"] == "BUY" else "bottom",
                           arrowprops=dict(arrowstyle="-", color=clr, lw=0.7, alpha=0.6),
                           zorder=9)

        for x in recent_chikou:
            clr = BULL_CLR if x["direction"] == "BUY" else BEAR_CLR
            li = to_local(x["bar"]) - DISPLACEMENT
            if 0 <= li < nr:
                ax2_p.scatter(li, x["chikou"], marker="x", color=clr, s=80,
                              linewidths=1.5, zorder=8, alpha=0.9)

        # Volume
        vol_r = df_r["Volume"].values
        vol_colors_r = [BULL_CLR if float(df_r.loc[i,"Close"]) >= float(df_r.loc[i,"Open"])
                        else BEAR_CLR for i in range(nr)]
        ax2_v.bar(x_r, vol_r, color=vol_colors_r, width=0.7, alpha=0.5)
        ax2_v.set_ylabel("Volume", color="#9e9e9e", fontsize=8)
        ax2_v.tick_params(colors="#9e9e9e", labelsize=7)
        ax2_v.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else str(int(x))))

        tick_lbl2 = [str(df_r.loc[i, "Date"])[:10] for i in range(nr)]
        ax2_v.set_xticks(list(range(nr)))
        ax2_v.set_xticklabels(tick_lbl2, rotation=45, ha="right", fontsize=6, color="#9e9e9e")
        ax2_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
        ax2_p.set_xlim(-0.5, nr_total)

        ax2_p.axhline(cur_price, color="white", lw=0.8, ls=":", alpha=0.8)
        ax2_p.text(nr_total - 0.5, cur_price, f" {cur_price:.2f}",
                   va="center", fontsize=8, color="white", fontweight="bold")

        ax2_p.set_title(f"{ticker} · Recent {nr} Candles + {DISPLACEMENT}-bar Forward Cloud · Ichimoku",
                        color="white", fontsize=12, pad=10)
        ax2_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
        for ax in [ax2_p, ax2_v]:
            ax.grid(axis="y", color="#1a1a1a", lw=0.5)
            ax.spines[:].set_color("#2a2a2a")

        ax2_p.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                     labelcolor="white", fontsize=8, ncol=2)

        ax2_s.set_xlim(0, 1); ax2_s.set_ylim(0, 1); ax2_s.axis("off")
        lines_r = []
        if recent_tk:
            lx = recent_tk[-1]
            lines_r.append((f"TK Cross: {lx['direction']} ({lx['strength']})",
                            BULL_CLR if lx["direction"] == "BUY" else BEAR_CLR))
        else:
            lines_r.append(("TK Cross: none in window", "#9e9e9e"))
        if recent_breakout:
            lx = recent_breakout[-1]
            lines_r.append((f"Kumo Breakout: {lx['direction']} ({lx['strength']})",
                            BULL_CLR if lx["direction"] == "BUY" else BEAR_CLR))
        else:
            lines_r.append(("Kumo Breakout: none in window", "#9e9e9e"))
        if recent_chikou:
            lx = recent_chikou[-1]
            lines_r.append((f"Chikou Cross: {lx['direction']}",
                            BULL_CLR if lx["direction"] == "BUY" else BEAR_CLR))
        else:
            lines_r.append(("Chikou Cross: none in window", "#9e9e9e"))

        for (txt, clr), yp in zip(lines_r, [0.78, 0.50, 0.22]):
            ax2_s.text(0.02, yp, txt, ha="left", va="center", fontsize=9, fontweight="bold",
                       color=clr, transform=ax2_s.transAxes)

        score_col2 = BULL_CLR if score > 0 else (BEAR_CLR if score < 0 else "#ffd54f")
        ax2_s.text(0.72, 0.50, f"Overall: {consensus}  ({score:+d}/6)",
                   ha="left", va="center", fontsize=9, fontweight="bold",
                   color=score_col2, transform=ax2_s.transAxes)

        ax2_s.text(0.5, -0.18,
                   "⚠  Same detection rules as the full-period scan. Not financial advice.",
                   ha="center", va="top", fontsize=7, color="#555",
                   transform=ax2_s.transAxes, style="italic")

        plt.savefig(recent_chart, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)

        chart_paths["full"] = full_chart
        chart_paths["recent"] = recent_chart

    # ── 8. RETURN ────────────────────────────────
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "current_price": cur_price,
        "signal": signal,
        "confidence": confidence,
        "reason": verdict_dict.get("verdict", "") + " | " + " ".join(verdict_dict.get("reasons", [])),
        "chart_paths": chart_paths,
        "raw": {
            "score": score,
            "consensus": consensus,
            "tenkan": cur_tenkan,
            "kijun": cur_kijun,
            "cloud_top": cur_cloud_top,
            "cloud_bottom": cur_cloud_bottom,
            "cloud_position": cur_pos,
            "chikou_vs_price": chikou_vs_price,
            "num_tk_crosses": len(tk_crosses),
            "num_chikou_crosses": len(chikou_crosses),
            "num_kijun_crosses": len(kijun_crosses),
            "num_kumo_breakouts": len(kumo_breakouts),
            "num_kumo_twists": len(kumo_twists),
        },
        "series": {
            **ohlc_payload(df),
            "tenkan": clean_list(tenkan_v),
            "kijun": clean_list(kijun_v),
            "span_a": clean_list(span_a_plot[:n]),
            "span_b": clean_list(span_b_plot[:n]),
            "chikou": clean_list(close_v),
            "tk_crosses": tk_crosses,
            "chikou_crosses": chikou_crosses,
            "kijun_crosses": kijun_crosses,
            "kumo_breakouts": kumo_breakouts,
            "kumo_twists": [t for t in kumo_twists if not t.get("is_future")],
        },
    }

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    result = analyze_ichimoku(ticker, save_chart=True)
    print(f"Signal: {result['signal']}, Confidence: {result['confidence']}, Reason: {result['reason']}")
    print("Charts:", result['chart_paths'])