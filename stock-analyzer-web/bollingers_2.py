"""
Bollinger Bands Analysis  ·  NEPSE / any ticker
─────────────────────────────────────────────────────────────
Standard/international convention used (John Bollinger's original spec —
the universal default on TradingView/Bloomberg/MetaTrader):
...
"""
import matplotlib; 
matplotlib.use('Agg')
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import os

from series_utils import clean_list, ohlc_payload

def analyze_bollinger(ticker, period='1y', interval='1d', save_chart=True):
    """
    Run Bollinger Bands analysis and return structured results.
    """
    # ──────────────────────────────────────────────
    #  CONFIG (can be overridden via parameters)
    # ──────────────────────────────────────────────
    BB_PERIOD  = 20
    BB_STD     = 2.0
    PIVOT_LEFT  = 3
    PIVOT_RIGHT = 3
    PIVOT_ORDER = max(PIVOT_LEFT, PIVOT_RIGHT)
    DOUBLE_TAP_PCT    = 0.02
    TREND_FILTER_PCT  = 0.0008
    WALK_MIN_BARS     = 3
    SQUEEZE_LOOKBACK  = 120
    SQUEEZE_PCTL      = 10
    EXPANSION_MULT    = 1.5
    RECENT_BARS = 20
    VERDICT_HORIZON = 5

    # ── 1. DOWNLOAD ──────────────────────────────
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df = df.reset_index()
    n = len(df)
    if n == 0:
        return {"error": f"No data for {ticker}"}

    # ── 2. BOLLINGER BANDS ───────────────────────
    def calc_bollinger(close, period=BB_PERIOD, num_std=BB_STD):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        bandwidth = (upper - lower) / mid
        pct_b = (close - lower) / (upper - lower)
        return mid, upper, lower, bandwidth, pct_b

    df["Mid"], df["Upper"], df["Lower"], df["Bandwidth"], df["PctB"] = calc_bollinger(df["Close"])

    close = df["Close"].values
    highs = df["High"].values
    lows  = df["Low"].values
    mid   = df["Mid"].values
    upper = df["Upper"].values
    lower = df["Lower"].values
    bw    = df["Bandwidth"].values
    pctb  = df["PctB"].values

    # ── 3. BAND BREAKOUT CROSSINGS ──────────────
    def find_band_crossings(close_arr, upper_arr, lower_arr):
        crossings = []
        for i in range(1, len(close_arr)):
            if np.isnan(upper_arr[i]) or np.isnan(lower_arr[i]) or \
               np.isnan(upper_arr[i-1]) or np.isnan(lower_arr[i-1]):
                continue
            prev_c, cur_c = close_arr[i-1], close_arr[i]
            if prev_c <= upper_arr[i-1] and cur_c > upper_arr[i]:
                crossings.append({"bar": i, "type": "UPPER_BREAK", "direction": "SELL_WATCH"})
            elif prev_c >= lower_arr[i-1] and cur_c < lower_arr[i]:
                crossings.append({"bar": i, "type": "LOWER_BREAK", "direction": "BUY_WATCH"})
        return crossings

    band_crossings = find_band_crossings(close, upper, lower)

    # ── 4. WALK-THE-BAND ────────────────────────
    def find_band_walks(close_arr, upper_arr, lower_arr, min_bars=WALK_MIN_BARS):
        n_ = len(close_arr)
        at_upper = np.zeros(n_, dtype=bool)
        at_lower = np.zeros(n_, dtype=bool)
        for i in range(n_):
            if np.isnan(upper_arr[i]) or np.isnan(lower_arr[i]):
                continue
            at_upper[i] = close_arr[i] >= upper_arr[i]
            at_lower[i] = close_arr[i] <= lower_arr[i]

        walking = np.zeros(n_, dtype=bool)
        segments = []

        def scan_side(flags, side):
            i = 0
            while i < n_:
                if flags[i]:
                    j = i
                    while j < n_ and flags[j]:
                        j += 1
                    if (j - i) >= min_bars:
                        walking[i:j] = True
                        segments.append({"start": i, "end": j - 1, "side": side})
                    i = j
                else:
                    i += 1

        scan_side(at_upper, "UPPER")
        scan_side(at_lower, "LOWER")
        return walking, segments

    is_walking, walk_segments = find_band_walks(close, upper, lower)

    # ── 5. SQUEEZE → EXPANSION ──────────────────
    def find_squeeze_events(bandwidth_arr, close_arr, upper_arr, lower_arr,
                             lookback=SQUEEZE_LOOKBACK, pctl=SQUEEZE_PCTL,
                             expansion_mult=EXPANSION_MULT):
        n_ = len(bandwidth_arr)
        is_squeeze = np.zeros(n_, dtype=bool)
        for i in range(n_):
            if np.isnan(bandwidth_arr[i]):
                continue
            start = max(0, i - lookback)
            window = bandwidth_arr[start:i + 1]
            window = window[~np.isnan(window)]
            if len(window) < 20:
                continue
            thresh = np.percentile(window, pctl)
            is_squeeze[i] = bandwidth_arr[i] <= thresh

        events = []
        i = 0
        while i < n_:
            if is_squeeze[i]:
                j = i
                while j < n_ and is_squeeze[j]:
                    j += 1
                squeeze_end = j - 1
                squeeze_bw = bandwidth_arr[squeeze_end]
                horizon = min(n_, j + max(10, lookback // 4))
                for k in range(j, horizon):
                    if np.isnan(bandwidth_arr[k]):
                        continue
                    if bandwidth_arr[k] >= squeeze_bw * expansion_mult:
                        if close_arr[k] > upper_arr[k]:
                            events.append({"squeeze_start": i, "squeeze_end": squeeze_end,
                                           "breakout_bar": k, "direction": "BUY"})
                            break
                        elif close_arr[k] < lower_arr[k]:
                            events.append({"squeeze_start": i, "squeeze_end": squeeze_end,
                                           "breakout_bar": k, "direction": "SELL"})
                            break
                i = j
            else:
                i += 1
        return is_squeeze, events

    is_squeeze, squeeze_events = find_squeeze_events(bw, close, upper, lower)

    # ── 6. %B MEAN-REVERSION EXTREMES ──────────
    def find_pctb_extremes(pctb_arr, walking_arr):
        events = []
        for i in range(len(pctb_arr)):
            if np.isnan(pctb_arr[i]):
                continue
            if pctb_arr[i] >= 1.0:
                events.append({"bar": i, "direction": "SELL",
                                "strength": "WEAK" if walking_arr[i] else "STRONG",
                                "pctb": pctb_arr[i]})
            elif pctb_arr[i] <= 0.0:
                events.append({"bar": i, "direction": "BUY",
                                "strength": "WEAK" if walking_arr[i] else "STRONG",
                                "pctb": pctb_arr[i]})
        return events

    pctb_extremes = find_pctb_extremes(pctb, is_walking)

    # ── 7. DOUBLE-TAP AT BAND ───────────────────
    def find_pivots(arr, order, mode="low"):
        comparator = np.less if mode == "low" else np.greater
        idx = argrelextrema(np.asarray(arr, dtype=float), comparator, order=order)[0]
        return idx.tolist()

    swing_low_idx  = find_pivots(lows,  PIVOT_ORDER, "low")
    swing_high_idx = find_pivots(highs, PIVOT_ORDER, "high")

    def price_same_level(p1, p2, tol=DOUBLE_TAP_PCT):
        return abs(p1 - p2) / ((abs(p1) + abs(p2)) / 2) <= tol

    def trend_slope_pct(close_arr, start_idx, end_idx):
        seg = close_arr[start_idx:end_idx + 1]
        if len(seg) < 3:
            return 0.0
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0]
        mean_price = np.mean(seg)
        return slope / mean_price if mean_price else 0.0

    def find_band_double_taps(sl_idx, sh_idx, lows_arr, highs_arr, close_arr,
                               lower_arr, upper_arr, tol=DOUBLE_TAP_PCT):
        signals = []
        for k in range(len(sl_idx) - 1):
            prev_pi, cur_pi = sl_idx[k], sl_idx[k + 1]
            prev_price, cur_price = lows_arr[prev_pi], lows_arr[cur_pi]
            if np.isnan(lower_arr[prev_pi]) or np.isnan(lower_arr[cur_pi]):
                continue
            prev_near_band = prev_price <= lower_arr[prev_pi] * 1.01
            cur_near_band  = cur_price  <= lower_arr[cur_pi]  * 1.01
            if not (prev_near_band and cur_near_band):
                continue
            if not price_same_level(prev_price, cur_price, tol):
                continue
            if cur_price < prev_price:
                continue
            slope = trend_slope_pct(close_arr, prev_pi, cur_pi)
            if slope <= -TREND_FILTER_PCT:
                continue
            signals.append({"type": "DOUBLE_TAP_LOWER", "direction": "BUY",
                             "bar": cur_pi, "prev_bar": prev_pi,
                             "price_cur": cur_price, "price_prev": prev_price})

        for k in range(len(sh_idx) - 1):
            prev_pi, cur_pi = sh_idx[k], sh_idx[k + 1]
            prev_price, cur_price = highs_arr[prev_pi], highs_arr[cur_pi]
            if np.isnan(upper_arr[prev_pi]) or np.isnan(upper_arr[cur_pi]):
                continue
            prev_near_band = prev_price >= upper_arr[prev_pi] * 0.99
            cur_near_band  = cur_price  >= upper_arr[cur_pi]  * 0.99
            if not (prev_near_band and cur_near_band):
                continue
            if not price_same_level(prev_price, cur_price, tol):
                continue
            if cur_price > prev_price:
                continue
            slope = trend_slope_pct(close_arr, prev_pi, cur_pi)
            if slope >= TREND_FILTER_PCT:
                continue
            signals.append({"type": "DOUBLE_TAP_UPPER", "direction": "SELL",
                             "bar": cur_pi, "prev_bar": prev_pi,
                             "price_cur": cur_price, "price_prev": prev_price})

        return sorted(signals, key=lambda s: s["bar"])

    band_double_taps = find_band_double_taps(swing_low_idx, swing_high_idx,
                                              lows, highs, close, lower, upper)

    # ── 8. RECENT WINDOW ────────────────────────
    recent_start = max(0, n - RECENT_BARS)
    recent_crossings   = [c for c in band_crossings if c["bar"] >= recent_start]
    recent_walks       = [w for w in walk_segments if w["end"] >= recent_start]
    recent_squeeze_evt = [e for e in squeeze_events if e["breakout_bar"] >= recent_start]
    recent_pctb        = [e for e in pctb_extremes if e["bar"] >= recent_start]
    recent_dtaps       = [s for s in band_double_taps if s["bar"] >= recent_start]
    in_squeeze_now      = bool(is_squeeze[-1]) if not np.isnan(bw[-1]) else False
    strong_extremes     = [e for e in pctb_extremes if e["strength"] == "STRONG"]

    # ── 9. BUILD VERDICT ─────────────────────────
    def build_verdict():
        if is_walking[-1]:
            side = "UPPER" if close[-1] >= upper[-1] else "LOWER"
            direction = "HOLD/RIDE TREND (do not fade)"
            reason = (f"Price is currently WALKING the {side} band — under Bollinger's own rule, "
                      f"this signals trend strength, not a reversal. Acting against it (e.g. selling "
                      f"into an upper-band walk) goes against the standard interpretation.")
            return {"direction": direction, "reason": reason, "bar": n - 1, "confidence": "—"}

        if squeeze_events and (n - 1 - squeeze_events[-1]["breakout_bar"]) <= VERDICT_HORIZON:
            e = squeeze_events[-1]
            d = str(df.loc[e["breakout_bar"], "Date"])[:10]
            return {"direction": e["direction"],
                    "reason": f"Volatility squeeze resolved into a confirmed {e['direction']} breakout on {d}.",
                    "bar": e["breakout_bar"], "confidence": "STRONG (squeeze breakout)"}

        if band_double_taps and (n - 1 - band_double_taps[-1]["bar"]) <= VERDICT_HORIZON:
            s = band_double_taps[-1]
            d = str(df.loc[s["bar"], "Date"])[:10]
            return {"direction": s["direction"],
                    "reason": f"{s['type'].replace('_',' ').title()} pattern confirmed on {d}.",
                    "bar": s["bar"], "confidence": "MODERATE (price structure)"}

        if strong_extremes and (n - 1 - strong_extremes[-1]["bar"]) <= VERDICT_HORIZON:
            e = strong_extremes[-1]
            d = str(df.loc[e["bar"], "Date"])[:10]
            return {"direction": e["direction"],
                    "reason": f"Isolated %B extreme ({e['pctb']:.2f}) on {d} — not part of a band walk, "
                              f"the classic mean-reversion read.",
                    "bar": e["bar"], "confidence": "MODERATE (mean-reversion)"}

        if in_squeeze_now:
            return {"direction": "WATCH (no entry yet)",
                    "reason": "Bandwidth is at a multi-month low — a breakout is statistically likely "
                              "soon, but direction is not yet confirmed. Wait for a close outside a band "
                              "with expanding bandwidth before acting.",
                    "bar": n - 1, "confidence": "—"}

        return {"direction": "NO SIGNAL",
                "reason": "Price is inside the bands, not walking a band, not in a squeeze, and no "
                          "recent double-tap or isolated extreme qualifies. Reported honestly — no "
                          "signal is manufactured when none exists.",
                "bar": n - 1, "confidence": "—"}

    verdict = build_verdict()

    # ── 10. CHARTS ──────────────────────────────
    chart_paths = {"full": None, "recent": None}
    if save_chart:
        os.makedirs("static", exist_ok=True)
        base = f"static/{ticker}_Bollinger"
        full_chart = f"{base}_full.png"
        recent_chart = f"{base}_recent.png"

        # ----- Full chart (same as original, but with paths) -----
        fig = plt.figure(figsize=(20, 12), facecolor="#0d1117")
        gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1.6, 0.8], hspace=0.06)
        ax_p = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1], sharex=ax_p)
        ax_s = fig.add_subplot(gs[2])
        for ax in [ax_p, ax_b, ax_s]:
            ax.set_facecolor("#0d1117")

        x = np.arange(n)
        ax_p.plot(x, upper, color="#64b5f6", lw=1.1, alpha=0.85, zorder=2, label="Upper Band")
        ax_p.plot(x, mid,   color="#bdbdbd", lw=1.0, alpha=0.8,  zorder=2, label="Middle (SMA20)")
        ax_p.plot(x, lower, color="#64b5f6", lw=1.1, alpha=0.85, zorder=2, label="Lower Band")
        ax_p.fill_between(x, upper, lower, color="#64b5f6", alpha=0.05, zorder=1)

        for i in range(n):
            o = float(df.loc[i, "Open"]);  c = float(df.loc[i, "Close"])
            h = float(df.loc[i, "High"]); lo = float(df.loc[i, "Low"])
            col = "#26a69a" if c >= o else "#ef5350"
            ax_p.plot([i, i], [o, c],  linewidth=4, color=col, solid_capstyle="round", zorder=3)
            ax_p.plot([i, i], [lo, h], linewidth=1, color=col, alpha=0.6, zorder=2)

        for idx in swing_low_idx:
            ax_p.scatter(idx, lows[idx],  marker="^", color="#26a69a", s=18, alpha=0.35, zorder=4)
        for idx in swing_high_idx:
            ax_p.scatter(idx, highs[idx], marker="v", color="#ef5350", s=18, alpha=0.35, zorder=4)

        for w in walk_segments:
            col = "#ef5350" if w["side"] == "UPPER" else "#26a69a"
            ax_p.axvspan(w["start"] - 0.5, w["end"] + 0.5, color=col, alpha=0.08, zorder=0)

        sq_idx = np.where(is_squeeze)[0]
        for idx in sq_idx:
            ax_p.axvspan(idx - 0.5, idx + 0.5, color="#ffd54f", alpha=0.06, zorder=0)

        for e in squeeze_events:
            col = "#26a69a" if e["direction"] == "BUY" else "#ef5350"
            b = e["breakout_bar"]
            ax_p.scatter(b, close[b], marker="*", color=col, s=180, edgecolors="white",
                         linewidths=0.6, zorder=8)
            ax_p.annotate(f"Squeeze {e['direction']}", xy=(b, close[b]),
                          xytext=(b, close[b] * (1.03 if e["direction"] == "BUY" else 0.97)),
                          fontsize=7.5, fontweight="bold", color=col, ha="center", zorder=8,
                          arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.7))

        for s in band_double_taps:
            col = "#66bb6a" if s["direction"] == "BUY" else "#ff8a65"
            pi, ci = s["prev_bar"], s["bar"]
            pp, cp = s["price_prev"], s["price_cur"]
            ax_p.plot([pi, ci], [pp, cp], color=col, lw=2, ls="--", alpha=0.9, zorder=5)
            ax_p.scatter([pi, ci], [pp, cp], color=col, s=70, zorder=6)

        for e in strong_extremes:
            col = "#26a69a" if e["direction"] == "BUY" else "#ef5350"
            marker = "^" if e["direction"] == "BUY" else "v"
            b = e["bar"]
            ax_p.scatter(b, close[b], marker=marker, color=col, s=55, alpha=0.8,
                         edgecolors="white", linewidths=0.4, zorder=7)

        tick_step = max(1, n // 12)
        tick_pos  = list(range(0, n, tick_step))
        tick_lbl  = [str(df.loc[i, "Date"])[:10] for i in tick_pos]

        cur = float(df["Close"].iloc[-1])
        ax_p.axhline(cur, color="white", lw=0.8, ls=":", alpha=0.8)
        ax_p.text(n + 0.5, cur, f" {cur:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
        ax_p.set_title(f"{ticker} · Daily ({period}) · Bollinger Bands ({BB_PERIOD}, {BB_STD}σ) Analysis",
                       color="white", fontsize=13, pad=10)
        ax_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
        ax_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
        ax_p.set_xlim(-1, n + 8)
        ax_p.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                    labelcolor="white", fontsize=8)

        ax_b.plot(x, pctb, color="#ba68c8", lw=1.3, label="%B", zorder=3)
        ax_b.axhline(1.0, color="#ef5350", lw=0.8, ls="--", alpha=0.5)
        ax_b.axhline(0.0, color="#26a69a", lw=0.8, ls="--", alpha=0.5)
        ax_b.axhline(0.5, color="#9e9e9e", lw=0.6, ls=":", alpha=0.4)
        ax_b.fill_between(x, pctb, 1.0, where=(pctb >= 1.0), color="#ef5350", alpha=0.15)
        ax_b.fill_between(x, pctb, 0.0, where=(pctb <= 0.0), color="#26a69a", alpha=0.15)
        ax_b.set_ylabel("%B", color="#9e9e9e", fontsize=9)
        ax_b.set_ylim(min(-0.3, np.nanmin(pctb) - 0.1) if not np.all(np.isnan(pctb)) else -0.3,
                      max(1.3, np.nanmax(pctb) + 0.1) if not np.all(np.isnan(pctb)) else 1.3)

        ax_b2 = ax_b.twinx()
        ax_b2.plot(x, bw, color="#ffa726", lw=1.0, alpha=0.7, label="Bandwidth", zorder=2)
        ax_b2.set_ylabel("Bandwidth", color="#ffa726", fontsize=8)
        ax_b2.tick_params(colors="#ffa726", labelsize=7)
        for idx in sq_idx:
            ax_b2.axvspan(idx - 0.5, idx + 0.5, color="#ffd54f", alpha=0.06, zorder=0)

        ax_b.set_xticks(tick_pos)
        ax_b.set_xticklabels(tick_lbl, rotation=35, ha="right", fontsize=7.5, color="#9e9e9e")
        ax_b.tick_params(colors="#9e9e9e", labelsize=8)
        ax_b.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                    labelcolor="white", fontsize=7.5)

        for ax in [ax_p, ax_b]:
            ax.grid(axis="y", color="#1a1a1a", lw=0.5)
            ax.spines[:].set_color("#2a2a2a")

        ax_p.axvspan(recent_start, n - 0.5, color="#ffffff", alpha=0.03, zorder=0)
        ax_b.axvspan(recent_start, n - 0.5, color="#ffffff", alpha=0.03, zorder=0)
        ax_p.axvline(recent_start, color="#fff176", lw=1, ls=":", alpha=0.6)
        ax_b.axvline(recent_start, color="#fff176", lw=1, ls=":", alpha=0.6)
        ax_p.text(recent_start + 0.5, ax_p.get_ylim()[0], f"◀ {RECENT_BARS}-bar window",
                  fontsize=7, color="#fff176", va="bottom", alpha=0.8)

        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
        sq_txt = "INSIDE SQUEEZE — watch for breakout" if in_squeeze_now else "Not currently squeezed"
        walk_txt = "—"
        if walk_segments and walk_segments[-1]["end"] >= n - 5:
            walk_txt = f"{walk_segments[-1]['side']} band walk active/recent"
        verdict_chart_col = ("#26a69a" if verdict["direction"] == "BUY"
                              else "#ef5350" if verdict["direction"] == "SELL"
                              else "#ffd54f" if "HOLD" in verdict["direction"] or "WATCH" in verdict["direction"]
                              else "#9e9e9e")
        ax_s.text(0.01, 0.82, f"VERDICT: {verdict['direction']}  ({verdict['confidence']})",
                  ha="left", va="center", fontsize=10, fontweight="bold",
                  color=verdict_chart_col, transform=ax_s.transAxes)
        ax_s.text(0.01, 0.52, f"%B: {pctb[-1]:.3f}   Bandwidth: {bw[-1]:.4f}   {sq_txt}",
                  ha="left", va="center", fontsize=8.5, color="#ffd54f", transform=ax_s.transAxes)
        ax_s.text(0.01, 0.22, f"Band walk status: {walk_txt}   |   Double-taps found: {len(band_double_taps)}",
                  ha="left", va="center", fontsize=8.5, color="#cccccc", transform=ax_s.transAxes)
        ax_s.text(0.5, -0.15, "⚠  For educational purposes only — not financial advice.",
                  ha="center", va="top", fontsize=7, color="#555",
                  transform=ax_s.transAxes, style="italic")

        plt.savefig(full_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)

        # ----- Recent chart -----
        df_r   = df.iloc[recent_start:].reset_index(drop=True)
        nr     = len(df_r)
        lows_r   = df_r["Low"].values
        highs_r  = df_r["High"].values
        mid_r    = df_r["Mid"].values
        upper_r  = df_r["Upper"].values
        lower_r  = df_r["Lower"].values
        pctb_r   = df_r["PctB"].values
        bw_r     = df_r["Bandwidth"].values
        close_r  = df_r["Close"].values

        sl_r = [i - recent_start for i in swing_low_idx  if i >= recent_start]
        sh_r = [i - recent_start for i in swing_high_idx if i >= recent_start]

        fig2 = plt.figure(figsize=(16, 10), facecolor="#0d1117")
        gs2  = gridspec.GridSpec(3, 1, height_ratios=[4, 1.6, 0.9], hspace=0.06)
        ax2_p = fig2.add_subplot(gs2[0])
        ax2_b = fig2.add_subplot(gs2[1], sharex=ax2_p)
        ax2_s = fig2.add_subplot(gs2[2])
        for ax in [ax2_p, ax2_b, ax2_s]:
            ax.set_facecolor("#0d1117")

        xr = np.arange(nr)
        ax2_p.plot(xr, upper_r, color="#64b5f6", lw=1.4, alpha=0.9, zorder=2, label="Upper Band")
        ax2_p.plot(xr, mid_r,   color="#bdbdbd", lw=1.2, alpha=0.85, zorder=2, label="Middle (SMA20)")
        ax2_p.plot(xr, lower_r, color="#64b5f6", lw=1.4, alpha=0.9, zorder=2, label="Lower Band")
        ax2_p.fill_between(xr, upper_r, lower_r, color="#64b5f6", alpha=0.06, zorder=1)

        for i in range(nr):
            o = float(df_r.loc[i, "Open"]); c = float(df_r.loc[i, "Close"])
            h = float(df_r.loc[i, "High"]); lo = float(df_r.loc[i, "Low"])
            col = "#26a69a" if c >= o else "#ef5350"
            ax2_p.plot([i, i], [o, c],  linewidth=6, color=col, solid_capstyle="round", zorder=3)
            ax2_p.plot([i, i], [lo, h], linewidth=1.5, color=col, alpha=0.7, zorder=2)

        for idx in sl_r:
            ax2_p.scatter(idx, lows_r[idx],  marker="^", color="#26a69a", s=45, alpha=0.5, zorder=4)
        for idx in sh_r:
            ax2_p.scatter(idx, highs_r[idx], marker="v", color="#ef5350", s=45, alpha=0.5, zorder=4)

        for w in walk_segments:
            s_local = w["start"] - recent_start
            e_local = w["end"] - recent_start
            if e_local < 0 or s_local > nr:
                continue
            s_local = max(0, s_local); e_local = min(nr - 1, e_local)
            col = "#ef5350" if w["side"] == "UPPER" else "#26a69a"
            ax2_p.axvspan(s_local - 0.5, e_local + 0.5, color=col, alpha=0.1, zorder=0)

        for e in squeeze_events:
            b = e["breakout_bar"]
            if b < recent_start:
                continue
            bl = b - recent_start
            col = "#26a69a" if e["direction"] == "BUY" else "#ef5350"
            ax2_p.scatter(bl, close_r[bl], marker="*", color=col, s=260, edgecolors="white",
                          linewidths=0.8, zorder=8)
            ax2_p.annotate(f"Squeeze {e['direction']}", xy=(bl, close_r[bl]),
                            xytext=(bl, close_r[bl] * (1.03 if e["direction"] == "BUY" else 0.97)),
                            fontsize=8.5, fontweight="bold", color=col, ha="center", zorder=8,
                            arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.8))

        for s in band_double_taps:
            if s["bar"] < recent_start:
                continue
            col = "#66bb6a" if s["direction"] == "BUY" else "#ff8a65"
            pi, ci = s["prev_bar"] - recent_start, s["bar"] - recent_start
            if pi < 0:
                continue
            pp, cp = s["price_prev"], s["price_cur"]
            ax2_p.plot([pi, ci], [pp, cp], color=col, lw=2.5, ls="--", alpha=0.9, zorder=5)
            ax2_p.scatter([pi, ci], [pp, cp], color=col, s=100, zorder=6)

        for e in strong_extremes:
            if e["bar"] < recent_start:
                continue
            bl = e["bar"] - recent_start
            col = "#26a69a" if e["direction"] == "BUY" else "#ef5350"
            marker = "^" if e["direction"] == "BUY" else "v"
            ax2_p.scatter(bl, close_r[bl], marker=marker, color=col, s=90, alpha=0.9,
                          edgecolors="white", linewidths=0.6, zorder=7)

        tick_lbl2 = [str(df_r.loc[i, "Date"])[:10] for i in range(nr)]

        cur2 = float(df_r["Close"].iloc[-1])
        ax2_p.axhline(cur2, color="white", lw=0.8, ls=":", alpha=0.8)
        ax2_p.text(nr - 0.5, cur2, f" {cur2:.2f}", va="center", fontsize=8,
                   color="white", fontweight="bold")
        ax2_p.set_title(f"{ticker} · Recent {nr} Candles · Bollinger Bands (zoomed)",
                        color="white", fontsize=12, pad=10)
        ax2_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
        ax2_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
        ax2_p.set_xlim(-0.5, nr)
        ax2_p.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                     labelcolor="white", fontsize=8)

        ax2_b.plot(xr, pctb_r, color="#ba68c8", lw=1.5, label="%B", zorder=3)
        ax2_b.axhline(1.0, color="#ef5350", lw=0.8, ls="--", alpha=0.5)
        ax2_b.axhline(0.0, color="#26a69a", lw=0.8, ls="--", alpha=0.5)
        ax2_b.fill_between(xr, pctb_r, 1.0, where=(pctb_r >= 1.0), color="#ef5350", alpha=0.15)
        ax2_b.fill_between(xr, pctb_r, 0.0, where=(pctb_r <= 0.0), color="#26a69a", alpha=0.15)
        ax2_b.set_ylabel("%B", color="#9e9e9e", fontsize=9)

        ax2_b2 = ax2_b.twinx()
        ax2_b2.plot(xr, bw_r, color="#ffa726", lw=1.2, alpha=0.8, label="Bandwidth", zorder=2)
        ax2_b2.set_ylabel("Bandwidth", color="#ffa726", fontsize=8)
        ax2_b2.tick_params(colors="#ffa726", labelsize=7)

        ax2_b.set_xticks(list(range(nr)))
        ax2_b.set_xticklabels(tick_lbl2, rotation=45, ha="right", fontsize=6.5, color="#9e9e9e")
        ax2_b.tick_params(colors="#9e9e9e", labelsize=8)
        ax2_b.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                     labelcolor="white", fontsize=7.5)

        for ax in [ax2_p, ax2_b]:
            ax.grid(axis="y", color="#1a1a1a", lw=0.5)
            ax.spines[:].set_color("#2a2a2a")

        ax2_s.set_xlim(0, 1); ax2_s.set_ylim(0, 1); ax2_s.axis("off")
        lines = []
        lines.append((f"VERDICT: {verdict['direction']}  ({verdict['confidence']})",
                      "#26a69a" if verdict["direction"] == "BUY"
                      else "#ef5350" if verdict["direction"] == "SELL"
                      else "#ffd54f" if ("HOLD" in verdict["direction"] or "WATCH" in verdict["direction"])
                      else "#9e9e9e"))
        lines.append((f"%B: {pctb_r[-1]:.3f}   Bandwidth: {bw_r[-1]:.4f}",
                      "#ef5350" if pctb_r[-1] >= 1.0 else ("#26a69a" if pctb_r[-1] <= 0.0 else "#cccccc")))
        lines.append((f"Squeeze status: {'INSIDE SQUEEZE' if in_squeeze_now else 'Normal volatility'}",
                      "#ffd54f" if in_squeeze_now else "#9e9e9e"))
        if recent_dtaps:
            s = recent_dtaps[-1]
            lines.append((f"Double-tap: {s['type']} ({s['direction']})",
                          "#26a69a" if s["direction"] == "BUY" else "#ef5350"))
        else:
            lines.append(("Double-tap: none in window", "#9e9e9e"))

        y_positions = [0.85, 0.60, 0.35, 0.10]
        for (txt, col), yp in zip(lines, y_positions):
            ax2_s.text(0.02, yp, txt, ha="left", va="center", fontsize=9, fontweight="bold",
                       color=col, transform=ax2_s.transAxes)

        ax2_s.text(0.5, -0.18, "⚠  Same detection rules as the full-period scan. Not financial advice.",
                   ha="center", va="top", fontsize=7, color="#555",
                   transform=ax2_s.transAxes, style="italic")

        plt.savefig(recent_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig2)

        chart_paths["full"] = full_chart
        chart_paths["recent"] = recent_chart

    # ── 11. RETURN STRUCTURED RESULT ─────────────
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "current_price": float(df["Close"].iloc[-1]),
        "signal": verdict["direction"],
        "confidence": (100 if verdict["confidence"] == "STRONG" else
                       70 if "MODERATE" in verdict["confidence"] else
                       50 if verdict["confidence"] == "—" else 0),
        "reason": verdict["reason"],
        "chart_paths": chart_paths,
        "raw": {
            "mid": float(mid[-1]),
            "upper": float(upper[-1]),
            "lower": float(lower[-1]),
            "pct_b": float(pctb[-1]),
            "bandwidth": float(bw[-1]),
            "in_squeeze": in_squeeze_now,
            "walking": bool(is_walking[-1]),
            "num_crossings": len(band_crossings),
            "num_walk_segments": len(walk_segments),
            "num_squeeze_events": len(squeeze_events),
            "num_double_taps": len(band_double_taps),
        },
        "series": {
            **ohlc_payload(df),
            "mid": clean_list(mid),
            "upper": clean_list(upper),
            "lower": clean_list(lower),
            "pct_b": clean_list(pctb),
            "bandwidth": clean_list(bw),
            "band_crossings": band_crossings,
            "squeeze_events": squeeze_events,
            "double_taps": band_double_taps,
        },
    }

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    result = analyze_bollinger(ticker, save_chart=True)
    print(f"Signal: {result['signal']}, Confidence: {result['confidence']}, Reason: {result['reason']}")
    print("Charts:", result['chart_paths'])