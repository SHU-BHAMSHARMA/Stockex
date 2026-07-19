"""
MACD Divergence + Crossover Detector  ·  NEPSE / any ticker
─────────────────────────────────────────────────────────────
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

def analyze_macd(ticker, period='1y', interval='1d', save_chart=True):
    """
    Run MACD analysis and return structured results.
    """
    # ── CONFIG ──────────────────────────────────
    FAST_PERIOD   = 12
    SLOW_PERIOD   = 26
    SIGNAL_PERIOD = 9
    PIVOT_LEFT  = 3
    PIVOT_RIGHT = 3
    PIVOT_ORDER = max(PIVOT_LEFT, PIVOT_RIGHT)
    DOUBLE_TAP_PCT   = 0.02
    TREND_FILTER_PCT = 0.0008
    MACD_MIN_DIFF_STD = 0.35
    MACD_STD_WINDOW   = 100
    EXTREME_PCTL = 80
    RECENT_BARS = 20

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

    # ── 2. MACD ─────────────────────────────────
    def ema_sma_seeded(series, period):
        vals = series.values.astype(float)
        out = np.full(len(vals), np.nan)
        if len(vals) < period:
            return pd.Series(out, index=series.index)
        seed = vals[:period].mean()
        out[period - 1] = seed
        k = 2.0 / (period + 1.0)
        for i in range(period, len(vals)):
            out[i] = vals[i] * k + out[i - 1] * (1 - k)
        return pd.Series(out, index=series.index)

    def calc_macd(close, fast=FAST_PERIOD, slow=SLOW_PERIOD, signal=SIGNAL_PERIOD):
        ema_fast = ema_sma_seeded(close, fast)
        ema_slow = ema_sma_seeded(close, slow)
        macd_line = ema_fast - ema_slow
        valid_start = slow - 1
        macd_valid = macd_line.iloc[valid_start:].reset_index(drop=True)
        sig_partial = ema_sma_seeded(macd_valid, signal)
        signal_line = pd.Series(np.full(len(close), np.nan), index=close.index)
        signal_line.iloc[valid_start: valid_start + len(sig_partial)] = sig_partial.values
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    df["MACD"], df["Signal"], df["Hist"] = calc_macd(df["Close"])
    macd  = df["MACD"].values
    sig   = df["Signal"].values
    hist  = df["Hist"].values
    close = df["Close"].values
    lows  = df["Low"].values
    highs = df["High"].values

    macd_series = df["MACD"]
    roll_std = macd_series.rolling(MACD_STD_WINDOW, min_periods=20).std()
    roll_std_vals = roll_std.bfill().values
    roll_std_vals = np.where(
        np.isnan(roll_std_vals) | (roll_std_vals == 0), np.nanstd(macd) or 1e-6, roll_std_vals
    )

    # ── 3. PIVOTS ───────────────────────────────
    def find_pivots(arr, order, mode="low"):
        comparator = np.less if mode == "low" else np.greater
        idx = argrelextrema(np.asarray(arr, dtype=float), comparator, order=order)[0]
        return idx.tolist()

    swing_low_idx  = find_pivots(lows,  PIVOT_ORDER, "low")
    swing_high_idx = find_pivots(highs, PIVOT_ORDER, "high")
    macd_low_idx   = find_pivots(macd,  PIVOT_ORDER, "low")
    macd_high_idx  = find_pivots(macd,  PIVOT_ORDER, "high")

    # ── 4. HELPERS ──────────────────────────────
    MATCH_WINDOW = PIVOT_LEFT + PIVOT_RIGHT

    def nearest_pivot(pivot_list, bar_idx, match_window):
        candidates = [p for p in pivot_list if abs(p - bar_idx) <= match_window]
        if not candidates: return None
        return min(candidates, key=lambda p: abs(p - bar_idx))

    def price_same_level(p1, p2, tol=DOUBLE_TAP_PCT):
        return abs(p1 - p2) / ((abs(p1) + abs(p2)) / 2) <= tol

    def macd_higher(m_new, m_old, idx, min_diff_std=MACD_MIN_DIFF_STD):
        return (m_new - m_old) >= min_diff_std * roll_std_vals[idx]

    def macd_lower(m_new, m_old, idx, min_diff_std=MACD_MIN_DIFF_STD):
        return (m_old - m_new) >= min_diff_std * roll_std_vals[idx]

    def trend_slope_pct(close_arr, start_idx, end_idx):
        seg = close_arr[start_idx:end_idx + 1]
        if len(seg) < 3: return 0.0
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0]
        mean_price = np.mean(seg)
        return slope / mean_price if mean_price else 0.0

    def zone_strength(macd_value, idx, direction):
        window = macd_series.iloc[max(0, idx - MACD_STD_WINDOW): idx + 1].dropna()
        if len(window) < 20: return "WEAK"
        if direction == "BUY":
            thresh = np.percentile(window, 100 - EXTREME_PCTL)
            return "STRONG" if macd_value <= thresh else "WEAK"
        else:
            thresh = np.percentile(window, EXTREME_PCTL)
            return "STRONG" if macd_value >= thresh else "WEAK"

    # ── 5. DIVERGENCE SCAN ──────────────────────
    TYPE_PRIORITY = {"CLASSIC_BULL":0, "CLASSIC_BEAR":0,
                     "DOUBLE_TAP_BULL":1, "DOUBLE_TAP_BEAR":1}

    def run_divergence_scan(sl_idx, sh_idx, ml_idx, mh_idx,
                             lows_arr, highs_arr, close_arr, macd_arr,
                             match_window):
        signals = []
        # Bullish
        for k in range(len(sl_idx)-1):
            prev_pi = sl_idx[k]; cur_pi = sl_idx[k+1]
            prev_price = lows_arr[prev_pi]; cur_price = lows_arr[cur_pi]
            prev_mi = nearest_pivot(ml_idx, prev_pi, match_window)
            cur_mi  = nearest_pivot(ml_idx, cur_pi,  match_window)
            if prev_mi is None or cur_mi is None or prev_mi >= cur_mi:
                continue
            if np.isnan(macd_arr[prev_mi]) or np.isnan(macd_arr[cur_mi]):
                continue
            prev_mv = macd_arr[prev_mi]; cur_mv = macd_arr[cur_mi]
            if cur_price < prev_price and macd_higher(cur_mv, prev_mv, cur_mi):
                signals.append({"type":"CLASSIC_BULL","label":"Classic Bullish","direction":"BUY",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "macd_cur":cur_mv,"macd_prev":prev_mv,
                                "macd_bar":cur_mi,"macd_prev_bar":prev_mi,
                                "strength":zone_strength(cur_mv, cur_mi, "BUY")})
            elif price_same_level(cur_price, prev_price) and macd_higher(cur_mv, prev_mv, cur_mi):
                slope = trend_slope_pct(close_arr, prev_pi, cur_pi)
                if slope <= -TREND_FILTER_PCT: continue
                signals.append({"type":"DOUBLE_TAP_BULL","label":"Double Tap Bullish","direction":"BUY",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "macd_cur":cur_mv,"macd_prev":prev_mv,
                                "macd_bar":cur_mi,"macd_prev_bar":prev_mi,
                                "strength":zone_strength(cur_mv, cur_mi, "BUY")})
        # Bearish
        for k in range(len(sh_idx)-1):
            prev_pi = sh_idx[k]; cur_pi = sh_idx[k+1]
            prev_price = highs_arr[prev_pi]; cur_price = highs_arr[cur_pi]
            prev_mi = nearest_pivot(mh_idx, prev_pi, match_window)
            cur_mi  = nearest_pivot(mh_idx, cur_pi,  match_window)
            if prev_mi is None or cur_mi is None or prev_mi >= cur_mi:
                continue
            if np.isnan(macd_arr[prev_mi]) or np.isnan(macd_arr[cur_mi]):
                continue
            prev_mv = macd_arr[prev_mi]; cur_mv = macd_arr[cur_mi]
            if cur_price > prev_price and macd_lower(cur_mv, prev_mv, cur_mi):
                signals.append({"type":"CLASSIC_BEAR","label":"Classic Bearish","direction":"SELL",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "macd_cur":cur_mv,"macd_prev":prev_mv,
                                "macd_bar":cur_mi,"macd_prev_bar":prev_mi,
                                "strength":zone_strength(cur_mv, cur_mi, "SELL")})
            elif price_same_level(cur_price, prev_price) and macd_lower(cur_mv, prev_mv, cur_mi):
                slope = trend_slope_pct(close_arr, prev_pi, cur_pi)
                if slope >= TREND_FILTER_PCT: continue
                signals.append({"type":"DOUBLE_TAP_BEAR","label":"Double Tap Bearish","direction":"SELL",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "macd_cur":cur_mv,"macd_prev":prev_mv,
                                "macd_bar":cur_mi,"macd_prev_bar":prev_mi,
                                "strength":zone_strength(cur_mv, cur_mi, "SELL")})
        seen = {}
        for s in signals:
            b = s["bar"]
            if b not in seen or TYPE_PRIORITY[s["type"]] < TYPE_PRIORITY[seen[b]["type"]]:
                seen[b] = s
        return sorted(seen.values(), key=lambda s: s["bar"])

    unique_signals = run_divergence_scan(
        swing_low_idx, swing_high_idx, macd_low_idx, macd_high_idx,
        lows, highs, close, macd, MATCH_WINDOW
    )

    # ── 6. SIGNAL-LINE CROSSOVERS ──────────────
    def find_signal_crossovers(macd_arr, sig_arr):
        crossovers = []
        for i in range(1, len(macd_arr)):
            m0, m1 = macd_arr[i-1], macd_arr[i]
            s0, s1 = sig_arr[i-1], sig_arr[i]
            if np.isnan(m0) or np.isnan(m1) or np.isnan(s0) or np.isnan(s1): continue
            prev_diff = m0 - s0; cur_diff = m1 - s1
            if prev_diff <= 0 and cur_diff > 0:
                direction = "BUY"; strength = "STRONG" if m1 < 0 else "WEAK"
            elif prev_diff >= 0 and cur_diff < 0:
                direction = "SELL"; strength = "STRONG" if m1 > 0 else "WEAK"
            else: continue
            crossovers.append({"bar": i, "direction": direction, "strength": strength,
                               "macd": m1, "signal": s1})
        return crossovers

    signal_crossovers = find_signal_crossovers(macd, sig)
    latest_cross = signal_crossovers[-1] if signal_crossovers else None

    # ── 7. ZERO-LINE CROSSOVERS ─────────────────
    def find_zero_crossovers(macd_arr):
        crossovers = []
        for i in range(1, len(macd_arr)):
            m0, m1 = macd_arr[i-1], macd_arr[i]
            if np.isnan(m0) or np.isnan(m1): continue
            if m0 <= 0 and m1 > 0: crossovers.append({"bar": i, "direction": "BUY", "macd": m1})
            elif m0 >= 0 and m1 < 0: crossovers.append({"bar": i, "direction": "SELL", "macd": m1})
        return crossovers

    zero_crossovers = find_zero_crossovers(macd)

    # ── 8. RECENT WINDOW ────────────────────────
    recent_start = max(0, n - RECENT_BARS)
    recent_signals = [s for s in unique_signals if s["bar"] >= recent_start]
    recent_crossovers = [c for c in signal_crossovers if c["bar"] >= recent_start]
    recent_zero_crossovers = [c for c in zero_crossovers if c["bar"] >= recent_start]

    # ── 9. COMPUTE OVERALL VERDICT ──────────────
    # Weighted divergence signals
    buy_score = 0; sell_score = 0
    for s in unique_signals:
        w = 2 if s["strength"] == "STRONG" else 1
        if s["direction"] == "BUY": buy_score += w
        else: sell_score += w

    # Signal crossovers add to score
    for c in signal_crossovers:
        w = 1.5 if c["strength"] == "STRONG" else 0.8
        if c["direction"] == "BUY": buy_score += w
        else: sell_score += w

    # Zero-line crossovers (trend shift) add smaller weight
    for z in zero_crossovers:
        if z["bar"] >= recent_start:  # only recent zero crosses count
            w = 0.5
            if z["direction"] == "BUY": buy_score += w
            else: sell_score += w

    total = buy_score + sell_score
    if total == 0:
        signal = "NEUTRAL"; confidence = 0; reason = "No significant signals detected."
    else:
        net = (buy_score - sell_score) / total
        if net > 0.15:
            signal = "BUY"; confidence = min(100, int(50 + 50 * net))
        elif net < -0.15:
            signal = "SELL"; confidence = min(100, int(50 + 50 * abs(net)))
        else:
            signal = "NEUTRAL"; confidence = 40
        reason = f"Weighted signals: BUY={buy_score:.1f}, SELL={sell_score:.1f}"

    # ── 10. CHARTS ──────────────────────────────
    chart_paths = {"full": None, "recent": None}
    if save_chart:
        os.makedirs("static", exist_ok=True)
        base = f"static/{ticker}_MACD"
        full_chart = f"{base}_full.png"
        recent_chart = f"{base}_recent.png"

        # ----- Full chart (adapted) -----
        SIG_COLOR = {"CLASSIC_BULL":"#26a69a","DOUBLE_TAP_BULL":"#66bb6a",
                     "CLASSIC_BEAR":"#ef5350","DOUBLE_TAP_BEAR":"#ff8a65"}

        fig = plt.figure(figsize=(20, 12), facecolor="#0d1117")
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 2, 0.8], hspace=0.05)
        ax_p = fig.add_subplot(gs[0]); ax_m = fig.add_subplot(gs[1]); ax_s = fig.add_subplot(gs[2])
        for ax in [ax_p, ax_m, ax_s]:
            ax.set_facecolor("#0d1117")

        for i in range(n):
            o = float(df.loc[i,"Open"]); c = float(df.loc[i,"Close"])
            h = float(df.loc[i,"High"]); lo = float(df.loc[i,"Low"])
            col = "#26a69a" if c >= o else "#ef5350"
            ax_p.plot([i,i], [o,c], linewidth=4, color=col, solid_capstyle="round", zorder=3)
            ax_p.plot([i,i], [lo,h], linewidth=1, color=col, alpha=0.6, zorder=2)

        for idx in swing_low_idx:
            ax_p.scatter(idx, lows[idx], marker="^", color="#26a69a", s=18, alpha=0.35, zorder=4)
        for idx in swing_high_idx:
            ax_p.scatter(idx, highs[idx], marker="v", color="#ef5350", s=18, alpha=0.35, zorder=4)

        x = np.arange(n)
        hist_colors = np.where(hist >= 0,
                                np.where(np.append(np.diff(hist), 0) >= 0, "#26a69a", "#80cbc4"),
                                np.where(np.append(np.diff(hist), 0) <= 0, "#ef5350", "#ef9a9a"))
        ax_m.bar(x, hist, color=hist_colors, width=0.7, alpha=0.6, zorder=1)
        ax_m.plot(x, macd, color="#42a5f5", linewidth=1.4, label="MACD", zorder=3)
        ax_m.plot(x, sig,  color="#ffa726", linewidth=1.2, label="Signal", zorder=3)
        ax_m.axhline(0, color="#9e9e9e", lw=0.9, ls="-", alpha=0.6)
        ax_m.set_ylabel("MACD", color="#9e9e9e", fontsize=9)
        ax_m.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                    labelcolor="white", fontsize=8)

        for idx in macd_low_idx:
            if not np.isnan(macd[idx]):
                ax_m.scatter(idx, macd[idx], marker="^", color="#26a69a", s=18, alpha=0.35, zorder=4)
        for idx in macd_high_idx:
            if not np.isnan(macd[idx]):
                ax_m.scatter(idx, macd[idx], marker="v", color="#ef5350", s=18, alpha=0.35, zorder=4)

        for s in unique_signals:
            c = SIG_COLOR[s["type"]]
            pi, ci = s["prev_bar"], s["bar"]
            mpi, mci = s["macd_prev_bar"], s["macd_bar"]
            pp, cp = s["price_prev"], s["price_cur"]
            mp, mc = s["macd_prev"], s["macd_cur"]
            ax_p.plot([pi,ci], [pp,cp], color=c, lw=2, ls="--", alpha=0.9, zorder=5)
            ax_p.scatter([pi,ci], [pp,cp], color=c, s=70, zorder=6)
            ax_m.plot([mpi,mci], [mp,mc], color=c, lw=2, ls="--", alpha=0.9, zorder=5)
            ax_m.scatter([mpi,mci], [mp,mc], color=c, s=70, zorder=6)
            is_bull = s["direction"]=="BUY"
            star = " ★" if s["strength"]=="STRONG" else ""
            yoff = cp*(0.974 if is_bull else 1.026)
            ax_p.annotate(("▲ DIV BUY" if is_bull else "▼ DIV SELL")+star,
                          xy=(ci,cp), xytext=(ci,yoff),
                          fontsize=7.5, fontweight="bold", color=c,
                          ha="center", va="top" if is_bull else "bottom", zorder=7,
                          arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.6))

        for cobj in signal_crossovers:
            col = "#26a69a" if cobj["direction"]=="BUY" else "#ef5350"
            marker = "^" if cobj["direction"]=="BUY" else "v"
            sz = 90 if cobj["strength"]=="STRONG" else 55
            ax_m.scatter(cobj["bar"], cobj["macd"], marker=marker, color=col, s=sz,
                         edgecolors="white", linewidths=0.6, zorder=8)

        for cobj in zero_crossovers:
            col = "#26a69a" if cobj["direction"]=="BUY" else "#ef5350"
            ax_m.axvline(cobj["bar"], color=col, lw=0.8, ls=":", alpha=0.5, zorder=2)

        tick_step = max(1, n // 12)
        tick_pos = list(range(0,n,tick_step))
        tick_lbl = [str(df.loc[i,"Date"])[:10] for i in tick_pos]
        ax_m.set_xticks(tick_pos)
        ax_m.set_xticklabels(tick_lbl, rotation=35, ha="right", fontsize=7.5, color="#9e9e9e")
        ax_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
        ax_m.tick_params(colors="#9e9e9e", labelsize=8)
        ax_p.set_xlim(-1, n+8)

        cur = float(df["Close"].iloc[-1])
        ax_p.axhline(cur, color="white", lw=0.8, ls=":", alpha=0.8)
        ax_p.text(n+0.5, cur, f" {cur:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
        ax_p.set_title(f"{ticker} · Daily ({period}) · MACD Divergence + Crossover Detection",
                       color="white", fontsize=13, pad=10)
        ax_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
        for ax in [ax_p, ax_m]:
            ax.grid(axis="y", color="#1a1a1a", lw=0.5)
            ax.spines[:].set_color("#2a2a2a")

        ax_p.axvspan(recent_start, n-0.5, color="#ffffff", alpha=0.03, zorder=0)
        ax_m.axvspan(recent_start, n-0.5, color="#ffffff", alpha=0.03, zorder=0)
        ax_p.axvline(recent_start, color="#ffd54f", lw=1, ls=":", alpha=0.6)
        ax_m.axvline(recent_start, color="#ffd54f", lw=1, ls=":", alpha=0.6)
        ax_p.text(recent_start+0.5, ax_p.get_ylim()[0], f"◀ {RECENT_BARS}-bar window",
                  fontsize=7, color="#ffd54f", va="bottom", alpha=0.8)

        leg = [
            mpatches.Patch(color="#26a69a", label="Classic Bullish Div."),
            mpatches.Patch(color="#66bb6a", label="Double Tap Bullish Div."),
            mpatches.Patch(color="#ef5350", label="Classic Bearish Div."),
            mpatches.Patch(color="#ff8a65", label="Double Tap Bearish Div."),
            mpatches.Patch(color="#ffd54f", label=f"Recent {RECENT_BARS}-bar zone"),
        ]
        ax_p.legend(handles=leg, loc="upper left",
                    facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=8)

        ax_s.set_xlim(0,1); ax_s.set_ylim(0,1); ax_s.axis("off")
        buy_n = sum(1 for s in unique_signals if s["direction"]=="BUY")
        sell_n = sum(1 for s in unique_signals if s["direction"]=="SELL")
        strong_n = sum(1 for s in unique_signals if s["strength"]=="STRONG")
        regime_txt = "Bullish (MACD > 0)" if macd[-1] > 0 else "Bearish (MACD < 0)"
        cross_txt = "—"
        if latest_cross:
            cross_txt = f"{latest_cross['direction']} ({latest_cross['strength']}) on {str(df.loc[latest_cross['bar'],'Date'])[:10]}"
        ax_s.text(0.01, 0.70, f"Divergence — BUY: {buy_n}  SELL: {sell_n}  STRONG: {strong_n}/{len(unique_signals)}",
                  ha="left", va="center", fontsize=8.5, color="#cccccc", transform=ax_s.transAxes)
        ax_s.text(0.01, 0.35, f"Current regime: {regime_txt}   |   Latest signal-cross: {cross_txt}",
                  ha="left", va="center", fontsize=8.5, color="#ffd54f", transform=ax_s.transAxes)
        ax_s.text(0.5, -0.15, "⚠  For educational purposes only — not financial advice.",
                  ha="center", va="top", fontsize=7, color="#555",
                  transform=ax_s.transAxes, style="italic")

        plt.savefig(full_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)

        # ----- Recent chart (simplified) -----
        df_r = df.iloc[recent_start:].reset_index(drop=True)
        nr = len(df_r)
        lows_r = df_r["Low"].values; highs_r = df_r["High"].values
        macd_r = df_r["MACD"].values; sig_r = df_r["Signal"].values; hist_r = df_r["Hist"].values
        sl_r = [i-recent_start for i in swing_low_idx if i>=recent_start]
        sh_r = [i-recent_start for i in swing_high_idx if i>=recent_start]
        ml_r = [i-recent_start for i in macd_low_idx if i>=recent_start]
        mh_r = [i-recent_start for i in macd_high_idx if i>=recent_start]
        plot_signals_recent = []
        for s in recent_signals:
            sc = dict(s)
            sc["bar"] -= recent_start; sc["prev_bar"] -= recent_start
            sc["macd_bar"] -= recent_start; sc["macd_prev_bar"] -= recent_start
            plot_signals_recent.append(sc)
        plot_cross_recent = []
        for c in recent_crossovers:
            cc = dict(c); cc["bar"] -= recent_start; plot_cross_recent.append(cc)
        plot_zero_recent = []
        for c in recent_zero_crossovers:
            zc = dict(c); zc["bar"] -= recent_start; plot_zero_recent.append(zc)

        fig2 = plt.figure(figsize=(16,10), facecolor="#0d1117")
        gs2 = gridspec.GridSpec(3,1, height_ratios=[4,2,0.9], hspace=0.05)
        ax2_p = fig2.add_subplot(gs2[0]); ax2_m = fig2.add_subplot(gs2[1]); ax2_s = fig2.add_subplot(gs2[2])
        for ax in [ax2_p, ax2_m, ax2_s]:
            ax.set_facecolor("#0d1117")

        for i in range(nr):
            o = float(df_r.loc[i,"Open"]); c = float(df_r.loc[i,"Close"])
            h = float(df_r.loc[i,"High"]); lo = float(df_r.loc[i,"Low"])
            col = "#26a69a" if c >= o else "#ef5350"
            ax2_p.plot([i,i], [o,c], linewidth=6, color=col, solid_capstyle="round", zorder=3)
            ax2_p.plot([i,i], [lo,h], linewidth=1.5, color=col, alpha=0.7, zorder=2)

        for idx in sl_r:
            ax2_p.scatter(idx, lows_r[idx], marker="^", color="#26a69a", s=45, alpha=0.5, zorder=4)
        for idx in sh_r:
            ax2_p.scatter(idx, highs_r[idx], marker="v", color="#ef5350", s=45, alpha=0.5, zorder=4)

        xr = np.arange(nr)
        hist_colors_r = np.where(hist_r >= 0,
                                  np.where(np.append(np.diff(hist_r),0) >= 0, "#26a69a", "#80cbc4"),
                                  np.where(np.append(np.diff(hist_r),0) <= 0, "#ef5350", "#ef9a9a"))
        ax2_m.bar(xr, hist_r, color=hist_colors_r, width=0.7, alpha=0.6, zorder=1)
        ax2_m.plot(xr, macd_r, color="#42a5f5", linewidth=1.6, label="MACD", zorder=3)
        ax2_m.plot(xr, sig_r,  color="#ffa726", linewidth=1.4, label="Signal", zorder=3)
        ax2_m.axhline(0, color="#9e9e9e", lw=0.9, ls="-", alpha=0.6)
        ax2_m.set_ylabel("MACD", color="#9e9e9e", fontsize=9)
        ax2_m.legend(loc="upper left", facecolor="#1a1a1a", edgecolor="#444",
                     labelcolor="white", fontsize=8)

        for idx in ml_r:
            if not np.isnan(macd_r[idx]):
                ax2_m.scatter(idx, macd_r[idx], marker="^", color="#26a69a", s=45, alpha=0.5, zorder=4)
        for idx in mh_r:
            if not np.isnan(macd_r[idx]):
                ax2_m.scatter(idx, macd_r[idx], marker="v", color="#ef5350", s=45, alpha=0.5, zorder=4)

        for s in plot_signals_recent:
            c = SIG_COLOR[s["type"]]
            pi, ci = s["prev_bar"], s["bar"]
            mpi, mci = s["macd_prev_bar"], s["macd_bar"]
            pp, cp = s["price_prev"], s["price_cur"]
            mp, mc = s["macd_prev"], s["macd_cur"]
            ax2_p.plot([pi,ci], [pp,cp], color=c, lw=2.5, ls="--", alpha=0.9, zorder=5)
            ax2_p.scatter([pi,ci], [pp,cp], color=c, s=100, zorder=6)
            ax2_m.plot([mpi,mci], [mp,mc], color=c, lw=2.5, ls="--", alpha=0.9, zorder=5)
            ax2_m.scatter([mpi,mci], [mp,mc], color=c, s=100, zorder=6)
            is_bull = s["direction"]=="BUY"
            star = " ★" if s["strength"]=="STRONG" else ""
            yoff = cp*(0.971 if is_bull else 1.029)
            ax2_p.annotate(("▲ DIV BUY" if is_bull else "▼ DIV SELL")+star,
                           xy=(ci,cp), xytext=(ci,yoff),
                           fontsize=8.5, fontweight="bold", color=c,
                           ha="center", va="top" if is_bull else "bottom", zorder=7,
                           arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.7))

        for cobj in plot_cross_recent:
            col = "#26a69a" if cobj["direction"]=="BUY" else "#ef5350"
            marker = "^" if cobj["direction"]=="BUY" else "v"
            sz = 130 if cobj["strength"]=="STRONG" else 80
            ax2_m.scatter(cobj["bar"], cobj["macd"], marker=marker, color=col, s=sz,
                          edgecolors="white", linewidths=0.8, zorder=8)
            ax2_m.annotate(f"{cobj['direction']} cross" + (" ★" if cobj["strength"]=="STRONG" else ""),
                           xy=(cobj["bar"], cobj["macd"]),
                           xytext=(cobj["bar"], cobj["macd"] + (0.6 if cobj["direction"]=="BUY" else -0.6)),
                           fontsize=7, color=col, ha="center", fontweight="bold", zorder=8)

        for cobj in plot_zero_recent:
            col = "#26a69a" if cobj["direction"]=="BUY" else "#ef5350"
            ax2_m.axvline(cobj["bar"], color=col, lw=1, ls=":", alpha=0.6, zorder=2)

        tick_lbl2 = [str(df_r.loc[i,"Date"])[:10] for i in range(nr)]
        ax2_m.set_xticks(list(range(nr)))
        ax2_m.set_xticklabels(tick_lbl2, rotation=45, ha="right", fontsize=6.5, color="#9e9e9e")
        ax2_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
        ax2_m.tick_params(colors="#9e9e9e", labelsize=8)
        ax2_p.set_xlim(-0.5, nr)

        cur2 = float(df_r["Close"].iloc[-1])
        ax2_p.axhline(cur2, color="white", lw=0.8, ls=":", alpha=0.8)
        ax2_p.text(nr-0.5, cur2, f" {cur2:.2f}", va="center", fontsize=8,
                   color="white", fontweight="bold")
        ax2_p.set_title(f"{ticker} · Recent {nr} Candles · MACD (zoomed)",
                        color="white", fontsize=12, pad=10)
        ax2_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
        for ax in [ax2_p, ax2_m]:
            ax.grid(axis="y", color="#1a1a1a", lw=0.5)
            ax.spines[:].set_color("#2a2a2a")

        ax2_s.set_xlim(0,1); ax2_s.set_ylim(0,1); ax2_s.axis("off")
        lines = []
        if recent_signals:
            latest_r = recent_signals[-1]
            is_buy = latest_r["direction"]=="BUY"
            lines.append((f"Divergence: {latest_r['label']} ({latest_r['strength']})",
                          "#26a69a" if is_buy else "#ef5350"))
        else:
            lines.append(("Divergence: none in window", "#9e9e9e"))
        if recent_crossovers:
            lc = recent_crossovers[-1]
            lines.append((f"Signal cross: {lc['direction']} ({lc['strength']})",
                          "#26a69a" if lc["direction"]=="BUY" else "#ef5350"))
        else:
            lines.append(("Signal cross: none in window", "#9e9e9e"))
        if recent_zero_crossovers:
            zc = recent_zero_crossovers[-1]
            lines.append((f"Zero-line cross: {zc['direction']}",
                          "#26a69a" if zc["direction"]=="BUY" else "#ef5350"))
        else:
            lines.append(("Zero-line cross: none in window", "#9e9e9e"))

        y_positions = [0.78, 0.50, 0.22]
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

    # ── 11. RETURN ──────────────────────────────
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "current_price": float(df["Close"].iloc[-1]),
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "chart_paths": chart_paths,
        "raw": {
            "macd": float(macd[-1]),
            "signal": float(sig[-1]),
            "hist": float(hist[-1]),
            "num_divergences": len(unique_signals),
            "num_signal_crosses": len(signal_crossovers),
            "num_zero_crosses": len(zero_crossovers),
        },
        "series": {
            **ohlc_payload(df),
            "macd": clean_list(macd),
            "signal": clean_list(sig),
            "hist": clean_list(hist),
            "swing_low_bars": swing_low_idx,
            "swing_high_bars": swing_high_idx,
            "macd_low_bars": macd_low_idx,
            "macd_high_bars": macd_high_idx,
            "divergences": [
                {"type": s["type"], "label": s["label"], "direction": s["direction"],
                 "strength": s["strength"], "bar": int(s["bar"]), "prev_bar": int(s["prev_bar"])}
                for s in unique_signals
            ],
            "signal_crossovers": signal_crossovers,
            "zero_crossovers": zero_crossovers,
        },
    }

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    result = analyze_macd(ticker, save_chart=True)
    print(f"Signal: {result['signal']}, Confidence: {result['confidence']}, Reason: {result['reason']}")
    print("Charts:", result['chart_paths'])