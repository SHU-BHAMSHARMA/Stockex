"""
RSI Divergence Detector  ·  NEPSE / any ticker
────────────────────────────────────────────────
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

# ── Configuration (can be overridden) ──────────────────────────────────────
RSI_PERIOD      = 14
PIVOT_LEFT      = 3
PIVOT_RIGHT     = 3
PIVOT_ORDER     = max(PIVOT_LEFT, PIVOT_RIGHT)
DOUBLE_TAP_PCT  = 0.02
RSI_MIN_DIFF    = 1.5
TREND_FILTER_PCT= 0.0008
RSI_OVERBOUGHT  = 70
RSI_OVERSOLD    = 30
RECENT_BARS     = 20

def analyze_rsi(ticker, period='1y', interval='1d', save_chart=True, verbose=False):
    """
    Run RSI divergence analysis and return structured results.
    """
    # ── 1. Download ──────────────────────────────────────────────────────────
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df = df.reset_index()
    n = len(df)
    if n == 0:
        return {"error": f"No data for {ticker}"}

    # ── 2. RSI (Wilder) ─────────────────────────────────────────────────────
    def calc_rsi(close, period=RSI_PERIOD):
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = np.full(len(close), np.nan)
        avg_loss = np.full(len(close), np.nan)
        if len(close) > period:
            avg_gain[period] = gain.iloc[1:period+1].mean()
            avg_loss[period] = loss.iloc[1:period+1].mean()
            gain_vals = gain.values
            loss_vals = loss.values
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1]*(period-1) + gain_vals[i]) / period
                avg_loss[i] = (avg_loss[i-1]*(period-1) + loss_vals[i]) / period
        avg_gain_s = pd.Series(avg_gain, index=close.index)
        avg_loss_s = pd.Series(avg_loss, index=close.index)
        rs = avg_gain_s / avg_loss_s
        rsi = 100 - 100 / (1 + rs)
        rsi[(avg_loss_s == 0) & (avg_gain_s > 0)] = 100
        rsi[(avg_loss_s == 0) & (avg_gain_s == 0)] = 50
        return rsi

    df["RSI"] = calc_rsi(df["Close"])
    lows  = df["Low"].values
    highs = df["High"].values
    close = df["Close"].values
    rsi   = df["RSI"].values

    # ── 3. Pivot detection ──────────────────────────────────────────────────
    def find_pivots(arr, order, mode="low"):
        comparator = np.less if mode == "low" else np.greater
        idx = argrelextrema(np.asarray(arr, dtype=float), comparator, order=order)[0]
        return idx.tolist()

    swing_low_idx  = find_pivots(lows,  PIVOT_ORDER, "low")
    swing_high_idx = find_pivots(highs, PIVOT_ORDER, "high")
    rsi_low_idx    = find_pivots(rsi,   PIVOT_ORDER, "low")
    rsi_high_idx   = find_pivots(rsi,   PIVOT_ORDER, "high")

    # ── 4. Helpers ──────────────────────────────────────────────────────────
    MATCH_WINDOW = PIVOT_LEFT + PIVOT_RIGHT

    def nearest_rsi_pivot(rsi_pivot_list, bar_idx, match_window):
        candidates = [p for p in rsi_pivot_list if abs(p - bar_idx) <= match_window]
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p - bar_idx))

    def price_same_level(p1, p2, tol=DOUBLE_TAP_PCT):
        return abs(p1 - p2) / ((abs(p1) + abs(p2)) / 2) <= tol

    def rsi_higher(r_new, r_old, min_diff=RSI_MIN_DIFF):
        return r_new - r_old >= min_diff

    def rsi_lower(r_new, r_old, min_diff=RSI_MIN_DIFF):
        return r_old - r_new >= min_diff

    def trend_slope_pct(close_arr, start_idx, end_idx):
        seg = close_arr[start_idx:end_idx+1]
        if len(seg) < 3:
            return 0.0
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0]
        mean_price = np.mean(seg)
        return slope / mean_price if mean_price else 0.0

    def zone_strength(rsi_value, direction):
        if direction == "BUY" and rsi_value <= RSI_OVERSOLD + 5:
            return "STRONG"
        if direction == "SELL" and rsi_value >= RSI_OVERBOUGHT - 5:
            return "STRONG"
        return "WEAK"

    # ── 5. Divergence scan ──────────────────────────────────────────────────
    TYPE_PRIORITY = {"CLASSIC_BULL":0, "CLASSIC_BEAR":0,
                     "DOUBLE_TAP_BULL":1, "DOUBLE_TAP_BEAR":1}

    def run_divergence_scan(sl_idx, sh_idx, rl_idx, rh_idx,
                            lows_arr, highs_arr, close_arr, rsi_arr,
                            match_window):
        signals = []
        # Bullish
        for k in range(len(sl_idx)-1):
            prev_pi = sl_idx[k]; cur_pi = sl_idx[k+1]
            prev_price = lows_arr[prev_pi]; cur_price = lows_arr[cur_pi]
            prev_ri = nearest_rsi_pivot(rl_idx, prev_pi, match_window)
            cur_ri  = nearest_rsi_pivot(rl_idx, cur_pi,  match_window)
            if prev_ri is None or cur_ri is None or prev_ri >= cur_ri:
                continue
            if np.isnan(rsi_arr[prev_ri]) or np.isnan(rsi_arr[cur_ri]):
                continue
            prev_rv = rsi_arr[prev_ri]; cur_rv = rsi_arr[cur_ri]
            if cur_price < prev_price and rsi_higher(cur_rv, prev_rv):
                signals.append({"type":"CLASSIC_BULL","label":"Classic Bullish","direction":"BUY",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "rsi_cur":cur_rv,"rsi_prev":prev_rv,
                                "rsi_bar":cur_ri,"rsi_prev_bar":prev_ri,
                                "strength":zone_strength(cur_rv,"BUY")})
            elif price_same_level(cur_price, prev_price) and rsi_higher(cur_rv, prev_rv):
                slope = trend_slope_pct(close_arr, prev_pi, cur_pi)
                if slope <= -TREND_FILTER_PCT:
                    continue
                signals.append({"type":"DOUBLE_TAP_BULL","label":"Double Tap Bullish","direction":"BUY",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "rsi_cur":cur_rv,"rsi_prev":prev_rv,
                                "rsi_bar":cur_ri,"rsi_prev_bar":prev_ri,
                                "strength":zone_strength(cur_rv,"BUY")})
        # Bearish
        for k in range(len(sh_idx)-1):
            prev_pi = sh_idx[k]; cur_pi = sh_idx[k+1]
            prev_price = highs_arr[prev_pi]; cur_price = highs_arr[cur_pi]
            prev_ri = nearest_rsi_pivot(rh_idx, prev_pi, match_window)
            cur_ri  = nearest_rsi_pivot(rh_idx, cur_pi,  match_window)
            if prev_ri is None or cur_ri is None or prev_ri >= cur_ri:
                continue
            if np.isnan(rsi_arr[prev_ri]) or np.isnan(rsi_arr[cur_ri]):
                continue
            prev_rv = rsi_arr[prev_ri]; cur_rv = rsi_arr[cur_ri]
            if cur_price > prev_price and rsi_lower(cur_rv, prev_rv):
                signals.append({"type":"CLASSIC_BEAR","label":"Classic Bearish","direction":"SELL",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "rsi_cur":cur_rv,"rsi_prev":prev_rv,
                                "rsi_bar":cur_ri,"rsi_prev_bar":prev_ri,
                                "strength":zone_strength(cur_rv,"SELL")})
            elif price_same_level(cur_price, prev_price) and rsi_lower(cur_rv, prev_rv):
                slope = trend_slope_pct(close_arr, prev_pi, cur_pi)
                if slope >= TREND_FILTER_PCT:
                    continue
                signals.append({"type":"DOUBLE_TAP_BEAR","label":"Double Tap Bearish","direction":"SELL",
                                "bar":cur_pi,"prev_bar":prev_pi,
                                "price_cur":cur_price,"price_prev":prev_price,
                                "rsi_cur":cur_rv,"rsi_prev":prev_rv,
                                "rsi_bar":cur_ri,"rsi_prev_bar":prev_ri,
                                "strength":zone_strength(cur_rv,"SELL")})
        # deduplicate
        seen = {}
        for s in signals:
            b = s["bar"]
            if b not in seen or TYPE_PRIORITY[s["type"]] < TYPE_PRIORITY[seen[b]["type"]]:
                seen[b] = s
        return sorted(seen.values(), key=lambda s: s["bar"])

    unique_signals = run_divergence_scan(
        swing_low_idx, swing_high_idx, rsi_low_idx, rsi_high_idx,
        lows, highs, close, rsi, MATCH_WINDOW
    )

    # ── 6. Compute overall verdict ──────────────────────────────────────────
    buy_score = 0
    sell_score = 0
    for s in unique_signals:
        w = 2 if s["strength"] == "STRONG" else 1
        if s["direction"] == "BUY":
            buy_score += w
        else:
            sell_score += w

    if buy_score == 0 and sell_score == 0:
        signal = "NEUTRAL"
        confidence = 0
        reason = "No divergence signals detected."
    else:
        total = buy_score + sell_score
        net = (buy_score - sell_score) / total
        if net > 0.2:
            signal = "BUY"
            confidence = min(100, int(50 + 50 * net))
        elif net < -0.2:
            signal = "SELL"
            confidence = min(100, int(50 + 50 * abs(net)))
        else:
            signal = "NEUTRAL"
            confidence = 50
        reason = f"Divergence signals: {buy_score} bullish vs {sell_score} bearish (weighted)."

    # recent window summary
    recent_start = max(0, n - RECENT_BARS)
    recent_signals = [s for s in unique_signals if s["bar"] >= recent_start]
    latest = unique_signals[-1] if unique_signals else None

    # ── 7. Charts ────────────────────────────────────────────────────────────
    chart_paths = {"full": None, "recent": None}
    if save_chart:
        # Create directory for charts
        os.makedirs("static", exist_ok=True)
        base = f"static/{ticker}_RSI"
        full_chart = f"{base}_full.png"
        recent_chart = f"{base}_recent.png"

        # Plotting code (same as original, adapted)
        def plot_full():
            fig = plt.figure(figsize=(20,12), facecolor="#0d1117")
            gs = gridspec.GridSpec(3,1, height_ratios=[4,2,0.7], hspace=0.05)
            ax_p = fig.add_subplot(gs[0]); ax_r = fig.add_subplot(gs[1]); ax_s = fig.add_subplot(gs[2])
            for ax in [ax_p, ax_r, ax_s]:
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
            ax_r.plot(x, rsi, color="#ba68c8", linewidth=1.4, zorder=3)
            ax_r.axhline(70, color="#ef5350", lw=0.8, ls="--", alpha=0.5)
            ax_r.axhline(30, color="#26a69a", lw=0.8, ls="--", alpha=0.5)
            ax_r.fill_between(x, rsi, 70, where=(rsi >= 70), color="#ef5350", alpha=0.12)
            ax_r.fill_between(x, rsi, 30, where=(rsi <= 30), color="#26a69a", alpha=0.12)
            ax_r.set_ylim(0,100)
            ax_r.set_ylabel("RSI (14)", color="#9e9e9e", fontsize=9)
            ax_r.text(n+0.5, 70, " 70", va="center", fontsize=7, color="#ef5350")
            ax_r.text(n+0.5, 30, " 30", va="center", fontsize=7, color="#26a69a")
            for idx in rsi_low_idx:
                if not np.isnan(rsi[idx]):
                    ax_r.scatter(idx, rsi[idx], marker="^", color="#26a69a", s=18, alpha=0.35, zorder=4)
            for idx in rsi_high_idx:
                if not np.isnan(rsi[idx]):
                    ax_r.scatter(idx, rsi[idx], marker="v", color="#ef5350", s=18, alpha=0.35, zorder=4)
            SIG_COLOR = {"CLASSIC_BULL":"#26a69a","DOUBLE_TAP_BULL":"#66bb6a",
                         "CLASSIC_BEAR":"#ef5350","DOUBLE_TAP_BEAR":"#ff8a65"}
            for s in unique_signals:
                c = SIG_COLOR[s["type"]]
                pi, ci = s["prev_bar"], s["bar"]
                rpi, rci = s["rsi_prev_bar"], s["rsi_bar"]
                pp, cp = s["price_prev"], s["price_cur"]
                pr, cr = s["rsi_prev"], s["rsi_cur"]
                ax_p.plot([pi,ci], [pp,cp], color=c, lw=2, ls="--", alpha=0.9, zorder=5)
                ax_p.scatter([pi,ci], [pp,cp], color=c, s=70, zorder=6)
                ax_r.plot([rpi,rci], [pr,cr], color=c, lw=2, ls="--", alpha=0.9, zorder=5)
                ax_r.scatter([rpi,rci], [pr,cr], color=c, s=70, zorder=6)
                is_bull = s["direction"]=="BUY"
                star = " ★" if s["strength"]=="STRONG" else ""
                yoff = cp*(0.974 if is_bull else 1.026)
                ax_p.annotate(("▲ BUY" if is_bull else "▼ SELL")+star,
                              xy=(ci,cp), xytext=(ci,yoff),
                              fontsize=8, fontweight="bold", color=c,
                              ha="center", va="top" if is_bull else "bottom", zorder=7,
                              arrowprops=dict(arrowstyle="-", color=c, lw=0.7, alpha=0.6))
            tick_step = max(1, n//12)
            tick_pos = list(range(0,n,tick_step))
            tick_lbl = [str(df.loc[i,"Date"])[:10] for i in tick_pos]
            ax_r.set_xticks(tick_pos)
            ax_r.set_xticklabels(tick_lbl, rotation=35, ha="right", fontsize=7.5, color="#9e9e9e")
            ax_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
            ax_r.tick_params(colors="#9e9e9e", labelsize=8)
            ax_p.set_xlim(-1, n+8)
            cur = float(df["Close"].iloc[-1])
            ax_p.axhline(cur, color="white", lw=0.8, ls=":", alpha=0.8)
            ax_p.text(n+0.5, cur, f" {cur:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
            ax_p.set_title(f"{ticker} · Daily ({period}) · RSI Divergence Detection",
                           color="white", fontsize=13, pad=10)
            ax_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
            for ax in [ax_p, ax_r]:
                ax.grid(axis="y", color="#1a1a1a", lw=0.5)
                ax.spines[:].set_color("#2a2a2a")
            ax_p.axvspan(recent_start, n-0.5, color="#ffffff", alpha=0.03, zorder=0)
            ax_r.axvspan(recent_start, n-0.5, color="#ffffff", alpha=0.03, zorder=0)
            ax_p.axvline(recent_start, color="#ffd54f", lw=1, ls=":", alpha=0.6)
            ax_r.axvline(recent_start, color="#ffd54f", lw=1, ls=":", alpha=0.6)
            ax_p.text(recent_start+0.5, ax_p.get_ylim()[0], f"◀ {RECENT_BARS}-bar window",
                      fontsize=7, color="#ffd54f", va="bottom", alpha=0.8)
            leg = [
                mpatches.Patch(color="#26a69a", label="Classic Bullish"),
                mpatches.Patch(color="#66bb6a", label="Double Tap Bullish"),
                mpatches.Patch(color="#ef5350", label="Classic Bearish"),
                mpatches.Patch(color="#ff8a65", label="Double Tap Bearish"),
                mpatches.Patch(color="#ffd54f", label=f"Recent {RECENT_BARS}-bar zone"),
            ]
            ax_p.legend(handles=leg, loc="upper left",
                        facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=8)
            ax_s.set_xlim(0,1); ax_s.set_ylim(0,1); ax_s.axis("off")
            if latest:
                col = "#26a69a" if latest["direction"]=="BUY" else "#ef5350"
                bg = "#0d2b28" if latest["direction"]=="BUY" else "#2b0d0d"
                d = str(df.loc[latest["bar"], "Date"])[:10]
                ax_s.add_patch(mpatches.FancyBboxPatch((0.01,0.05),0.98,0.90,
                    boxstyle="round,pad=0.01", facecolor=bg, edgecolor=col, lw=1.5,
                    transform=ax_s.transAxes, zorder=1))
                ax_s.add_patch(mpatches.FancyBboxPatch((0.02,0.18),0.09,0.64,
                    boxstyle="round,pad=0.01", facecolor=col, edgecolor="none",
                    transform=ax_s.transAxes, zorder=2))
                ax_s.text(0.065,0.52, latest["direction"], ha="center", va="center",
                          fontsize=9, fontweight="bold", color="#0d1117",
                          transform=ax_s.transAxes, zorder=3)
                ax_s.text(0.13,0.70, f"Full-period latest: {latest['label']} ({latest['strength']})",
                          ha="left", va="center", fontsize=9, fontweight="bold",
                          color=col, transform=ax_s.transAxes)
                ax_s.text(0.13,0.35, f"Date: {d}   Price: {latest['price_cur']:.2f}   RSI: {latest['rsi_cur']:.1f}",
                          ha="left", va="center", fontsize=8, color="#cccccc",
                          transform=ax_s.transAxes)
                buy_n = sum(1 for s in unique_signals if s["direction"]=="BUY")
                sell_n = sum(1 for s in unique_signals if s["direction"]=="SELL")
                strong_n = sum(1 for s in unique_signals if s["strength"]=="STRONG")
                ax_s.text(0.60,0.70, f"BUY signals:  {buy_n}   SELL signals: {sell_n}",
                          ha="left", va="center", fontsize=8, color="#cccccc",
                          transform=ax_s.transAxes)
                ax_s.text(0.60,0.35, f"STRONG (near RSI extreme): {strong_n} / {len(unique_signals)}",
                          ha="left", va="center", fontsize=8, color="#ffd54f",
                          transform=ax_s.transAxes)
            else:
                ax_s.text(0.5,0.52, "No divergence signals detected over this period.",
                          ha="center", va="center", fontsize=9, color="#9e9e9e",
                          transform=ax_s.transAxes)
            ax_s.text(0.5,-0.1, "⚠  For educational purposes only — not financial advice.",
                      ha="center", va="top", fontsize=7, color="#555",
                      transform=ax_s.transAxes, style="italic")
            plt.savefig(full_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)

        def plot_recent():
            df_r = df.iloc[recent_start:].reset_index(drop=True)
            nr = len(df_r)
            lows_r = df_r["Low"].values; highs_r = df_r["High"].values; rsi_r = df_r["RSI"].values
            sl_r = [i-recent_start for i in swing_low_idx if i>=recent_start]
            sh_r = [i-recent_start for i in swing_high_idx if i>=recent_start]
            rl_r = [i-recent_start for i in rsi_low_idx if i>=recent_start]
            rh_r = [i-recent_start for i in rsi_high_idx if i>=recent_start]
            plot_signals_recent = []
            for s in recent_signals:
                sc = dict(s)
                sc["bar"] -= recent_start; sc["prev_bar"] -= recent_start
                sc["rsi_bar"] -= recent_start; sc["rsi_prev_bar"] -= recent_start
                plot_signals_recent.append(sc)
            fig2 = plt.figure(figsize=(16,10), facecolor="#0d1117")
            gs2 = gridspec.GridSpec(3,1, height_ratios=[4,2,0.8], hspace=0.05)
            ax2_p = fig2.add_subplot(gs2[0]); ax2_r = fig2.add_subplot(gs2[1]); ax2_s = fig2.add_subplot(gs2[2])
            for ax in [ax2_p, ax2_r, ax2_s]:
                ax.set_facecolor("#0d1117")
            for i in range(nr):
                o = float(df_r.loc[i,"Open"]); c = float(df_r.loc[i,"Close"])
                h = float(df_r.loc[i,"High"]); lo = float(df_r.loc[i,"Low"])
                col = "#26a69a" if c>=o else "#ef5350"
                ax2_p.plot([i,i], [o,c], linewidth=6, color=col, solid_capstyle="round", zorder=3)
                ax2_p.plot([i,i], [lo,h], linewidth=1.5, color=col, alpha=0.7, zorder=2)
            for idx in sl_r:
                ax2_p.scatter(idx, lows_r[idx], marker="^", color="#26a69a", s=45, alpha=0.5, zorder=4)
            for idx in sh_r:
                ax2_p.scatter(idx, highs_r[idx], marker="v", color="#ef5350", s=45, alpha=0.5, zorder=4)
            xr = np.arange(nr)
            ax2_r.plot(xr, rsi_r, color="#ba68c8", linewidth=1.6, zorder=3)
            ax2_r.axhline(70, color="#ef5350", lw=0.8, ls="--", alpha=0.5)
            ax2_r.axhline(30, color="#26a69a", lw=0.8, ls="--", alpha=0.5)
            ax2_r.fill_between(xr, rsi_r, 70, where=(rsi_r>=70), color="#ef5350", alpha=0.12)
            ax2_r.fill_between(xr, rsi_r, 30, where=(rsi_r<=30), color="#26a69a", alpha=0.12)
            ax2_r.set_ylim(0,100)
            ax2_r.set_ylabel("RSI (14)", color="#9e9e9e", fontsize=9)
            for idx in rl_r:
                if not np.isnan(rsi_r[idx]):
                    ax2_r.scatter(idx, rsi_r[idx], marker="^", color="#26a69a", s=45, alpha=0.5, zorder=4)
            for idx in rh_r:
                if not np.isnan(rsi_r[idx]):
                    ax2_r.scatter(idx, rsi_r[idx], marker="v", color="#ef5350", s=45, alpha=0.5, zorder=4)
            SIG_COLOR = {"CLASSIC_BULL":"#26a69a","DOUBLE_TAP_BULL":"#66bb6a",
                         "CLASSIC_BEAR":"#ef5350","DOUBLE_TAP_BEAR":"#ff8a65"}
            for s in plot_signals_recent:
                c = SIG_COLOR[s["type"]]
                pi, ci = s["prev_bar"], s["bar"]
                rpi, rci = s["rsi_prev_bar"], s["rsi_bar"]
                pp, cp = s["price_prev"], s["price_cur"]
                pr, cr = s["rsi_prev"], s["rsi_cur"]
                ax2_p.plot([pi,ci], [pp,cp], color=c, lw=2.5, ls="--", alpha=0.9, zorder=5)
                ax2_p.scatter([pi,ci], [pp,cp], color=c, s=100, zorder=6)
                ax2_r.plot([rpi,rci], [pr,cr], color=c, lw=2.5, ls="--", alpha=0.9, zorder=5)
                ax2_r.scatter([rpi,rci], [pr,cr], color=c, s=100, zorder=6)
                is_bull = s["direction"]=="BUY"
                star = " ★" if s["strength"]=="STRONG" else ""
                yoff = cp*(0.971 if is_bull else 1.029)
                ax2_p.annotate(("▲ BUY" if is_bull else "▼ SELL")+star,
                               xy=(ci,cp), xytext=(ci,yoff),
                               fontsize=9, fontweight="bold", color=c,
                               ha="center", va="top" if is_bull else "bottom", zorder=7,
                               arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.7))
            tick_lbl2 = [str(df_r.loc[i,"Date"])[:10] for i in range(nr)]
            ax2_r.set_xticks(list(range(nr)))
            ax2_r.set_xticklabels(tick_lbl2, rotation=45, ha="right", fontsize=6.5, color="#9e9e9e")
            ax2_p.tick_params(labelbottom=False, colors="#9e9e9e", labelsize=8)
            ax2_r.tick_params(colors="#9e9e9e", labelsize=8)
            ax2_p.set_xlim(-0.5, nr)
            cur2 = float(df_r["Close"].iloc[-1])
            ax2_p.axhline(cur2, color="white", lw=0.8, ls=":", alpha=0.8)
            ax2_p.text(nr-0.5, cur2, f" {cur2:.2f}", va="center", fontsize=8,
                       color="white", fontweight="bold")
            ax2_p.set_title(f"{ticker} · Recent {nr} Candles · RSI Divergence (zoomed)",
                            color="white", fontsize=12, pad=10)
            ax2_p.set_ylabel("Price", color="#9e9e9e", fontsize=9)
            for ax in [ax2_p, ax2_r]:
                ax.grid(axis="y", color="#1a1a1a", lw=0.5)
                ax.spines[:].set_color("#2a2a2a")
            ax2_s.set_xlim(0,1); ax2_s.set_ylim(0,1); ax2_s.axis("off")
            if recent_signals:
                latest_r = recent_signals[-1]
                is_buy = latest_r["direction"]=="BUY"
                col_c = "#26a69a" if is_buy else "#ef5350"
                bg_c = "#0d2b28" if is_buy else "#2b0d0d"
                d = str(df.loc[latest_r["bar"], "Date"])[:10]
                ax2_s.add_patch(mpatches.FancyBboxPatch((0.01,0.05),0.98,0.90,
                    boxstyle="round,pad=0.01", facecolor=bg_c, edgecolor=col_c, lw=1.5,
                    transform=ax2_s.transAxes, zorder=1))
                ax2_s.add_patch(mpatches.FancyBboxPatch((0.02,0.18),0.09,0.64,
                    boxstyle="round,pad=0.01", facecolor=col_c, edgecolor="none",
                    transform=ax2_s.transAxes, zorder=2))
                ax2_s.text(0.065,0.52, latest_r["direction"], ha="center", va="center",
                           fontsize=8, fontweight="bold", color="#0d1117",
                           transform=ax2_s.transAxes, zorder=3)
                ax2_s.text(0.13,0.70, f"Recent verdict: {latest_r['label']} ({latest_r['strength']})",
                           ha="left", va="center", fontsize=9, fontweight="bold",
                           color=col_c, transform=ax2_s.transAxes)
                ax2_s.text(0.13,0.35,
                           f"Date: {d}   Price: {latest_r['price_cur']:.2f}   RSI: {latest_r['rsi_cur']:.1f}",
                           ha="left", va="center", fontsize=8, color="#cccccc",
                           transform=ax2_s.transAxes)
            else:
                ax2_s.text(0.5,0.52, "⬜ No qualifying divergence signal in this window",
                           ha="center", va="center", fontsize=9, color="#9e9e9e",
                           transform=ax2_s.transAxes)
            ax2_s.text(0.5,-0.1, "⚠  Same detection rules as the full-period scan. Not financial advice.",
                       ha="center", va="top", fontsize=7, color="#555",
                       transform=ax2_s.transAxes, style="italic")
            plt.savefig(recent_chart, dpi=150, bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig2)

        plot_full()
        plot_recent()
        chart_paths["full"] = full_chart
        chart_paths["recent"] = recent_chart

    # ── 8. Build return dict ──────────────────────────────────────────────
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
            "num_signals": len(unique_signals),
            "buy_signals": [s for s in unique_signals if s["direction"]=="BUY"],
            "sell_signals": [s for s in unique_signals if s["direction"]=="SELL"],
        },
        "series": {
            **ohlc_payload(df),
            "rsi": clean_list(rsi),
            "swing_low_bars": swing_low_idx,
            "swing_high_bars": swing_high_idx,
            "rsi_low_bars": rsi_low_idx,
            "rsi_high_bars": rsi_high_idx,
            "divergences": [
                {"type": s["type"], "label": s["label"], "direction": s["direction"],
                 "strength": s["strength"], "bar": int(s["bar"]), "prev_bar": int(s["prev_bar"]),
                 "rsi_bar": int(s["rsi_bar"]), "rsi_prev_bar": int(s["rsi_prev_bar"]),
                 "price_cur": float(s["price_cur"]), "price_prev": float(s["price_prev"]),
                 "rsi_cur": float(s["rsi_cur"]), "rsi_prev": float(s["rsi_prev"])}
                for s in unique_signals
            ],
        },
    }

if __name__ == "__main__":
    # For standalone testing
    ticker = input("Enter stock ticker: ").strip().upper()
    result = analyze_rsi(ticker, save_chart=True, verbose=True)
    print(f"\nSignal: {result['signal']}, Confidence: {result['confidence']}, Reason: {result['reason']}")
    print("Charts saved:", result['chart_paths'])