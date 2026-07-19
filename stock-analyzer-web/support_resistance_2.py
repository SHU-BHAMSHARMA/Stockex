# support_resistance_2.py (adapted)
"""
Support/Resistance Zone Detector with signal engine.
"""
import matplotlib; 
matplotlib.use('Agg')
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import os

from series_utils import clean_list, ohlc_payload

def analyze_sr(ticker, period='2y', interval='1d', save_chart=True):
    """
    Run Support/Resistance zone analysis and return structured results.
    """
    # ── 1. Download & resample to weekly ──────
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    weekly = df.resample("W").agg(
        {"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"}
    )
    weekly.dropna(inplace=True)
    weekly = weekly.reset_index()
    if len(weekly) == 0:
        return {"error": f"No weekly data for {ticker}"}

    current_price = float(weekly["Close"].iloc[-1])

    # ── 2. Find pivots ──────────────────────────
    ORDER = 3
    highs = weekly["High"].values
    lows  = weekly["Low"].values
    local_max_idx = argrelextrema(highs, np.greater_equal, order=ORDER)[0]
    local_min_idx = argrelextrema(lows,  np.less_equal,    order=ORDER)[0]
    pivot_highs = highs[local_max_idx]
    pivot_lows  = lows[local_min_idx]

    # ── 3. Filter to ±20% of current price ──────
    RANGE_PCT = 0.20
    lo_bound = current_price * (1 - RANGE_PCT)
    hi_bound = current_price * (1 + RANGE_PCT)
    pivot_highs_near = pivot_highs[(pivot_highs >= lo_bound) & (pivot_highs <= hi_bound)]
    pivot_lows_near  = pivot_lows [(pivot_lows  >= lo_bound) & (pivot_lows  <= hi_bound)]

    # ── 4. Cluster pivots ───────────────────────
    CLUSTER_TOL = 0.02

    def cluster_pivots(pivots, current_price, tol=CLUSTER_TOL):
        if len(pivots) == 0: return []
        prices = np.sort(pivots)
        zones = []
        used = np.zeros(len(prices), dtype=bool)
        for i in range(len(prices)):
            if used[i]: continue
            cluster = [prices[i]]
            for j in range(i+1, len(prices)):
                if used[j]: continue
                if abs(prices[j] - np.mean(cluster)) / current_price <= tol:
                    cluster.append(prices[j])
                    used[j] = True
            used[i] = True
            zones.append({
                "low": round(min(cluster), 2),
                "high": round(max(cluster), 2),
                "mid": round(np.mean(cluster), 2),
                "strength": len(cluster),
            })
        return zones

    resistance_zones = cluster_pivots(pivot_highs_near, current_price)
    support_zones    = cluster_pivots(pivot_lows_near,  current_price)
    resistance_zones.sort(key=lambda z: z["strength"], reverse=True)
    support_zones.sort(key=lambda z: z["strength"], reverse=True)

    def strength_label(n):
        if n >= 4: return "Very Strong"
        if n == 3: return "Strong"
        if n == 2: return "Moderate"
        return "Weak"

    # ── 5. Signal engine ────────────────────────
    PROXIMITY_PCT = 0.015

    def evaluate_signal(price, support_zones, resistance_zones):
        signals = []
        for z in support_zones:
            w = z["strength"]
            if z["low"] <= price <= z["high"]:
                signals.append(("STRONG BUY",  2 * w, z, "price inside support zone"))
            elif z["high"] < price <= z["high"] * (1 + PROXIMITY_PCT):
                signals.append(("BUY",         1 * w, z, "price just above support zone"))

        for z in resistance_zones:
            w = z["strength"]
            if z["low"] <= price <= z["high"]:
                signals.append(("STRONG SELL", 2 * w, z, "price inside resistance zone"))
            elif z["low"] * (1 - PROXIMITY_PCT) <= price < z["low"]:
                signals.append(("SELL",        1 * w, z, "price just below resistance zone"))

        if not signals:
            nearest_s = min(support_zones,    key=lambda z: abs(z["mid"] - price)) if support_zones else None
            nearest_r = min(resistance_zones, key=lambda z: abs(z["mid"] - price)) if resistance_zones else None
            return "NEUTRAL", 0, nearest_s, nearest_r, "price between zones — wait for edge"

        signals.sort(key=lambda x: x[1], reverse=True)
        best_action, best_score, best_zone, reason = signals[0]
        max_possible = 2 * 4
        confidence = min(int(best_score / max_possible * 100), 100)
        return best_action, confidence, best_zone, None, reason

    signal, confidence, trigger_zone, nearest_r, reason = evaluate_signal(
        current_price, support_zones, resistance_zones
    )

    nearest_support = min(support_zones, key=lambda z: abs(z["mid"] - current_price)) if support_zones else None
    nearest_resistance = min(resistance_zones, key=lambda z: abs(z["mid"] - current_price)) if resistance_zones else None

    stop_loss = take_profit = rr = None
    if signal in ("STRONG BUY", "BUY") and nearest_support and nearest_resistance:
        stop_loss = round(nearest_support["low"] * 0.99, 2)
        take_profit = round(nearest_resistance["mid"], 2)
        rr_raw = (take_profit - current_price) / (current_price - stop_loss) if current_price != stop_loss else 0
        rr = round(rr_raw, 2)
    elif signal in ("STRONG SELL", "SELL") and nearest_support and nearest_resistance:
        stop_loss = round(nearest_resistance["high"] * 1.01, 2)
        take_profit = round(nearest_support["mid"], 2)
        rr_raw = (current_price - take_profit) / (stop_loss - current_price) if current_price != stop_loss else 0
        rr = round(rr_raw, 2)

    # ── 6. CHARTS ────────────────────────────────
    chart_paths = {"full": None}
    if save_chart:
        os.makedirs("static", exist_ok=True)
        chart_file = f"static/{ticker}_SR.png"
        n_weeks = len(weekly)

        fig, (ax, ax_sig) = plt.subplots(
            2, 1, figsize=(18, 11),
            gridspec_kw={"height_ratios": [4, 1]},
            facecolor="#0d1117"
        )
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax_sig.set_facecolor("#0d1117")

        for i in range(n_weeks):
            o = float(weekly.loc[i, "Open"])
            c = float(weekly.loc[i, "Close"])
            h = float(weekly.loc[i, "High"])
            lo = float(weekly.loc[i, "Low"])
            color = "#26a69a" if c >= o else "#ef5350"
            ax.plot([i, i], [o, c], linewidth=7, color=color, solid_capstyle="round", zorder=3)
            ax.plot([i, i], [lo, h], linewidth=1, color=color, alpha=0.7, zorder=2)

        ALPHA_MAP = {1:0.12, 2:0.20, 3:0.28, 4:0.36}
        for z in resistance_zones:
            alpha = ALPHA_MAP.get(min(z["strength"],4), 0.36)
            ax.axhspan(z["low"], z["high"], color="#ef5350", alpha=alpha, zorder=1)
            ax.axhline(z["mid"], color="#ef5350", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
            label = f"R  ${z['low']:.0f}–${z['high']:.0f}  [{strength_label(z['strength'])}]"
            ax.text(n_weeks + 0.3, z["mid"], label, va="center", ha="left",
                    fontsize=8, color="#ef5350", fontweight="bold")

        for z in support_zones:
            alpha = ALPHA_MAP.get(min(z["strength"],4), 0.36)
            ax.axhspan(z["low"], z["high"], color="#26a69a", alpha=alpha, zorder=1)
            ax.axhline(z["mid"], color="#26a69a", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)
            label = f"S  ${z['low']:.0f}–${z['high']:.0f}  [{strength_label(z['strength'])}]"
            ax.text(n_weeks + 0.3, z["mid"], label, va="center", ha="left",
                    fontsize=8, color="#26a69a", fontweight="bold")

        ax.axhline(current_price, color="white", linewidth=1, linestyle=":", alpha=0.9, zorder=4)
        ax.text(n_weeks + 0.3, current_price, f"  NOW ${current_price:.2f}",
                va="center", ha="left", fontsize=8, color="white", fontweight="bold")

        tick_step = 8
        tick_pos = list(range(0, n_weeks, tick_step))
        tick_label = [str(weekly.loc[i, "Date"])[:10] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_label, rotation=35, ha="right", fontsize=8, color="#9e9e9e")
        ax.tick_params(axis="y", colors="#9e9e9e", labelsize=9)
        ax.set_xlim(-1, n_weeks + 18)
        ax.set_title(f"{ticker} · Weekly Candlestick  ·  Support & Resistance Zones (±20% of price)",
                     color="white", fontsize=13, pad=12)
        ax.set_xlabel("Date", color="#9e9e9e", fontsize=9)
        ax.set_ylabel("Price (USD)", color="#9e9e9e", fontsize=9)
        ax.grid(axis="y", color="#2a2a2a", linewidth=0.5, zorder=0)
        ax.spines[:].set_color("#2a2a2a")

        legend_handles = [
            mpatches.Patch(color="#ef5350", alpha=0.5, label="Resistance zone"),
            mpatches.Patch(color="#26a69a", alpha=0.5, label="Support zone"),
            mpatches.Patch(color="#26a69a", label="Bullish candle"),
            mpatches.Patch(color="#ef5350", label="Bearish candle"),
        ]
        ax.legend(handles=legend_handles, loc="upper left",
                  facecolor="#1a1a1a", edgecolor="#444", labelcolor="white", fontsize=8)

        # Signal panel
        ax_sig.set_xlim(0,1); ax_sig.set_ylim(0,1); ax_sig.axis("off")
        SIGNAL_CHART_COLORS = {
            "STRONG BUY":  ("#26a69a", "#0d2b28"),
            "BUY":         ("#66bb6a", "#0d2318"),
            "STRONG SELL": ("#ef5350", "#2b0d0d"),
            "SELL":        ("#ff8a65", "#2b1a0d"),
            "NEUTRAL":     ("#ffd54f", "#2b280d"),
        }
        fg_color, bg_color = SIGNAL_CHART_COLORS.get(signal, ("#ffffff", "#1a1a1a"))

        card = mpatches.FancyBboxPatch(
            (0.01, 0.08), 0.98, 0.84,
            boxstyle="round,pad=0.01",
            facecolor=bg_color, edgecolor=fg_color, linewidth=1.5,
            transform=ax_sig.transAxes, zorder=1
        )
        ax_sig.add_patch(card)

        badge = mpatches.FancyBboxPatch(
            (0.02, 0.25), 0.13, 0.50,
            boxstyle="round,pad=0.01",
            facecolor=fg_color, edgecolor="none",
            transform=ax_sig.transAxes, zorder=2
        )
        ax_sig.add_patch(badge)
        ax_sig.text(0.085, 0.52, signal, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#0d1117",
                    transform=ax_sig.transAxes, zorder=3)

        if signal != "NEUTRAL":
            bar_x, bar_y, bar_w, bar_h = 0.17, 0.58, 0.30, 0.12
            ax_sig.add_patch(mpatches.FancyBboxPatch(
                (bar_x, bar_y), bar_w, bar_h,
                boxstyle="round,pad=0.005", facecolor="#1e1e1e", edgecolor="#444",
                transform=ax_sig.transAxes, zorder=2
            ))
            ax_sig.add_patch(mpatches.FancyBboxPatch(
                (bar_x, bar_y), bar_w * confidence / 100, bar_h,
                boxstyle="round,pad=0.005", facecolor=fg_color, edgecolor="none",
                transform=ax_sig.transAxes, zorder=3
            ))
            ax_sig.text(bar_x - 0.01, bar_y + bar_h/2,
                        "Confidence", ha="right", va="center",
                        fontsize=8, color="#9e9e9e", transform=ax_sig.transAxes)
            ax_sig.text(bar_x + bar_w + 0.01, bar_y + bar_h/2,
                        f"{confidence}%", ha="left", va="center",
                        fontsize=8, fontweight="bold", color=fg_color, transform=ax_sig.transAxes)

        ax_sig.text(0.17, 0.36, f"Reason: {reason}", ha="left", va="center",
                    fontsize=8.5, color="#cccccc", transform=ax_sig.transAxes)

        if stop_loss and take_profit:
            ax_sig.text(0.52, 0.70, f"Stop-loss", ha="center", va="center",
                        fontsize=7, color="#9e9e9e", transform=ax_sig.transAxes)
            ax_sig.text(0.52, 0.48, f"${stop_loss:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#ef5350", transform=ax_sig.transAxes)

            ax_sig.text(0.66, 0.70, f"Target", ha="center", va="center",
                        fontsize=7, color="#9e9e9e", transform=ax_sig.transAxes)
            ax_sig.text(0.66, 0.48, f"${take_profit:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#26a69a", transform=ax_sig.transAxes)

            ax_sig.text(0.80, 0.70, "Risk / Reward", ha="center", va="center",
                        fontsize=7, color="#9e9e9e", transform=ax_sig.transAxes)
            rr_color = "#26a69a" if rr >= 1.5 else "#ffd54f" if rr >= 1.0 else "#ef5350"
            ax_sig.text(0.80, 0.48, f"1 : {rr}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=rr_color, transform=ax_sig.transAxes)

        if nearest_support:
            ax_sig.text(0.52, 0.25,
                        f"Nearest support:    ${nearest_support['low']:.2f}–${nearest_support['high']:.2f}  "
                        f"[{strength_label(nearest_support['strength'])}]",
                        ha="left", va="center", fontsize=7.5, color="#26a69a", transform=ax_sig.transAxes)
        if nearest_resistance:
            ax_sig.text(0.52, 0.10,
                        f"Nearest resistance: ${nearest_resistance['low']:.2f}–${nearest_resistance['high']:.2f}  "
                        f"[{strength_label(nearest_resistance['strength'])}]",
                        ha="left", va="center", fontsize=7.5, color="#ef5350", transform=ax_sig.transAxes)

        ax_sig.text(0.5, -0.05,
                    "⚠  For educational purposes only — not financial advice.",
                    ha="center", va="top", fontsize=7, color="#555555",
                    transform=ax_sig.transAxes, style="italic")

        plt.subplots_adjust(hspace=0.08)
        plt.tight_layout()
        plt.savefig(chart_file, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        chart_paths["full"] = chart_file

    # ── 7. RETURN ────────────────────────────────
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "current_price": current_price,
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": rr,
        "support_zones": support_zones,
        "resistance_zones": resistance_zones,
        "chart_paths": chart_paths,
        "series": {
            **ohlc_payload(weekly),
        },
    }

if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    result = analyze_sr(ticker, save_chart=True)
    print(f"Signal: {result['signal']}, Confidence: {result['confidence']}, Reason: {result['reason']}")
    if result.get("stop_loss"):
        print(f"Stop: {result['stop_loss']}, Target: {result['take_profit']}, RR: {result['risk_reward']}")
    print("Chart:", result['chart_paths'])