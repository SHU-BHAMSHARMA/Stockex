"""
scoring.py  ·  composite verdict + momentum gauge
──────────────────────────────────────────────────────────────────
Aggregation-only layer. Every individual indicator (RSI, MACD, Bollinger,
Ichimoku, Order Blocks, S/R, "Other") keeps deciding its own signal and
confidence exactly as before — this module only combines those already
-computed outputs into one composite reading, the way TradingView's
"Technical Rating" widget aggregates oscillators + moving averages into
a single Strong Sell..Strong Buy scale. Nothing here re-derives or
overrides a single indicator's own verdict.

Composite Technical Rating (an established convention, not invented
here): each indicator's signal is mapped to -1 (SELL) / 0 (NEUTRAL) /
+1 (BUY), scaled by that indicator's own confidence (0-1) and by a
fixed importance weight, then averaged into a single score in [-1, +1].
That score is multiplied by 100 to give a "Composite Rating" from -100
to +100 and banded into the standard 5-tier scale used by most
technical-rating widgets:

    rating >= 60          STRONG BUY
    20 <= rating < 60      BUY
    -20 < rating < 20      NEUTRAL
    -60 < rating <= -20    SELL
    rating <= -60          STRONG SELL
"""

INDICATOR_WEIGHTS = {
    "RSI": 1.5,
    "Bollinger": 1.0,
    "Ichimoku": 1.2,
    "MACD": 1.3,
    "Other": 0.8,
    "SR": 1.0,
    "OrderBlock": 1.4,
}

RATING_BANDS = [
    (60, "STRONG BUY"),
    (20, "BUY"),
    (-20, "NEUTRAL"),
    (-60, "SELL"),
    (-101, "STRONG SELL"),
]


def normalize_signal(signal_str):
    s = (signal_str or "").upper()
    if "BUY" in s and "WATCH" not in s:
        return 1
    if "SELL" in s and "WATCH" not in s:
        return -1
    return 0


def rating_label(rating):
    for threshold, label in RATING_BANDS:
        if rating >= threshold:
            return label
    return "STRONG SELL"


def composite_rating(indicator_results):
    """
    indicator_results: dict name -> result dict (each with .signal/.confidence
    or an 'error' key). Returns the composite rating payload used by the
    "Final Verdict" panel, plus a per-indicator score breakdown so the UI
    can show how each indicator contributed.
    """
    breakdown = []
    total_weight = 0.0
    weighted_sum = 0.0

    for name, res in indicator_results.items():
        weight = INDICATOR_WEIGHTS.get(name, 1.0)
        if not res or "error" in res:
            breakdown.append({"indicator": name, "included": False, "reason": res.get("error") if res else "no data"})
            continue
        sig = res.get("signal", "NEUTRAL")
        conf = max(0, min(100, res.get("confidence", 0))) / 100.0
        num_sig = normalize_signal(sig)
        score = num_sig * conf                     # -1..+1
        contribution = score * weight
        total_weight += weight
        weighted_sum += contribution
        breakdown.append({
            "indicator": name, "included": True, "signal": sig,
            "confidence": res.get("confidence", 0), "weight": weight,
            "score": round(score, 3), "weighted_contribution": round(contribution, 3),
        })

    if total_weight == 0:
        return {
            "rating": 0, "label": "NEUTRAL", "confidence": 0,
            "breakdown": breakdown, "reason": "No indicator data available.",
        }

    avg_score = weighted_sum / total_weight         # -1..+1
    rating = round(avg_score * 100, 1)               # -100..+100
    label = rating_label(rating)
    confidence = int(abs(avg_score) * 100)

    return {
        "rating": rating, "label": label, "confidence": confidence,
        "breakdown": breakdown,
        "reason": f"Weighted composite of {sum(1 for b in breakdown if b['included'])} "
                  f"of {len(breakdown)} indicators: {label} ({rating:+.1f}/100).",
    }


def momentum_gauge(other_result, rsi_result, macd_result):
    """
    A 0-100 momentum gauge (0 = extreme bearish momentum, 50 = neutral,
    100 = extreme bullish momentum), built only from values the
    "Other"/RSI/MACD modules already compute (RSI level, Stochastic %K,
    ADX + DI direction, MACD histogram sign). Purely a display aggregate.
    """
    components = []

    raw_other = (other_result or {}).get("raw", {})
    raw_rsi_price = (rsi_result or {}).get("current_price")
    rsi_val = None
    if rsi_result and "series" in rsi_result and rsi_result["series"].get("rsi"):
        vals = [v for v in rsi_result["series"]["rsi"] if v is not None]
        rsi_val = vals[-1] if vals else None
    if rsi_val is not None:
        components.append(("RSI", rsi_val))

    stoch_k = raw_other.get("stoch_k")
    if stoch_k is not None:
        components.append(("Stochastic %K", stoch_k))

    adx = raw_other.get("adx")
    plus_di = raw_other.get("plus_di")
    minus_di = raw_other.get("minus_di")
    if adx is not None and plus_di is not None and minus_di is not None:
        direction = 1 if plus_di >= minus_di else -1
        adx_component = 50 + direction * min(50, adx)
        components.append(("ADX/DI", adx_component))

    macd_raw = (macd_result or {}).get("raw", {})
    hist = macd_raw.get("hist")
    if hist is not None:
        # squash the histogram into a 0-100 push around 50 using a soft cap
        pct_price = 0
        cp = (macd_result or {}).get("current_price") or 0
        if cp:
            pct_price = hist / cp * 100
        macd_component = max(0, min(100, 50 + pct_price * 20))
        components.append(("MACD Histogram", macd_component))

    if not components:
        return {"score": 50, "label": "NEUTRAL", "components": []}

    score = sum(v for _, v in components) / len(components)
    score = max(0, min(100, round(score, 1)))

    if score >= 80:
        label = "EXTREME BULLISH"
    elif score >= 60:
        label = "BULLISH"
    elif score > 40:
        label = "NEUTRAL"
    elif score > 20:
        label = "BEARISH"
    else:
        label = "EXTREME BEARISH"

    return {
        "score": score, "label": label,
        "components": [{"name": n, "value": round(v, 1)} for n, v in components],
    }
