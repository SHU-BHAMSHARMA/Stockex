# SMC Analyzer — Web Dashboard

A Flask backend (your existing indicator engines, untouched logic) + a
light-themed, single-page frontend with dynamic (pan/zoom) charts built on
[Lightweight Charts](https://tradingview.github.io/lightweight-charts/).

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Then open **http://localhost:5000**.

## What you get

- **Home screen**: a live, pannable/zoomable NASDAQ Composite (`^IXIC`) candlestick
  chart, a search bar, and a Top Gainers / Top Losers rail.
- **Search a ticker** → full dashboard:
  - Big dynamic candlestick chart in the middle.
  - A **toggle bar** under the chart to switch what's overlaid: **Order Blocks**,
    **FVG**, **RSI Divergence**, **MACD Divergence**, **Bollinger Bands**,
    **Ichimoku Cloud**, **Support/Resistance**. RSI/MACD toggles also open a
    synced oscillator panel below the price chart, with divergence markers
    plotted directly on the price candles.
  - A **Momentum Gauge** (RSI + Stochastic + ADX/DI direction + MACD histogram,
    averaged into one 0–100 needle).
  - A **Final Verdict** panel: a weighted composite score from **-100 to +100**,
    banded into the standard 5-tier technical-rating scale (Strong Sell → Sell →
    Neutral → Buy → Strong Buy), with a per-indicator contribution breakdown.

## What changed in your files, and what didn't

**Nothing about how any indicator decides BUY/SELL/confidence was changed.**
Every file's decision logic, thresholds, and formulas are untouched.

What *was* changed, purely additive:

1. Each indicator file (`rsi5.py`, `macd.py`, `bollingers_2.py`, `ichimoku_2.py`,
   `other.py`, `support_resistance_2.py`, `order_blocks.py`) now also returns a
   `"series"` block in its result dict — the same arrays it already computed
   (RSI values, MACD/signal/hist, Bollinger bands, Ichimoku lines, OHLC, swing
   points, divergence bars, etc.), reshaped into JSON-safe lists via the new
   `series_utils.py` helper. This is what the frontend charts read; nothing
   about the signal/verdict calculation itself was touched.
2. `main.py` was rewritten as a slightly larger API (`/api/candles`,
   `/api/search`, `/api/movers`, `/api/analyze`, `/api/indicator/<name>`,
   plus the original `/backtest` and `/dataintegrity` endpoints) and now calls
   every indicator with `save_chart=False` by default, since the frontend
   draws its own dynamic charts and the matplotlib PNGs are no longer needed.
   The PNG-drawing code itself is still there in each file (untouched) behind
   its original `if save_chart:` gate — it simply isn't invoked by the API
   anymore. You can still call any `analyze_*()` function with
   `save_chart=True` yourself (e.g. from the `__main__` block of each file) to
   get the original PNGs.
3. New files, all additive: `series_utils.py` (array→JSON helper) and
   `scoring.py` (the composite rating + momentum gauge — pure aggregation of
   each indicator's already-computed `signal`/`confidence`, doesn't re-derive
   or override any individual indicator's own verdict).
4. `backtester.py`, `data_integrity.py`, `liquidity_fvg.py`, `mtf_confluence.py`
   are copied over completely unchanged.

## Composite scoring — the "international system"

This mirrors the widely-used technical-rating convention popularized by
TradingView's "Technical Rating" widget: map each indicator's signal to
-1 (SELL) / 0 (NEUTRAL) / +1 (BUY), scale by that indicator's own confidence
(0–1), scale again by a fixed importance weight, average across all
indicators that returned data, then multiply by 100 for a **-100..+100**
composite rating:

| Rating range      | Label        |
|--------------------|-------------|
| ≥ 60               | STRONG BUY  |
| 20 to 59.9         | BUY         |
| -19.9 to 19.9       | NEUTRAL     |
| -59.9 to -20        | SELL        |
| ≤ -60               | STRONG SELL |

Indicator weights (unchanged from your original `main.py`):
RSI 1.5, MACD 1.3, Ichimoku 1.2, Order Block 1.4, Bollinger 1.0, S/R 1.0,
Other 0.8.

## Notes / limitations

- The "Top Gainers/Losers" rail scans a fixed, liquid slice of large-cap
  NASDAQ names (`MOVERS_UNIVERSE` in `main.py`) rather than the full index
  live, so it loads fast. Swap in your own list/API if you need the full
  index.
- `/api/search` uses `yfinance`'s search endpoint when available and falls
  back to treating your query as a literal ticker.
- Intraday intervals (`1h`) keep full timestamps (not just the date) so bars
  on the same calendar day don't collide on the chart's time axis.
