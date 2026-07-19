# Stockex

A modern **Flask-powered stock analysis dashboard** that combines multiple technical indicators into a clean, interactive web interface. Stockex provides dynamic TradingView-style charts, Smart Money Concepts (SMC) tools, momentum analysis, and a composite technical rating—all while preserving each indicator's original decision logic.

---

## Screenshots

### Main Dashboard
![Main Dashboard](stock-analyzer-web/assets/Main_Dashboard.png)

### AAPL First Look
![AAPL First Look](stock-analyzer-web/assets/AAPL_First_Look.png)

### AAPL Scroll Look
![AAPL Scroll Look](stock-analyzer-web/assets/AAPL_Scroll_Look.png)

---

## Features

### Interactive Dashboard

* Modern light-themed single-page web application
* Live, pannable and zoomable candlestick charts powered by **Lightweight Charts**
* Fast ticker search
* Responsive layout
* Top Gainers and Top Losers market overview

### Technical Analysis

Stockex supports multiple technical indicators and Smart Money Concepts:

* Order Blocks
* Fair Value Gaps (FVG)
* RSI Divergence
* MACD Divergence
* Bollinger Bands
* Ichimoku Cloud
* Support & Resistance

Each indicator can be enabled or disabled independently from the dashboard.

### Dynamic Charts

* TradingView-style candlestick charts
* Smooth pan and zoom
* Indicator overlays
* Synced oscillator panels for RSI and MACD
* Divergence markers displayed directly on price candles

### Momentum Gauge

A 0–100 momentum score calculated from:

* RSI
* Stochastic
* ADX / DI Direction
* MACD Histogram

The result is displayed as an easy-to-read momentum gauge.

### Composite Technical Rating

Stockex combines signals from all available indicators into a single weighted score ranging from **-100 to +100**.

| Score         | Rating      |
| ------------- | ----------- |
| ≥ 60          | Strong Buy  |
| 20 to 59.9    | Buy         |
| -19.9 to 19.9 | Neutral     |
| -59.9 to -20  | Sell        |
| ≤ -60         | Strong Sell |

The rating also includes a breakdown showing how much each indicator contributed to the final score.

---

# Project Structure

```
Stockex/
│
├── main.py
├── requirements.txt
├── scoring.py
├── series_utils.py
├── backtester.py
├── data_integrity.py
├── liquidity_fvg.py
├── mtf_confluence.py
│
├── indicators/
│   ├── rsi.py
│   ├── macd.py
│   ├── bollinger_bands.py
│   ├── ichimoku.py
│   ├── support_resistance.py
│   ├── order_blocks.py
│   └── other_indicators.py
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── templates/
│   └── index.html
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stockex.git
cd stockex
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

Open your browser and visit:

```
http://localhost:5000
```

---

# Dashboard

## Home Page

* Live NASDAQ Composite (^IXIC) chart
* Search any stock ticker
* Top Gainers
* Top Losers

## Stock Dashboard

Searching for a ticker opens the complete analysis dashboard featuring:

* Interactive candlestick chart
* Indicator overlay toggles
* RSI panel
* MACD panel
* Momentum Gauge
* Composite Technical Rating
* Indicator contribution breakdown

---

# API Endpoints

| Endpoint                | Description                     |
| ----------------------- | ------------------------------- |
| `/api/search`           | Search ticker symbols           |
| `/api/candles`          | Retrieve OHLC candle data       |
| `/api/analyze`          | Run complete stock analysis     |
| `/api/indicator/<name>` | Execute an individual indicator |
| `/api/movers`           | Get top gainers and losers      |
| `/backtest`             | Backtesting endpoint            |
| `/dataintegrity`        | Data integrity validation       |

---

# Indicator Engine

Stockex includes:

* RSI
  
  ### RSI Divergence
![RSI Divergence](stock-analyzer-web/assets/RSI_Divergence.png)

* MACD
  
### MACD Divergence
![MACD Divergence](stock-analyzer-web/assets/MACD_Divergence.png)

* Bollinger Bands
  
  ### Bollinger Bands
![Bollinger Bands](stock-analyzer-web/assets/Bollinger_Bands.png)

* Ichimoku Cloud

  ### Ichimoku Cloud
![Ichimoku Cloud](stock-analyzer-web/assets/Ichimoku_Cloud.png)

* Order Blocks

### Order Blocks
![Order Blocks](stock-analyzer-web/assets/Order_Blocks.png)
  
* Support & Resistance

### Support & Resistance
![Support & Resistance](stock-analyzer-web/assets/Support_Resistance.png)

* Fair Value Gaps
* Divergence Detection

Each indicator exposes both:

* Analysis result
* Chart-ready JSON series

The frontend renders all visualizations dynamically using Lightweight Charts.

---

# Composite Scoring

Each indicator contributes to the final score using:

```
Signal × Confidence × Weight
```

Indicator weights:

| Indicator            | Weight |
| -------------------- | ------ |
| RSI                  | 1.5    |
| MACD                 | 1.3    |
| Ichimoku             | 1.2    |
| Order Blocks         | 1.4    |
| Bollinger Bands      | 1.0    |
| Support & Resistance | 1.0    |
| Other Indicators     | 0.8    |

The weighted average is converted into a score between **-100** and **100**.

---

# Performance

* Lightweight frontend
* Dynamic client-side chart rendering
* No static matplotlib images required
* Fast API responses
* Modular indicator architecture
* Easily extensible

---

# Technology Stack

### Backend

* Flask
* Python
* yfinance
* Pandas
* NumPy

### Frontend

* HTML5
* CSS3
* JavaScript
* Lightweight Charts

---

# Notes

* Indicator logic, formulas, thresholds, and BUY/SELL decisions remain unchanged.
* PNG chart generation is still available by calling indicator functions with `save_chart=True`.
* The web dashboard uses JSON data and renders all charts client-side for better performance.
* The Top Gainers and Top Losers panel scans a predefined universe of liquid NASDAQ stocks for fast loading.

---

# Future Improvements

* Multiple watchlists
* Portfolio tracking
* Dark mode
* News sentiment analysis
* AI-assisted trade insights
* WebSocket live price updates
* Cryptocurrency support
* Forex support
* Multi-timeframe comparison

---

# License

This project is intended for educational and research purposes.

Always perform your own analysis before making investment decisions.

---

**Stockex** — Smart Technical Analysis for Modern Traders.
