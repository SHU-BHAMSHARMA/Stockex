# main.py
"""
Main entry point for the multi-indicator stock analyzer.
Serves the dashboard (templates/index.html + static/) and a small JSON API.

IMPORTANT: none of the individual indicator files (rsi5.py, macd.py,
bollingers_2.py, ichimoku_2.py, order_blocks.py, support_resistance_2.py,
other.py) had their analysis/decision logic changed. Each one only gained
an additive "series" block in its return dict (raw arrays for charting)
and is now called with save_chart=False by default from here, since the
frontend renders its own dynamic (pan/zoom) charts instead of the
matplotlib PNGs those files can still optionally generate.
"""
import math
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
import yfinance as yf

# ── Import all indicator modules (logic untouched) ────────────
from rsi5 import analyze_rsi
from bollingers_2 import analyze_bollinger
from ichimoku_2 import analyze_ichimoku
from macd import analyze_macd
from other import analyze_other
from support_resistance_2 import analyze_sr
from scoring import composite_rating, momentum_gauge

try:
    from order_blocks import analyze_order_blocks
    HAS_OB = True
except ImportError:
    print("⚠️  order_blocks.py not found — order-block analysis disabled")
    HAS_OB = False
    def analyze_order_blocks(*args, **kwargs):
        return {"error": "order_blocks.py not available"}

try:
    from backtester import run_backtest
except ImportError:
    run_backtest = None
try:
    from data_integrity import check_data_integrity
except ImportError:
    check_data_integrity = None


# ── numpy-safe JSON so raw numpy types from the indicator modules
#    never crash jsonify() ──────────────────────────────────────
class NumpySafeJSONProvider(DefaultJSONProvider):
    @staticmethod
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return DefaultJSONProvider.default(o)


app = Flask(__name__)
app.json = NumpySafeJSONProvider(app)

# A compact, liquid slice of the NASDAQ-100 used for the "top movers" rail.
# (Scanning the full index live on every request would be slow; this list
# gives a representative, fast-loading snapshot.)
MOVERS_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
    "COST", "ADBE", "NFLX", "AMD", "CSCO", "INTC", "QCOM", "TXN", "AMGN",
    "INTU", "HON", "SBUX", "BKNG", "GILD", "MDLZ", "ADI", "PYPL", "REGN",
    "VRTX", "PANW", "CRWD",
]

INDICATOR_FUNCS = {
    "rsi": analyze_rsi,
    "bollinger": analyze_bollinger,
    "ichimoku": analyze_ichimoku,
    "macd": analyze_macd,
    "other": analyze_other,
    "sr": analyze_sr,
    "orderblock": analyze_order_blocks,
    "ob": analyze_order_blocks,
}


def analyze_all(ticker, period='1y', interval='1d', ob_filters=None):
    """Run all indicators (no PNGs) and return combined verdict + scoring."""
    ob_filters = ob_filters or {}
    results = {}
    for key, fn in [("RSI", analyze_rsi), ("Bollinger", analyze_bollinger),
                     ("Ichimoku", analyze_ichimoku), ("MACD", analyze_macd),
                     ("Other", analyze_other), ("SR", analyze_sr)]:
        try:
            results[key] = fn(ticker, period, interval, save_chart=False)
        except Exception as e:
            results[key] = {"error": str(e)}

    if HAS_OB:
        try:
            results["OrderBlock"] = analyze_order_blocks(ticker, period, interval, save_chart=False, **ob_filters)
        except Exception as e:
            results["OrderBlock"] = {"error": str(e)}
    else:
        results["OrderBlock"] = {"error": "order_blocks.py not found"}

    composite = composite_rating(results)
    gauge = momentum_gauge(results.get("Other"), results.get("RSI"), results.get("MACD"))

    levels = None
    if "SR" in results and "error" not in results["SR"]:
        sr = results["SR"]
        levels = {
            "support": sr.get("support_zones", []),
            "resistance": sr.get("resistance_zones", []),
            "stop_loss": sr.get("stop_loss"),
            "take_profit": sr.get("take_profit"),
        }
    elif "OrderBlock" in results and "error" not in results["OrderBlock"] and results["OrderBlock"].get("entry_plan"):
        ob = results["OrderBlock"]
        plan = ob["entry_plan"]
        levels = {
            "support": [{"low": plan["zone_low"], "high": plan["zone_high"],
                         "mid": (plan["zone_low"] + plan["zone_high"]) / 2,
                         "strength": plan["confidence_score"]}],
            "resistance": [],
            "stop_loss": plan["stop_loss"],
            "take_profit": plan["take_profit"],
        }

    current_price = (results.get("RSI", {}).get("current_price") or
                      results.get("Bollinger", {}).get("current_price") or 0)

    return {
        "ticker": ticker,
        "current_price": current_price,
        "composite": composite,
        "momentum_gauge": gauge,
        "indicators": results,
        "levels": levels,
    }


def _bool_arg(name, default):
    val = request.args.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Pages ──────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ── Core JSON API ──────────────────────────────────────────────
@app.route("/api/candles", methods=["GET"])
def api_candles():
    """OHLCV series for the dynamic candlestick chart (main NASDAQ chart
    or any searched ticker)."""
    ticker = request.args.get("ticker", "^IXIC").strip().upper()
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or len(df) == 0:
            return jsonify({"error": f"No data for {ticker}"}), 404
        if hasattr(df.columns, "get_level_values") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        payload = {
            "ticker": ticker,
            "dates": [str(d) for d in df.index],
            "open": [round(float(x), 4) for x in df["Open"]],
            "high": [round(float(x), 4) for x in df["High"]],
            "low": [round(float(x), 4) for x in df["Low"]],
            "close": [round(float(x), 4) for x in df["Close"]],
            "volume": [int(x) for x in df["Volume"]] if "Volume" in df.columns else [],
        }
        last_close = payload["close"][-1]
        prev_close = payload["close"][-2] if len(payload["close"]) > 1 else last_close
        payload["current_price"] = last_close
        payload["change"] = round(last_close - prev_close, 4)
        payload["change_pct"] = round((last_close - prev_close) / prev_close * 100, 3) if prev_close else 0
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def api_search():
    """Lightweight ticker lookup/validation for the search bar."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})
    try:
        candidates = []
        seen = set()
        try:
            search = yf.Search(q, max_results=8)
            for quote in getattr(search, "quotes", []) or []:
                sym = quote.get("symbol")
                if sym and sym not in seen:
                    seen.add(sym)
                    candidates.append({
                        "symbol": sym,
                        "name": quote.get("shortname") or quote.get("longname") or sym,
                        "exchange": quote.get("exchange", ""),
                    })
        except Exception:
            pass
        if not candidates:
            # fall back to treating the query itself as a ticker
            candidates = [{"symbol": q.upper(), "name": q.upper(), "exchange": ""}]
        return jsonify({"results": candidates})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/movers", methods=["GET"])
def api_movers():
    """Top gainers / losers rail, computed from MOVERS_UNIVERSE."""
    try:
        data = yf.download(MOVERS_UNIVERSE, period="5d", interval="1d",
                            progress=False, group_by="ticker")
        moves = []
        for tkr in MOVERS_UNIVERSE:
            try:
                closes = data[tkr]["Close"].dropna()
                if len(closes) < 2:
                    continue
                last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
                pct = (last - prev) / prev * 100 if prev else 0
                moves.append({"ticker": tkr, "price": round(last, 2), "change_pct": round(pct, 2)})
            except Exception:
                continue
        moves.sort(key=lambda m: m["change_pct"], reverse=True)
        return jsonify({
            "gainers": moves[:8],
            "losers": sorted(moves, key=lambda m: m["change_pct"])[:8],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["GET"])
@app.route("/api/analyze", methods=["GET"])
def api_analyze():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Missing ticker parameter"}), 400
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")

    ob_filters = {
        "use_mtf_filter": _bool_arg("mtf", True),
        "require_liquidity_sweep": _bool_arg("sweep", True),
        "require_fvg_confluence": _bool_arg("fvg", False),
        "require_mtf_alignment": _bool_arg("mtf_align", False),
    }
    result = analyze_all(ticker, period, interval, ob_filters=ob_filters)
    return jsonify(result)


@app.route("/indicator/<indicator>", methods=["GET"])
@app.route("/api/indicator/<indicator>", methods=["GET"])
def api_indicator(indicator):
    """Single-indicator payload (used to populate/toggle the indicator
    chart panel without re-running everything else)."""
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Missing ticker parameter"}), 400
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")

    key = indicator.lower()
    fn = INDICATOR_FUNCS.get(key)
    if fn is None or (key in ("orderblock", "ob") and not HAS_OB):
        return jsonify({"error": f"Unknown or disabled indicator: {indicator}"}), 400
    try:
        if key in ("orderblock", "ob"):
            res = fn(
                ticker, period, interval, save_chart=False,
                use_mtf_filter=_bool_arg("mtf", True),
                require_liquidity_sweep=_bool_arg("sweep", True),
                require_fvg_confluence=_bool_arg("fvg", False),
                require_mtf_alignment=_bool_arg("mtf_align", False),
            )
        else:
            res = fn(ticker, period, interval, save_chart=False)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Optional backtest and data integrity endpoints ──────────
if run_backtest is not None:
    @app.route("/backtest/orderblocks", methods=["GET"])
    @app.route("/api/backtest/orderblocks", methods=["GET"])
    def backtest_orderblocks():
        ticker = request.args.get("ticker", "").strip().upper()
        if not ticker:
            return jsonify({"error": "Missing ticker parameter"}), 400
        period = request.args.get("period", "2y")
        interval = request.args.get("interval", "1d")
        try:
            slippage_pct = float(request.args.get("slippage_pct", 0.0015))
            min_bars = int(request.args.get("min_bars", 60))
            max_bars = int(request.args.get("max_bars", 1500))
        except ValueError:
            return jsonify({"error": "slippage_pct/min_bars/max_bars must be numeric"}), 400
        try:
            report = run_backtest(
                ticker, period=period, interval=interval,
                require_liquidity_sweep=_bool_arg("sweep", True),
                require_fvg_confluence=_bool_arg("fvg", False),
                require_mtf_alignment=_bool_arg("mtf_align", False),
                slippage_pct=slippage_pct, min_bars=min_bars, max_bars=max_bars,
            )
            return jsonify(report)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if check_data_integrity is not None:
    @app.route("/dataintegrity", methods=["GET"])
    @app.route("/api/dataintegrity", methods=["GET"])
    def data_integrity_route():
        ticker = request.args.get("ticker", "").strip().upper()
        if not ticker:
            return jsonify({"error": "Missing ticker parameter"}), 400
        period = request.args.get("period", "1y")
        interval = request.args.get("interval", "1d")
        try:
            report = check_data_integrity(ticker, period=period, interval=interval,
                                           cross_check=_bool_arg("cross_check", True))
            return jsonify(report)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
