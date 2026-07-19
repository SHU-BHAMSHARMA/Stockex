"""
Walk-Forward Backtester  ·  companion module to order_blocks.py
─────────────────────────────────────────────────────────────────
Runs the exact same analysis the live API would run, using only the data
available up to each bar (no lookahead in the decision step), and — when
that produces a fresh, actionable entry ("price inside zone" today, on an
order block that hasn't already been traded) — opens a virtual trade at
the entry_plan's price and walks forward on the real subsequent bars until
stop-loss, take-profit, or the end of data.

This is the empirical confirmation layer: rather than trusting that a new
filter (multi-timeframe alignment, liquidity-sweep requirement, FVG
confluence) "should" help, this reports the actual win rate / expectancy
those filters produce, so filters can be compared to each other and to a
no-filter baseline on the same ticker/history before anyone trades on them.

Importable:
    from backtester import run_backtest
    report = run_backtest("NABIL.NP", period="2y", interval="1d")

Also importable directly for feeding other indicators (RSI, MACD, etc.)
into the same harness later — see `_summarize` / trade-record shape below,
which is intentionally indicator-agnostic (entry/stop/target/outcome/pnl).
"""
import numpy as np
import pandas as pd
import yfinance as yf

from order_blocks import _run_analysis

DEFAULT_SLIPPAGE_PCT = 0.0015   # 0.15% adverse slippage on entry/exit fills — tune per market/liquidity
DEFAULT_MIN_BARS = 60           # first bar index the engine is allowed to evaluate from
DEFAULT_MAX_BARS = 1500         # cap on how much history to re-scan (O(n^2)-ish cost otherwise)


def _download(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    return df


def _apply_slippage(price, direction, is_exit, slippage_pct):
    """Adverse slippage only — fills are always slightly worse than the
    theoretical price, in whichever direction actually hurts the trade."""
    if direction == "BUY":
        return price * (1 + slippage_pct) if not is_exit else price * (1 - slippage_pct)
    else:
        return price * (1 - slippage_pct) if not is_exit else price * (1 + slippage_pct)


def _pnl_pct(entry_price, exit_price, direction):
    if direction == "BUY":
        return (exit_price - entry_price) / entry_price * 100
    return (entry_price - exit_price) / entry_price * 100


def run_backtest(ticker, period="2y", interval="1d",
                  require_liquidity_sweep=True,
                  require_fvg_confluence=False,
                  require_mtf_alignment=False,
                  htf_trend=None,
                  slippage_pct=DEFAULT_SLIPPAGE_PCT,
                  min_bars=DEFAULT_MIN_BARS,
                  max_bars=DEFAULT_MAX_BARS):
    """
    NOTE ON MULTI-TIMEFRAME: a live higher-timeframe download at every
    single backtest bar is both prohibitively slow and not point-in-time
    correct (yfinance only ever returns the *current* higher-timeframe
    series, which leaks future bars into the past). This backtester
    therefore isolates the lower-timeframe engine (structure + order
    blocks + liquidity sweep + FVG confluence). Pass a fixed `htf_trend`
    ("up"/"down") only if you specifically want to test how the
    multi-timeframe filter would have behaved assuming that bias held
    steady for the whole test window — treat that mode as illustrative,
    not as a substitute for genuinely point-in-time MTF backtesting.

    Returns a JSON-safe report dict: num_trades, win_rate_pct, avg_win_pct,
    avg_loss_pct, expectancy_pct_per_trade, a breakdown by OB origin
    (BOS vs CHoCH), and the full trade log.
    """
    df = _download(ticker, period, interval)
    n = len(df)
    if n > max_bars:
        df = df.iloc[n - max_bars:].reset_index(drop=True)
        n = len(df)
    if n <= min_bars:
        return {"ticker": ticker, "period": period, "interval": interval, "bars": n,
                "num_trades": 0, "note": "Not enough bars to run a walk-forward test."}

    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)

    trades = []
    traded_ob_bars = set()   # dedupe: don't re-enter the same OB twice
    open_trade = None

    for t in range(min_bars, n):
        if open_trade is not None:
            direction = open_trade["direction"]
            if direction == "BUY":
                hit_sl = l[t] <= open_trade["stop_loss"]
                hit_tp = open_trade["take_profit"] is not None and h[t] >= open_trade["take_profit"]
            else:
                hit_sl = h[t] >= open_trade["stop_loss"]
                hit_tp = open_trade["take_profit"] is not None and l[t] <= open_trade["take_profit"]

            if hit_sl or hit_tp:
                # if both trigger intrabar, treat the stop as hit first (conservative assumption)
                outcome = "SL" if hit_sl else "TP"
                raw_exit = open_trade["stop_loss"] if hit_sl else open_trade["take_profit"]
                exit_price = _apply_slippage(raw_exit, direction, is_exit=True, slippage_pct=slippage_pct)
                pnl_pct = _pnl_pct(open_trade["entry_price"], exit_price, direction)
                open_trade.update(exit_bar=t, exit_date=str(df.loc[t, "Date"])[:10],
                                   exit_price=round(exit_price, 4), outcome=outcome,
                                   pnl_pct=round(pnl_pct, 4))
                trades.append(open_trade)
                open_trade = None
            elif t == n - 1:
                exit_price = c[t]
                pnl_pct = _pnl_pct(open_trade["entry_price"], exit_price, direction)
                open_trade.update(exit_bar=t, exit_date=str(df.loc[t, "Date"])[:10],
                                   exit_price=round(exit_price, 4), outcome="OPEN_AT_END",
                                   pnl_pct=round(pnl_pct, 4))
                trades.append(open_trade)
                open_trade = None
            continue  # one trade open at a time -> don't evaluate new entries mid-trade

        sub_df = df.iloc[:t + 1].reset_index(drop=True)
        try:
            res = _run_analysis(sub_df, ticker, period, interval, save_chart=False,
                                 htf_trend=htf_trend, htf_info=None,
                                 require_liquidity_sweep=require_liquidity_sweep,
                                 require_fvg_confluence=require_fvg_confluence,
                                 require_mtf_alignment=require_mtf_alignment)
        except Exception:
            continue
        if "error" in res or not res.get("entry_plan"):
            continue

        plan = res["entry_plan"]
        if "ENTRY" not in plan["status"]:
            continue  # only trade live triggers, not WATCH-only levels
        ob_key = (plan["direction"], plan["order_block_bar"])
        if ob_key in traded_ob_bars:
            continue
        traded_ob_bars.add(ob_key)

        entry_price = _apply_slippage(plan["entry_price"], plan["direction"], is_exit=False,
                                       slippage_pct=slippage_pct)
        open_trade = {
            "entry_bar": t, "entry_date": str(df.loc[t, "Date"])[:10],
            "direction": plan["direction"], "entry_price": round(entry_price, 4),
            "stop_loss": plan["stop_loss"], "take_profit": plan["take_profit"],
            "origin": plan["born_from"], "confidence_score": plan["confidence_score"],
        }

    return _summarize(trades, ticker, period, interval, n)


def _summarize(trades, ticker, period, interval, n_bars):
    if not trades:
        return {"ticker": ticker, "period": period, "interval": interval, "bars": n_bars,
                "num_trades": 0, "note": "No qualifying entries were triggered over this window."}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    by_origin = {}
    for origin in sorted(set(t["origin"] for t in trades)):
        sub = [t for t in trades if t["origin"] == origin]
        sub_wins = [t for t in sub if t["pnl_pct"] > 0]
        by_origin[origin] = {
            "num_trades": len(sub),
            "win_rate_pct": round(len(sub_wins) / len(sub) * 100, 2),
            "avg_pnl_pct": round(float(np.mean([t["pnl_pct"] for t in sub])), 4),
        }

    return {
        "ticker": ticker, "period": period, "interval": interval, "bars": n_bars,
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "expectancy_pct_per_trade": round(expectancy, 4),
        "cumulative_pnl_pct_sum": round(float(np.sum([t["pnl_pct"] for t in trades])), 4),
        "by_origin": by_origin,
        "trades": trades,
    }


if __name__ == "__main__":
    tkr = input("Enter stock ticker: ").strip().upper()
    rep = run_backtest(tkr, period="2y", interval="1d")
    if rep.get("num_trades", 0) == 0:
        print(rep.get("note", "No trades."))
    else:
        print(f"Trades: {rep['num_trades']}  WinRate: {rep['win_rate_pct']}%  "
              f"Expectancy: {rep['expectancy_pct_per_trade']}%/trade  "
              f"CumPnL: {rep['cumulative_pnl_pct_sum']}%")
        print("By origin:", rep["by_origin"])
