"""Trade Review page — interactive chart visualization for all strategy variants."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tradingagents.dashboard.chart_builder import TradeChartBuilder
from tradingagents.storage.database import TradingDatabase

VARIANT_CONFIG = {
    "chan": {
        "db_path": "trading_chan.db",
        "ohlcv_db": "research_data/intraday_30m_broad.duckdb",
        "ohlcv_table": "bars_30m",
        "ts_col": "ts",
        "timeframe": "30m",
        "default_overlays": ["chan", "macd"],
    },
    "mechanical": {
        "db_path": "trading_mechanical.db",
        "ohlcv_db": "research_data/market_data.duckdb",
        "ohlcv_table": "daily_bars",
        "ts_col": "trade_date",
        "timeframe": "daily",
        "default_overlays": ["sma", "volume"],
    },
    "llm": {
        "db_path": "trading_llm.db",
        "ohlcv_db": "research_data/market_data.duckdb",
        "ohlcv_table": "daily_bars",
        "ts_col": "trade_date",
        "timeframe": "daily",
        "default_overlays": ["sma", "volume"],
    },
}


def _load_ohlcv(db_path: str, table: str, ts_col: str, symbol: str,
                begin: str, end: str) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute(
        f"SELECT {ts_col} as ts, open, high, low, close, volume "
        f"FROM {table} WHERE symbol = ? AND {ts_col} >= ? AND {ts_col} <= ? "
        f"ORDER BY {ts_col}",
        [symbol, begin, end],
    ).fetchdf()
    conn.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    return df


def _load_trades(db: TradingDatabase, limit: int = 100) -> pd.DataFrame:
    trades = db.get_recent_trades(limit=limit)
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    # Mixed timestamp formats across live-insert and reconciler paths —
    # see pages/1_Performance.py for the full story.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="mixed", errors="coerce", utc=True,
    )
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp", ascending=False)


def _get_signal_reasoning(db: TradingDatabase, signal_id) -> str:
    if not signal_id:
        return ""
    try:
        row = db.conn.execute(
            "SELECT reasoning FROM signals WHERE id = ?", [signal_id]
        ).fetchone()
        return row[0] if row else ""
    except Exception:
        return ""


def main():
    st.set_page_config(page_title="Trade Review", layout="wide")
    st.title("Trade Review")

    with st.sidebar:
        variant = st.selectbox("Strategy Variant", list(VARIANT_CONFIG.keys()))
        vcfg = VARIANT_CONFIG[variant]

        trade_db_path = vcfg["db_path"]
        if not Path(trade_db_path).exists():
            st.warning(f"Trade DB not found: {trade_db_path}")
            st.info("No trades to review yet. Run the strategy first.")
            return

        db = TradingDatabase(trade_db_path)
        trades_df = _load_trades(db)

        if trades_df.empty:
            st.info("No trades found.")
            return

        st.subheader("Overlay Options")
        show_volume = st.checkbox("Volume", value=True)
        show_sma = st.checkbox("SMA (50/150/200)", value="sma" in vcfg["default_overlays"])
        show_bollinger = st.checkbox("Bollinger Bands", value=False)
        show_macd = st.checkbox("MACD", value="macd" in vcfg["default_overlays"])
        show_rsi = st.checkbox("RSI", value=False)
        show_atr = st.checkbox("ATR", value=False)
        show_chan = st.checkbox(
            "Chan Structures (笔/中枢/买卖点)",
            value="chan" in vcfg["default_overlays"],
            disabled=(variant != "chan"),
        )
        context_days = st.slider("Chart context (days)", 7, 90, 30)

    symbols_traded = sorted(trades_df["symbol"].unique())
    selected_symbol = st.selectbox("Symbol", symbols_traded)

    symbol_trades = trades_df[trades_df["symbol"] == selected_symbol].copy()

    st.subheader(f"Trades — {selected_symbol}")
    display_cols = ["timestamp", "side", "qty", "filled_price", "status", "reasoning"]
    available_cols = [c for c in display_cols if c in symbol_trades.columns]
    st.dataframe(
        symbol_trades[available_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    if symbol_trades.empty:
        return

    trade_options = []
    for _, row in symbol_trades.iterrows():
        ts = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        side = row.get("side", "?").upper()
        price = row.get("filled_price", 0) or 0
        trade_options.append(f"{ts} | {side} @ ${price:.2f}")

    selected_idx = st.selectbox("Select trade to chart", range(len(trade_options)),
                                format_func=lambda i: trade_options[i])
    selected_trade = symbol_trades.iloc[selected_idx]

    trade_time = selected_trade["timestamp"]
    begin = (trade_time - timedelta(days=context_days)).strftime("%Y-%m-%d")
    end = (trade_time + timedelta(days=5)).strftime("%Y-%m-%d")

    ohlcv = _load_ohlcv(
        vcfg["ohlcv_db"], vcfg["ohlcv_table"], vcfg["ts_col"],
        selected_symbol, begin, end,
    )

    if ohlcv.empty:
        st.warning(f"No OHLCV data found for {selected_symbol} in {vcfg['ohlcv_db']}")
        return

    builder = TradeChartBuilder(selected_symbol, ohlcv).add_candlesticks()

    if show_volume:
        builder.add_volume()
    if show_sma:
        builder.add_sma([50, 150, 200])
    if show_bollinger:
        builder.add_bollinger()
    if show_macd:
        builder.add_macd()
    if show_rsi:
        builder.add_rsi()
    if show_atr:
        builder.add_atr()

    trade_markers = []
    for _, row in symbol_trades.iterrows():
        reasoning = row.get("reasoning", "") or _get_signal_reasoning(
            db, row.get("signal_id")
        )
        trade_markers.append({
            "timestamp": row["timestamp"],
            "side": row.get("side", ""),
            "filled_price": row.get("filled_price", 0),
            "qty": row.get("qty") or row.get("filled_qty", ""),
            "reasoning": reasoning,
        })
    builder.add_trade_markers(trade_markers)

    if show_chan and variant == "chan":
        # Pre-flight: Chan's CChan constructor raises cryptic errors when the
        # symbol has no 30m bars in the window, or when the window is too
        # narrow for the pivot calculation. Surface a clearer message and
        # log the full traceback so the terminal shows the cause.
        try:
            if ohlcv.empty:
                raise RuntimeError("no 30m OHLCV bars in window")
            # Chan wants at least ~40 bars (Bi + ZS pivots). If the window
            # is too small, the extractor returns degenerate output at best.
            if len(ohlcv) < 20:
                raise RuntimeError(
                    f"only {len(ohlcv)} 30m bars in window "
                    f"(need ~40+ for meaningful Bi/ZS extraction)"
                )
            from tradingagents.dashboard.chan_structures import extract_chan_structures
            structures = extract_chan_structures(
                selected_symbol, begin, end, vcfg["ohlcv_db"],
            )
            builder.add_chan_bi(structures["bi_list"])
            builder.add_chan_zs(structures["zs_list"])
            builder.add_chan_bsp(structures["bsp_list"])
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            is_bare_assert = (
                isinstance(e, AssertionError) and not str(e)
            )
            st.warning(
                f"**Chan structure extraction failed for {selected_symbol}:**\n\n"
                f"`{type(e).__name__}: {e}`\n\n"
                f"Window: {begin} → {end} · bars loaded: {len(ohlcv)}\n\n"
                + (
                    "AssertionError with empty message usually comes from a "
                    "bare `assert` in the Chan library. Often caused by "
                    "in-process state leaking from a prior extraction — "
                    "click **Clear cache + rerun** below, or try a "
                    "different window/symbol.\n\n"
                    if is_bare_assert else
                    "Tip: widen the context window in the sidebar, or confirm "
                    f"this symbol has 30m data in `{vcfg['ohlcv_db']}`.\n\n"
                )
            )
            with st.expander("Full traceback", expanded=False):
                st.code(tb, language="python")
            if st.button("Clear cache + rerun", key=f"chan_retry_{selected_symbol}"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

    fig = builder.build()
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Trade Details"):
        detail = {
            "Symbol": selected_symbol,
            "Side": selected_trade.get("side", ""),
            "Qty": selected_trade.get("qty") or selected_trade.get("filled_qty", ""),
            "Price": f"${selected_trade.get('filled_price', 0) or 0:.2f}",
            "Status": selected_trade.get("status", ""),
            "Time": str(selected_trade["timestamp"]),
            "Reasoning": selected_trade.get("reasoning", ""),
        }
        for k, v in detail.items():
            st.write(f"**{k}:** {v}")


if __name__ == "__main__":
    main()
