"""Alpaca Market Data API integration for real-time and historical data."""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> StockHistoricalDataClient:
    global _client
    if _client is None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        _client = StockHistoricalDataClient(api_key, secret_key)
    return _client


def get_alpaca_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Get OHLCV daily bars from Alpaca. Drop-in replacement for yfinance."""
    try:
        client = _get_client()
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return f"No data available for {symbol} from {start_date} to {end_date}"

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        df.index = df.index.strftime("%Y-%m-%d")
        return df.to_string()

    except Exception as e:
        logger.error(f"Alpaca stock data error for {symbol}: {e}")
        return f"Error fetching data for {symbol}: {e}"


def get_alpaca_latest_price(symbol: str) -> float:
    """Get the latest mid-price for a symbol."""
    client = _get_client()
    quotes = client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    )
    q = quotes[symbol]
    mid = (float(q.ask_price) + float(q.bid_price)) / 2
    return mid if mid > 0 else float(q.ask_price or q.bid_price)


def get_alpaca_intraday_bars(symbol: str, timeframe_minutes: int = 5,
                              lookback_hours: int = 8) -> str:
    """Get intraday bars for day-trading analysis."""
    try:
        client = _get_client()
        end = datetime.now()
        start = end - timedelta(hours=lookback_hours)

        tf_map = {
            1: TimeFrame.Minute,
            5: TimeFrame(5, "Min"),
            15: TimeFrame(15, "Min"),
            30: TimeFrame(30, "Min"),
            60: TimeFrame.Hour,
        }
        tf = tf_map.get(timeframe_minutes, TimeFrame(5, "Min"))

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
        )
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return f"No intraday data for {symbol}"

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        return df.to_string()

    except Exception as e:
        logger.error(f"Alpaca intraday data error for {symbol}: {e}")
        return f"Error: {e}"


def get_alpaca_snapshot(symbol: str) -> str:
    """Get real-time snapshot: latest trade, quote, minute/daily bar."""
    try:
        client = _get_client()
        snapshots = client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbol)
        )
        snap = snapshots[symbol]

        lines = [f"=== {symbol} Snapshot ==="]
        if snap.latest_trade:
            lines.append(f"Latest Trade: ${snap.latest_trade.price:.2f} (size: {snap.latest_trade.size})")
        if snap.latest_quote:
            lines.append(f"Bid: ${snap.latest_quote.bid_price:.2f} x {snap.latest_quote.bid_size}")
            lines.append(f"Ask: ${snap.latest_quote.ask_price:.2f} x {snap.latest_quote.ask_size}")
        if snap.daily_bar:
            b = snap.daily_bar
            lines.append(f"Today: O={b.open:.2f} H={b.high:.2f} L={b.low:.2f} C={b.close:.2f} V={b.volume}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Alpaca snapshot error for {symbol}: {e}")
        return f"Error: {e}"
