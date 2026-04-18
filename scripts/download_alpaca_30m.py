#!/usr/bin/env python3
"""Download 30-minute OHLCV bars from Alpaca into a dedicated DuckDB.

Used for the chan.py spike: measures canonical multi-level buy-point
density on a small universe of liquid US large-caps. Separate from the
daily warehouse so it's easy to discard if the spike kills.

Usage:
    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...
    python scripts/download_alpaca_30m.py \
        --universe research_data/spike_universe.json \
        --start 2023-01-01 --end 2025-12-30 \
        --db research_data/intraday_30m.duckdb
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alpaca_30m")


def get_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
    return StockHistoricalDataClient(key, secret)


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bars_30m (
            symbol VARCHAR,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            trade_count BIGINT,
            vwap DOUBLE,
            source VARCHAR,
            updated_at TIMESTAMP,
            PRIMARY KEY (symbol, ts)
        )
        """
    )


def fetch_symbol(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(30, TimeFrameUnit.Minute),
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()
        df.insert(0, "symbol", symbol)
    return df


def upsert_bars(
    conn: duckdb.DuckDBPyConnection, symbol: str, df: pd.DataFrame
) -> int:
    if df.empty:
        return 0
    now = datetime.utcnow()
    rows = []
    for _, row in df.iterrows():
        ts = row["timestamp"] if "timestamp" in df.columns else row["time"]
        rows.append((
            symbol,
            pd.Timestamp(ts).to_pydatetime(),
            float(row.get("open", 0) or 0),
            float(row.get("high", 0) or 0),
            float(row.get("low", 0) or 0),
            float(row.get("close", 0) or 0),
            float(row.get("volume", 0) or 0),
            int(row.get("trade_count", 0) or 0),
            float(row.get("vwap", 0) or 0),
            "alpaca",
            now,
        ))
    conn.executemany(
        """
        INSERT OR REPLACE INTO bars_30m
        (symbol, ts, open, high, low, close, volume, trade_count, vwap, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Download Alpaca 30-min bars")
    p.add_argument("--universe", default="research_data/spike_universe.json")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-30")
    p.add_argument("--db", default="research_data/intraday_30m.duckdb")
    return p.parse_args()


def main():
    args = parse_args()

    universe_path = Path(args.universe)
    data = json.loads(universe_path.read_text())
    symbols = data["symbols"] if isinstance(data, dict) else data
    log.info("Universe: %d symbols from %s", len(symbols), args.universe)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    client = get_client()

    success = 0
    failed: list[str] = []
    total_bars = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_symbol(client, symbol, start, end)
            n = upsert_bars(conn, symbol, df)
            total_bars += n
            success += 1
            log.info("  [%d/%d] %s: %d bars", i, len(symbols), symbol, n)
        except Exception as e:
            failed.append(symbol)
            log.warning("  [%d/%d] %s FAILED: %s", i, len(symbols), symbol, e)
        time.sleep(0.1)

    log.info("=" * 60)
    log.info("Success: %d/%d symbols, %d total bars", success, len(symbols), total_bars)
    if failed:
        log.warning("Failed: %s", ", ".join(failed))

    counts = conn.execute(
        "SELECT symbol, COUNT(*) AS n FROM bars_30m GROUP BY symbol ORDER BY symbol"
    ).fetchall()
    log.info("Warehouse now contains:")
    for sym, n in counts:
        log.info("  %-6s %d bars", sym, n)

    conn.close()


if __name__ == "__main__":
    main()
