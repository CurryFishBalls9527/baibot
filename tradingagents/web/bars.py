"""Bar fetchers — DuckDB readers for each timeframe used by the web UI.

The chan family stores 30m bars in ``intraday_30m_broad.duckdb`` (table
``bars_30m``). chan_daily uses ``market_data.duckdb.daily_bars``. The
intraday_mechanical strategy is 15m and shares ``intraday_30m.duckdb``
keyed by ``interval_minutes``? — no: it uses its own 15m table. We keep
fetchers small and explicit rather than abstracting prematurely.

All readers return ``List[Bar]`` (see ``overlays/base.py``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from .overlays.base import Bar


# Bars in our DuckDB stores (bars_30m, bars_15m, daily_bars) are written as
# naive timestamps in ET wall-clock time. Localizing as UTC would shift the
# unix epoch by 4-5h, which renders the chart's time axis wrong in any non-UTC
# browser. Treat naive intraday timestamps as ET, then convert to UTC for the
# wire. Daily timestamps are conceptually a *date*, not an instant — so we
# strip the time-of-day and emit midnight-UTC (lightweight-charts requires
# uniform 86400s deltas for its daily-resolution detection; emitting 04:00
# UTC bars from naive-ET conversion would break that detection AND mis-align
# markers/overlays whose times may fall anywhere in the trading day).
_MARKET_TZ = ZoneInfo("America/New_York")


def _to_unix(ts) -> int:
    """Intraday-grade unix conversion (treats naive timestamps as ET)."""
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize(_MARKET_TZ)
        return int(ts.timestamp())
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_MARKET_TZ)
        return int(ts.timestamp())
    raise TypeError(f"unsupported timestamp type: {type(ts)}")


def _to_unix_date(ts) -> int:
    """Daily-grade unix conversion: align to midnight UTC of the calendar date.

    Use this for everything that lands on a daily chart — bars, overlay
    line/zone/level endpoints, fill markers, BSP markers — so lightweight-
    charts' daily resolution detection works and markers align to bars.
    """
    if isinstance(ts, (int, float)):
        # Already a unix epoch — round down to the date in UTC.
        d = datetime.fromtimestamp(int(ts), tz=__import__("datetime").timezone.utc).date()
    else:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        if isinstance(ts, pd.Timestamp):
            d = ts.date()
        elif isinstance(ts, datetime):
            d = ts.date()
        else:
            raise TypeError(f"unsupported timestamp type: {type(ts)}")
    # Build a midnight-UTC Timestamp for that date and emit unix.
    return int(pd.Timestamp(d).tz_localize("UTC").timestamp())


def _df_to_bars(df: pd.DataFrame, daily: bool = False) -> List[Bar]:
    convert = _to_unix_date if daily else _to_unix
    bars: List[Bar] = []
    for _, r in df.iterrows():
        bars.append({
            "time":   convert(r["ts"]),
            "open":   float(r["open"]),
            "high":   float(r["high"]),
            "low":    float(r["low"]),
            "close":  float(r["close"]),
            "volume": float(r["volume"]) if r.get("volume") is not None else None,
        })
    return bars


# ── 30m bars (chan, chan_v2) ─────────────────────────────────────────


def fetch_30m(
    symbol: str,
    db_path: str,
    bars_before: int = 80,
    bars_after: int = 20,
    pivot_ts: Optional[str] = None,
) -> List[Bar]:
    """Window of 30m bars around ``pivot_ts``.

    If ``pivot_ts`` is None, returns the most recent ``bars_before``.
    """
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        if pivot_ts is None:
            df = con.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM bars_30m
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [symbol, bars_before],
            ).fetch_df()
            df = df.sort_values("ts").reset_index(drop=True)
        else:
            df_before = con.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM bars_30m
                WHERE symbol = ? AND ts <= ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [symbol, pivot_ts, bars_before],
            ).fetch_df()
            df_after = con.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM bars_30m
                WHERE symbol = ? AND ts > ?
                ORDER BY ts ASC
                LIMIT ?
                """,
                [symbol, pivot_ts, bars_after],
            ).fetch_df()
            df = (
                pd.concat([df_before, df_after], ignore_index=True)
                .drop_duplicates(subset=["ts"])
                .sort_values("ts")
                .reset_index(drop=True)
            )
        return _df_to_bars(df, daily=False)
    finally:
        con.close()


# ── Daily bars (chan_daily) ──────────────────────────────────────────


def fetch_daily(
    symbol: str,
    db_path: str,
    bars_before: int = 250,
    bars_after: int = 20,
    pivot_date: Optional[str] = None,
) -> List[Bar]:
    """Window of daily bars from ``market_data.duckdb.daily_bars``."""
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        if pivot_date is None:
            df = con.execute(
                """
                SELECT trade_date AS ts, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol = ?
                ORDER BY trade_date DESC
                LIMIT ?
                """,
                [symbol, bars_before],
            ).fetch_df()
            df = df.sort_values("ts").reset_index(drop=True)
        else:
            df_before = con.execute(
                """
                SELECT trade_date AS ts, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol = ? AND trade_date <= ?
                ORDER BY trade_date DESC
                LIMIT ?
                """,
                [symbol, pivot_date, bars_before],
            ).fetch_df()
            df_after = con.execute(
                """
                SELECT trade_date AS ts, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol = ? AND trade_date > ?
                ORDER BY trade_date ASC
                LIMIT ?
                """,
                [symbol, pivot_date, bars_after],
            ).fetch_df()
            df = (
                pd.concat([df_before, df_after], ignore_index=True)
                .drop_duplicates(subset=["ts"])
                .sort_values("ts")
                .reset_index(drop=True)
            )
        return _df_to_bars(df, daily=True)
    finally:
        con.close()


# ── 15m bars (intraday_mechanical) ───────────────────────────────────


def fetch_15m(
    symbol: str,
    db_path: str,
    bars_before: int = 80,
    bars_after: int = 30,
    pivot_ts: Optional[str] = None,
    table: str = "bars_15m",
) -> List[Bar]:
    """15m bars around ``pivot_ts``. Falls back gracefully if the 15m
    DuckDB or table doesn't exist on this host."""
    if not Path(db_path).exists():
        return []
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Quick existence check — intraday DBs are sometimes 30m only.
        existing = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()
        if not existing:
            return []
        if pivot_ts is None:
            df = con.execute(
                f"""
                SELECT ts, open, high, low, close, volume
                FROM {table}
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [symbol, bars_before],
            ).fetch_df()
            df = df.sort_values("ts").reset_index(drop=True)
        else:
            df_before = con.execute(
                f"""
                SELECT ts, open, high, low, close, volume
                FROM {table}
                WHERE symbol = ? AND ts <= ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [symbol, pivot_ts, bars_before],
            ).fetch_df()
            df_after = con.execute(
                f"""
                SELECT ts, open, high, low, close, volume
                FROM {table}
                WHERE symbol = ? AND ts > ?
                ORDER BY ts ASC
                LIMIT ?
                """,
                [symbol, pivot_ts, bars_after],
            ).fetch_df()
            df = (
                pd.concat([df_before, df_after], ignore_index=True)
                .drop_duplicates(subset=["ts"])
                .sort_values("ts")
                .reset_index(drop=True)
            )
        return _df_to_bars(df)
    finally:
        con.close()
