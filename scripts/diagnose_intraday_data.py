#!/usr/bin/env python3
"""Inspect intraday DuckDB quality for research-only mechanical experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose intraday DuckDB quality")
    p.add_argument("--db", required=True, help="Path to intraday DuckDB")
    p.add_argument("--interval", type=int, choices=[5, 15, 30], default=30)
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    table_name = f"bars_{args.interval}m"
    conn = duckdb.connect(args.db, read_only=True)
    try:
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = [
                row[0]
                for row in conn.execute(
                    f"""
                    SELECT symbol
                    FROM {table_name}
                    GROUP BY symbol
                    ORDER BY COUNT(*) DESC, symbol
                    LIMIT ?
                    """,
                    [args.limit],
                ).fetchall()
            ]

        rows = []
        for symbol in symbols:
            metrics = conn.execute(
                f"""
                WITH base AS (
                    SELECT ts, open, high, low, close, volume
                    FROM {table_name}
                    WHERE symbol = ?
                ),
                session_stats AS (
                    SELECT
                        CAST(ts AS DATE) AS session_date,
                        MIN(CAST(ts AS TIME)) AS first_bar_time,
                        MAX(CAST(ts AS TIME)) AS last_bar_time,
                        COUNT(*) AS bars_in_session
                    FROM base
                    GROUP BY 1
                )
                SELECT
                    ? AS symbol,
                    COUNT(*) AS bar_count,
                    COUNT(DISTINCT CAST(ts AS DATE)) AS sessions,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts,
                    SUM(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 1 ELSE 0 END) AS null_rows,
                    COUNT(*) - COUNT(DISTINCT ts) AS duplicate_ts,
                    MIN(session_stats.first_bar_time) AS min_first_bar_time,
                    MAX(session_stats.first_bar_time) AS max_first_bar_time,
                    MIN(session_stats.last_bar_time) AS min_last_bar_time,
                    MAX(session_stats.last_bar_time) AS max_last_bar_time,
                    MIN(session_stats.bars_in_session) AS min_bars_in_session,
                    MAX(session_stats.bars_in_session) AS max_bars_in_session,
                    AVG(session_stats.bars_in_session) AS avg_bars_in_session
                FROM base
                LEFT JOIN session_stats ON CAST(base.ts AS DATE) = session_stats.session_date
                """,
                [symbol, symbol],
            ).fetchone()
            rows.append(
                {
                    "symbol": metrics[0],
                    "bar_count": int(metrics[1]),
                    "sessions": int(metrics[2]),
                    "first_ts": str(metrics[3]) if metrics[3] is not None else None,
                    "last_ts": str(metrics[4]) if metrics[4] is not None else None,
                    "null_rows": int(metrics[5]),
                    "duplicate_ts": int(metrics[6]),
                    "min_first_bar_time": str(metrics[7]) if metrics[7] is not None else None,
                    "max_first_bar_time": str(metrics[8]) if metrics[8] is not None else None,
                    "min_last_bar_time": str(metrics[9]) if metrics[9] is not None else None,
                    "max_last_bar_time": str(metrics[10]) if metrics[10] is not None else None,
                    "min_bars_in_session": int(metrics[11]) if metrics[11] is not None else 0,
                    "max_bars_in_session": int(metrics[12]) if metrics[12] is not None else 0,
                    "avg_bars_in_session": round(float(metrics[13]), 2) if metrics[13] is not None else 0.0,
                }
            )

        payload = {
            "db": args.db,
            "interval_minutes": args.interval,
            "table_name": table_name,
            "symbols": rows,
        }
    finally:
        conn.close()

    print()
    print("=" * 96)
    print("  INTRADAY DATA DIAGNOSTIC")
    print("=" * 96)
    for row in rows:
        print(
            f"  {row['symbol']:<8} bars {row['bar_count']:>7}"
            f"  sessions {row['sessions']:>5}"
            f"  first {row['min_first_bar_time']}"
            f"  last {row['max_last_bar_time']}"
            f"  dup {row['duplicate_ts']:>4}"
            f"  null {row['null_rows']:>4}"
            f"  avg/session {row['avg_bars_in_session']:>5.1f}"
        )
    print("=" * 96)

    out_path = (
        Path(args.out)
        if args.out
        else Path("results/intraday_breakout/diagnostics") / f"{Path(args.db).stem}_{table_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
