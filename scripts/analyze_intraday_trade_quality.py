#!/usr/bin/env python3
"""Analyze entry/exit timing quality for research-only intraday backtest artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze intraday trade timing quality")
    p.add_argument("--artifact", required=True, help="Path to intraday backtest JSON artifact")
    p.add_argument("--period", default="2023_2025", help="Period key inside period-matrix artifacts")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _extract_trade_payload(payload: dict, period: str) -> tuple[list[dict], dict]:
    if "details" in payload:
        detail = payload["details"][period]
        return detail.get("trades", []), detail.get("metadata", {})
    return payload.get("trades", []), payload.get("metadata", {})


def _load_bar_slice(conn: duckdb.DuckDBPyConnection, table_name: str, symbol: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    df = conn.execute(
        f"""
        SELECT ts, open, high, low, close, volume
        FROM {table_name}
        WHERE symbol = ? AND ts >= ? AND ts <= ?
        ORDER BY ts
        """,
        [symbol, start_ts, end_ts],
    ).fetchdf()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=False)
    return df.set_index("ts").sort_index()


def _analyze_trade(conn: duckdb.DuckDBPyConnection, table_name: str, trade: dict) -> dict:
    entry_ts = pd.Timestamp(trade["entry_time"])
    exit_ts = pd.Timestamp(trade["exit_time"])
    bars = _load_bar_slice(conn, table_name, trade["symbol"], trade["entry_time"], trade["exit_time"])
    if bars.empty:
        return {
            **trade,
            "mfe_pct": None,
            "mae_pct": None,
            "close_vs_exit_pct": None,
            "entry_bar_hour": entry_ts.hour,
            "entry_bar_minute": entry_ts.minute,
            "entry_bar_in_session_est": None,
        }

    entry_price = float(trade["entry_price"])
    max_high = float(bars["high"].max())
    min_low = float(bars["low"].min())
    final_close = float(bars["close"].iloc[-1])
    bars_held = int(len(bars))
    # 08:30 local session open for current research warehouse convention.
    session_open = pd.Timestamp(entry_ts.date()).replace(hour=8, minute=30)
    entry_bar_in_session_est = int((entry_ts - session_open).total_seconds() // (30 * 60))
    return {
        **trade,
        "mfe_pct": round((max_high / entry_price) - 1.0, 4),
        "mae_pct": round((min_low / entry_price) - 1.0, 4),
        "close_vs_exit_pct": round((final_close / float(trade["exit_price"])) - 1.0, 4),
        "entry_bar_hour": entry_ts.hour,
        "entry_bar_minute": entry_ts.minute,
        "entry_bar_in_session_est": entry_bar_in_session_est,
        "bars_held_check": bars_held,
    }


def _group_summary(frame: pd.DataFrame, group_col: str) -> list[dict]:
    if frame.empty or group_col not in frame.columns:
        return []
    grouped = frame.groupby(group_col).agg(
        trades=(group_col, "count"),
        total_pnl=("pnl", "sum"),
        avg_return=("return_pct", "mean"),
        win_rate=("pnl", lambda s: float((s > 0).mean())),
        avg_mfe=("mfe_pct", "mean"),
        avg_mae=("mae_pct", "mean"),
        avg_close_vs_exit=("close_vs_exit_pct", "mean"),
        avg_bars_held=("bars_held", "mean"),
    )
    return grouped.reset_index().to_dict("records")


def main():
    args = parse_args()
    artifact_path = Path(args.artifact)
    payload = json.loads(artifact_path.read_text())
    trades, metadata = _extract_trade_payload(payload, args.period)
    if not trades:
        raise SystemExit("No trades found in artifact")

    config = metadata.get("config", {})
    interval = int(config.get("interval_minutes", 30))
    table_name = f"bars_{interval}m"
    db_path = metadata.get("db_path")
    if not db_path:
        raise SystemExit("Artifact metadata missing db_path")

    conn = duckdb.connect(db_path, read_only=True)
    try:
        enriched = [_analyze_trade(conn, table_name, trade) for trade in trades]
    finally:
        conn.close()

    frame = pd.DataFrame(enriched)
    summary = {
        "artifact": str(artifact_path),
        "period": args.period,
        "trade_count": int(len(frame)),
        "avg_return": round(float(frame["return_pct"].mean()), 4),
        "avg_mfe": round(float(frame["mfe_pct"].dropna().mean()), 4),
        "avg_mae": round(float(frame["mae_pct"].dropna().mean()), 4),
        "avg_close_vs_exit": round(float(frame["close_vs_exit_pct"].dropna().mean()), 4),
    }

    report = {
        "summary": summary,
        "by_setup_family": _group_summary(frame, "setup_family"),
        "by_exit_reason": _group_summary(frame, "exit_reason"),
        "by_entry_bar": _group_summary(frame, "entry_bar_in_session_est"),
        "by_bars_held": _group_summary(frame.assign(bars_held_bucket=frame["bars_held"].clip(upper=15)), "bars_held_bucket"),
        "sample_worst_gap": frame.sort_values("close_vs_exit_pct", ascending=True).head(20).to_dict("records"),
        "sample_best_gap": frame.sort_values("close_vs_exit_pct", ascending=False).head(20).to_dict("records"),
    }

    print()
    print("=" * 96)
    print("  INTRADAY TRADE QUALITY")
    print("=" * 96)
    print(
        f"  trades {summary['trade_count']:>5}"
        f"  avg_return {summary['avg_return']:>7.4f}"
        f"  avg_mfe {summary['avg_mfe']:>7.4f}"
        f"  avg_mae {summary['avg_mae']:>7.4f}"
        f"  avg_close_vs_exit {summary['avg_close_vs_exit']:>7.4f}"
    )
    print("=" * 96)
    for row in report["by_setup_family"]:
        print(
            f"  {row['setup_family']:<26}"
            f" trades {row['trades']:>4}"
            f" avg_ret {row['avg_return']:>7.4f}"
            f" avg_mfe {row['avg_mfe']:>7.4f}"
            f" avg_mae {row['avg_mae']:>7.4f}"
            f" exit_gap {row['avg_close_vs_exit']:>7.4f}"
        )
    print("=" * 96)

    out_path = (
        Path(args.out)
        if args.out
        else artifact_path.with_name(f"{artifact_path.stem}_{args.period}_trade_quality.json")
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
