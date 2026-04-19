#!/usr/bin/env python3
"""Postmortem stopped trades versus non-stopped trades for intraday research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze stopped trades versus non-stopped trades")
    p.add_argument("--artifact", required=True, help="Path to intraday backtest JSON artifact")
    p.add_argument("--period", default="2023_2025")
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
        SELECT ts, open, high, low, close
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


def _enrich_trade(conn: duckdb.DuckDBPyConnection, table_name: str, trade: dict) -> dict:
    bars = _load_bar_slice(conn, table_name, trade["symbol"], trade["entry_time"], trade["exit_time"])
    entry_price = float(trade["entry_price"])
    if bars.empty:
        return {
            **trade,
            "mfe_pct": None,
            "mae_pct": None,
            "stopped": trade.get("exit_reason") == "stop",
        }
    return {
        **trade,
        "mfe_pct": round((float(bars["high"].max()) / entry_price) - 1.0, 4),
        "mae_pct": round((float(bars["low"].min()) / entry_price) - 1.0, 4),
        "stopped": trade.get("exit_reason") == "stop",
    }


def _summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"label": label, "trades": 0}
    return {
        "label": label,
        "trades": int(len(frame)),
        "avg_return": round(float(frame["return_pct"].mean()), 4),
        "avg_candidate_score": round(float(frame["candidate_score"].dropna().mean()), 4)
        if "candidate_score" in frame.columns
        else None,
        "avg_selection_score": round(float(frame["selection_score"].dropna().mean()), 4)
        if "selection_score" in frame.columns
        else None,
        "avg_mfe": round(float(frame["mfe_pct"].dropna().mean()), 4),
        "avg_mae": round(float(frame["mae_pct"].dropna().mean()), 4),
        "avg_bars_held": round(float(frame["bars_held"].mean()), 2),
        "win_rate": round(float((frame["pnl"] > 0).mean()), 4),
    }


def _group(frame: pd.DataFrame, group_col: str) -> list[dict]:
    if frame.empty or group_col not in frame.columns:
        return []
    agg_map = {
        "trades": (group_col, "count"),
        "avg_return": ("return_pct", "mean"),
        "avg_mfe": ("mfe_pct", "mean"),
        "avg_mae": ("mae_pct", "mean"),
        "avg_bars_held": ("bars_held", "mean"),
        "win_rate": ("pnl", lambda s: float((s > 0).mean())),
    }
    if "candidate_score" in frame.columns:
        agg_map["avg_candidate_score"] = ("candidate_score", "mean")
    if "selection_score" in frame.columns:
        agg_map["avg_selection_score"] = ("selection_score", "mean")
    grouped = frame.groupby(group_col).agg(**agg_map)
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
        enriched = [_enrich_trade(conn, table_name, trade) for trade in trades]
    finally:
        conn.close()

    frame = pd.DataFrame(enriched)
    stopped = frame.loc[frame["stopped"]].copy()
    survivors = frame.loc[~frame["stopped"]].copy()

    report = {
        "summary": {
            "artifact": str(artifact_path),
            "period": args.period,
            "all_trades": int(len(frame)),
            "stopped_trades": int(len(stopped)),
            "non_stopped_trades": int(len(survivors)),
        },
        "stopped_summary": _summary(stopped, "stopped"),
        "non_stopped_summary": _summary(survivors, "non_stopped"),
        "stopped_by_setup_family": _group(stopped, "setup_family"),
        "non_stopped_by_setup_family": _group(survivors, "setup_family"),
        "stopped_by_entry_bar": _group(
            stopped.assign(
                entry_bar_in_session_est=pd.to_datetime(stopped["entry_time"]).apply(
                    lambda ts: int((ts - ts.normalize().replace(hour=8, minute=30)).total_seconds() // (30 * 60))
                )
            ),
            "entry_bar_in_session_est",
        ),
        "worst_stops": stopped.sort_values("return_pct").head(25).to_dict("records"),
    }

    print()
    print("=" * 96)
    print("  INTRADAY STOP POSTMORTEM")
    print("=" * 96)
    print(
        f"  stopped {report['summary']['stopped_trades']:>4}"
        f"  non_stopped {report['summary']['non_stopped_trades']:>4}"
        f"  stop_avg_return {report['stopped_summary'].get('avg_return', 0):>7.4f}"
        f"  survivor_avg_return {report['non_stopped_summary'].get('avg_return', 0):>7.4f}"
    )
    print("=" * 96)
    for row in report["stopped_by_setup_family"]:
        avg_candidate_score = row.get("avg_candidate_score")
        avg_selection_score = row.get("avg_selection_score")
        print(
            f"  STOP {row['setup_family']:<22}"
            f" trades {row['trades']:>3}"
            f" avg_ret {row['avg_return']:>7.4f}"
            f" cand {(avg_candidate_score if avg_candidate_score is not None else float('nan')):>6.3f}"
            f" sel {(avg_selection_score if avg_selection_score is not None else float('nan')):>6.3f}"
            f" mfe {row['avg_mfe']:>7.4f}"
            f" mae {row['avg_mae']:>7.4f}"
        )
    print("=" * 96)

    out_path = (
        Path(args.out)
        if args.out
        else artifact_path.with_name(f"{artifact_path.stem}_{args.period}_stops.json")
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
