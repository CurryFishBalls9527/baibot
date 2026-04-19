#!/usr/bin/env python3
"""Compare top-decile winners versus low-gain non-stopped trades for intraday research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Compare winners versus low-gain drift trades")
    p.add_argument("--artifact", required=True, help="Path to intraday backtest JSON artifact")
    p.add_argument("--period", default="2023_2025")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _extract_detail(payload: dict, period: str) -> dict:
    if "details" not in payload:
        raise SystemExit("Expected a period-matrix artifact with details")
    return payload["details"][period]


def _group(frame: pd.DataFrame, group_col: str) -> list[dict]:
    if frame.empty or group_col not in frame.columns:
        return []
    grouped = frame.groupby(group_col).agg(
        trades=(group_col, "count"),
        avg_return=("return_pct", "mean"),
        avg_candidate_score=("candidate_score", "mean"),
        avg_breakout_distance=("breakout_distance_pct", "mean"),
        avg_vwap_distance=("distance_from_vwap_pct", "mean"),
        avg_volume_ratio=("volume_ratio", "mean"),
        avg_entry_bar=("bar_in_session", "mean"),
    )
    return grouped.reset_index().to_dict("records")


def _summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"label": label, "trades": 0}
    return {
        "label": label,
        "trades": int(len(frame)),
        "avg_return": round(float(frame["return_pct"].mean()), 4),
        "avg_candidate_score": round(float(frame["candidate_score"].mean()), 4),
        "avg_breakout_distance_pct": round(float(frame["breakout_distance_pct"].mean()), 4),
        "avg_distance_from_vwap_pct": round(float(frame["distance_from_vwap_pct"].mean()), 4),
        "avg_volume_ratio": round(float(frame["volume_ratio"].mean()), 4),
        "avg_entry_bar": round(float(frame["bar_in_session"].mean()), 2),
    }


def main():
    args = parse_args()
    artifact_path = Path(args.artifact)
    payload = json.loads(artifact_path.read_text())
    detail = _extract_detail(payload, args.period)

    trades = pd.DataFrame(detail.get("trades", []))
    candidates = pd.DataFrame(detail.get("candidate_log", []))
    if trades.empty or candidates.empty:
        raise SystemExit("Artifact missing trades or candidate_log")

    candidates = candidates.rename(columns={"ts": "signal_time"})
    candidate_cols = [
        "symbol",
        "signal_time",
        "setup_family",
        "candidate_score",
        "breakout_distance_pct",
        "distance_from_vwap_pct",
        "volume_ratio",
        "bar_in_session",
    ]
    if "relative_strength_pct" in candidates.columns:
        candidate_cols.append("relative_strength_pct")

    merged = trades.merge(
        candidates[candidate_cols],
        on=["symbol", "signal_time", "setup_family", "candidate_score"],
        how="left",
        suffixes=("", "_candidate"),
    )

    non_stopped = merged.loc[merged["exit_reason"] != "stop"].copy()
    if non_stopped.empty:
        raise SystemExit("No non-stopped trades found")

    winner_cutoff = float(non_stopped["return_pct"].quantile(0.9))
    drift_cutoff = float(non_stopped["return_pct"].quantile(0.5))
    winners = non_stopped.loc[non_stopped["return_pct"] >= winner_cutoff].copy()
    drifters = non_stopped.loc[
        (non_stopped["return_pct"] >= 0.0) & (non_stopped["return_pct"] <= drift_cutoff)
    ].copy()

    report = {
        "summary": {
            "artifact": str(artifact_path),
            "period": args.period,
            "non_stopped_trades": int(len(non_stopped)),
            "winner_cutoff_return_pct": round(winner_cutoff, 4),
            "drift_cutoff_return_pct": round(drift_cutoff, 4),
        },
        "winner_summary": _summary(winners, "top_decile_winners"),
        "drifter_summary": _summary(drifters, "low_gain_drifters"),
        "winners_by_setup_family": _group(winners, "setup_family"),
        "drifters_by_setup_family": _group(drifters, "setup_family"),
        "winners_by_entry_bar": _group(winners, "bar_in_session"),
        "drifters_by_entry_bar": _group(drifters, "bar_in_session"),
        "sample_winners": winners.sort_values("return_pct", ascending=False).head(20).to_dict("records"),
        "sample_drifters": drifters.sort_values("return_pct", ascending=True).head(20).to_dict("records"),
    }

    print()
    print("=" * 96)
    print("  INTRADAY WINNERS VS DRIFTS")
    print("=" * 96)
    print(
        f"  non_stopped {report['summary']['non_stopped_trades']:>4}"
        f"  winner_cutoff {report['summary']['winner_cutoff_return_pct']:>7.4f}"
        f"  drift_cutoff {report['summary']['drift_cutoff_return_pct']:>7.4f}"
    )
    print("=" * 96)
    for label, block in [("WIN", report["winner_summary"]), ("DRIFT", report["drifter_summary"])]:
        print(
            f"  {label:<5}"
            f" trades {block.get('trades', 0):>4}"
            f" avg_ret {block.get('avg_return', 0):>7.4f}"
            f" cand {block.get('avg_candidate_score', 0):>6.3f}"
            f" breakout {block.get('avg_breakout_distance_pct', 0):>7.4f}"
            f" vwap {block.get('avg_distance_from_vwap_pct', 0):>7.4f}"
            f" vol {block.get('avg_volume_ratio', 0):>6.3f}"
            f" bar {block.get('avg_entry_bar', 0):>5.2f}"
        )
    print("=" * 96)

    out_path = (
        Path(args.out)
        if args.out
        else artifact_path.with_name(f"{artifact_path.stem}_{args.period}_winners_vs_drifts.json")
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
