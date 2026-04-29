#!/usr/bin/env python3
"""Pareto re-analysis of Phase 4 + Phase 5 sweep results.

Aggregates every cell across multi-axis metrics:
  - avg R/DD (Phase metric)
  - min R/DD (worst-period robustness)
  - total return % (sum across 4 periods)
  - max single-period DD (worst tail risk)
  - num positive periods (out of 4)

Identifies Pareto frontier — cells not dominated by any other on all axes.
"""
from __future__ import annotations
import json
from pathlib import Path

PERIODS = ["2023_2025", "2020_2022", "2017_2019", "2014_2016"]
ROOT = Path("/Users/myu/code/baibot/results/chan_daily_etf")

# Sweep dirs to scan + their cell-name extractor
SWEEP_DIRS = [
    "sweep4/baseline",
    "sweep4/test1",
    "sweep4/test2",
    "sweep4/test3",
    "phase5_credit",
    "phase5_calendar",
    "phase5_voladapt",
    "phase5_reentry",
    "phase5_voltarget",
]


def collect():
    """Walk every cell dir, aggregate per-period summaries into one record."""
    records = []
    for sd in SWEEP_DIRS:
        dir_path = ROOT / sd
        if not dir_path.exists():
            continue
        # Each subdir = one cell; or directly files for top-level baseline-like dirs
        cells = [d for d in dir_path.iterdir() if d.is_dir()]
        if not cells and any((dir_path / f"{p}.json").exists() for p in PERIODS):
            cells = [dir_path]
        for cell in cells:
            cell_name = f"{sd}/{cell.name}" if cell != dir_path else sd
            sums = {}
            for p in PERIODS:
                f = cell / f"{p}.json"
                if not f.exists():
                    continue
                try:
                    sums[p] = json.load(open(f))["summary"]
                except Exception:
                    pass
            if len(sums) < 4:  # need full coverage
                continue
            r = aggregate(cell_name, sums)
            if r:
                records.append(r)
    return records


def aggregate(name, sums):
    rdds, returns, dds = [], [], []
    pos_periods = 0
    for p in PERIODS:
        s = sums[p]
        ret = s["total_return_pct"]
        dd = abs(s["max_drawdown_pct"])
        rdd = ret / dd if dd > 0 else 0.0
        rdds.append(rdd)
        returns.append(ret)
        dds.append(dd)
        if ret > 0:
            pos_periods += 1
    return {
        "name": name,
        "avg_rdd": sum(rdds) / len(rdds),
        "min_rdd": min(rdds),
        "total_ret": sum(returns),
        "max_dd": max(dds),  # worst single-period DD (tail risk)
        "pos_periods": pos_periods,  # 0..4
    }


def is_pareto_optimal(rec, others):
    """Pareto-optimal: no other record dominates it on ALL axes (>= on each, > on at least one)."""
    # Maximize avg_rdd, min_rdd, total_ret, pos_periods. Minimize max_dd.
    for o in others:
        if o["name"] == rec["name"]:
            continue
        if (o["avg_rdd"] >= rec["avg_rdd"]
                and o["min_rdd"] >= rec["min_rdd"]
                and o["total_ret"] >= rec["total_ret"]
                and o["max_dd"] <= rec["max_dd"]
                and o["pos_periods"] >= rec["pos_periods"]
                and (o["avg_rdd"] > rec["avg_rdd"]
                     or o["min_rdd"] > rec["min_rdd"]
                     or o["total_ret"] > rec["total_ret"]
                     or o["max_dd"] < rec["max_dd"]
                     or o["pos_periods"] > rec["pos_periods"])):
            return False
    return True


def main():
    recs = collect()
    print(f"Loaded {len(recs)} cells with full 4-period coverage.\n")

    pareto = [r for r in recs if is_pareto_optimal(r, recs)]
    print(f"Pareto-optimal cells (not dominated on any axis): {len(pareto)}\n")

    # Sort Pareto by total_ret descending
    pareto.sort(key=lambda r: -r["total_ret"])

    print(f"  {'cell':<55} {'avgRDD':>7} {'minRDD':>7} {'sumRet':>8} {'maxDD':>7} {'pos':>4}")
    print(f"  {'-'*55} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*4}")
    for r in pareto:
        print(f"  {r['name']:<55} {r['avg_rdd']:+7.2f} {r['min_rdd']:+7.2f} {r['total_ret']:+7.1f}% {r['max_dd']:+6.1f}% {r['pos_periods']:>4}")

    # Highlight: how many of these are wins?
    print()
    print("Per-axis ranking (top 5 each):")
    for axis, key, sign, fmt in [
        ("Best avg R/DD", "avg_rdd", -1, "+.2f"),
        ("Best min R/DD", "min_rdd", -1, "+.2f"),
        ("Best total return %", "total_ret", -1, "+.1f"),
        ("Lowest max DD %", "max_dd", 1, "+.1f"),
        ("Most positive periods", "pos_periods", -1, "d"),
    ]:
        print(f"\n{axis}:")
        ranked = sorted(recs, key=lambda r: sign * r[key])
        for r in ranked[:5]:
            print(f"  {r['name']:<55} {format(r[key], fmt)}")


if __name__ == "__main__":
    main()
