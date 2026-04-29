#!/usr/bin/env python3
"""Phase 4 multi-variable optimization sweeps.

Three tests:
  1. Conditional pyramid (3 conditions × combos = 8 cells)
  2. Sector cap × max_positions joint sweep (16 cells)
  3. Per-signal-type stops (5 cells)

Reports avg R/DD per cell + flags candidates for lag-1 follow-up.
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("phase4")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()

OPTIMAL_FLAGS = [
    "--no-require-sure", "--no-bi-strict",
    "--bs-type", "1,1p,2,2s,3a,3b",
    "--buy-types", "T1,T1P,T2,T2S,T3A,T3B",
    "--sell-types", "T1",
    "--min-zs-cnt", "0",
    "--sizing-mode", "atr_parity",
    "--risk-per-trade", "0.020",
    "--position-pct", "0.25",
    "--no-shorts", "--max-positions", "6",
    "--entry-mode", "donchian_or_seg",
    "--donchian-period", "30",
    "--macd-algo", "slope",
    "--divergence-rate", "0.7",
    "--momentum-lookback", "63",
    "--momentum-top-k", "10",
    "--time-stop-bars", "100",
    "--entry-priority", "momentum",
    "--trend-type-filter", "trend_only",
    "--entry-lag-extra-days", "1",
]
PERIODS = ["2023_2025", "2020_2022", "2017_2019", "2014_2016"]


def run_cell(cell_name: str, override_flags: list[str], out_base: str) -> dict:
    out_dir = f"{out_base}/{cell_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    override_keys = set(f for f in override_flags if f.startswith("--"))
    base_clean = []
    i = 0
    while i < len(OPTIMAL_FLAGS):
        f = OPTIMAL_FLAGS[i]
        if f in override_keys:
            if i + 1 < len(OPTIMAL_FLAGS) and not OPTIMAL_FLAGS[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        base_clean.append(f)
        i += 1
    cmd = [
        sys.executable, "scripts/run_chan_daily_etf_backtest.py",
        "--symbols", *UNIV16,
        "--periods", *PERIODS,
        *base_clean,
        *override_flags,
        "--out-dir", out_dir,
    ]
    log.info("Running %s ...", cell_name)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("FAIL %s: %s", cell_name, r.stderr[-300:])
        return {}
    return {p: json.loads((Path(out_dir) / f"{p}.json").read_text())["summary"]
            for p in PERIODS if (Path(out_dir) / f"{p}.json").exists()}


def fmt(label: str, cells: dict[str, dict], baseline_avg: float = None) -> dict[str, float]:
    print(f"\n===== {label} =====")
    print(f"  {'config':<35} {'2023-25':>11} {'2020-22':>11} {'2017-19':>11} {'2014-16':>11}  {'avgRDD':>7}  {'minRDD':>7}")
    avgs = {}
    for name, sums in cells.items():
        rs = []
        rdds = []
        for p in PERIODS:
            s = sums.get(p, {})
            if not s:
                rs.append("    n/a")
                continue
            ret = s["total_return_pct"]
            dd = abs(s["max_drawdown_pct"])
            rdd = ret / dd if dd > 0 else 0
            rs.append(f"{ret:>+5.1f}/{rdd:>+4.1f}")
            rdds.append(rdd)
        avg_rdd = sum(rdds) / len(rdds) if rdds else 0
        min_rdd = min(rdds) if rdds else 0
        delta = ""
        if baseline_avg is not None:
            d = avg_rdd - baseline_avg
            delta = f"  Δ{d:+.2f}"
        print(f"  {name:<35} " + " ".join(f"{r:>11}" for r in rs) + f"  {avg_rdd:>+6.2f}  {min_rdd:>+6.2f}{delta}")
        avgs[name] = avg_rdd
    return avgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="all", choices=["all", "1", "2", "3"])
    return p.parse_args()


def main():
    args = parse_args()
    out_base = "results/chan_daily_etf/sweep4"

    # Always run baseline first (used as comparison)
    log.info("===== Baseline =====")
    baseline = {"baseline": run_cell("baseline", [], out_base)}
    baseline_avgs = fmt("BASELINE (current OPTIMAL)", baseline)
    bavg = list(baseline_avgs.values())[0]

    if args.test in ("all", "1"):
        # Test 1: Conditional pyramid
        cells = {}
        common_pyramid = ["--pyramid", "--pyramid-thresholds", "1.5,3.0",
                          "--pyramid-fractions", "0.5,0.33"]
        cells["pyramid_plain"] = run_cell("pyramid_plain", common_pyramid, f"{out_base}/test1")
        cells["pyr_donchian_only"] = run_cell(
            "pyr_donchian_only", common_pyramid + ["--pyramid-donchian-only"], f"{out_base}/test1")
        cells["pyr_segseg_only"] = run_cell(
            "pyr_segseg_only", common_pyramid + ["--pyramid-require-up-segseg-sure"], f"{out_base}/test1")
        cells["pyr_zsbroken_only"] = run_cell(
            "pyr_zsbroken_only", common_pyramid + ["--pyramid-require-zs-broken"], f"{out_base}/test1")
        cells["pyr_donch_AND_segseg"] = run_cell(
            "pyr_donch_AND_segseg",
            common_pyramid + ["--pyramid-donchian-only", "--pyramid-require-up-segseg-sure"],
            f"{out_base}/test1")
        cells["pyr_donch_AND_zsbroken"] = run_cell(
            "pyr_donch_AND_zsbroken",
            common_pyramid + ["--pyramid-donchian-only", "--pyramid-require-zs-broken"],
            f"{out_base}/test1")
        cells["pyr_all_three"] = run_cell(
            "pyr_all_three",
            common_pyramid + ["--pyramid-donchian-only",
                               "--pyramid-require-up-segseg-sure",
                               "--pyramid-require-zs-broken"],
            f"{out_base}/test1")
        fmt("TEST 1: Conditional pyramid (vs baseline)", cells, baseline_avg=bavg)

    if args.test in ("all", "2"):
        # Test 2: Sector cap × max_positions
        cells = {}
        for max_pos in [6, 7, 8, 10]:
            for cap in [0, 3, 4, 5]:
                if cap == 0:
                    label = f"max{max_pos}_no_cap"
                    extra = ["--max-positions", str(max_pos)]
                else:
                    label = f"max{max_pos}_cap{cap}"
                    extra = ["--max-positions", str(max_pos),
                             "--equity-sector-cap", str(cap)]
                cells[label] = run_cell(label, extra, f"{out_base}/test2")
        fmt("TEST 2: Sector cap × max_positions (vs baseline)", cells, baseline_avg=bavg)

    if args.test in ("all", "3"):
        # Test 3: Per-signal-type stops
        cells = {}
        # baseline matches current (uniform 2.0 ATR / 100 bars)
        cells["uniform_2.0_ts100"] = run_cell("uniform_2.0_ts100", [], f"{out_base}/test3")
        cells["donch_2.5_seg_1.5"] = run_cell(
            "donch_2.5_seg_1.5",
            ["--stop-atr-mult", "2.5", "--stop-atr-mult-seg", "1.5"],
            f"{out_base}/test3")
        cells["donch_2.0_seg_1.5"] = run_cell(
            "donch_2.0_seg_1.5",
            ["--stop-atr-mult-seg", "1.5"],
            f"{out_base}/test3")
        cells["donch_2.5_seg_2.0"] = run_cell(
            "donch_2.5_seg_2.0",
            ["--stop-atr-mult", "2.5"],
            f"{out_base}/test3")
        cells["donch_3.0_seg_1.5"] = run_cell(
            "donch_3.0_seg_1.5",
            ["--stop-atr-mult", "3.0", "--stop-atr-mult-seg", "1.5"],
            f"{out_base}/test3")
        cells["donch_ts120_seg_ts60"] = run_cell(
            "donch_ts120_seg_ts60",
            ["--time-stop-bars", "120", "--time-stop-bars-seg", "60"],
            f"{out_base}/test3")
        cells["combined_split"] = run_cell(
            "combined_split",
            ["--stop-atr-mult", "2.5", "--stop-atr-mult-seg", "1.5",
             "--time-stop-bars", "120", "--time-stop-bars-seg", "60"],
            f"{out_base}/test3")
        fmt("TEST 3: Per-signal-type stops (vs baseline)", cells, baseline_avg=bavg)


if __name__ == "__main__":
    main()
