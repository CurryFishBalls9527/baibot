#!/usr/bin/env python3
"""Phase 5 — Credit-spread regime gate sweep.

Tests HYG/LQD and HYG/TLT ratios as risk-off detectors. Modes:
  - below_sma: ratio < N-day SMA → block
  - negative_delta: ratio - ratio[N days ago] < 0 → block
  - z_below: z-score below threshold → block
"""
from __future__ import annotations
import json, logging, subprocess, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("phase5_credit")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()
PERIODS = ["2023_2025", "2020_2022", "2017_2019", "2014_2016"]

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


def run_cell(cell_name: str, override_flags: list[str], out_base: str) -> dict:
    out_dir = f"{out_base}/{cell_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    keys = set(f for f in override_flags if f.startswith("--"))
    base_clean = []
    i = 0
    while i < len(OPTIMAL_FLAGS):
        f = OPTIMAL_FLAGS[i]
        if f in keys:
            if i + 1 < len(OPTIMAL_FLAGS) and not OPTIMAL_FLAGS[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        base_clean.append(f); i += 1
    cmd = [sys.executable, "scripts/run_chan_daily_etf_backtest.py",
           "--symbols", *UNIV16, "--periods", *PERIODS,
           *base_clean, *override_flags, "--out-dir", out_dir]
    log.info("Running %s ...", cell_name)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("FAIL %s: %s", cell_name, r.stderr[-400:])
        return {}
    return {p: json.loads((Path(out_dir) / f"{p}.json").read_text())["summary"]
            for p in PERIODS if (Path(out_dir) / f"{p}.json").exists()}


def fmt(label, cells, baseline_avg=None):
    print(f"\n===== {label} =====")
    print(f"  {'cell':<35} {'2023-25':>13} {'2020-22':>13} {'2017-19':>13} {'2014-16':>13}  {'avg':>6}  {'min':>6}")
    avgs = {}
    for name, sums in cells.items():
        rdds, rs = [], []
        for p in PERIODS:
            s = sums.get(p, {})
            if not s:
                rs.append("       n/a"); continue
            ret = s["total_return_pct"]; dd = abs(s["max_drawdown_pct"])
            rdd = ret/dd if dd > 0 else 0.0
            rs.append(f"{ret:+5.1f}/{rdd:+4.1f}"); rdds.append(rdd)
        avg = sum(rdds)/len(rdds) if rdds else 0
        mn = min(rdds) if rdds else 0
        delta = ""
        if baseline_avg is not None:
            delta = f"  Δ{avg-baseline_avg:+.2f}"
        print(f"  {name:<35} " + " ".join(f"{c:>13}" for c in rs) + f"  {avg:+6.2f}  {mn:+6.2f}{delta}")
        avgs[name] = avg
    return avgs


def main():
    out_base = "results/chan_daily_etf/phase5_credit"
    Path(out_base).mkdir(parents=True, exist_ok=True)

    log.info("Baseline (no gate) ...")
    baseline = {"baseline": run_cell("baseline", [], out_base)}
    bavgs = fmt("BASELINE", baseline)
    bavg = list(bavgs.values())[0]

    cells = {}
    # HYG/LQD ratio (pure credit spread)
    for lb in [20, 60, 120]:
        for mode in ["below_sma", "negative_delta"]:
            label = f"hyglqd_{mode}_lb{lb}"
            cells[label] = run_cell(label,
                ["--credit-spread-filter",
                 "--credit-spread-numerator", "HYG",
                 "--credit-spread-denominator", "LQD",
                 "--credit-spread-lookback", str(lb),
                 "--credit-spread-block-mode", mode],
                out_base)

    # HYG/LQD with z-score
    for lb in [60, 120]:
        for z in [-0.5, -1.0, -1.5]:
            label = f"hyglqd_z{z}_lb{lb}"
            cells[label] = run_cell(label,
                ["--credit-spread-filter",
                 "--credit-spread-numerator", "HYG",
                 "--credit-spread-denominator", "LQD",
                 "--credit-spread-lookback", str(lb),
                 "--credit-spread-block-mode", "z_below",
                 "--credit-spread-z-threshold", str(z)],
                out_base)

    # HYG/TLT ratio (risk-on/off via flight-to-quality)
    for lb in [60, 120]:
        for mode in ["below_sma", "negative_delta"]:
            label = f"hygtlt_{mode}_lb{lb}"
            cells[label] = run_cell(label,
                ["--credit-spread-filter",
                 "--credit-spread-numerator", "HYG",
                 "--credit-spread-denominator", "TLT",
                 "--credit-spread-lookback", str(lb),
                 "--credit-spread-block-mode", mode],
                out_base)

    fmt("CREDIT-SPREAD GATES (vs baseline)", cells, baseline_avg=bavg)


if __name__ == "__main__":
    main()
