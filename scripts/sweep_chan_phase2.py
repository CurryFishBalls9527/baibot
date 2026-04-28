#!/usr/bin/env python3
"""Phase 2 optimization sweeps: divergence_rate, atr_stop_mult.

Builds on NEW OPTIMAL config from Phase 1.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sweep2")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()

# NEW OPTIMAL config from Phase 1 (4x sizing)
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
    "--momentum-lookback", "63",
    "--momentum-top-k", "10",
    "--time-stop-bars", "100",
    "--entry-lag-extra-days", "1",
]

PERIODS = ["2023_2025", "2020_2022", "2017_2019", "2014_2016"]


def run_cell(cell_name: str, override_flags: list[str], out_base: str) -> dict:
    out_dir = f"{out_base}/{cell_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Strip overridden keys from base
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


def fmt(label: str, cells: dict[str, dict]) -> None:
    print(f"\n===== {label} =====")
    print(f"  {'config':<30} {'2023-25':>11} {'2020-22':>11} {'2017-19':>11} {'2014-16':>11}  {'avgRDD':>7}")
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
        row = f"  {name:<30} " + " ".join(f"{r:>11}" for r in rs) + f"  {avg_rdd:>+6.2f}"
        print(row)


# === Sweep 1: divergence_rate ===
divrate_cells = {}
for dr in ["0.3", "0.5", "0.7", "0.9"]:
    cell_name = f"divrate_{dr.replace('.', '')}"
    divrate_cells[cell_name] = run_cell(cell_name, ["--divergence-rate", dr],
                                         "results/chan_daily_etf/sweep2/divrate")
fmt("DIVERGENCE_RATE sweep (lag-1, ret%/RDD)", divrate_cells)

# === Sweep 2: atr_stop_mult ===
atrmult_cells = {}
for am in ["1.5", "2.0", "2.5", "3.0"]:
    cell_name = f"atrmult_{am.replace('.', '')}"
    atrmult_cells[cell_name] = run_cell(cell_name, ["--stop-atr-mult", am],
                                         "results/chan_daily_etf/sweep2/atrmult")
fmt("STOP_ATR_MULT sweep (lag-1, ret%/RDD)", atrmult_cells)
