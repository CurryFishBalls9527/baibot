#!/usr/bin/env python3
"""Phase 3 sweeps: trailing stop variants, partial-exit variants, combinations."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sweep3")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()

# Config from Phase 2 with divergence_rate=0.7 (NEW NEW OPTIMAL)
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


def fmt(label: str, cells: dict[str, dict]) -> None:
    print(f"\n===== {label} =====")
    print(f"  {'config':<35} {'2023-25':>11} {'2020-22':>11} {'2017-19':>11} {'2014-16':>11}  {'avgRDD':>7}  {'sumRet':>7}")
    for name, sums in cells.items():
        rs = []
        rdds = []
        rets = []
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
            rets.append(ret)
        avg_rdd = sum(rdds) / len(rdds) if rdds else 0
        sum_ret = sum(rets)
        row = f"  {name:<35} " + " ".join(f"{r:>11}" for r in rs) + f"  {avg_rdd:>+6.2f}  {sum_ret:>+6.1f}"
        print(row)


# Baseline: NEW OPTIMAL with divrate=0.7 (no trailing/partial)
baseline_cells = {"baseline_no_overlay": run_cell("baseline_no_overlay", [],
                                                    "results/chan_daily_etf/sweep3")}

# Trailing stop variants
trail_cells = {}
for be_r, t_r, mult in [
    (1.5, 3.0, 1.5),
    (2.0, 4.0, 1.5),
    (2.0, 4.0, 2.0),
    (2.5, 5.0, 1.5),
    (1.0, 2.0, 1.5),  # aggressive trail
]:
    name = f"trail_be{be_r}_t{t_r}_m{mult}"
    flags = ["--trailing-stop",
             "--trail-breakeven-r", str(be_r),
             "--trail-at-r", str(t_r),
             "--trail-atr-mult", str(mult)]
    trail_cells[name] = run_cell(name, flags, "results/chan_daily_etf/sweep3/trail")

# Partial exit variants
partial_cells = {}
for at_r, pct in [(2.0, 0.33), (2.0, 0.50), (2.0, 0.67), (3.0, 0.50), (1.5, 0.50)]:
    name = f"partial_r{at_r}_pct{pct}"
    flags = ["--partial-exit", "--partial-at-r", str(at_r), "--partial-pct", str(pct)]
    partial_cells[name] = run_cell(name, flags, "results/chan_daily_etf/sweep3/partial")

# Combo: best trail + best partial
combo_cells = {}
for label, flags in [
    ("trail_only_be2_t4",          ["--trailing-stop",
                                     "--trail-breakeven-r", "2.0", "--trail-at-r", "4.0", "--trail-atr-mult", "1.5"]),
    ("partial_only_r2_p50",        ["--partial-exit", "--partial-at-r", "2.0", "--partial-pct", "0.5"]),
    ("trail_AND_partial",          ["--trailing-stop",
                                     "--trail-breakeven-r", "2.0", "--trail-at-r", "4.0", "--trail-atr-mult", "1.5",
                                     "--partial-exit", "--partial-at-r", "2.0", "--partial-pct", "0.5"]),
]:
    combo_cells[label] = run_cell(label, flags, "results/chan_daily_etf/sweep3/combo")

fmt("BASELINE (NEW OPTIMAL + divrate 0.7, no overlays)", baseline_cells)
fmt("TRAILING STOP variants", trail_cells)
fmt("PARTIAL EXIT variants", partial_cells)
fmt("COMBO trail + partial", combo_cells)
