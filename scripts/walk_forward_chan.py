#!/usr/bin/env python3
"""Walk-forward parameter stability audit.

For each test year T in 2017-2025:
  1. Train on years T-2..T-1 — try a grid of {time_stop, divergence_rate} cells
  2. Pick the cell with highest avg R/DD on training years
  3. Evaluate that cell on test year T
  4. Record: test-year R/DD, test return, picked params

Output: a year-by-year table showing whether the OPTIMAL params from in-sample
training generalize to out-of-sample year. Stable params → not overfit. Drift → overfit.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("wf")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()

# Param grid (time_stop × divergence_rate) — kept small for runtime
GRID = [(ts, dr) for ts in [60, 100] for dr in [0.5, 0.7, 0.9]]

BASE_FLAGS = [
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
    "--entry-lag-extra-days", "1",
]


def run_year(period: str, ts: int, dr: float, out_dir: str) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_chan_daily_etf_backtest.py",
        "--symbols", *UNIV16,
        "--periods", period,
        *BASE_FLAGS,
        "--time-stop-bars", str(ts),
        "--divergence-rate", str(dr),
        "--out-dir", out_dir,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    f = Path(out_dir) / f"{period}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else {}


def avg_rdd(years: list[str], cell: tuple[int, float], out_base: str) -> float:
    """Run cell across given years, return avg R/DD."""
    rdds = []
    for y in years:
        out_dir = f"{out_base}/y{y}_ts{cell[0]}_dr{cell[1]}"
        s = run_year(y, cell[0], cell[1], out_dir)
        if not s:
            continue
        dd = abs(s["max_drawdown_pct"])
        rdd = s["total_return_pct"] / dd if dd > 0 else 0
        rdds.append(rdd)
    return sum(rdds) / len(rdds) if rdds else 0


def main():
    test_years = [str(y) for y in range(2017, 2026)]   # 2017 ... 2025

    print()
    print(f"{'TestYr':<8} {'BestTrainCell':<18} {'TrainAvgRDD':>12} {'TestRet':>10} {'TestDD':>10} {'TestRDD':>10}")
    print("-" * 80)

    for test_y in test_years:
        train_years = [str(int(test_y) - 2), str(int(test_y) - 1)]
        log.info("Test year %s, train years %s", test_y, train_years)

        # Find best cell on training years
        best_cell = None
        best_avg = -float("inf")
        for cell in GRID:
            avg = avg_rdd(train_years, cell, f"results/chan_daily_etf/walkfwd")
            if avg > best_avg:
                best_avg = avg
                best_cell = cell

        # Evaluate on test year
        test_dir = f"results/chan_daily_etf/walkfwd/test_y{test_y}_ts{best_cell[0]}_dr{best_cell[1]}"
        s = run_year(test_y, best_cell[0], best_cell[1], test_dir)
        if not s:
            print(f"{test_y:<8} ts{best_cell[0]}_dr{best_cell[1]:<10}     {best_avg:>+11.2f}     n/a")
            continue
        ret = s["total_return_pct"]
        dd = abs(s["max_drawdown_pct"])
        rdd = ret / dd if dd > 0 else 0
        print(f"{test_y:<8} ts{best_cell[0]}_dr{best_cell[1]:<14} {best_avg:>+11.2f}  {ret:>+8.2f}%  {dd:>+8.2f}%  {rdd:>+8.2f}")


if __name__ == "__main__":
    main()
