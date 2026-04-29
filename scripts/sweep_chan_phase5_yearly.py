#!/usr/bin/env python3
"""Phase 5 — Per-year decomposition: baseline vs block_aug_sep."""
from __future__ import annotations
import json, logging, subprocess, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("yr")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()
YEARS = [str(y) for y in range(2014, 2026)] + ["2026_ytd"]

BASE = [
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


def run_year(year, override_flags, out_base):
    out_dir = f"{out_base}/y{year}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/run_chan_daily_etf_backtest.py",
           "--symbols", *UNIV16, "--periods", year,
           *BASE, *override_flags, "--out-dir", out_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("FAIL y%s: %s", year, r.stderr[-300:])
        return {}
    f = Path(out_dir) / f"{year}.json"
    return json.loads(f.read_text())["summary"] if f.exists() else {}


def main():
    bb = "results/chan_daily_etf/phase5_yearly/baseline"
    cb = "results/chan_daily_etf/phase5_yearly/block_aug_sep"
    Path(bb).mkdir(parents=True, exist_ok=True)
    Path(cb).mkdir(parents=True, exist_ok=True)

    print(f"\n{'Year':<10}  {'Baseline ret/dd/RDD':<24}  {'BlockAugSep ret/dd/RDD':<24}  {'Δret':>6}  {'ΔRDD':>6}")
    print("-" * 90)

    sum_b_ret, sum_c_ret = 0, 0
    n_pos_b, n_pos_c = 0, 0

    for y in YEARS:
        log.info("Year %s ...", y)
        sb = run_year(y, [], bb)
        sc = run_year(y, ["--calendar-filter", "block_months",
                          "--calendar-block-months", "8,9"], cb)
        if not sb or not sc:
            print(f"{y:<10}  data missing")
            continue
        bret, bdd = sb["total_return_pct"], abs(sb["max_drawdown_pct"])
        cret, cdd = sc["total_return_pct"], abs(sc["max_drawdown_pct"])
        brdd = bret/bdd if bdd > 0 else 0
        crdd = cret/cdd if cdd > 0 else 0
        sum_b_ret += bret; sum_c_ret += cret
        if bret > 0: n_pos_b += 1
        if cret > 0: n_pos_c += 1
        print(f"{y:<10}  {bret:+6.2f}% / {bdd:5.2f}% / {brdd:+5.2f}     "
              f"{cret:+6.2f}% / {cdd:5.2f}% / {crdd:+5.2f}      {cret-bret:+6.2f}  {crdd-brdd:+6.2f}")

    print("-" * 90)
    print(f"  Sum: baseline {sum_b_ret:+.2f}%   blockAugSep {sum_c_ret:+.2f}%   Δ={sum_c_ret-sum_b_ret:+.2f}pp")
    print(f"  Positive years: baseline {n_pos_b}/{len(YEARS)}   blockAugSep {n_pos_c}/{len(YEARS)}")


if __name__ == "__main__":
    main()
