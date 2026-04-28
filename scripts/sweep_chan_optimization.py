#!/usr/bin/env python3
"""Batch optimization sweeps for the donchian_or_seg candidate.

Runs multiple parameter cells in sequence, prints a single comparative table.
Prints lag-1 numbers (which are the published candidate's truth).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sweep")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()
UNIV61 = (
    "ACWI BIL BND DBA DBC DIA EEM EFA EWA EWG EWJ EWZ FXE FXI GLD GSG HYG IBB "
    "IEFA IEMG INDA IWM IYR LQD MCHI MTUM OIH QQQ QUAL SCHD SCO SHY SLV SOXX "
    "SPY TIP TLT UCO UNG USMV USO UUP VEA VEU VIG VLUE VNQ VWO VYM XBI XLB "
    "XLE XLF XLI XLK XLP XLU XLV XLY XME XOP"
).split()

BASE_FLAGS = [
    "--no-require-sure", "--no-bi-strict",
    "--bs-type", "1,1p,2,2s,3a,3b",
    "--buy-types", "T1,T1P,T2,T2S,T3A,T3B",
    "--sell-types", "T1",
    "--min-zs-cnt", "0",
    "--sizing-mode", "atr_parity",
    "--no-shorts", "--max-positions", "6",
    "--entry-mode", "donchian_or_seg",
    "--macd-algo", "slope",
    "--entry-lag-extra-days", "1",
]

PERIODS_4 = ["2023_2025", "2020_2022", "2017_2019", "2014_2016"]
PERIODS_3 = ["2023_2025", "2020_2022", "2017_2019"]


def run_cell(symbols: list[str], extra_flags: list[str], out_dir: str, periods: list[str]) -> dict:
    cmd = [
        sys.executable, "scripts/run_chan_daily_etf_backtest.py",
        "--symbols", *symbols,
        "--periods", *periods,
        *BASE_FLAGS,
        *extra_flags,
        "--out-dir", out_dir,
    ]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log.info("Running %s", out_dir.split("/")[-1])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("Cell failed: %s\n%s", out_dir, result.stderr[-500:])
        return {}
    summaries = {}
    for p in periods:
        f = Path(out_dir) / f"{p}.json"
        if f.exists():
            d = json.loads(f.read_text())
            summaries[p] = d["summary"]
    return summaries


def fmt_table(label: str, cells: dict[str, dict], periods: list[str]) -> None:
    print()
    print(f"===== {label} =====")
    print(f"  {'config':<40} " + " ".join(f"{p[:7]:>10}" for p in periods))
    for name, sums in cells.items():
        row = []
        for p in periods:
            s = sums.get(p, {})
            if not s:
                row.append("    n/a")
                continue
            ret = s["total_return_pct"]
            dd = abs(s["max_drawdown_pct"])
            rdd = ret / dd if dd > 0 else 0
            row.append(f"{ret:>+5.1f}/{rdd:>+4.1f}")
        print(f"  {name:<40} " + " ".join(f"{r:>10}" for r in row))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True,
                   choices=["sizing", "time_stop", "momentum", "donchian", "buy_types",
                            "optimal_check", "all"])
    p.add_argument("--universe", default="16", choices=["16", "61"])
    p.add_argument("--out-base", default="results/chan_daily_etf/sweep")
    return p.parse_args()


def main():
    args = parse_args()
    universe = UNIV16 if args.universe == "16" else UNIV61
    periods = PERIODS_4 if args.universe == "16" else PERIODS_3

    sweeps = {
        "sizing": [
            # Risk per trade × notional cap matrix. The notional cap binds first
            # for tight stops; need both knobs to scale.
            ("risk0.005_cap0.10", ["--risk-per-trade", "0.005", "--position-pct", "0.10",
                                    "--time-stop-bars", "60",
                                    "--momentum-lookback", "63", "--momentum-top-k", "6",
                                    "--donchian-period", "20"]),
            ("risk0.010_cap0.15", ["--risk-per-trade", "0.010", "--position-pct", "0.15",
                                    "--time-stop-bars", "60",
                                    "--momentum-lookback", "63", "--momentum-top-k", "6",
                                    "--donchian-period", "20"]),
            ("risk0.015_cap0.20", ["--risk-per-trade", "0.015", "--position-pct", "0.20",
                                    "--time-stop-bars", "60",
                                    "--momentum-lookback", "63", "--momentum-top-k", "6",
                                    "--donchian-period", "20"]),
            ("risk0.020_cap0.25", ["--risk-per-trade", "0.020", "--position-pct", "0.25",
                                    "--time-stop-bars", "60",
                                    "--momentum-lookback", "63", "--momentum-top-k", "6",
                                    "--donchian-period", "20"]),
        ],
        "time_stop": [
            ("ts_40",  ["--time-stop-bars", "40",  "--risk-per-trade", "0.005",
                        "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
            ("ts_60",  ["--time-stop-bars", "60",  "--risk-per-trade", "0.005",
                        "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
            ("ts_80",  ["--time-stop-bars", "80",  "--risk-per-trade", "0.005",
                        "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
            ("ts_100", ["--time-stop-bars", "100", "--risk-per-trade", "0.005",
                        "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
            ("ts_120", ["--time-stop-bars", "120", "--risk-per-trade", "0.005",
                        "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
        ],
        "momentum": [
            ("mom_lb20_top6",   ["--momentum-lookback", "20",  "--momentum-top-k", "6",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
            ("mom_lb63_top4",   ["--momentum-lookback", "63",  "--momentum-top-k", "4",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
            ("mom_lb63_top6",   ["--momentum-lookback", "63",  "--momentum-top-k", "6",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
            ("mom_lb63_top8",   ["--momentum-lookback", "63",  "--momentum-top-k", "8",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
            ("mom_lb63_top10",  ["--momentum-lookback", "63",  "--momentum-top-k", "10",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
            ("mom_lb126_top6",  ["--momentum-lookback", "126", "--momentum-top-k", "6",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
            ("mom_lb250_top6",  ["--momentum-lookback", "250", "--momentum-top-k", "6",
                                  "--risk-per-trade", "0.005", "--time-stop-bars", "60", "--donchian-period", "20"]),
        ],
        "donchian": [
            ("dn_10",  ["--donchian-period", "10",  "--risk-per-trade", "0.005",
                         "--momentum-lookback", "63", "--momentum-top-k", "6", "--time-stop-bars", "60"]),
            ("dn_20",  ["--donchian-period", "20",  "--risk-per-trade", "0.005",
                         "--momentum-lookback", "63", "--momentum-top-k", "6", "--time-stop-bars", "60"]),
            ("dn_30",  ["--donchian-period", "30",  "--risk-per-trade", "0.005",
                         "--momentum-lookback", "63", "--momentum-top-k", "6", "--time-stop-bars", "60"]),
            ("dn_55",  ["--donchian-period", "55",  "--risk-per-trade", "0.005",
                         "--momentum-lookback", "63", "--momentum-top-k", "6", "--time-stop-bars", "60"]),
        ],
        "buy_types": [
            ("buy_all_6",   ["--buy-types", "T1,T1P,T2,T2S,T3A,T3B", "--bs-type", "1,1p,2,2s,3a,3b",
                              "--risk-per-trade", "0.005", "--time-stop-bars", "60",
                              "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
            ("buy_T1_T3",   ["--buy-types", "T1,T1P,T3A,T3B", "--bs-type", "1,1p,3a,3b",
                              "--risk-per-trade", "0.005", "--time-stop-bars", "60",
                              "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
            ("buy_T3_only", ["--buy-types", "T3A,T3B", "--bs-type", "3a,3b",
                              "--risk-per-trade", "0.005", "--time-stop-bars", "60",
                              "--momentum-lookback", "63", "--momentum-top-k", "6", "--donchian-period", "20"]),
        ],
    }

    if args.sweep == "all":
        run_keys = ["sizing", "time_stop", "momentum", "donchian", "buy_types"]
    else:
        run_keys = [args.sweep]

    # Override base buy_types when not running the buy_types sweep, since BASE_FLAGS already sets them.
    # The sweep cells override via flags.
    for sweep_key in run_keys:
        cells = {}
        for cell_name, cell_flags in sweeps[sweep_key]:
            out_dir = f"{args.out_base}/{sweep_key}_{args.universe}/{cell_name}"
            # Build clean flag list — remove any base flags that the cell overrides
            override_keys = set()
            i = 0
            while i < len(cell_flags):
                if cell_flags[i].startswith("--"):
                    override_keys.add(cell_flags[i])
                i += 1

            # Rebuild base without overridden flags
            base_clean = []
            i = 0
            while i < len(BASE_FLAGS):
                f = BASE_FLAGS[i]
                if f in override_keys:
                    # skip this flag and its arg
                    if i + 1 < len(BASE_FLAGS) and not BASE_FLAGS[i + 1].startswith("--"):
                        i += 2
                    else:
                        i += 1
                    continue
                base_clean.append(f)
                i += 1

            cmd = [
                sys.executable, "scripts/run_chan_daily_etf_backtest.py",
                "--symbols", *universe,
                "--periods", *periods,
                *base_clean,
                *cell_flags,
                "--out-dir", out_dir,
            ]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            log.info("Running %s/%s ...", sweep_key, cell_name)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                log.warning("FAIL %s: %s", cell_name, r.stderr[-300:])
                continue
            cells[cell_name] = {p: json.loads((Path(out_dir) / f"{p}.json").read_text())["summary"]
                                for p in periods if (Path(out_dir) / f"{p}.json").exists()}

        fmt_table(f"{sweep_key.upper()} sweep on {args.universe}-ETF (lag-1, ret%/RDD)", cells, periods)


if __name__ == "__main__":
    main()
