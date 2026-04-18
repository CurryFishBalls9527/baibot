#!/usr/bin/env python3
"""Ablation sweep over chan.py config parameters.

Tests parameter changes incrementally against the baseline across multiple
periods. Each config builds on the previous one to isolate the contribution
of each change.

Usage:
    python scripts/chan_param_sweep.py --limit 10          # quick 10-symbol test
    python scripts/chan_param_sweep.py                      # full 51-symbol sweep
    python scripts/chan_param_sweep.py --configs baseline divergence  # specific configs
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
CHAN_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "chan.py"
sys.path.insert(0, str(CHAN_ROOT))

from scripts.chan_spike_backtest import run_backtest  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_sweep")

# NOTE (C-BE, 2026-04-17): breakeven_atr / trail_tighten_atr / trail_tighten_mult
# are exit-side params that live in ChanBacktestConfig (chan_backtester.py), NOT
# in CChanConfig. The `overrides` dict here is passed to CChanConfig only, so
# adding those keys here is a no-op. A proper C-BE sweep would need a separate
# harness that invokes `chan_backtester.py` directly with varying BacktestConfig
# values on the 2018 + 2023_25 periods. Deferred until Track C becomes priority.

CONFIGS = {
    "baseline": {},
    "divergence": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
    },
    "div+drop_T3A": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s,3b",
    },
    "div+T3A+bs2_786": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s,3b",
        "max_bs2_rate": 0.786,
    },
    "div+T3A+bs2_618": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s,3b",
        "max_bs2_rate": 0.618,
    },
    "div+T3A+zs2": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s,3b",
        "min_zs_cnt": 2,
    },
    "full_stack": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s,3b",
        "max_bs2_rate": 0.786,
        "min_zs_cnt": 2,
    },
    "full_stack_618": {
        "divergence_rate": 0.8,
        "macd_algo": "area",
        "bs_type": "1,1p,2,2s,3b",
        "max_bs2_rate": 0.618,
        "min_zs_cnt": 2,
    },
}

PERIODS = {
    "2018_flat": {
        "begin": "2018-01-01",
        "end": "2018-12-31",
        "db": "research_data/intraday_30m_2018.duckdb",
    },
    "2020_crash": {
        "begin": "2020-01-01",
        "end": "2020-12-31",
        "db": "research_data/intraday_30m_2020.duckdb",
    },
    "2023_25_bull": {
        "begin": "2023-01-01",
        "end": "2025-12-30",
        "db": "research_data/intraday_30m.duckdb",
    },
}


def _buy_types_from_config(overrides: dict) -> set:
    bs_type = overrides.get("bs_type", "1,2,3a,1p,2s,3b")
    mapping = {"1": "T1", "1p": "T1P", "2": "T2", "2s": "T2S", "3a": "T3A", "3b": "T3B"}
    return {mapping[t.strip()] for t in bs_type.split(",")}


def run_sweep(symbols: list[str], config_names: list[str], period_names: list[str], exit_mode: str = "atr_trail") -> list[dict]:
    results = []
    total_combos = len(config_names) * len(period_names)
    combo = 0

    for cfg_name in config_names:
        overrides = CONFIGS[cfg_name]
        buy_types = _buy_types_from_config(overrides)

        for period_name in period_names:
            combo += 1
            period = PERIODS[period_name]

            if not Path(period["db"]).exists():
                log.warning("  [%d/%d] %s × %s — DB not found, skipping", combo, total_combos, cfg_name, period_name)
                continue

            log.info("[%d/%d] Config=%s  Period=%s  (%d symbols)", combo, total_combos, cfg_name, period_name, len(symbols))

            all_trades = []
            sym_returns = []
            t0 = time.perf_counter()

            for sym in symbols:
                try:
                    r = run_backtest(
                        sym, period["begin"], period["end"], period["db"],
                        buy_types=buy_types,
                        chan_config_overrides=overrides,
                        exit_mode=exit_mode,
                    )
                    all_trades.extend(r["trades"])
                    sym_returns.append(r["compounded_return"])
                except Exception as e:
                    log.warning("    %s FAILED: %s", sym, e)

            elapsed = time.perf_counter() - t0
            total_t = len(all_trades)
            wins = sum(1 for t in all_trades if t["pnl_pct"] > 0)
            losses = total_t - wins

            entry = {
                "config": cfg_name,
                "period": period_name,
                "overrides": overrides,
                "symbols_run": len(sym_returns),
                "total_trades": total_t,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / total_t, 4) if total_t else 0,
                "avg_return": round(sum(t["pnl_pct"] for t in all_trades) / total_t, 6) if total_t else 0,
                "avg_win": round(sum(t["pnl_pct"] for t in all_trades if t["pnl_pct"] > 0) / wins, 6) if wins else 0,
                "avg_loss": round(sum(t["pnl_pct"] for t in all_trades if t["pnl_pct"] <= 0) / losses, 6) if losses else 0,
                "avg_bars_held": round(sum(t["bars_held"] for t in all_trades) / total_t, 1) if total_t else 0,
                "equal_weight_return": round(sum(sym_returns) / len(sym_returns) * 100, 2) if sym_returns else 0,
                "elapsed_sec": round(elapsed, 1),
            }

            by_type = {}
            for t in all_trades:
                for tp in t["bsp_types"].split(","):
                    by_type.setdefault(tp, []).append(t["pnl_pct"])
            entry["by_entry_type"] = {
                tp: {
                    "count": len(v),
                    "avg_ret": round(sum(v) / len(v), 6),
                    "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4),
                }
                for tp, v in sorted(by_type.items())
            }

            by_exit = {}
            for t in all_trades:
                by_exit.setdefault(t["exit_reason"], []).append(t["pnl_pct"])
            entry["by_exit_reason"] = {
                reason: {
                    "count": len(v),
                    "avg_ret": round(sum(v) / len(v), 6),
                    "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4),
                }
                for reason, v in sorted(by_exit.items())
            }

            results.append(entry)
            log.info(
                "    → %d trades  WR %.1f%%  avg %+.3f%%  portfolio %+.1f%%  %.1fs",
                total_t, entry["win_rate"] * 100, entry["avg_return"] * 100,
                entry["equal_weight_return"], elapsed,
            )

    return results


def print_comparison(results: list[dict]):
    periods = sorted(set(r["period"] for r in results))
    configs = []
    seen = set()
    for r in results:
        if r["config"] not in seen:
            configs.append(r["config"])
            seen.add(r["config"])

    lookup = {(r["config"], r["period"]): r for r in results}

    print()
    print("=" * 100)
    print("  CHAN.PY PARAMETER SWEEP — COMPARISON")
    print("=" * 100)

    header = f"  {'Config':<22}"
    for p in periods:
        header += f" | {'Trades':>6} {'WR':>6} {'AvgR':>8} {'Port%':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline_results = {}
    for p in periods:
        r = lookup.get(("baseline", p))
        if r:
            baseline_results[p] = r

    for cfg in configs:
        row = f"  {cfg:<22}"
        for p in periods:
            r = lookup.get((cfg, p))
            if r:
                row += f" | {r['total_trades']:>6} {r['win_rate']*100:>5.1f}% {r['avg_return']*100:>+7.3f}% {r['equal_weight_return']:>+6.1f}%"
            else:
                row += f" | {'—':>6} {'—':>6} {'—':>8} {'—':>7}"
        print(row)

    if baseline_results:
        print()
        print("  Δ vs baseline:")
        print("  " + "-" * 60)
        for cfg in configs:
            if cfg == "baseline":
                continue
            row = f"  {cfg:<22}"
            for p in periods:
                r = lookup.get((cfg, p))
                bl = baseline_results.get(p)
                if r and bl:
                    dwr = (r["win_rate"] - bl["win_rate"]) * 100
                    davg = (r["avg_return"] - bl["avg_return"]) * 100
                    dport = r["equal_weight_return"] - bl["equal_weight_return"]
                    row += f" |        {dwr:>+5.1f}pp {davg:>+7.3f}% {dport:>+6.1f}%"
                else:
                    row += f" |        {'—':>6} {'—':>8} {'—':>7}"
            print(row)

    print("=" * 100)


def parse_args():
    p = argparse.ArgumentParser(description="chan.py parameter sweep")
    p.add_argument("--universe", default="research_data/spike_universe.json")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--configs", nargs="*", default=None,
                    help=f"Configs to test (default: all). Options: {list(CONFIGS.keys())}")
    p.add_argument("--periods", nargs="*", default=None,
                    help=f"Periods to test (default: all). Options: {list(PERIODS.keys())}")
    p.add_argument("--exit-mode", default="atr_trail",
                    choices=["atr_trail", "zs_structural"])
    p.add_argument("--out", default="results/chan_params/sweep_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        data = json.loads(Path(args.universe).read_text())
        symbols = data["symbols"] if isinstance(data, dict) else data
    if args.limit:
        symbols = symbols[: args.limit]
    symbols = [s for s in symbols if s != "BRK-B"]

    config_names = args.configs or list(CONFIGS.keys())
    period_names = args.periods or list(PERIODS.keys())

    for c in config_names:
        if c not in CONFIGS:
            log.error("Unknown config: %s. Options: %s", c, list(CONFIGS.keys()))
            sys.exit(1)
    for p in period_names:
        if p not in PERIODS:
            log.error("Unknown period: %s. Options: %s", p, list(PERIODS.keys()))
            sys.exit(1)

    log.info("Sweep: %d configs × %d periods × %d symbols = %d runs",
             len(config_names), len(period_names), len(symbols),
             len(config_names) * len(period_names))

    results = run_sweep(symbols, config_names, period_names, exit_mode=args.exit_mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print_comparison(results)
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
