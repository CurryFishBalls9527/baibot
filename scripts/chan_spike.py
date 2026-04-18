#!/usr/bin/env python3
"""Measurement spike: run chan.py on 30-min bars and count signals.

Four numbers we're trying to nail down:

1. Signal density — buy/sell points per week per symbol, split by bsp type
2. is_sure rate — fraction of signals that become confirmed vs stay provisional
3. Churn rate — fraction of provisional signals that get retracted/shifted
   within the next 10 bars (direct measure of lookahead risk)
4. step_load speed — seconds per symbol, predicts cost of a full engine

Usage:
    python scripts/chan_spike.py --symbols AAPL MSFT NVDA --limit 5
    python scripts/chan_spike.py --universe research_data/spike_universe.json
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# chan.py root must be on sys.path for top-level imports like `from Common...`
CHAN_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "chan.py"
sys.path.insert(0, str(CHAN_ROOT))

from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import AUTYPE, KL_TYPE  # noqa: E402

from tradingagents.research.chan_adapter import DuckDBIntradayAPI  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_spike")


def _bsp_signature(bsp) -> tuple:
    """Stable identifier for a bsp — klu index + types + is_buy."""
    return (
        bsp.klu.idx,
        bsp.is_buy,
        tuple(sorted(t.name for t in bsp.type)),
    )


def run_symbol(
    symbol: str,
    begin: str,
    end: str,
    db_path: str,
    churn_window: int = 10,
) -> dict:
    """Walk one symbol bar-by-bar and collect signal stats."""
    DuckDBIntradayAPI.DB_PATH = db_path

    config = CChanConfig({
        "trigger_step": True,
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 1,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": "1,2,3a,1p,2s,3b",
        "print_warning": False,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=symbol,
        begin_time=begin,
        end_time=end,
        data_src="custom:DuckDBAPI.DuckDB30mAPI",
        lv_list=[KL_TYPE.K_30M],
        config=config,
        autype=AUTYPE.QFQ,
    )

    t0 = time.perf_counter()
    steps = 0
    first_seen: dict[tuple, dict] = {}
    bars_sure_at_first: list[bool] = []
    retracted = 0
    survived = 0
    sure_upgrades = 0
    by_type: Counter = Counter()
    by_type_sure: Counter = Counter()
    previous_sigs: set = set()

    for snapshot in chan.step_load():
        steps += 1
        try:
            bsp_list = snapshot.get_latest_bsp(idx=0, number=200)
        except Exception:
            bsp_list = []

        current_sigs: set = set()
        for bsp in bsp_list:
            sig = _bsp_signature(bsp)
            current_sigs.add(sig)
            if sig not in first_seen:
                for t in bsp.type:
                    by_type[t.name] += 1
                    if bsp.bi.is_sure:
                        by_type_sure[t.name] += 1
                first_seen[sig] = {
                    "step": steps,
                    "is_sure": bsp.bi.is_sure,
                }
            else:
                if not first_seen[sig]["is_sure"] and bsp.bi.is_sure:
                    sure_upgrades += 1
                    first_seen[sig]["is_sure"] = True

        dropped = previous_sigs - current_sigs
        for sig in dropped:
            seen = first_seen.get(sig)
            if seen and (steps - seen["step"]) <= churn_window:
                retracted += 1
        previous_sigs = current_sigs

    for sig, seen in first_seen.items():
        if sig in previous_sigs:
            survived += 1

    elapsed = time.perf_counter() - t0
    total_sigs = len(first_seen)
    provisional = sum(1 for v in first_seen.values() if not v["is_sure"])
    sure_now = total_sigs - provisional

    return {
        "symbol": symbol,
        "bars_consumed": steps,
        "elapsed_sec": round(elapsed, 2),
        "total_unique_signals": total_sigs,
        "sure_at_first_sight": sum(1 for v in first_seen.values() if v["is_sure"]),
        "provisional_at_first_sight": sum(1 for v in first_seen.values() if not v["is_sure"]),
        "sure_upgrades": sure_upgrades,
        "final_sure_count": sure_now,
        "final_provisional_count": provisional,
        "retracted_within_window": retracted,
        "survived_to_end": survived,
        "by_type": dict(by_type),
        "by_type_sure_first_sight": dict(by_type_sure),
    }


def parse_args():
    p = argparse.ArgumentParser(description="chan.py measurement spike")
    p.add_argument("--universe", default="research_data/spike_universe.json")
    p.add_argument("--symbols", nargs="*", help="Override universe with explicit symbols")
    p.add_argument("--limit", type=int, default=None, help="Only run first N symbols")
    p.add_argument("--begin", default="2023-01-01")
    p.add_argument("--end", default="2025-12-30")
    p.add_argument("--db", default="research_data/intraday_30m.duckdb")
    p.add_argument("--out", default="results/chan_spike/metrics.json")
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

    log.info("Running chan.py spike on %d symbols (%s to %s)", len(symbols), args.begin, args.end)

    all_metrics = []
    t_total = time.perf_counter()
    for i, sym in enumerate(symbols, 1):
        try:
            m = run_symbol(sym, args.begin, args.end, args.db)
            all_metrics.append(m)
            log.info(
                "  [%d/%d] %-6s %d bars  %d uniq sigs  %d sure-first  %d retracted  %.1fs",
                i, len(symbols), sym,
                m["bars_consumed"], m["total_unique_signals"],
                m["sure_at_first_sight"], m["retracted_within_window"],
                m["elapsed_sec"],
            )
        except Exception as e:
            log.warning("  [%d/%d] %-6s FAILED: %s", i, len(symbols), sym, e)
    total_elapsed = time.perf_counter() - t_total

    agg = {
        "symbols_total": len(symbols),
        "symbols_succeeded": len(all_metrics),
        "wall_clock_sec": round(total_elapsed, 2),
        "total_bars": sum(m["bars_consumed"] for m in all_metrics),
        "total_unique_signals": sum(m["total_unique_signals"] for m in all_metrics),
        "total_sure_first_sight": sum(m["sure_at_first_sight"] for m in all_metrics),
        "total_provisional_first_sight": sum(m["provisional_at_first_sight"] for m in all_metrics),
        "total_retracted": sum(m["retracted_within_window"] for m in all_metrics),
        "total_sure_upgrades": sum(m["sure_upgrades"] for m in all_metrics),
        "by_type": dict(_sum_counters(m["by_type"] for m in all_metrics)),
        "by_type_sure_first_sight": dict(_sum_counters(
            m["by_type_sure_first_sight"] for m in all_metrics
        )),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"aggregate": agg, "per_symbol": all_metrics}, indent=2))

    _print_summary(agg, all_metrics, args)
    log.info("Saved metrics to %s", out_path)


def _sum_counters(counters):
    result: Counter = Counter()
    for c in counters:
        for k, v in c.items():
            result[k] += v
    return result


def _print_summary(agg, all_metrics, args):
    print()
    print("=" * 70)
    print("  CHAN.PY SPIKE — METRICS")
    print("=" * 70)
    print(f"  Symbols:                   {agg['symbols_succeeded']}/{agg['symbols_total']}")
    print(f"  Period:                    {args.begin} to {args.end}")
    print(f"  Total 30m bars consumed:   {agg['total_bars']:,}")
    print(f"  Wall clock:                {agg['wall_clock_sec']}s")
    if agg["symbols_succeeded"]:
        print(f"  Avg sec/symbol:            {agg['wall_clock_sec'] / agg['symbols_succeeded']:.2f}")
    print()
    print(f"  Total unique signals:      {agg['total_unique_signals']:,}")
    print(f"  Sure at first sight:       {agg['total_sure_first_sight']:,}")
    print(f"  Provisional at first:      {agg['total_provisional_first_sight']:,}")
    print(f"  Later upgraded to sure:    {agg['total_sure_upgrades']:,}")
    print(f"  Retracted (<=10 bars):     {agg['total_retracted']:,}")
    if agg["total_provisional_first_sight"]:
        churn_rate = agg["total_retracted"] / agg["total_provisional_first_sight"] * 100
        print(f"  Provisional churn rate:    {churn_rate:.1f}%")
    if agg["total_unique_signals"]:
        sure_rate = agg["total_sure_first_sight"] / agg["total_unique_signals"] * 100
        print(f"  Sure-at-first rate:        {sure_rate:.1f}%")

    print()
    print("  Signals by bsp type (unique, first sight):")
    for k, v in sorted(agg["by_type"].items(), key=lambda x: -x[1]):
        sure_count = agg["by_type_sure_first_sight"].get(k, 0)
        print(f"    {k:<6} {v:>8,}  (sure: {sure_count:,})")

    if agg["symbols_succeeded"]:
        total_weeks = (
            agg["total_bars"]
            / (13 * 5)  # ~13 bars/day * 5 trading days = 65 bars/week
            / agg["symbols_succeeded"]
        )
        if total_weeks > 0:
            per_wk_sym = agg["total_unique_signals"] / agg["symbols_succeeded"] / total_weeks
            sure_per_wk_sym = agg["total_sure_first_sight"] / agg["symbols_succeeded"] / total_weeks
            print()
            print(f"  Signals per week per symbol:    {per_wk_sym:.2f}")
            print(f"  Sure signals per week per sym:  {sure_per_wk_sym:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
