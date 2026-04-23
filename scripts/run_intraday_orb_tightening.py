#!/usr/bin/env python3
"""Tighten ORB so it survives 2018 without killing 2023-25 alpha.

Baseline enable_orb on broad250:
  2023-25: R/DD 2.38 (baseline 0.88, +170%), +146%, 4.5 tr/day
  2020:    R/DD 0.75 (baseline 0.22, +240%), +32.8%, 4.3 tr/day
  2018:    R/DD 0.23 (baseline 2.43, COLLAPSED), +14.2%, 60% DD, 3.3 tr/day

Each variant keeps allow_orb_breakout=True and changes ONE ORB gate at a time.
Goal: raise 2018 R/DD above ~1.0 without dropping 2023-25 R/DD below baseline.
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.intraday_backtester import (  # noqa: E402
    IntradayBacktestConfig,
    IntradayBreakoutBacktester,
)

PERIODS = [
    ("2023_2025", "2023-01-01", "2025-12-30", "research_data/intraday_15m.duckdb"),
    ("2020", "2020-01-01", "2020-12-31", "research_data/intraday_15m_2020.duckdb"),
    ("2018", "2018-01-01", "2018-12-31", "research_data/intraday_15m_2018.duckdb"),
]

UNIVERSE_PATH = "research_data/intraday_top250_universe.json"

# Live config + enable_orb + the just-shipped nr4_vol_1.3 tweak.
BASE_CONFIG = IntradayBacktestConfig(
    initial_cash=100_000.0,
    max_positions=6,
    max_position_pct=0.20,
    stop_loss_pct=0.03,
    trail_stop_pct=0.04,
    opening_range_bars=2,
    require_above_prior_high=True,
    latest_entry_bar_in_session=20,
    max_trades_per_symbol_per_day=1,
    min_volume_ratio=1.5,
    continuation_min_volume_ratio=2.0,
    continuation_max_distance_from_vwap_pct=0.005,
    continuation_latest_entry_bar_in_session=8,
    allow_continuation_setup=False,
    allow_overextended_setup=False,
    allow_expansion_setup=False,
    allow_pullback_vwap=False,
    allow_gap_reclaim_long=True,
    allow_nr4_breakout=True,
    allow_orb_breakout=True,   # ON for this sweep
    gap_reclaim_min_gap_down_pct=0.012,
    gap_reclaim_max_gap_down_pct=0.06,
    gap_reclaim_min_reclaim_fraction=0.5,
    gap_reclaim_min_volume_ratio=1.3,
    gap_reclaim_earliest_entry_bar=2,
    gap_reclaim_latest_entry_bar=8,
    nr4_lookback_days=4,
    nr4_earliest_entry_bar=1,
    nr4_latest_entry_bar=12,
    nr4_min_volume_ratio=1.3,  # post-ship value
    orb_range_bars=2,
    orb_min_volume_ratio=1.5,
    orb_min_breakout_distance_pct=0.001,
    orb_earliest_entry_bar=2,
    orb_latest_entry_bar=10,
    orb_require_above_vwap=True,
    require_above_vwap=True,
    flatten_at_close=True,
    interval_minutes=15,
    daily_trend_filter=True,
    daily_trend_sma=20,
    execution_half_spread_bps=1.0,
    execution_stop_slippage_bps=5.0,
    daily_db_path="/tmp/market_data_snapshot.duckdb",
)

VARIANTS = {
    "orb_base":            dict(),
    "orb_vol_2.0":         dict(orb_min_volume_ratio=2.0),
    "orb_vol_2.5":         dict(orb_min_volume_ratio=2.5),
    "orb_dist_0.003":      dict(orb_min_breakout_distance_pct=0.003),
    "orb_dist_0.005":      dict(orb_min_breakout_distance_pct=0.005),
    "orb_window_2_5":      dict(orb_latest_entry_bar=5),
    "orb_window_2_4":      dict(orb_latest_entry_bar=4),
    "orb_range_4":         dict(orb_range_bars=4),
}


def main() -> int:
    universe = json.loads(Path(UNIVERSE_PATH).read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe
    out_dir = Path("results/intraday_orb_tight")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} syms)")

    rows = []
    for variant_name, overrides in VARIANTS.items():
        for period_name, begin, end, db_path in PERIODS:
            cfg = replace(BASE_CONFIG, **overrides)
            bt = IntradayBreakoutBacktester(cfg)
            print(f"-> {variant_name} / {period_name}", flush=True)
            res = bt.backtest_portfolio(symbols, begin, end, db_path)
            s = res.summary
            print(
                f"  return {s['total_return_pct']:+7.2f}%  "
                f"dd {s['max_drawdown_pct']:>6.2f}%  "
                f"trades {s['total_trades']:>4}  "
                f"win {s['win_rate']*100:>5.1f}%",
                flush=True,
            )
            (out_dir / f"{variant_name}_{period_name}.json").write_text(
                json.dumps({
                    "variant": variant_name,
                    "period": period_name,
                    "summary": s,
                    "config": asdict(cfg),
                }, indent=2, default=str)
            )
            rows.append({
                "variant": variant_name,
                "period": period_name,
                **{k: s.get(k) for k in ("total_return_pct", "max_drawdown_pct", "total_trades", "win_rate")},
            })

    print()
    print(f"{'variant':<20}{'period':<12}{'ret%':>8}{'DD%':>7}{'R/DD':>7}{'trades':>8}{'tr/day':>8}")
    print("-" * 72)
    period_days = {"2023_2025": 756, "2020": 253, "2018": 251}
    for r in rows:
        rdd = r["total_return_pct"] / abs(r["max_drawdown_pct"]) if r["max_drawdown_pct"] else 0
        days = period_days.get(r["period"], 252)
        tr_per_day = r["total_trades"] / days
        print(
            f"{r['variant']:<20}{r['period']:<12}"
            f"{r['total_return_pct']:>+8.2f}{r['max_drawdown_pct']:>7.2f}"
            f"{rdd:>7.2f}{r['total_trades']:>8d}{tr_per_day:>8.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
