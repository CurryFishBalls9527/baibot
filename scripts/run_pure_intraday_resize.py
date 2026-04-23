#!/usr/bin/env python3
"""Phase B add-on: bump max_position_pct to 0.20 and max_positions to 6.

Re-runs baseline_nr4_gap12 + pure_all_with_orb on all 3 periods to test
whether the pure-bundle DD blowout is a sizing artifact or a correlation
issue intrinsic to the family bundle.
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

# Same as run_pure_intraday_validation BASE_CONFIG but with bumped sizing.
BASE_CONFIG = IntradayBacktestConfig(
    initial_cash=100_000.0,
    max_positions=6,                      # was 4
    max_position_pct=0.20,                # was 0.08
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
    allow_gap_reclaim_long=False,
    allow_nr4_breakout=False,
    allow_orb_breakout=False,
    gap_reclaim_min_gap_down_pct=0.012,
    gap_reclaim_max_gap_down_pct=0.06,
    gap_reclaim_min_reclaim_fraction=0.5,
    gap_reclaim_min_volume_ratio=1.3,
    gap_reclaim_earliest_entry_bar=2,
    gap_reclaim_latest_entry_bar=8,
    nr4_lookback_days=4,
    nr4_earliest_entry_bar=1,
    nr4_latest_entry_bar=12,
    nr4_min_volume_ratio=2.0,
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
)

BUNDLES = {
    "baseline_nr4_gap12_resized": dict(
        allow_gap_reclaim_long=True,
        allow_nr4_breakout=True,
    ),
    "pure_all_with_orb_resized": dict(
        allow_expansion_setup=True,
        allow_pullback_vwap=True,
        allow_continuation_setup=True,
        allow_overextended_setup=True,
        allow_orb_breakout=True,
    ),
}


def main() -> int:
    universe = json.loads(Path("research_data/spike_universe.json").read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe
    out_dir = Path("results/intraday_pure")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bundle_name, overrides in BUNDLES.items():
        for period_name, begin, end, db_path in PERIODS:
            cfg = replace(BASE_CONFIG, **overrides)
            bt = IntradayBreakoutBacktester(cfg)
            print(f"→ {bundle_name} / {period_name}", flush=True)
            res = bt.backtest_portfolio(symbols, begin, end, db_path)
            s = res.summary
            print(
                f"  return {s['total_return_pct']:+7.2f}%  dd {s['max_drawdown_pct']:>6.2f}%  "
                f"trades {s['total_trades']:>4}  win {s['win_rate']*100:>5.1f}%",
                flush=True,
            )
            (out_dir / f"{bundle_name}_{period_name}.json").write_text(
                json.dumps({"bundle": bundle_name, "period": period_name, "summary": s,
                            "config": asdict(cfg),
                            "setup_summary": res.setup_summary.to_dict("records") if not res.setup_summary.empty else []},
                           indent=2, default=str)
            )
            rows.append({"bundle": bundle_name, "period": period_name, **{k: s.get(k) for k in
                         ("total_return_pct", "max_drawdown_pct", "total_trades", "win_rate")}})

    print()
    print(f"{'bundle':<32}{'period':<12}{'ret%':>8}{'DD%':>7}{'R/DD':>7}{'trades':>8}")
    print("-" * 80)
    for r in rows:
        rdd = r["total_return_pct"] / abs(r["max_drawdown_pct"]) if r["max_drawdown_pct"] else 0
        print(f"{r['bundle']:<32}{r['period']:<12}{r['total_return_pct']:>+8.2f}{r['max_drawdown_pct']:>7.2f}{rdd:>7.2f}{r['total_trades']:>8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
