#!/usr/bin/env python3
"""Re-run live intraday config (orb_vol_2.5 broad250) with bias fixes applied.

Compares post-fix R/DD against the pre-fix numbers cached in
results/intraday_orb_tight/orb_vol_2.5_*.json. Only Bug 1 (daily-trend lookahead)
affects this run — Bug 2 (tradability filter) requires --apply-tradability-filter
in the periods runner, which orb_tight doesn't use.
"""
from __future__ import annotations
import json
import sys
from dataclasses import replace
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

LIVE_CONFIG = IntradayBacktestConfig(
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
    allow_orb_breakout=True,
    gap_reclaim_min_gap_down_pct=0.012,
    gap_reclaim_max_gap_down_pct=0.06,
    gap_reclaim_min_reclaim_fraction=0.5,
    gap_reclaim_min_volume_ratio=1.3,
    gap_reclaim_earliest_entry_bar=2,
    gap_reclaim_latest_entry_bar=8,
    nr4_lookback_days=4,
    nr4_earliest_entry_bar=1,
    nr4_latest_entry_bar=12,
    nr4_min_volume_ratio=1.3,
    orb_range_bars=2,
    orb_min_volume_ratio=2.5,
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

PRE_FIX = {
    "2023_2025": {"ret": 90.09, "dd": 42.02, "trades": 2329, "wr": 52.4},
    "2020":      {"ret": 12.02, "dd": 23.87, "trades": 738,  "wr": 46.5},
    "2018":      {"ret": 18.86, "dd": 21.35, "trades": 546,  "wr": 52.8},
}


def main() -> int:
    universe = json.loads(Path(UNIVERSE_PATH).read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe
    out_dir = Path("results/intraday_bias_fix")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} syms)")
    print(f"Config: orb_vol_2.5 (live), bias fixes applied (daily trend < trend_day)\n")

    bt = IntradayBreakoutBacktester(LIVE_CONFIG)

    rows = []
    for period_name, begin, end, db_path in PERIODS:
        if not Path(db_path).exists():
            print(f"  SKIP {period_name} — {db_path} missing")
            continue
        print(f"  Running {period_name} ({begin} → {end})...", flush=True)
        result = bt.backtest_portfolio(symbols=symbols, begin=begin, end=end, db_path=db_path)
        s = result.summary
        ret = float(s.get("total_return_pct", 0.0))
        dd = float(s.get("max_drawdown_pct", 0.0))
        tr = int(s.get("total_trades", 0))
        wr = float(s.get("win_rate", 0.0)) * 100
        rdd = abs(ret / dd) if dd else 0.0
        rows.append((period_name, ret, dd, rdd, tr, wr))
        out_path = out_dir / f"orb_vol_2.5_postfix_{period_name}.json"
        out_path.write_text(json.dumps({
            "period": period_name, "begin": begin, "end": end,
            "summary": s,
        }, indent=2, default=str))

    print()
    print(f"{'period':12s} | {'pre-fix':35s} | {'post-fix':35s} | delta")
    print(f"{'':12s} | {'ret%/DD%/RDD/trades':35s} | {'ret%/DD%/RDD/trades':35s} |")
    print("-" * 110)
    for name, ret, dd, rdd, tr, wr in rows:
        pre = PRE_FIX.get(name, {})
        pre_rdd = abs(pre['ret']/pre['dd']) if pre.get('dd') else 0
        pre_str = f"{pre.get('ret',0):+6.1f}/{pre.get('dd',0):5.1f}/{pre_rdd:5.2f}/{pre.get('trades',0):5d}"
        post_str = f"{ret:+6.1f}/{dd:5.1f}/{rdd:5.2f}/{tr:5d}"
        d_ret = ret - pre.get('ret', 0)
        d_rdd = rdd - pre_rdd
        d_tr = tr - pre.get('trades', 0)
        delta_str = f"Δret={d_ret:+5.1f}pp Δrdd={d_rdd:+5.2f} Δtrades={d_tr:+5d}"
        print(f"{name:12s} | {pre_str:35s} | {post_str:35s} | {delta_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
