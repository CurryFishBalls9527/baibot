#!/usr/bin/env python3
"""Re-run baseline NR4+gap on the 250-symbol broadened universe.

Goal: measure trade-frequency lift vs the 52-symbol spike universe at the
shipped (resized) sizing of 6 positions x 20%.
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

UNIVERSES = {
    "spike52": "research_data/spike_universe.json",
    "broad250": "research_data/intraday_top250_universe.json",
}

# Mirror live intraday_mechanical post-resize sizing.
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
    require_above_vwap=True,
    flatten_at_close=True,
    interval_minutes=15,
    daily_trend_filter=True,
    daily_trend_sma=20,
    execution_half_spread_bps=1.0,
    execution_stop_slippage_bps=5.0,
)


def load_symbols(path: str) -> list[str]:
    d = json.loads(Path(path).read_text())
    return d["symbols"] if isinstance(d, dict) else d


def main() -> int:
    out_dir = Path("results/intraday_broad")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for univ_name, univ_path in UNIVERSES.items():
        symbols = load_symbols(univ_path)
        for period_name, begin, end, db_path in PERIODS:
            cfg = replace(BASE_CONFIG)
            bt = IntradayBreakoutBacktester(cfg)
            print(f"-> {univ_name} ({len(symbols)} syms) / {period_name}", flush=True)
            res = bt.backtest_portfolio(symbols, begin, end, db_path)
            s = res.summary
            print(
                f"  return {s['total_return_pct']:+7.2f}%  "
                f"dd {s['max_drawdown_pct']:>6.2f}%  "
                f"trades {s['total_trades']:>4}  "
                f"win {s['win_rate']*100:>5.1f}%",
                flush=True,
            )
            (out_dir / f"{univ_name}_{period_name}.json").write_text(
                json.dumps({
                    "universe": univ_name,
                    "universe_size": len(symbols),
                    "period": period_name,
                    "summary": s,
                    "config": asdict(cfg),
                    "setup_summary": res.setup_summary.to_dict("records") if not res.setup_summary.empty else [],
                }, indent=2, default=str)
            )
            rows.append({
                "universe": univ_name,
                "u_size": len(symbols),
                "period": period_name,
                **{k: s.get(k) for k in ("total_return_pct", "max_drawdown_pct", "total_trades", "win_rate")},
            })

    print()
    print(f"{'universe':<10}{'size':>5}  {'period':<10}{'ret%':>8}{'DD%':>7}{'R/DD':>7}{'trades':>8}{'tr/yr':>8}")
    print("-" * 70)
    period_years = {"2023_2025": 3.0, "2020": 1.0, "2018": 1.0}
    for r in rows:
        rdd = r["total_return_pct"] / abs(r["max_drawdown_pct"]) if r["max_drawdown_pct"] else 0
        yrs = period_years.get(r["period"], 1.0)
        tr_per_yr = r["total_trades"] / yrs
        print(
            f"{r['universe']:<10}{r['u_size']:>5}  {r['period']:<10}"
            f"{r['total_return_pct']:>+8.2f}{r['max_drawdown_pct']:>7.2f}"
            f"{rdd:>7.2f}{r['total_trades']:>8d}{tr_per_yr:>8.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
