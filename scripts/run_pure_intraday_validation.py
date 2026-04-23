#!/usr/bin/env python3
"""Phase B: OOS validation for pure-intraday bundle vs current NR4+gap12 baseline.

Runs 4 bundles × 3 periods (2023-2025 IS, 2020 OOS, 2018 OOS), each on the
51-symbol spike universe at 15m bars with the live `intraday_mechanical`
execution params (cash=100k, max_positions=4, position_pct=8%, stop=3%,
trail=4%, exec costs=1bps half-spread + 5bps stop slippage).

Bundles:
  baseline_nr4_gap12     — gap_reclaim_long + nr4_breakout (current live)
  pure_expansion_pullback — opening_drive_expansion + pullback_vwap
  pure_all_no_orb         — + continuation + overextended
  pure_all_with_orb       — + orb_breakout

Sequential to respect the 3-concurrent backtest cap and keep disk I/O sane.
Output: results/intraday_pure/{bundle}_{period}.json
"""

from __future__ import annotations

import json
import sys
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


# Shared base config — matches live `intraday_mechanical` variant in
# experiments/paper_launch_v2.yaml. Bundles override only the allow_* flags.
BASE_CONFIG = IntradayBacktestConfig(
    initial_cash=100_000.0,
    max_positions=4,
    max_position_pct=0.08,
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
    # Default bundle gates — overridden below per bundle
    allow_continuation_setup=False,
    allow_overextended_setup=False,
    allow_expansion_setup=False,
    allow_pullback_vwap=False,
    allow_gap_reclaim_long=False,
    allow_nr4_breakout=False,
    allow_orb_breakout=False,
    # Gap reclaim params (live)
    gap_reclaim_min_gap_down_pct=0.012,
    gap_reclaim_max_gap_down_pct=0.06,
    gap_reclaim_min_reclaim_fraction=0.5,
    gap_reclaim_min_volume_ratio=1.3,
    gap_reclaim_earliest_entry_bar=2,
    gap_reclaim_latest_entry_bar=8,
    # NR4 params (live)
    nr4_lookback_days=4,
    nr4_earliest_entry_bar=1,
    nr4_latest_entry_bar=12,
    nr4_min_volume_ratio=2.0,
    # ORB params (new — defaults from plan)
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
    "baseline_nr4_gap12": dict(
        allow_gap_reclaim_long=True,
        allow_nr4_breakout=True,
    ),
    "pure_expansion_pullback": dict(
        allow_expansion_setup=True,
        allow_pullback_vwap=True,
    ),
    "pure_all_no_orb": dict(
        allow_expansion_setup=True,
        allow_pullback_vwap=True,
        allow_continuation_setup=True,
        allow_overextended_setup=True,
    ),
    "pure_all_with_orb": dict(
        allow_expansion_setup=True,
        allow_pullback_vwap=True,
        allow_continuation_setup=True,
        allow_overextended_setup=True,
        allow_orb_breakout=True,
    ),
}


def main() -> int:
    universe_path = Path("research_data/spike_universe.json")
    universe = json.loads(universe_path.read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe

    out_dir = Path("results/intraday_pure")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for bundle_name, overrides in BUNDLES.items():
        for period_name, begin, end, db_path in PERIODS:
            cfg = replace(BASE_CONFIG, **overrides)
            bt = IntradayBreakoutBacktester(cfg)
            print(f"\n→ {bundle_name} / {period_name}  ({db_path})", flush=True)
            try:
                result = bt.backtest_portfolio(symbols, begin, end, db_path)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                continue
            s = result.summary
            print(
                f"  return {s['total_return_pct']:+7.2f}%  "
                f"dd {s['max_drawdown_pct']:>6.2f}%  "
                f"trades {s['total_trades']:>4}  "
                f"win {s['win_rate']*100:>5.1f}%",
                flush=True,
            )
            payload = {
                "bundle": bundle_name,
                "period": period_name,
                "summary": s,
                "config": asdict(cfg),
                "setup_summary": result.setup_summary.to_dict("records") if not result.setup_summary.empty else [],
                "symbol_summary": result.symbol_summary.to_dict("records") if not result.symbol_summary.empty else [],
            }
            (out_dir / f"{bundle_name}_{period_name}.json").write_text(json.dumps(payload, indent=2, default=str))
            summary_rows.append({
                "bundle": bundle_name,
                "period": period_name,
                **{k: s.get(k) for k in ("total_return_pct", "max_drawdown_pct", "total_trades", "win_rate", "avg_bars_held")},
            })

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2, default=str))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
