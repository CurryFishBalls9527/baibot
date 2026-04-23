#!/usr/bin/env python3
"""Q3 from `memory/project_intraday_baseline_study.md`: does gap-FADE
(no reclaim requirement) work across regimes where gap_reclaim_long fails?

Hypothesis: 2020 was high-vol with big intraday mean-reversion on gaps.
gap_reclaim required a half-gap-close before entry (momentum), so it
caught reversals that were already fading their fades. Pure fade fires
on a fixed bar inside the gap-down session unconditionally — bets the
gap closes regardless of in-session price action.

Variants (all on broad250 @ 15m, post-fix backtester):
- baseline: existing gap_reclaim (reclaim 0.5, above_open required)
- fade_bar2: disable reclaim check, fire on bar 2, still require close > open
- fade_bar4: disable reclaim check, fire on bar 4 (mid window)
- fade_bar4_pure: disable reclaim, fire on bar 4, NO above-open requirement

Same gap range (1.2-6%) and min_volume (1.3) across all variants so the
only difference is the reclaim-check + entry-bar logic.
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

BASE = IntradayBacktestConfig(
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
    allow_nr4_breakout=False,
    allow_orb_breakout=False,
    gap_reclaim_min_gap_down_pct=0.012,
    gap_reclaim_max_gap_down_pct=0.06,
    gap_reclaim_min_reclaim_fraction=0.5,
    gap_reclaim_min_volume_ratio=1.3,
    gap_reclaim_earliest_entry_bar=2,
    gap_reclaim_latest_entry_bar=8,
    gap_reclaim_require_above_session_open=True,
    gap_reclaim_disable_reclaim_check=False,
    gap_reclaim_fade_entry_bar=4,
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
    market_context_min_score=None,
    execution_half_spread_bps=1.0,
    execution_stop_slippage_bps=5.0,
    daily_db_path="/tmp/market_data_snapshot.duckdb",
)

VARIANTS = {
    "baseline":       {"gap_reclaim_disable_reclaim_check": False},
    "fade_bar2":      {"gap_reclaim_disable_reclaim_check": True,
                       "gap_reclaim_fade_entry_bar": 2,
                       "gap_reclaim_require_above_session_open": True},
    "fade_bar4":      {"gap_reclaim_disable_reclaim_check": True,
                       "gap_reclaim_fade_entry_bar": 4,
                       "gap_reclaim_require_above_session_open": True},
    "fade_bar4_pure": {"gap_reclaim_disable_reclaim_check": True,
                       "gap_reclaim_fade_entry_bar": 4,
                       "gap_reclaim_require_above_session_open": False},
}


def main() -> int:
    universe = json.loads(Path(UNIVERSE_PATH).read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe
    out_dir = Path("results/intraday_gap_fade_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} syms)")
    print("Post-fix backtester + gap-FADE variants\n")

    rows = []
    for variant_name, overrides in VARIANTS.items():
        cfg = replace(BASE, **overrides)
        bt = IntradayBreakoutBacktester(cfg)
        for period_name, begin, end, db_path in PERIODS:
            if not Path(db_path).exists():
                print(f"  SKIP {variant_name}/{period_name}")
                continue
            tag = f"{variant_name}__{period_name}"
            print(f"  {tag}", flush=True)
            res = bt.backtest_portfolio(symbols=symbols, begin=begin, end=end, db_path=db_path)
            s = res.summary
            ret = float(s.get("total_return_pct", 0.0))
            dd = float(s.get("max_drawdown_pct", 0.0))
            tr = int(s.get("total_trades", 0))
            wr = float(s.get("win_rate", 0.0)) * 100
            rdd = abs(ret / dd) if dd else 0.0
            rows.append((variant_name, period_name, ret, dd, rdd, tr, wr))
            (out_dir / f"{tag}.json").write_text(
                json.dumps({"variant": variant_name, "period": period_name,
                            "summary": s}, indent=2, default=str)
            )

    print()
    print(f"{'variant':16s}{'period':12s}{'ret%':>9s}{'DD%':>7s}{'R/DD':>7s}{'trades':>8s}{'WR%':>6s}")
    print("-" * 65)
    for v, p, ret, dd, rdd, tr, wr in rows:
        print(f"{v:16s}{p:12s}{ret:+9.2f}{dd:7.2f}{rdd:7.2f}{tr:8d}{wr:6.1f}")

    print()
    print("Signed R/DD pivot (+ = profit, − = loss):")
    print(f"  {'variant':16s}{'2023_2025':>12s}{'2020':>10s}{'2018':>10s}")
    for v in VARIANTS:
        cells = []
        for p in ("2023_2025", "2020", "2018"):
            hit = next((r for r in rows if r[0] == v and r[1] == p), None)
            if hit:
                signed = hit[4] if hit[2] >= 0 else -hit[4]
                cells.append(f"{signed:>+10.2f}")
            else:
                cells.append(f"{'--':>10s}")
        print(f"  {v:16s}" + "".join(cells))

    print("\nTrade counts:")
    print(f"  {'variant':16s}{'2023_2025':>12s}{'2020':>10s}{'2018':>10s}")
    for v in VARIANTS:
        cells = []
        for p in ("2023_2025", "2020", "2018"):
            hit = next((r for r in rows if r[0] == v and r[1] == p), None)
            cells.append(f"{hit[5]:>10d}" if hit else f"{'--':>10s}")
        print(f"  {v:16s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
