#!/usr/bin/env python3
"""Post-fix intraday baseline study: what edge (if any) survives lookahead removal?

After the daily_trend_filter `<=` lookahead bug was fixed, the live config
collapsed from R/DD 2.14 to 0.15 on 2023-25 IS. This study isolates which
component(s) actually carry signal vs noise:

- Filter on/off: was the filter doing real regime gating, or was the entire
  effect the lookahead?
- Family solo: gap_reclaim / nr4 / orb individually — which (if any) has
  edge on its own?
- Combined: matches live config to anchor the comparison.

8 variants × 3 periods = 24 backtests, sequential (~15 min).
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

# Live config (= post-fix orb_vol_2.5 broad250). Variants flip allow_* flags
# and daily_trend_filter on/off; everything else stays at the live config.
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


def variants() -> dict[str, dict]:
    fams = ["allow_gap_reclaim_long", "allow_nr4_breakout", "allow_orb_breakout"]

    def solo(family: str) -> dict:
        return {f: (f == family) for f in fams}

    return {
        # Combined bundle — matches live config; with/without filter.
        "combined_filter_on":  {"daily_trend_filter": True},
        "combined_filter_off": {"daily_trend_filter": False,
                                **{f: True for f in fams}},

        # Each family solo, with daily_trend filter ON (post-fix `<` semantics)
        "gap_only_filter_on":  {"daily_trend_filter": True,  **solo("allow_gap_reclaim_long")},
        "nr4_only_filter_on":  {"daily_trend_filter": True,  **solo("allow_nr4_breakout")},
        "orb_only_filter_on":  {"daily_trend_filter": True,  **solo("allow_orb_breakout")},

        # Each family solo, no daily_trend filter at all
        "gap_only_filter_off": {"daily_trend_filter": False, **solo("allow_gap_reclaim_long")},
        "nr4_only_filter_off": {"daily_trend_filter": False, **solo("allow_nr4_breakout")},
        "orb_only_filter_off": {"daily_trend_filter": False, **solo("allow_orb_breakout")},
    }


def main() -> int:
    universe = json.loads(Path(UNIVERSE_PATH).read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe
    out_dir = Path("results/intraday_baseline_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} syms)")
    print("Post-fix backtester (daily_trend uses `<` not `<=`)\n")

    rows = []
    for variant_name, overrides in variants().items():
        cfg = replace(BASE, **overrides)
        bt = IntradayBreakoutBacktester(cfg)
        for period_name, begin, end, db_path in PERIODS:
            if not Path(db_path).exists():
                print(f"  SKIP {variant_name}/{period_name} — {db_path} missing")
                continue
            print(f"  {variant_name:24s} {period_name}", flush=True)
            res = bt.backtest_portfolio(symbols=symbols, begin=begin, end=end, db_path=db_path)
            s = res.summary
            ret = float(s.get("total_return_pct", 0.0))
            dd = float(s.get("max_drawdown_pct", 0.0))
            tr = int(s.get("total_trades", 0))
            wr = float(s.get("win_rate", 0.0)) * 100
            rdd = abs(ret / dd) if dd else 0.0
            rows.append((variant_name, period_name, ret, dd, rdd, tr, wr))
            (out_dir / f"{variant_name}_{period_name}.json").write_text(
                json.dumps({"variant": variant_name, "period": period_name,
                            "summary": s}, indent=2, default=str)
            )

    print()
    print(f"{'variant':25s}{'period':12s}{'ret%':>9s}{'DD%':>7s}{'R/DD':>7s}{'trades':>8s}{'WR%':>6s}")
    print("-" * 74)
    for v, p, ret, dd, rdd, tr, wr in rows:
        print(f"{v:25s}{p:12s}{ret:+9.2f}{dd:7.2f}{rdd:7.2f}{tr:8d}{wr:6.1f}")

    # Pivot: variant × period summary R/DD
    print()
    print("R/DD matrix (key takeaway):")
    print(f"{'variant':25s}{'2023_2025':>12s}{'2020':>10s}{'2018':>10s}")
    print("-" * 57)
    by_var = {}
    for v, p, *_, in rows:
        by_var.setdefault(v, {})[p] = next(r for r in rows if r[0] == v and r[1] == p)
    for v in variants().keys():
        cells = []
        for p in ("2023_2025", "2020", "2018"):
            row = by_var.get(v, {}).get(p)
            if row is None:
                cells.append(f"{'--':>10s}")
            else:
                rdd = row[4]
                cells.append(f"{rdd:>10.2f}")
        print(f"{v:25s}" + "".join(cells))

    return 0


if __name__ == "__main__":
    sys.exit(main())
