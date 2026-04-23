#!/usr/bin/env python3
"""Does gap_reclaim_long edge transfer to less-mega-liquid stocks?

Open question 1 from `memory/project_intraday_baseline_study.md`: the
post-fix baseline showed gap_reclaim_long is the only family with edge,
but it's fair-weather (R/DD 2.31 IS / 0.54 2020 / 0.76 2018). The
hypothesis: smaller-cap stocks have less HFT/institutional crowding,
so the gap-reclaim signal might survive in non-trending regimes there.

**Universe construction caveat (READ BEFORE INTERPRETING)**:
- We use the 489 symbols with 30m bars in ALL THREE period DBs
  (2018/2020/2023-25), so the universe is heavily survival-biased
  toward names that existed and were liquid throughout 2017-2025.
- We can't get a clean true "$500M-$5B mid-cap" universe from this set;
  even bottom-quartile $-volume here is $22M-$157M/day, mid-cap-ish.
- Stratification is by **2025** daily $-volume — forward-looking when
  applied to 2018/2020. Symmetric across both buckets, so the
  bottom-vs-top comparison is apples-to-apples even if the absolute
  numbers may overstate edge from survivor cherry-picking.
- 30m bars used (matches broad DB granularity); live uses 15m. If a
  bottom bucket signal looks promising, RE-VALIDATE at 15m on a
  freshly-pulled small-cap universe before any live config change.

Variants:
- gap_only_filter_on  (= the post-fix-edge candidate) on smallmid
- gap_only_filter_on on megaliquid (sanity check vs broad250 baseline)
- combined_filter_off on both (no-filter sanity floor)
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
    ("2023_2025", "2023-01-01", "2025-12-30", "research_data/intraday_30m_broad.duckdb"),
    ("2020", "2020-01-01", "2020-12-31", "research_data/intraday_30m_broad_2020.duckdb"),
    ("2018", "2018-01-01", "2018-12-31", "research_data/intraday_30m_broad_2018.duckdb"),
]

UNIVERSES = {
    "smallmid":   "research_data/intraday_smallmid_universe.json",
    "megaliquid": "research_data/intraday_megaliquid_universe.json",
}

# Same as baseline study BUT interval_minutes=30 to match broad DBs
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
    interval_minutes=30,
    daily_trend_filter=True,
    daily_trend_sma=20,
    execution_half_spread_bps=1.0,
    execution_stop_slippage_bps=5.0,
    daily_db_path="/tmp/market_data_snapshot.duckdb",
)


VARIANTS = {
    "gap_only_filter_on":  {"daily_trend_filter": True,  "allow_gap_reclaim_long": True},
    "combined_filter_off": {"daily_trend_filter": False, "allow_gap_reclaim_long": True,
                            "allow_nr4_breakout": True, "allow_orb_breakout": True},
}


def main() -> int:
    out_dir = Path("results/intraday_smallmid_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Post-fix backtester (daily_trend uses `<`)\n")

    rows = []
    for univ_name, univ_path in UNIVERSES.items():
        u = json.loads(Path(univ_path).read_text())
        symbols = u["symbols"]
        print(f"Universe {univ_name}: {len(symbols)} syms ({univ_path})")

        for variant_name, overrides in VARIANTS.items():
            cfg = replace(BASE, **overrides)
            bt = IntradayBreakoutBacktester(cfg)
            for period_name, begin, end, db_path in PERIODS:
                if not Path(db_path).exists():
                    print(f"  SKIP {univ_name}/{variant_name}/{period_name}")
                    continue
                tag = f"{univ_name}__{variant_name}__{period_name}"
                print(f"  {tag}", flush=True)
                res = bt.backtest_portfolio(symbols=symbols, begin=begin, end=end, db_path=db_path)
                s = res.summary
                ret = float(s.get("total_return_pct", 0.0))
                dd = float(s.get("max_drawdown_pct", 0.0))
                tr = int(s.get("total_trades", 0))
                wr = float(s.get("win_rate", 0.0)) * 100
                rdd = abs(ret / dd) if dd else 0.0
                rows.append((univ_name, variant_name, period_name, ret, dd, rdd, tr, wr))
                (out_dir / f"{tag}.json").write_text(
                    json.dumps({"universe": univ_name, "variant": variant_name,
                                "period": period_name, "summary": s}, indent=2, default=str)
                )

    print()
    print(f"{'universe':10s}{'variant':22s}{'period':12s}{'ret%':>9s}{'DD%':>7s}{'R/DD':>7s}{'trades':>8s}{'WR%':>6s}")
    print("-" * 81)
    for u, v, p, ret, dd, rdd, tr, wr in rows:
        print(f"{u:10s}{v:22s}{p:12s}{ret:+9.2f}{dd:7.2f}{rdd:7.2f}{tr:8d}{wr:6.1f}")

    print()
    print("R/DD pivot — universe × period (per variant):")
    for variant_name in VARIANTS:
        print(f"\n{variant_name}:")
        print(f"  {'universe':12s}{'2023_2025':>12s}{'2020':>10s}{'2018':>10s}")
        for u in UNIVERSES:
            cells = []
            for p in ("2023_2025", "2020", "2018"):
                hit = next((r for r in rows if r[0] == u and r[1] == variant_name and r[2] == p), None)
                cells.append(f"{hit[5]:>10.2f}" if hit else f"{'--':>10s}")
            print(f"  {u:12s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
