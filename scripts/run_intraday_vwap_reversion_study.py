#!/usr/bin/env python3
"""True VWAP-mean-reversion long: discount-from-VWAP fade.

Distinct from `pullback_vwap` (which is a continuation entry in an
uptrending session: buy when price reclaims VWAP from below after a
pullback). True reversion is: buy WHILE price is below VWAP, betting
on mean-reversion back to VWAP. Falling-knife guard via max discount.

Sweep:
- 3 min-discount thresholds (0.4 / 0.7 / 1.0%) at bar 4 entry
- 1 timing variant (0.7% at bar 6, later entry = more "confirmed" deviation)

4 variants × 3 periods = 12 backtests on broad250 @ 15m, post-fix.
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
    allow_gap_reclaim_long=False,
    allow_nr4_breakout=False,
    allow_orb_breakout=False,
    allow_vwap_reversion_long=True,  # all variants enable this
    vwap_reversion_min_discount_pct=0.005,
    vwap_reversion_max_discount_pct=0.03,
    vwap_reversion_earliest_entry_bar=4,
    vwap_reversion_latest_entry_bar=18,
    vwap_reversion_min_volume_ratio=1.3,
    require_above_vwap=False,  # SIGNAL fires below VWAP, so disable global gate
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
    "vwapr_d0.4_bar4": {"vwap_reversion_min_discount_pct": 0.004},
    "vwapr_d0.7_bar4": {"vwap_reversion_min_discount_pct": 0.007},
    "vwapr_d1.0_bar4": {"vwap_reversion_min_discount_pct": 0.010},
    "vwapr_d0.7_bar6": {"vwap_reversion_min_discount_pct": 0.007,
                        "vwap_reversion_earliest_entry_bar": 6},
}


def main() -> int:
    universe = json.loads(Path(UNIVERSE_PATH).read_text())
    symbols = universe["symbols"] if isinstance(universe, dict) else universe
    out_dir = Path("results/intraday_vwap_reversion_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} syms)")
    print("Post-fix backtester — true VWAP-reversion long\n")

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
    print(f"{'variant':18s}{'period':12s}{'ret%':>9s}{'DD%':>7s}{'R/DD':>7s}{'trades':>8s}{'WR%':>6s}")
    print("-" * 67)
    for v, p, ret, dd, rdd, tr, wr in rows:
        print(f"{v:18s}{p:12s}{ret:+9.2f}{dd:7.2f}{rdd:7.2f}{tr:8d}{wr:6.1f}")

    print()
    print("Signed R/DD pivot (+ = profit):")
    print(f"  {'variant':18s}{'2023_2025':>12s}{'2020':>10s}{'2018':>10s}")
    for v in VARIANTS:
        cells = []
        for p in ("2023_2025", "2020", "2018"):
            hit = next((r for r in rows if r[0] == v and r[1] == p), None)
            if hit:
                signed = hit[4] if hit[2] >= 0 else -hit[4]
                cells.append(f"{signed:>+10.2f}")
            else:
                cells.append(f"{'--':>10s}")
        print(f"  {v:18s}" + "".join(cells))

    print("\nTrade counts:")
    print(f"  {'variant':18s}{'2023_2025':>12s}{'2020':>10s}{'2018':>10s}")
    for v in VARIANTS:
        cells = []
        for p in ("2023_2025", "2020", "2018"):
            hit = next((r for r in rows if r[0] == v and r[1] == p), None)
            cells.append(f"{hit[5]:>10d}" if hit else f"{'--':>10s}")
        print(f"  {v:18s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
