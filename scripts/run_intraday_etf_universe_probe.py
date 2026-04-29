#!/usr/bin/env python3
"""Test current live intraday signal config (gap_reclaim + NR4 + ORB) on
the 8-macro-ETF universe at 15m.

Adapts the chan_daily breakthrough lesson — "the unlock was the universe,
not the parameters" — to intraday. The 8-macro-ETF universe (SPY, QQQ,
IWM, DIA, GLD, SLV, TLT, USO) is fundamentally different from the broad250
single-stock universe: tighter spreads, mean-reverting flow, sector
rotation, no idiosyncratic event risk. Different microstructure may
favor different breakout-style signals.

Cheap probe (~1 hr) before committing to:
 - Backfilling sector SPDRs (XLF/XLE/XLK/...) for the full 16-ETF test
 - Building a Chan-structural 15m signal as a 4th OR entry path

If the 8-ETF universe shows R/DD > 0.5 on any 4-period subset (using
mandatory CLAUDE.md 2023_25 / 2020 / 2018 set; 2015 intraday data not
yet available), worth investing in the full backfill.

Mirror live intraday_mechanical config (paper_launch_v2.yaml v2 variant).
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
    ("2023_2025", "2023-01-01", "2025-12-30", "research_data/intraday_15m_etf.duckdb"),
    ("2020", "2020-01-01", "2020-12-31", "research_data/intraday_15m_etf_2020.duckdb"),
    ("2018", "2018-01-01", "2018-12-31", "research_data/intraday_15m_etf_2018.duckdb"),
]

ETF_8 = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO"]

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
    allow_orb_breakout=True,  # mirrors live (memory says dead but live still fires it)
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


def main() -> int:
    out_dir = Path("results/intraday_etf_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for period_name, begin, end, db_path in PERIODS:
        cfg = replace(BASE_CONFIG)
        bt = IntradayBreakoutBacktester(cfg)
        print(f"-> 8-ETF / {period_name}", flush=True)
        res = bt.backtest_portfolio(ETF_8, begin, end, db_path)
        s = res.summary
        rdd = (
            s["total_return_pct"] / abs(s["max_drawdown_pct"])
            if s.get("max_drawdown_pct")
            else 0
        )
        print(
            f"  return {s['total_return_pct']:+7.2f}%  "
            f"dd {s['max_drawdown_pct']:>6.2f}%  "
            f"R/DD {rdd:+5.2f}  "
            f"trades {s['total_trades']:>4}  "
            f"win {s['win_rate']*100:>5.1f}%",
            flush=True,
        )
        (out_dir / f"etf8_{period_name}.json").write_text(
            json.dumps({
                "universe": "8-macro-ETF",
                "period": period_name,
                "summary": s,
                "config": asdict(cfg),
                "setup_summary": (
                    res.setup_summary.to_dict("records")
                    if not res.setup_summary.empty else []
                ),
            }, indent=2, default=str)
        )
        rows.append({
            "period": period_name,
            "ret": s["total_return_pct"],
            "dd": s["max_drawdown_pct"],
            "rdd": rdd,
            "trades": s["total_trades"],
            "wr": s["win_rate"],
        })

    print()
    print(f"{'period':<12}{'ret%':>8}{'DD%':>7}{'R/DD':>7}{'trades':>8}{'WR%':>7}")
    print("-" * 50)
    for r in rows:
        print(
            f"{r['period']:<12}{r['ret']:>+7.2f}%"
            f"{r['dd']:>+6.2f}%"
            f"{r['rdd']:>+6.2f}"
            f"{r['trades']:>8}"
            f"{r['wr']*100:>6.1f}%"
        )

    # Quick verdict: per chan_daily lesson, R/DD > 0.5 on any OOS period
    # is the bar to consider further investment (sector backfill, deeper
    # universe analysis). Below that → universe flip is null, save effort.
    return 0


if __name__ == "__main__":
    sys.exit(main())
