#!/usr/bin/env python3
"""ORB breakeven-lock exit-tightening probe.

Question: ORB's failure mode is "gap up small, hit a quick high, then
mean-revert all day to -3% stop." Many ORB trades touched +0.5-1% before
reversing. Locking those small gains in (move stop to +0.5% once high
hits +1%) could plausibly flip ORB's negative EV into positive territory
without giving up the 2023-25 alpha where it works as-is.

Per memory `[Pure-intraday bundle null]`, ORB was labeled dead — but the
prior `run_intraday_orb_tightening.py` sweep tested ONLY entry-side
filters (RVOL, distance, window). Exit-side asymmetry has not been
tested. This is a genuinely new mechanism.

Configs (all with allow_orb_breakout=True, ORB filters at live-shipped
values from paper_launch_v2.yaml):
  baseline                 — current ORB exit (-3% hard, 4% trail)
  lock_t1.0_o0.5           — trigger at +1%, lock at +0.5%
  lock_t1.0_o0.3           — trigger at +1%, lock at +0.3% (tighter)
  lock_t0.5_o0.3           — trigger at +0.5%, lock at +0.3% (most aggressive)
  lock_t1.5_o0.5           — trigger at +1.5%, lock at +0.5% (more room to run)

Mandatory CLAUDE.md gates:
  1. 4-period rule: 2023_2025 / 2020 / 2018 (no 2015 intraday data).
  2. Edge claim: pass = 4-period non-degrade + at least one period
     materially better (>2pp return AND DD not worse).
  3. R/DD > 1.5 on multi-year IS = lookahead-suspect (would warrant
     tighter audit).
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _load_symbols(path: str) -> list[str]:
    d = json.loads(Path(path).read_text())
    return d["symbols"] if isinstance(d, dict) else d


# Mirror live intraday_mechanical config EXCEPT:
#  - allow_orb_breakout=True (re-enabled for the test)
#  - orb_min_volume_ratio=2.5 (the orb_vol_2.5 winner from prior sweep)
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
    nr4_min_volume_ratio=2.0,
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
)


def _summarize(s: dict) -> dict:
    rdd = (
        s["total_return_pct"] / abs(s["max_drawdown_pct"])
        if s.get("max_drawdown_pct") else 0
    )
    return {
        "ret": s["total_return_pct"],
        "dd": s["max_drawdown_pct"],
        "rdd": rdd,
        "trades": s["total_trades"],
        "wr": s["win_rate"],
    }


def main() -> int:
    out_dir = Path("results/intraday_orb_breakeven_lock")
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = _load_symbols(UNIVERSE_PATH)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} symbols)\n")

    configs = {
        "baseline":         {},
        "lock_t1.0_o0.5":   {"orb_breakeven_trigger_pct": 0.010, "orb_breakeven_lock_offset_pct": 0.005},
        "lock_t1.0_o0.3":   {"orb_breakeven_trigger_pct": 0.010, "orb_breakeven_lock_offset_pct": 0.003},
        "lock_t0.5_o0.3":   {"orb_breakeven_trigger_pct": 0.005, "orb_breakeven_lock_offset_pct": 0.003},
        "lock_t1.5_o0.5":   {"orb_breakeven_trigger_pct": 0.015, "orb_breakeven_lock_offset_pct": 0.005},
    }

    grid = {}
    for cfg_name, ovr in configs.items():
        cfg = replace(BASE_CONFIG, **ovr)
        print(f"=== {cfg_name} ===")
        grid[cfg_name] = {}
        for period_name, begin, end, db_path in PERIODS:
            bt = IntradayBreakoutBacktester(cfg)
            print(f"  -> {period_name}", flush=True)
            try:
                res = bt.backtest_portfolio(symbols, begin, end, db_path)
            except Exception as exc:
                print(f"     CRASHED: {exc}")
                continue
            s = res.summary
            row = _summarize(s)
            grid[cfg_name][period_name] = row
            print(
                f"     ret {row['ret']:+7.2f}%  "
                f"dd {row['dd']:>6.2f}%  "
                f"R/DD {row['rdd']:+5.2f}  "
                f"trades {row['trades']:>4}  "
                f"win {row['wr']*100:>5.1f}%",
                flush=True,
            )
            (out_dir / f"{cfg_name}_{period_name}.json").write_text(
                json.dumps({
                    "config_name": cfg_name,
                    "period": period_name,
                    "summary": s,
                    "config": asdict(cfg),
                    "setup_summary": (
                        res.setup_summary.to_dict("records")
                        if not res.setup_summary.empty else []
                    ),
                }, indent=2, default=str)
            )
        print()

    # Side-by-side
    periods = [p[0] for p in PERIODS]
    print()
    print(f"{'config':<18}", end="")
    for p in periods:
        print(f"{p:>14s}", end="")
    print()
    print("-" * (18 + 14 * len(periods)))
    for cfg_name in configs.keys():
        print(f"{cfg_name:<18}", end="")
        for p in periods:
            row = grid.get(cfg_name, {}).get(p)
            if row is None:
                print(f"{'(crash)':>14s}", end="")
            else:
                print(f"  {row['ret']:>+5.1f}%/{row['rdd']:>+4.1f}", end="")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
