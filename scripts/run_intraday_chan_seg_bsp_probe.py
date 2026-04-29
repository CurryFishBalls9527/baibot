#!/usr/bin/env python3
"""Test the new Chan-structural 15m segment-level BSP signal as a 4th OR
entry path on intraday_mechanical, on the broad250 universe.

Per chan_daily breakthrough lesson: the unlock for daily was a different
*mechanism* (Chan structural pivot), not a parameter tweak. This script
tests the same idea at 15m: does adding seg-BSP entries alongside the
existing post-open breakout signals (gap_reclaim, NR4, ORB) produce edge?

Mandatory CLAUDE.md gates:
  1. 4-period rule: 2023_2025 / 2020 / 2018. (No 2015 intraday data.)
  2. Future-blanked probe: lag_bars=0 AND lag_bars=1 — if multi-year IS
     edge collapses by >2pp at lag=1, lookahead is suspected.
  3. Treat suspicious R/DD as a bug, not a feature. R/DD > 1.5 on multi-
     year IS deserves extra scrutiny.

Three configs:
  A. baseline    — current live config (gap+nr4+orb, no chan)
  B. +chan_lag0  — add chan_seg_bsp signal at lag 0 (production semantics)
  C. +chan_lag1  — same but lagged 1 bar (probe for lookahead)

Universe: broad250 (mirrors live intraday_mechanical).
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict, replace
from pathlib import Path

# Make chan.py importable for the signal precomputer.
_CHAN_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "chan.py"
if _CHAN_ROOT.exists() and str(_CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAN_ROOT))

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


# Mirror live intraday_mechanical (paper_launch_v2.yaml).
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
    out_dir = Path("results/intraday_chan_seg_bsp")
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = _load_symbols(UNIVERSE_PATH)
    print(f"Universe: {UNIVERSE_PATH} ({len(symbols)} symbols)\n")

    configs = {
        "baseline":         replace(BASE_CONFIG, allow_chan_seg_bsp_long=False),
        "+chan_seg_lag0":   replace(BASE_CONFIG, allow_chan_seg_bsp_long=True,  chan_seg_bsp_lag_bars=0),
        "+chan_seg_lag1":   replace(BASE_CONFIG, allow_chan_seg_bsp_long=True,  chan_seg_bsp_lag_bars=1),
    }

    grid = {}
    for cfg_name, cfg in configs.items():
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
        print(f"{p:>12s}", end="")
    print()
    print("-" * (18 + 12 * len(periods)))
    for cfg_name in configs.keys():
        print(f"{cfg_name:<18}", end="")
        for p in periods:
            row = grid.get(cfg_name, {}).get(p)
            if row is None:
                print(f"{'(crash)':>12s}", end="")
            else:
                print(f"  {row['ret']:>+5.1f}%/{row['rdd']:>+4.1f}", end="")
        print()

    # Probe: lag0 vs lag1 delta (per CLAUDE.md edge-claim rule #3)
    print()
    print("Future-blanked probe (lag0 → lag1 R/DD delta):")
    for p in periods:
        l0 = grid.get("+chan_seg_lag0", {}).get(p)
        l1 = grid.get("+chan_seg_lag1", {}).get(p)
        if l0 and l1:
            d = l1["rdd"] - l0["rdd"]
            print(f"  {p:<10s}  lag0 R/DD {l0['rdd']:+5.2f}  lag1 R/DD {l1['rdd']:+5.2f}  Δ {d:+5.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
