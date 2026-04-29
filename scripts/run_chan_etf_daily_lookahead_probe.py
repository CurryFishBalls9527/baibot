#!/usr/bin/env python3
"""Future-blanked probe: re-run multi-level Chan with daily filter shifted by 1 day.

The base `_get_daily_data` returns sym_dir[bar_date] when present, which
includes the FULL daily bar for bar_date — i.e., uses today's post-close
daily Chan direction to gate today's intraday entries. That's same-day
lookahead.

This script monkey-patches the lookup to use the strictly-prior daily
date (D-1), then re-runs the 3-period sweep. Per CLAUDE.md edge-claim
discipline: if results drop materially under the lag, the apparent edge
is lookahead.
"""
from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research import chan_backtester as cb_mod
from tradingagents.research.chan_backtester import PortfolioChanBacktester


def _strictly_prior_daily_data(daily_directions: dict, symbol: str, bar_date: str):
    """Same as _get_daily_data but returns the largest date STRICTLY LESS than bar_date.

    Eliminates same-day lookahead: the daily Chan direction for day D was
    computed from day D's full bar, which is not observable during day D's
    intraday session.
    """
    sym_dir = daily_directions.get(symbol)
    if not sym_dir:
        return "neutral"
    dates = sorted(sym_dir.keys())
    idx = bisect_left(dates, bar_date) - 1
    if idx < 0:
        return "neutral"
    return sym_dir[dates[idx]]


# Monkey-patch BEFORE running anything that imports the unpatched function.
PortfolioChanBacktester._get_daily_data = staticmethod(_strictly_prior_daily_data)
print("[probe] Patched _get_daily_data → strictly-prior date (no same-day lookahead)")


# Now hand off to the standard wrapper but force --daily-filter and a probe out-dir.
import importlib.util  # noqa: E402

sys.argv = [
    "run_chan_etf_backtest",
    "--daily-filter",
    "--out-dir", "results/chan_etf/daily_gate_probe",
]
wrapper_path = Path(__file__).parent / "run_chan_etf_backtest.py"
spec = importlib.util.spec_from_file_location("run_chan_etf_backtest", wrapper_path)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
wrapper.main()
