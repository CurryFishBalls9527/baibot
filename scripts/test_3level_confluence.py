#!/usr/bin/env python3
"""Research-2: 3-level intraday confluence test.

Hypothesis: daily entry signals (Donchian / seg-BSP) confirmed by 15m sub-level
Chan buy BSP in the last N hours produce HIGHER R/DD than daily signal alone.

Setup:
- Universe: 8 macro ETFs (only ones with intraday history): SPY QQQ IWM DIA GLD SLV TLT USO
- Period: 2023-2025 (where we have 15m data continuously via intraday_15m_etf.duckdb)
- Daily backtest baseline: NEW NEW OPTIMAL config
- Test: gate entries by intraday confirmation

Method:
1. Run daily backtester to identify all daily entry signals
2. For each signal date, query 15m chan to check if buy BSP fired in last N hours
3. Filter signals → compute filtered backtest stats
4. Compare R/DD with vs without filter

Result determines whether to integrate into main backtester.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tradingagents.automation  # noqa
from tradingagents.research.chan_daily_backtester import (
    ChanDailyBacktestConfig, PortfolioChanDailyBacktester,
)

# chan engine
CHAN_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "chan.py"
if str(CHAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAN_ROOT))
from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import AUTYPE, KL_TYPE  # noqa: E402

from tradingagents.research.chan_adapter import DuckDBIntradayAPI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("3level")

UNIV_8 = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "USO"]


def get_intraday_buy_bsp_dates(symbol: str, begin: str, end: str, db_path: str) -> set[pd.Timestamp]:
    """For a symbol, return set of DATES (just the date part) on which a buy BSP
    fired at the 15m level. Used as confluence-confirm signal for daily."""
    DuckDBIntradayAPI.DB_PATH = db_path
    chan_cfg = CChanConfig({
        "trigger_step": True, "bi_strict": False, "min_zs_cnt": 0,
        "bs_type": "1,1p,2,2s,3a,3b", "print_warning": False,
        "zs_algo": "normal", "macd_algo": "slope", "divergence_rate": 0.7,
    })
    out = set()
    try:
        chan = CChan(
            code=symbol, begin_time=begin, end_time=end,
            data_src="custom:DuckDBAPI.DuckDB30mAPI",  # this is for 30m; we need 15m
            lv_list=[KL_TYPE.K_15M], config=chan_cfg, autype=AUTYPE.QFQ,
        )
        seen = set()
        for snapshot in chan.step_load():
            try:
                lvl = snapshot[0]
                bsps = lvl.bs_point_lst.getSortedBspList()
                for bsp in bsps:
                    try:
                        if not bsp.is_buy:
                            continue
                        klu_idx = bsp.klu.idx
                        if klu_idx in seen:
                            continue
                        seen.add(klu_idx)
                        ct = bsp.klu.time
                        out.add(pd.Timestamp(year=ct.year, month=ct.month, day=ct.day))
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception as e:
        log.warning("15m chan failed for %s: %s", symbol, e)
    return out


def main():
    # First, make sure DuckDBIntradayAPI shim is registered for 15m
    # Check what data_src works
    log.info("Loading intraday adapter...")
    DuckDBIntradayAPI.DB_PATH = "research_data/intraday_15m_etf.duckdb"
    api = DuckDBIntradayAPI("SPY", k_type=KL_TYPE.K_15M, begin_date="2024-01-01", end_date="2024-01-15")
    bars = list(api.get_kl_data())
    log.info("SPY 15m bars sampled: %d", len(bars))

    # Step 1: collect intraday-buy-bsp dates per symbol
    log.info("Pre-loading 15m buy BSP dates per symbol (2023-2025)...")
    intraday_buy_dates = {}
    for sym in UNIV_8:
        dates = get_intraday_buy_bsp_dates(
            sym, "2023-01-01", "2025-12-30",
            db_path="research_data/intraday_15m_etf.duckdb",
        )
        intraday_buy_dates[sym] = dates
        log.info("  %s: %d 15m buy BSPs (~%d unique days)", sym, len(dates), len(dates))

    # Step 2: run daily backtest baseline AND filtered version
    cfg_base = ChanDailyBacktestConfig(
        chan_bs_type="1,1p,2,2s,3a,3b",
        buy_types=("T1", "T1P", "T2", "T2S", "T3A", "T3B"),
        sell_types=("T1",),
        chan_min_zs_cnt=0, chan_bi_strict=False, require_sure=False,
        sizing_mode="atr_parity", risk_per_trade=0.020, position_pct=0.25,
        max_positions=6, enable_shorts=False,
        entry_mode="donchian_or_seg", donchian_period=30,
        chan_macd_algo="slope", chan_divergence_rate=0.7,
        momentum_rank_lookback=63, momentum_rank_top_k=10,
        time_stop_bars=100, entry_priority_mode="momentum",
        trend_type_filter_mode="trend_only",
        entry_lag_extra_days=1,
    )
    bt = PortfolioChanDailyBacktester(cfg_base)
    log.info("Running daily baseline backtest (8 macro ETFs, 2023-25)...")
    res_base = bt.backtest_portfolio(UNIV_8, "2021-11-27", "2025-12-30")
    # Filter to IS
    res_base.trades["entry_date"] = pd.to_datetime(res_base.trades["entry_date"])
    is_trades = res_base.trades[res_base.trades["entry_date"] >= "2023-01-01"].copy()
    log.info("Baseline: %d IS entry trades", len(is_trades))

    # Step 3: per-trade confluence check
    is_trades["intraday_confirm"] = is_trades.apply(
        lambda r: r["entry_date"] in intraday_buy_dates.get(r["symbol"], set()),
        axis=1,
    )

    # Step 4: compare baseline vs filtered
    print()
    print("=" * 70)
    print("BASELINE (no intraday filter)")
    print("=" * 70)
    _print_stats(is_trades)

    print()
    print("=" * 70)
    print("FILTERED (require 15m buy BSP same day)")
    print("=" * 70)
    filtered = is_trades[is_trades["intraday_confirm"]].copy()
    _print_stats(filtered)

    print()
    print("=" * 70)
    print("FILTERED OUT (no 15m confluence — what we'd miss)")
    print("=" * 70)
    missed = is_trades[~is_trades["intraday_confirm"]].copy()
    _print_stats(missed)


def _print_stats(trades: pd.DataFrame):
    if trades.empty:
        print("  (no trades)")
        return
    print(f"  N trades: {len(trades)}")
    wr = (trades["return"] > 0).mean()
    print(f"  Win rate: {wr*100:.1f}%")
    avg_ret = trades["return"].mean() * 100
    print(f"  Avg per-trade return: {avg_ret:+.3f}%")
    sum_ret = trades["return"].sum() * 100
    print(f"  Sum of per-trade returns: {sum_ret:+.2f}%")
    if (trades["return"] > 0).any():
        avg_w = trades[trades["return"] > 0]["return"].mean() * 100
    else:
        avg_w = 0
    if (trades["return"] < 0).any():
        avg_l = trades[trades["return"] < 0]["return"].mean() * 100
    else:
        avg_l = 0
    print(f"  Avg winner: {avg_w:+.2f}%   Avg loser: {avg_l:+.2f}%")


if __name__ == "__main__":
    main()
