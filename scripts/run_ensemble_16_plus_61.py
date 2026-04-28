#!/usr/bin/env python3
"""Compute the 50/50 ensemble of 16-ETF + 61-ETF NEW NEW OPTIMAL candidates.

Each gets 50% of capital and runs at full sizing (the strategy's R/DD profile is preserved).
Equity curves are combined daily; combined metrics computed.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradingagents.research.chan_daily_backtester import (
    ChanDailyBacktestConfig, PortfolioChanDailyBacktester,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ensemble")

UNIV16 = "SPY QQQ IWM DIA GLD SLV TLT USO XLF XLE XLK XLV XLI XLY XLP XLU".split()
UNIV61 = (
    "ACWI BIL BND DBA DBC DIA EEM EFA EWA EWG EWJ EWZ FXE FXI GLD GSG HYG IBB "
    "IEFA IEMG INDA IWM IYR LQD MCHI MTUM OIH QQQ QUAL SCHD SCO SHY SLV SOXX "
    "SPY TIP TLT UCO UNG USMV USO UUP VEA VEU VIG VLUE VNQ VWO VYM XBI XLB "
    "XLE XLF XLI XLK XLP XLU XLV XLY XME XOP"
).split()

PERIODS = {
    "2023_2025": ("2023-01-01", "2025-12-30"),
    "2020_2022": ("2020-01-01", "2022-12-30"),
    "2017_2019": ("2017-01-01", "2019-12-30"),
    "2014_2016": ("2014-01-01", "2016-12-30"),
}
LOOKBACK_DAYS = 400


def run_one(universe: list[str], begin: str, end: str) -> pd.DataFrame:
    cfg = ChanDailyBacktestConfig(
        chan_bs_type="1,1p,2,2s,3a,3b",
        buy_types=("T1", "T1P", "T2", "T2S", "T3A", "T3B"),
        sell_types=("T1",),
        chan_min_zs_cnt=0,
        chan_bi_strict=False,
        require_sure=False,
        sizing_mode="atr_parity",
        risk_per_trade=0.020,
        position_pct=0.25,
        max_positions=6,
        enable_shorts=False,
        entry_mode="donchian_or_seg",
        donchian_period=30,
        chan_macd_algo="slope",
        chan_divergence_rate=0.7,
        momentum_rank_lookback=63,
        momentum_rank_top_k=10,
        time_stop_bars=100,
        entry_lag_extra_days=1,
        initial_cash=50_000.0,  # half of nominal $100k
    )
    bt = PortfolioChanDailyBacktester(cfg)
    res = bt.backtest_portfolio(universe, begin, end)
    eq = res.equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    return eq[["date", "equity"]].set_index("date")


def main():
    # Cache equity curves per period × universe (re-used across weight sweeps)
    period_caches = {}
    for period, (is_begin, is_end) in PERIODS.items():
        load_begin = (pd.Timestamp(is_begin) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        log.info("Period %s ...", period)
        eq16 = run_one(UNIV16, load_begin, is_end)
        eq61 = run_one(UNIV61, load_begin, is_end)
        is_ts = pd.Timestamp(is_begin)
        period_caches[period] = (eq16[eq16.index >= is_ts], eq61[eq61.index >= is_ts])

    for w16 in [1.0, 0.8, 0.7, 0.6, 0.5, 0.3]:
        w61 = 1.0 - w16
        print()
        print(f"===== Ensemble weights: 16-ETF={w16:.0%}  61-ETF={w61:.0%} =====")
        print(f"{'Period':<12} {'Return':>12} {'MaxDD':>10} {'R/DD':>8} {'AnnRet':>8}")
        print("-" * 55)
        period_results = []
        for period, (is_begin, _) in PERIODS.items():
            eq16, eq61 = period_caches[period]
            # Each leg starts with 100*w_i thousand; combined = 100k
            cash_16 = 100_000 * w16
            cash_61 = 100_000 * w61
            # Rescale leg equity (each leg's bt initialized with $50k internally;
            # actual leg equity scales linearly so multiply by ratio)
            scale_16 = cash_16 / 50_000.0
            scale_61 = cash_61 / 50_000.0
            scaled16 = eq16["equity"] * scale_16
            scaled61 = eq61["equity"] * scale_61
            merged = pd.DataFrame({"e16": scaled16, "e61": scaled61}).ffill()
            merged["e16"] = merged["e16"].fillna(cash_16)
            merged["e61"] = merged["e61"].fillna(cash_61)
            merged["total"] = merged["e16"] + merged["e61"]
            initial = merged["total"].iloc[0]
            final = merged["total"].iloc[-1]
            ret_pct = (final / initial - 1.0) * 100
            running_max = merged["total"].cummax()
            dd_pct = float(((merged["total"] / running_max - 1.0) * 100).min())
            rdd = ret_pct / abs(dd_pct) if dd_pct != 0 else 0
            ann = ((1 + ret_pct / 100) ** (1/3) - 1) * 100
            print(f"{period:<12} {ret_pct:>+10.2f}% {dd_pct:>+9.2f}% {rdd:>+7.2f} {ann:>+7.2f}%")
            period_results.append((period, ret_pct, dd_pct, rdd, ann))
        avg_rdd = sum(r[3] for r in period_results) / len(period_results)
        avg_ann = sum(r[4] for r in period_results) / len(period_results)
        min_rdd = min(r[3] for r in period_results)
        print("-" * 55)
        print(f"{'AVG':<12} {'':>12} {'':>10} {avg_rdd:>+7.2f} {avg_ann:>+7.2f}%   min_RDD={min_rdd:+.2f}")


if __name__ == "__main__":
    main()
