#!/usr/bin/env python3
"""Futures backtest with NEW NEW OPTIMAL config + proper contract multipliers."""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradingagents.research.chan_daily_backtester import (
    ChanDailyBacktestConfig, PortfolioChanDailyBacktester,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("futures")

# CME standard contract multipliers
CONTRACT_MULTIPLIERS = {
    "ES=F": 50, "NQ=F": 20, "YM=F": 5, "RTY=F": 50,
    "GC=F": 100, "SI=F": 5000, "CL=F": 1000,
    "ZB=F": 1000, "ZN=F": 1000,
    "HG=F": 25000, "6E=F": 125000, "6J=F": 12500000,
    "ZS=F": 50, "ZW=F": 50, "NG=F": 10000,
}
# Approx CME initial margins
INITIAL_MARGINS = {
    "ES=F": 13000, "NQ=F": 18000, "YM=F": 8000, "RTY=F": 5000,
    "GC=F": 10000, "SI=F": 15000, "CL=F": 8000,
    "ZB=F": 5000, "ZN=F": 2000,
    "HG=F": 6000, "6E=F": 3000, "6J=F": 3000,
    "ZS=F": 3000, "ZW=F": 2500, "NG=F": 5000,
}

PERIODS = {
    "2017_2019": ("2017-01-01", "2019-12-30"),
    "2020_2022": ("2020-01-01", "2022-12-30"),
    "2023_2025": ("2023-01-01", "2025-12-30"),
}
LOOKBACK_DAYS = 400


def main(universe: list[str], lag_extra: int, label: str, risk_pct: float = 0.020, pos_pct: float = 0.25):
    summaries = {}
    for period, (is_begin, is_end) in PERIODS.items():
        load_begin = (pd.Timestamp(is_begin) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        log.info("Period %s ...", period)
        cfg = ChanDailyBacktestConfig(
            chan_bs_type="1,1p,2,2s,3a,3b",
            buy_types=("T1", "T1P", "T2", "T2S", "T3A", "T3B"),
            sell_types=("T1",),
            chan_min_zs_cnt=0,
            chan_bi_strict=False,
            require_sure=False,
            sizing_mode="atr_parity",
            risk_per_trade=risk_pct,
            position_pct=pos_pct,
            max_positions=6,
            enable_shorts=False,
            entry_mode="donchian_or_seg",
            donchian_period=30,
            chan_macd_algo="slope",
            chan_divergence_rate=0.7,
            momentum_rank_lookback=63,
            momentum_rank_top_k=10,
            time_stop_bars=100,
            entry_lag_extra_days=lag_extra,
            contract_multipliers=CONTRACT_MULTIPLIERS,
            initial_margins=INITIAL_MARGINS,
        )
        bt = PortfolioChanDailyBacktester(cfg)
        res = bt.backtest_portfolio(universe, load_begin, is_end)
        # Filter to IS
        is_ts = pd.Timestamp(is_begin)
        eq = res.equity_curve.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq_is = eq[eq["date"] >= is_ts]
        if eq_is.empty:
            continue
        initial = eq_is["equity"].iloc[0]
        final = eq_is["equity"].iloc[-1]
        ret = (final / initial - 1.0) * 100
        running_max = eq_is["equity"].cummax()
        dd = float(((eq_is["equity"] / running_max - 1.0) * 100).min())
        rdd = ret / abs(dd) if dd < 0 else 0
        n_trades = len(res.trades[pd.to_datetime(res.trades["entry_date"]) >= is_ts]) if not res.trades.empty else 0
        summaries[period] = (ret, dd, rdd, n_trades)
        print(f"  {period}: {ret:+.2f}% / DD {dd:+.2f}% / R/DD {rdd:+.2f} / trades {n_trades}")

    print()
    print(f"===== {label} (lag {lag_extra}, {len(universe)} futures) =====")
    print(f"{'Period':<12} {'Return':>10} {'MaxDD':>10} {'R/DD':>8} {'Trades':>7} {'Annual':>8}")
    rdds = []
    for p, (r, d, rdd, n) in summaries.items():
        ann = ((1 + r/100) ** (1/3) - 1) * 100
        print(f"{p:<12} {r:>+9.2f}% {d:>+9.2f}% {rdd:>+7.2f} {n:>7} {ann:>+7.2f}%")
        rdds.append(rdd)
    print(f"AVG R/DD: {sum(rdds)/len(rdds):+.2f}")


if __name__ == "__main__":
    UNIV8 = ["ES=F", "NQ=F", "RTY=F", "YM=F", "GC=F", "SI=F", "ZB=F", "CL=F"]

    print("="*70)
    print("Standard $100k account: futures contracts too big — DD hits 22-61%")
    print("="*70)
    main(UNIV8, lag_extra=1, label="8-futures @ $100k 4x sizing")

    # Try with micro contracts: MES = $5/pt (1/10 of ES). Simulate by
    # dividing all multipliers by 10 (mimics MES, MNQ, MYM, M2K, MGC, etc.)
    print()
    print("="*70)
    print("Micro futures @ 4x sizing (matches research config)")
    print("="*70)
    CONTRACT_MULTIPLIERS.update({k: v / 10 for k, v in dict(CONTRACT_MULTIPLIERS).items()})
    INITIAL_MARGINS.update({k: v / 10 for k, v in dict(INITIAL_MARGINS).items()})
    main(UNIV8, lag_extra=1, label="8-MICRO @ 4x sizing")

    print()
    print("="*70)
    print("Micro futures @ 1x sizing (CONSERVATIVE for live)")
    print("="*70)
    main(UNIV8, lag_extra=1, label="8-MICRO @ 1x sizing", risk_pct=0.005, pos_pct=0.10)

    print()
    print("="*70)
    print("Micro futures @ 2x sizing (moderate live)")
    print("="*70)
    main(UNIV8, lag_extra=1, label="8-MICRO @ 2x sizing", risk_pct=0.010, pos_pct=0.15)
