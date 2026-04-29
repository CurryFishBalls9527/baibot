#!/usr/bin/env python3
"""Chan v2 backtest on the 8-ETF futures-proxy universe.

Universe: SPY, QQQ, IWM, DIA, GLD, SLV, TLT, USO (proxies for
ES, NQ, RTY, YM, GC, SI, ZN, CL futures respectively).

Three-period sweep at 15-minute bars, mirroring the live `chan_v2`
variant settings from experiments/paper_launch_v2.yaml. RS filter is
intentionally OFF — the universe is already curated to 8 names.

Usage:
    # default: all 3 periods, regime gating on
    python scripts/run_chan_etf_backtest.py

    # single period
    python scripts/run_chan_etf_backtest.py --periods 2023_2025

    # future-blanked probe (CLAUDE.md edge-claim discipline)
    python scripts/run_chan_etf_backtest.py --no-regime
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from tradingagents.research.chan_v2_backtester import (
    ChanV2BacktestConfig,
    PortfolioChanV2Backtester,
)
from tradingagents.research.market_context import build_market_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_etf")

UNIVERSE_PATH = "research_data/etf_futures_proxy_universe.json"

PERIODS = {
    "2023_2025": {
        "begin": "2023-01-01",
        "end":   "2025-12-30",
        "db":    "research_data/intraday_15m_etf.duckdb",
    },
    "2020": {
        "begin": "2020-01-01",
        "end":   "2020-12-30",
        "db":    "research_data/intraday_15m_etf_2020.duckdb",
    },
    "2018": {
        "begin": "2018-01-01",
        "end":   "2018-12-30",
        "db":    "research_data/intraday_15m_etf_2018.duckdb",
    },
}


def _load_regime(begin: str, end: str) -> pd.DataFrame | None:
    """Load SPY/QQQ/IWM/SMH/VIX regime context (same as live)."""
    from tradingagents.research import MarketDataWarehouse

    daily_db = Path("research_data/market_data.duckdb")
    if not daily_db.exists():
        log.warning("No daily warehouse — running without regime gating")
        return None
    try:
        symbols = ["SPY", "QQQ", "IWM", "SMH", "^VIX"]
        wh = MarketDataWarehouse(str(daily_db))
        try:
            frames = {s: wh.get_daily_bars(s, begin, end) for s in symbols}
        finally:
            wh.close()
        return build_market_context(frames) if frames else None
    except Exception as e:
        log.warning("Failed to load regime data: %s", e)
        return None


def _make_config(args: argparse.Namespace) -> ChanV2BacktestConfig:
    """Mirror live chan_v2 settings (paper_launch_v2.yaml lines 115-134)."""
    return ChanV2BacktestConfig(
        initial_cash=args.cash,
        intraday_interval_minutes=args.interval,
        max_positions=args.max_positions,
        exit_mode=args.exit_mode,
        # Live chan_v2 filter settings
        chan_macd_algo=args.macd_algo,
        buy_types=("T1", "T2", "T2S"),
        filter_macd_zero_axis=True,
        filter_divergence_max=0.6,
        # v2 dead-money exit (live chan_v2 setting)
        dead_money_bars=150,
        dead_money_min_gain=0.05,
        # RS filter OFF — universe is already 8 names
        rs_filter=False,
        # Regime gate — mirrors chan_v2 (which doesn't enable it; pass-through context only)
        regime_gate=False,
        # Multi-level Chan: daily Chan as gate, intraday BSP for entry
        daily_filter=args.daily_filter,
        daily_filter_mode=args.daily_filter_mode,
        daily_db_path="research_data/market_data.duckdb",
    )


def _run_one_period(
    period: str,
    cfg: ChanV2BacktestConfig,
    symbols: list[str],
    use_regime: bool,
    out_dir: Path,
) -> dict:
    spec = PERIODS[period]
    db_path = Path(spec["db"])
    if not db_path.exists():
        log.error("DuckDB missing for period %s: %s", period, db_path)
        return {"period": period, "error": "db_missing"}

    regime_df = _load_regime(spec["begin"], spec["end"]) if use_regime else None
    if regime_df is not None:
        log.info("Loaded regime data: %d days", len(regime_df))

    log.info("=" * 70)
    log.info("PERIOD %s | %s -> %s | db=%s | symbols=%d",
             period, spec["begin"], spec["end"], db_path, len(symbols))
    log.info("=" * 70)

    bt = PortfolioChanV2Backtester(cfg)
    result = bt.backtest_portfolio(
        symbols, spec["begin"], spec["end"], str(db_path), regime_df=regime_df,
    )
    s = result.summary

    print()
    print("=" * 70)
    print(f"  CHAN ETF BACKTEST — PERIOD {period}")
    print("=" * 70)
    print(f"  Initial capital:       ${s['initial_cash']:,.0f}")
    print(f"  Final equity:          ${s['final_equity']:,.2f}")
    print(f"  Total return:          {s['total_return_pct']:+.2f}%")
    print(f"  Max drawdown:          {s['max_drawdown_pct']:.2f}%")
    dd_abs = abs(s['max_drawdown_pct'])
    rdd = (s['total_return_pct'] / dd_abs) if dd_abs > 0 else float('nan')
    print(f"  Return / DD:           {rdd:+.2f}")
    print(f"  Total trades:          {s['total_trades']}")
    print(f"  Win rate:              {s['win_rate']*100:.1f}%")
    print(f"  Avg return:            {s['avg_return']*100:+.3f}%")
    print(f"  Avg winner:            {s['avg_win']*100:+.3f}%")
    print(f"  Avg loser:             {s['avg_loss']*100:+.3f}%")
    print(f"  Avg bars held:         {s['avg_bars_held']:.1f}")
    if s.get("by_exit_reason"):
        print("  By exit reason:")
        for reason, st in s["by_exit_reason"].items():
            print(f"    {reason:<15} {st['count']:>4}  WR {st['win_rate']*100:>5.1f}%  avg {st['avg_ret']*100:>+7.3f}%")
    if not result.symbol_summary.empty:
        print("  By symbol:")
        for _, row in result.symbol_summary.sort_values("total_pnl", ascending=False).iterrows():
            print(f"    {row['symbol']:<5} {row['trades']:>3} trades  PnL ${row['total_pnl']:>+8,.0f}  WR {row['win_rate']*100:.0f}%")
    print()

    out_path = out_dir / f"{period}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "period": period,
        "summary": s,
        "config": {
            "interval_minutes": cfg.intraday_interval_minutes,
            "max_positions": cfg.max_positions,
            "exit_mode": cfg.exit_mode,
            "buy_types": list(cfg.buy_types),
            "dead_money_bars": cfg.dead_money_bars,
            "filter_divergence_max": cfg.filter_divergence_max,
            "filter_macd_zero_axis": cfg.filter_macd_zero_axis,
            "daily_filter": cfg.daily_filter,
            "daily_filter_mode": cfg.daily_filter_mode,
            "use_regime": use_regime,
        },
        "symbols": symbols,
        "trades": result.trades.to_dict("records") if not result.trades.empty else [],
    }, indent=2, default=str))
    log.info("Wrote %s", out_path)

    return {"period": period, "summary": s}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chan v2 ETF futures-proxy backtest")
    p.add_argument("--periods", nargs="*", default=list(PERIODS.keys()),
                   choices=list(PERIODS.keys()),
                   help="Periods to run (default: all)")
    p.add_argument("--symbols", nargs="*",
                   help="Override universe (default: load etf_futures_proxy_universe.json)")
    p.add_argument("--interval", type=int, default=15, choices=[5, 15, 30],
                   help="Bar interval minutes (default 15)")
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=8)
    p.add_argument("--exit-mode", default="zs_structural",
                   choices=["atr_trail", "zs_structural"])
    p.add_argument("--macd-algo", default="area",
                   choices=["area", "peak", "full_area", "diff", "slope", "amp"])
    p.add_argument("--no-regime", action="store_true",
                   help="Disable regime gating (also serves as the future-blanked probe)")
    p.add_argument("--daily-filter", action="store_true",
                   help="Multi-level Chan: only enter when daily Chan is bullish")
    p.add_argument("--daily-filter-mode", default="bullish_only",
                   choices=["bullish_only", "not_bearish"])
    p.add_argument("--out-dir", default="results/chan_etf",
                   help="Output directory for per-period JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        data = json.loads(Path(UNIVERSE_PATH).read_text())
        symbols = data["symbols"] if isinstance(data, dict) else data

    cfg = _make_config(args)
    out_dir = Path(args.out_dir)

    summaries: list[dict] = []
    for period in args.periods:
        summaries.append(_run_one_period(
            period, cfg, symbols,
            use_regime=not args.no_regime,
            out_dir=out_dir,
        ))

    print()
    print("=" * 70)
    print("  CROSS-PERIOD SUMMARY")
    print("=" * 70)
    print(f"  {'Period':<12} {'Return':>10} {'MaxDD':>10} {'R/DD':>8} {'Trades':>8} {'WR':>7}")
    for r in summaries:
        if "error" in r:
            print(f"  {r['period']:<12} ERROR: {r['error']}")
            continue
        s = r["summary"]
        dd_abs = abs(s['max_drawdown_pct'])
        rdd = (s['total_return_pct'] / dd_abs) if dd_abs > 0 else float('nan')
        print(f"  {r['period']:<12} {s['total_return_pct']:>+9.2f}% {s['max_drawdown_pct']:>9.2f}% "
              f"{rdd:>+7.2f} {s['total_trades']:>8} {s['win_rate']*100:>6.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
