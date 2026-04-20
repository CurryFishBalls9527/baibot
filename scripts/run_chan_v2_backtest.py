#!/usr/bin/env python3
"""Run Chan v2 portfolio backtest with exit and sizing improvements.

Usage:
    python scripts/run_chan_v2_backtest.py
    python scripts/run_chan_v2_backtest.py --sell-cooldown-bars 5
    python scripts/run_chan_v2_backtest.py --sell-cooldown-bars 5 --breakeven-atr 2.0
    python scripts/run_chan_v2_backtest.py --macd-algo peak
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from tradingagents.research.chan_v2_backtester import ChanV2BacktestConfig, PortfolioChanV2Backtester
from tradingagents.research.market_context import build_market_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_v2")


def _load_regime(db_path: str, begin: str, end: str) -> pd.DataFrame | None:
    """Load regime data using MarketDataWarehouse (same as live system)."""
    from tradingagents.research import MarketDataWarehouse

    daily_db = Path("research_data/market_data.duckdb")
    if not daily_db.exists():
        log.warning("No daily warehouse found, running without regime gating")
        return None

    try:
        symbols = ["SPY", "QQQ", "IWM", "SMH", "^VIX"]
        warehouse = MarketDataWarehouse(str(daily_db))
        try:
            frames = {s: warehouse.get_daily_bars(s, begin, end) for s in symbols}
        finally:
            warehouse.close()
        if frames:
            return build_market_context(frames)
    except Exception as e:
        log.warning("Failed to load regime data: %s", e)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Chan v2 portfolio backtest")
    p.add_argument("--universe", default="research_data/spike_universe.json")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--begin", default="2023-01-01")
    p.add_argument("--end", default="2025-12-30")
    p.add_argument("--db", default="research_data/intraday_30m.duckdb")
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=8)
    p.add_argument("--exit-mode", default="zs_structural",
                    choices=["atr_trail", "zs_structural"])
    p.add_argument("--no-regime", action="store_true", help="Disable regime gating")

    # --- V1 inherited flags ---
    p.add_argument("--daily-filter", action="store_true",
                    help="Only enter when daily Chan bi is bullish")
    p.add_argument("--daily-filter-mode", default="bullish_only",
                    choices=["bullish_only", "not_bearish"])
    p.add_argument("--daily-db", default="research_data/market_data.duckdb")
    p.add_argument("--sepa-filter", action="store_true")
    p.add_argument("--dead-money-bars", type=int, default=0)
    p.add_argument("--dead-money-min-gain", type=float, default=0.05)
    p.add_argument("--ma-trend-filter", action="store_true")
    p.add_argument("--regime-gate", action="store_true")
    p.add_argument("--regime-min-score", type=int, default=4)
    p.add_argument("--seg-bsp-boost", action="store_true")
    p.add_argument("--seg-bsp-boost-factor", type=float, default=1.5)
    p.add_argument("--vol-divergence-filter", action="store_true")
    p.add_argument("--hub-peak-exit", action="store_true")
    p.add_argument("--daily-zs-filter", action="store_true")
    p.add_argument("--daily-bsp-confirm", action="store_true")
    p.add_argument("--bs-type", default="1,1p,2,2s")
    p.add_argument("--buy-types", default="T1,T2,T2S")
    p.add_argument("--macd-algo", default="area",
                    choices=["area", "peak", "full_area", "diff", "slope", "amp"],
                    help="MACD algorithm for BSP divergence detection")

    # --- V2 exit improvements ---
    p.add_argument("--sell-cooldown-bars", type=int, default=0,
                    help="Ignore sell BSPs for first N bars after entry (0=off)")
    p.add_argument("--breakeven-atr", type=float, default=0.0,
                    help="Raise stop to entry after +N ATR gain (0=off)")
    p.add_argument("--trail-tighten-atr", type=float, default=0.0,
                    help="Switch to tight trail after +N ATR gain (0=off)")
    p.add_argument("--trail-tighten-mult", type=float, default=2.0,
                    help="Tight trailing stop distance: N × ATR from high")
    p.add_argument("--stop-grace-bars", type=int, default=0,
                    help="Use wide 5×ATR safety stop for first N bars (0=off)")
    p.add_argument("--zs-oscillation-exit", action="store_true",
                    help="Exit when inside ZS with N+ oscillations")
    p.add_argument("--zs-max-oscillations", type=int, default=5,
                    help="Oscillation threshold for ZS exhaustion exit")

    # --- V2 sizing improvements ---
    p.add_argument("--divergence-sizing", action="store_true",
                    help="Scale position size by divergence strength")
    p.add_argument("--divergence-sizing-base", type=float, default=0.6,
                    help="Divergence rate at which sizing = 1.0x")
    p.add_argument("--divergence-sizing-max-boost", type=float, default=1.5,
                    help="Max sizing multiplier for strong divergence")

    p.add_argument("--out", default="results/chan_v2/backtest.json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        data = json.loads(Path(args.universe).read_text())
        symbols = data["symbols"] if isinstance(data, dict) else data
    if args.limit:
        symbols = symbols[: args.limit]
    symbols = [s for s in symbols if s != "BRK-B"]

    config = ChanV2BacktestConfig(
        initial_cash=args.cash,
        max_positions=args.max_positions,
        exit_mode=args.exit_mode,
        daily_filter=args.daily_filter,
        daily_filter_mode=args.daily_filter_mode,
        daily_db_path=args.daily_db,
        sepa_filter=args.sepa_filter,
        dead_money_bars=args.dead_money_bars,
        dead_money_min_gain=args.dead_money_min_gain,
        ma_trend_filter=args.ma_trend_filter,
        regime_gate=args.regime_gate,
        regime_min_score=args.regime_min_score,
        seg_bsp_boost=args.seg_bsp_boost,
        seg_bsp_boost_factor=args.seg_bsp_boost_factor,
        vol_divergence_filter=args.vol_divergence_filter,
        hub_peak_exit=args.hub_peak_exit,
        daily_zs_filter=args.daily_zs_filter,
        daily_bsp_confirm=args.daily_bsp_confirm,
        chan_bs_type=args.bs_type,
        chan_macd_algo=args.macd_algo,
        buy_types=tuple(args.buy_types.split(",")),
        # V2 fields
        sell_cooldown_bars=args.sell_cooldown_bars,
        breakeven_atr=args.breakeven_atr,
        trail_tighten_atr=args.trail_tighten_atr,
        trail_tighten_mult=args.trail_tighten_mult,
        stop_grace_bars=args.stop_grace_bars,
        zs_oscillation_exit=args.zs_oscillation_exit,
        zs_max_oscillations=args.zs_max_oscillations,
        divergence_sizing=args.divergence_sizing,
        divergence_sizing_base=args.divergence_sizing_base,
        divergence_sizing_max_boost=args.divergence_sizing_max_boost,
    )

    regime_df = None
    if not args.no_regime:
        regime_df = _load_regime(args.db, args.begin, args.end)
        if regime_df is not None:
            log.info("Loaded regime data: %d days", len(regime_df))

    bt = PortfolioChanV2Backtester(config)
    result = bt.backtest_portfolio(symbols, args.begin, args.end, args.db, regime_df=regime_df)

    s = result.summary
    print()
    print("=" * 70)
    print("  CHAN V2 PORTFOLIO BACKTEST — RESULTS")
    print("=" * 70)
    print(f"  Initial capital:       ${s['initial_cash']:,.0f}")
    print(f"  Final equity:          ${s['final_equity']:,.2f}")
    print(f"  Total return:          {s['total_return_pct']:+.2f}%")
    print(f"  Max drawdown:          {s['max_drawdown_pct']:.2f}%")
    print(f"  Total trades:          {s['total_trades']}")
    print(f"  Win rate:              {s['win_rate']*100:.1f}%")
    print(f"  Avg return:            {s['avg_return']*100:+.3f}%")
    print(f"  Avg winner:            {s['avg_win']*100:+.3f}%")
    print(f"  Avg loser:             {s['avg_loss']*100:+.3f}%")
    print(f"  Avg bars held:         {s['avg_bars_held']:.1f}")
    print()
    print("  By exit reason:")
    for reason, stats in s["by_exit_reason"].items():
        print(f"    {reason:<15} {stats['count']:>5} trades  "
              f"WR {stats['win_rate']*100:>5.1f}%  avg {stats['avg_ret']*100:>+7.3f}%")
    print()
    print("  By entry bsp type:")
    for tp, stats in s["by_entry_type"].items():
        print(f"    {tp:<6} {stats['count']:>5} trades  "
              f"WR {stats['win_rate']*100:>5.1f}%  avg {stats['avg_ret']*100:>+7.3f}%")

    if not result.symbol_summary.empty:
        print()
        top = result.symbol_summary.nlargest(5, "total_pnl")
        bottom = result.symbol_summary.nsmallest(5, "total_pnl")
        print("  Top 5 symbols by PnL:")
        for _, row in top.iterrows():
            print(f"    {row['symbol']:<6} {row['trades']:>3} trades  PnL ${row['total_pnl']:>+8,.0f}  WR {row['win_rate']*100:.0f}%")
        print("  Bottom 5:")
        for _, row in bottom.iterrows():
            print(f"    {row['symbol']:<6} {row['trades']:>3} trades  PnL ${row['total_pnl']:>+8,.0f}  WR {row['win_rate']*100:.0f}%")

    print("=" * 70)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": s,
        "equity_curve": result.equity_curve.to_dict("records") if not result.equity_curve.empty else [],
        "trades": result.trades.to_dict("records") if not result.trades.empty else [],
    }
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
