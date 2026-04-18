#!/usr/bin/env python3
"""Run Chan theory portfolio backtest with shared capital management.

Usage:
    python scripts/run_chan_portfolio_backtest.py
    python scripts/run_chan_portfolio_backtest.py --begin 2018-01-01 --end 2018-12-31 --db research_data/intraday_30m_2018.duckdb
    python scripts/run_chan_portfolio_backtest.py --symbols AAPL MSFT NVDA --limit 10
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from tradingagents.research.chan_backtester import ChanBacktestConfig, PortfolioChanBacktester
from tradingagents.research.market_context import build_market_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_portfolio")


def _load_regime(db_path: str, begin: str, end: str) -> pd.DataFrame | None:
    """Load regime data from daily warehouse if available."""
    daily_db = Path("research_data/market_data.duckdb")
    if not daily_db.exists():
        log.warning("No daily warehouse found, running without regime gating")
        return None

    try:
        import duckdb
        conn = duckdb.connect(str(daily_db), read_only=True)
        data_by_symbol = {}
        for sym in ["SPY", "QQQ", "IWM"]:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume FROM daily_bars "
                "WHERE symbol = ? AND date >= ? AND date <= ? ORDER BY date",
                [sym, begin, end],
            ).fetchdf()
            if not rows.empty:
                rows["date"] = pd.to_datetime(rows["date"])
                rows = rows.set_index("date")
                data_by_symbol[sym] = rows
        conn.close()
        if data_by_symbol:
            return build_market_context(data_by_symbol)
    except Exception as e:
        log.warning("Failed to load regime data: %s", e)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Chan portfolio backtest")
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
    p.add_argument("--daily-filter", action="store_true",
                    help="Only enter when daily Chan bi is bullish")
    p.add_argument("--daily-filter-mode", default="bullish_only",
                    choices=["bullish_only", "not_bearish"],
                    help="bullish_only: require UP bi; not_bearish: allow neutral+bullish")
    p.add_argument("--daily-db", default="research_data/market_data.duckdb",
                    help="DuckDB path for daily bars")
    p.add_argument("--sepa-filter", action="store_true",
                    help="Only enter when stock passes SEPA trend template")
    p.add_argument("--out", default="results/chan_portfolio/backtest.json")
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

    config = ChanBacktestConfig(
        initial_cash=args.cash,
        max_positions=args.max_positions,
        exit_mode=args.exit_mode,
        daily_filter=args.daily_filter,
        daily_filter_mode=args.daily_filter_mode,
        daily_db_path=args.daily_db,
        sepa_filter=args.sepa_filter,
    )

    regime_df = None
    if not args.no_regime:
        regime_df = _load_regime(args.db, args.begin, args.end)
        if regime_df is not None:
            log.info("Loaded regime data: %d days", len(regime_df))

    bt = PortfolioChanBacktester(config)
    result = bt.backtest_portfolio(symbols, args.begin, args.end, args.db, regime_df=regime_df)

    s = result.summary
    print()
    print("=" * 70)
    print("  CHAN PORTFOLIO BACKTEST — RESULTS")
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
