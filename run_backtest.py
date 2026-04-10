#!/usr/bin/env python3
"""Run backtests comparing existing strategies vs. the new exit-managed approach.

Usage:
    .venv/bin/python run_backtest.py                          # defaults
    .venv/bin/python run_backtest.py --symbols AAPL,NVDA,MSFT --start 2024-01-01
    .venv/bin/python run_backtest.py --portfolio --symbols AAPL,NVDA,MSFT,GOOGL,META
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

from tradingagents.backtesting.engine import BacktestEngine
from tradingagents.backtesting.strategies import SwingStrategy, CombinedStrategy
from tradingagents.backtesting.portfolio_backtest import PortfolioBacktestEngine

# New: Exit-managed strategy for comparison
import backtrader as bt


class ExitManagedSwingStrategy(bt.Strategy):
    """Swing strategy with Phase 2 exit management logic.

    Mirrors our live ExitManager:
    - ATR trailing stop (ratchets up)
    - Breakeven stop once up 5%
    - Partial profit at 12% gain (sell 33%)
    - 50 DMA exit
    - Max hold 60 days

    Entry: same as SwingStrategy (EMA crossover + RSI + MACD + trend).
    """

    params = dict(
        # Entry
        ema_fast=9,
        ema_slow=21,
        ema_trend=50,
        rsi_period=14,
        rsi_low=35,
        rsi_high=75,
        # Exit manager params (matching our live config)
        atr_period=14,
        trail_stop_pct=0.10,
        breakeven_trigger_pct=0.05,
        partial_profit_trigger_pct=0.12,
        partial_profit_fraction=0.33,
        max_hold_days=60,
        use_50dma_exit=True,
        sma_50_period=50,
        position_pct=0.15,
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(self.data.close)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.sma_50 = bt.indicators.SMA(self.data.close, period=self.p.sma_50_period)

        self.order = None
        self.entry_price = 0
        self.entry_bar = 0
        self.highest_close = 0
        self.current_stop = 0
        self.partial_taken = False

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.entry_bar = len(self)
                self.highest_close = order.executed.price
                self.current_stop = order.executed.price * (1 - self.p.trail_stop_pct)
                self.partial_taken = False
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Entry logic (same as SwingStrategy)
            bullish = (
                self.ema_fast[0] > self.ema_slow[0]
                and self.p.rsi_low < self.rsi[0] < self.p.rsi_high
                and self.macd.macd[0] > self.macd.signal[0]
                and self.data.close[0] > self.ema_trend[0]
                and self.atr[0] > 0
            )
            if bullish:
                size = int(self.broker.getvalue() * self.p.position_pct / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # Exit manager logic
            price = self.data.close[0]
            self.highest_close = max(self.highest_close, price)
            hold_days = len(self) - self.entry_bar

            # Trailing stop
            trail_stop = self.highest_close * (1.0 - self.p.trail_stop_pct)
            self.current_stop = max(self.current_stop, trail_stop)

            # Breakeven stop
            if price >= self.entry_price * (1.0 + self.p.breakeven_trigger_pct):
                self.current_stop = max(self.current_stop, self.entry_price)

            # Check exit conditions
            if price <= self.current_stop:
                self.order = self.close()
                return

            if self.p.use_50dma_exit and price < self.sma_50[0]:
                self.order = self.close()
                return

            if hold_days >= self.p.max_hold_days:
                self.order = self.close()
                return

            # Partial profit
            if (
                not self.partial_taken
                and price >= self.entry_price * (1.0 + self.p.partial_profit_trigger_pct)
            ):
                partial_qty = max(1, int(self.position.size * self.p.partial_profit_fraction))
                self.order = self.sell(size=partial_qty)
                self.partial_taken = True
                self.current_stop = max(self.current_stop, self.entry_price)


def run_single_stock_comparison(symbols, start_date, end_date, initial_cash):
    """Compare strategies on individual stocks."""
    engine = BacktestEngine(initial_cash=initial_cash)

    strategies = [
        ("SwingStrategy (baseline)", SwingStrategy, {}),
        ("CombinedStrategy (scoring)", CombinedStrategy, {}),
        ("ExitManaged (Phase 2)", ExitManagedSwingStrategy, {}),
    ]

    all_results = []
    for symbol in symbols:
        print(f"\n{'#' * 70}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'#' * 70}")

        for name, strat_class, params in strategies:
            try:
                result = engine.run_single(
                    symbol, strat_class,
                    start_date=start_date, end_date=end_date,
                    strategy_params=params,
                )
                result["strategy"] = name
                engine.print_results(result)
                all_results.append(result)
            except Exception as e:
                print(f"\n  {name}: ERROR - {e}")
                all_results.append({"symbol": symbol, "strategy": name, "error": str(e)})

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"  {'Symbol':<8} {'Strategy':<30} {'Return':>8} {'B&H':>8} "
        f"{'Beat?':>5} {'Sharpe':>7} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7}"
    )
    print(f"  {'-' * 86}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['symbol']:<8} {r['strategy']:<30} ERROR: {r['error'][:30]}")
            continue
        beat = "YES" if r.get("beats_buy_hold") else "no"
        sharpe = f"{r['sharpe_ratio']:.2f}" if r.get("sharpe_ratio") is not None else "n/a"
        print(
            f"  {r['symbol']:<8} {r['strategy']:<30} "
            f"{r['total_return_pct']:>8} {r['buy_hold_return_pct']:>8} "
            f"{beat:>5} {sharpe:>7} {r['max_drawdown_pct']:>7} "
            f"{r['win_rate_pct']:>8} {r['total_trades']:>7}"
        )
    print()

    return all_results


def run_portfolio_comparison(symbols, start_date, end_date, initial_cash):
    """Compare portfolio-level strategies."""
    from tradingagents.backtesting.portfolio_backtest import (
        PortfolioBacktestEngine,
        PortfolioSwingStrategy,
    )
    from tradingagents.backtesting.smart_strategies import (
        MomentumRankingStrategy,
        RegimeAwareStrategy,
    )

    print(f"\n{'#' * 70}")
    print(f"  PORTFOLIO BACKTEST: {', '.join(symbols)}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Capital: ${initial_cash:,.0f}")
    print(f"{'#' * 70}")

    backtester = PortfolioBacktestEngine(initial_cash=initial_cash)

    strategies = [
        ("PortfolioSwing", PortfolioSwingStrategy, {}),
        ("MomentumRanking", MomentumRankingStrategy, {}),
    ]

    # RegimeAware needs SPY as first feed
    if "SPY" not in symbols:
        regime_symbols = ["SPY"] + symbols
    else:
        regime_symbols = symbols

    for name, strat_class, params in strategies:
        try:
            syms = regime_symbols if name == "RegimeAware" else symbols
            result = backtester.run(
                syms, strat_class,
                start_date=start_date, end_date=end_date,
                strategy_params=params,
            )
            backtester.print_result(result)
        except Exception as e:
            print(f"\n  {name}: ERROR - {e}")

    # Also run RegimeAware
    try:
        result = backtester.run(
            regime_symbols, RegimeAwareStrategy,
            start_date=start_date, end_date=end_date,
        )
        backtester.print_result(result)
    except Exception as e:
        print(f"\n  RegimeAware: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument(
        "--symbols", type=str, default="NVDA,AAPL,MSFT,META,GOOGL",
        help="Comma-separated symbols"
    )
    parser.add_argument(
        "--start", type=str,
        default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cash", type=float, default=100_000,
        help="Initial cash"
    )
    parser.add_argument(
        "--portfolio", action="store_true",
        help="Also run portfolio-level backtests"
    )

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    print(f"\n  Symbols:  {', '.join(symbols)}")
    print(f"  Period:   {args.start} to {args.end}")
    print(f"  Capital:  ${args.cash:,.0f}")

    # Single-stock comparison
    run_single_stock_comparison(symbols, args.start, args.end, args.cash)

    # Portfolio comparison
    if args.portfolio:
        run_portfolio_comparison(symbols, args.start, args.end, args.cash)


if __name__ == "__main__":
    main()
