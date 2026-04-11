#!/usr/bin/env python3
"""Walk-forward backtest with LLM entry decisions.

Uses the TradingAgentsGraph LLM pipeline to make entry decisions instead of
purely mechanical rules. The Minervini screener identifies candidates, then
the LLM (market analyst only for backtesting) makes the final call.

Usage:
    # Small test (5 symbols, short period)
    python run_llm_backtest.py --test-mode

    # Full run
    python run_llm_backtest.py --start 2023-01-01 --end 2025-12-31 --trade-start 2024-01-01

    # With specific LLM model
    python run_llm_backtest.py --llm-model gpt-4o-mini --min-confidence 0.7
"""

import argparse
import json

import pandas as pd
import logging
import sys
from datetime import datetime
from pathlib import Path

from tradingagents.research.backtester import BacktestConfig
from tradingagents.research.llm_backtester import LLMBacktester, LLMBacktestConfig
from tradingagents.research.minervini import MinerviniConfig
from tradingagents.research.seed_universe import build_seed_universe, load_seed_universe
from tradingagents.research.walk_forward import WalkForwardConfig
from tradingagents.research.warehouse import MarketDataWarehouse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest with LLM entry decisions"
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="Data start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--trade-start", type=str, default=None,
        help="Trading start date (default: start + 1yr warmup)",
    )
    parser.add_argument(
        "--rebalance", type=str, default="weekly",
        choices=["weekly", "monthly"],
    )
    parser.add_argument(
        "--db", type=str, default="research_data/market_data.duckdb",
    )
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Path to seed universe JSON",
    )
    # LLM settings
    parser.add_argument(
        "--llm-provider", type=str, default="openai",
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o-mini",
        help="LLM model for entry decisions",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.6,
        help="Min LLM confidence to enter (0.0-1.0)",
    )
    parser.add_argument(
        "--analysts", type=str, default="market",
        help="Comma-separated analyst types (market,social,news,fundamentals)",
    )
    # Portfolio settings
    parser.add_argument(
        "--initial-cash", type=float, default=100_000,
    )
    parser.add_argument(
        "--max-positions", type=int, default=10,
    )
    parser.add_argument(
        "--max-position-pct", type=float, default=0.15,
    )
    # Screening
    parser.add_argument(
        "--min-rs", type=float, default=50.0,
    )
    parser.add_argument(
        "--min-template-score", type=int, default=6,
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50,
    )
    # Output
    parser.add_argument(
        "--results-dir", type=str, default="results/llm_backtest",
    )
    # Test mode
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Run with 5 symbols and 1 month for quick validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Test mode overrides
    if args.test_mode:
        args.start = "2023-06-01"
        args.end = "2025-03-31"
        args.trade_start = "2024-06-01"
        args.max_candidates = 10
        args.rebalance = "monthly"

    # Load universe
    if args.universe:
        seed_symbols = load_seed_universe(args.universe)
    else:
        seed_symbols = build_seed_universe()

    if args.test_mode:
        # Use a small subset for testing
        seed_symbols = seed_symbols[:50]

    analysts = [a.strip() for a in args.analysts.split(",")]

    print(f"\n{'=' * 70}")
    print(f"  LLM WALK-FORWARD BACKTEST")
    print(f"{'=' * 70}")
    print(f"  Seed universe:    {len(seed_symbols)} symbols")
    print(f"  Period:           {args.start} to {args.end}")
    print(f"  Trade start:      {args.trade_start or 'auto (1yr warmup)'}")
    print(f"  Rebalance:        {args.rebalance}")
    print(f"  LLM:              {args.llm_provider}/{args.llm_model}")
    print(f"  Analysts:         {', '.join(analysts)}")
    print(f"  Min confidence:   {args.min_confidence}")
    print(f"  Capital:          ${args.initial_cash:,.0f}")
    print(f"  Max positions:    {args.max_positions}")
    print(f"{'=' * 70}\n")

    # Initialize
    warehouse = MarketDataWarehouse(db_path=args.db, read_only=True)

    llm_config = LLMBacktestConfig(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        selected_analysts=analysts,
        min_confidence=args.min_confidence,
    )

    wf_config = WalkForwardConfig(
        rebalance_frequency=args.rebalance,
        min_template_score=args.min_template_score,
        min_rs_percentile=args.min_rs,
        max_screen_candidates=args.max_candidates,
    )

    backtest_config = BacktestConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        max_position_pct=args.max_position_pct,
        min_template_score=5,
        require_volume_surge=False,
        require_base_pattern=False,
        require_market_regime=False,
    )

    screener_config = MinerviniConfig(
        require_fundamentals=False,
        require_market_uptrend=False,
    )

    # Run
    backtester = LLMBacktester(
        warehouse=warehouse,
        llm_config=llm_config,
        wf_config=wf_config,
        backtest_config=backtest_config,
        screener_config=screener_config,
    )

    result = backtester.run(
        seed_symbols=seed_symbols,
        start_date=args.start,
        end_date=args.end,
        trade_start_date=args.trade_start,
    )

    warehouse.close()

    # Print comparison
    llm_summary = result.llm_result.summary
    mech_summary = result.mechanical_result.summary

    if not llm_summary or not mech_summary:
        print("  No trades executed. Check data availability.")
        return

    print(f"\n{'=' * 70}")
    print(f"  RESULTS COMPARISON: LLM vs Mechanical")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25} {'LLM':>15} {'Mechanical':>15} {'SPY':>15}")
    print(f"  {'-' * 65}")

    spy_ret = llm_summary.get("benchmark_return", 0)

    rows = [
        ("Return", "total_return", True),
        ("Max Drawdown", "max_drawdown", True),
        ("Total Trades", "total_trades", False),
        ("Win Rate", "trade_win_rate", True),
        ("Avg Trade Return", "avg_trade_return", True),
        ("Symbols Traded", "symbols_with_trades", False),
    ]

    for label, key, is_pct in rows:
        llm_val = llm_summary.get(key, 0)
        mech_val = mech_summary.get(key, 0)
        if is_pct:
            print(f"  {label:<25} {llm_val:>14.2%} {mech_val:>14.2%}", end="")
            if key == "total_return":
                print(f" {spy_ret:>14.2%}")
            else:
                print()
        else:
            print(f"  {label:<25} {llm_val:>15} {mech_val:>15}")

    # LLM decision stats
    decisions = result.llm_decisions
    if decisions:
        buy_count = sum(1 for d in decisions if d["action"] == "BUY")
        hold_count = sum(1 for d in decisions if d["action"] == "HOLD")
        sell_count = sum(1 for d in decisions if d["action"] == "SELL")
        avg_conf = sum(d["confidence"] for d in decisions) / len(decisions)
        buy_conf = (
            sum(d["confidence"] for d in decisions if d["action"] == "BUY") / buy_count
            if buy_count > 0 else 0
        )

        print(f"\n  {'LLM Decision Stats':<25}")
        print(f"  {'-' * 40}")
        print(f"  {'Total decisions':<25} {len(decisions):>15}")
        print(f"  {'BUY signals':<25} {buy_count:>15}")
        print(f"  {'HOLD signals':<25} {hold_count:>15}")
        print(f"  {'SELL signals':<25} {sell_count:>15}")
        print(f"  {'Avg confidence':<25} {avg_conf:>14.2f}")
        print(f"  {'Avg BUY confidence':<25} {buy_conf:>14.2f}")
        print(f"  {'LLM API calls':<25} {result.llm_calls:>15}")
        print(f"  {'Cache hits':<25} {result.cache_hits:>15}")

    # LLM added value
    llm_ret = llm_summary.get("total_return", 0)
    mech_ret = mech_summary.get("total_return", 0)
    alpha = llm_ret - mech_ret
    print(f"\n  LLM Alpha vs Mechanical: {alpha:+.2%}")
    if llm_ret > spy_ret:
        print(f"  LLM BEATS SPY by {llm_ret - spy_ret:+.2%}")
    else:
        print(f"  LLM underperforms SPY by {spy_ret - llm_ret:.2%}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # LLM equity curve
    eq = result.llm_result.equity_curve
    if not eq.empty:
        eq.to_csv(results_dir / "llm_equity_curve.csv", index=False)

    # Mechanical equity curve
    meq = result.mechanical_result.equity_curve
    if not meq.empty:
        meq.to_csv(results_dir / "mechanical_equity_curve.csv", index=False)

    # LLM trades
    trades = result.llm_result.trades
    if not trades.empty:
        trades.to_csv(results_dir / "llm_trades.csv", index=False)

    # LLM decisions log
    if decisions:
        pd.DataFrame(decisions).to_csv(results_dir / "llm_decisions.csv", index=False)

    # Summary
    summary_out = {
        "llm": llm_summary,
        "mechanical": mech_summary,
        "llm_config": {
            "model": args.llm_model,
            "analysts": analysts,
            "min_confidence": args.min_confidence,
        },
        "llm_stats": {
            "total_decisions": len(decisions),
            "buy_signals": buy_count,
            "hold_signals": hold_count,
            "sell_signals": sell_count,
            "api_calls": result.llm_calls,
            "cache_hits": result.cache_hits,
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary_out, indent=2, default=str))

    print(f"\n  Results saved to: {results_dir}/")
    print()


if __name__ == "__main__":
    main()
