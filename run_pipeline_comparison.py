#!/usr/bin/env python3
"""Compare simple single-call LLM vs full 13-call agent pipeline.

For a small set of symbols across a short date range, run BOTH:
1. Simple backtest-style single LLM call (blinded, technicals only)
2. Full TradingAgentsGraph pipeline (13 LLM calls, with news/fundamentals)

Report agreement rate, confidence correlation, and sample reasoning
to determine if the full pipeline adds meaningful value over the
simple single-call approach.

NOTE: Full pipeline uses CURRENT news/fundamentals regardless of trade_date
(known limitation). This comparison is about pipeline complexity, not
historical accuracy.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.research.llm_backtester import LLMBacktester, LLMBacktestConfig
from tradingagents.research.minervini import MinerviniConfig, MinerviniScreener
from tradingagents.research.warehouse import MarketDataWarehouse


# Pick 10 symbols with mixed 2024 performance (avoid survivorship bias)
DEFAULT_SYMBOLS = [
    "NVDA",   # Big winner
    "PLTR",   # Big winner
    "AAPL",   # Flat/modest
    "AMZN",   # Modest winner
    "GOOGL",  # Modest winner
    "TSLA",   # Volatile
    "INTC",   # Loser
    "BA",     # Loser
    "XOM",    # Cyclical
    "WMT",    # Defensive
]

# Bi-weekly checkpoints across Oct-Dec 2024
DEFAULT_DATES = [
    "2024-10-07",
    "2024-10-21",
    "2024-11-04",
    "2024-11-18",
    "2024-12-02",
    "2024-12-16",
]


def run_simple_pipeline(
    backtester: LLMBacktester,
    symbol: str,
    trade_date: str,
    prepared_frames: dict,
    as_of_ts: pd.Timestamp,
) -> dict:
    """Run the simple single-call pipeline (backtest version)."""
    context = backtester._build_screener_context(symbol, prepared_frames, as_of_ts)
    return backtester.get_llm_decision(symbol, trade_date, screener_context=context)


def run_full_pipeline(graph, symbol: str, trade_date: str) -> dict:
    """Run the full 13-call TradingAgentsGraph pipeline."""
    try:
        state, _ = graph.propagate(symbol, trade_date)
        signal = graph.signal_processor.process_signal_structured(
            state["final_trade_decision"], symbol
        )
        return {
            "action": signal.get("action", "HOLD").upper(),
            "confidence": float(signal.get("confidence", 0.5)),
            "reasoning": signal.get("reasoning", "")[:300],
        }
    except Exception as e:
        return {
            "action": "ERROR",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)[:200]}",
        }


def main():
    parser = argparse.ArgumentParser(description="Compare simple vs full LLM pipelines")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--dates", type=str, default=",".join(DEFAULT_DATES))
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--db", type=str, default="research_data/market_data.duckdb")
    parser.add_argument("--output", type=str, default="results/pipeline_comparison")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("tradingagents").setLevel(logging.WARNING)

    symbols = [s.strip() for s in args.symbols.split(",")]
    dates = [d.strip() for d in args.dates.split(",")]

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPARISON: Simple vs Full Agent Graph")
    print(f"{'=' * 70}")
    print(f"  Symbols:   {', '.join(symbols)} ({len(symbols)})")
    print(f"  Dates:     {len(dates)} checkpoints ({dates[0]} to {dates[-1]})")
    print(f"  LLM:       {args.llm_provider}/{args.llm_model}")
    print(f"  Total:     {len(symbols) * len(dates)} pairs")
    print(f"  Est. cost: ~${len(symbols) * len(dates) * 14 * 0.001:.2f}")
    print(f"{'=' * 70}\n")

    # 1. Load data for screener context
    print("Loading market data...")
    warehouse = MarketDataWarehouse(db_path=args.db, read_only=True)
    start_date = (pd.Timestamp(dates[0]) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    data_by_symbol = warehouse.get_daily_bars_bulk(symbols, start_date, end_date)
    warehouse.close()

    screener = MinerviniScreener(
        MinerviniConfig(require_fundamentals=False, require_market_uptrend=False)
    )
    prepared_frames = {}
    for sym, df in data_by_symbol.items():
        prepared = screener.prepare_features(df)
        if not prepared.empty:
            prepared_frames[sym] = prepared
    print(f"  Loaded features for {len(prepared_frames)} symbols\n")

    # 2. Initialize simple backtester
    simple_config = LLMBacktestConfig(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        cache_db_path="research_data/pipeline_comparison_cache.db",
    )
    simple_bt = LLMBacktester(warehouse=None, llm_config=simple_config)

    # 3. Initialize full pipeline (TradingAgentsGraph)
    print("Initializing full agent graph...")
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    full_config = {
        **DEFAULT_CONFIG,
        "llm_provider": args.llm_provider,
        "deep_think_llm": args.llm_model,
        "quick_think_llm": args.llm_model,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }
    graph = TradingAgentsGraph(
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config=full_config,
    )
    print("  Ready\n")

    # 4. Run comparisons
    results = []
    total = len(symbols) * len(dates)
    start_time = time.time()

    for i, symbol in enumerate(symbols):
        if symbol not in prepared_frames:
            print(f"[{symbol}] No data, skipping")
            continue

        for j, date_str in enumerate(dates):
            idx = i * len(dates) + j + 1
            elapsed = time.time() - start_time
            print(f"[{idx}/{total}] {symbol} @ {date_str} (elapsed: {elapsed:.0f}s)")

            as_of_ts = pd.Timestamp(date_str)

            # Simple pipeline
            t0 = time.time()
            simple_result = run_simple_pipeline(
                simple_bt, symbol, date_str, prepared_frames, as_of_ts
            )
            simple_time = time.time() - t0
            print(f"  Simple: {simple_result['action']} "
                  f"(conf={simple_result['confidence']:.2f}, {simple_time:.1f}s)")

            # Full pipeline
            t0 = time.time()
            full_result = run_full_pipeline(graph, symbol, date_str)
            full_time = time.time() - t0
            print(f"  Full:   {full_result['action']} "
                  f"(conf={full_result['confidence']:.2f}, {full_time:.1f}s)")

            results.append({
                "symbol": symbol,
                "date": date_str,
                "simple_action": simple_result["action"],
                "simple_confidence": simple_result["confidence"],
                "simple_reasoning": simple_result.get("reasoning", "")[:200],
                "simple_time_s": round(simple_time, 1),
                "full_action": full_result["action"],
                "full_confidence": full_result["confidence"],
                "full_reasoning": full_result.get("reasoning", "")[:200],
                "full_time_s": round(full_time, 1),
                "agree": simple_result["action"] == full_result["action"],
            })

    # 5. Analyze
    df = pd.DataFrame(results)
    if df.empty:
        print("\nNo results collected.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "comparison.csv", index=False)

    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total pairs:          {len(df)}")
    print(f"  Total time:           {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()
    print(f"  Simple pipeline avg:  {df['simple_time_s'].mean():.1f}s per call")
    print(f"  Full pipeline avg:    {df['full_time_s'].mean():.1f}s per call")
    print(f"  Full is {df['full_time_s'].mean() / df['simple_time_s'].mean():.1f}x slower")
    print()

    # Agreement
    agree_count = df["agree"].sum()
    agree_rate = agree_count / len(df)
    print(f"  Agreement rate:       {agree_count}/{len(df)} ({agree_rate:.0%})")
    print()

    # Action breakdown
    print(f"  {'Action':<10} {'Simple':>10} {'Full':>10}")
    print(f"  {'-' * 32}")
    for action in ["BUY", "HOLD", "SELL", "ERROR"]:
        s = (df["simple_action"] == action).sum()
        f = (df["full_action"] == action).sum()
        print(f"  {action:<10} {s:>10} {f:>10}")
    print()

    # Confidence correlation (only on agreeing decisions)
    if agree_count > 0:
        agreeing = df[df["agree"]]
        print(f"  Avg simple conf:      {df['simple_confidence'].mean():.2f}")
        print(f"  Avg full conf:        {df['full_confidence'].mean():.2f}")
        if len(agreeing) >= 2:
            corr = agreeing["simple_confidence"].corr(agreeing["full_confidence"])
            print(f"  Conf correlation:     {corr:.2f} (on {len(agreeing)} agreeing)")
    print()

    # Disagreements
    disagreements = df[~df["agree"]]
    if not disagreements.empty:
        print(f"  DISAGREEMENTS ({len(disagreements)}):")
        print(f"  {'-' * 65}")
        for _, row in disagreements.iterrows():
            print(f"  {row['symbol']:6} @ {row['date']}: "
                  f"Simple={row['simple_action']:4}({row['simple_confidence']:.2f}) vs "
                  f"Full={row['full_action']:4}({row['full_confidence']:.2f})")

    print(f"\n  Results saved to: {output_dir}/comparison.csv")
    print()


if __name__ == "__main__":
    main()
