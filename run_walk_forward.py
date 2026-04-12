#!/usr/bin/env python3
"""Walk-forward backtest with dynamic stock screening.

Eliminates survivorship bias by screening stocks at each rebalance point
using only data available on that date.

Usage:
    python run_walk_forward.py --start 2023-01-01 --end 2025-12-31 --rebalance weekly
    python run_walk_forward.py --universe research_data/seed_universe.json --rebalance monthly
    python run_walk_forward.py --start 2023-06-01 --end 2025-12-31 --max-positions 8
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from tradingagents.research.backtester import BacktestConfig
from tradingagents.research.minervini import MinerviniConfig
from tradingagents.research.seed_universe import build_seed_universe, load_seed_universe
from tradingagents.research.walk_forward import WalkForwardBacktester, WalkForwardConfig
from tradingagents.research.warehouse import MarketDataWarehouse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest with dynamic Minervini screening"
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01",
        help="Data start date including warmup (YYYY-MM-DD)",
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
        help="How often to re-screen the universe",
    )
    parser.add_argument(
        "--db", type=str, default="research_data/market_data.duckdb",
        help="Path to DuckDB warehouse",
    )
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Path to seed universe JSON (builds default if not provided)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="SPY",
        help="Benchmark symbol",
    )
    parser.add_argument(
        "--initial-cash", type=float, default=100_000,
        help="Starting capital",
    )
    parser.add_argument(
        "--max-positions", type=int, default=6,
        help="Maximum concurrent positions",
    )
    parser.add_argument(
        "--max-position-pct", type=float, default=0.10,
        help="Max single position as fraction of equity",
    )
    parser.add_argument(
        "--min-rs", type=float, default=70.0,
        help="Minimum RS percentile for screening",
    )
    parser.add_argument(
        "--min-template-score", type=int, default=6,
        help="Minimum Minervini template score for screening",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=50,
        help="Max candidates per rebalance screen",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/walk_forward",
        help="Directory for output files",
    )
    parser.add_argument(
        "--buy-point-tolerance", type=float, default=1.0,
        help="Entry price tolerance below pivot buy point (1.0 = strict, 0.98 = allow 2% below)",
    )
    parser.add_argument(
        "--max-stage", type=int, default=2,
        help="Maximum Minervini stage number for entries (2 = early breakouts only, 3 = mid-cycle ok)",
    )
    parser.add_argument(
        "--no-50dma-exit", action="store_true",
        help="Disable 50 DMA break as an exit condition",
    )
    parser.add_argument(
        "--no-require-breakout-ready", action="store_true",
        help="Disable the breakout_ready flag gate (rely on pivot check alone)",
    )
    parser.add_argument(
        "--allow-continuation-entry", action="store_true",
        help="Enable alt entry path for persistent leaders (pullback to EMA, not breakout)",
    )
    parser.add_argument(
        "--trail-stop", type=float, default=0.10,
        help="Trailing stop percent (0.10 = 10%)",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Park idle cash in SPY to match target exposure",
    )
    parser.add_argument(
        "--overlay-threshold", type=float, default=0.05,
        help="Min gap (fraction of equity) to trigger overlay rebalance",
    )
    parser.add_argument(
        "--target-exposure", type=float, default=None,
        help="Override regime-based target exposure for all regimes (e.g. 1.0 for full overlay)",
    )
    parser.add_argument(
        "--vol-target", action="store_true",
        help="Enable Barroso-Santa Clara strategy-vol-targeted exposure scaling",
    )
    parser.add_argument(
        "--vol-target-level", type=float, default=0.15,
        help="Annualized strategy vol target (default 0.15 = 15%)",
    )
    parser.add_argument(
        "--vol-target-halflife", type=int, default=20,
        help="EWM halflife in trading days for vol estimation (default 20)",
    )
    parser.add_argument(
        "--min-rvol", type=float, default=0.0,
        help="Hard gate: require breakout_volume_ratio >= N on breakout entries (default 0 = off)",
    )
    parser.add_argument(
        "--disable-breakouts-in-uptrend", action="store_true",
        help="In confirmed_uptrend regime, skip breakout entries and run continuation-only",
    )
    parser.add_argument(
        "--min-group-rank", type=float, default=0.0,
        help="Industry group rank percentile gate (0-100). 0 = off. 60 = require top 40% group.",
    )
    parser.add_argument(
        "--group-by", type=str, default="industry",
        choices=["industry", "sector"],
        help="Grouping level for group rank filter",
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

    # Load or build seed universe
    if args.universe:
        seed_symbols = load_seed_universe(args.universe)
    else:
        seed_symbols = build_seed_universe()

    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD BACKTEST (Survivorship-Bias-Free)")
    print(f"{'=' * 70}")
    print(f"  Seed universe:    {len(seed_symbols)} symbols")
    print(f"  Period:           {args.start} to {args.end}")
    print(f"  Rebalance:        {args.rebalance}")
    print(f"  Benchmark:        {args.benchmark}")
    print(f"  Capital:          ${args.initial_cash:,.0f}")
    print(f"  Max positions:    {args.max_positions}")
    print(f"  Min RS percentile: {args.min_rs}%")
    print(f"  Min template score: {args.min_template_score}")
    print(f"{'=' * 70}\n")

    # Initialize
    warehouse = MarketDataWarehouse(db_path=args.db, read_only=True)

    wf_config = WalkForwardConfig(
        rebalance_frequency=args.rebalance,
        min_template_score=args.min_template_score,
        min_rs_percentile=args.min_rs,
        max_screen_candidates=args.max_candidates,
        benchmark=args.benchmark,
        min_group_rank_pct=args.min_group_rank,
        group_by=args.group_by,
    )

    backtest_config = BacktestConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        max_position_pct=args.max_position_pct,
        # Relax entry conditions since the walk-forward screener
        # already pre-filters the candidate universe
        min_template_score=5,
        require_volume_surge=False,
        require_base_pattern=False,
        require_breakout_ready=not args.no_require_breakout_ready,
        require_market_regime=False,
        allow_continuation_entry=args.allow_continuation_entry,
        buy_point_tolerance=args.buy_point_tolerance,
        trail_stop_pct=args.trail_stop,
        use_50dma_exit=not args.no_50dma_exit,
        overlay_enabled=args.overlay,
        overlay_rebalance_threshold=args.overlay_threshold,
        vol_target_enabled=args.vol_target,
        vol_target_annual=args.vol_target_level,
        vol_target_halflife_days=args.vol_target_halflife,
        min_breakout_volume_ratio=args.min_rvol,
        disable_breakouts_in_uptrend=args.disable_breakouts_in_uptrend,
        **(
            {
                "target_exposure_confirmed_uptrend": args.target_exposure,
                "target_exposure_uptrend_under_pressure": args.target_exposure,
                "target_exposure_market_correction": args.target_exposure,
            }
            if args.target_exposure is not None
            else {}
        ),
    )

    screener_config = MinerviniConfig(
        require_fundamentals=False,
        require_market_uptrend=False,
        max_stage_number=args.max_stage,
    )

    # Run
    backtester = WalkForwardBacktester(
        warehouse=warehouse,
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

    # Print summary
    summary = result.portfolio_result.summary
    if not summary:
        print("  No trades executed. Check if data is available in the warehouse.")
        print("  Run: python scripts/download_broad_universe.py --start 2022-01-01")
        return

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Starting capital:    ${summary.get('start_value', 0):>12,.2f}")
    print(f"  Ending capital:      ${summary.get('end_value', 0):>12,.2f}")
    print(f"  Strategy return:     {summary.get('total_return', 0):>12.2%}")
    print(f"  Benchmark return:    {summary.get('benchmark_return', 0):>12.2%}")
    beats = summary.get('total_return', 0) > summary.get('benchmark_return', 0)
    print(f"  Beats benchmark:     {'YES' if beats else 'NO':>12}")
    print(f"  Max drawdown:        {summary.get('max_drawdown', 0):>12.2%}")
    print(f"  Total trades:        {summary.get('total_trades', 0):>12}")
    print(f"  Win rate:            {summary.get('trade_win_rate', 0):>12.2%}")
    print(f"  Avg trade return:    {summary.get('avg_trade_return', 0):>12.2%}")
    print(f"  Symbols traded:      {summary.get('symbols_with_trades', 0):>12}")

    # Rebalance stats
    log = result.rebalance_log
    print(f"\n  Rebalance points:    {len(log)}")
    print(f"  Avg approved/screen: {log['approved_count'].mean():>12.1f}")
    print(f"  Max approved:        {log['approved_count'].max():>12}")
    print(f"  Min approved:        {log['approved_count'].min():>12}")

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve
    eq = result.portfolio_result.equity_curve
    if not eq.empty:
        eq.to_csv(results_dir / "equity_curve.csv", index=False)

    # Trades
    trades = result.portfolio_result.trades
    if not trades.empty:
        trades.to_csv(results_dir / "trades.csv", index=False)

    # Rebalance log
    log.to_csv(results_dir / "rebalance_log.csv", index=False)

    # Summary JSON
    summary_out = {
        **summary,
        "config": {
            "start_date": args.start,
            "end_date": args.end,
            "rebalance": args.rebalance,
            "seed_universe_size": len(seed_symbols),
            "initial_cash": args.initial_cash,
            "max_positions": args.max_positions,
            "min_rs_percentile": args.min_rs,
            "min_template_score": args.min_template_score,
        },
        "rebalance_stats": {
            "total_rebalances": len(log),
            "avg_approved": float(log["approved_count"].mean()),
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary_out, indent=2, default=str))

    print(f"\n  Results saved to: {results_dir}/")
    print()


if __name__ == "__main__":
    main()
