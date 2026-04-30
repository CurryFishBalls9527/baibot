#!/usr/bin/env python3
"""Run the intraday cross-sectional mean-reversion backtester.

Default mode runs the broad250 universe across the standard intraday periods:
2023_2025, 2020, and 2018.

Example:
    ./.venv/bin/python scripts/run_intraday_xsection_backtest.py \
        --formation-minutes 60 \
        --hold-minutes 60 \
        --n-long 20 \
        --n-short 20 \
        --target-gross-exposure 1.5 \
        --sector-neutral \
        --formation-lag-bars 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.intraday_xsection_backtester import (  # noqa: E402
    XSectionReversionBacktester,
    XSectionReversionConfig,
)


DEFAULT_PERIODS = {
    "2023_2025": (
        "2023-01-01",
        "2025-12-30",
        "research_data/intraday_15m.duckdb",
    ),
    "2020": (
        "2020-01-01",
        "2020-12-31",
        "research_data/intraday_15m_2020.duckdb",
    ),
    "2018": (
        "2018-01-01",
        "2018-12-31",
        "research_data/intraday_15m_2018.duckdb",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intraday cross-sectional mean-reversion backtests."
    )
    parser.add_argument(
        "--universe",
        default="research_data/intraday_top250_universe.json",
        help="JSON universe path (default: broad250 intraday universe)",
    )
    parser.add_argument(
        "--periods",
        nargs="*",
        default=list(DEFAULT_PERIODS.keys()),
        choices=list(DEFAULT_PERIODS.keys()),
        help="Named default periods to run",
    )
    parser.add_argument("--daily-db", default="research_data/market_data.duckdb")
    parser.add_argument("--cash", type=float, default=100_000.0)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--adv-lookback-days", type=int, default=20)
    parser.add_argument("--max-position-pct-of-adv", type=float, default=None)
    parser.add_argument("--formation-minutes", type=int, default=60)
    parser.add_argument("--hold-minutes", type=int, default=60)
    parser.add_argument(
        "--tilt-refresh-minutes",
        type=int,
        default=None,
        help="Optional cadence for re-sizing the directional bias without re-ranking names",
    )
    parser.add_argument("--formation-lag-bars", type=int, default=0)
    parser.add_argument(
        "--signal-direction",
        choices=["reversion", "momentum"],
        default="reversion",
    )
    parser.add_argument("--n-long", type=int, default=20)
    parser.add_argument("--n-short", type=int, default=20)
    parser.add_argument(
        "--target-gross-exposure",
        type=float,
        default=1.0,
        help=(
            "When --dollar-neutral is on, 1.0 means 100%% long + 100%% short "
            "(200%% gross)."
        ),
    )
    parser.add_argument("--min-dollar-volume-avg", type=float, default=5_000_000.0)
    parser.add_argument("--sector-neutral", action="store_true")
    parser.add_argument("--market-context-min-score", type=int, default=None)
    parser.add_argument("--market-context-max-score", type=int, default=None)
    parser.add_argument("--market-context-qqq-above-ema21-pct-min", type=float, default=None)
    parser.add_argument("--market-context-qqq-above-ema21-pct-max", type=float, default=None)
    parser.add_argument("--market-context-qqq-roc-5-min", type=float, default=None)
    parser.add_argument("--market-context-qqq-roc-5-max", type=float, default=None)
    parser.add_argument(
        "--intraday-market-tilt-symbol",
        default=None,
        help="Optional same-day proxy symbol (for example SPY or QQQ) used to lean long/short",
    )
    parser.add_argument(
        "--intraday-market-tilt-threshold",
        type=float,
        default=0.0,
        help="Minimum absolute session return in the proxy before applying a tilt",
    )
    parser.add_argument(
        "--intraday-market-tilt-strength",
        type=float,
        default=0.0,
        help=(
            "Directional lean magnitude in [0,1]. With 0.25 and dollar-neutral sizing, "
            "a bullish signal becomes 125%% long / 75%% short."
        ),
    )
    parser.add_argument(
        "--intraday-market-tilt-strong-threshold",
        type=float,
        default=None,
        help="Optional stronger state threshold for the same-day proxy return",
    )
    parser.add_argument(
        "--intraday-market-tilt-strong-strength",
        type=float,
        default=None,
        help="Optional stronger state lean magnitude used beyond the strong threshold",
    )
    parser.add_argument(
        "--allowed-market-regime",
        action="append",
        choices=["confirmed_uptrend", "uptrend_under_pressure", "market_correction"],
        default=None,
        help="Repeatable prior-session regime filter",
    )
    parser.add_argument(
        "--no-dollar-neutral",
        action="store_true",
        help="Disable equal long/short side notional sizing",
    )
    parser.add_argument("--half-spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--short-borrow-bps-per-day", type=float, default=5.0)
    parser.add_argument(
        "--out",
        default="results/intraday_xsection/backtest.json",
        help="Output JSON artifact path",
    )
    parser.add_argument(
        "--include-trades",
        action="store_true",
        help="Include full trade blotters in the output JSON",
    )
    return parser.parse_args()


def load_symbols(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return payload["symbols"] if isinstance(payload, dict) else payload


def build_config(args: argparse.Namespace) -> XSectionReversionConfig:
    return XSectionReversionConfig(
        initial_cash=args.cash,
        universe=args.universe,
        min_dollar_volume_avg=args.min_dollar_volume_avg,
        adv_lookback_days=args.adv_lookback_days,
        max_position_pct_of_adv=args.max_position_pct_of_adv,
        sector_neutral=bool(args.sector_neutral),
        market_context_min_score=args.market_context_min_score,
        market_context_max_score=args.market_context_max_score,
        market_context_qqq_above_ema21_pct_min=args.market_context_qqq_above_ema21_pct_min,
        market_context_qqq_above_ema21_pct_max=args.market_context_qqq_above_ema21_pct_max,
        market_context_qqq_roc_5_min=args.market_context_qqq_roc_5_min,
        market_context_qqq_roc_5_max=args.market_context_qqq_roc_5_max,
        intraday_market_tilt_symbol=args.intraday_market_tilt_symbol,
        intraday_market_tilt_threshold=args.intraday_market_tilt_threshold,
        intraday_market_tilt_strength=args.intraday_market_tilt_strength,
        intraday_market_tilt_strong_threshold=args.intraday_market_tilt_strong_threshold,
        intraday_market_tilt_strong_strength=args.intraday_market_tilt_strong_strength,
        allowed_market_regimes=tuple(args.allowed_market_regime or ()),
        interval_minutes=args.interval,
        formation_minutes=args.formation_minutes,
        hold_minutes=args.hold_minutes,
        tilt_refresh_minutes=args.tilt_refresh_minutes,
        formation_lag_bars=args.formation_lag_bars,
        signal_direction=args.signal_direction,
        n_long=args.n_long,
        n_short=args.n_short,
        dollar_neutral=not bool(args.no_dollar_neutral),
        target_gross_exposure=args.target_gross_exposure,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
        short_borrow_bps_per_day=args.short_borrow_bps_per_day,
    )


def main() -> int:
    args = parse_args()
    symbols = load_symbols(args.universe)
    cfg = build_config(args)
    backtester = XSectionReversionBacktester(cfg)

    rows: list[dict] = []
    details: dict[str, dict] = {}

    for period_name in args.periods:
        begin, end, db_path = DEFAULT_PERIODS[period_name]
        print(f"-> {period_name} / {db_path}", flush=True)
        result = backtester.backtest(
            symbols=symbols,
            begin=begin,
            end=end,
            intraday_db_path=db_path,
            daily_db_path=args.daily_db,
        )
        summary = result.summary
        rdd = (
            summary["total_return_pct"] / abs(summary["max_drawdown_pct"])
            if summary.get("max_drawdown_pct")
            else 0.0
        )
        print(
            f"  return {summary['total_return_pct']:+8.2f}%  "
            f"dd {summary['max_drawdown_pct']:>7.2f}%  "
            f"R/DD {rdd:+6.2f}  "
            f"trades {summary['total_trades']:>6}",
            flush=True,
        )

        row = {
            "period": period_name,
            "begin": begin,
            "end": end,
            "db_path": db_path,
            "rdd": round(rdd, 4),
            **summary,
        }
        rows.append(row)

        details[period_name] = {
            "summary": summary,
            "config": asdict(cfg),
            "equity_curve": result.equity_curve.to_dict("records"),
        }
        if args.include_trades:
            details[period_name]["trades"] = [
                asdict(trade) for trade in result.trades
            ]

    print()
    print(
        f"{'period':<12}{'ret%':>9}{'DD%':>8}{'R/DD':>8}{'trades':>8}{'gross':>8}"
    )
    print("-" * 53)
    for row in rows:
        print(
            f"{row['period']:<12}"
            f"{row['total_return_pct']:>+9.2f}"
            f"{row['max_drawdown_pct']:>8.2f}"
            f"{row['rdd']:>8.2f}"
            f"{row['total_trades']:>8d}"
            f"{row.get('avg_gross_exposure', 0.0):>8.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary_rows": rows,
                "details": details,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
