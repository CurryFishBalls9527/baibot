#!/usr/bin/env python3
"""Daily-bar Chan strategy on the 8-ETF futures-proxy universe.

Universe: SPY, QQQ, IWM, DIA, GLD, SLV, TLT, USO.
Periods: 2023-2025 IS, 2020 OOS, 2018 OOS.

Fresh strategy module — see tradingagents/research/chan_daily_backtester.py.
Daily Chan signals (T1 + T3 buys), structural-or-ATR stop, time stop at 60
days, fills at next-day open. Uses market_data.duckdb for daily bars.

Usage:
    python scripts/run_chan_daily_etf_backtest.py
    python scripts/run_chan_daily_etf_backtest.py --periods 2020
    python scripts/run_chan_daily_etf_backtest.py --buy-types T1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.chan_daily_backtester import (
    ChanDailyBacktestConfig,
    PortfolioChanDailyBacktester,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_daily_etf")

UNIVERSE_PATH = "research_data/etf_futures_proxy_universe.json"

PERIODS = {
    "2023_2025": ("2023-01-01", "2025-12-30"),
    "2020_2022": ("2020-01-01", "2022-12-30"),
    "2017_2019": ("2017-01-01", "2019-12-30"),
    "2014_2016": ("2014-01-01", "2016-12-30"),
    # Single-year windows for annual decomposition + 2026 fresh-data test
    "2014":      ("2014-01-01", "2014-12-30"),
    "2015":      ("2015-01-01", "2015-12-30"),
    "2016":      ("2016-01-01", "2016-12-30"),
    "2017":      ("2017-01-01", "2017-12-30"),
    "2018":      ("2018-01-01", "2018-12-30"),
    "2019":      ("2019-01-01", "2019-12-30"),
    "2020":      ("2020-01-01", "2020-12-30"),
    "2021":      ("2021-01-01", "2021-12-30"),
    "2022":      ("2022-01-01", "2022-12-30"),
    "2023":      ("2023-01-01", "2023-12-30"),
    "2024":      ("2024-01-01", "2024-12-30"),
    "2025":      ("2025-01-01", "2025-12-30"),
    "2026_ytd":  ("2026-01-01", "2026-04-24"),
}

# Each Chan run needs lookback for bi/seg structure to mature.
# Pull ~250 trading days (~14 months) before the backtest start.
LOOKBACK_DAYS = 400


def _make_config(args: argparse.Namespace) -> ChanDailyBacktestConfig:
    sell_types_str = args.sell_types if args.sell_types else args.buy_types
    return ChanDailyBacktestConfig(
        initial_cash=args.cash,
        max_positions=args.max_positions,
        position_pct=args.position_pct,
        sizing_mode=args.sizing_mode,
        risk_per_trade=args.risk_per_trade,
        chan_bs_type=args.bs_type,
        buy_types=tuple(args.buy_types.split(",")),
        sell_types=tuple(sell_types_str.split(",")),
        enable_longs=not args.no_longs,
        enable_shorts=not args.no_shorts,
        chan_macd_algo=args.macd_algo,
        chan_divergence_rate=args.divergence_rate,
        chan_min_zs_cnt=args.min_zs_cnt,
        chan_bi_strict=not args.no_bi_strict,
        chan_bsp2_follow_1=not args.no_bsp2_follow_1,
        chan_bsp3_follow_1=not args.no_bsp3_follow_1,
        chan_bsp3_peak=args.bsp3_peak,
        chan_strict_bsp3=args.strict_bsp3,
        chan_bsp3a_max_zs_cnt=args.bsp3a_max_zs_cnt,
        chan_zs_algo=args.zs_algo,
        atr_period=args.atr_period,
        stop_atr_mult=args.stop_atr_mult,
        time_stop_bars=args.time_stop_bars,
        trend_sma_period=args.trend_sma,
        momentum_rank_lookback=args.momentum_lookback,
        momentum_rank_top_k=args.momentum_top_k,
        entry_mode=args.entry_mode,
        donchian_period=args.donchian_period,
        exit_on_sell_signal=not args.no_sell_signal_exit,
        require_sure=not args.no_require_sure,
        entry_lag_extra_days=args.entry_lag_extra_days,
        kline_level=args.kline_level,
        weekly_filter_enabled=args.weekly_filter,
        weekly_filter_mode=args.weekly_filter_mode,
        segseg_filter_enabled=args.segseg_filter,
        zs_divergence_required=args.require_zs_divergence,
        exit_long_on_zs_broken=args.exit_on_zs_broken,
        momentum_filter_seg_branch=args.momentum_filter_seg_branch,
        trailing_stop_enabled=args.trailing_stop,
        trail_breakeven_r=args.trail_breakeven_r,
        trail_at_r=args.trail_at_r,
        trail_atr_mult=args.trail_atr_mult,
        partial_exit_enabled=args.partial_exit,
        partial_at_r=args.partial_at_r,
        partial_pct=args.partial_pct,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        vix_filter_enabled=args.vix_filter,
        vix_block_threshold=args.vix_threshold,
        vol_scale_enabled=args.vol_scale,
        vol_scale_target_pct=args.vol_scale_target,
        entry_priority_mode=args.entry_priority,
        volume_confirm_enabled=args.volume_confirm,
        volume_confirm_lookback=args.volume_confirm_lookback,
        volume_confirm_mult=args.volume_confirm_mult,
        trend_type_filter_mode=args.trend_type_filter,
        require_bi_up_at_entry=args.require_bi_up,
        equity_dd_threshold_pct=args.equity_dd_threshold,
        equity_dd_resume_pct=args.equity_dd_resume,
        equity_sector_max_positions=args.equity_sector_cap,
        donchian_breakout_min_pct=args.donchian_breakout_min_pct,
        pyramid_enabled=args.pyramid,
        pyramid_thresholds_r=tuple(float(x) for x in args.pyramid_thresholds.split(",")),
        pyramid_add_fractions=tuple(float(x) for x in args.pyramid_fractions.split(",")),
        pyramid_donchian_only=args.pyramid_donchian_only,
        pyramid_require_up_segseg_sure=args.pyramid_require_up_segseg_sure,
        pyramid_require_zs_broken=args.pyramid_require_zs_broken,
        stop_atr_mult_seg=args.stop_atr_mult_seg,
        time_stop_bars_seg=args.time_stop_bars_seg,
        credit_spread_filter_enabled=args.credit_spread_filter,
        credit_spread_numerator=args.credit_spread_numerator,
        credit_spread_denominator=args.credit_spread_denominator,
        credit_spread_lookback=args.credit_spread_lookback,
        credit_spread_block_mode=args.credit_spread_block_mode,
        credit_spread_z_threshold=args.credit_spread_z_threshold,
        credit_spread_drop_pct=args.credit_spread_drop_pct,
        calendar_filter_mode=args.calendar_filter,
        calendar_block_months=tuple(int(m) for m in args.calendar_block_months.split(",") if m.strip()),
        vol_adaptive_exit_mode=args.vol_adaptive_exit,
        vol_expansion_ratio=args.vol_expansion_ratio,
        vol_tightened_atr_mult=args.vol_tightened_atr_mult,
        reentry_after_stop_enabled=args.reentry_after_stop,
        reentry_window_bars=args.reentry_window_bars,
        reentry_max_count=args.reentry_max_count,
        portfolio_vol_target=args.portfolio_vol_target,
        portfolio_vol_lookback=args.portfolio_vol_lookback,
    )


def _expand_begin(begin: str, kline_level: str = "daily") -> str:
    """Expand begin date by lookback so Chan structure matures by the IS start.

    Daily: LOOKBACK_DAYS (400 cal ≈ 280 trading bars).
    Weekly: 730 cal days ≈ 104 weekly bars warmup. (Bounded by data availability;
    for the 2014 IS we get only ~52 weekly bars since data starts 2013-01-02.)
    """
    import pandas as pd
    days = 730 if kline_level == "weekly" else LOOKBACK_DAYS
    return (pd.Timestamp(begin) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _filter_to_period(result, begin: str, end: str):
    """Drop trades and equity points before the IS start so warm-up isn't counted."""
    import pandas as pd

    begin_ts = pd.Timestamp(begin)
    if not result.trades.empty:
        result.trades = result.trades[
            pd.to_datetime(result.trades["entry_date"]) >= begin_ts
        ].reset_index(drop=True)
    if not result.equity_curve.empty:
        eq = result.equity_curve.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        result.equity_curve = eq[eq["date"] >= begin_ts].reset_index(drop=True)
        # Recompute summary on filtered slice
        if not result.equity_curve.empty:
            initial = float(result.equity_curve["equity"].iloc[0])
            final = float(result.equity_curve["equity"].iloc[-1])
            running_max = result.equity_curve["equity"].cummax()
            dd = (result.equity_curve["equity"] / running_max - 1.0) * 100.0
            result.summary["initial_cash"] = initial
            result.summary["final_equity"] = final
            result.summary["total_return_pct"] = (final / initial - 1.0) * 100.0
            result.summary["max_drawdown_pct"] = float(dd.min())
    if not result.trades.empty:
        win = result.trades["return"] > 0
        result.summary["total_trades"] = int(len(result.trades))
        result.summary["win_rate"] = float(win.mean())
        result.summary["avg_return"] = float(result.trades["return"].mean())
        wins = result.trades[win]
        losses = result.trades[~win]
        result.summary["avg_win"] = float(wins["return"].mean()) if not wins.empty else 0.0
        result.summary["avg_loss"] = float(losses["return"].mean()) if not losses.empty else 0.0
        result.summary["avg_bars_held"] = float(result.trades["bars_held"].mean())
        result.summary["by_exit_reason"] = {
            r: {
                "count": int(len(g)),
                "win_rate": float((g["return"] > 0).mean()),
                "avg_ret": float(g["return"].mean()),
            }
            for r, g in result.trades.groupby("exit_reason")
        }
        result.summary["by_entry_type"] = {
            t: {
                "count": int(len(g)),
                "win_rate": float((g["return"] > 0).mean()),
                "avg_ret": float(g["return"].mean()),
            }
            for t, g in result.trades.groupby("bsp_types")
        }
    return result


def _print_period(period: str, summary: dict, sym_summary) -> None:
    s = summary
    dd_abs = abs(s["max_drawdown_pct"])
    rdd = (s["total_return_pct"] / dd_abs) if dd_abs > 0 else float("nan")
    print()
    print("=" * 70)
    print(f"  CHAN DAILY ETF — PERIOD {period}")
    print("=" * 70)
    print(f"  Initial capital:       ${s['initial_cash']:,.0f}")
    print(f"  Final equity:          ${s['final_equity']:,.2f}")
    print(f"  Total return:          {s['total_return_pct']:+.2f}%")
    print(f"  Max drawdown:          {s['max_drawdown_pct']:.2f}%")
    print(f"  Return / |DD|:         {rdd:+.2f}")
    print(f"  Total trades:          {s['total_trades']}")
    print(f"  Win rate:              {s['win_rate']*100:.1f}%")
    print(f"  Avg return:            {s['avg_return']*100:+.3f}%")
    if s["total_trades"] > 0:
        print(f"  Avg winner:            {s['avg_win']*100:+.3f}%")
        print(f"  Avg loser:             {s['avg_loss']*100:+.3f}%")
        print(f"  Avg bars held:         {s['avg_bars_held']:.1f}")
    if s.get("by_exit_reason"):
        print("  By exit reason:")
        for r, st in s["by_exit_reason"].items():
            print(f"    {r:<14} {st['count']:>4}  WR {st['win_rate']*100:>5.1f}%  avg {st['avg_ret']*100:>+7.3f}%")
    if s.get("by_entry_type"):
        print("  By entry type:")
        for t, st in s["by_entry_type"].items():
            print(f"    {t:<14} {st['count']:>4}  WR {st['win_rate']*100:>5.1f}%  avg {st['avg_ret']*100:>+7.3f}%")
    if sym_summary is not None and not sym_summary.empty:
        print("  By symbol:")
        for _, row in sym_summary.sort_values("total_pnl", ascending=False).iterrows():
            print(f"    {row['symbol']:<5} {row['trades']:>3} trades  PnL ${row['total_pnl']:>+8,.0f}  WR {row['win_rate']*100:.0f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily Chan ETF backtest")
    p.add_argument("--periods", nargs="*",
                   default=["2023_2025", "2020_2022", "2017_2019", "2014_2016"],
                   choices=list(PERIODS.keys()))
    p.add_argument("--symbols", nargs="*",
                   help="Override universe (default: load etf_futures_proxy_universe.json)")
    p.add_argument("--cash", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=4)
    p.add_argument("--position-pct", type=float, default=0.10)
    p.add_argument("--sizing-mode", default="fixed", choices=["fixed", "atr_parity"])
    p.add_argument("--risk-per-trade", type=float, default=0.005,
                   help="atr_parity only: equity at risk per trade (default 0.5%%)")
    p.add_argument("--bs-type", default="1,2,2s,3a,3b")
    p.add_argument("--buy-types", default="T1,T2,T2S,T3A,T3B")
    p.add_argument("--no-bi-strict", action="store_true",
                   help="Relax bi formation rules (more bis, more BSPs)")
    p.add_argument("--entry-lag-extra-days", type=int, default=0,
                   help="Future-blanked probe: delay entry execution by N extra days")
    p.add_argument("--macd-algo", default="area")
    p.add_argument("--divergence-rate", type=float, default=0.6)
    p.add_argument("--min-zs-cnt", type=int, default=1)
    p.add_argument("--no-bsp2-follow-1", action="store_true",
                   help="Allow second buy/sell points without a prior first point (small-to-large reversal probe).")
    p.add_argument("--no-bsp3-follow-1", action="store_true",
                   help="Allow third buy/sell points without a prior first point.")
    p.add_argument("--bsp3-peak", action="store_true",
                   help="Require third buy/sell breakout leg to clear the ZS peak/trough.")
    p.add_argument("--strict-bsp3", action="store_true",
                   help="Require the third buy/sell ZS to be adjacent to its related first point.")
    p.add_argument("--bsp3a-max-zs-cnt", type=int, default=1,
                   help="Maximum ZS count crossed by 3a buy/sell detection.")
    p.add_argument("--zs-algo", default="normal", choices=["normal", "over_seg", "auto"],
                   help="Chan ZS algorithm: normal=inside segment, over_seg=cross-segment, auto=mixed.")
    p.add_argument("--atr-period", type=int, default=20)
    p.add_argument("--stop-atr-mult", type=float, default=2.0)
    p.add_argument("--time-stop-bars", type=int, default=60)
    p.add_argument("--trend-sma", type=int, default=0,
                   help="Trend-overlay gate: only take Chan signals when aligned with SMA(N). "
                        "0=off; 200=require close>SMA(200) for longs.")
    p.add_argument("--momentum-lookback", type=int, default=0,
                   help="Cross-sectional momentum overlay: rank symbols by N-day return. 0=off.")
    p.add_argument("--momentum-top-k", type=int, default=4,
                   help="When momentum-lookback>0, only take entries on top-K names.")
    p.add_argument("--entry-mode", default="chan_bsp", choices=["chan_bsp", "donchian", "seg_bsp", "any_bsp", "donchian_or_seg", "zs_boundary", "donchian_seg_zs"],
                   help="Entry signal source. donchian=N-day-high breakout. seg_bsp=segment-level Chan BSPs. donchian_or_seg=combo. zs_boundary=ZS-edge. donchian_seg_zs=triple union.")
    p.add_argument("--momentum-filter-seg-branch", action="store_true",
                   help="In donchian_or_seg mode, also apply momentum filter to seg-branch entries (default: seg bypasses momentum).")
    p.add_argument("--donchian-period", type=int, default=20,
                   help="When entry-mode=donchian, lookback for breakout high/low.")
    p.add_argument("--no-sell-signal-exit", action="store_true")
    p.add_argument("--no-shorts", action="store_true",
                   help="Long-only mode (skip short entries on sell BSPs)")
    p.add_argument("--no-longs", action="store_true",
                   help="Short-only mode (for diagnostic)")
    p.add_argument("--sell-types", default=None,
                   help="Sell BSP types that trigger shorts (default: same as buy-types)")
    p.add_argument("--no-require-sure", action="store_true",
                   help="Act on BSPs on the live (unsure) bi — more signals, more noise")
    p.add_argument("--kline-level", default="daily", choices=["daily", "weekly"],
                   help="Bar level for Chan signals. weekly resamples daily on the fly.")
    p.add_argument("--weekly-filter", action="store_true",
                   help="Multi-level: gate daily entries by weekly Chan direction (only meaningful with kline-level=daily).")
    p.add_argument("--weekly-filter-mode", default="seg", choices=["seg", "bi", "confirmed_seg"],
                   help="Source of weekly direction. seg=last seg dir (default), bi=last bi dir (faster), confirmed_seg=last is_sure seg (laggier).")
    p.add_argument("--segseg-filter", action="store_true",
                   help="Gate entries by lvl.segseg_list[-1].dir (chan.py-internal higher-level structure; same data source as base level).")
    p.add_argument("--require-zs-divergence", action="store_true",
                   help="Only allow long entries when most recent ZS shows MACD divergence (T1-quality reversal filter).")
    p.add_argument("--exit-on-zs-broken", action="store_true",
                   help="Treat ZS end_bi_break as additional long exit signal (中枢被破出场).")
    p.add_argument("--trailing-stop", action="store_true",
                   help="Enable trailing stop: move to breakeven at +trail_breakeven_r, then ATR-trail.")
    p.add_argument("--trail-breakeven-r", type=float, default=2.0)
    p.add_argument("--trail-at-r", type=float, default=4.0)
    p.add_argument("--trail-atr-mult", type=float, default=1.5)
    p.add_argument("--partial-exit", action="store_true",
                   help="Take partial off at +partial_at_r profit.")
    p.add_argument("--partial-at-r", type=float, default=2.0)
    p.add_argument("--partial-pct", type=float, default=0.5)
    p.add_argument("--slippage-bps", type=float, default=5.0,
                   help="One-side slippage in basis points (default 5).")
    p.add_argument("--commission-bps", type=float, default=1.0,
                   help="Commission per side in basis points (default 1).")
    p.add_argument("--vix-filter", action="store_true",
                   help="Block new entries when VIX > vix_threshold")
    p.add_argument("--vix-threshold", type=float, default=30.0,
                   help="VIX level above which entries are blocked (default 30)")
    p.add_argument("--credit-spread-filter", action="store_true",
                   help="Block new entries when credit-spread proxy ratio is risk-off")
    p.add_argument("--credit-spread-numerator", default="HYG")
    p.add_argument("--credit-spread-denominator", default="LQD")
    p.add_argument("--credit-spread-lookback", type=int, default=60)
    p.add_argument("--credit-spread-block-mode", default="below_sma",
                   choices=["below_sma", "negative_delta", "z_below"])
    p.add_argument("--credit-spread-z-threshold", type=float, default=-0.5)
    p.add_argument("--credit-spread-drop-pct", type=float, default=0.0)
    p.add_argument("--calendar-filter", default="off",
                   choices=["off", "sell_in_may", "santa_only", "block_months"],
                   help="Calendar gate. sell_in_may=block May-Oct, santa_only=Nov-Jan only.")
    p.add_argument("--calendar-block-months", default="",
                   help="Comma-separated month numbers to block (when --calendar-filter block_months)")
    p.add_argument("--vol-adaptive-exit", default="off",
                   choices=["off", "tighten_stop", "exit"],
                   help="Vol-adaptive exit overlay. tighten_stop=raise stop on vol expansion, exit=close at next open.")
    p.add_argument("--vol-expansion-ratio", type=float, default=1.5,
                   help="ATR(today)/ATR(entry) ratio threshold for vol-adaptive exit")
    p.add_argument("--vol-tightened-atr-mult", type=float, default=1.0,
                   help="ATR multiplier for tightened stop in vol-adaptive tighten_stop mode")
    p.add_argument("--reentry-after-stop", action="store_true",
                   help="Allow re-entry after stop if price reclaims entry within reentry_window_bars")
    p.add_argument("--reentry-window-bars", type=int, default=5)
    p.add_argument("--reentry-max-count", type=int, default=1)
    p.add_argument("--portfolio-vol-target", type=float, default=0.0,
                   help="Annualized vol target (e.g., 0.12 = 12%%); 0=disabled")
    p.add_argument("--portfolio-vol-lookback", type=int, default=30)
    p.add_argument("--vol-scale", action="store_true",
                   help="Scale per-trade risk inversely to recent ATR/price (target/actual, clipped 0.5-2x)")
    p.add_argument("--vol-scale-target", type=float, default=0.02,
                   help="Target ATR/price ratio (default 2%%)")
    p.add_argument("--entry-priority", default="fifo", choices=["fifo", "momentum", "rank_random"],
                   help="When cash binds, which signal gets funded first. fifo=alphabetical (current), momentum=strongest first.")
    p.add_argument("--volume-confirm", action="store_true",
                   help="Donchian breakout requires volume > N-day avg × M (CTA-style confirmation)")
    p.add_argument("--volume-confirm-lookback", type=int, default=20)
    p.add_argument("--volume-confirm-mult", type=float, default=1.5)
    p.add_argument("--trend-type-filter", default="off",
                   choices=["off", "trend_only", "up_segseg_only", "up_trend_strict"],
                   help="走势类型 gate. trend_only=block during active ZS (consolidation). "
                        "up_segseg_only=require confirmed up segseg. "
                        "up_trend_strict=both.")
    p.add_argument("--require-bi-up", action="store_true",
                   help="同级别分解: only enter when last bi is confirmed UP "
                        "(pullback bottomed). Avoids chasing mid-decline.")
    p.add_argument("--equity-dd-threshold", type=float, default=0.0,
                   help="Pause entries when account DD vs high-water > X (e.g. 0.05 for 5%%). 0=off")
    p.add_argument("--equity-dd-resume", type=float, default=0.0,
                   help="Resume entries when DD recovers below X (e.g. 0.03 for 3%%). 0=off")
    p.add_argument("--equity-sector-cap", type=int, default=0,
                   help="Max concurrent positions in equity-correlated group (SPY/QQQ/IWM/DIA/XL*). 0=off")
    p.add_argument("--donchian-breakout-min-pct", type=float, default=0.0,
                   help="Donchian breakout requires close > Donchian-high × (1+X). 0.005 = 0.5%% margin")
    p.add_argument("--pyramid", action="store_true",
                   help="Pyramid scale-in: add to winners as MFE crosses thresholds")
    p.add_argument("--pyramid-thresholds", default="1.5,3.0",
                   help="R-multiple thresholds for adds, comma-sep")
    p.add_argument("--pyramid-fractions", default="0.5,0.33",
                   help="Fraction of initial shares to add at each threshold")
    p.add_argument("--pyramid-donchian-only", action="store_true",
                   help="Conditional pyramid: only add for Donchian-branch entries")
    p.add_argument("--pyramid-require-up-segseg-sure", action="store_true",
                   help="Conditional pyramid: only add when entry segseg dir=up + sure")
    p.add_argument("--pyramid-require-zs-broken", action="store_true",
                   help="Conditional pyramid: only add when entry zs_broken=True")
    p.add_argument("--stop-atr-mult-seg", type=float, default=0.0,
                   help="Per-signal stop: seg-bsp branch ATR multiplier (0=use main)")
    p.add_argument("--time-stop-bars-seg", type=int, default=0,
                   help="Per-signal time stop: seg-bsp branch bar count (0=use main)")
    p.add_argument("--out-dir", default="results/chan_daily_etf")
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
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for period in args.periods:
        is_begin, is_end = PERIODS[period]
        load_begin = _expand_begin(is_begin, args.kline_level)
        log.info("=" * 70)
        log.info("PERIOD %s | IS=[%s, %s] | warmup=%s | symbols=%d",
                 period, is_begin, is_end, load_begin, len(symbols))
        log.info("=" * 70)
        bt = PortfolioChanDailyBacktester(cfg)
        result = bt.backtest_portfolio(symbols, load_begin, is_end)
        result = _filter_to_period(result, is_begin, is_end)

        _print_period(period, result.summary, result.symbol_summary)

        out_path = out_dir / f"{period}.json"
        out_path.write_text(json.dumps({
            "period": period,
            "summary": result.summary,
            "config": {
                "max_positions": cfg.max_positions,
                "position_pct": cfg.position_pct,
                "buy_types": list(cfg.buy_types),
                "bs_type": cfg.chan_bs_type,
                "bsp2_follow_1": cfg.chan_bsp2_follow_1,
                "bsp3_follow_1": cfg.chan_bsp3_follow_1,
                "bsp3_peak": cfg.chan_bsp3_peak,
                "strict_bsp3": cfg.chan_strict_bsp3,
                "bsp3a_max_zs_cnt": cfg.chan_bsp3a_max_zs_cnt,
                "zs_algo": cfg.chan_zs_algo,
                "atr_period": cfg.atr_period,
                "stop_atr_mult": cfg.stop_atr_mult,
                "time_stop_bars": cfg.time_stop_bars,
                "exit_on_sell_signal": cfg.exit_on_sell_signal,
                "warmup_lookback_days": LOOKBACK_DAYS,
            },
            "symbols": symbols,
            "trades": result.trades.to_dict("records") if not result.trades.empty else [],
        }, indent=2, default=str))
        log.info("Wrote %s", out_path)
        summaries.append({"period": period, "summary": result.summary})

    print()
    print("=" * 70)
    print("  CROSS-PERIOD SUMMARY")
    print("=" * 70)
    print(f"  {'Period':<12} {'Return':>10} {'MaxDD':>10} {'R/|DD|':>8} {'Trades':>8} {'WR':>7}")
    for r in summaries:
        s = r["summary"]
        dd_abs = abs(s["max_drawdown_pct"])
        rdd = (s["total_return_pct"] / dd_abs) if dd_abs > 0 else float("nan")
        print(f"  {r['period']:<12} {s['total_return_pct']:>+9.2f}% {s['max_drawdown_pct']:>9.2f}% "
              f"{rdd:>+7.2f} {s['total_trades']:>8} {s['win_rate']*100:>6.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
