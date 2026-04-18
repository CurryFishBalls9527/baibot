#!/usr/bin/env python3
"""Minimal chan.py backtest: trade is_sure buy/sell signals, measure edge.

Walks each symbol bar-by-bar via step_load(). On a confirmed buy signal,
enters at the next bar's open. Exits on a confirmed sell signal, a time
stop (max_hold bars), or an ATR-based trailing stop. Reports win rate,
avg return, and P&L by bsp type.

Usage:
    python scripts/chan_spike_backtest.py --symbols AAPL MSFT NVDA
    python scripts/chan_spike_backtest.py --universe research_data/spike_universe.json
    python scripts/chan_spike_backtest.py --symbols SPY --begin 2024-01-01 --end 2024-12-31
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHAN_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "chan.py"
sys.path.insert(0, str(CHAN_ROOT))

from Chan import CChan  # noqa: E402
from ChanConfig import CChanConfig  # noqa: E402
from Common.CEnum import AUTYPE, KL_TYPE  # noqa: E402

from tradingagents.research.chan_adapter import DuckDBIntradayAPI  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chan_bt")


@dataclass
class Trade:
    symbol: str
    entry_bar: int
    entry_price: float
    entry_time: str
    entry_bsp_types: str
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    bars_held: int = 0
    pnl_pct: float = 0.0
    high_since_entry: float = 0.0
    entry_bi_low: float = 0.0


DEFAULT_CHAN_CONFIG = {
    "trigger_step": True,
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 1,
    "bs1_peak": False,
    "macd_algo": "peak",
    "bs_type": "1,2,3a,1p,2s,3b",
    "print_warning": False,
    "zs_algo": "normal",
}


def run_backtest(
    symbol: str,
    begin: str,
    end: str,
    db_path: str,
    max_hold_bars: int = 130,  # ~10 trading days of 30m bars
    trail_atr_mult: float = 3.0,
    atr_period: int = 14,
    require_sure: bool = True,
    buy_types: set | None = None,
    chan_config_overrides: dict | None = None,
    exit_mode: str = "atr_trail",  # "atr_trail" | "zs_structural"
) -> dict:
    """Walk one symbol and trade buy/sell signals."""
    DuckDBIntradayAPI.DB_PATH = db_path

    if buy_types is None:
        buy_types = {"T1", "T2", "T3A", "T1P", "T2S", "T3B"}

    chan_cfg = {**DEFAULT_CHAN_CONFIG}
    if chan_config_overrides:
        chan_cfg.update(chan_config_overrides)

    config = CChanConfig(chan_cfg)

    chan = CChan(
        code=symbol,
        begin_time=begin,
        end_time=end,
        data_src="custom:DuckDBAPI.DuckDB30mAPI",
        lv_list=[KL_TYPE.K_30M],
        config=config,
        autype=AUTYPE.QFQ,
    )

    trades: list[Trade] = []
    position: Optional[Trade] = None
    step = 0
    prices: list[float] = []
    seen_buy_sigs: set = set()
    seen_sell_sigs: set = set()

    t0 = time.perf_counter()

    for snapshot in chan.step_load():
        step += 1

        # Get current bar's close price from the latest kline unit
        try:
            lvl = snapshot[0]  # CKLine_List for first level (K_30M)
            if not lvl.lst:
                continue
            ckline = lvl.lst[-1]  # CKLine (combined kline)
            cur_klu = ckline.lst[-1]  # CKLine_Unit (raw bar)
            cur_close = float(cur_klu.close)
            cur_high = float(cur_klu.high)
            cur_low = float(cur_klu.low)
            cur_time = str(cur_klu.time)
        except Exception:
            continue

        prices.append(cur_close)

        # Compute simple ATR proxy (true range over recent bars)
        atr = _simple_atr(prices, atr_period)

        # Check for exit if in position
        if position is not None:
            position.bars_held += 1
            if cur_high > position.high_since_entry:
                position.high_since_entry = cur_high

            exit_reason = None

            if exit_mode == "zs_structural":
                # 1a. ZS structural stop: find valid ZS with low BELOW entry price
                zs_list = list(lvl.zs_list) if lvl.zs_list else []
                zs_stop = None
                for zs in reversed(zs_list):
                    zs_low = float(zs.low)
                    if zs_low < position.entry_price:
                        zs_stop = zs_low
                        break
                # 1b. Entry bi invalidation
                bi_stop = position.entry_bi_low if position.entry_bi_low > 0 else None

                structural_stop = None
                if zs_stop is not None and bi_stop is not None:
                    structural_stop = max(zs_stop, bi_stop)
                elif zs_stop is not None:
                    structural_stop = zs_stop
                elif bi_stop is not None:
                    structural_stop = bi_stop

                if structural_stop is not None and cur_low <= structural_stop:
                    exit_reason = "zs_break"
                    exit_price = max(structural_stop, cur_low)
                elif structural_stop is None and atr > 0:
                    safety_stop = position.high_since_entry - 5.0 * atr
                    if cur_low <= safety_stop:
                        exit_reason = "safety_stop"
                        exit_price = max(safety_stop, cur_low)
            else:
                # 1. Trailing stop: highest high since entry - N * ATR
                if atr > 0:
                    trail_stop = position.high_since_entry - trail_atr_mult * atr
                    if cur_low <= trail_stop:
                        exit_reason = "trail_stop"
                        exit_price = max(trail_stop, cur_low)

            # 2. Time stop (widened for zs mode)
            effective_max_hold = max_hold_bars if exit_mode == "atr_trail" else 200
            if exit_reason is None and position.bars_held >= effective_max_hold:
                exit_reason = "time_stop"
                exit_price = cur_close

            # 3. Sell signal from chan.py
            if exit_reason is None:
                try:
                    bsp_list = snapshot.get_latest_bsp(idx=0, number=50)
                except Exception:
                    bsp_list = []
                for bsp in bsp_list:
                    if bsp.is_buy:
                        continue
                    if require_sure and not bsp.bi.is_sure:
                        continue
                    sig_id = (bsp.klu.idx, False, tuple(t.name for t in bsp.type))
                    if sig_id in seen_sell_sigs:
                        continue
                    seen_sell_sigs.add(sig_id)
                    exit_reason = "sell_signal"
                    exit_price = cur_close
                    break

            if exit_reason:
                position.exit_bar = step
                position.exit_price = exit_price
                position.exit_time = cur_time
                position.exit_reason = exit_reason
                position.pnl_pct = (exit_price - position.entry_price) / position.entry_price
                trades.append(position)
                position = None

        # Check for entry if flat
        if position is None:
            try:
                bsp_list = snapshot.get_latest_bsp(idx=0, number=50)
            except Exception:
                bsp_list = []
            for bsp in bsp_list:
                if not bsp.is_buy:
                    continue
                if require_sure and not bsp.bi.is_sure:
                    continue
                types = {t.name for t in bsp.type}
                if not types.intersection(buy_types):
                    continue
                sig_id = (bsp.klu.idx, True, tuple(sorted(types)))
                if sig_id in seen_buy_sigs:
                    continue
                seen_buy_sigs.add(sig_id)

                entry_bi_low = 0.0
                try:
                    entry_bi_low = float(bsp.bi._low())
                except Exception:
                    pass

                position = Trade(
                    symbol=symbol,
                    entry_bar=step,
                    entry_price=cur_close,
                    entry_time=cur_time,
                    entry_bsp_types=",".join(sorted(types)),
                    high_since_entry=cur_high,
                    entry_bi_low=entry_bi_low,
                )
                break

    # Close any open position at end
    if position is not None:
        position.exit_bar = step
        position.exit_price = prices[-1] if prices else position.entry_price
        position.exit_time = cur_time if prices else ""
        position.exit_reason = "end_of_data"
        position.pnl_pct = (position.exit_price - position.entry_price) / position.entry_price
        trades.append(position)

    elapsed = time.perf_counter() - t0

    # Aggregate
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    total = len(trades)
    avg_ret = sum(t.pnl_pct for t in trades) / total if total > 0 else 0
    avg_win = (
        sum(t.pnl_pct for t in trades if t.pnl_pct > 0) / wins if wins else 0
    )
    losses = total - wins
    avg_loss = (
        sum(t.pnl_pct for t in trades if t.pnl_pct <= 0) / losses if losses else 0
    )
    total_ret = 1.0
    for t in trades:
        total_ret *= (1 + t.pnl_pct)
    total_ret -= 1.0

    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t.exit_reason].append(t.pnl_pct)

    by_type = defaultdict(list)
    for t in trades:
        for tp in t.entry_bsp_types.split(","):
            by_type[tp].append(t.pnl_pct)

    return {
        "symbol": symbol,
        "bars": step,
        "elapsed_sec": round(elapsed, 2),
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total, 4) if total else 0,
        "avg_return": round(avg_ret, 6),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "compounded_return": round(total_ret, 6),
        "avg_bars_held": round(sum(t.bars_held for t in trades) / total, 1) if total else 0,
        "by_exit_reason": {
            k: {"count": len(v), "avg_ret": round(sum(v) / len(v), 6)}
            for k, v in by_exit.items()
        },
        "by_entry_type": {
            k: {
                "count": len(v),
                "avg_ret": round(sum(v) / len(v), 6),
                "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4) if v else 0,
            }
            for k, v in by_type.items()
        },
        "trades": [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "pnl_pct": round(t.pnl_pct, 6),
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
                "bsp_types": t.entry_bsp_types,
            }
            for t in trades
        ],
    }


def _simple_atr(prices: list[float], period: int) -> float:
    """Quick ATR proxy from close-to-close true range."""
    if len(prices) < period + 1:
        return 0.0
    trs = [abs(prices[i] - prices[i - 1]) for i in range(-period, 0)]
    return sum(trs) / len(trs)


def parse_args():
    p = argparse.ArgumentParser(description="chan.py signal backtest")
    p.add_argument("--universe", default="research_data/spike_universe.json")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--begin", default="2023-01-01")
    p.add_argument("--end", default="2025-12-30")
    p.add_argument("--db", default="research_data/intraday_30m.duckdb")
    p.add_argument("--max-hold", type=int, default=130,
                    help="Max bars to hold (130 ≈ 10 trading days of 30m)")
    p.add_argument("--trail-atr", type=float, default=3.0)
    p.add_argument("--no-require-sure", action="store_true",
                    help="Trade unconfirmed signals too (dangerous)")
    p.add_argument("--buy-types", default="T1,T2,T3A,T1P,T2S,T3B",
                    help="Comma-separated bsp types to trade")
    p.add_argument("--exit-mode", default="atr_trail",
                    choices=["atr_trail", "zs_structural"],
                    help="Exit logic: atr_trail (3x ATR) or zs_structural (zhongshu-based)")
    p.add_argument("--out", default="results/chan_spike/backtest.json")
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

    # Remove BRK-B (not in warehouse)
    symbols = [s for s in symbols if s != "BRK-B"]

    buy_types = set(args.buy_types.split(","))

    log.info(
        "Chan.py backtest: %d symbols, %s to %s, buy_types=%s, trail=%.1fx ATR, max_hold=%d",
        len(symbols), args.begin, args.end, buy_types, args.trail_atr, args.max_hold,
    )

    all_results = []
    t_total = time.perf_counter()

    for i, sym in enumerate(symbols, 1):
        try:
            r = run_backtest(
                sym, args.begin, args.end, args.db,
                max_hold_bars=args.max_hold,
                trail_atr_mult=args.trail_atr,
                require_sure=not args.no_require_sure,
                buy_types=buy_types,
                exit_mode=args.exit_mode,
            )
            all_results.append(r)
            log.info(
                "  [%d/%d] %-6s %3d trades  WR %.1f%%  avg %.2f%%  comp %.2f%%  %.1fs",
                i, len(symbols), sym,
                r["total_trades"],
                r["win_rate"] * 100,
                r["avg_return"] * 100,
                r["compounded_return"] * 100,
                r["elapsed_sec"],
            )
        except Exception as e:
            log.warning("  [%d/%d] %-6s FAILED: %s", i, len(symbols), sym, e)

    total_elapsed = time.perf_counter() - t_total

    # Aggregate across all symbols
    all_trades_flat = [t for r in all_results for t in r["trades"]]
    total_trades = len(all_trades_flat)
    total_wins = sum(1 for t in all_trades_flat if t["pnl_pct"] > 0)
    avg_ret = sum(t["pnl_pct"] for t in all_trades_flat) / total_trades if total_trades else 0
    avg_win = (
        sum(t["pnl_pct"] for t in all_trades_flat if t["pnl_pct"] > 0) / total_wins
        if total_wins else 0
    )
    total_losses = total_trades - total_wins
    avg_loss = (
        sum(t["pnl_pct"] for t in all_trades_flat if t["pnl_pct"] <= 0) / total_losses
        if total_losses else 0
    )

    # By exit reason
    by_exit: dict[str, list] = defaultdict(list)
    for t in all_trades_flat:
        by_exit[t["exit_reason"]].append(t["pnl_pct"])

    # By entry bsp type
    by_type: dict[str, list] = defaultdict(list)
    for t in all_trades_flat:
        for tp in t["bsp_types"].split(","):
            by_type[tp].append(t["pnl_pct"])

    agg = {
        "symbols_run": len(all_results),
        "wall_clock_sec": round(total_elapsed, 2),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "win_rate": round(total_wins / total_trades, 4) if total_trades else 0,
        "avg_return": round(avg_ret, 6),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "avg_bars_held": round(
            sum(t["bars_held"] for t in all_trades_flat) / total_trades, 1
        ) if total_trades else 0,
        "by_exit_reason": {
            k: {
                "count": len(v),
                "avg_ret": round(sum(v) / len(v), 6),
                "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4),
            }
            for k, v in sorted(by_exit.items())
        },
        "by_entry_type": {
            k: {
                "count": len(v),
                "avg_ret": round(sum(v) / len(v), 6),
                "win_rate": round(sum(1 for x in v if x > 0) / len(v), 4),
            }
            for k, v in sorted(by_type.items())
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_symbol_summary = [
        {k: v for k, v in r.items() if k != "trades"} for r in all_results
    ]
    out_path.write_text(json.dumps({
        "aggregate": agg,
        "per_symbol": per_symbol_summary,
    }, indent=2))

    # Also save trade log
    trades_path = out_path.parent / "trades.json"
    trades_path.write_text(json.dumps(all_trades_flat, indent=2))

    _print_summary(agg, args)
    log.info("Saved to %s", out_path)


def _print_summary(agg, args):
    print()
    print("=" * 70)
    print("  CHAN.PY BACKTEST — RESULTS")
    print("=" * 70)
    print(f"  Symbols:          {agg['symbols_run']}")
    print(f"  Period:           {args.begin} to {args.end}")
    print(f"  Total trades:     {agg['total_trades']:,}")
    print(f"  Win rate:         {agg['win_rate'] * 100:.1f}%")
    print(f"  Avg return:       {agg['avg_return'] * 100:+.3f}%")
    print(f"  Avg winner:       {agg['avg_win'] * 100:+.3f}%")
    print(f"  Avg loser:        {agg['avg_loss'] * 100:+.3f}%")
    print(f"  Avg bars held:    {agg['avg_bars_held']:.1f}")
    print(f"  Wall clock:       {agg['wall_clock_sec']:.1f}s")
    print()
    print("  By exit reason:")
    for reason, stats in agg["by_exit_reason"].items():
        print(f"    {reason:<15} {stats['count']:>6} trades  "
              f"WR {stats['win_rate'] * 100:>5.1f}%  "
              f"avg {stats['avg_ret'] * 100:>+7.3f}%")
    print()
    print("  By entry bsp type:")
    for tp, stats in agg["by_entry_type"].items():
        print(f"    {tp:<6} {stats['count']:>6} trades  "
              f"WR {stats['win_rate'] * 100:>5.1f}%  "
              f"avg {stats['avg_ret'] * 100:>+7.3f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
