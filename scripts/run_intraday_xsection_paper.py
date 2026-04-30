#!/usr/bin/env python3
"""Standalone live runner for the cross-sectional intraday paper trader.

This is a separate process from the main scheduler (`launchd
com.tradingagents.scheduler`). It uses its own Alpaca paper account so it
will NOT touch the existing mechanical / llm / chan / chan_v2 / intraday_v2
/ chan_daily variants.

Required env vars:
    ALPACA_XSECTION_API_KEY
    ALPACA_XSECTION_SECRET_KEY

Optional env var:
    XSECTION_PAPER_KILL=1     # graceful shutdown on next tick
    ALPACA_XSECTION_FEED=sip  # default IEX (free paper plans)

Default behavior is dry-run — orders are NOT submitted. Verify the JSONL
output, then re-launch with `--live-submit` to send real paper orders.

Example (dry-run, 30-min hold variant matching the cadence sweep):

    ALPACA_XSECTION_API_KEY=xxx ALPACA_XSECTION_SECRET_KEY=yyy \
    ./.venv/bin/python scripts/run_intraday_xsection_paper.py \
        --hold-minutes 30 --formation-minutes 60 \
        --n-long 10 --n-short 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time as time_mod
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

# Pre-warm tradingagents.automation BEFORE touching broker.alpaca_broker.
# AlpacaBroker imports `tradingagents.automation.events`, and if the automation
# package has not been loaded yet, broker/__init__.py → alpaca_broker.py →
# automation/__init__.py → orchestrator.py → broker.alpaca_broker (mid-load)
# trips a circular ImportError. Importing automation.config first fully
# resolves the automation package and breaks the cycle.
import tradingagents.automation.config  # noqa: E402,F401

from tradingagents.broker.alpaca_broker import AlpacaBroker  # noqa: E402
from tradingagents.research.intraday_xsection_backtester import (  # noqa: E402
    XSectionReversionConfig,
)
from tradingagents.research.intraday_xsection_paper_trader import (  # noqa: E402
    XSectionPaperTrader,
    XSectionPaperTraderConfig,
)


# Tick cadence: how often the loop wakes to check whether a rebalance bar has
# closed. 5 seconds gives plenty of headroom — Alpaca publishes the just-closed
# bar within a couple seconds, and we wait `bar_grace_seconds` (default 8) past
# the bar boundary before fetching.
LOOP_TICK_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone live paper trader for the cross-sectional intraday "
            "reversion strategy. Defaults to DRY-RUN — no orders are sent "
            "until --live-submit is passed."
        ),
    )
    p.add_argument("--universe", default="research_data/intraday_top250_universe.json")
    p.add_argument("--daily-db", default="research_data/market_data.duckdb")
    p.add_argument("--log-dir", default="results/intraday_xsection/paper")
    p.add_argument("--interval", type=int, default=15)
    p.add_argument("--formation-minutes", type=int, default=60)
    p.add_argument("--hold-minutes", type=int, default=30)
    p.add_argument("--n-long", type=int, default=10)
    p.add_argument("--n-short", type=int, default=10)
    p.add_argument("--target-gross-exposure", type=float, default=1.0)
    p.add_argument("--max-gross-exposure", type=float, default=0.5)
    p.add_argument("--min-dollar-volume-avg", type=float, default=5_000_000.0)
    p.add_argument("--earliest-rebalance", default="08:30",
                   help="HH:MM CDT — first eligible rebalance time")
    p.add_argument("--latest-rebalance", default="14:30",
                   help="HH:MM CDT — last eligible rebalance time")
    p.add_argument("--flatten-at", default="14:55",
                   help="HH:MM CDT — flush all positions at/after this time")
    p.add_argument("--bar-grace-seconds", type=float, default=8.0,
                   help="Seconds to wait after bar boundary before fetching")
    p.add_argument("--live-submit", action="store_true",
                   help="DANGER — actually submit orders. Default is dry-run.")
    p.add_argument("--max-rebalances", type=int, default=None,
                   help="Stop after N rebalance steps (smoke-test bound)")
    return p.parse_args()


def parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def load_symbols(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return payload["symbols"] if isinstance(payload, dict) else payload


def build_strategy(args: argparse.Namespace) -> XSectionReversionConfig:
    return XSectionReversionConfig(
        universe=args.universe,
        min_dollar_volume_avg=args.min_dollar_volume_avg,
        interval_minutes=args.interval,
        formation_minutes=args.formation_minutes,
        hold_minutes=args.hold_minutes,
        signal_direction="reversion",
        n_long=args.n_long,
        n_short=args.n_short,
        dollar_neutral=True,
        target_gross_exposure=args.target_gross_exposure,
        earliest_rebalance_time=parse_hhmm(args.earliest_rebalance),
        latest_rebalance_time=parse_hhmm(args.latest_rebalance),
        flatten_at_close_time=parse_hhmm(args.flatten_at),
    )


def build_live(args: argparse.Namespace) -> XSectionPaperTraderConfig:
    return XSectionPaperTraderConfig(
        log_dir=Path(args.log_dir),
        dry_run=not args.live_submit,
        max_gross_exposure=args.max_gross_exposure,
        bar_grace_seconds=args.bar_grace_seconds,
        universe_path=args.universe,
        daily_db_path=args.daily_db,
        alpaca_data_feed=os.getenv("ALPACA_XSECTION_FEED", "iex"),
    )


def build_broker() -> AlpacaBroker:
    api_key = os.environ.get("ALPACA_XSECTION_API_KEY")
    secret = os.environ.get("ALPACA_XSECTION_SECRET_KEY")
    if not api_key or not secret:
        raise SystemExit(
            "ALPACA_XSECTION_API_KEY and ALPACA_XSECTION_SECRET_KEY must be set. "
            "This must be a NEW Alpaca paper account, separate from the existing "
            "mechanical / llm / chan accounts."
        )
    return AlpacaBroker(api_key=api_key, secret_key=secret, paper=True)


def is_in_window(now_local: dtime, earliest: dtime, latest: dtime) -> bool:
    return earliest <= now_local <= latest


def floor_to_interval(ts: pd.Timestamp, interval_minutes: int) -> pd.Timestamp:
    """Floor to the nearest bar boundary (e.g., 9:31:42 → 9:30 for 15-min)."""
    minute_of_day = ts.hour * 60 + ts.minute
    floored_minute = (minute_of_day // interval_minutes) * interval_minutes
    return ts.replace(
        hour=floored_minute // 60,
        minute=floored_minute % 60,
        second=0,
        microsecond=0,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()

    broker = build_broker()
    strategy = build_strategy(args)
    live = build_live(args)
    symbols = load_symbols(args.universe)

    trader = XSectionPaperTrader(strategy, live, broker, symbols)
    trader.install_signal_handlers()

    flatten_time = parse_hhmm(args.flatten_at)
    earliest = parse_hhmm(args.earliest_rebalance)
    latest = parse_hhmm(args.latest_rebalance)
    bars_per_hold = max(1, strategy.hold_minutes // strategy.interval_minutes)

    mode = "LIVE-SUBMIT" if args.live_submit else "DRY-RUN"
    logging.warning(
        "xsection paper trader starting in %s mode | universe=%d symbols | "
        "n_long=%d n_short=%d | hold=%dmin | gross_cap=%.2f",
        mode, len(symbols), strategy.n_long, strategy.n_short,
        strategy.hold_minutes, live.max_gross_exposure,
    )

    last_rebalance_floor: pd.Timestamp | None = None
    rebalance_count = 0
    eod_flushed_for_date: pd.Timestamp | None = None

    while not trader.should_stop():
        time_mod.sleep(LOOP_TICK_SECONDS)
        # Use America/Chicago to match the backtester's bar timestamps.
        now_aware = pd.Timestamp.now(tz="America/Chicago")
        now_naive = now_aware.tz_localize(None)
        now_local = now_naive.time()
        today = pd.Timestamp(now_naive.normalize())

        # 1. Market-open guard. Use Alpaca clock (handles holidays).
        try:
            if not broker.is_market_open():
                continue
        except Exception as exc:
            logging.warning("is_market_open failed: %s", exc)
            continue

        # 2. EOD flatten gate. Once-per-day, atomic.
        if now_local >= flatten_time and eod_flushed_for_date != today:
            logging.info("EOD flatten window — closing all positions.")
            trader.run_eod_flatten(now_naive)
            eod_flushed_for_date = today
            continue

        # 3. Rebalance gate. Compute the bar we'd trade on FIRST, then check
        # the window against THAT bar's time — not wall-clock. Otherwise
        # `latest_rebalance=14:30` would skip the legitimate 14:30-bar trade
        # because by the time we process it (wall ≈ 14:45) we're past 14:30.
        # First: never open new positions after wall-clock EOD flatten time.
        # The window check is on bar-time, but the flatten guarantee is on
        # wall-clock — past flatten time we should be flat for the day.
        if now_local >= flatten_time:
            continue
        floor = floor_to_interval(now_naive, strategy.interval_minutes)
        seconds_past_close = (now_naive - floor).total_seconds()
        if seconds_past_close < live.bar_grace_seconds:
            continue

        # Rebalance on the JUST-CLOSED bar (one interval before floor), not the
        # just-opened bar. The just-opened bar is not yet aggregated for many
        # symbols at wall-clock floor + bar_grace, which causes master_index
        # to include floor only for a partial subset of the universe and
        # corrupts cross-sectional ranking. The just-closed bar is finalized
        # for all liquid symbols by floor + a few seconds.
        rebalance_bar = pd.Timestamp(floor) - timedelta(minutes=strategy.interval_minutes)

        # Window check on the rebalance bar's time, NOT wall-clock.
        if not is_in_window(rebalance_bar.time(), earliest, latest):
            continue
        if last_rebalance_floor == floor:
            continue
        if last_rebalance_floor is not None:
            bars_since = int(
                (floor - last_rebalance_floor).total_seconds()
                / 60
                / strategy.interval_minutes
            )
            if bars_since < bars_per_hold:
                # Wait until enough bars have passed. Do NOT bump
                # last_rebalance_floor here — that would reset the cadence
                # counter every bar boundary and the strategy would only
                # fire once per restart.
                continue

        logging.info("rebalance fire @ bar %s (wall=%s, %s)", rebalance_bar, floor, mode)
        try:
            decision = trader.run_one_rebalance(rebalance_bar)
        except Exception:
            logging.exception("run_one_rebalance crashed — staying in loop")
            last_rebalance_floor = floor
            continue
        last_rebalance_floor = floor
        rebalance_count += 1
        logging.info(
            "rebalance %d done — action=%s longs=%d shorts=%d",
            rebalance_count,
            decision.get("action"),
            len(decision.get("longs", [])) if isinstance(decision.get("longs"), list) else 0,
            len(decision.get("shorts", [])) if isinstance(decision.get("shorts"), list) else 0,
        )
        if args.max_rebalances is not None and rebalance_count >= args.max_rebalances:
            logging.warning("max_rebalances=%d reached — exiting", args.max_rebalances)
            break

    logging.warning("xsection paper trader stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
