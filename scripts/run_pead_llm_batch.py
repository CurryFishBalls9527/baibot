#!/usr/bin/env python3
"""Batch runner for PEAD-LLM analysis.

Two cron windows (see plan ethereal-strolling-rocket.md §C):
  --window amc  → runs at 17:30 CDT, analyzes yesterday-AMC + overnight reporters
                  (event_datetime in [yesterday 15:00, today 06:00))
  --window bmo  → runs at 08:10 CDT, analyzes today's pre-open reporters
                  (event_datetime in [today 06:00, today 09:30])

For each event, calls
:func:`tradingagents.research.pead_llm_analyzer.analyze_event` which always
writes a row to ``earnings_llm_decisions`` (success or failure). Re-running
on the same window is idempotent — UPSERT semantics on
``(symbol, event_date)``.

PEAD's morning fire (08:35 CDT) reads this table via INNER JOIN and only
enters positions where ``llm_decision = 'BUY'``.

Usage:
    ./.venv/bin/python scripts/run_pead_llm_batch.py --window amc
    ./.venv/bin/python scripts/run_pead_llm_batch.py --window bmo
    ./.venv/bin/python scripts/run_pead_llm_batch.py --window amc --max-symbols 3 --dry-run
    ./.venv/bin/python scripts/run_pead_llm_batch.py --window bmo \\
        --deep-model gpt-5.4-pro --quick-model gpt-5-mini

Environment:
    OPENAI_API_KEY (required — analyzer uses OpenAI provider)
    PEAD_DEEP_MODEL  (optional override of --deep-model default)
    PEAD_QUICK_MODEL (optional override of --quick-model default)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.research.pead_llm_analyzer import (  # noqa: E402
    EARNINGS_DB_DEFAULT,
    PEAD_DEEP_MODEL_DEFAULT,
    PEAD_QUICK_MODEL_DEFAULT,
    analyze_event,
    ensure_schema,
    load_window_events,
)

logger = logging.getLogger("pead_llm_batch")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--window", choices=("amc", "bmo"), required=True,
                   help="Which batch window to process")
    p.add_argument("--db", default=EARNINGS_DB_DEFAULT,
                   help="Path to earnings_data.duckdb")
    p.add_argument("--max-symbols", type=int, default=None,
                   help="Cap symbols processed (useful for cost-controlled testing)")
    p.add_argument("--deep-model", default=PEAD_DEEP_MODEL_DEFAULT,
                   help=f"Deep-think LLM (default {PEAD_DEEP_MODEL_DEFAULT})")
    p.add_argument("--quick-model", default=PEAD_QUICK_MODEL_DEFAULT,
                   help=f"Quick-think LLM (default {PEAD_QUICK_MODEL_DEFAULT})")
    p.add_argument("--surprise-min", type=float, default=5.0,
                   help="Minimum surprise%% to consider (matches PEAD gate)")
    p.add_argument("--surprise-max", type=float, default=50.0,
                   help="Max surprise%% (data-noise cap, matches PEAD gate)")
    p.add_argument("--dry-run", action="store_true",
                   help="Load events + log them, but do NOT call LLM or write cache")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY missing — required for live LLM calls. "
            "Use --dry-run to test the loader path without API access."
        )

    # Idempotent — schema lands on first run, no-op afterwards.
    ensure_schema(args.db)

    today = date.today()
    events = load_window_events(
        args.window,
        today=today,
        db_path=args.db,
        surprise_min=args.surprise_min,
        surprise_max=args.surprise_max,
    )

    if args.max_symbols:
        events = events[: args.max_symbols]

    logger.warning(
        "PEAD-LLM batch %s | window=%s | %d events | deep=%s quick=%s",
        "DRY-RUN" if args.dry_run else "LIVE",
        args.window, len(events), args.deep_model, args.quick_model,
    )

    if args.dry_run:
        for ev in events:
            logger.info(
                "  [dry] %s @ %s surprise=%+.2f%% time=%s",
                ev.symbol, ev.event_datetime.strftime("%Y-%m-%d %H:%M"),
                ev.surprise_pct or 0.0, ev.time_hint or "?",
            )
        logger.warning("DRY-RUN complete — 0 LLM calls, 0 cache writes.")
        return 0

    if not events:
        logger.warning("No events in window — nothing to analyze.")
        return 0

    counters = {"buy": 0, "sell": 0, "hold": 0, "error": 0, "no_decision": 0}
    cost_total = 0.0
    started = time.time()

    for i, ev in enumerate(events, 1):
        # BMO window has a 2h budget (06:30 → 08:30, 5 min before PEAD's
        # 08:35 fire). Bail at 115 min to leave a safety margin.
        if args.window == "bmo":
            elapsed_min = (time.time() - started) / 60.0
            if elapsed_min > 115:
                logger.warning(
                    "BMO budget at %.1f min — skipping remaining %d events to "
                    "let PEAD fire on time. They'll be missed today.",
                    elapsed_min, len(events) - i + 1,
                )
                break

        logger.info(
            "[%d/%d] analyzing %s (event %s, surprise=%+.2f%%)",
            i, len(events), ev.symbol,
            ev.event_datetime.strftime("%Y-%m-%d %H:%M"),
            ev.surprise_pct or 0.0,
        )
        result = analyze_event(
            ev,
            db_path=args.db,
            deep_model=args.deep_model,
            quick_model=args.quick_model,
        )
        cost_total += result.cost_estimate_usd
        if result.error:
            counters["error"] += 1
            logger.warning(
                "  → ERROR: %s (took %.1fs, cost ~$%.3f)",
                result.error.split("\n")[0][:140],
                result.duration_seconds, result.cost_estimate_usd,
            )
        elif result.llm_decision is None:
            counters["no_decision"] += 1
            logger.warning(
                "  → NO DECISION (parse failed); took %.1fs, cost ~$%.3f",
                result.duration_seconds, result.cost_estimate_usd,
            )
        else:
            counters[result.llm_decision.lower()] += 1
            logger.info(
                "  → %s | took %.1fs, cost ~$%.3f",
                result.llm_decision, result.duration_seconds,
                result.cost_estimate_usd,
            )

    elapsed = time.time() - started
    logger.warning(
        "Done. %d events in %.1f min. Decisions: %s. Total cost: ~$%.2f",
        len(events), elapsed / 60.0, counters, cost_total,
    )
    # Cost-overrun guard: alert above $20/day (per plan §risks #4).
    if cost_total > 20.0:
        logger.error(
            "DAILY COST OVERRUN: $%.2f exceeds $20 cap — investigate model "
            "selection or candidate-count spike.",
            cost_total,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
