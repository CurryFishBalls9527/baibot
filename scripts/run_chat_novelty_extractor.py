#!/usr/bin/env python3
"""CLI runner for the weekly chat novelty extractor.

Usage:
    .venv/bin/python scripts/run_chat_novelty_extractor.py
    .venv/bin/python scripts/run_chat_novelty_extractor.py --days 14
    .venv/bin/python scripts/run_chat_novelty_extractor.py --model gpt-5.4-pro
    .venv/bin/python scripts/run_chat_novelty_extractor.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from tradingagents.research.chat_novelty_extractor import (  # noqa: E402
    CORPUS_DB_DEFAULT, MEMORY_INDEX_DEFAULT, NOVELTY_OUT_DEFAULT,
    _filter_messages, _load_recent_messages, run_extraction,
)


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7,
                   help="Window size (default 7d). Saturday job uses 7.")
    p.add_argument("--db", default=CORPUS_DB_DEFAULT)
    p.add_argument("--out-dir", default=NOVELTY_OUT_DEFAULT)
    p.add_argument("--memory-path", default=MEMORY_INDEX_DEFAULT)
    p.add_argument("--model", default="gpt-5.4",
                   help="Override LLM model (default gpt-5.4).")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip LLM call; just report message counts after filter.")
    args = p.parse_args()

    if args.dry_run:
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        raw = _load_recent_messages(args.db, start, end)
        filtered = _filter_messages(raw)
        print(f"Window {start.date()} → {end.date()}")
        print(f"  raw messages: {len(raw)}")
        print(f"  after noise-filter: {len(filtered)}")
        print(f"  estimated prompt tokens: ~{sum(len(m['text']) for m in filtered) // 4}")
        if filtered[:3]:
            print("\n  First 3 surviving messages (preview):")
            for m in filtered[:3]:
                print(f"    [{m['timestamp']}] @{m.get('author_username','?')}: {m['text'][:100]}")
        return 0

    result = run_extraction(
        end=None,  # = now
        days=args.days,
        db_path=args.db,
        out_dir=args.out_dir,
        memory_path=args.memory_path,
        model=args.model,
    )
    print(f"Wrote digest: {result.digest_path}")
    print(f"  iso_week: {result.iso_week}")
    print(f"  messages: {result.n_messages_after_filter} of {result.n_messages_raw}")
    print(f"  model: {result.model}")
    print(f"  cost (est): ${result.cost_estimate_usd:.3f}")
    print(f"  duration: {result.duration_s:.1f}s")
    if result.error:
        print(f"  ERROR: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
