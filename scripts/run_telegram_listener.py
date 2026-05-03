#!/usr/bin/env python3
"""Long-running Telegram User-API listener — entry point for launchd.

Reads config from env (.env via the wrapper sources it):
  TELEGRAM_API_ID
  TELEGRAM_API_HASH
  TELEGRAM_SESSION_PATH (default: results/.telegram_session)
  CHAN_CHAT_IDS (comma-separated)
  TELEGRAM_BACKFILL (default: 200, set 0 to skip)
  CHAT_CORPUS_DB (default: research_data/chat_corpus.duckdb)

Will run forever, reconnecting on transient errors. Crash → launchd
restarts via KeepAlive. Session file persists auth across restarts.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from tradingagents.research.telegram_listener import (  # noqa: E402
    CORPUS_DB_DEFAULT, SESSION_PATH_DEFAULT,
    _coerce_chat_ids, main_sync,
)


def main() -> int:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_id_raw = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id_raw or not api_hash:
        raise SystemExit(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set. "
            "Get from https://my.telegram.org/apps and add to .env."
        )

    session_path = os.environ.get("TELEGRAM_SESSION_PATH", SESSION_PATH_DEFAULT)
    chat_ids = _coerce_chat_ids(os.environ.get("CHAN_CHAT_IDS"))
    db_path = os.environ.get("CHAT_CORPUS_DB", CORPUS_DB_DEFAULT)
    backfill = int(os.environ.get("TELEGRAM_BACKFILL", "200"))

    main_sync(
        api_id=int(api_id_raw),
        api_hash=api_hash,
        session_path=session_path,
        chat_ids=chat_ids,
        db_path=db_path,
        backfill=backfill,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
