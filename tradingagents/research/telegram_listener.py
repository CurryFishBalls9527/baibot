"""Telegram User-API listener for personal-research chat ingestion.

Uses Telethon (User API) to lurk in groups the user is a member of and
write every message to a local DuckDB corpus. Bot API can't read groups
without admin-add permission; User API requires no such permission
(it's just "another client logged in as the user").

NOT a bot. Not added to groups. Not visible to other members. Just a
headless instance of the user's own account.

## Security notes (LOAD-BEARING)

The session file (`results/.telegram_session.session` by default) IS
the user's authenticated Telegram session. Anyone with that file can
post as them, read their DMs, etc. Treat as a password:
  - Never commit (already in .gitignore via `results/`)
  - Never copy off the local machine
  - Revoke from official Telegram client if it leaks
  - Ideally: dedicate a separate Telegram account for read-only research

## Storage

Corpus DB: `research_data/chat_corpus.duckdb` (separate from
market_data and earnings_data — different concern, different access
pattern, isolated to keep blast radius small).

Schema: `chan_chat_messages` table with idempotent UPSERT on
(chat_id, message_id) so re-runs don't duplicate.

## Env vars consumed

  TELEGRAM_API_ID           — from https://my.telegram.org/apps
  TELEGRAM_API_HASH         — same source
  TELEGRAM_SESSION_PATH     — default `results/.telegram_session` (no .session suffix)
  CHAN_CHAT_IDS             — comma-separated list of chat IDs to listen to
                              (use scripts/telegram_list_dialogs.py to find them)

See `scripts/telegram_auth_setup.py` for one-time interactive auth.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import duckdb

logger = logging.getLogger(__name__)

CORPUS_DB_DEFAULT = "research_data/chat_corpus.duckdb"
SESSION_PATH_DEFAULT = "results/.telegram_session"


_CREATE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS chan_chat_messages (
    chat_id        BIGINT       NOT NULL,
    message_id     BIGINT       NOT NULL,
    chat_title     VARCHAR,
    timestamp      TIMESTAMP    NOT NULL,
    author_id      BIGINT,
    author_username VARCHAR,
    author_display VARCHAR,
    text           VARCHAR,
    reply_to_msg_id BIGINT,
    raw_json       VARCHAR,
    ingested_at    TIMESTAMP    NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_chan_chat_messages_ts ON chan_chat_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_chan_chat_messages_author ON chan_chat_messages(author_id);
"""


def ensure_schema(db_path: str = CORPUS_DB_DEFAULT) -> None:
    """Create the chat_corpus table if missing. Idempotent."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    try:
        con.execute(_CREATE_TABLE_DDL)
    finally:
        con.close()


def _coerce_chat_ids(raw: Optional[str]) -> list[int]:
    """Parse CHAN_CHAT_IDS env var into list of ints. Empty/missing → []."""
    if not raw:
        return []
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            logger.warning("ignoring non-int chat id: %r", token)
    return out


def write_message(
    con: duckdb.DuckDBPyConnection,
    *,
    chat_id: int, message_id: int, chat_title: Optional[str],
    timestamp: datetime, author_id: Optional[int],
    author_username: Optional[str], author_display: Optional[str],
    text: Optional[str], reply_to_msg_id: Optional[int],
    raw_json: Optional[str],
) -> bool:
    """UPSERT one message. Returns True if newly inserted, False if dup."""
    existing = con.execute(
        "SELECT 1 FROM chan_chat_messages WHERE chat_id=? AND message_id=?",
        [chat_id, message_id],
    ).fetchone()
    con.execute(
        """
        INSERT OR REPLACE INTO chan_chat_messages
          (chat_id, message_id, chat_title, timestamp,
           author_id, author_username, author_display,
           text, reply_to_msg_id, raw_json, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            chat_id, message_id, chat_title, timestamp,
            author_id, author_username, author_display,
            text, reply_to_msg_id, raw_json,
            datetime.now(timezone.utc),
        ],
    )
    return existing is None


# ─── Telethon listener ────────────────────────────────────────────────


async def run_listener(
    api_id: int,
    api_hash: str,
    session_path: str,
    chat_ids: list[int],
    db_path: str = CORPUS_DB_DEFAULT,
    backfill_messages_per_chat: int = 200,
) -> None:
    """Long-running listener. Connects, optionally backfills recent
    history, then subscribes to live new-message events.

    `backfill_messages_per_chat`: on startup, pull this many recent
    messages per configured chat to fill any gap during downtime. Set
    to 0 to skip backfill.
    """
    from telethon import TelegramClient, events  # lazy

    if not chat_ids:
        raise SystemExit(
            "CHAN_CHAT_IDS env var is empty. Run "
            "scripts/telegram_list_dialogs.py first to find the chat IDs "
            "you want to listen to, then set CHAN_CHAT_IDS in .env."
        )

    ensure_schema(db_path)
    chat_id_set = set(chat_ids)
    logger.warning(
        "Telegram listener starting | session=%s | watching chat_ids=%s",
        session_path, chat_ids,
    )

    # Telethon connect; session file persists auth across restarts
    client = TelegramClient(session_path, api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        raise SystemExit(
            f"Session at {session_path} is not authorized. Run "
            "scripts/telegram_auth_setup.py first to do the one-time "
            "interactive login."
        )

    # ─── Optional backfill ─────────────────────────────────────────
    if backfill_messages_per_chat > 0:
        for chat_id in chat_ids:
            n_new = 0
            try:
                entity = await client.get_entity(chat_id)
                chat_title = getattr(entity, "title", None) or str(chat_id)
                con = duckdb.connect(db_path)
                try:
                    async for msg in client.iter_messages(
                        entity, limit=backfill_messages_per_chat,
                    ):
                        if not msg.text:
                            continue
                        sender = getattr(msg, "sender", None)
                        if write_message(
                            con,
                            chat_id=chat_id, message_id=msg.id,
                            chat_title=chat_title,
                            timestamp=msg.date,
                            author_id=getattr(sender, "id", None) if sender else None,
                            author_username=getattr(sender, "username", None) if sender else None,
                            author_display=(
                                f"{getattr(sender,'first_name','') or ''} "
                                f"{getattr(sender,'last_name','') or ''}".strip()
                                if sender else None
                            ),
                            text=msg.text,
                            reply_to_msg_id=getattr(msg, "reply_to_msg_id", None),
                            raw_json=None,  # skip on backfill — heavy
                        ):
                            n_new += 1
                finally:
                    con.close()
                logger.warning(
                    "Backfill chat %s (%s): %d new of %d most recent",
                    chat_id, chat_title, n_new, backfill_messages_per_chat,
                )
            except Exception as exc:
                logger.warning("Backfill failed for %s: %s", chat_id, exc)

    # ─── Live event subscription ───────────────────────────────────
    @client.on(events.NewMessage())
    async def _on_new(event):
        msg = event.message
        if msg.chat_id not in chat_id_set:
            return
        if not msg.text:
            return
        try:
            sender = await event.get_sender()
            chat = await event.get_chat()
            con = duckdb.connect(db_path)
            try:
                inserted = write_message(
                    con,
                    chat_id=msg.chat_id, message_id=msg.id,
                    chat_title=getattr(chat, "title", None) or str(msg.chat_id),
                    timestamp=msg.date,
                    author_id=getattr(sender, "id", None),
                    author_username=getattr(sender, "username", None),
                    author_display=(
                        f"{getattr(sender,'first_name','') or ''} "
                        f"{getattr(sender,'last_name','') or ''}".strip()
                        if sender else None
                    ),
                    text=msg.text,
                    reply_to_msg_id=getattr(msg, "reply_to_msg_id", None),
                    raw_json=None,
                )
            finally:
                con.close()
            if inserted:
                logger.info(
                    "[%s] %s: %s",
                    getattr(chat, "title", msg.chat_id),
                    getattr(sender, "username", "?"),
                    (msg.text or "")[:100].replace("\n", " "),
                )
        except Exception as exc:
            logger.warning("on_new handler error: %s", exc, exc_info=True)

    logger.warning("Subscribed to %d chats. Listening...", len(chat_ids))
    await client.run_until_disconnected()


def main_sync(
    api_id: int, api_hash: str, session_path: str,
    chat_ids: list[int], db_path: str,
    backfill: int = 200,
) -> None:
    """Sync entry point that wraps the asyncio listener."""
    asyncio.run(run_listener(
        api_id=api_id, api_hash=api_hash, session_path=session_path,
        chat_ids=chat_ids, db_path=db_path,
        backfill_messages_per_chat=backfill,
    ))
