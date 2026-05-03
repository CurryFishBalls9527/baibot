#!/usr/bin/env python3
"""List all chats/channels/groups your Telegram account is in. Use the
output to pick chat IDs for the CHAN_CHAT_IDS env var.

Run AFTER scripts/telegram_auth_setup.py has created the session file.

Usage:
    .venv/bin/python scripts/telegram_list_dialogs.py
    .venv/bin/python scripts/telegram_list_dialogs.py --filter chan
    .venv/bin/python scripts/telegram_list_dialogs.py --groups-only

Output is sorted by recency (most-recent activity first), so the chats
you actively use float to the top. Negative IDs are groups/channels;
positive IDs are private chats with users.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from tradingagents.research.telegram_listener import (  # noqa: E402
    SESSION_PATH_DEFAULT,
)


async def _list(args):
    from telethon import TelegramClient
    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    session = os.environ.get("TELEGRAM_SESSION_PATH", SESSION_PATH_DEFAULT)
    client = TelegramClient(session, api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        print(f"Session at {session} not authorized. "
              f"Run scripts/telegram_auth_setup.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"{'kind':<10} {'chat_id':>16}  title  (@username)")
    print("-" * 80)
    n = 0
    async for d in client.iter_dialogs():
        kind = (
            "channel" if d.is_channel
            else "group" if d.is_group
            else "user" if d.is_user
            else "?"
        )
        if args.groups_only and kind not in ("group", "channel"):
            continue
        title = d.title or d.name or ""
        if args.filter and args.filter.lower() not in title.lower():
            continue
        username = getattr(d.entity, "username", None)
        username_str = f" (@{username})" if username else ""
        print(f"{kind:<10} {d.id:>16}  {title}{username_str}")
        n += 1
    print()
    print(f"  {n} dialogs shown")
    print()
    print("To listen to a chat: copy its `chat_id` (negative number for "
          "groups/channels) into .env as:")
    print("  CHAN_CHAT_IDS=-1001234567890,-1009876543210")
    await client.disconnect()


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--filter", help="Substring filter on title (case-insensitive)")
    p.add_argument("--groups-only", action="store_true",
                   help="Hide private user chats; show only groups/channels")
    asyncio.run(_list(p.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
