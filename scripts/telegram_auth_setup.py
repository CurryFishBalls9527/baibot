#!/usr/bin/env python3
"""One-time interactive Telegram User-API auth.

Run this ONCE on the local machine to create the session file. After
that, the listener (run_telegram_listener.py) loads the session non-
interactively and runs forever.

Setup steps before running this:
  1. Visit https://my.telegram.org/apps and create an "application"
     (any name, any platform). Get back: api_id (int), api_hash (str).
  2. Add to .env:
       TELEGRAM_API_ID=<your api_id>
       TELEGRAM_API_HASH=<your api_hash>
  3. Run THIS script — interactive. Will prompt:
       - Phone number (international format, e.g. +14155551234)
       - SMS/Telegram code (from your phone)
       - 2FA password if you have one enabled
  4. After success, session file is at results/.telegram_session.session
     (gitignored via the umbrella results/ ignore).

After that, run scripts/telegram_list_dialogs.py to find chat IDs, then
add them to .env as CHAN_CHAT_IDS=<comma-separated>, then start the
listener via the launchd job.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from tradingagents.research.telegram_listener import (  # noqa: E402
    SESSION_PATH_DEFAULT,
)


async def _auth():
    from telethon import TelegramClient
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        print(
            "ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in "
            "your environment / .env. Get them from https://my.telegram.org/apps",
            file=sys.stderr,
        )
        sys.exit(1)
    session = os.environ.get("TELEGRAM_SESSION_PATH", SESSION_PATH_DEFAULT)
    Path(session).parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(session, int(api_id), api_hash)
    print(f"Connecting (session at {session})...")
    # client.start() handles all the interactive prompts
    await client.start()
    me = await client.get_me()
    print()
    print(f"✓ Authenticated as: {me.first_name} {me.last_name or ''} "
          f"(@{me.username or 'no_username'}, id={me.id})")
    print(f"✓ Session saved to {session}.session")
    print()
    print("Next steps:")
    print("  1. Run: .venv/bin/python scripts/telegram_list_dialogs.py")
    print("     → find the Chan group(s) you want to listen to, copy chat ID(s)")
    print("  2. Add to .env:")
    print("       CHAN_CHAT_IDS=<comma-separated chat ids, e.g. -1001234567890>")
    print("  3. Install + load the listener launchd job")
    await client.disconnect()


def main() -> int:
    load_dotenv()
    asyncio.run(_auth())
    return 0


if __name__ == "__main__":
    sys.exit(main())
