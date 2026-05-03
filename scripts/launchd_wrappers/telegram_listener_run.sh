#!/bin/bash
# Wrapper for the Telegram User-API listener launchd job.
# Sources .env so credentials never appear in the plist.

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

if [ -z "${TELEGRAM_API_ID:-}" ] || [ -z "${TELEGRAM_API_HASH:-}" ]; then
  echo "TELEGRAM_API_ID / TELEGRAM_API_HASH not set after sourcing $REPO/.env" >&2
  exit 1
fi
if [ -z "${CHAN_CHAT_IDS:-}" ]; then
  echo "CHAN_CHAT_IDS not set — listener has no chats to watch" >&2
  exit 1
fi
if [ ! -f "$REPO/results/.telegram_session.session" ]; then
  echo "Session file missing — run scripts/telegram_auth_setup.py first" >&2
  exit 1
fi

exec "$REPO/.venv/bin/python" "$REPO/scripts/run_telegram_listener.py"
