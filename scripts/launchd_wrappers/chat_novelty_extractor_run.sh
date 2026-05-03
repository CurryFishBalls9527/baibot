#!/bin/bash
# Wrapper for the weekly chat novelty extractor.
# Sat 10:00 CDT (1h after weekly_review fires at 09:00 ET = 08:00 CDT,
# leaves enough headroom). Reads chat_corpus.duckdb for last 7 days,
# runs gpt-5.4 against the memory file, writes digest to
# results/chat_novelty/<isoweek>.md.

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY not set after sourcing $REPO/.env" >&2
  exit 1
fi

exec "$REPO/.venv/bin/python" "$REPO/scripts/run_chat_novelty_extractor.py" \
  --days 7
