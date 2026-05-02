#!/bin/bash
# Wrapper for the PEAD-LLM AMC batch (17:30 CDT Mon-Fri).
# Sources .env so the launchd plist doesn't need API keys baked in.
# Logs to results/pead/llm/amc.{out,err}.log via plist redirects.
#
# Analyzes earnings events with event_datetime in [yesterday 15:00, today 06:00)
# — i.e. yesterday-AMC + overnight reporters. Plenty of overnight time budget.
#
# Reads / writes research_data/earnings_data.duckdb earnings_llm_decisions.

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

# Default models per plan ethereal-strolling-rocket.md §B (gpt-5.4-pro deep,
# gpt-5-mini quick). Override via env var: PEAD_DEEP_MODEL, PEAD_QUICK_MODEL.
exec "$REPO/.venv/bin/python" "$REPO/scripts/run_pead_llm_batch.py" \
  --window amc \
  --db "$REPO/research_data/earnings_data.duckdb"
