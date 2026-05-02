#!/bin/bash
# Wrapper for the PEAD-LLM BMO batch (06:30 CDT Mon-Fri).
# Schedule: 30 min after AV ingest at 06:00, 2h before PEAD's 08:35 fire.
#
# Model split (vs AMC):
#   AMC uses gpt-5.4-pro deep + gpt-5-mini quick (~20 min/candidate, fine
#     overnight).
#   BMO uses gpt-5.4 (NON-PRO) deep + gpt-5-mini quick — gpt-5.4-pro's
#     reasoning tokens add ~5min/judge call which would not fit even the
#     2h budget reliably for busy days. gpt-5.4 brings runtime to ~5min/
#     candidate (~24 candidates fit in 2h budget) at ~$0.10/candidate
#     instead of $0.20.
#
# Analyzes earnings events with event_datetime in [today 06:00, today 09:30].
# Typical day: 0-8 names. The 2h budget is the durable fix for the
# old 25-min overflow risk.

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

# BMO-specific: faster non-reasoning model for deep_think. Keeps the
# AMC default (gpt-5.4-pro) untouched. Override if user wants the slower
# but stronger model — e.g. PEAD_DEEP_MODEL=gpt-5.4-pro launchctl ...
export PEAD_DEEP_MODEL="${PEAD_DEEP_MODEL:-gpt-5.4}"

exec "$REPO/.venv/bin/python" "$REPO/scripts/run_pead_llm_batch.py" \
  --window bmo \
  --db "$REPO/research_data/earnings_data.duckdb" \
  --deep-model "$PEAD_DEEP_MODEL"
