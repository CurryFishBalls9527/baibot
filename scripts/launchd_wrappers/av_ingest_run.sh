#!/bin/bash
# Wrapper for the Alpha Vantage earnings ingest launchd plist.
# Sources .env so launchd doesn't need API keys baked into the plist XML.
# Logs to results/pead/paper/av_ingest.{out,err}.log via plist redirects.

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

# Load .env (ALPHA_VANTAGE_API_KEY is required)
set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

if [ -z "${ALPHA_VANTAGE_API_KEY:-}" ] && [ -z "${ALPHAVANTAGE_API_KEY:-}" ]; then
  echo "ALPHA_VANTAGE_API_KEY not set after sourcing $REPO/.env" >&2
  exit 1
fi

# Max 24 fetches/night so the EARNINGS_CALENDAR call (1 quota) at the
# start fits within AV's 25/day free-tier limit. 13s delay = ~4.6 req/min,
# under the 5/min hard limit.
#
# Priority queue (rebuilt each night):
#   1. Calendar — symbols expected to report in next 3 days (per AV)
#   2. NaN-refresh — events in DB w/ NULL surprise_pct, < 30d old
#   3. Stale-refresh — latest non-future event > 60 days old
#   4. Backfill — broad250 names not yet in earnings_events
exec "$REPO/.venv/bin/python" "$REPO/scripts/ingest_earnings_alphavantage.py" \
  --max-symbols 24 \
  --delay-seconds 13 \
  --universe "$REPO/research_data/intraday_top250_universe.json" \
  --db "$REPO/research_data/market_data.duckdb"
