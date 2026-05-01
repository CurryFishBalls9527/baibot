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

# 25 symbols/night = AV free-tier daily limit (5/min × 5 min ≈ comfortably under cap).
# 13s delay = ~4.6 req/min, under the 5/min hard limit.
# Hits backfill first (159 broad250 names without data), then rotates refresh
# (symbols whose latest event is > 60 days old).
exec "$REPO/.venv/bin/python" "$REPO/scripts/ingest_earnings_alphavantage.py" \
  --max-symbols 25 \
  --delay-seconds 13 \
  --universe "$REPO/research_data/intraday_top250_universe.json" \
  --db "$REPO/research_data/market_data.duckdb"
