#!/bin/bash
# Wrapper for the Alpha Vantage earnings ingest launchd plist.
# Sources .env so launchd doesn't need API keys baked into the plist XML.
# Logs to results/pead/paper/av_ingest.{out,err}.log via plist redirects.

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

# Load .env (ALPHA_VANTAGE_INGEST_API_KEY is required — dedicated key
# isolated from the shared ALPHA_VANTAGE_API_KEY so other callers can't
# burn this 25/day quota.)
set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

if [ -z "${ALPHA_VANTAGE_INGEST_API_KEY:-}" ]; then
  echo "ALPHA_VANTAGE_INGEST_API_KEY not set after sourcing $REPO/.env" >&2
  exit 1
fi

# 25 fetches/night = AV free-tier daily limit. yfinance recent-reporter
# lookup is free (no AV quota). 13s delay = ~4.6 req/min, under AV's
# 5/min hard limit. Total runtime ~5-6 min: yfinance check (~1 min for
# 80 symbols) + 25 AV calls × 13s.
#
# Priority queue (rebuilt each night):
#   1. RECENT_REPORTER — symbols that actually reported in past 4 days
#                        (per yfinance earnings_dates). Highest priority.
#   2. NAN-REFRESH — events in DB w/ NULL surprise_pct, < 30d old
#   3. STALE-REFRESH — latest non-future event > 60 days old
#   4. BACKFILL — broad250 names not yet in earnings_events
# Step 1: AV ingest (paid-quality data, capped at 25/day on free tier).
# We don't `exec` so step 2 still runs even if AV is rate-limited.
"$REPO/.venv/bin/python" "$REPO/scripts/ingest_earnings_alphavantage.py" \
  --max-symbols 25 \
  --delay-seconds 13 \
  --universe "$REPO/research_data/intraday_top250_universe.json" \
  --db "$REPO/research_data/earnings_data.duckdb" || true

# Step 2: yfinance fallback for the recent-reporter list. Fill-only: never
# overwrites existing rows, so AV data always wins when present. This
# unblocks PEAD on days AV is rate-limited (the IP-level throttle from
# 2026-05-01 was the trigger for adding this).
exec "$REPO/.venv/bin/python" "$REPO/scripts/ingest_earnings_yfinance_recent.py" \
  --db "$REPO/research_data/earnings_data.duckdb" \
  --lookback-days 4
