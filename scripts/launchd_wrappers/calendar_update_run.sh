#!/bin/bash
# Wrapper for the earnings calendar maintenance launchd plist.
# Sources .env (no API keys actually needed by yfinance, but consistent).
# Updates earnings_calendar table from yfinance (free, no quota).

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

set -a
# shellcheck disable=SC1091
source "$REPO/.env" 2>/dev/null || true
set +a

# Full broad250 universe (minus ETFs). yfinance ~1s/symbol → ~4 min total.
# Updates BOTH historical [-90d] and forecast [+180d] entries.
exec "$REPO/.venv/bin/python" "$REPO/scripts/update_earnings_calendar.py" \
  --universe "$REPO/research_data/intraday_top250_universe.json" \
  --db "$REPO/research_data/market_data.duckdb" \
  --lookback-days 90 \
  --lookahead-days 180 \
  --delay-ms 200
