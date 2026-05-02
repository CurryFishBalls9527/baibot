#!/bin/bash
# Wrapper for the PEAD end-of-day dashboard sync.
# Runs at 16:30 ET (15:30 CDT) — after market close, after Alpaca settles
# the day's P&L. Mirrors PEAD's positions.json + Alpaca PEAD account
# state into trading_pead.db so the dashboard's daily_snapshot for PEAD
# reflects EOD numbers (not the morning numbers from the 08:35 run).

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

if [ -z "${ALPACA_PEAD_API_KEY:-}" ] || [ -z "${ALPACA_PEAD_SECRET_KEY:-}" ]; then
  echo "ALPACA_PEAD_* env vars not set after sourcing $REPO/.env" >&2
  exit 1
fi

exec "$REPO/.venv/bin/python" "$REPO/scripts/sync_pead_to_dashboard.py" \
  --state-dir "$REPO/results/pead/paper" \
  --db "$REPO/trading_pead.db"
