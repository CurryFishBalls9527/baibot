#!/bin/bash
# Wrapper for the PEAD launchd plist.
# Sources .env so launchd doesn't need API keys baked into the plist XML.
# Logs to results/pead/paper/launchd.{out,err}.log via the plist redirects.

set -euo pipefail

REPO="/Users/myu/code/baibot"
cd "$REPO"

# Load .env so ALPACA_PEAD_* are available
set -a
# shellcheck disable=SC1091
source "$REPO/.env"
set +a

# Sanity check
if [ -z "${ALPACA_PEAD_API_KEY:-}" ] || [ -z "${ALPACA_PEAD_SECRET_KEY:-}" ]; then
  echo "ALPACA_PEAD_* env vars not set after sourcing $REPO/.env" >&2
  exit 1
fi

exec "$REPO/.venv/bin/python" "$REPO/scripts/run_pead_paper.py" \
  --account-prefix PEAD \
  --min-surprise-pct 5.0 \
  --hold-days 20 \
  --position-pct 0.05 \
  --max-concurrent 10 \
  --max-gross-exposure 0.5 \
  --log-dir "$REPO/results/pead/paper" \
  --live-submit
