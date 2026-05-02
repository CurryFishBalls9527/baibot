#!/bin/bash
# Wrapper for the PEAD LLM-gated TREATMENT arm of the A/B forward test.
#
# Fires Mon-Fri 08:35 CDT (same time as control arm com.baibot.pead).
# Reads earnings_llm_decisions populated by com.baibot.pead_llm_amc/bmo.
# Runs in DRY-RUN mode — no broker submissions. Logs to a separate dir
# so the control arm's results/pead/paper/ stays untouched.
#
# After 60-90 trading days, compare cumulative P&L of:
#   Control:   results/pead/paper/        (existing, --no-llm-gate, live)
#   Treatment: results/pead/llm_dryrun/   (this wrapper, --require-llm-buy, dry)
# Promote treatment to live-submit only if it shows positive Sharpe delta
# net of LLM costs (~$1-3/day).

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

# Step 1: PEAD strategy run with LLM gate ON, in DRY-RUN. Don't `exec`
# so step 2 still runs even if step 1 errors.
"$REPO/.venv/bin/python" "$REPO/scripts/run_pead_paper.py" \
  --account-prefix PEAD \
  --min-surprise-pct 5.0 \
  --hold-days 20 \
  --position-pct 0.05 \
  --max-concurrent 10 \
  --max-gross-exposure 0.5 \
  --log-dir "$REPO/results/pead/llm_dryrun" \
  --require-llm-buy || true

# Step 2: mirror to a SEPARATE dashboard DB (trading_pead_llm.db) so the
# unified dashboard can show the LLM-gated arm's hypothetical P&L
# alongside the live control arm's actual P&L. Reads --state-dir from
# the treatment arm's log dir, NOT the control arm's.
exec "$REPO/.venv/bin/python" "$REPO/scripts/sync_pead_to_dashboard.py" \
  --state-dir "$REPO/results/pead/llm_dryrun" \
  --db "$REPO/trading_pead_llm.db"
