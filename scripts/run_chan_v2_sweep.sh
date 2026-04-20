#!/usr/bin/env bash
# Run a list of chan v2 backtests in batches of N (default 3).
set -u
BATCH=${BATCH:-3}
mkdir -p results/chan_v2/logs

# Each line: <out_tag>|<extra args>
SWEEPS=(
  "cooldown_1|--sell-cooldown-bars 1"
  "cooldown_3|--sell-cooldown-bars 3"
  "cooldown_5|--sell-cooldown-bars 5"
  "cooldown_10|--sell-cooldown-bars 10"
  "cooldown_15|--sell-cooldown-bars 15"
  "cooldown_20|--sell-cooldown-bars 20"
  "breakeven_1_0|--breakeven-atr 1.0"
  "breakeven_1_5|--breakeven-atr 1.5"
  "breakeven_2_0|--breakeven-atr 2.0"
  "breakeven_3_0|--breakeven-atr 3.0"
  "grace_1|--stop-grace-bars 1"
  "grace_2|--stop-grace-bars 2"
  "grace_3|--stop-grace-bars 3"
  "grace_5|--stop-grace-bars 5"
  "macd_peak|--macd-algo peak"
  "macd_full_area|--macd-algo full_area"
  "macd_diff|--macd-algo diff"
  "macd_slope|--macd-algo slope"
)

i=0
total=${#SWEEPS[@]}
while [ $i -lt $total ]; do
  pids=()
  echo "=== batch starting at index $i ==="
  for j in 0 1 2; do
    idx=$((i+j))
    [ $idx -ge $total ] && break
    entry="${SWEEPS[$idx]}"
    tag="${entry%%|*}"
    extra="${entry#*|}"
    out="results/chan_v2/${tag}.json"
    log="results/chan_v2/logs/${tag}.log"
    if [ -f "$out" ]; then
      echo "  skip $tag (exists)"
      continue
    fi
    echo "  launching $tag :: $extra"
    .venv/bin/python scripts/run_chan_v2_backtest.py --no-regime $extra --out "$out" > "$log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  i=$((i+BATCH))
done
echo "=== sweep complete ==="
ls results/chan_v2/*.json | wc -l
