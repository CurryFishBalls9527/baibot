# TradingAgents

Multi-agent LLM trading framework using LangGraph. Specialized AI agents (analysts, researchers, traders, risk managers) collaborate to make trading decisions.

## Architecture

```
Scheduler → Orchestrator → AI Analysis (LangGraph) → Signal → Sizing → Risk Check → Alpaca Order
```

### Core Workflow (per symbol)
1. **4 Analysts** (market/social/news/fundamentals) → reports
2. **Bull & Bear Researchers** → debate (configurable rounds)
3. **Research Manager** (judge) → investment decision
4. **Trader** → BUY/SELL/HOLD proposal
5. **Risk Debaters** (aggressive/neutral/conservative) → risk discussion
6. **Risk Manager** → approve/reject
7. **Signal Processor** → structured JSON signal
8. **Position Sizer + Risk Engine** → hard risk gates → Alpaca bracket order

### Key Directories
- `tradingagents/agents/` — All agent definitions (analysts, researchers, trader, managers)
- `tradingagents/automation/` — Scheduler, orchestrator, config
- `tradingagents/broker/` — Alpaca integration (paper + live)
- `tradingagents/dataflows/` — Data sources (yfinance, Alpha Vantage, Alpaca)
- `tradingagents/graph/` — LangGraph workflow, signal processing, reflection
- `tradingagents/llm_clients/` — Multi-provider LLM support (OpenAI, Anthropic, Google, xAI, Ollama)
- `tradingagents/risk/` — Hard risk gates (position limits, drawdown, daily loss)
- `tradingagents/portfolio/` — Position sizing, portfolio tracking
- `tradingagents/storage/` — SQLite database for trades, signals, memories

## LLM Configuration

Default config in `tradingagents/default_config.py`. Automation overrides in `tradingagents/automation/config.py`.

| Context | deep_think_llm | quick_think_llm | Provider |
|---------|---------------|-----------------|----------|
| Research/Manual | gpt-5.2 | gpt-5-mini | openai |
| Automation | gpt-4o-mini | gpt-4o-mini | openai |

Supports: OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama. Change via `llm_provider` config key.

## Data Sources

Configured via `data_vendors` dict in config. Default: all yfinance.
- **Stock OHLCV**: yfinance / Alpaca
- **Technical indicators**: stockstats (computed locally, no API)
- **Fundamentals**: yfinance / Alpha Vantage
- **News/Sentiment**: yfinance / Alpha Vantage
- **Real-time prices**: Alpaca (`get_alpaca_latest_price()`)

## Entry Points

```bash
python run_trading.py run          # Run analysis now
python run_trading.py schedule     # Start automated scheduler
python run_trading.py status       # Account status
python run_trading.py close-all    # Emergency close
python run_trading.py trades       # Recent trades
```

Or programmatically:
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
ta = TradingAgentsGraph(debug=True)
state, decision = ta.propagate("NVDA", "2026-01-15")
```

## Environment Variables

```
OPENAI_API_KEY=       # Required if using OpenAI
ANTHROPIC_API_KEY=    # Required if using Anthropic
GOOGLE_API_KEY=       # Required if using Google
ALPACA_API_KEY=       # Required for trading
ALPACA_SECRET_KEY=    # Required for trading
```

## Risk Controls (hard gates, not overridable by AI)

- Max single position: 10% equity
- Max total exposure: 80% equity
- Max daily loss: 3%
- Max drawdown: 10%
- Min cash reserve: 20%
- Max open positions: 10
- Default SL: 5%, TP: 15%

## LLM API Call Map (per symbol analysis)

Each `propagate()` call triggers ~10-15 LLM invocations:
1. Market Analyst (tool-calling chain)
2. Social Media Analyst (tool-calling chain)
3. News Analyst (tool-calling chain)
4. Fundamentals Analyst (tool-calling chain)
5. Bull Researcher
6. Bear Researcher
7. Research Manager/Judge (deep_think)
8. Trader
9. Aggressive Risk Debater
10. Conservative Risk Debater
11. Neutral Risk Debater
12. Risk Manager (deep_think)
13. Signal Processor (structured extraction)
14. Reflection (5 calls, post-trade only)

With 7 watchlist stocks: ~70-105 LLM calls per daily analysis cycle.

## Commands

```bash
pip install -e .              # Install in dev mode
python -m pytest tests/       # Run tests
```

## Restart the scheduler after code changes that affect live trading

The scheduler runs under launchd as `com.tradingagents.scheduler`
(`~/Library/LaunchAgents/com.tradingagents.scheduler.plist`). It loads
every `tradingagents/` module once at startup and holds the imports in
memory for the life of the process. Editing code does NOT propagate to
the running scheduler.

**Standard operation:** whenever a code change is expected to affect
live behavior — broker integration, orchestrator logic, exit manager,
risk gates, config schema — the scheduler must be restarted to pick it
up. Ask the user before restarting.

```bash
launchctl kickstart -k gui/$UID/com.tradingagents.scheduler
```

The `-k` flag stops the current process and relaunches it under launchd.
This is a brief (seconds) gap in the scheduler loop but does not cancel
open Alpaca orders or clear position state. Confirm the restart took by
checking the PID changed:

```bash
launchctl list | grep com.tradingagents.scheduler
```

Changes that do NOT require a restart:
- Backtest / research scripts under `scripts/` (not imported by scheduler)
- Tests under `tests/`
- DB rows modified out-of-band (scheduler re-reads the DB each tick)
- `experiments/paper_launch_v2.yaml` (re-read at next scheduler tick)

When in doubt, restart.

## Strategy edge claims — MANDATORY before reporting any backtest result

A one-character lookahead (`<=` vs `<`) in `intraday_backtester.py:1235`
turned a no-edge strategy into a 2.14 R/DD apparent winner. We almost
shipped it live. See `memory/project_intraday_lookahead_blowup.md`.

Before claiming a strategy has edge — i.e., before reporting any backtest
return, R/DD, win rate, or Sharpe — you MUST:

1. **Enumerate filters before auditing.** When asked "are there biases?",
   first `grep -n "shift\|<=\|<.*ts\|loc\[.*index\|as_of\|trend_day"` in
   the relevant backtester. Produce an exhaustive list of every place
   the simulation consults a time-indexed series. THEN check each one.
   Do not extrapolate from spot-checks of feature math to a clean bill
   of health on the whole pipeline.

2. **State scope explicitly.** When reporting on bias, list what you
   checked AND what you did not check. "I checked feature math; I did
   NOT check filter applications in the selection loop or universe
   construction" beats "the code looks clean."

3. **Run the future-blanked probe before reporting edge.** Re-run the
   backtest with the current bar's daily/weekly/external data blanked
   (set to NaN). If returns drop by more than ~2pp, the apparent edge
   is lookahead — find the leak before reporting numbers.

4. **Treat suspicious R/DD as a bug, not a feature.** A mechanical
   intraday strategy with R/DD > 2.0 on multi-year IS data is, by prior,
   far more likely to be lookahead than to be a real edge. When you see
   one, dig harder, not less.

5. **When the user asks "are we sure?" twice, treat it as evidence you
   are wrong.** Re-audit from scratch with stronger scope.

These rules apply to every backtester in the repo (`intraday_backtester`,
`backtest.py`, `chan_backtester`, `strategy_ab_runner`, etc.), not just
intraday.

## PEAD + earnings ingest architecture (added 2026-05-01)

PEAD is a **standalone launchd job** (`com.baibot.pead`), not an ABRunner
variant. The three earnings ingest crons (`com.baibot.av_ingest`,
`com.baibot.calendar_update`, `com.baibot.pead_eod_sync`) are likewise
standalone. This separation is **intentional** — do not migrate them
into the swing scheduler / ABRunner.

**Why standalone:**
- PEAD's thesis is "no stop, exit by date." If PEAD positions ever flowed
  into the swing reconciler's `position_states` table, the orphan branch
  at `reconciler.py:462-522` would fabricate an 8%-below-entry stop and
  ExitManager would start ratcheting it — strategy-killing bug.
- Process isolation: a crash in PEAD or an ingest cron cannot affect
  the live swing scheduler.

**DB split (also 2026-05-01):**
- `research_data/market_data.duckdb` — owned by scheduler. Holds
  `daily_bars`, `fundamentals_snapshots`, `quarterly_fundamentals`.
- `research_data/earnings_data.duckdb` — owned by earnings ingest crons +
  PEAD (read). Holds `earnings_events`, `earnings_calendar`.
- Cross-DB reads use DuckDB `ATTACH 'earnings_data.duckdb' AS ed (READ_ONLY)`
  — see `warehouse.py:_attach_earnings_readonly`. Read-only attach takes
  no writer lock on the attached file, so ingest crons can write
  concurrently with scheduler reads.
- Migration: `scripts/migrate_split_earnings_db.py` (idempotent).

**Dashboard integration (one-way bridge):**
- PEAD writes its authoritative state to `results/pead/paper/positions.json`
  (atomic) and `fills.jsonl` (append-only).
- `tradingagents/automation/pead_ingest.py` mirrors that JSON state +
  the live PEAD Alpaca account into `trading_pead.db` (a SQLite shaped
  like the other per-variant trading DBs). Bridge runs at 08:35 CDT
  (chained into PEAD's wrapper) and 15:30 CDT (separate
  `com.baibot.pead_eod_sync` job).
- The dashboard's `multi_variant.get_variant_dbs()` auto-discovers
  `trading_pead.db` via the entry in `experiments/paper_launch_v2.yaml`
  (`name: pead`, `strategy_type: pead_view`). PEAD shows up on
  Performance + Reviews + Today pages alongside the 6 trading variants.
- The bridge is **fill-only / mirror-only** — never modifies PEAD's
  authoritative JSON files. A sync failure cannot corrupt PEAD itself.
- Position rows for PEAD use `current_stop=0.0` (sentinel for "no stop")
  and `base_pattern='earnings_surprise'` so cross-variant analytics can
  filter them out where stop-based logic would be misleading.

**Watchdog freshness (added 2026-05-01, extended 2026-05-02):**
Five checks fire daily at 09:00 ET (`monitor.py` jobs 10-14):
- `check_pead_freshness` — `positions.json` mtime > 36h on a trading day
- `check_pead_dashboard_sync` — `trading_pead.db` mtime > 36h
- `check_calendar_freshness` — `MAX(fetched_at) FROM earnings_calendar` > 36h
- `check_av_or_yfinance_freshness` — `MAX(updated_at) FROM earnings_events` > 48h
- `check_pead_llm_decisions_fresh` — `MAX(analyzed_at) FROM earnings_llm_decisions` > 18h (added for LLM gate)

## PEAD LLM gate — A/B forward test (added 2026-05-02)

Layered on top of the standalone PEAD architecture above:
`tradingagents/research/pead_llm_analyzer.py` wraps the existing 13-agent
`TradingAgentsGraph` pipeline (used by the live `llm` swing variant) to
score each PEAD candidate as BUY/HOLD/SELL.

**Two batch crons populate the cache:**
- `com.baibot.pead_llm_amc` (Mon-Fri 17:30 CDT) — yesterday-AMC + overnight reporters
- `com.baibot.pead_llm_bmo` (Mon-Fri 06:30 CDT) — today's pre-open reporters
  (2h budget — fits 24+ candidates serial; guard at 115 min skips overflow)
  - **Note**: `com.baibot.av_ingest` was moved 08:00 → **06:00 CDT** the same
    day so the BMO LLM batch has fresh `earnings_events.surprise_pct` data
    to work against. The yfinance fallback chained inside that wrapper still
    covers AV-blocked days.

Cache: `earnings_llm_decisions` table in `earnings_data.duckdb`.
PRIMARY KEY `(symbol, event_date)`. Failed analyses still write a row
(`error` populated, `llm_decision = NULL`) — auditable but PEAD's INNER
JOIN naturally drops them.

**Models** (OpenAI only — no Anthropic key):
- AMC batch deep-think (research_manager + risk_manager judges): `gpt-5.4-pro`
  (~20 min/candidate; overnight latency is fine)
- **BMO batch deep-think: `gpt-5.4` (NON-pro)** — set in
  `pead_llm_bmo.sh`. gpt-5.4-pro's reasoning tokens add ~5min/judge call
  which would risk overflowing even the 2h BMO budget on busy days.
  gpt-5.4 brings runtime to ~5 min/candidate at ~$0.10/candidate.
- Quick-think (4 analysts + 2 researchers + trader + 3 risk debaters):
  `gpt-5-mini` for both windows
- Override via env: `PEAD_DEEP_MODEL` / `PEAD_QUICK_MODEL`
- Total cost: ~$0.10-0.20/analysis × 5-10 candidates/day ≈ $0.5-2/day

**A/B forward test** (60-90 trading days):
- Control: `com.baibot.pead` (existing, `--no-llm-gate`, **live-submit unchanged**)
- Treatment: `com.baibot.pead_llm` (new, `--require-llm-buy`, **dry-run only**)
- Treatment writes to `results/pead/llm_dryrun/` and mirrors to
  `trading_pead_llm.db` (registered in `multi_variant._STANDALONE_VIEW_DBS`
  so it appears in the dashboard alongside the control arm)
- Promote treatment to live-submit only on positive Sharpe delta net of LLM costs

**Earnings catalyst injection** (the "Option B" decision in the plan):
The 13 agent prompts are generic investment-decision prompts — none
mention "earnings". The analyzer wraps the graph to:
- Inject an `EARNINGS CATALYST CONTEXT` preamble (symbol, EPS estimate/
  reported/surprise%, time_hint, source, plus the PEAD thesis: weigh
  beat quality, guidance, mgmt tone, sector context, analyst reaction)
- Pass via `screener_context` (trader sees it) AND prepend as the
  initial human message (analysts see it as their first user turn)
- No file forks — monkey-patches `propagator.create_initial_state` for
  one graph instance only.

**LOAD-BEARING GUARDRAIL — DO NOT regress:**
PEAD-LLM analyzer must NEVER import `tradingagents.testing.ab_runner`
or instantiate any `Orchestrator` subclass. Adding PEAD to
`paper_launch_v2.yaml` corrupted the PEAD Alpaca account on 2026-05-01
(scheduler built a default Minervini Orchestrator on it, opened ~$50k
of unwanted positions, fabricated 8% stops). The `_build_orchestrator`
in `ab_runner.py:20-32` now raises on unknown `strategy_type` to make
this fail loud — but the analyzer must also not need ABRunner at all.

**AV API key isolation:**
- Shared `ALPHA_VANTAGE_API_KEY` is now used by **zero** live code paths.
  The live preflight (`prescreener.py:312`) passes
  `use_alpha_vantage_fallback=False` to keep the IP-level quota free.
- The nightly ingest cron uses dedicated `ALPHA_VANTAGE_INGEST_API_KEY`
  exclusively (no fallback to the shared key).
- AV throttles by IP, not just by key — a fresh key alone does not solve
  IP exhaustion. The yfinance fallback (`scripts/ingest_earnings_yfinance_recent.py`,
  chained after AV ingest in the cron wrapper) covers AV-blocked days.
