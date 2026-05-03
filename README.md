# baibot

Personal multi-strategy paper-trading + research framework. Forked from
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
(Apache 2.0); the upstream multi-agent LLM trading-decision graph remains the
core, with substantial additions for live automation, broker integration,
backtesting, and a custom dashboard.

> ⚠️ Personal repo. Paper trading on Alpaca by default. Not investment advice.
> The automation will place orders against whatever broker keys it finds in
> `.env` — read every config before running.

## What's in here

- **Multi-agent decision graph** — analysts → researchers → trader → risk
  manager, built on LangGraph. Multi-provider (OpenAI / Anthropic / Google /
  xAI / OpenRouter / Ollama). See `tradingagents/graph/`.
- **Live paper-trading scheduler** — runs end-to-end pipelines on a cron,
  posts orders to Alpaca, tracks state in per-variant SQLite databases.
  Multiple strategies run in parallel as separate launchd jobs against
  separate Alpaca accounts. Configured in `experiments/paper_launch_v2.yaml`.
- **Research / backtest harness** — DuckDB market-data store under
  `research_data/`, screener + backtester runners (`run_backtest.py`,
  `run_minervini_research.py`, `run_walk_forward.py`).
- **Web dashboard** — FastAPI + lightweight-charts, served by `run_web.py`.
  Live trade telemetry, per-trade reasoning + chart with structural overlays,
  performance / risk / proposals / daily+weekly review pages, raw log tail.
  See `tradingagents/web/`.
- **Watchdog + alerting** — file-tailing event watcher with structured
  notifications via ntfy and Telegram. See `run_watchdog.py`.
- **Test suite** — `pytest tests/` runs unit tests + a web-dashboard
  regression suite (HTTP API + headless-Chrome UI). See *Tests* below.

## Repo layout

```
tradingagents/
  agents/        per-role LLM prompts (analyst, researcher, trader, risk)
  automation/    schedulers, orchestrators, watchdog, ingest jobs
  broker/        Alpaca paper + live integration
  dataflows/     yfinance / Alpha Vantage / Alpaca readers
  graph/         LangGraph workflow + signal processing + reflection
  llm_clients/   provider factories (openai/anthropic/google/xai/...)
  research/      backtesters, screeners, feature builders
  risk/          hard risk gates (position limits, drawdown, daily loss)
  storage/       SQLite + DuckDB schemas and helpers
  testing/       A/B harness for paper variants
  web/           FastAPI app + chart.js frontend (dashboard)
  dashboard/     Streamlit dashboard (legacy; running in parallel with web/)

scripts/         one-off + scheduled helper scripts; launchd plists
experiments/     YAML configs for paper-trading variants
research_data/   local DuckDB warehouses (market, earnings, intraday)
results/         scheduler outputs, daily/weekly reviews, service logs
tests/           pytest suite (unit + web-dashboard regression)
```

## Install

Python 3.11 recommended (3.10+ supported).

```bash
git clone git@github.com:CurryFishBalls9527/baibot.git
cd baibot
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env  # then fill in the keys you actually use
```

## Configure

Edit `.env`. Only fill what you need:

| Variable | When |
|---|---|
| `OPENAI_API_KEY` | If using OpenAI models (the default). |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / `XAI_API_KEY` / `OPENROUTER_API_KEY` | Alternate LLM providers. |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Required for any live or paper trading. |
| `ALPHA_VANTAGE_API_KEY` | Some fundamentals / earnings ingest paths. |
| `NTFY_*` | Phone notifications. |
| `TELEGRAM_*` | Telegram bot alerts. |

`.env.example` is the canonical reference. Secrets stay local — `.gitignore`
excludes `.env`, `.venv/`, `tests/node_modules/`, and runtime DB files.

## Common commands

```bash
# Manual one-shot analysis (LangGraph pipeline, prints decision)
python run_trading.py run

# Account status + open positions
python run_trading.py status

# Recent trades
python run_trading.py trades

# Start the scheduler (foreground)
python run_trading.py schedule

# Emergency: cancel all orders + flatten positions
python run_trading.py close-all

# Web dashboard (FastAPI + lightweight-charts)
python run_web.py                    # http://127.0.0.1:8765
python run_web.py --host 0.0.0.0     # LAN-reachable (no auth — be careful)

# Streamlit dashboard (legacy; will be retired once web/ has parity)
python run_trading.py dashboard

# Watchdog / alerting
python run_watchdog.py

# Backtest
python run_backtest.py --help
python run_minervini_research.py --refresh-data --screen --backtest
```

The scheduler can install itself as a macOS background service via launchd:

```bash
python run_trading.py install-service --mode both
launchctl list | grep com.tradingagents.scheduler
```

After any change that affects live behavior (broker, orchestrator, exit
manager, risk gates), restart the scheduler so the long-lived process picks
up the new code:

```bash
launchctl kickstart -k gui/$UID/com.tradingagents.scheduler
```

## Programmatic use

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG.copy()
cfg["llm_provider"]   = "openai"
cfg["deep_think_llm"] = "gpt-5.2"
cfg["quick_think_llm"] = "gpt-5-mini"

ta = TradingAgentsGraph(debug=True, config=cfg)
state, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

`tradingagents/default_config.py` is the full schema; per-variant overrides
go in `experiments/paper_launch_v2.yaml`.

## Web dashboard

`run_web.py` boots a FastAPI app at `http://127.0.0.1:8765`. Tabs:

- **TRADE** — live trade ticker + per-trade chart with structural overlays
  (chan BI/SEG/ZS/BSP, MA50/150/200, fill price lines)
- **TODAY** — single-variant operational view (KPIs, equity curve,
  positions, recent trades, scheduler health)
- **PERFORMANCE** — cross-variant equity / returns / drawdown / activity
- **REVIEWS** — daily + weekly post-mortem markdown with embedded charts
- **RISK** — correlation, risk-parity sizing, regime-conditional Sharpe,
  per-pattern attribution, on-demand AI synthesis
- **PROPOSALS** — manage weekly improvement queue
- **LOG** — tail `automation_service.out.log`

Live-event SSE endpoint at `/events/stream` tails
`results/service_logs/events.jsonl`.

## Tests

```bash
pytest                           # whole suite
pytest tests/test_web_api.py     # web API tests (~3s)
pytest tests/test_web_ui.py      # web UI tests (~30s, headless Chrome)
```

The UI suite drives a real Chrome via Chrome DevTools Protocol. It pulls
`chrome-remote-interface` into `tests/node_modules/` on first run via npm.
If Node, npm, or Chrome aren't available the UI tests skip cleanly without
breaking the rest of the suite.

## Outputs

These are runtime artifacts, not source:

- `trading*.db` — per-variant SQLite (signals, trades, snapshots, proposals)
- `research_data/*.duckdb` — market / earnings / intraday warehouses
- `results/daily_reviews/<date>/` — per-trade markdown post-mortems + charts
- `results/weekly_reviews/<isoweek>/` — Saturday strategy reviews
- `results/service_logs/` — scheduler stdout/stderr + structured events

## Attribution

The multi-agent LLM decision graph is from the upstream
[TradingAgents](https://github.com/TauricResearch/TradingAgents) project
(Apache 2.0). Citation:

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
  title  = {TradingAgents: Multi-Agents LLM Financial Trading Framework},
  author = {Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
  year   = {2025},
  eprint = {2412.20138},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.TR},
  url    = {https://arxiv.org/abs/2412.20138},
}
```
