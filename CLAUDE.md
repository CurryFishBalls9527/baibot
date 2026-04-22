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
