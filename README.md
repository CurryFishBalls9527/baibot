<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/wechat.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-TauricResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

## News
- [2026-03] **TradingAgents v0.2.1** released with GPT-5.4, Gemini 3.1, Claude 4.6 model coverage and improved system stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

<div align="center">
<a href="https://www.star-history.com/#TauricResearch/TradingAgents&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" />
   <img alt="TradingAgents Star History" src="https://api.star-history.com/svg?repos=TauricResearch/TradingAgents&type=Date" style="width: 80%; height: auto;" />
 </picture>
</a>
</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles. This ensures the system achieves a robust, scalable approach to market analysis and decision-making.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone TradingAgents:
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.13
conda activate tradingagents
```

Or with the standard library `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

This repo can be used in two different ways:

- `Research mode`: local screening/backtests, no Alpaca account required.
- `Automation mode`: paper trading with Alpaca, scheduled scans, dashboard, and phone notifications.

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export OPENROUTER_API_KEY=...      # OpenRouter
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
```

For local models, configure Ollama with `llm_provider: "ollama"` in your config.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### `.env` Guide For Collaborators

Use `.env.example` as the template.

- `OPENAI_API_KEY`: needed for most agent-based research and trading flows if you use OpenAI models.
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`: needed only for Alpaca paper/live trading, account status, and automated execution.
- `ALPACA_MECHANICAL_API_KEY` / `ALPACA_MECHANICAL_SECRET_KEY`: optional, used by the `mechanical` variant of the A/B paper-trading experiment (see *A/B Paper Trading* below).
- `ALPACA_LLM_API_KEY` / `ALPACA_LLM_SECRET_KEY`: optional, used by the `llm` variant of the A/B paper-trading experiment.
- `ALPHA_VANTAGE_API_KEY`: optional, used by some market-data and research paths.
- `NTFY_TOPIC`: optional, used for phone push notifications.
- `SOCIAL_NTFY_TOPIC`: optional, used for social-monitor alerts.
- `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`: optional alternatives to OpenAI.

Recommended sharing workflow:

```bash
cp .env.example .env
```

Then fill in only the keys you actually need.

- If you only want research / CLI usage: usually `OPENAI_API_KEY` is enough.
- If you want automated paper trading: add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
- If you want phone notifications: add `NTFY_TOPIC`.

Do not commit `.env`, and do not share real API keys in the repo.

### What Each Key Is For

- `OPENAI_API_KEY`
  - Needed for most agent-driven trading flows in `run_trading.py`.
  - Also needed if you enable social-monitor translation.
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
  - Needed for account status, positions, orders, and automated paper trading.
  - If these are missing, trading commands that touch the broker will not work.
- `ALPHA_VANTAGE_API_KEY`
  - Optional. Some research/data workflows may use it.
- `NTFY_TOPIC`
  - Optional. Enables phone push notifications through `ntfy`.
- `NTFY_ENABLED`
  - Set to `1` if you want trading notifications enabled.
- `SOCIAL_NTFY_TOPIC`
  - Optional. Separate topic for social-feed alerts.
- `SOCIAL_MONITOR_ENABLED`
  - Set to `1` if you want RSS-based social monitoring enabled.
- `SOCIAL_NTFY_ENABLED`
  - Set to `1` if you want social alerts pushed through `ntfy`.
- `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`
  - Optional alternatives if you want to switch model providers.

### Secrets And Local State

These should stay local and should not be committed:

- `.env`
- `trading.db`
- `trading_*.db` (per-variant A/B dbs like `trading_mechanical.db`, `trading_llm.db`)
- `results/`
- `research_data/`
- `trading.log`

The repo already ignores those paths. Do not paste real tokens or API keys into issues, PRs, or the README.

### Quick Start By Use Case

#### 1. Research only

If your collaborator only wants to run screens/backtests:

```bash
cp .env.example .env
```

Fill in:

- `OPENAI_API_KEY` only if they also want to use the agent workflows
- otherwise, the Minervini research runner can be used without Alpaca

Then run:

```bash
python run_minervini_research.py --refresh-data --screen --backtest
```

#### 2. Paper trading / automation

If they want the automated trading system:

```bash
cp .env.example .env
```

Fill in at minimum:

- `OPENAI_API_KEY`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

Optional:

- `NTFY_ENABLED=1`
- `NTFY_TOPIC`
- `SOCIAL_MONITOR_ENABLED=1`
- `SOCIAL_NTFY_ENABLED=1`
- `SOCIAL_NTFY_TOPIC`

Then sanity-check the setup:

```bash
python run_trading.py status
python run_trading.py setups
```

#### 3. Phone notifications

To use `ntfy`:

- install the `ntfy` app on the phone
- subscribe to a topic
- set `NTFY_ENABLED=1` and `NTFY_TOPIC` in `.env`

Then test it:

```bash
python run_trading.py notify-test
```

If using social alerts too:

```bash
python run_trading.py social-test
```

For RSS-based social monitoring, also set:

- `SOCIAL_MONITOR_ENABLED=1`
- `SOCIAL_NTFY_ENABLED=1`
- `SOCIAL_NTFY_TOPIC=your-topic`

### Repo-Specific Trading Extensions

This repo is not just the upstream `TradingAgents` framework. It also includes:

- `run_trading.py`: automated paper-trading CLI
- `tradingagents/automation/`: scheduler, notifications, social monitor, launchd integration
- `tradingagents/research/`: Minervini-style screening, local market-data warehouse, regime logic
- `tradingagents/dashboard/`: local Streamlit dashboard
- `run_minervini_research.py`: research/backtest runner

The automated trading layer currently supports:

- Minervini-style screening
- dynamic broad-market coarse scan
- leader continuation / add-on setups
- Alpaca paper trading
- `ntfy` phone notifications
- optional RSS-based social monitoring

### Running The Main Commands

#### Manual trading analysis

Runs one analysis cycle immediately. If the broker is connected and rules are satisfied, this may place paper orders.

```bash
python run_trading.py run
```

With specific symbols:

```bash
python run_trading.py run --symbols AAPL,NVDA
```

#### Live status

Shows account, positions, daily P&L, approved setups, and overlay status.

```bash
python run_trading.py status
```

#### Recent trades

```bash
python run_trading.py trades --limit 20
```

#### Daily report

Shows today’s trades, open positions, performance, and latest screening batch.

```bash
python run_trading.py report
```

#### Latest setups

Shows the latest saved screening candidates from automation.

```bash
python run_trading.py setups
```

#### Local dashboard

Starts a local Streamlit dashboard.

```bash
python run_trading.py dashboard
```

Then open:

```text
http://127.0.0.1:8501
```

#### Background scheduler

Runs the automation loop in the foreground:

```bash
python run_trading.py schedule --mode both
```

Current schedule is based on `US/Eastern`:

- `09:25 ET`: market-open snapshot and morning scan
- `10:00-15:00 ET`: intraday scan every hour
- `15:30 ET`: swing analysis
- `16:30 ET`: daily reflection and report

#### macOS background service

On macOS, install the scheduler as a `launchd` service:

```bash
python run_trading.py install-service --mode both
```

To install the service bound to an A/B experiment config (see below):

```bash
python run_trading.py install-service --mode swing \
  --experiment experiments/paper_launch.yaml
```

Remove it:

```bash
python run_trading.py remove-service
```

Service logs are written under:

```text
results/service_logs/
```

#### A/B paper trading (mechanical vs LLM)

You can run two paper-trading variants side by side against separate Alpaca
paper accounts and separate local SQLite dbs. The bundled example at
`experiments/paper_launch.yaml` defines two variants:

- `mechanical` — pure rule-based Minervini entries, `ExitManager` exits, no
  LLM calls (sets `mechanical_only_mode: true`)
- `llm` — full 13-call LangGraph agent pipeline

Alpaca keys are resolved at runtime from environment variables via `${VAR}`
substitution, so the YAML is safe to commit. Set the following in `.env`:

```bash
ALPACA_MECHANICAL_API_KEY=...
ALPACA_MECHANICAL_SECRET_KEY=...
ALPACA_LLM_API_KEY=...
ALPACA_LLM_SECRET_KEY=...
```

Then run either ad-hoc:

```bash
python run_trading.py run --experiment experiments/paper_launch.yaml
```

or on the scheduler:

```bash
python run_trading.py schedule --mode swing \
  --experiment experiments/paper_launch.yaml
```

Each variant writes to its own db (`trading_mechanical.db`, `trading_llm.db`)
so positions, trades, and agent memories never mix.

#### Emergency close

```bash
python run_trading.py close-all --confirm
```

Use this carefully. It will attempt to close all open positions.

### Research / Backtest Runner

The research script builds a local DuckDB market-data store, runs Minervini-style screens, and can backtest candidate universes.

Basic run:

```bash
python run_minervini_research.py --refresh-data --screen --backtest
```

Example with explicit symbols:

```bash
python run_minervini_research.py \
  --symbols NVDA,MSFT,ANET \
  --refresh-data \
  --refresh-fundamentals \
  --screen \
  --backtest
```

Useful options:

- `--period 1y|2y|3y|5y`
- `--start YYYY-MM-DD --end YYYY-MM-DD`
- `--trade-start YYYY-MM-DD`
- `--screen-count N`
- `--min-rs`
- `--require-acceleration`
- `--allow-missing-fundamentals`
- `--allow-market-correction`

Artifacts are written to:

- `research_data/market_data.duckdb`
- `results/minervini/`

### Dashboard And Output Files

Common local artifacts:

- `trading.db`: local SQLite database for signals, trades, snapshots, and setups
- `results/daily_reports/`: JSON daily summaries
- `results/service_logs/`: scheduler stdout/stderr logs
- `results/minervini/`: research outputs and CSV exports
- `research_data/market_data.duckdb`: local market-data warehouse

These files are local runtime state, not source code.

### Recommended First Run For A Collaborator

If someone is new to the repo, this is the cleanest sequence:

```bash
git clone git@github.com:hakusama1024/tradingbot.git
cd tradingbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then:

1. Fill in the required keys in `.env`.
2. Validate the install:

```bash
python run_trading.py status
python run_trading.py notify-test
```

3. If they only want research:

```bash
python run_minervini_research.py --refresh-data --screen
```

4. If they want the dashboard:

```bash
python run_trading.py dashboard
```

5. If they want background automation on macOS:

```bash
python run_trading.py install-service --mode both
```

### Notes For Collaborators

- The trading system is configured for Alpaca paper trading by default.
- Automation may place paper orders when rules are satisfied.
- `run_trading.py run` is not a dry run.
- `run_trading.py status` and `dashboard` both talk to the broker and local DB.
- Social monitoring uses RSS by default in this repo, not the paid X API.
- Some research/data downloads may take a while on first run because the local market-data warehouse is being built.

### CLI Usage

You can also try out the CLI directly by running:
```bash
python -m cli.main
```
You will see a screen where you can select your desired tickers, date, LLMs, research depth, etc.

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, OpenRouter, and Ollama.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # openai, google, anthropic, xai, openrouter, ollama
config["deep_think_llm"] = "gpt-5.2"     # Model for complex reasoning
config["quick_think_llm"] = "gpt-5-mini" # Model for quick tasks
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better. If you are interested in this line of research, please consider joining our open-source financial AI research community [Tauric Research](https://tauric.ai/).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
