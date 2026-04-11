"""LLM-augmented portfolio backtester.

Integrates the TradingAgentsGraph LLM pipeline into the walk-forward
backtest loop. The Minervini screener identifies candidates mechanically,
then the LLM makes the final entry decision. Exits remain rule-based.

For backtesting, only the market analyst (price/technicals) is used
since historical news and fundamentals aren't available point-in-time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .backtester import BacktestConfig
from .minervini import MinerviniConfig, MinerviniScreener
from .portfolio_backtester import PortfolioBacktestResult, PortfolioMinerviniBacktester
from .walk_forward import WalkForwardBacktester, WalkForwardConfig, WalkForwardResult
from .warehouse import MarketDataWarehouse

logger = logging.getLogger(__name__)


BACKTEST_ENTRY_PROMPT = """You are grading momentum trade setups for a Minervini SEPA portfolio.

IMPORTANT CONTEXT: this candidate has ALREADY passed a strict mechanical screen
(Stage 2 uptrend, SMA stacking, RS leadership, base pattern, and more). Your
job is NOT to re-validate the screen — assume it's a qualifying setup. Your
job is to decide whether it's an A-grade entry worth taking a slot for, or
a marginal one worth skipping.

Default to BUY. Passing has real opportunity cost: the portfolio holds at
most 10 names and slots go empty when you pass. Only PASS when you see a
specific disqualifying condition.

{screener_context}

Decision rule:
- BUY if the setup looks tradable under Minervini rules. Most pre-screened
  candidates qualify. This should be your default answer.
- PASS only if you spot a concrete red flag, such as:
    * RSI > 80 AND price > 20% above SMA50 (blow-off extension)
    * Price below SMA50 (trend break, not a momentum entry)
    * ADX < 15 (no directional strength, choppy)
    * Rate of change negative over 60 AND 120 days (momentum reversal)
  Ambiguity or "could go either way" is NOT a red flag — that's a BUY.

Confidence calibration:
- 0.85-0.95: clean momentum setup, all indicators aligned
- 0.70-0.84: solid setup with one minor concern
- 0.60-0.69: marginal but still tradable (still BUY)
- < 0.60: PASS (use only when a red flag above is present)

Respond with ONLY valid JSON (no markdown, no extra text):
{{"action": "BUY" or "PASS", "confidence": 0.0 to 1.0, "reasoning": "1-2 sentences naming the specific factor that drove the decision"}}"""


@dataclass
class LLMBacktestConfig:
    """Configuration for the LLM-augmented backtester."""

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    selected_analysts: list = None  # Default: ["market"] for backtest
    min_confidence: float = 0.6  # Minimum LLM confidence to enter

    # Caching
    cache_db_path: str = "research_data/llm_backtest_cache.db"

    # Cost controls
    max_llm_calls_per_day: int = 20  # Safety limit
    skip_hold_signals: bool = True  # Don't count HOLD as rejection

    def __post_init__(self):
        if self.selected_analysts is None:
            self.selected_analysts = ["market"]


class LLMResponseCache:
    """SQLite cache for LLM backtest responses to avoid re-running identical calls."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_responses (
                cache_key TEXT PRIMARY KEY,
                symbol TEXT,
                trade_date TEXT,
                model TEXT,
                action TEXT,
                confidence REAL,
                reasoning TEXT,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                timeframe TEXT,
                full_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    @staticmethod
    def _make_key(symbol: str, trade_date: str, model: str, analysts: list) -> str:
        raw = f"{symbol}|{trade_date}|{model}|{','.join(sorted(analysts))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, symbol: str, trade_date: str, model: str, analysts: list) -> dict | None:
        key = self._make_key(symbol, trade_date, model, analysts)
        row = self.conn.execute(
            "SELECT action, confidence, reasoning, stop_loss_pct, take_profit_pct, timeframe "
            "FROM llm_responses WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "action": row[0],
            "confidence": row[1],
            "reasoning": row[2],
            "stop_loss_pct": row[3],
            "take_profit_pct": row[4],
            "timeframe": row[5],
        }

    def put(self, symbol: str, trade_date: str, model: str, analysts: list, signal: dict):
        key = self._make_key(symbol, trade_date, model, analysts)
        self.conn.execute(
            """INSERT OR REPLACE INTO llm_responses
               (cache_key, symbol, trade_date, model, action, confidence,
                reasoning, stop_loss_pct, take_profit_pct, timeframe, full_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                symbol,
                trade_date,
                model,
                signal.get("action", "HOLD"),
                signal.get("confidence", 0.0),
                signal.get("reasoning", ""),
                signal.get("stop_loss_pct"),
                signal.get("take_profit_pct"),
                signal.get("timeframe", ""),
                json.dumps(signal, default=str),
            ),
        )
        self.conn.commit()

    def stats(self) -> dict:
        row = self.conn.execute("SELECT COUNT(*) FROM llm_responses").fetchone()
        return {"total_cached": row[0]}

    def close(self):
        self.conn.close()


class LLMBacktester:
    """Walk-forward backtester with LLM entry decisions.

    Flow:
    1. Walk-forward screener identifies candidates (mechanical)
    2. For each candidate passing entry rules, call LLM for final decision
    3. Only enter if LLM says BUY with confidence >= threshold
    4. Exits remain rule-based (stops, 50 DMA, time)
    """

    def __init__(
        self,
        warehouse: MarketDataWarehouse,
        llm_config: LLMBacktestConfig | None = None,
        wf_config: WalkForwardConfig | None = None,
        backtest_config: BacktestConfig | None = None,
        screener_config: MinerviniConfig | None = None,
    ):
        self.warehouse = warehouse
        self.llm_config = llm_config or LLMBacktestConfig()
        self.wf_config = wf_config or WalkForwardConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.screener_config = screener_config or MinerviniConfig(
            require_fundamentals=False,
            require_market_uptrend=False,
        )

        self.cache = LLMResponseCache(self.llm_config.cache_db_path)
        self._llm = None  # Lazy-init
        self._llm_calls = 0
        self._cache_hits = 0

    def _get_llm(self):
        """Lazy-initialize a single LLM client (no agent graph needed)."""
        if self._llm is None:
            from tradingagents.llm_clients import create_llm_client

            client = create_llm_client(
                provider=self.llm_config.llm_provider,
                model=self.llm_config.llm_model,
            )
            self._llm = client.get_llm()
        return self._llm

    def get_llm_decision(
        self,
        symbol: str,
        trade_date: str,
        screener_context: str = "",
    ) -> dict:
        """Get LLM trading decision via a single direct call.

        Instead of running the full 13-call agent pipeline, sends one prompt
        with all screener data and gets back a BUY/PASS decision. ~13x faster.
        """
        model = self.llm_config.llm_model
        analysts = self.llm_config.selected_analysts

        # Check cache first
        cached = self.cache.get(symbol, trade_date, model, analysts)
        if cached is not None:
            self._cache_hits += 1
            return cached

        # Single LLM call
        llm = self._get_llm()
        prompt = BACKTEST_ENTRY_PROMPT.format(screener_context=screener_context)

        try:
            response = llm.invoke([
                ("system", prompt),
                ("human", "Based on this technical setup, should I buy or pass?"),
            ])

            raw = response.content.strip()
            # Strip markdown fences and thinking tags if present
            if "<think>" in raw:
                # Remove thinking block (common with qwen/deepseek)
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            signal = json.loads(raw)

            # Normalize action
            action = signal.get("action", "PASS").upper()
            if action not in ("BUY", "PASS", "SELL", "HOLD"):
                action = "PASS"
            if action == "PASS":
                action = "HOLD"  # Normalize for consistency

            result = {
                "action": action,
                "confidence": float(signal.get("confidence", 0.5)),
                "reasoning": signal.get("reasoning", ""),
            }

            self._llm_calls += 1
            self.cache.put(symbol, trade_date, model, analysts, result)

            logger.info(
                f"  LLM {symbol} @ {trade_date}: "
                f"{result['action']} (conf={result['confidence']:.2f}) "
                f"- {result.get('reasoning', '')[:80]}"
            )
            return result

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"  LLM parse error for {symbol} @ {trade_date}: {e} | raw: {raw[:200]}")
            fallback = {"action": "HOLD", "confidence": 0.0, "reasoning": f"Parse error: {e}"}
            self._llm_calls += 1
            self.cache.put(symbol, trade_date, model, analysts, fallback)
            return fallback
        except Exception as e:
            logger.warning(f"  LLM error for {symbol} @ {trade_date}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"LLM error: {e}",
            }

    def run(
        self,
        seed_symbols: list[str],
        start_date: str,
        end_date: str,
        trade_start_date: str | None = None,
    ) -> LLMBacktestResult:
        """Execute walk-forward backtest with LLM entry gate.

        Phase 1: Run mechanical walk-forward to identify candidates
        Phase 2: Re-run with LLM filtering on candidates
        """
        benchmark = self.wf_config.benchmark
        screener = MinerviniScreener(self.screener_config)

        # 1. Load data
        logger.info(f"Loading data for {len(seed_symbols)} symbols...")
        all_symbols = sorted(set(seed_symbols + [benchmark]))
        all_data = self.warehouse.get_daily_bars_bulk(all_symbols, start_date, end_date)

        if benchmark not in all_data:
            raise ValueError(f"Benchmark {benchmark} not found in warehouse")

        benchmark_df = all_data.pop(benchmark)
        data_by_symbol = {
            sym: df for sym, df in all_data.items()
            if len(df) >= self.wf_config.min_data_bars
        }
        logger.info(f"Loaded {len(data_by_symbol)} symbols")

        # 2. Pre-compute features
        logger.info("Computing features...")
        prepared_frames: dict[str, pd.DataFrame] = {}
        for symbol, df in data_by_symbol.items():
            prepared = screener.prepare_features(df)
            if not prepared.empty:
                prepared_frames[symbol] = prepared

        # 3. Build walk-forward candidate schedule (mechanical)
        wf_backtester = WalkForwardBacktester(
            warehouse=self.warehouse,
            wf_config=self.wf_config,
            backtest_config=self.backtest_config,
            screener_config=self.screener_config,
        )

        all_dates = sorted(
            {ts for frame in prepared_frames.values() for ts in frame.index}
        )
        if trade_start_date:
            trade_start_ts = pd.Timestamp(trade_start_date)
        else:
            warmup_idx = min(self.wf_config.min_data_bars, len(all_dates) - 1)
            trade_start_ts = all_dates[warmup_idx]

        rebalance_dates = wf_backtester._generate_rebalance_dates(
            trade_start_ts, all_dates[-1]
        )
        logger.info(f"Generated {len(rebalance_dates)} rebalance dates")

        candidate_schedule, rebalance_log = wf_backtester._build_candidate_schedule(
            prepared_frames, benchmark_df, rebalance_dates
        )

        # 4. Run portfolio backtest with LLM-filtered schedule
        logger.info("Running LLM-filtered backtest...")
        llm_schedule = self._build_llm_filtered_schedule(
            candidate_schedule=candidate_schedule,
            prepared_frames=prepared_frames,
            trade_start_ts=trade_start_ts,
        )

        # 5. Run the portfolio backtest with LLM-filtered candidates
        backtester = PortfolioMinerviniBacktester(
            config=self.backtest_config,
            screener=screener,
        )
        portfolio_result = backtester.backtest_portfolio(
            data_by_symbol=data_by_symbol,
            benchmark_df=benchmark_df,
            trade_start_date=trade_start_ts.strftime("%Y-%m-%d"),
            candidate_schedule=llm_schedule,
        )

        # 6. Also run mechanical-only for comparison
        logger.info("Running mechanical-only backtest for comparison...")
        mech_backtester = PortfolioMinerviniBacktester(
            config=self.backtest_config,
            screener=screener,
        )
        mechanical_result = mech_backtester.backtest_portfolio(
            data_by_symbol=data_by_symbol,
            benchmark_df=benchmark_df,
            trade_start_date=trade_start_ts.strftime("%Y-%m-%d"),
            candidate_schedule=candidate_schedule,
        )

        logger.info(
            f"LLM calls: {self._llm_calls}, Cache hits: {self._cache_hits}, "
            f"Total cache: {self.cache.stats()['total_cached']}"
        )

        self.cache.close()

        return LLMBacktestResult(
            llm_result=portfolio_result,
            mechanical_result=mechanical_result,
            rebalance_log=rebalance_log,
            llm_calls=self._llm_calls,
            cache_hits=self._cache_hits,
            llm_decisions=self._llm_decisions_log,
        )

    def _build_llm_filtered_schedule(
        self,
        candidate_schedule: dict[pd.Timestamp, set[str]],
        prepared_frames: dict[str, pd.DataFrame],
        trade_start_ts: pd.Timestamp,
    ) -> dict[pd.Timestamp, set[str]]:
        """Filter candidate schedule through LLM decisions.

        For each rebalance date, take the mechanically-approved candidates
        and ask the LLM whether to buy each one. Only pass through BUY signals.
        """
        self._llm_decisions_log: list[dict] = []
        llm_schedule: dict[pd.Timestamp, set[str]] = {}

        total_rebalances = len(candidate_schedule)
        for i, (rebalance_date, candidates) in enumerate(sorted(candidate_schedule.items())):
            trade_date_str = rebalance_date.strftime("%Y-%m-%d")

            llm_approved = set()
            for symbol in sorted(candidates):
                # Build screener context from prepared features
                context = self._build_screener_context(
                    symbol, prepared_frames, rebalance_date
                )

                decision = self.get_llm_decision(
                    symbol=symbol,
                    trade_date=trade_date_str,
                    screener_context=context,
                )

                self._llm_decisions_log.append({
                    "rebalance_date": trade_date_str,
                    "symbol": symbol,
                    "action": decision.get("action", "HOLD"),
                    "confidence": decision.get("confidence", 0.0),
                    "reasoning": decision.get("reasoning", "")[:200],
                })

                action = decision.get("action", "HOLD").upper()
                confidence = decision.get("confidence", 0.0)

                if action == "BUY" and confidence >= self.llm_config.min_confidence:
                    llm_approved.add(symbol)

            llm_schedule[rebalance_date] = llm_approved

            if (i + 1) % 5 == 0 or i == total_rebalances - 1:
                logger.info(
                    f"  Rebalance {i + 1}/{total_rebalances} ({trade_date_str}): "
                    f"{len(candidates)} candidates → {len(llm_approved)} LLM-approved"
                )

        return llm_schedule

    def _build_screener_context(
        self,
        symbol: str,
        prepared_frames: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> str:
        """Build a blinded text summary of screener data for LLM context.

        IMPORTANT: No ticker name or date is included to prevent the LLM
        from using future knowledge about specific stocks. All values are
        presented as relative/percentage terms where possible.
        """
        frame = prepared_frames.get(symbol)
        if frame is None:
            return ""

        valid = frame.loc[frame.index <= as_of_date]
        if valid.empty:
            return ""

        row = valid.iloc[-1]
        parts = ["Technical Setup:"]

        # Price position relative to moving averages (percentages only, no dollar values)
        close = float(row.get("close", 0) or 0)
        sma_50 = float(row.get("sma_50", 0) or 0)
        sma_150 = float(row.get("sma_150", 0) or 0)
        sma_200 = float(row.get("sma_200", 0) or 0)

        if close > 0 and sma_50 > 0:
            parts.append(f"  Price vs SMA50: {(close / sma_50 - 1) * 100:+.1f}%")
        if close > 0 and sma_150 > 0:
            parts.append(f"  Price vs SMA150: {(close / sma_150 - 1) * 100:+.1f}%")
        if close > 0 and sma_200 > 0:
            parts.append(f"  Price vs SMA200: {(close / sma_200 - 1) * 100:+.1f}%")

        # SMA stacking
        if sma_50 > 0 and sma_150 > 0 and sma_200 > 0:
            stacking = "SMA50 > SMA150 > SMA200" if sma_50 > sma_150 > sma_200 else "Not aligned"
            parts.append(f"  MA Stacking: {stacking}")

        # Momentum
        roc_60 = float(row.get("roc_60", 0) or 0)
        roc_120 = float(row.get("roc_120", 0) or 0)
        rsi = float(row.get("rsi_14", 0) or 0)
        parts.append(f"  Rate of Change (60d): {roc_60:+.1f}%")
        parts.append(f"  Rate of Change (120d): {roc_120:+.1f}%")
        parts.append(f"  RSI(14): {rsi:.1f}")

        # Trend strength
        adx = float(row.get("adx_14", 0) or 0)
        parts.append(f"  ADX(14): {adx:.1f}")

        # 52-week position (relative only)
        high_52w = float(row.get("52w_high", 0) or 0)
        low_52w = float(row.get("52w_low", 0) or 0)
        if high_52w > 0 and close > 0:
            parts.append(f"  Distance from 52W High: {(close / high_52w - 1) * 100:+.1f}%")
        if low_52w > 0 and close > 0:
            parts.append(f"  Above 52W Low: {(close / low_52w - 1) * 100:+.1f}%")

        # Pattern flags
        flags = []
        for flag in ["breakout_ready", "vcp_candidate", "base_candidate",
                      "flat_base_candidate", "cup_handle_candidate", "breakout_signal"]:
            if row.get(flag):
                flags.append(flag.replace("_", " "))
        if flags:
            parts.append(f"  Chart Patterns: {', '.join(flags)}")
        else:
            parts.append("  Chart Patterns: none detected")

        # Volume
        vol_ratio = float(row.get("breakout_volume_ratio", 0) or 0)
        if vol_ratio > 0:
            parts.append(f"  Volume vs 50d Avg: {vol_ratio:.1f}x")

        return "\n".join(parts)


@dataclass
class LLMBacktestResult:
    """Results from LLM-augmented backtest."""
    llm_result: PortfolioBacktestResult
    mechanical_result: PortfolioBacktestResult
    rebalance_log: pd.DataFrame
    llm_calls: int
    cache_hits: int
    llm_decisions: list[dict]
