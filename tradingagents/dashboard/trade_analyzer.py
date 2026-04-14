"""LLM-powered analysis of individual closed trades."""

from __future__ import annotations

import logging

from tradingagents.automation.config import build_config
from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)


class TradeAnalyzer:
    def __init__(self):
        config = build_config()
        client = create_llm_client(
            provider=config.get("llm_provider", "openai"),
            model=config.get("quick_think_llm", "gpt-4o-mini"),
        )
        self.llm = client.get_llm()

    def analyze_trade(self, outcome: dict, entry_signal: dict | None) -> str:
        """Generate LLM analysis for a closed trade. Returns markdown string."""
        prompt = self._build_prompt(outcome, entry_signal)
        result = self.llm.invoke(prompt)
        return result.content

    def _build_prompt(self, outcome: dict, entry_signal: dict | None) -> str:
        symbol = outcome.get("symbol", "?")
        entry_price = outcome.get("entry_price")
        exit_price = outcome.get("exit_price")
        return_pct = outcome.get("return_pct")
        hold_days = outcome.get("hold_days")
        exit_reason = outcome.get("exit_reason", "unknown")
        base_pattern = outcome.get("base_pattern", "N/A")
        regime = outcome.get("regime_at_entry", "N/A")
        entry_date = outcome.get("entry_date", "?")
        exit_date = outcome.get("exit_date", "?")

        signal_section = ""
        if entry_signal:
            reasoning = entry_signal.get("reasoning") or entry_signal.get("full_analysis") or ""
            confidence = entry_signal.get("confidence")
            action = entry_signal.get("action")
            signal_section = f"""
Entry Signal:
- Action: {action}
- Confidence: {confidence}
- Reasoning: {reasoning[:1000]}
"""

        return f"""You are a trading journal analyst. Analyze this closed trade concisely.

Trade Details:
- Symbol: {symbol}
- Entry: ${entry_price:.2f} on {entry_date}
- Exit: ${exit_price:.2f} on {exit_date}
- Return: {return_pct:+.2f}%
- Hold Duration: {hold_days} days
- Exit Reason: {exit_reason}
- Base Pattern: {base_pattern}
- Market Regime at Entry: {regime}
{signal_section}
Write a brief analysis in 4 sections (use markdown headers):

### Entry Thesis
Why was this trade entered? (Based on signal reasoning and pattern.)

### What Happened
Price action summary, hold duration, and what triggered the exit.

### Assessment
Why did this trade {"succeed" if return_pct and return_pct > 0 else "fail"}? What market conditions helped or hurt?

### Lesson
One actionable takeaway for future trades.

Keep total response under 250 words."""
