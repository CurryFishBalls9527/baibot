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

        # Optional enriched context — populated by the automation review loop
        # (backwards compatible: omitted when called from the UI with a plain
        # trade_outcomes row).
        mfe = outcome.get("max_favorable_excursion")
        mae = outcome.get("max_adverse_excursion")
        setup_context = outcome.get("_setup_context") or ""
        variant = outcome.get("_variant", "")

        excursion_section = ""
        if mfe is not None or mae is not None:
            excursion_section = (
                "\nExcursion (hourly bars over trade window):\n"
                f"- Max Favorable (MFE): {mfe:+.2%}\n" if mfe is not None else ""
            )
            if mae is not None:
                excursion_section += f"- Max Adverse (MAE): {mae:+.2%}\n"

        setup_section = ""
        if setup_context:
            setup_section = f"\nSetup Context:\n{setup_context}\n"

        variant_section = f"- Strategy variant: {variant}\n" if variant else ""

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

        return f"""You are a trading journal analyst writing a post-mortem for a human trader who is learning technical analysis. Explain this closed trade in plain, concrete, teaching-oriented language. Reference specific levels, indicators, or price events — no vague platitudes.

Trade Details:
- Symbol: {symbol}
{variant_section}- Entry: ${entry_price:.2f} on {entry_date}
- Exit: ${exit_price:.2f} on {exit_date}
- Return: {return_pct:+.2f}%
- Hold Duration: {hold_days} days
- Exit Reason: {exit_reason}
- Base Pattern: {base_pattern}
- Market Regime at Entry: {regime}
{excursion_section}{setup_section}{signal_section}
Write in 4 sections (use markdown H3 headers):

### Entry Thesis
Why did we buy? Name the specific setup and the TA indicators that fired (pivot / ORB level / Chan BSP type / SMA stack / VWAP / volume surge — whichever apply). Be concrete about the price levels.

### What Happened
Narrate the price path from entry to exit in 2-3 short beats. Reference MFE/MAE if they tell a story (e.g. "ran to +8% before giving back").

### Assessment
Did this trade {"succeed" if return_pct and return_pct > 0 else "fail"} for the reason we expected, or by luck? What market condition helped/hurt? Contrast against what a textbook outcome would look like.

### Lesson
One actionable, testable takeaway for future trades with this setup. Avoid generic advice.

Keep total response under 350 words. Quote prices and indicators precisely."""
