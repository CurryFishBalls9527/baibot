# TradingAgents/graph/signal_processing.py

import json
import logging
from typing import Union
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

STRUCTURED_SIGNAL_PROMPT = """You are a trading signal extractor. Analyze the financial report and extract a structured trading signal.

Return ONLY valid JSON with these fields:
{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": 0.0 to 1.0,
  "reasoning": "1-2 sentence summary",
  "stop_loss_pct": suggested stop-loss as percentage below entry (e.g. 0.05 for 5%), or null,
  "take_profit_pct": suggested take-profit as percentage above entry (e.g. 0.15 for 15%), or null,
  "timeframe": "day" or "swing" or "position"
}

Rules:
- confidence: 0.9+ = very strong conviction, 0.7-0.9 = strong, 0.5-0.7 = moderate, <0.5 = weak
- For HOLD signals, confidence reflects how certain you are that holding is correct
- stop_loss_pct and take_profit_pct are positive decimals (0.05 = 5%)
- timeframe: "day" = exit today, "swing" = days to weeks, "position" = weeks to months
- Return ONLY the JSON, no markdown fences, no extra text"""


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Legacy method: extract simple BUY/SELL/HOLD string."""
        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]
        return self.quick_thinking_llm.invoke(messages).content

    def process_signal_structured(self, full_signal: str, symbol: str = "") -> dict:
        """Extract a structured trading signal with confidence and risk levels.

        Returns:
            dict with keys: action, confidence, reasoning, stop_loss_pct,
            take_profit_pct, timeframe, symbol
        """
        messages = [
            ("system", STRUCTURED_SIGNAL_PROMPT),
            ("human", full_signal),
        ]

        raw = self.quick_thinking_llm.invoke(messages).content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse structured signal, falling back: {raw[:200]}")
            simple = self.process_signal(full_signal).strip().upper()
            parsed = {
                "action": simple if simple in ("BUY", "SELL", "HOLD") else "HOLD",
                "confidence": 0.5,
                "reasoning": "Fallback: could not parse structured signal",
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.15,
                "timeframe": "swing",
            }

        action = parsed.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        return {
            "symbol": symbol,
            "action": action,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": parsed.get("reasoning", ""),
            "stop_loss_pct": parsed.get("stop_loss_pct"),
            "take_profit_pct": parsed.get("take_profit_pct"),
            "timeframe": parsed.get("timeframe", "swing"),
        }
