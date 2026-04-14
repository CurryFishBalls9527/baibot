"""Telegram Bot API notification helper."""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib import parse, request

logger = logging.getLogger(__name__)

# Map ntfy tag shortcodes used in the codebase to Unicode emoji.
_EMOJI_MAP = {
    "moneybag": "\U0001f4b0",
    "money_with_wings": "\U0001f4b8",
    "chart_with_upwards_trend": "\U0001f4c8",
    "rotating_light": "\U0001f6a8",
    "warning": "\u26a0\ufe0f",
    "eyes": "\U0001f440",
    "sunrise": "\U0001f305",
    "memo": "\U0001f4dd",
    "mag": "\U0001f50d",
    "bar_chart": "\U0001f4ca",
    "newspaper": "\U0001f4f0",
    "speech_balloon": "\U0001f4ac",
    "iphone": "\U0001f4f1",
    "test_tube": "\U0001f9ea",
    "trading_chart": "\U0001f4c8",
}

_MAX_MESSAGE_LEN = 4096


class TelegramNotifier:
    """Send notifications via the Telegram Bot API."""

    def __init__(self, config: dict):
        token = str(config.get("telegram_bot_token", "") or "").strip()
        raw_chat_ids = str(config.get("telegram_chat_id", "") or "").strip()
        self.enabled = bool(config.get("telegram_enabled", False) and token and raw_chat_ids)
        self.token = token
        self.chat_ids = [
            cid.strip() for cid in raw_chat_ids.split(",") if cid.strip()
        ]
        self.strategy_tag = str(config.get("strategy_tag", "") or "").strip()
        self.state_path = (
            Path(config.get("results_dir", "./results"))
            / "notifications"
            / "telegram_state.json"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Expose for CompositeNotifier / status reporting compatibility
        self.topic = raw_chat_ids
        self.server = "https://api.telegram.org"

    def send(
        self,
        title: str,
        message: str,
        *,
        priority: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        dedupe_key: Optional[str] = None,
        click: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            return False
        if dedupe_key and self._already_sent(dedupe_key):
            return False

        text = self._format(title, message, priority=priority, tags=tags, click=click)

        success = False
        for chat_id in self.chat_ids:
            if self._send_to_chat(chat_id, text):
                success = True

        if success and dedupe_key:
            self._mark_sent(dedupe_key)
        return success

    # ── Formatting ──────────────────────────────────────────────────

    def _format(
        self,
        title: str,
        message: str,
        *,
        priority: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        click: Optional[str] = None,
    ) -> str:
        parts: list[str] = []

        # Strategy hashtag
        if self.strategy_tag:
            parts.append(f"#{html.escape(self.strategy_tag)}")

        # Emoji from tags
        emoji = ""
        if tags:
            emoji = "".join(_EMOJI_MAP.get(t, "") for t in tags)

        # Priority indicator
        prio = priority or ""
        if prio == "urgent":
            emoji = "\U0001f534 " + emoji  # 🔴
        elif prio == "high":
            emoji = "\U0001f6a8 " + emoji  # 🚨

        # Title line
        title_line = f"<b>{emoji} {html.escape(title)}</b>" if emoji else f"<b>{html.escape(title)}</b>"
        parts.append(title_line)

        # Body
        parts.append(html.escape(message))

        # Click URL
        if click:
            parts.append(f'<a href="{html.escape(click)}">Open</a>')

        text = "\n".join(parts)

        # Truncate to Telegram limit
        if len(text) > _MAX_MESSAGE_LEN:
            text = text[: _MAX_MESSAGE_LEN - 20] + "\n[truncated]"

        return text

    # ── HTTP ────────────────────────────────────────────────────────

    def _send_to_chat(self, chat_id: str, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as resp:
                resp.read()
            logger.info("Sent Telegram notification to chat %s", chat_id)
            return True
        except Exception as exc:
            logger.warning("Failed to send Telegram notification to %s: %s", chat_id, exc)
            return False

    # ── Deduplication ───────────────────────────────────────────────

    def _already_sent(self, key: str) -> bool:
        state = self._load_state()
        return key in state

    def _mark_sent(self, key: str):
        state = self._load_state()
        state[key] = datetime.now(timezone.utc).isoformat()
        if len(state) > 500:
            items = sorted(state.items(), key=lambda item: item[1], reverse=True)[:500]
            state = dict(items)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
