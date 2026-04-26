"""Build the watchdog's CompositeNotifier with its own ntfy topic and
Telegram chat — separate channel from trade-fill notifications.

Reads ``WATCHDOG_*`` env vars so system alerts land somewhere the user
can subscribe to independently. If neither backend is configured, the
process exits with code 2 — the watchdog must never run silent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from tradingagents.automation.composite_notifier import CompositeNotifier
from tradingagents.automation.notifier import NtfyNotifier
from tradingagents.automation.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def build_or_die(results_dir: str = "./results") -> CompositeNotifier:
    """Construct the watchdog notifier. Exit(2) if neither backend works.

    Watchdog uses dedicated env vars so trade fills (NTFY_TOPIC,
    TELEGRAM_CHAT_ID) and system alerts (WATCHDOG_NTFY_TOPIC,
    WATCHDOG_TELEGRAM_CHAT_ID) don't share a channel. Falls back to the
    trade channel only if the user explicitly opts in via
    WATCHDOG_REUSE_TRADE_CHANNEL=1 — otherwise the watchdog refuses to
    start, surfacing the misconfiguration via launchd.
    """
    ntfy_topic = (os.environ.get("WATCHDOG_NTFY_TOPIC") or "").strip()
    ntfy_server = (
        os.environ.get("WATCHDOG_NTFY_URL")
        or os.environ.get("NTFY_SERVER_URL")
        or "https://ntfy.sh"
    ).strip()
    tg_token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    tg_chat_id = (os.environ.get("WATCHDOG_TELEGRAM_CHAT_ID") or "").strip()

    trade_topic = (os.environ.get("NTFY_TOPIC") or "").strip()
    trade_chat = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()

    reuse = _bool_env("WATCHDOG_REUSE_TRADE_CHANNEL", default=False)
    if reuse:
        # Inherit unset watchdog channels from the trade channel.
        if not ntfy_topic and trade_topic:
            ntfy_topic = trade_topic
        if not tg_chat_id and trade_chat:
            tg_chat_id = trade_chat
    else:
        if ntfy_topic and trade_topic and ntfy_topic == trade_topic:
            logger.error(
                "WATCHDOG_NTFY_TOPIC equals NTFY_TOPIC — system alerts would mix "
                "with trade fills. Use a distinct topic, or set "
                "WATCHDOG_REUSE_TRADE_CHANNEL=1 to override."
            )
            sys.exit(2)
        if tg_chat_id and trade_chat and tg_chat_id == trade_chat:
            logger.error(
                "WATCHDOG_TELEGRAM_CHAT_ID equals TELEGRAM_CHAT_ID — system alerts "
                "would mix with trade fills. Use a distinct chat, or set "
                "WATCHDOG_REUSE_TRADE_CHANNEL=1 to override."
            )
            sys.exit(2)

    # Force separate dedupe state files so watchdog dedupe doesn't collide
    # with the scheduler's notifier dedupe.
    base = Path(results_dir).expanduser()
    watchdog_results_dir = base / "watchdog_notify"
    watchdog_results_dir.mkdir(parents=True, exist_ok=True)
    # NtfyNotifier and TelegramNotifier both build their state files under
    # `${results_dir}/notifications/`, so passing this as results_dir gives
    # us watchdog-scoped dedupe files at:
    #   results/watchdog_notify/notifications/{ntfy,telegram}_state.json

    ntfy_cfg = {
        "ntfy_enabled": bool(ntfy_topic),
        "ntfy_topic": ntfy_topic,
        "ntfy_server": ntfy_server,
        "ntfy_priority": "default",
        "ntfy_tags": ["warning"],
        "results_dir": str(watchdog_results_dir),
        "strategy_tag": "watchdog",
    }
    tg_cfg = {
        "telegram_enabled": bool(tg_token and tg_chat_id),
        "telegram_bot_token": tg_token,
        "telegram_chat_id": tg_chat_id,
        "results_dir": str(watchdog_results_dir),
        "strategy_tag": "watchdog",
    }

    ntfy = NtfyNotifier(ntfy_cfg)
    tg = TelegramNotifier(tg_cfg)
    composite = CompositeNotifier([ntfy, tg])

    if not composite.enabled:
        logger.error(
            "Watchdog notifier has no enabled backend. Set "
            "WATCHDOG_NTFY_TOPIC and/or WATCHDOG_TELEGRAM_CHAT_ID + "
            "TELEGRAM_BOT_TOKEN. Refusing to run silent."
        )
        sys.exit(2)

    backends = []
    if ntfy.enabled:
        backends.append(f"ntfy(topic={ntfy.topic} server={ntfy.server})")
    if tg.enabled:
        backends.append(f"telegram(chats={tg_chat_id})")
    logger.info("Watchdog notifier backends: %s", ", ".join(backends))
    return composite
