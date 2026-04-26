"""External watchdog for the paper-trading scheduler.

Runs as a separate launchd job (`com.tradingagents.watchdog`). Tails the
structured-event stream and live state from each variant's DB / Alpaca
account, and emits ntfy / Telegram alerts on a dedicated channel
(``WATCHDOG_NTFY_TOPIC`` / ``WATCHDOG_TELEGRAM_CHAT_ID``) whenever
something looks wrong.

Strict silence: in healthy state the watchdog produces zero alerts.
"""
