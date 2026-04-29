"""Structured-event emitter for the watchdog to consume.

Writes one JSON line per anomaly to ``results/service_logs/events.jsonl``
(append-only). Sits *alongside* existing ``logger.error`` /
``logger.warning`` calls — does not replace them. The watchdog tails the
JSONL stream so it can parse without regex-grepping text logs.

Never raises. A disk error here must not take down the scheduler; the
fallback is to log via the standard ``logger``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Categories ──────────────────────────────────────────────────────


class Categories:
    """Stable identifiers for every event category the watchdog knows about.

    Keep these as plain strings — the watchdog matches by string equality
    so renaming any constant is a wire-protocol break.
    """

    # Broker / order rejects
    ORDER_REJECT = "order_reject"
    STALE_QUOTE_RETRY = "stale_quote_retry"
    WASH_TRADE_REJECT = "wash_trade_reject"
    BRACKET_LEG_MISSING = "bracket_leg_missing"

    # Position / exit invariants
    POSITION_STRANDED = "position_stranded"
    NAKED_POSITION = "naked_position"
    EXIT_SKIPPED = "exit_skipped"
    RS_FILTER_BYPASSED = "rs_filter_bypassed"
    ADD_ON_SUPERSESSION = "add_on_supersession"

    # Drift / reconciler
    DRIFT_DETECTED = "drift_detected"
    RECONCILER_TICK = "reconciler_tick"  # info-level liveness pulse

    # Scheduler / activity
    JOB_FAILED = "job_failed"
    ACTIVITY_GAP = "activity_gap"


_VALID_LEVELS = ("info", "warning", "error", "critical")


# ── Module-level state ──────────────────────────────────────────────

_seq_counter = count(1)
_seq_lock = threading.Lock()
_host = socket.gethostname()


def _events_path() -> Path:
    """Resolve the events.jsonl path — overridable by ``BAIBOT_RESULTS_DIR``
    so tests can redirect to a tmpdir."""
    base = Path(os.environ.get("BAIBOT_RESULTS_DIR", "results"))
    return base / "service_logs" / "events.jsonl"


def _next_seq() -> int:
    with _seq_lock:
        return next(_seq_counter)


# ── Public API ──────────────────────────────────────────────────────


def emit_event(
    category: str,
    *,
    level: str = "warning",
    variant: Optional[str] = None,
    symbol: Optional[str] = None,
    code: Optional[str] = None,
    message: str = "",
    context: Optional[dict] = None,
    fingerprint: Optional[str] = None,
) -> None:
    """Append one JSON event to ``results/service_logs/events.jsonl``.

    Never raises. If the write fails (disk full, permission, anything),
    the failure is logged via ``logger`` and we move on — the scheduler
    must keep running even if the watchdog's input file is unavailable.
    """
    try:
        if level not in _VALID_LEVELS:
            level = "warning"

        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seq": _next_seq(),
            "pid": os.getpid(),
            "host": _host,
            "level": level,
            "category": category,
            "variant": variant,
            "symbol": symbol,
            "code": code,
            "message": message,
            "fingerprint": fingerprint or _default_fingerprint(category, variant, symbol, code),
            "context": context or {},
        }

        path = _events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"

        # O_APPEND is atomic for single writes < PIPE_BUF (4 KB on macOS).
        # Records are well under that limit; concurrent emits from the
        # scheduler's APScheduler threads won't interleave.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as e:
        # Last-ditch: log and swallow. Never bubble up.
        try:
            logger.warning("emit_event failed (category=%s): %s", category, e)
        except Exception:
            pass


def _default_fingerprint(
    category: str,
    variant: Optional[str],
    symbol: Optional[str],
    code: Optional[str],
) -> str:
    parts = [category]
    if variant:
        parts.append(variant)
    if symbol:
        parts.append(symbol)
    if code:
        parts.append(str(code))
    return ":".join(parts)
