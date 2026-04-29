"""Watchdog state — JSON file holding offsets and dedupe memory.

Atomic write via tmp + os.replace so a SIGKILL mid-write doesn't corrupt
the file. Prunes alert_dedupe to last 14 days on every save.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEDUPE_TTL_DAYS = 14

# APScheduler runs every check on its own thread, so two jobs scheduled on
# the same minute boundary (e.g. scheduler_liveness + log_error_sweep) can
# race on the .tmp → state.json rename. Serialize save_state to one writer.
_SAVE_LOCK = threading.Lock()


@dataclass
class TailedFileState:
    path: str
    offset: int = 0
    inode: Optional[int] = None


@dataclass
class WatchdogState:
    events_jsonl: TailedFileState
    service_log: TailedFileState
    # alert_dedupe: dedupe_key -> ISO timestamp of last send
    alert_dedupe: Dict[str, str] = field(default_factory=dict)
    # last_tick: check_name -> ISO timestamp of last successful run
    last_tick: Dict[str, str] = field(default_factory=dict)
    # rolling_buckets: arbitrary in-memory counters (e.g. recent rejects)
    rolling_buckets: Dict[str, Any] = field(default_factory=dict)
    # last_seen_event_seq: monotonic dedupe per (host, pid, seq) for crash-safety
    last_event_seq: Dict[str, int] = field(default_factory=dict)


def state_path() -> Path:
    base = Path(os.environ.get("BAIBOT_RESULTS_DIR", "results"))
    return base / "watchdog" / "state.json"


def load_state(default_event_path: Path, default_log_path: Path) -> WatchdogState:
    path = state_path()
    if not path.exists():
        return WatchdogState(
            events_jsonl=TailedFileState(path=str(default_event_path)),
            service_log=TailedFileState(path=str(default_log_path)),
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("watchdog state.json unreadable, resetting: %s", e)
        return WatchdogState(
            events_jsonl=TailedFileState(path=str(default_event_path)),
            service_log=TailedFileState(path=str(default_log_path)),
        )

    def _tailed(d: dict, default_path: Path) -> TailedFileState:
        return TailedFileState(
            path=d.get("path", str(default_path)),
            offset=int(d.get("offset", 0) or 0),
            inode=d.get("inode"),
        )

    return WatchdogState(
        events_jsonl=_tailed(raw.get("events_jsonl", {}), default_event_path),
        service_log=_tailed(raw.get("service_log", {}), default_log_path),
        alert_dedupe=dict(raw.get("alert_dedupe", {})),
        last_tick=dict(raw.get("last_tick", {})),
        rolling_buckets=dict(raw.get("rolling_buckets", {})),
        last_event_seq=dict(raw.get("last_event_seq", {})),
    )


def save_state(state: WatchdogState) -> None:
    state.alert_dedupe = _prune_dedupe(state.alert_dedupe)
    payload = {
        "events_jsonl": {
            "path": state.events_jsonl.path,
            "offset": state.events_jsonl.offset,
            "inode": state.events_jsonl.inode,
        },
        "service_log": {
            "path": state.service_log.path,
            "offset": state.service_log.offset,
            "inode": state.service_log.inode,
        },
        "alert_dedupe": state.alert_dedupe,
        "last_tick": state.last_tick,
        "rolling_buckets": state.rolling_buckets,
        "last_event_seq": state.last_event_seq,
    }
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-write unique tmp name keeps concurrent writers from racing on the
    # same .tmp path; the lock serializes the rename so the last writer wins.
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with _SAVE_LOCK:
        os.replace(tmp, path)


def _prune_dedupe(dedupe: Dict[str, str]) -> Dict[str, str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=_DEDUPE_TTL_DAYS)
    pruned = {}
    for key, ts_iso in dedupe.items():
        try:
            ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        except Exception:
            continue
        if ts >= cutoff:
            pruned[key] = ts_iso
    return pruned


def already_alerted(state: WatchdogState, dedupe_key: str, ttl_hours: int = 24) -> bool:
    """True if `dedupe_key` was sent within the last `ttl_hours`."""
    last = state.alert_dedupe.get(dedupe_key)
    if not last:
        return False
    try:
        ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except Exception:
        return False
    return (datetime.now(timezone.utc) - ts) < timedelta(hours=ttl_hours)


def mark_alerted(state: WatchdogState, dedupe_key: str) -> None:
    state.alert_dedupe[dedupe_key] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
