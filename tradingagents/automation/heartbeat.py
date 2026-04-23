"""Scheduler heartbeats — alert when a cron job was supposed to run but didn't.

`_run_with_logging` in scheduler.py already alerts when a cron raises.
This module covers the complementary case: the cron never fires at all
(launchd died, APScheduler crashed silently, etc.). For each critical
cron, a heartbeat job runs ~1h later and asserts that the job's
expected output exists. If missing, notifier.send fires.

Everything here is read-only against the filesystem. No DB writes,
no broker touches.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_DAILY_REVIEW_ROOT = Path("results/daily_reviews")
_WEEKLY_REVIEW_ROOT = Path("results/weekly_reviews")


def _iso_week(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _notifier_for(config: dict):
    """Build notifier if one isn't on the orchestrator's hot path."""
    try:
        from tradingagents.automation.notifier import build_notifier
        return build_notifier(config)
    except Exception as e:
        logger.warning("heartbeat: notifier build failed: %s", e)
        return None


# ─────────────────── daily review heartbeat ──────────────────

def check_daily_review_ran(config: dict, review_date: Optional[date] = None) -> dict:
    """Verify today's daily review produced at least one output file.

    Expected artifacts (either is enough):
      - `results/daily_reviews/YYYY-MM-DD/` directory exists AND contains
        at least one `*.md` file (variant summary or per-trade review)
      - OR the variant had zero closed trades today — in which case the
        cron still runs and logs "no closed trades" without writing
        files. We treat a missing directory on a no-trade day as OK
        IFF the scheduler log shows "JOB DONE: Daily Trade Review" in
        the last 90 minutes.

    Returns a summary dict. Sends notifier alert if the directory is
    missing AND no recent DONE log line is found.
    """
    if not config.get("daily_review_heartbeat_enabled", True):
        return {"status": "disabled"}

    review_date = review_date or date.today()
    # Only expect reviews on trading days (Mon-Fri).
    if review_date.weekday() >= 5:
        return {"status": "weekend_skip", "date": review_date.isoformat()}

    day_dir = _DAILY_REVIEW_ROOT / review_date.isoformat()
    md_files = list(day_dir.glob("*.md")) if day_dir.exists() else []

    # A recent "JOB DONE" entry in the scheduler log also counts as proof
    # the cron fired — covers the zero-closed-trades case where no files
    # would be written.
    log_ok = _recent_job_success(
        "Daily Trade Review",
        max_age=timedelta(minutes=90),
    )

    if md_files or log_ok:
        return {
            "status": "ok", "date": review_date.isoformat(),
            "files": len(md_files), "log_seen": log_ok,
        }

    # Silent skip — alert.
    notifier = _notifier_for(config)
    if notifier is not None:
        try:
            notifier.send(
                title="Daily review did not run",
                message=(
                    f"No output in {day_dir} and no JOB DONE in the last "
                    f"90 min. Scheduler may have crashed or been stopped."
                ),
                priority="high",
                tags=["warning", "rotating_light"],
                dedupe_key=f"heartbeat:daily_review:{review_date.isoformat()}",
            )
        except Exception as e:
            logger.warning("heartbeat alert send failed: %s", e)
    logger.warning(
        "heartbeat: daily review for %s did not produce output",
        review_date.isoformat(),
    )
    return {
        "status": "missing", "date": review_date.isoformat(),
        "files": 0, "log_seen": False,
    }


# ─────────────────── weekly review heartbeat ─────────────────

def check_weekly_review_ran(config: dict, review_date: Optional[date] = None) -> dict:
    """Same idea for the Saturday weekly review."""
    if not config.get("weekly_review_heartbeat_enabled", True):
        return {"status": "disabled"}

    review_date = review_date or date.today()
    iso = _iso_week(review_date)
    week_dir = _WEEKLY_REVIEW_ROOT / iso
    md_files = list(week_dir.glob("*.md")) if week_dir.exists() else []
    log_ok = _recent_job_success(
        "Weekly Strategy Review",
        max_age=timedelta(hours=3),
    )

    if md_files or log_ok:
        return {"status": "ok", "iso_week": iso, "files": len(md_files),
                "log_seen": log_ok}

    notifier = _notifier_for(config)
    if notifier is not None:
        try:
            notifier.send(
                title="Weekly review did not run",
                message=(
                    f"No output in {week_dir} and no JOB DONE in the last "
                    f"3h. Scheduler may have crashed."
                ),
                priority="high",
                tags=["warning", "rotating_light"],
                dedupe_key=f"heartbeat:weekly_review:{iso}",
            )
        except Exception as e:
            logger.warning("heartbeat alert send failed: %s", e)
    return {"status": "missing", "iso_week": iso,
            "files": 0, "log_seen": False}


# ─────────────────── helpers ─────────────────────────────────

_SCHEDULER_LOG_PATH = Path("results/service_logs/automation_service.out.log")


def _recent_job_success(label: str, max_age: timedelta) -> bool:
    """True if the scheduler log has `JOB DONE: {label}` within max_age.

    `_run_with_logging` writes `JOB DONE: {label} — {result}` on success
    (scheduler.py:L288+). Tail the last ~500 lines to avoid scanning
    multi-MB files.
    """
    if not _SCHEDULER_LOG_PATH.exists():
        return False
    try:
        # Tail last 500 lines — cheap for ~100 KB files.
        lines = _SCHEDULER_LOG_PATH.read_text(encoding="utf-8").splitlines()[-500:]
    except Exception:
        return False
    needle = f"JOB DONE: {label}"
    cutoff = datetime.now() - max_age
    for line in reversed(lines):
        if needle not in line:
            continue
        # Line starts with "YYYY-MM-DD HH:MM:SS ..."
        try:
            ts_str = line[:19]
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        return ts >= cutoff
    return False


def log_notifier_banner(config: dict) -> None:
    """Log which notifier backends have credentials at scheduler startup.

    If NEITHER is configured we emit a WARNING so crashed-cron alerts
    don't go silently nowhere. Called once from `TradingScheduler.start`.
    """
    try:
        from tradingagents.automation.notifier import NtfyNotifier
        from tradingagents.automation.telegram_notifier import TelegramNotifier
    except Exception:
        return

    ntfy = NtfyNotifier(config)
    tg = TelegramNotifier(config)
    backends = []
    if ntfy.enabled:
        backends.append(f"ntfy(topic={ntfy.topic})")
    if tg.enabled:
        backends.append("telegram")
    if backends:
        logger.info(
            "Notifier backends enabled: %s — scheduler alerts WILL deliver.",
            ", ".join(backends),
        )
    else:
        logger.warning(
            "No notifier backends are enabled (neither ntfy nor telegram). "
            "Scheduler failure alerts and heartbeat misses WILL go silent. "
            "Set ntfy_enabled or telegram_enabled in config + provide creds."
        )
