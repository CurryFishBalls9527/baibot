"""Watchdog main loop. Long-lived BackgroundScheduler that runs each
check on its own cadence, dedupes alerts, and forwards to the watchdog
notifier. Strict silence — no proactive heartbeats."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from . import checks
from .checks import Alert
from .notifier_factory import build_or_die
from .state import (
    WatchdogState,
    already_alerted,
    load_state,
    mark_alerted,
    save_state,
)

logger = logging.getLogger(__name__)


class Watchdog:
    def __init__(self) -> None:
        results_dir = os.environ.get("BAIBOT_RESULTS_DIR", "results")
        self.notifier = build_or_die(results_dir=results_dir)
        self.state = load_state(
            default_event_path=checks.events_jsonl_path(),
            default_log_path=checks.service_log_path(),
        )
        self.scheduler = BackgroundScheduler(timezone="US/Eastern")

    # ── Scheduling ──────────────────────────────────────────────────

    def _register_jobs(self) -> None:
        # Wrap every check so a single failing check doesn't poison the scheduler.
        def wrap(check_fn: Callable[[WatchdogState], List[Alert]], name: str):
            def runner():
                self._run_check(check_fn, name)
            return runner

        # 1. Scheduler liveness: 5min mkt-hrs, 30min off-hrs.
        self.scheduler.add_job(
            wrap(checks.check_scheduler_liveness, "scheduler_liveness"),
            trigger=CronTrigger(minute="*/5"),
            id="scheduler_liveness",
            misfire_grace_time=120,
        )

        # 2. Event tail: every 60s.
        self.scheduler.add_job(
            wrap(checks.check_event_tail, "event_tail"),
            trigger=CronTrigger(second="0,30"),
            id="event_tail",
            misfire_grace_time=30,
        )

        # 3. Drift: every 10min during market hours.
        self.scheduler.add_job(
            wrap(checks.check_drift, "drift"),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*/10",
            ),
            id="drift",
            misfire_grace_time=300,
        )

        # 4. Naked positions: every 15min during market hours.
        self.scheduler.add_job(
            wrap(checks.check_naked_positions, "naked_positions"),
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*/15",
            ),
            id="naked_positions",
            misfire_grace_time=300,
        )

        # 5. Stranded intraday: 16:10 ET Mon-Fri.
        self.scheduler.add_job(
            wrap(checks.check_stranded_intraday, "stranded_intraday"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=10),
            id="stranded_intraday",
            misfire_grace_time=600,
        )

        # 6. Job execution sanity: 16:30 ET Mon-Fri.
        self.scheduler.add_job(
            wrap(checks.check_job_execution_sanity, "job_execution_sanity"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
            id="job_execution_sanity",
            misfire_grace_time=600,
        )

        # 7. Daily activity sanity: 11:00 ET Mon-Fri.
        self.scheduler.add_job(
            wrap(checks.check_daily_activity_sanity, "daily_activity_sanity"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=11, minute=0),
            id="daily_activity_sanity",
            misfire_grace_time=600,
        )

        # 8. Log error sweep: every 5min.
        self.scheduler.add_job(
            wrap(checks.check_log_error_sweep, "log_error_sweep"),
            trigger=CronTrigger(minute="*/5"),
            id="log_error_sweep",
            misfire_grace_time=120,
        )

        # 9. Intraday regime gate stuck: 16:15 ET Mon-Fri (after stranded
        # check, before job-execution sanity). Catches the case where the
        # Minervini regime gate has been blocking every intraday scan for
        # 3+ recent trading days (config bug, stale SPY, or extended
        # correction the user should know about).
        self.scheduler.add_job(
            wrap(checks.check_intraday_regime_gate_stuck, "regime_gate_stuck"),
            trigger=CronTrigger(day_of_week="mon-fri", hour=16, minute=15),
            id="regime_gate_stuck",
            misfire_grace_time=600,
        )

    # ── Check execution ──────────────────────────────────────────────

    def _run_check(self, check_fn, name: str) -> None:
        try:
            alerts = check_fn(self.state)
        except Exception as e:
            logger.exception("watchdog check %s raised: %s", name, e)
            return
        for alert in alerts:
            self._maybe_send(alert)
        self.state.last_tick[name] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            save_state(self.state)
        except Exception as e:
            logger.warning("watchdog: save_state failed: %s", e)

    def _maybe_send(self, alert: Alert) -> None:
        if already_alerted(self.state, alert.dedupe_key, ttl_hours=alert.ttl_hours):
            return
        try:
            sent = self.notifier.send(
                alert.title,
                alert.body,
                priority=alert.priority,
                tags=alert.tags,
                dedupe_key=alert.dedupe_key,
            )
        except Exception as e:
            logger.warning("watchdog: notifier.send raised: %s", e)
            return
        if sent:
            mark_alerted(self.state, alert.dedupe_key)
            logger.info("watchdog: alerted %s", alert.dedupe_key)
        else:
            logger.warning(
                "watchdog: notifier.send returned False for %s", alert.dedupe_key
            )

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        self._register_jobs()
        self.scheduler.start()
        logger.info(
            "Watchdog started — %d jobs registered. Strict silence mode.",
            len(self.scheduler.get_jobs()),
        )

        def _shutdown(signum, frame):
            logger.info("watchdog: shutdown signal %s — stopping scheduler", signum)
            self.scheduler.shutdown(wait=False)
            try:
                save_state(self.state)
            except Exception:
                pass
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Block forever until signal arrives. APScheduler runs jobs on its
        # own threads; we just need to keep the main thread alive.
        while True:
            time_mod.sleep(3600)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    Watchdog().start()
    return 0
