"""Scheduler for automated daily trading runs."""

import logging
import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .orchestrator import Orchestrator
from .social_monitor import SocialFeedMonitor

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Wraps APScheduler to run the orchestrator on a daily schedule."""

    def __init__(self, config: dict):
        self.config = config
        self.experiment = None
        self.ab_runner = None
        if config.get("experiment_config_path"):
            from tradingagents.testing.ab_config import load_experiment
            from tradingagents.testing.ab_runner import ABRunner
            self.experiment = load_experiment(config["experiment_config_path"])
            self.ab_runner = ABRunner(self.experiment, config)
            self.orchestrator = None
            if self.experiment.reconciler_enabled:
                self.config.setdefault("reconciler_enabled", True)
                self.config.setdefault(
                    "reconciler_interval_minutes",
                    self.experiment.reconciler_interval_minutes,
                )
        else:
            self.orchestrator = Orchestrator(config)
        self.social_monitor = SocialFeedMonitor(config)
        self.scheduler = BlockingScheduler(timezone="US/Eastern")
        self._setup_jobs()
        self._setup_signal_handlers()

    def _setup_jobs(self):
        mode = self.config.get("trading_mode", "swing")

        snapshot_func = (
            self.ab_runner.take_market_snapshot
            if self.ab_runner
            else self.orchestrator.take_market_snapshot
        )
        analysis_func = (
            self.ab_runner.run_daily_analysis
            if self.ab_runner
            else self.orchestrator.run_daily_analysis
        )
        reflection_func = (
            self.ab_runner.run_daily_reflection
            if self.ab_runner
            else self.orchestrator.run_daily_reflection
        )

        open_snapshot_time = self.config.get("market_open_snapshot_time", "09:25")
        hour, minute = open_snapshot_time.split(":")
        self.scheduler.add_job(
            self._run_with_logging,
            args=[snapshot_func, "Market Open Snapshot"],
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=int(hour),
                minute=int(minute),
                timezone="US/Eastern",
            ),
            id="market_open_snapshot",
            name="Market Open Snapshot",
            misfire_grace_time=300,
        )
        logger.info(
            "Scheduled market open snapshot at %s ET (Mon-Fri)",
            open_snapshot_time,
        )

        # Morning bulk refresh of 30m bars for Chan universe
        if self.ab_runner:
            chan_refresh_time = self.config.get("chan_bulk_refresh_time", "09:35")
            hour, minute = chan_refresh_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.ab_runner.bulk_refresh_chan_data, "Chan Bulk 30m Refresh"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="chan_bulk_refresh",
                name="Chan Bulk 30m Refresh",
                misfire_grace_time=600,
            )
            logger.info(
                "Scheduled Chan bulk 30m refresh at %s ET (Mon-Fri)",
                chan_refresh_time,
            )

        # Swing trade: daily analysis before close
        if mode in ("swing", "both"):
            swing_time = self.config.get("swing_analysis_time", "15:30")
            hour, minute = swing_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[analysis_func, "Daily Swing Analysis"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="swing_analysis",
                name="Daily Swing Analysis",
                misfire_grace_time=300,
            )
            logger.info(f"Scheduled swing analysis at {swing_time} ET (Mon-Fri)")

        # Day trade: intraday scans
        if mode in ("day", "both"):
            interval = self.config.get("intraday_interval_minutes", 60)
            self.scheduler.add_job(
                self._run_with_logging,
                args=[analysis_func, "Intraday Scan"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour="10-15",
                    minute=f"*/{interval}" if interval < 60 else "0",
                    timezone="US/Eastern",
                ),
                id="intraday_scan",
                name="Intraday Scan",
                misfire_grace_time=120,
            )
            logger.info(f"Scheduled intraday scans every {interval}min (10AM-3PM ET)")

        # Intraday scan — mechanical (price checks) + Chan (30m signals)
        if self.ab_runner:
            scan_interval = self.config.get("intraday_scan_interval_minutes", 10)
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.ab_runner.run_intraday_scan, "Intraday Entry Scan"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour="10-15",
                    minute=f"*/{scan_interval}",
                    timezone="US/Eastern",
                ),
                id="intraday_entry_scan",
                name="Intraday Entry Scan (mechanical + Chan)",
                misfire_grace_time=120,
            )
            logger.info(
                "Scheduled intraday entry scan every %dmin (10AM-3:50PM ET)",
                scan_interval,
            )

        # Intraday mechanical EOD flatten — close any open intraday positions
        # before market close. Runs only when ab_runner is configured (the
        # variant lives inside the A/B framework).
        if self.ab_runner:
            flatten_time = self.config.get("intraday_flatten_time", "15:55")
            hour, minute = flatten_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.ab_runner.flatten_all_intraday, "Intraday EOD Flatten"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="intraday_eod_flatten",
                name="Intraday EOD Flatten",
                misfire_grace_time=60,
            )
            logger.info(
                "Scheduled intraday EOD flatten at %s ET (Mon-Fri)",
                flatten_time,
            )

        # Weekly strategy review — Saturday 09:00 ET. Uses gpt-5.2 (deep-think)
        # to produce per-variant markdown proposals. Gated by
        # `weekly_strategy_review_enabled` (default off).
        if self.config.get("weekly_strategy_review_enabled", False) and self.ab_runner:
            weekly_time = self.config.get("weekly_review_time", "09:00")
            hour, minute = weekly_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.ab_runner.run_weekly_strategy_review, "Weekly Strategy Review"],
                trigger=CronTrigger(
                    day_of_week="sat",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="weekly_strategy_review",
                name="Weekly Strategy Review",
                misfire_grace_time=3600,
            )
            logger.info(
                "Scheduled weekly strategy review at %s ET (Saturday)", weekly_time,
            )

        # Daily per-trade review — Mon-Fri 17:00 ET (after the 16:30 reflection
        # cron, so trade_outcomes has latest fills). Chart + gpt-4o-mini LLM
        # per closed trade. Gated by `daily_trade_review_enabled` (default off).
        if self.config.get("daily_trade_review_enabled", False) and mode in ("swing", "both"):
            review_time = self.config.get("daily_trade_review_time", "17:00")
            hour, minute = review_time.split(":")
            review_func = (
                self.ab_runner.run_daily_trade_review
                if self.ab_runner
                else self.orchestrator.run_daily_trade_review
            )
            self.scheduler.add_job(
                self._run_with_logging,
                args=[review_func, "Daily Trade Review"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="daily_trade_review",
                name="Daily Trade Review",
                misfire_grace_time=1800,
            )
            logger.info(
                "Scheduled daily trade review at %s ET (Mon-Fri)", review_time,
            )

        # Exit-check pass — 5-min cadence, rules-based only (no LLM). Runs
        # `Orchestrator.run_exit_check_pass` for each swing variant so
        # trailing stops / dead-money / 50-DMA exits don't wait for the
        # once-daily scan to fire. Idempotent via in-flight-order guard.
        # Chan / intraday variants short-circuit to not_applicable.
        if self.config.get("exit_check_enabled", True) and mode in ("swing", "both"):
            exit_check_interval = max(
                int(self.config.get("exit_check_interval_minutes", 5)), 1
            )
            exit_check_func = (
                self.ab_runner.run_exit_check_pass
                if self.ab_runner
                else self.orchestrator.run_exit_check_pass
            )
            self.scheduler.add_job(
                self._run_with_logging,
                args=[exit_check_func, "Exit Check Pass"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour="9-16",
                    minute=f"*/{exit_check_interval}"
                    if exit_check_interval < 60
                    else "0",
                    timezone="US/Eastern",
                ),
                id="exit_check_pass",
                name="Exit Check Pass",
                misfire_grace_time=120,
            )
            logger.info(
                "Scheduled exit-check pass every %dmin (9AM-4PM ET, swing variants)",
                exit_check_interval,
            )

        # Order reconciliation (Track P-SYNC) — sync local DB against broker.
        # Off by default; enable via `reconciler_enabled: true` in config.
        if self.config.get("reconciler_enabled", False):
            reconcile_interval = max(
                int(self.config.get("reconciler_interval_minutes", 5)), 1
            )
            reconcile_func = (
                self.ab_runner.reconcile_orders
                if self.ab_runner
                else self.orchestrator.reconcile_orders
            )
            self.scheduler.add_job(
                self._run_with_logging,
                args=[reconcile_func, "Order Reconciliation"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour="9-16",
                    minute=f"*/{reconcile_interval}"
                    if reconcile_interval < 60
                    else "0",
                    timezone="US/Eastern",
                ),
                id="order_reconciliation",
                name="Order Reconciliation",
                misfire_grace_time=120,
            )
            logger.info(
                "Scheduled order reconciliation every %dmin (9AM-4PM ET)",
                reconcile_interval,
            )

        # Daily reflection after market close
        reflection_time = self.config.get("reflection_time", "16:30")
        hour, minute = reflection_time.split(":")
        self.scheduler.add_job(
            self._run_with_logging,
            args=[reflection_func, "Daily Reflection & Report"],
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=int(hour),
                minute=int(minute),
                timezone="US/Eastern",
            ),
            id="daily_reflection",
            name="Daily Reflection & Report",
            misfire_grace_time=600,
        )
        logger.info(f"Scheduled daily reflection at {reflection_time} ET (Mon-Fri)")

        # Held-position health check — Mon-Fri 17:15 ET (after the 17:00
        # closed-trade review runs, so both sets of files land together
        # in results/daily_reviews/YYYY-MM-DD/). Gated by
        # `held_position_review_enabled` (default True).
        if self.config.get("held_position_review_enabled", True) and mode in ("swing", "both"):
            held_time = self.config.get("held_position_review_time", "17:15")
            hour, minute = held_time.split(":")
            held_func = (
                self.ab_runner.run_held_position_review
                if self.ab_runner
                else self.orchestrator.run_held_position_review
            )
            self.scheduler.add_job(
                self._run_with_logging,
                args=[held_func, "Held Position Review"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="held_position_review",
                name="Held Position Review",
                misfire_grace_time=1800,
            )
            logger.info(
                "Scheduled held-position review at %s ET (Mon-Fri)", held_time,
            )

        # Heartbeats — alert when the daily / weekly review cron was
        # supposed to fire but didn't (launchd died, scheduler crashed
        # silently). `_run_with_logging` already alerts on raises; this
        # covers the complementary silent-skip case.
        if self.config.get("daily_review_heartbeat_enabled", True) and mode in ("swing", "both"):
            from tradingagents.automation.heartbeat import check_daily_review_ran
            heartbeat_time = self.config.get("daily_review_heartbeat_time", "18:00")
            hour, minute = heartbeat_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[lambda: check_daily_review_ran(self.config), "Daily Review Heartbeat"],
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="daily_review_heartbeat",
                name="Daily Review Heartbeat",
                misfire_grace_time=1800,
            )
            logger.info(
                "Scheduled daily review heartbeat at %s ET (Mon-Fri)",
                heartbeat_time,
            )

        if self.config.get("weekly_review_heartbeat_enabled", True) and self.ab_runner:
            from tradingagents.automation.heartbeat import check_weekly_review_ran
            weekly_hb_time = self.config.get("weekly_review_heartbeat_time", "11:00")
            hour, minute = weekly_hb_time.split(":")
            self.scheduler.add_job(
                self._run_with_logging,
                args=[lambda: check_weekly_review_ran(self.config), "Weekly Review Heartbeat"],
                trigger=CronTrigger(
                    day_of_week="sat",
                    hour=int(hour),
                    minute=int(minute),
                    timezone="US/Eastern",
                ),
                id="weekly_review_heartbeat",
                name="Weekly Review Heartbeat",
                misfire_grace_time=3600,
            )
            logger.info(
                "Scheduled weekly review heartbeat at %s ET (Saturday)",
                weekly_hb_time,
            )

        if self.config.get("social_monitor_enabled", False):
            interval = max(int(self.config.get("social_check_interval_minutes", 30)), 1)
            self.scheduler.add_job(
                self._run_with_logging,
                args=[self.social_monitor.check_once, "Social Feed Monitor"],
                trigger=CronTrigger(
                    hour="7-22",
                    minute=f"*/{interval}" if interval < 60 else "0",
                    timezone="US/Eastern",
                ),
                id="social_feed_monitor",
                name="Social Feed Monitor",
                misfire_grace_time=300,
            )
            logger.info(
                "Scheduled social monitor every %smin (7AM-10PM ET)",
                interval,
            )

    def _run_with_logging(self, func, label: str):
        """Execute a job with structured logging."""
        logger.info(f"{'=' * 50}")
        logger.info(f"JOB START: {label} at {datetime.now()}")
        logger.info(f"{'=' * 50}")
        try:
            result = func()
            logger.info(f"JOB DONE: {label} — {result}")
        except Exception as e:
            logger.error(f"JOB FAILED: {label} — {e}", exc_info=True)
            notifier = (
                self.orchestrator.notifier
                if self.orchestrator
                else list(self.ab_runner.orchestrators.values())[0].notifier
            )
            notifier.send(
                "TradingAgents Job Failed",
                f"{label}\n{type(e).__name__}: {e}",
                priority="high",
                tags=["warning", "rotating_light"],
                dedupe_key=f"job-failed:{label}:{datetime.now().date().isoformat()}:{type(e).__name__}",
            )

    def _setup_signal_handlers(self):
        """Graceful shutdown on SIGINT/SIGTERM."""
        def _shutdown(signum, frame):
            logger.info("Shutdown signal received. Stopping scheduler...")
            self.scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

    def start(self):
        """Start the scheduler (blocks forever)."""
        logger.info("=" * 60)
        logger.info("TradingAgents Automation Scheduler Starting")
        logger.info(f"Mode: {self.config.get('trading_mode', 'swing')}")
        logger.info(f"Paper: {self.config.get('paper_trading', True)}")
        logger.info(f"Watchlist: {self.config.get('watchlist', [])}")
        logger.info("=" * 60)

        # Log which notifier backends have credentials — without this line,
        # a misconfigured notifier sends job-failure alerts to the void and
        # the user has no visibility that alerts are silently dropping.
        try:
            from tradingagents.automation.heartbeat import log_notifier_banner
            log_notifier_banner(self.config)
        except Exception as e:
            logger.debug(f"Notifier banner failed: {e}")

        # Take initial snapshot
        if self.orchestrator:
            try:
                tracker = getattr(self.orchestrator, "tracker", None)
                if tracker is not None:
                    tracker.take_daily_snapshot()
            except Exception as e:
                logger.warning(f"Could not take initial snapshot: {e}")
        elif self.ab_runner:
            for name, orch in self.ab_runner.orchestrators.items():
                tracker = getattr(orch, "tracker", None)
                if tracker is None:
                    logger.debug(
                        "Skipping initial snapshot for variant '%s': no tracker",
                        name,
                    )
                    continue
                try:
                    tracker.take_daily_snapshot()
                except Exception as e:
                    logger.warning(
                        "Could not take initial snapshot for variant '%s': %s",
                        name,
                        e,
                    )

        # Print upcoming jobs
        jobs = self.scheduler.get_jobs()
        for job in jobs:
            next_run = getattr(job, "next_run_time", None)
            if next_run is not None:
                logger.info(f"  Job: {job.name} -> next run: {next_run}")
            else:
                logger.info(f"  Job: {job.name} -> trigger: {job.trigger}")

        self.scheduler.start()

    def run_now(self):
        """Immediately run analysis (for manual/testing use)."""
        logger.info("Manual trigger: running analysis NOW")
        if self.ab_runner:
            return self.ab_runner.run_daily_analysis()
        return self.orchestrator.run_daily_analysis()
