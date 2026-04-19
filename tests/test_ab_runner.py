"""Tests for ABRunner's grouped-parallel variant dispatch."""

import threading
import time
import unittest
from unittest.mock import patch

from tradingagents.testing.ab_runner import ABRunner


class _FakeChan:
    """Mimics ChanOrchestrator's `.intraday_db` attribute for writer-group keying."""

    def __init__(self, name, intraday_db, sleep_s=0.2):
        self.name = name
        self.intraday_db = intraday_db
        self._sleep_s = sleep_s

    def _work(self, label):
        start = time.perf_counter()
        time.sleep(self._sleep_s)
        return {"variant": self.name, "label": label,
                "thread": threading.current_thread().name,
                "elapsed": time.perf_counter() - start}

    def run_daily_analysis(self):
        return self._work("daily_analysis")

    def bulk_refresh_30m_data(self):
        return self._work("bulk_refresh")

    def reconcile_orders(self):
        return self._work("reconcile")

    def run_daily_reflection(self):
        return self._work("reflection")

    def take_market_snapshot(self):
        return self._work("snapshot")


class _FakeMechanical:
    """Mimics Minervini Orchestrator surface (no shared DuckDB writer)."""

    def __init__(self, name, sleep_s=0.2):
        self.name = name
        self._sleep_s = sleep_s

    def _work(self, label):
        time.sleep(self._sleep_s)
        return {"variant": self.name, "label": label,
                "thread": threading.current_thread().name}

    def run_daily_analysis(self):
        return self._work("daily_analysis")

    def run_intraday_entry_scan(self):
        return self._work("intraday_entry_scan")

    def reconcile_orders(self):
        return self._work("reconcile")

    def run_daily_reflection(self):
        return self._work("reflection")

    def take_market_snapshot(self):
        return self._work("snapshot")


def _build_runner(orchestrators):
    runner = ABRunner.__new__(ABRunner)
    runner.experiment = None
    runner.orchestrators = orchestrators
    return runner


class ABRunnerParallelismTests(unittest.TestCase):

    def _patch_chan_isinstance(self):
        """Make `isinstance(_FakeChan, ChanOrchestrator)` return True.

        The runner imports ChanOrchestrator lazily; we patch the symbol at
        its usage site inside ab_runner.
        """
        # The runner calls `from tradingagents.automation.chan_orchestrator import ChanOrchestrator`
        # each time. Patch the source module to alias ChanOrchestrator to _FakeChan.
        return patch(
            "tradingagents.automation.chan_orchestrator.ChanOrchestrator",
            _FakeChan,
        )

    def test_independent_variants_run_in_parallel(self):
        """Non-Chan variants have distinct writer keys → run concurrently."""
        runner = _build_runner({
            "mech1": _FakeMechanical("mech1", sleep_s=0.3),
            "mech2": _FakeMechanical("mech2", sleep_s=0.3),
            "mech3": _FakeMechanical("mech3", sleep_s=0.3),
        })
        start = time.perf_counter()
        results = runner.run_daily_analysis()
        elapsed = time.perf_counter() - start

        self.assertEqual(set(results.keys()), {"mech1", "mech2", "mech3"})
        # Three 0.3s jobs with 3 workers should finish in ~0.35-0.5s, not 0.9s
        self.assertLess(elapsed, 0.7,
                        f"expected parallel (<0.7s), got {elapsed:.2f}s")

    def test_shared_duckdb_variants_serialize_within_group(self):
        """Chan variants sharing the same DuckDB path must run sequentially."""
        with self._patch_chan_isinstance():
            runner = _build_runner({
                "chan":    _FakeChan("chan",    "/tmp/shared.duckdb", sleep_s=0.3),
                "chan_v2": _FakeChan("chan_v2", "/tmp/shared.duckdb", sleep_s=0.3),
            })
            start = time.perf_counter()
            results = runner.run_daily_analysis()
            elapsed = time.perf_counter() - start

        self.assertEqual(set(results.keys()), {"chan", "chan_v2"})
        # Same writer key → serialized → should take at least 2 * sleep
        self.assertGreaterEqual(elapsed, 0.55,
                                f"expected serialized (>=0.55s), got {elapsed:.2f}s")

    def test_mixed_variants_serialize_chan_group_only(self):
        """Chan+chan_v2 serialize; mechanical runs in parallel with the group."""
        with self._patch_chan_isinstance():
            runner = _build_runner({
                "mech":    _FakeMechanical("mech",    sleep_s=0.4),
                "chan":    _FakeChan("chan",    "/tmp/shared.duckdb", sleep_s=0.3),
                "chan_v2": _FakeChan("chan_v2", "/tmp/shared.duckdb", sleep_s=0.3),
            })
            start = time.perf_counter()
            results = runner.run_daily_analysis()
            elapsed = time.perf_counter() - start

        self.assertEqual(set(results.keys()), {"mech", "chan", "chan_v2"})
        # mech (0.4s) runs in parallel with chan group (0.3+0.3=0.6s).
        # Total should be dominated by the chan group: ~0.6s, not 1.0s.
        self.assertLess(elapsed, 0.85,
                        f"expected parallel groups (<0.85s), got {elapsed:.2f}s")
        self.assertGreaterEqual(elapsed, 0.55,
                                f"expected chan group serialized (>=0.55s), got {elapsed:.2f}s")

    def test_variant_exception_isolated(self):
        """A failure in one variant must not break the others."""
        class _Boom(_FakeMechanical):
            def run_daily_analysis(self):
                raise RuntimeError("kaboom")

        runner = _build_runner({
            "ok":   _FakeMechanical("ok",   sleep_s=0.05),
            "boom": _Boom("boom",           sleep_s=0.05),
        })
        results = runner.run_daily_analysis()
        self.assertEqual(results["ok"]["variant"], "ok")
        self.assertIn("error", results["boom"])
        self.assertIn("kaboom", results["boom"]["error"])

    def test_bulk_refresh_dedupes_shared_db(self):
        """Variants pointing at the same intraday DB share one refresh call."""
        with self._patch_chan_isinstance():
            chan = _FakeChan("chan", "/tmp/shared.duckdb", sleep_s=0.05)
            chan_v2 = _FakeChan("chan_v2", "/tmp/shared.duckdb", sleep_s=0.05)
            runner = _build_runner({"chan": chan, "chan_v2": chan_v2})
            results = runner.bulk_refresh_chan_data()

        # One variant gets the real refresh, the other gets a deduped status
        refreshed = [n for n, r in results.items() if r.get("label") == "bulk_refresh"]
        deduped   = [n for n, r in results.items() if r.get("status") == "deduped"]
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(results[deduped[0]]["shared_with"], refreshed[0])

    def test_bulk_refresh_runs_distinct_db_paths_in_parallel(self):
        """Chan variants on different DB files each get their own refresh."""
        with self._patch_chan_isinstance():
            runner = _build_runner({
                "chan":    _FakeChan("chan",    "/tmp/a.duckdb", sleep_s=0.3),
                "chan_v2": _FakeChan("chan_v2", "/tmp/b.duckdb", sleep_s=0.3),
            })
            start = time.perf_counter()
            results = runner.bulk_refresh_chan_data()
            elapsed = time.perf_counter() - start

        self.assertEqual(
            {r.get("label") for r in results.values()},
            {"bulk_refresh"},
        )
        self.assertLess(elapsed, 0.55,
                        f"expected parallel (<0.55s), got {elapsed:.2f}s")

    def test_intraday_scan_dispatches_per_orchestrator_type(self):
        """Chan orchestrators use run_daily_analysis; others use run_intraday_entry_scan."""
        with self._patch_chan_isinstance():
            runner = _build_runner({
                "mech": _FakeMechanical("mech", sleep_s=0.05),
                "chan": _FakeChan("chan", "/tmp/shared.duckdb", sleep_s=0.05),
            })
            results = runner.run_intraday_scan()

        self.assertEqual(results["mech"]["label"], "intraday_entry_scan")
        self.assertEqual(results["chan"]["label"], "daily_analysis")

    def test_flatten_all_intraday_only_targets_intraday_variants(self):
        """Non-intraday variants must get a not_applicable status; intraday
        variants must have their flatten_all() called."""
        class _FakeIntraday:
            def __init__(self, name):
                self.name = name
                self.flatten_calls = 0
            def flatten_all(self):
                self.flatten_calls += 1
                return {"closed": [], "positions_closed": 0, "dry_run": True}

        intraday = _FakeIntraday("intraday_mechanical")
        runner = _build_runner({
            "mech": _FakeMechanical("mech"),
            "intraday_mechanical": intraday,
        })
        with patch(
            "tradingagents.automation.intraday_orchestrator.IntradayOrchestrator",
            _FakeIntraday,
        ):
            results = runner.flatten_all_intraday()

        self.assertEqual(results["mech"]["status"], "not_applicable")
        self.assertEqual(results["intraday_mechanical"]["positions_closed"], 0)
        self.assertEqual(intraday.flatten_calls, 1)


if __name__ == "__main__":
    unittest.main()
