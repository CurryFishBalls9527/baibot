"""Pytest-wide fixtures.

Critical: set ``BAIBOT_RESULTS_DIR`` to a per-session tmpdir so tests
that exercise modules calling ``emit_event`` do not write to the real
``results/service_logs/events.jsonl``. Without this, the live watchdog
picks up synthetic test events on its next tail tick and sends real
ntfy/Telegram alerts (we caught this on 2026-04-24).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_results_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("baibot_test_results")
    prior = os.environ.get("BAIBOT_RESULTS_DIR")
    os.environ["BAIBOT_RESULTS_DIR"] = str(base)
    yield base
    if prior is None:
        os.environ.pop("BAIBOT_RESULTS_DIR", None)
    else:
        os.environ["BAIBOT_RESULTS_DIR"] = prior
