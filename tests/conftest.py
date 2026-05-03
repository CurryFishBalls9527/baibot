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


# ── Web dashboard test fixtures ──────────────────────────────────────


@pytest.fixture(scope="session")
def web_server():
    """Boot the FastAPI dashboard on a free port; yield base URL.

    Used by tests/test_web_*.py. Only starts when those tests request it.
    A session-scoped subprocess so the boot cost is amortized across
    every API + UI test in the run.
    """
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path
    import requests as _requests

    repo_root = Path(__file__).resolve().parents[1]
    venv_py = repo_root / ".venv" / "bin" / "python"
    py_exe = str(venv_py) if venv_py.exists() else sys.executable

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [py_exe, "-m", "uvicorn", "tradingagents.web.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn died with rc={proc.returncode}")
        try:
            if _requests.get(f"{base}/api/health", timeout=0.5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("web server failed to come up within 20s")

    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
