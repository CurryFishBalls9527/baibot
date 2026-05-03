"""Headless-Chrome UI regression tests for the dashboard.

Spawns the Node-based runner (``tests/web_ui_runner.js``) which drives
a real Chrome via Chrome DevTools Protocol. Each line of the runner's
stdout is one test result (JSON). We parse the line stream and surface
each as an individual pytest test via parametrize-after-collection so
failures point at the specific UI scenario.

Skips cleanly if Node, the chrome-remote-interface npm package, or
Google Chrome aren't available — these tests are integration-grade and
not all environments will have them.

The bugs each test pins:
  tab_switch_chart_recovery        — chart blank after tab-switch + back
  tab_switch_chart_recovery_stress — same, under repeated toggles
  today_loads_account_on_direct_url — TODAY blank when hit via /#today
  variant_change_refreshes_today   — variant dropdown didn't refresh TODAY
  daily_review_not_truncated       — markdown clipped under chart iframe
  ideas_tab_hidden_from_nav        — IDEAS removed from nav while disabled
  no_js_exceptions_*                — uncaught console errors during session
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_RUNNER = _HERE / "web_ui_runner.js"
_NODE_DEPS = _HERE / "node_modules" / "chrome-remote-interface"
_CHROME_DEFAULT = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _have_node() -> bool:
    return shutil.which("node") is not None


def _have_chrome() -> bool:
    return os.path.exists(os.environ.get("CHROME_BINARY", _CHROME_DEFAULT))


def _ensure_node_deps() -> bool:
    """Install chrome-remote-interface in tests/ if missing. Returns True
    on success, False if npm isn't available."""
    if _NODE_DEPS.exists():
        return True
    if shutil.which("npm") is None:
        return False
    # tests/package.json declares the dep; npm install pulls it locally
    pkg = _HERE / "package.json"
    if not pkg.exists():
        pkg.write_text(json.dumps({
            "name": "baibot-web-ui-tests",
            "private": True,
            "dependencies": {"chrome-remote-interface": "^0.33.0"},
        }, indent=2))
    try:
        subprocess.run(
            ["npm", "install", "--silent", "--no-audit", "--no-fund"],
            cwd=str(_HERE),
            check=True, timeout=120,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return _NODE_DEPS.exists()


def _run_ui_suite(base_url: str) -> tuple[list[dict], dict]:
    """Run the Node runner; return (results_per_test, summary)."""
    proc = subprocess.run(
        ["node", str(_RUNNER), base_url],
        cwd=str(_HERE),
        capture_output=True, text=True, timeout=180,
    )
    results: list[dict] = []
    summary: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in obj:
            summary = obj["summary"]
        elif "test" in obj:
            results.append(obj)
    return results, summary


@pytest.fixture(scope="module")
def ui_results(web_server):
    """Run the UI suite once per module; return all test results."""
    if not _have_node():
        pytest.skip("node not available — UI tests require Node.js")
    if not _have_chrome():
        pytest.skip(f"Chrome not found at {_CHROME_DEFAULT}; set CHROME_BINARY")
    if not _ensure_node_deps():
        pytest.skip("chrome-remote-interface not installable (npm missing or offline)")

    results, summary = _run_ui_suite(web_server)
    if not results:
        pytest.fail("UI runner produced no results — runner crashed or no output")
    return {"results": results, "summary": summary}


def _result(ui_results, name: str) -> dict:
    for r in ui_results["results"]:
        if r["test"] == name:
            return r
    pytest.fail(f"UI runner did not emit a result for test {name!r}")


# ── Each scenario as its own pytest test, fed by the same runner ────


@pytest.mark.parametrize("scenario", [
    "no_js_exceptions_on_load",
    "initial_chart_renders",
    "tab_switch_chart_recovery",
    "tab_switch_chart_recovery_stress",
    "today_loads_account_on_direct_url",
    "variant_change_refreshes_today",
    "daily_review_not_truncated",
    "ideas_tab_hidden_from_nav",
    "reviews_overview_cards_present",
    "risk_page_renders_all_sections",
    "ai_synthesis_does_not_auto_fire",          # cost-protection guard
    "performance_renders_four_charts",
    "escapeHtml_handles_non_strings",
    "log_page_renders_and_stops_on_leave",
    "closed_trade_shows_entry_and_exit_price_lines",
    "no_js_exceptions_during_session",
])
def test_ui_scenario(ui_results, scenario):
    r = _result(ui_results, scenario)
    assert r["ok"], f"{scenario}: {r.get('detail', '(no detail)')}"
