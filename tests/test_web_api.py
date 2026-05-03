"""Web dashboard API regression tests.

Covers the bugs we hit during the parity build so they don't silently
come back:

  * daily-bar timestamps must be midnight-UTC and a fill marker must
    align to a bar's time (lightweight-charts daily resolution + marker
    alignment) — was 04:00 UTC after the ET tz fix, broke chart render
  * intraday-bar timestamps must NOT be midnight-aligned (they're real
    instants) — guard against accidentally daily-aligning them
  * /api/today/{variant} returns the shape the frontend reads
  * /api/risk/* endpoints don't 500 even with sparse data
  * /api/reviews/overview, /api/log/tail, /api/service_status return
    well-formed JSON

The session fixture lives in tests/conftest.py and boots uvicorn on a
free port; tests are read-only against the live DBs so they don't
mutate state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import pytest
import requests


# ── Helpers ──────────────────────────────────────────────────────────


def _get(base: str, path: str, **kwargs) -> requests.Response:
    return requests.get(f"{base}{path}", timeout=15, **kwargs)


def _is_midnight_utc(unix_ts: int) -> bool:
    """True iff the unix timestamp lands on 00:00:00 UTC of some date."""
    return unix_ts % 86400 == 0


# ── Smoke: every documented route returns 200 ────────────────────────


@pytest.mark.parametrize("path", [
    "/api/health",
    "/api/variants",
    "/api/service_status",
    "/api/performance/snapshots",
    "/api/performance/trades",
    "/api/risk/correlation",
    "/api/risk/sizing",
    "/api/risk/regime",
    "/api/risk/patterns",
    "/api/risk/llm_cost",
    "/api/risk/outcomes",
    "/api/reviews/overview",
    "/api/proposals",
    "/api/log/tail",
])
def test_all_routes_200(web_server, path):
    r = _get(web_server, path)
    assert r.status_code == 200, f"{path} → {r.status_code}: {r.text[:200]}"


def test_index_html_served(web_server):
    r = _get(web_server, "/")
    assert r.status_code == 200
    assert "<title>" in r.text
    # Sanity: the static bundle wires up the trade route so the chart
    # toolbar is present in the page source.
    assert 'data-route="trade"' in r.text


# ── /api/variants schema ─────────────────────────────────────────────


def test_variants_payload(web_server):
    r = _get(web_server, "/api/variants").json()
    assert isinstance(r, list)
    assert r, "expected at least one variant on disk"
    for v in r:
        assert {"name", "strategy_type", "db_path"} <= set(v)


# ── /api/today/{variant} schema (the bug where TODAY wouldn't render
#   if hit directly via /#today before selectedVariant was set) ───────


def test_today_payload_shape(web_server):
    variants = _get(web_server, "/api/variants").json()
    name = variants[0]["name"]
    r = _get(web_server, f"/api/today/{name}").json()
    assert r["variant"] == name
    assert "account" in r and isinstance(r["account"], dict)
    assert "positions" in r and isinstance(r["positions"], list)
    assert "snapshots" in r and isinstance(r["snapshots"], list)
    assert "trades" in r and isinstance(r["trades"], list)
    assert "starting_equity" in r           # may be None for view-only variants


# ── Trade chart timezone alignment ──────────────────────────────────
#   The bug: after the ET tz fix in bars.py, daily bars sat at 04:00 UTC.
#   lightweight-charts requires midnight-UTC for daily-resolution
#   detection AND markers must coincide with a bar's time. We split the
#   helper into _to_unix (intraday, ET-localize) and _to_unix_date
#   (daily, midnight-UTC). Tests below pin both behaviors.


def _first_trade_id(base: str, variant: str) -> int | None:
    rows = _get(base, f"/api/variants/{variant}/trades?limit=1").json()
    return rows[0]["trade_id"] if rows else None


def _find_variant_with_trade(base: str, predicate) -> tuple[str, int] | None:
    """Return (variant_name, trade_id) for the first variant matching predicate
    that has at least one trade. predicate(variant_dict) -> bool."""
    for v in _get(base, "/api/variants").json():
        if not predicate(v):
            continue
        tid = _first_trade_id(base, v["name"])
        if tid is not None:
            return v["name"], tid
    return None


def test_daily_chart_bars_at_midnight_utc(web_server):
    """All daily-strategy bars must be at 00:00:00 UTC."""
    pair = _find_variant_with_trade(
        web_server,
        lambda v: v["strategy_type"] in ("mechanical", "llm", "chan_daily", "pead", "pead_llm"),
    )
    if pair is None:
        pytest.skip("no daily-strategy trade on disk")
    variant, tid = pair
    chart = _get(web_server, f"/api/trade/{variant}/{tid}/chart").json()
    if chart.get("timeframe") != "1d" or not chart.get("bars"):
        pytest.skip("first trade had no daily bars")
    for b in chart["bars"]:
        assert _is_midnight_utc(b["time"]), (
            f"daily bar {datetime.utcfromtimestamp(b['time'])} not midnight-UTC "
            f"(was the ET-tz regression reintroduced?)"
        )


def test_daily_fill_midnight_aligned(web_server):
    """Daily-strategy fill markers must be at midnight-UTC.

    The bug we want to catch: ET-localizing the fill time put it at
    ~14:00 UTC instead of 00:00 UTC, so the marker landed between bars
    and lightweight-charts dropped/misplaced it. An exact in-bar-set
    match is too strict (a very recent exit can be past the bar window)
    — the real contract is midnight-UTC alignment.
    """
    pair = _find_variant_with_trade(
        web_server,
        lambda v: v["strategy_type"] in ("mechanical", "llm", "chan_daily", "pead", "pead_llm"),
    )
    if pair is None:
        pytest.skip("no daily-strategy trade on disk")
    variant, tid = pair
    chart = _get(web_server, f"/api/trade/{variant}/{tid}/chart").json()
    if chart.get("timeframe") != "1d" or not chart.get("fills"):
        pytest.skip("trade has no fills")
    for f in chart["fills"]:
        assert _is_midnight_utc(f["time"]), (
            f"daily fill at unix {f['time']} ("
            f"{datetime.utcfromtimestamp(f['time'])}) not midnight-UTC "
            f"— ET localization regression?"
        )


def test_intraday_chart_bars_NOT_midnight_aligned(web_server):
    """Intraday bars are real instants, not dates. They should NOT be
    midnight-UTC — guards against accidentally date-aligning them."""
    pair = _find_variant_with_trade(
        web_server,
        lambda v: v["strategy_type"] in ("chan", "intraday_mechanical"),
    )
    if pair is None:
        pytest.skip("no intraday-strategy trade on disk")
    variant, tid = pair
    chart = _get(web_server, f"/api/trade/{variant}/{tid}/chart").json()
    if not chart.get("bars"):
        pytest.skip("trade has no bars")
    # Most intraday bars span market hours; at least one must be sub-day.
    assert any(not _is_midnight_utc(b["time"]) for b in chart["bars"]), (
        "all intraday bars are midnight-UTC — daily alignment leaked into intraday path"
    )


def test_intraday_consecutive_bar_delta_is_30m(web_server):
    pair = _find_variant_with_trade(
        web_server,
        lambda v: v["strategy_type"] in ("chan",),  # chan/chan_v2 use 30m bars
    )
    if pair is None:
        pytest.skip("no chan trade")
    variant, tid = pair
    chart = _get(web_server, f"/api/trade/{variant}/{tid}/chart").json()
    if len(chart.get("bars", [])) < 2:
        pytest.skip("not enough bars")
    # Find ANY consecutive pair within a session whose delta == 1800s
    deltas = [chart["bars"][i + 1]["time"] - chart["bars"][i]["time"]
              for i in range(min(len(chart["bars"]) - 1, 200))]
    assert 1800 in deltas, f"no 30m delta found in first 200 bars: {sorted(set(deltas))[:10]}"


# ── Risk-page endpoints (added during parity build) ─────────────────


def test_risk_correlation_schema(web_server):
    r = _get(web_server, "/api/risk/correlation").json()
    assert "variants" in r and "matrix" in r and "n_days" in r
    assert isinstance(r["matrix"], list)
    if r["variants"]:
        # Square matrix
        assert all(len(row) == len(r["variants"]) for row in r["matrix"])


def test_risk_sizing_returns_well_formed_even_when_sparse(web_server):
    """The bug: sizing returned 500 if cov was undefined. Should return
    rows=[] with a note instead."""
    r = _get(web_server, "/api/risk/sizing").json()
    assert "rows" in r
    assert isinstance(r["rows"], list)
    if r["rows"]:
        for row in r["rows"]:
            assert {"variant", "current_dollars", "current_weight",
                    "daily_vol_pct", "pct_of_portfolio_swings",
                    "risk_parity_weight", "shift_pp"} <= set(row)


def test_risk_regime_schema(web_server):
    r = _get(web_server, "/api/risk/regime").json()
    if r.get("available") is False:
        pytest.skip("SPY history unavailable")
    assert "rows" in r
    valid_regimes = {"confirmed_uptrend", "uptrend_under_pressure", "market_correction"}
    for row in r["rows"]:
        assert row["regime"] in valid_regimes


def test_risk_patterns_schema(web_server):
    r = _get(web_server, "/api/risk/patterns").json()
    assert "rows" in r
    for row in r["rows"]:
        assert {"variant", "base_pattern", "n_trades", "win_rate",
                "avg_return_pct", "total_return_pct"} <= set(row)


def test_risk_rolling_corr_pair(web_server):
    variants = [v["name"] for v in _get(web_server, "/api/variants").json()]
    if len(variants) < 2:
        pytest.skip("need ≥2 variants")
    r = _get(web_server,
             f"/api/risk/rolling_corr?a={variants[0]}&b={variants[1]}&window=5").json()
    assert r["a"] == variants[0] and r["b"] == variants[1]
    assert "points" in r


# ── Reviews overview (parity-build addition) ────────────────────────


def test_reviews_overview_schema(web_server):
    r = _get(web_server, "/api/reviews/overview").json()
    assert {"cards", "totals", "normalized"} <= set(r)
    for c in r["cards"]:
        assert {"variant", "equity", "daily_pl", "trades_closed",
                "wins", "losses"} <= set(c)
    assert {"equity", "daily_pl", "trades_closed", "wins", "losses"} <= set(r["totals"])


# ── Service status pill ──────────────────────────────────────────────


def test_service_status_returns_bool(web_server):
    r = _get(web_server, "/api/service_status").json()
    assert isinstance(r["running"], bool)
    assert r["label"] in ("RUNNING", "STOPPED")


# ── Log tail ─────────────────────────────────────────────────────────


def test_log_tail_returns_lines(web_server):
    r = _get(web_server, "/api/log/tail?n=5").json()
    assert "exists" in r
    assert "lines" in r
    if r["exists"]:
        assert len(r["lines"]) <= 5


# ── Trade-cycle: clicking a SELL row resolves to the entry ──────────
#   Bug: previously the SELL's reasoning panel showed empty meta (so all
#   criteria rendered red ✗) and the chart only had the SELL marker, no
#   entry. Fix: every overlay redirects to find_entry_id and rebuilds the
#   chart from there.


def _find_closed_buy_sell_pair(base: str) -> Optional[Dict]:
    """Find a (variant, buy_id, sell_id) where the buy was followed by a
    sell of the same symbol and the variant has an overlay extractor.
    Returns dict with keys variant, buy_id, sell_id, symbol, or None."""
    for v in _get(base, "/api/variants").json():
        if v["strategy_type"] not in ("chan", "chan_v2", "chan_daily", "mechanical",
                                       "llm", "intraday_mechanical", "pead", "pead_llm"):
            continue
        rows = _get(base, f"/api/variants/{v['name']}/trades?limit=200").json()
        # Walk back: for each SELL, find a recent BUY of the same symbol.
        sells = [r for r in rows if (r.get("side") or "").lower() == "sell"]
        for s in sells:
            sym = s["symbol"]
            buys_before = [r for r in rows
                           if (r.get("side") or "").lower() == "buy"
                           and r["symbol"] == sym
                           and str(r["timestamp"]) < str(s["timestamp"])]
            if buys_before:
                latest_buy = max(buys_before, key=lambda r: r["timestamp"])
                return {"variant": v["name"], "buy_id": latest_buy["trade_id"],
                        "sell_id": s["trade_id"], "symbol": sym,
                        "strategy_type": v["strategy_type"]}
    return None


def test_clicking_sell_returns_same_chart_as_clicking_buy(web_server):
    """Clicking the SELL trade row should show the entry's reasoning +
    both buy + sell fills (same payload as clicking the BUY)."""
    pair = _find_closed_buy_sell_pair(web_server)
    if pair is None:
        pytest.skip("no buy→sell pair on disk")
    buy_chart = _get(web_server, f"/api/trade/{pair['variant']}/{pair['buy_id']}/chart").json()
    sell_chart = _get(web_server, f"/api/trade/{pair['variant']}/{pair['sell_id']}/chart").json()
    # Symbol identical
    assert buy_chart["symbol"] == sell_chart["symbol"] == pair["symbol"]
    # Both should have at least 2 fills (one buy, one sell)
    sell_sides_buy = {(f["side"] or "").lower() for f in sell_chart.get("fills", [])}
    sell_sides_sell = {(f["side"] or "").lower() for f in sell_chart.get("fills", [])}
    assert "buy" in sell_sides_buy, (
        f"clicking SELL didn't surface BUY fill (regression): {sell_chart.get('fills')}"
    )
    assert "sell" in sell_sides_sell
    # Reasoning headline should reflect the closed trade ("Closed +X.XX%")
    headline = sell_chart["reasoning"]["headline"]
    assert "Closed" in headline or any(
        c.get("passed") for c in sell_chart["reasoning"]["criteria"]
    ), f"SELL-click rendered all-red criteria: {headline} criteria={[c['passed'] for c in sell_chart['reasoning']['criteria']]}"


def test_closed_trade_has_return_metric(web_server):
    """For a closed cycle the reasoning metrics include Return %."""
    pair = _find_closed_buy_sell_pair(web_server)
    if pair is None:
        pytest.skip("no closed cycles")
    chart = _get(web_server, f"/api/trade/{pair['variant']}/{pair['sell_id']}/chart").json()
    labels = [m["label"] for m in chart["reasoning"]["metrics"]]
    assert "Return %" in labels, (
        f"closed-trade metrics missing Return %: {labels}"
    )


# ── Daily review chart payload (parity with TRADE-tab chart) ────────


def test_daily_review_file_strips_interactive_chart_link(web_server):
    """The "[Interactive chart](charts/...)" hyperlink in raw review
    markdown 404s under the API route AND duplicates the embedded
    chart. Server should strip it from `content`."""
    from datetime import date as _date, timedelta as _td
    for offset in range(0, 14):
        d = (_date.today() - _td(days=offset)).isoformat()
        listing = _get(web_server, f"/api/reviews/daily/{d}").json()
        if not (listing.get("exists") and listing.get("by_variant")):
            continue
        for variant, slot in listing["by_variant"].items():
            for fname in (slot.get("closed") or []):
                payload = _get(web_server, f"/api/reviews/daily/{d}/file/{fname}").json()
                content = payload.get("content") or ""
                assert "[Interactive chart](" not in content, (
                    f"{d}/{fname}: legacy interactive-chart link not stripped"
                )
                return
    pytest.skip("no closed daily reviews found in last 14 days")


def test_daily_review_file_includes_chart_payload(web_server):
    """Daily-review file endpoint should include a `chart_payload` so
    the frontend can render the same lightweight-charts visualization
    as the TRADE tab — not the legacy Plotly iframe."""
    # Walk recent dates until we find a directory with reviews.
    from datetime import date as _date, timedelta as _td
    base_url = web_server
    for offset in range(0, 14):
        d = (_date.today() - _td(days=offset)).isoformat()
        listing = _get(base_url, f"/api/reviews/daily/{d}").json()
        if listing.get("exists") and listing.get("by_variant"):
            for variant, slot in listing["by_variant"].items():
                files = (slot.get("closed") or []) + (slot.get("held") or [])
                for fname in files:
                    payload = _get(base_url, f"/api/reviews/daily/{d}/file/{fname}").json()
                    if payload.get("chart_payload"):
                        cp = payload["chart_payload"]
                        assert "bars" in cp and "fills" in cp, (
                            f"chart_payload malformed for {d}/{fname}: {list(cp.keys())}"
                        )
                        # Closed reviews should have ≥2 fills (entry + ≥1 exit)
                        if not fname.endswith("_HELD"):
                            sides = {(f["side"] or "").lower() for f in cp["fills"]}
                            assert "buy" in sides, (
                                f"closed review {fname} missing BUY in fills"
                            )
                        return  # success
    pytest.skip("no daily reviews found in last 14 days")


# ── Path-traversal guards (defenses are silent — pin them down) ─────


def test_review_file_rejects_path_traversal(web_server):
    """The endpoint must refuse `..` segments; if this regresses, an
    attacker on the LAN-exposed deploy could read arbitrary files."""
    today = datetime.utcnow().date().isoformat()
    r = _get(web_server, f"/api/reviews/daily/{today}/file/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code == 404


def test_digest_rejects_path_traversal(web_server):
    r = _get(web_server, "/api/ideas/digest/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code == 404
