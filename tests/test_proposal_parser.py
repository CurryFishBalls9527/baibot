"""Tests for the weekly-review proposal parser + proposals table helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.automation.weekly_review import _parse_and_store_proposals
from tradingagents.storage.database import TradingDatabase


@pytest.fixture
def db(tmp_path: Path) -> TradingDatabase:
    return TradingDatabase(str(tmp_path / "t.db"))


SAMPLE_REVIEW = """# Weekly Review — chan_v2 — 2026-W17

## Scope
Analyzed 4 closed trades this week.

## Performance
Outperformed SPY by 5pp.

## Proposals (Open-Ended)

### Proposal: Tighten entry filter
- **What:** Raise RS threshold from 15% to 10% top.
- **Why:** Top-RS trades had +3% higher avg return.
- **How to validate:** Backtest the same 2023-2025 window with new threshold; pass if avg return unchanged and trade count > 20.
- **Risk:** May lose 30% of trade count; easy rollback via config.

### Proposal: Skip Monday entries
- **What:** Disable new entries on Monday; exits still fire.
- **Why:** Monday WR 0%, other days 65%.
- **How to validate:** Paper-test 2 weeks vs baseline; pass if Monday-excluded total return >= baseline.
- **Risk:** Tiny sample; may be noise.

## Other ignored section
"""


class TestParser:
    def test_parses_two_proposals(self, db):
        n = _parse_and_store_proposals(
            db, variant="chan_v2", iso_week="2026-W17", review_md=SAMPLE_REVIEW,
        )
        assert n == 2
        rows = db.get_proposals(variant="chan_v2")
        titles = [r["title"] for r in rows]
        assert "Tighten entry filter" in titles
        assert "Skip Monday entries" in titles

    def test_populates_all_fields(self, db):
        _parse_and_store_proposals(
            db, variant="chan_v2", iso_week="2026-W17", review_md=SAMPLE_REVIEW,
        )
        rows = db.get_proposals(variant="chan_v2", status="open")
        p = next(r for r in rows if r["title"] == "Tighten entry filter")
        assert "RS threshold" in (p["what"] or "")
        assert "higher avg return" in (p["why"] or "")
        assert "Backtest" in (p["how_to_validate"] or "")
        assert "rollback" in (p["risk"] or "")

    def test_empty_review_yields_zero(self, db):
        assert _parse_and_store_proposals(db, variant="x", iso_week="W1", review_md="") == 0

    def test_review_without_proposals_section_yields_zero(self, db):
        md = "# Review\n## Performance\nDid fine."
        assert _parse_and_store_proposals(db, variant="x", iso_week="W1", review_md=md) == 0

    def test_tolerates_partial_bullet_format(self, db):
        md = """## Proposals (Open-Ended)
### Proposal: Fix stop width
- **What:** Widen stop 3% -> 4%.
- No other fields provided
"""
        n = _parse_and_store_proposals(db, variant="x", iso_week="W1", review_md=md)
        assert n == 1
        p = db.get_proposals(variant="x")[0]
        assert p["what"] and "Widen stop" in p["what"]
        assert p["why"] is None
        assert p["how_to_validate"] is None

    def test_status_update(self, db):
        _parse_and_store_proposals(
            db, variant="chan_v2", iso_week="2026-W17", review_md=SAMPLE_REVIEW,
        )
        rows = db.get_proposals(variant="chan_v2")
        pid = rows[0]["id"]
        db.update_proposal_status(pid, "accepted")
        updated = [r for r in db.get_proposals(variant="chan_v2") if r["id"] == pid][0]
        assert updated["status"] == "accepted"
        assert updated["status_updated_at"] is not None

    def test_status_update_with_outcome(self, db):
        _parse_and_store_proposals(
            db, variant="chan_v2", iso_week="2026-W17", review_md=SAMPLE_REVIEW,
        )
        pid = db.get_proposals(variant="chan_v2")[0]["id"]
        db.update_proposal_status(pid, "tested", outcome_summary="backtest showed -5pp")
        row = [r for r in db.get_proposals() if r["id"] == pid][0]
        assert row["status"] == "tested"
        assert "backtest showed" in row["outcome_summary"]
