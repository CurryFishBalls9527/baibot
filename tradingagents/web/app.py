"""FastAPI live UI for trade reasoning + event streaming.

Routes:
  GET  /                                  → static index.html
  GET  /api/variants                      → list of (name, strategy_type, db_path)
  GET  /api/variants/{v}/trades?limit=    → recent trades + their reasoning blob
  GET  /api/trade/{v}/{trade_id}/chart    → ChartPayload for the trade-detail view
  GET  /events/stream                     → SSE: tail results/service_logs/events.jsonl
  GET  /api/health                        → liveness probe

Runs alongside the existing Streamlit dashboard (which we keep intact).
Start with ``python run_web.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load broker API keys + per-variant Alpaca creds from .env, the same way
# the Streamlit dashboard does. Without this, the Today page can't build
# Orchestrator instances and shows empty account data.
load_dotenv()

from tradingagents.storage.database import TradingDatabase
from tradingagents.testing.ab_config import load_experiment

from . import overlays


# ── Static assets ────────────────────────────────────────────────────


_STATIC_DIR = Path(__file__).parent / "static"


# ── Variant discovery ────────────────────────────────────────────────


_DEFAULT_EXPERIMENT = "experiments/paper_launch_v2.yaml"

# Standalone variants not in the experiment YAML (PEAD bridges).
# Mirrors dashboard.multi_variant._STANDALONE_VIEW_DBS.
_STANDALONE_VARIANTS = [
    {"name": "pead",     "strategy_type": "pead",     "db_path": "trading_pead.db"},
    {"name": "pead_llm", "strategy_type": "pead_llm", "db_path": "trading_pead_llm.db"},
]


def _discover_variants() -> List[Dict]:
    """All variants (experiment + standalone) with non-empty data on disk."""
    yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", _DEFAULT_EXPERIMENT)
    found: List[Dict] = []
    seen: set = set()

    if Path(yaml_path).exists():
        exp = load_experiment(yaml_path)
        for v in exp.variants:
            if not v.db_path or not Path(v.db_path).exists():
                continue
            overrides = v.config_overrides or {}
            found.append({
                "name": v.name,
                "strategy_type": overrides.get("strategy_type") or _infer_strategy_type(v.name),
                "db_path": v.db_path,
                "config_overrides": overrides,
            })
            seen.add(v.name)

    for s in _STANDALONE_VARIANTS:
        if s["name"] in seen or not Path(s["db_path"]).exists():
            continue
        found.append({**s, "config_overrides": {}})

    return found


def _infer_strategy_type(name: str) -> str:
    """Fallback when YAML doesn't set strategy_type explicitly.

    The Minervini variants (mechanical, mechanical_v2, llm) don't set
    strategy_type in the YAML — they're the implicit "default". Keep that
    contract by inferring from the variant name.
    """
    if name in ("mechanical", "mechanical_v2"):
        return "mechanical"
    if name == "llm":
        return "llm"
    return name  # chan, chan_v2, chan_daily, intraday_mechanical, pead*


def _variant(name: str) -> Optional[Dict]:
    for v in _discover_variants():
        if v["name"] == name:
            return v
    return None


# ── FastAPI app ──────────────────────────────────────────────────────


app = FastAPI(title="baibot live UI")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "static_dir": str(_STATIC_DIR)}


@app.get("/api/variants")
def list_variants() -> List[dict]:
    return [
        {"name": v["name"], "strategy_type": v["strategy_type"], "db_path": v["db_path"]}
        for v in _discover_variants()
    ]


@app.get("/api/variants/{variant}/trades")
def list_trades(variant: str, limit: int = 50) -> List[dict]:
    v = _variant(variant)
    if v is None:
        raise HTTPException(404, f"variant {variant!r} not found")
    db = TradingDatabase(v["db_path"])

    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp, t.symbol, t.side, t.status,
               t.filled_qty, t.filled_price, t.reasoning,
               s.confidence, s.signal_metadata
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        ORDER BY t.timestamp DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        meta = None
        if d.get("signal_metadata"):
            try:
                meta = json.loads(d["signal_metadata"])
            except Exception:
                meta = None
        d["signal_metadata"] = meta
        out.append(d)
    return out


@app.get("/api/today/{variant}")
def today(variant: str) -> dict:
    """Single-variant operational view — KPIs, equity history, open positions,
    recent trades. Mirrors what ``dashboard/app.py`` shows."""
    v = _variant(variant)
    if v is None:
        raise HTTPException(404, f"variant {variant!r} not found")

    # Build the orchestrator the same way the Streamlit dashboard does so
    # we get account+positions in one shot.
    try:
        from tradingagents.automation.config import build_config
        from tradingagents.automation.orchestrator import Orchestrator
        from tradingagents.testing.ab_config import build_variant_config, load_experiment
        yaml_path = os.getenv("EXPERIMENT_CONFIG_PATH", _DEFAULT_EXPERIMENT)
        base_cfg = build_config()
        # Find this variant's config_overrides from the YAML if it's there
        if Path(yaml_path).exists():
            exp = load_experiment(yaml_path)
            v_obj = next((x for x in exp.variants if x.name == variant), None)
            vconfig = build_variant_config(base_cfg, v_obj) if v_obj else base_cfg
        else:
            vconfig = base_cfg
        orch = Orchestrator(vconfig)
        status = orch.get_status()
    except Exception as exc:
        # Fall back to read-only snapshot+trade data; account/positions are
        # broker-side and unavailable for view-only variants like pead.
        status = {"account": {}, "positions": []}
        orch = None

    db = TradingDatabase(v["db_path"])

    snapshots = db.get_snapshots(days=180) or []
    snapshots = list(reversed(snapshots))  # ascending date

    starting_equity = None
    try:
        starting_equity = db.get_starting_equity()
    except Exception:
        pass
    if starting_equity is None and snapshots:
        starting_equity = float(snapshots[0]["equity"])

    trades = db.get_recent_trades(limit=50) or []

    return {
        "variant": variant,
        "strategy_type": v["strategy_type"],
        "account": status.get("account", {}),
        "positions": status.get("positions", []),
        "snapshots": snapshots,
        "trades": trades,
        "starting_equity": starting_equity,
        "max_positions": int(v["config_overrides"].get("max_positions", 10)),
    }


@app.get("/api/performance/snapshots")
def performance_snapshots(days: int = 180) -> dict:
    """Cross-variant snapshots for equity curves / returns / drawdown."""
    out: Dict[str, list] = {}
    for v in _discover_variants():
        try:
            db = TradingDatabase(v["db_path"])
            rows = db.get_snapshots(days=days) or []
            out[v["name"]] = list(reversed(rows))
        except Exception:
            out[v["name"]] = []
    return {"variants": out}


@app.get("/api/performance/trades")
def performance_trades(limit: int = 500) -> dict:
    """Cross-variant recent trades for the activity histogram."""
    out: Dict[str, list] = {}
    for v in _discover_variants():
        try:
            db = TradingDatabase(v["db_path"])
            out[v["name"]] = db.get_recent_trades(limit=limit) or []
        except Exception:
            out[v["name"]] = []
    return {"variants": out}


# ── PROPOSALS ────────────────────────────────────────────────────────


from pydantic import BaseModel  # noqa: E402  (kept local — only used here)


class ProposalUpdate(BaseModel):
    status: str
    outcome_summary: Optional[str] = None


@app.get("/api/proposals")
def list_proposals(variant: Optional[str] = None) -> dict:
    """All proposals across variants (or one variant if filtered).

    Each proposal carries its source ``_db`` so the client can address
    the write-back endpoint correctly.
    """
    out: List[dict] = []
    for v in _discover_variants():
        if variant and v["name"] != variant:
            continue
        try:
            db = TradingDatabase(v["db_path"])
            rows = db.get_proposals(variant=v["name"]) or []
        except Exception:
            rows = []
        for r in rows:
            r["_db"] = v["name"]
            out.append(r)
    return {"proposals": out}


@app.post("/api/proposals/{variant}/{pid}")
def update_proposal(variant: str, pid: int, body: ProposalUpdate) -> dict:
    v = _variant(variant)
    if v is None:
        raise HTTPException(404, f"variant {variant!r} not found")
    valid = ("open", "accepted", "rejected", "tested")
    if body.status not in valid:
        raise HTTPException(400, f"status must be one of {valid}")
    try:
        db = TradingDatabase(v["db_path"])
        db.update_proposal_status(
            pid,
            status=body.status,
            outcome_summary=body.outcome_summary if body.status == "tested" else None,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"ok": True, "id": pid, "status": body.status}


# ── IDEAS (chat-novelty digest + corpus search) ──────────────────────


def _novelty_paths():
    """Return (digest_dir, corpus_db) — Path objects, may not exist."""
    try:
        from tradingagents.research.chat_novelty_extractor import (
            CORPUS_DB_DEFAULT, NOVELTY_OUT_DEFAULT,
        )
        return Path(NOVELTY_OUT_DEFAULT), Path(CORPUS_DB_DEFAULT)
    except Exception:
        return Path("results/chat_novelty"), Path("research_data/chan_chat_corpus.duckdb")


@app.get("/api/ideas/digests")
def list_digests() -> dict:
    digest_dir, _ = _novelty_paths()
    if not digest_dir.exists():
        return {"digests": []}
    items = []
    for p in sorted(digest_dir.glob("*.md"), reverse=True):
        items.append({
            "name": p.stem,
            "path": str(p),
            "modified": int(p.stat().st_mtime),
            "bytes": p.stat().st_size,
        })
    return {"digests": items}


@app.get("/api/ideas/digest/{name}")
def read_digest(name: str) -> dict:
    digest_dir, _ = _novelty_paths()
    # Reject path traversal attempts: only allow the .md sibling whose stem
    # matches `name` exactly.
    target = digest_dir / f"{name}.md"
    if not target.exists() or target.parent != digest_dir:
        raise HTTPException(404, f"digest {name!r} not found")
    return {"name": name, "content": target.read_text(encoding="utf-8")}


@app.get("/api/ideas/corpus/meta")
def corpus_meta() -> dict:
    _, corpus_db = _novelty_paths()
    if not corpus_db.exists():
        return {"available": False}
    try:
        import duckdb
        con = duckdb.connect(str(corpus_db), read_only=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM chan_chat_messages").fetchone()[0]
            ts_min, ts_max = con.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM chan_chat_messages"
            ).fetchone()
            chats = con.execute(
                "SELECT chat_id, chat_title, COUNT(*) AS n FROM chan_chat_messages "
                "GROUP BY chat_id, chat_title ORDER BY n DESC LIMIT 10"
            ).fetchall()
        finally:
            con.close()
        return {
            "available": True,
            "n_messages": n,
            "min_ts": str(ts_min) if ts_min else None,
            "max_ts": str(ts_max) if ts_max else None,
            "chats": [{"chat_id": c[0], "chat_title": c[1], "n": c[2]} for c in chats],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@app.get("/api/ideas/corpus/search")
def corpus_search(
    q: str = "", days: int = 30, author: str = "", limit: int = 200,
) -> dict:
    _, corpus_db = _novelty_paths()
    if not corpus_db.exists():
        return {"messages": []}
    from datetime import datetime, timedelta, timezone
    since = datetime.now(timezone.utc) - timedelta(days=int(days))
    try:
        import duckdb
        con = duckdb.connect(str(corpus_db), read_only=True)
        try:
            where = ["timestamp >= ?"]
            params: list = [since]
            if q:
                where.append("text ILIKE ?"); params.append(f"%{q}%")
            if author:
                where.append("(author_username ILIKE ? OR author_display ILIKE ?)")
                params.extend([f"%{author}%", f"%{author}%"])
            sql = (
                "SELECT timestamp, chat_title, author_username, author_display, text "
                "FROM chan_chat_messages WHERE "
                + " AND ".join(where)
                + " ORDER BY timestamp DESC LIMIT ?"
            )
            params.append(int(limit))
            rows = con.execute(sql, params).fetchall()
            cols = ["timestamp", "chat_title", "author_username", "author_display", "text"]
        finally:
            con.close()
        msgs = [dict(zip(cols, r)) for r in rows]
        for m in msgs:
            if m.get("timestamp") is not None:
                m["timestamp"] = str(m["timestamp"])
        return {"messages": msgs}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── REVIEWS (daily + weekly markdown post-mortems) ───────────────────


_KNOWN_VARIANT_PREFIXES = (
    "mechanical_v2", "chan_daily", "chan_v2", "mechanical", "chan",
    "llm", "intraday_mechanical", "pead_llm", "pead",
)


def _variant_of_filename(stem: str) -> str:
    for v in _KNOWN_VARIANT_PREFIXES:
        if stem.startswith(f"{v}_"):
            return v
    return stem.split("_", 1)[0]


@app.get("/api/reviews/daily/{review_date}")
def daily_reviews(review_date: str, include_dry_run: bool = False) -> dict:
    """List all daily-review markdown files for a given date.

    Returns:
      ``{ "directory": str, "exists": bool, "by_variant": {variant: {"closed":[], "held":[], "summary": str|None}} }``
    """
    base = Path("results/daily_reviews") / review_date
    if not (base.exists() and any(base.iterdir())):
        if include_dry_run:
            base = Path("results/daily_reviews_dryrun") / review_date
    if not base.exists():
        return {"directory": str(base), "exists": False, "by_variant": {}}

    md_files = sorted(p for p in base.glob("*.md") if not p.name.endswith("_summary.md"))
    by_variant: Dict[str, dict] = {}
    for p in md_files:
        v = _variant_of_filename(p.stem)
        slot = by_variant.setdefault(v, {"closed": [], "held": [], "summary": None})
        if p.stem.endswith("_HELD"):
            slot["held"].append(p.stem)
        else:
            slot["closed"].append(p.stem)
    for v, slot in by_variant.items():
        s = base / f"{v}_summary.md"
        if s.exists():
            slot["summary"] = s.stem
    return {"directory": str(base), "exists": True, "by_variant": by_variant}


@app.get("/api/reviews/daily/{review_date}/file/{name}")
def daily_review_file(review_date: str, name: str, include_dry_run: bool = False) -> dict:
    base = Path("results/daily_reviews") / review_date
    if not base.exists() and include_dry_run:
        base = Path("results/daily_reviews_dryrun") / review_date
    target = base / f"{name}.md"
    if not target.exists() or target.parent != base:
        raise HTTPException(404, f"file {name!r} not found")
    chart_html = None
    chart_path = base / "charts" / f"{name}.html"
    if chart_path.exists():
        chart_html = chart_path.read_text(encoding="utf-8")
    return {
        "name": name,
        "content": target.read_text(encoding="utf-8"),
        "chart_html": chart_html,
    }


def _iso_week(d: str) -> str:
    """ISO-week tag for a YYYY-MM-DD string: "2026-W18"."""
    from datetime import date as _date
    dt = _date.fromisoformat(d)
    y, w, _x = dt.isocalendar()
    return f"{y}-W{w:02d}"


@app.get("/api/reviews/weekly/{ref_date}")
def weekly_reviews(ref_date: str, include_dry_run: bool = False) -> dict:
    iso = _iso_week(ref_date)
    base = Path("results/weekly_reviews") / iso
    if not (base.exists() and any(base.iterdir())):
        if include_dry_run:
            base = Path("results/weekly_reviews_dryrun") / iso
    if not base.exists():
        return {"iso_week": iso, "directory": str(base), "exists": False, "variants": []}
    files = sorted(p for p in base.glob("*.md") if p.name != "_index.md")
    return {
        "iso_week": iso,
        "directory": str(base),
        "exists": True,
        "variants": [p.stem for p in files],
        "has_index": (base / "_index.md").exists(),
    }


@app.get("/api/reviews/weekly/{ref_date}/file/{name}")
def weekly_review_file(ref_date: str, name: str, include_dry_run: bool = False) -> dict:
    iso = _iso_week(ref_date)
    base = Path("results/weekly_reviews") / iso
    if not base.exists() and include_dry_run:
        base = Path("results/weekly_reviews_dryrun") / iso
    target = base / f"{name}.md"
    if not target.exists() or target.parent != base:
        raise HTTPException(404, f"file {name!r} not found")
    return {
        "name": name,
        "iso_week": iso,
        "content": target.read_text(encoding="utf-8"),
    }


# ── PORTFOLIO RISK (correlation matrix + outcomes + LLM cost) ────────


@app.get("/api/risk/correlation")
def risk_correlation(days: int = 60) -> dict:
    """Daily-return correlation matrix across variants.

    Returns ``{ "variants": [...], "matrix": [[...]], "n_days": N }`` —
    correlation of daily equity returns over the last ``days`` calendar
    days. Variants with too few overlapping days are dropped.
    """
    import math
    snaps_by_v: Dict[str, Dict[str, float]] = {}  # variant → {date: equity}
    for v in _discover_variants():
        try:
            db = TradingDatabase(v["db_path"])
            rows = db.get_snapshots(days=days) or []
        except Exception:
            rows = []
        snaps_by_v[v["name"]] = {r["date"]: float(r["equity"]) for r in rows}

    # Returns per (variant, date)
    rets_by_v: Dict[str, Dict[str, float]] = {}
    for variant, eq in snaps_by_v.items():
        dates = sorted(eq.keys())
        if len(dates) < 2:
            rets_by_v[variant] = {}; continue
        out: Dict[str, float] = {}
        for i in range(1, len(dates)):
            prev_eq = eq[dates[i - 1]]
            cur_eq  = eq[dates[i]]
            if prev_eq > 0:
                out[dates[i]] = (cur_eq / prev_eq) - 1.0
        rets_by_v[variant] = out

    variants = [v for v, m in rets_by_v.items() if len(m) >= 5]
    if len(variants) < 2:
        return {"variants": variants, "matrix": [], "n_days": 0}

    def _corr(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
        common = sorted(set(a) & set(b))
        if len(common) < 5:
            return None
        xs = [a[d] for d in common]; ys = [b[d] for d in common]
        n = len(common)
        mx = sum(xs) / n; my = sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if dx == 0 or dy == 0:
            return None
        return num / (dx * dy)

    matrix = [[_corr(rets_by_v[a], rets_by_v[b]) for b in variants] for a in variants]
    n_days = max(len(rets_by_v[v]) for v in variants)
    return {"variants": variants, "matrix": matrix, "n_days": n_days}


@app.get("/api/risk/outcomes")
def risk_outcomes(days: int = 90) -> dict:
    """Cross-variant trade outcomes for the diagnostics table."""
    from datetime import date as _date, timedelta as _td
    end = _date.today().isoformat()
    start = (_date.today() - _td(days=days)).isoformat()
    rows: List[dict] = []
    for v in _discover_variants():
        try:
            db = TradingDatabase(v["db_path"])
            outcomes = db.get_trade_outcomes_in_range(start, end) or []
        except Exception:
            outcomes = []
        for o in outcomes:
            o["_variant"] = v["name"]
            rows.append(o)
    return {"start": start, "end": end, "outcomes": rows}


@app.get("/api/risk/llm_cost")
def risk_llm_cost(days: int = 60) -> dict:
    """Daily LLM cost rollup from earnings_llm_decisions."""
    db_path = Path("research_data/earnings_data.duckdb")
    if not db_path.exists():
        return {"available": False, "rows": []}
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                """
                SELECT CAST(analyzed_at AS DATE) AS day,
                       SUM(cost_estimate_usd) AS daily_cost,
                       COUNT(*) AS n_analyses,
                       SUM(CASE WHEN llm_decision IS NULL THEN 1 ELSE 0 END) AS n_errors
                FROM earnings_llm_decisions
                WHERE analyzed_at IS NOT NULL
                  AND analyzed_at >= NOW() - INTERVAL ? DAY
                GROUP BY day
                ORDER BY day
                """,
                [int(days)],
            ).fetchall()
        finally:
            con.close()
        return {
            "available": True,
            "rows": [
                {"day": str(r[0]), "daily_cost": float(r[1] or 0),
                 "n_analyses": int(r[2] or 0), "n_errors": int(r[3] or 0)}
                for r in rows
            ],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "rows": []}


@app.get("/api/trade/{variant}/{trade_id}/chart")
def trade_chart(variant: str, trade_id: int) -> dict:
    v = _variant(variant)
    if v is None:
        raise HTTPException(404, f"variant {variant!r} not found")

    db = TradingDatabase(v["db_path"])
    variant_config = {
        "name": v["name"],
        "strategy_type": v["strategy_type"],
        **v["config_overrides"],
    }

    payload = overlays.build_for(
        strategy_type=v["strategy_type"],
        db=db,
        trade_id=trade_id,
        variant_config=variant_config,
    )
    if payload is None:
        raise HTTPException(
            501,
            f"strategy_type {v['strategy_type']!r} not yet supported by web UI",
        )
    return payload


# ── SSE: tail events.jsonl ───────────────────────────────────────────


_EVENTS_PATH = Path(os.environ.get("BAIBOT_RESULTS_DIR", "results")) / "service_logs" / "events.jsonl"


async def _tail_events() -> AsyncIterator[str]:
    """Tail-follow events.jsonl. Yields one SSE-formatted message per line.

    The watchdog already writes structured JSON events here (see
    ``tradingagents/automation/events.py``) — categories like
    ``order_reject``, ``position_stranded``, ``drift_detected``,
    ``reconciler_tick``. Browsers subscribe via EventSource.
    """
    path = _EVENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()

    # On open: send the last 50 events so the page can populate without
    # waiting for the next event to fire.
    try:
        with path.open("r", encoding="utf-8", errors="replace") as h:
            tail = h.readlines()[-50:]
        for line in tail:
            line = line.strip()
            if line:
                yield f"data: {line}\n\n"
    except Exception:
        pass

    # Then follow the file for new lines.
    f = path.open("r", encoding="utf-8", errors="replace")
    try:
        f.seek(0, 2)  # EOF
        while True:
            line = f.readline()
            if not line:
                # Heartbeat every 15s so proxies don't drop the connection,
                # and so the client can detect server-side liveness.
                await asyncio.sleep(1.0)
                # Send a comment line every ~15 polls (~15s).
                # Comment frames are ignored by EventSource clients.
                yield ": ping\n\n"
                continue
            line = line.strip()
            if line:
                yield f"data: {line}\n\n"
    finally:
        try:
            f.close()
        except Exception:
            pass


@app.get("/events/stream")
async def events_stream() -> StreamingResponse:
    return StreamingResponse(
        _tail_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Per-variant trade stream ─────────────────────────────────────────
#
# The orchestrator(s) write to per-variant SQLite files. When a new trade
# row is committed, the file's mtime advances. We poll the mtime and
# push any new trade rows over SSE — no orchestrator coupling needed.


def _trade_row(db: TradingDatabase, trade_id: int) -> Optional[dict]:
    """Same shape as ``GET /api/variants/{v}/trades`` rows."""
    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp, t.symbol, t.side, t.status,
               t.filled_qty, t.filled_price, t.reasoning,
               s.confidence, s.signal_metadata
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        WHERE t.id = ?
        """,
        [trade_id],
    ).fetchall()
    if not rows:
        return None
    d = dict(rows[0])
    if d.get("signal_metadata"):
        try:
            d["signal_metadata"] = json.loads(d["signal_metadata"])
        except Exception:
            d["signal_metadata"] = None
    return d


def _trades_after(db: TradingDatabase, since_id: int, limit: int = 50) -> List[dict]:
    rows = db.conn.execute(
        """
        SELECT t.id AS trade_id, t.timestamp, t.symbol, t.side, t.status,
               t.filled_qty, t.filled_price, t.reasoning,
               s.confidence, s.signal_metadata
        FROM trades t
        LEFT JOIN signals s ON s.id = t.signal_id
        WHERE t.id > ?
        ORDER BY t.id ASC
        LIMIT ?
        """,
        [since_id, limit],
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("signal_metadata"):
            try:
                d["signal_metadata"] = json.loads(d["signal_metadata"])
            except Exception:
                d["signal_metadata"] = None
        out.append(d)
    return out


def _latest_trade_id(db: TradingDatabase) -> int:
    row = db.conn.execute("SELECT MAX(id) AS mx FROM trades").fetchone()
    return int(row["mx"]) if row and row["mx"] is not None else 0


async def _variant_event_stream(v: dict, request: Request) -> AsyncIterator[str]:
    """Watch the variant's SQLite mtime; push any newly-inserted trades.

    Each yielded message is one SSE frame. The client receives:
      { "type": "snapshot", "last_trade_id": N }
      { "type": "trade",    "trade": {...} }
      { "type": "ping",     "ts": "..." }      (every ~10s, keep-alive)
    """
    db_path = Path(v["db_path"])
    db = TradingDatabase(v["db_path"])
    last_id = _latest_trade_id(db)
    last_mtime = db_path.stat().st_mtime if db_path.exists() else 0

    yield f"data: {json.dumps({'type': 'snapshot', 'last_trade_id': last_id})}\n\n"

    poll_secs = 1.5
    ping_every = 10.0
    elapsed_since_ping = 0.0

    while True:
        if await request.is_disconnected():
            break

        try:
            current_mtime = db_path.stat().st_mtime
        except FileNotFoundError:
            await asyncio.sleep(poll_secs)
            continue

        if current_mtime != last_mtime:
            last_mtime = current_mtime
            try:
                # Reuse the existing connection — TradingDatabase wraps
                # sqlite3 in WAL mode so reads see committed writes from
                # the orchestrator process without needing reconnects.
                new_trades = _trades_after(db, last_id)
            except Exception as exc:
                # Log to the events stream-style format and keep going.
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                new_trades = []
            for t in new_trades:
                last_id = max(last_id, int(t["trade_id"]))
                yield f"data: {json.dumps({'type': 'trade', 'trade': t}, default=str)}\n\n"
                elapsed_since_ping = 0.0  # an event counts as a heartbeat

        await asyncio.sleep(poll_secs)
        elapsed_since_ping += poll_secs

        if elapsed_since_ping >= ping_every:
            yield ": ping\n\n"
            elapsed_since_ping = 0.0


@app.get("/api/variants/{variant}/stream")
async def variant_stream(variant: str, request: Request) -> StreamingResponse:
    v = _variant(variant)
    if v is None:
        raise HTTPException(404, f"variant {variant!r} not found")
    return StreamingResponse(
        _variant_event_stream(v, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Static frontend ──────────────────────────────────────────────────
# Mount last so the API routes take precedence over the static handler.


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
