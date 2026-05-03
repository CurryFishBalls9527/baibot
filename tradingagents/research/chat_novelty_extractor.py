"""Weekly LLM novelty extractor for the Telegram chat corpus.

Reads last N days of messages from `chan_chat_messages`, filters out
obvious noise via simple heuristics, and feeds what's left + the
project's MEMORY.md (a compact list of everything already tried) to
gpt-5.4 with a prompt that asks ONE question:

  "Are there any concrete trading ideas / setups / filters discussed
  here that are NOT already in the user's tried/tested list?"

Output: a markdown digest at results/chat_novelty/<isoweek>.md plus a
DuckDB row in `chat_novelty_runs` (audit trail of cost + decisions).

The Memory file is the negative filter — kills 80% of false positives
("oh, another volume gate proposal" → not novel). Without it the LLM
has no way to know what's been tested.

## Why gpt-5.4 not gpt-5.4-pro
- Need a model that handles bilingual Chinese/English context well
- Need long-ish context for the memory file + a week of messages
- Don't need maximum reasoning depth — this is extraction, not judgment
- gpt-5.4 = ~$0.30/run; pro = ~$1.50/run. Cost not material either way
  but pro adds latency + reasoning-block complications we already hit
  with PEAD-LLM.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


CORPUS_DB_DEFAULT = "research_data/chat_corpus.duckdb"
NOVELTY_OUT_DEFAULT = "results/chat_novelty"
MEMORY_INDEX_DEFAULT = (
    "/Users/myu/.claude/projects/-Users-myu-code-baibot/memory/MEMORY.md"
)

# Heuristics for dropping pure noise. Tuned for Chinese/English mixed
# Chan groups. Aggressive — better to drop interesting messages than
# pay LLM to read junk.
_DROP_PATTERNS = [
    re.compile(r"^[\s。.,，!?]+$"),          # punctuation only
    re.compile(r"^[a-zA-Z]{1,3}$"),         # single short English word (lol, ok, bro)
    re.compile(r"^\d+(\.\d+)?$"),           # bare number (price quote)
    re.compile(r"^\$?[A-Z]{1,5}$"),         # bare ticker only
    re.compile(r"^(?:啊|哦|嗯|哈|呵|嘿|嗨|对|是|不|好|行|可以|没事|加油|牛|强|赞|顶|哈哈+|呵呵+)\W*$"),
    re.compile(r"^[一-鿿]{1,2}$"),  # 1-2 chinese chars (likely interjection)
]
_MIN_LEN = 8  # very short messages are usually noise


@dataclass
class NoveltyRunResult:
    iso_week: str
    window_start: datetime
    window_end: datetime
    n_messages_raw: int
    n_messages_after_filter: int
    digest_md: str
    digest_path: str
    model: str
    duration_s: float
    cost_estimate_usd: float
    error: Optional[str] = None


def _is_noise(text: str) -> bool:
    if not text or len(text.strip()) < _MIN_LEN:
        return True
    s = text.strip()
    return any(p.search(s) for p in _DROP_PATTERNS)


def _load_recent_messages(
    db_path: str, since: datetime, until: datetime,
) -> list[dict]:
    """Load chat messages in (since, until], returning a list of dicts.
    Read-only, tolerant of missing DB / table."""
    if not Path(db_path).exists():
        return []
    try:
        con = duckdb.connect(db_path, read_only=True)
        try:
            rows = con.execute(
                """
                SELECT chat_id, chat_title, message_id, timestamp,
                       author_username, author_display, text
                FROM chan_chat_messages
                WHERE timestamp > ? AND timestamp <= ?
                ORDER BY timestamp
                """,
                [since, until],
            ).fetchall()
        finally:
            con.close()
    except Exception as exc:
        logger.warning("chat corpus load failed: %s", exc)
        return []
    return [
        dict(zip(
            ["chat_id", "chat_title", "message_id", "timestamp",
             "author_username", "author_display", "text"],
            r,
        ))
        for r in rows
    ]


def _filter_messages(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not _is_noise(r["text"])]


def _format_messages_for_llm(rows: list[dict], max_chars: int = 60_000) -> str:
    """Compact representation. Each line: [time] @author: text.
    Truncate to max_chars to keep prompt cost bounded."""
    lines = []
    total = 0
    for r in rows:
        author = r.get("author_username") or r.get("author_display") or "anon"
        ts = r["timestamp"].strftime("%m-%d %H:%M") if r.get("timestamp") else "??"
        line = f"[{ts}] @{author}: {r['text'][:600]}"
        line_len = len(line) + 1
        if total + line_len > max_chars:
            lines.append(f"... [truncated; {len(rows) - len(lines)} more messages dropped]")
            break
        lines.append(line)
        total += line_len
    return "\n".join(lines)


def _load_memory_index(memory_path: str) -> str:
    """Load the MEMORY.md project index. Returns empty string if missing."""
    p = Path(memory_path)
    if not p.exists():
        logger.warning("MEMORY.md not found at %s — extractor running without negative filter", memory_path)
        return ""
    return p.read_text()


_PROMPT_TEMPLATE = """You are a quantitative research assistant for a trader who has \
already done extensive research and wants to know if a Telegram \
discussion group surfaced any GENUINELY NEW IDEAS in the past week.

The user's research history is below as a memory index. Each line \
summarizes a strategy, filter, or idea that has ALREADY been:
  - tested (passed → already shipped, or failed → null finding)
  - ruled out for a documented reason
  - or actively run live

Your job: read the chat messages and identify ONLY ideas that meet \
ALL of these criteria:
1. Concrete trading approach, setup, filter, or signal (not generic theory)
2. NOT obviously already in the memory index (use semantic match, not literal)
3. Could plausibly be backtested in a few hours
4. Not just a position-talking, after-the-fact rationalization, or \
   late "buy XYZ now" signal

STRICTLY EXCLUDE (do not output, do not even mention):
- Generic Chan T1/T2/T3 BSP discussions (the user has tested these deeply)
- Volume-based gates of any flavor (3 separate null memories)
- Standard technical indicators (MACD/RSI/stochastic confirmations)
- Pyramid / partial-exit / scaling ideas (universe-survivorship null pattern)
- Signal-style "buy XXX now" calls without methodology
- Rehashes of academic strategies the user already runs
- Pure sentiment/regime takes without an actionable rule

If NO genuinely novel ideas: say so directly with one short sentence. \
False positives (pretending an old idea is new) are MUCH worse than \
false negatives.

Format:
## Novel ideas this week ({iso_week}, {start} → {end})

For each (1-5 max, ranked by novelty × researchability):
### {{idea_name}}
- **Proposed by**: @username (or "multiple" if discussed by several)
- **Idea**: 2-3 sentences describing the concrete approach
- **Why potentially novel**: 1 sentence — what's not in the memory list
- **Quick test sketch**: 1 sentence — what backtest would validate this

If nothing novel surfaced this week, output exactly:
## No novel ideas this week ({iso_week})
The chat had {n_messages} substantive messages but everything discussed \
was either: [brief reason — e.g., "already-tested approaches", "pure \
signal-style calls", "off-topic"].

────────────────────────────────────────
USER'S RESEARCH MEMORY INDEX (already-tried list):

{memory_index}

────────────────────────────────────────
CHAT MESSAGES from {start} to {end} ({n_messages} after noise-filter, \
from {n_raw} raw):

{messages}
"""


# Rough cost estimate (matching pead_llm_analyzer's rates). Updated by
# the novelty extractor based on actual token counts when available.
_RATES = {
    "gpt-5.4":     {"in": 0.010, "out": 0.030},
    "gpt-5.4-pro": {"in": 0.015, "out": 0.060},
    "gpt-5-mini":  {"in": 0.0008, "out": 0.003},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = _RATES.get(model, _RATES["gpt-5.4"])
    return (prompt_tokens * rate["in"] + completion_tokens * rate["out"]) / 1000.0


def run_extraction(
    *,
    end: Optional[datetime] = None,
    days: int = 7,
    db_path: str = CORPUS_DB_DEFAULT,
    out_dir: str = NOVELTY_OUT_DEFAULT,
    memory_path: str = MEMORY_INDEX_DEFAULT,
    model: str = "gpt-5.4",
) -> NoveltyRunResult:
    """Run one extraction pass. Always returns a result; on LLM failure
    sets `error` and writes a failure-marker to disk."""
    import time

    if end is None:
        end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    iso_year, iso_week, _ = end.isocalendar()
    iso_label = f"{iso_year}-W{iso_week:02d}"

    raw_rows = _load_recent_messages(db_path, start, end)
    filtered = _filter_messages(raw_rows)
    logger.warning(
        "novelty extractor | %s | window %s → %s | %d raw → %d after filter",
        iso_label, start.date(), end.date(), len(raw_rows), len(filtered),
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    digest_path = str(Path(out_dir) / f"{iso_label}.md")

    if not filtered:
        text = (
            f"# Chat novelty digest — {iso_label}\n\n"
            f"## No data\n\n"
            f"Window {start.date()} → {end.date()}: {len(raw_rows)} raw messages, "
            "0 after noise-filter. Listener may not be running, or "
            "configured chats had no substantive activity.\n"
        )
        Path(digest_path).write_text(text)
        return NoveltyRunResult(
            iso_week=iso_label, window_start=start, window_end=end,
            n_messages_raw=len(raw_rows), n_messages_after_filter=0,
            digest_md=text, digest_path=digest_path,
            model=model, duration_s=0.0, cost_estimate_usd=0.0,
        )

    memory_index = _load_memory_index(memory_path)
    prompt = _PROMPT_TEMPLATE.format(
        iso_week=iso_label,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        n_messages=len(filtered),
        n_raw=len(raw_rows),
        memory_index=memory_index or "(memory index not available)",
        messages=_format_messages_for_llm(filtered),
    )

    started = time.time()
    error_text: Optional[str] = None
    text: str = ""
    try:
        from tradingagents.llm_clients.factory import create_llm_client
        client = create_llm_client(provider="openai", model=model)
        llm = client.get_llm()
        result = llm.invoke(prompt)
        # Coerce to text — same pattern as pead_llm_analyzer for safety
        raw = result.content
        if isinstance(raw, list):
            text = "\n".join(
                b.get("text", "") for b in raw
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(raw)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        logger.warning("LLM extraction failed: %s", exc)
        text = (
            f"# Chat novelty digest — {iso_label}\n\n"
            f"## Extraction failed\n\n"
            f"Error: `{error_text}`\n\n"
            f"Window {start.date()} → {end.date()} had {len(filtered)} "
            "substantive messages. Re-run after fixing the underlying issue.\n"
        )
    duration = time.time() - started

    digest_full = (
        f"# Chat novelty digest — {iso_label}\n\n"
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"model `{model}` · {duration:.1f}s · "
        f"{len(filtered)} of {len(raw_rows)} messages analysed_\n\n"
        + text
    )
    Path(digest_path).write_text(digest_full)

    # Approximate cost — token estimate from len/4 heuristic
    approx_in = len(prompt) // 4
    approx_out = len(text) // 4
    cost = _estimate_cost(model, approx_in, approx_out)

    return NoveltyRunResult(
        iso_week=iso_label, window_start=start, window_end=end,
        n_messages_raw=len(raw_rows), n_messages_after_filter=len(filtered),
        digest_md=digest_full, digest_path=digest_path,
        model=model, duration_s=duration, cost_estimate_usd=cost,
        error=error_text,
    )
