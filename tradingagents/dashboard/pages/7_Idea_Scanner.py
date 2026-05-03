"""Idea Scanner — surfaces the latest weekly chat-novelty digest plus
the searchable underlying chat corpus.

Two views:
  1. Latest digest (markdown rendered) + history navigation
  2. Raw corpus search by ticker / author / freetext / date range

This is research substrate. NOT a signal feed. Use it to spot ideas
the user hasn't already tried, manually decide which (if any) to
backtest.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tradingagents.research.chat_novelty_extractor import (
    CORPUS_DB_DEFAULT, NOVELTY_OUT_DEFAULT,
)

st.set_page_config(page_title="Idea Scanner", layout="wide")
st.title("Idea Scanner")
st.caption(
    "Weekly novelty digest from chat groups + searchable raw corpus. "
    "**Research substrate, not signals.** Use it to find ideas you "
    "haven't already tested. Manually decide what (if anything) to backtest."
)

tab_digest, tab_corpus = st.tabs(["Weekly digest", "Raw corpus search"])

# ─── Tab 1: weekly digest ─────────────────────────────────────────────

with tab_digest:
    out_dir = Path(NOVELTY_OUT_DEFAULT)
    if not out_dir.exists():
        st.info(
            "No digests yet. The extractor runs Saturday 10:00 CDT. "
            "Force a run with: `launchctl kickstart gui/$UID/com.baibot.chat_novelty_extractor` "
            "(after one-time setup — see scripts/telegram_auth_setup.py)."
        )
    else:
        digests = sorted(out_dir.glob("*.md"), reverse=True)
        if not digests:
            st.info("No digests in results/chat_novelty/ yet.")
        else:
            choice = st.selectbox(
                "Pick a week",
                options=[d.stem for d in digests],
                index=0,
                help="Most recent first.",
            )
            chosen = next(d for d in digests if d.stem == choice)
            st.markdown(f"_File: `{chosen}` · "
                       f"modified {datetime.fromtimestamp(chosen.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}_")
            st.divider()
            st.markdown(chosen.read_text())

# ─── Tab 2: raw corpus search ─────────────────────────────────────────

with tab_corpus:
    db_path = Path(CORPUS_DB_DEFAULT)
    if not db_path.exists():
        st.info(
            f"No chat corpus yet at `{db_path}`. The Telegram listener "
            "(`com.baibot.telegram_listener`) writes here. Verify it's running:\n"
            "`launchctl list | grep telegram_listener`"
        )
    else:
        @st.cache_data(ttl=60, show_spinner=False)
        def _load_corpus_meta(db_path: str):
            con = duckdb.connect(db_path, read_only=True)
            try:
                n_total = con.execute(
                    "SELECT COUNT(*) FROM chan_chat_messages"
                ).fetchone()[0]
                date_range = con.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM chan_chat_messages"
                ).fetchone()
                chats = con.execute(
                    "SELECT chat_id, chat_title, COUNT(*) AS n FROM chan_chat_messages "
                    "GROUP BY chat_id, chat_title ORDER BY n DESC"
                ).fetchall()
            finally:
                con.close()
            return n_total, date_range, chats

        n_total, (ts_min, ts_max), chats = _load_corpus_meta(str(db_path))
        st.caption(
            f"Corpus: **{n_total:,}** messages · "
            + (f"{ts_min.date()} → {ts_max.date()}" if ts_min and ts_max else "(empty)")
        )
        if chats:
            st.caption(
                "By chat: "
                + " · ".join(f"`{title or cid}`={n:,}" for cid, title, n in chats[:5])
            )

        if n_total == 0:
            st.info("Corpus has no messages yet. Listener may not be configured "
                    "(check CHAN_CHAT_IDS in .env) or just hasn't seen any "
                    "messages in the watched chats yet.")
        else:
            # Search controls
            c1, c2, c3 = st.columns([2, 1, 1])
            q = c1.text_input("Search text (ticker, keyword, name…)", "")
            days_back = c2.slider("Days back", 1, 365, 30)
            limit = c3.number_input("Max rows", 50, 5000, 500, step=50)
            author_filter = st.text_input("Author username filter (optional)", "")

            since = datetime.now(timezone.utc) - timedelta(days=days_back)
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                where = ["timestamp >= ?"]
                params: list = [since]
                if q:
                    where.append("text ILIKE ?")
                    params.append(f"%{q}%")
                if author_filter:
                    where.append("(author_username ILIKE ? OR author_display ILIKE ?)")
                    params.append(f"%{author_filter}%")
                    params.append(f"%{author_filter}%")
                sql = (
                    "SELECT timestamp, chat_title, author_username, "
                    "author_display, text "
                    "FROM chan_chat_messages WHERE "
                    + " AND ".join(where)
                    + " ORDER BY timestamp DESC LIMIT ?"
                )
                params.append(int(limit))
                df = con.execute(sql, params).df()
            finally:
                con.close()

            if df.empty:
                st.info("No messages match those filters in the time window.")
            else:
                st.caption(f"{len(df)} matches (most recent first)")
                # Combine username/display for compact view
                df["author"] = df["author_username"].fillna("") + " (" + df["author_display"].fillna("") + ")"
                df_show = df[["timestamp", "chat_title", "author", "text"]]
                df_show.columns = ["When", "Chat", "Author", "Text"]
                st.dataframe(df_show, use_container_width=True, hide_index=True,
                             column_config={
                                 "When": st.column_config.DatetimeColumn(format="MM-DD HH:mm"),
                                 "Text": st.column_config.TextColumn(width="large"),
                             })

# ─── Footer ───────────────────────────────────────────────────────────

st.divider()
with st.expander("Setup status & operations"):
    st.markdown(
        f"""
        **Components:**
        - `com.baibot.telegram_listener` — persistent Telethon listener.
          Reads chats configured via `CHAN_CHAT_IDS` in `.env`. Writes to
          `{CORPUS_DB_DEFAULT}`.
        - `com.baibot.chat_novelty_extractor` — Saturday 10:00 CDT job.
          Reads last 7 days, calls gpt-5.4 with the project memory file
          as negative filter, writes digest to `{NOVELTY_OUT_DEFAULT}/`.

        **One-time setup** (before either job works):
        1. Create app at https://my.telegram.org/apps → get `api_id` + `api_hash`
        2. Add to `.env`: `TELEGRAM_API_ID=...` `TELEGRAM_API_HASH=...`
        3. Run `.venv/bin/python scripts/telegram_auth_setup.py`
           — interactive (phone + SMS code, optional 2FA)
        4. Run `.venv/bin/python scripts/telegram_list_dialogs.py`
           — find your Chan group's `chat_id`
        5. Add to `.env`: `CHAN_CHAT_IDS=<comma-separated negative ids>`
        6. Install plists:
           ```
           cp scripts/launchd/com.baibot.telegram_listener.plist ~/Library/LaunchAgents/
           cp scripts/launchd/com.baibot.chat_novelty_extractor.plist ~/Library/LaunchAgents/
           launchctl load ~/Library/LaunchAgents/com.baibot.telegram_listener.plist
           launchctl load ~/Library/LaunchAgents/com.baibot.chat_novelty_extractor.plist
           ```

        **Force a digest run** (after corpus has data):
        `launchctl kickstart gui/$UID/com.baibot.chat_novelty_extractor`
        Or manually: `.venv/bin/python scripts/run_chat_novelty_extractor.py`

        **Privacy / security**: the session file at
        `results/.telegram_session.session` is your authenticated Telegram
        session. Treat it as a password. Never commit, never copy off the
        local machine.
        """
    )
