"""Live Log — tail the scheduler's automation_service log."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _log_path() -> Path:
    return _repo_root() / "results" / "service_logs" / "automation_service.out.log"


def _tail_lines(path: Path, n: int) -> List[str]:
    if not path.exists():
        return ["(log file not found)"]
    try:
        with path.open("r", encoding="utf-8", errors="replace") as h:
            return h.readlines()[-n:]
    except Exception as exc:
        return [f"(error reading log: {exc})"]


def _log_html(lines: List[str]) -> str:
    safe = "".join(lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    colored = []
    for raw in safe.splitlines():
        if "ERROR" in raw or "Traceback" in raw or " error " in raw.lower():
            colored.append(f'<span style="color:#ef4444;">{raw}</span>')
        else:
            colored.append(raw)
    body = "\n".join(colored)
    return (
        '<div id="livelog" style="background:#05080d;border:1px solid #1f2937;'
        'border-radius:4px;padding:0.6rem 0.75rem;height:75vh;overflow-y:auto;'
        'font-family:Menlo,monospace;font-size:0.74rem;color:#9ca3af;'
        f'white-space:pre-wrap;line-height:1.4;">{body}</div>'
        '<script>var el=document.getElementById("livelog"); '
        'if(el) el.scrollTop=el.scrollHeight;</script>'
    )


st.set_page_config(page_title="Live Log", layout="wide")
st.title("Live Log")

c1, c2, c3 = st.columns([1.2, 1.2, 6])
with c1:
    n = st.selectbox(
        "Tail length",
        [100, 200, 500, 1000, 2000],
        index=1,
        label_visibility="collapsed",
    )
with c2:
    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()
with c3:
    path = _log_path()
    mtime_str = ""
    if path.exists():
        mtime_str = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
    st.caption(
        f"`{path}` · last write {mtime_str}" if mtime_str else f"`{path}` · missing"
    )

lines = _tail_lines(_log_path(), n=int(n))
st.markdown(_log_html(lines), unsafe_allow_html=True)
