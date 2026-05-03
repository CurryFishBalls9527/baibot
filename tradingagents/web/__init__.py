"""Live web UI for trade reasoning + event streaming.

A FastAPI sidecar that runs alongside (not replacing) the Streamlit
dashboard. The contract: every strategy emits chart payloads in a
unified overlay vocabulary, so a single frontend renders them all.

See ``overlays/`` for per-strategy extractors and ``app.py`` for the
HTTP/SSE surface.
"""
