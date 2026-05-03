"""Entrypoint for the live web UI.

Runs the FastAPI app at ``tradingagents/web/app.py`` under uvicorn.
Sits *alongside* the existing Streamlit dashboard — does not replace it.

  python run_web.py            # http://127.0.0.1:8765
  python run_web.py --port 9000
"""

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--reload", action="store_true",
                        help="hot-reload on code changes (dev)")
    args = parser.parse_args()

    uvicorn.run(
        "tradingagents.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
