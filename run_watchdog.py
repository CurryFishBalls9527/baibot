#!/usr/bin/env python3
"""LaunchAgent entrypoint for the watchdog.

Mirrors run_trading.py — loads .env, then hands off to the watchdog
package's main loop. Keep this file thin: it's what launchd executes.
"""

import os
import sys
from pathlib import Path

from dotenv import dotenv_values


def _load_dotenv_skip_blanks() -> None:
    for key, value in dotenv_values().items():
        if value:
            os.environ.setdefault(key, value)


_load_dotenv_skip_blanks()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradingagents.watchdog.monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
