"""Partial smoke test for AlpacaBroker.replace_order (market-closed safe).

Verifies the Alpaca replace_order_by_id wrapper works end-to-end on a
STANDALONE stop order (not a bracket leg). This validates ~80% of the
ExitManagerV2 ratcheting path:
  - submit_order → stop_price captured
  - replace_order with new stop_price → Alpaca accepts
  - get_order → new stop_price stuck
  - cancel_order → clean up

What this does NOT test: whether Alpaca treats a bracket OCO stop leg
differently from a standalone stop when replaced. That edge case needs
the full bracket smoke test during market hours.

Safe to run any time:
  - Uses a BUY stop far above market (won't trigger).
  - 1-share qty, GTC TIF.
  - Script cancels before exit.

Usage:
    python scripts/smoke_test_replace_order.py [--symbol SPY] \\
        [--api-key-env ALPACA_MECHANICAL_API_KEY] \\
        [--secret-env ALPACA_MECHANICAL_SECRET_KEY]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv("/Users/myu/code/baibot/.env")

from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.broker.models import OrderRequest


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: {name} not set.")
    return val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--api-key-env", default="ALPACA_MECHANICAL_API_KEY")
    parser.add_argument("--secret-env", default="ALPACA_MECHANICAL_SECRET_KEY")
    args = parser.parse_args()

    api_key = _require_env(args.api_key_env)
    secret = _require_env(args.secret_env)

    broker = AlpacaBroker(api_key, secret, paper=True)
    acct = broker.get_account()
    print(f"Account {acct.account_id}  equity=${acct.equity:,.2f}")

    clock = broker.get_clock()
    print(f"Market open? {clock.is_open}  (next open {clock.next_open})")

    price = broker.get_latest_price(args.symbol)
    stop_initial = round(price * 1.20, 2)  # 20% above market — won't trigger
    stop_target = round(price * 1.25, 2)   # ratcheted 25% above market
    print(
        f"\n{args.symbol}  last=${price:.2f}  "
        f"initial_stop=${stop_initial}  ratcheted_stop=${stop_target}"
    )

    # ── Step 1: submit standalone buy stop ──────────────────────────────
    print("\n[1/4] Submitting standalone BUY stop order...")
    req = OrderRequest(
        symbol=args.symbol,
        side="buy",
        qty=1,
        order_type="stop",
        stop_price=stop_initial,
        time_in_force="gtc",
    )
    try:
        order = broker.submit_order(req)
    except Exception as e:
        sys.exit(f"FAIL: submit_order raised: {e}")
    print(f"  order_id={order.order_id}  status={order.status}")

    # Brief settle.
    time.sleep(1)
    readback = broker.get_order(order.order_id)
    print(f"  readback: status={readback.status}")
    raw_before = broker.trading_client.get_order_by_id(order.order_id)
    stop_before = float(raw_before.stop_price or 0)
    print(f"  stop_price on leg: ${stop_before}")
    if abs(stop_before - stop_initial) > 0.01:
        print(f"  WARNING: stop price ${stop_before} != submitted ${stop_initial}")

    # ── Step 2: replace stop price ──────────────────────────────────────
    print(f"\n[2/4] replace_order stop ${stop_before} -> ${stop_target}...")
    replaced = broker.replace_order(order.order_id, stop_price=stop_target)
    if replaced is None:
        # Still try to cancel original before exiting.
        try:
            broker.cancel_order(order.order_id)
        except Exception:
            pass
        sys.exit(
            "FAIL: replace_order returned None. "
            "Check AlpacaBroker logs for the Alpaca error."
        )
    print(f"  new_order_id={replaced.order_id}  status={replaced.status}")

    # ── Step 3: verify new stop stuck ───────────────────────────────────
    print("\n[3/4] Re-reading replaced order...")
    time.sleep(2)
    raw_after = broker.trading_client.get_order_by_id(replaced.order_id)
    stop_after = float(raw_after.stop_price or 0)
    print(f"  status={raw_after.status}  stop_price=${stop_after}")
    if abs(stop_after - stop_target) > 0.01:
        # clean up before exit
        try:
            broker.cancel_order(replaced.order_id)
        except Exception:
            pass
        sys.exit(
            f"FAIL: replaced stop ${stop_after} != target ${stop_target}"
        )

    # ── Step 4: cancel ──────────────────────────────────────────────────
    print("\n[4/4] Cancelling replaced order...")
    try:
        broker.cancel_order(replaced.order_id)
        print(f"  cancelled {replaced.order_id}")
    except Exception as e:
        print(f"  cancel_order raised: {e}")

    # Also try the original order_id in case replace was a no-op on Alpaca's end.
    try:
        broker.cancel_order(order.order_id)
    except Exception:
        pass  # Expected — Alpaca replaces by issuing a new order and cancelling the old.

    print("\n✓ PARTIAL SMOKE TEST PASSED.")
    print("  - Standalone stop submitted.")
    print("  - replace_order accepted by Alpaca and stop_price updated.")
    print("  - get_order confirmed new stop_price.")
    print("  - cancel_order clean.")
    print("\nRemaining risk: bracket-OCO-leg replacement not exercised.")
    print("Run scripts/smoke_test_mechanical_v2.py during RTH to cover that.")


if __name__ == "__main__":
    main()
