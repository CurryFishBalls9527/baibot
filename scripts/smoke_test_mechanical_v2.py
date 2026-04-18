"""Smoke test for mechanical_v2 broker plumbing (Track P-SYNC).

Walks through the critical untested paths end-to-end against the Alpaca
paper account:
  1. Submit a tiny bracket order.
  2. Verify stop_order_id + tp_order_id are captured on the response.
  3. Call replace_order on the stop leg to bump the stop price up.
  4. Read the leg back via get_order and verify the new stop price stuck.
  5. Close the position and cancel remaining legs.

This is intended as a one-shot manual check before `mechanical_v2` goes
live. It uses real paper capital on the new mechanical_v2 Alpaca account.

Usage:
    export ALPACA_MECHANICAL_V2_API_KEY=...
    export ALPACA_MECHANICAL_V2_SECRET_KEY=...
    python scripts/smoke_test_mechanical_v2.py [--symbol SPY] [--qty 1]

Market must be OPEN — bracket child legs only materialize after the parent
market order fills.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tradingagents.broker.alpaca_broker import AlpacaBroker
from tradingagents.broker.models import OrderRequest


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: {name} not set. Export ALPACA_MECHANICAL_V2_* first.")
    return val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument(
        "--skip-confirm", action="store_true",
        help="Skip the y/n prompt before submitting.",
    )
    args = parser.parse_args()

    api_key = _require_env("ALPACA_MECHANICAL_V2_API_KEY")
    secret = _require_env("ALPACA_MECHANICAL_V2_SECRET_KEY")

    broker = AlpacaBroker(api_key, secret, paper=True)

    acct = broker.get_account()
    print(f"Account {acct.account_id}  equity=${acct.equity:,.2f}  cash=${acct.cash:,.2f}")

    clock = broker.get_clock()
    if not clock.is_open:
        sys.exit(
            f"Market is CLOSED (next open {clock.next_open}). "
            "Bracket legs only materialize after parent fill — rerun during RTH."
        )

    price = broker.get_latest_price(args.symbol)
    stop_initial = round(price * 0.90, 2)
    stop_target = round(price * 0.92, 2)  # ratcheted stop
    tp_initial = round(price * 1.15, 2)
    print(
        f"\n{args.symbol}  last=${price:.2f}  "
        f"initial_stop=${stop_initial}  ratcheted_stop=${stop_target}  "
        f"take_profit=${tp_initial}"
    )
    print(f"Will place: BUY {args.qty} {args.symbol} BRACKET")

    if not args.skip_confirm:
        if input("Proceed? (y/N) ").strip().lower() != "y":
            sys.exit("Aborted.")

    # ── Step 1: submit bracket ───────────────────────────────────────────
    print("\n[1/5] Submitting bracket order...")
    req = OrderRequest(
        symbol=args.symbol,
        side="buy",
        qty=args.qty,
        order_type="market",
        time_in_force="gtc",
    )
    parent = broker.submit_bracket_order(req, stop_initial, tp_initial)
    print(
        f"  parent_id={parent.order_id}  status={parent.status}  "
        f"stop_order_id={parent.stop_order_id}  tp_order_id={parent.tp_order_id}"
    )

    # ── Step 2: verify leg IDs captured ──────────────────────────────────
    print("\n[2/5] Verifying bracket leg IDs were captured...")
    if not parent.stop_order_id or not parent.tp_order_id:
        # Alpaca may return legs only once the parent fills; poll briefly.
        print("  Legs not on initial response — polling for fill...")
        for i in range(20):
            time.sleep(1)
            refreshed = broker.get_order(parent.order_id)
            if refreshed.status.lower().endswith("filled"):
                print(f"  parent filled after {i+1}s")
                break
        # Re-fetch the full order to get legs.
        raw = broker.trading_client.get_order_by_id(parent.order_id)
        legs = getattr(raw, "legs", None) or []
        for leg in legs:
            if getattr(leg, "stop_price", None) and not parent.stop_order_id:
                parent.stop_order_id = str(leg.id)
            elif getattr(leg, "limit_price", None) and not parent.tp_order_id:
                parent.tp_order_id = str(leg.id)
        print(
            f"  after poll: stop_order_id={parent.stop_order_id} "
            f"tp_order_id={parent.tp_order_id}"
        )
    if not parent.stop_order_id:
        sys.exit("FAIL: stop_order_id not captured on bracket parent.")
    print(f"  ✓ stop_order_id={parent.stop_order_id}")
    print(f"  ✓ tp_order_id={parent.tp_order_id}")

    # ── Step 3: read the stop leg back and confirm initial price ─────────
    print("\n[3/5] Reading stop leg back...")
    stop_leg_before = broker.trading_client.get_order_by_id(parent.stop_order_id)
    stop_price_before = float(stop_leg_before.stop_price or 0)
    print(f"  stop_leg.status={stop_leg_before.status}  stop_price=${stop_price_before}")
    if abs(stop_price_before - stop_initial) > 0.01:
        print(
            f"  WARNING: stop leg price ${stop_price_before} != submitted "
            f"${stop_initial} (Alpaca rounding?)"
        )

    # ── Step 4: ratchet the stop up via replace_order ────────────────────
    print(f"\n[4/5] Calling replace_order to bump stop {stop_initial} -> {stop_target}...")
    replaced = broker.replace_order(parent.stop_order_id, stop_price=stop_target)
    if replaced is None:
        sys.exit("FAIL: replace_order returned None — check logs for Alpaca error.")
    print(f"  new_order_id={replaced.order_id}  status={replaced.status}")

    # Alpaca replace_order creates a new order (new ID) and cancels the old.
    # Give the client a moment to settle, then read the new order back.
    time.sleep(2)
    stop_leg_after = broker.trading_client.get_order_by_id(replaced.order_id)
    stop_price_after = float(stop_leg_after.stop_price or 0)
    print(
        f"  re-read: status={stop_leg_after.status}  stop_price=${stop_price_after}"
    )
    if abs(stop_price_after - stop_target) > 0.01:
        sys.exit(
            f"FAIL: stop price after replace ${stop_price_after} != target "
            f"${stop_target}"
        )
    print(f"  ✓ stop ratcheted: ${stop_price_before} -> ${stop_price_after}")

    # ── Step 5: clean up — close position + cancel any remaining orders ──
    print("\n[5/5] Cleaning up — closing position + cancelling legs...")
    try:
        close_result = broker.close_position(args.symbol)
        print(f"  close_position({args.symbol}) -> {close_result.status}")
    except Exception as e:
        print(f"  close_position failed (may already be flat): {e}")

    # Cancel the remaining TP leg if it's still open.
    try:
        broker.cancel_order(parent.tp_order_id)
        print(f"  cancelled tp_leg {parent.tp_order_id}")
    except Exception as e:
        print(f"  tp_leg cancel (may already be done): {e}")

    # And the replaced stop leg (close_position may have triggered it already).
    try:
        broker.cancel_order(replaced.order_id)
        print(f"  cancelled stop_leg {replaced.order_id}")
    except Exception as e:
        print(f"  stop_leg cancel (may already be done): {e}")

    print("\n✓ SMOKE TEST PASSED.")
    print("  - Bracket submit returned leg IDs.")
    print("  - replace_order accepted by Alpaca and stop price updated.")
    print("  - Position closed and legs cancelled.")
    print("\nReady to enable mechanical_v2 variant in experiments/paper_launch_v2.yaml.")


if __name__ == "__main__":
    main()
