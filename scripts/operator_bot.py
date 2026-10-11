"""A stand-in for a human operator, for the recorded handoff demo.

It does what a person at the operator console would: watches the console for
an open intervention, takes control, attaches to the *same* live browser over
CDP, does the manual work, and hands control back with a note. Every request
it makes goes through the same console endpoints a person's clicks would.

    python scripts/operator_bot.py --console http://127.0.0.1:8770 --policy dismiss|approve

  dismiss  click a button by its visible name in the main frame (default "Remind Me Later"),
           then resolve "done"
  approve  resolve "approve" (for approval_required interventions)
"""

import argparse
import time

import requests
from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--console", default="http://127.0.0.1:8770")
    parser.add_argument("--policy", choices=["dismiss", "approve", "deny"], default="dismiss")
    parser.add_argument("--button", default="Remind Me Later")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    deadline = time.time() + args.timeout
    handled = 0
    while time.time() < deadline:
        try:
            state = requests.get(f"{args.console}/api/state", timeout=2).json()
        except requests.RequestException:
            if handled:
                return 0          # the run finished and the console went away
            time.sleep(0.5)
            continue
        open_ = state.get("open")
        if not open_:
            time.sleep(0.5)
            continue
        print(f"[operator] intervention {open_['id']}: {open_['reason']} — {open_['message']}")
        time.sleep(1.5)           # a person reads the screen first
        requests.post(f"{args.console}/take", timeout=5)
        print("[operator] took control")
        if args.policy == "dismiss":
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(open_["cdp_endpoint"])
                page = browser.contexts[0].pages[0]
                frame = next((f for f in page.frames if f.name == "main"), page.main_frame)
                frame.get_by_role("button", name=args.button).click()
                frame.wait_for_load_state()
                print(f"[operator] clicked {args.button!r} in the live session")
                time.sleep(1)     # let the capture binding deliver the click
            requests.post(f"{args.console}/resolve", json={
                "action": "done", "note": f"Dismissed an unexpected '{args.button}' prompt."}, timeout=5)
        elif args.policy in open_["options"]:
            requests.post(f"{args.console}/resolve", json={
                "action": args.policy, "note": f"Reviewed the screen; {args.policy}d."}, timeout=5)
        else:
            # Asked for something this bot can't do (e.g. a stuck agent): a
            # person would look and decide; the bot gives up cleanly instead.
            fallback = "abort" if "abort" in open_["options"] else "deny"
            requests.post(f"{args.console}/resolve", json={
                "action": fallback, "note": f"Bot cannot handle a {open_['reason']} intervention."}, timeout=5)
            print(f"[operator] {open_['reason']} is beyond this bot; resolved {fallback}")
        print("[operator] handed control back")
        handled += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
