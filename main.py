"""Rote — Record Once Then Execute.

Entry point. Type a goal; Rote replays a saved workflow if it has one, and
otherwise plans the steps with the model, runs them, and records what worked so
the next time is a replay.

Inference runs on the GX10 (see ai.py). The layers underneath are in src/:
bridge orchestrates, analyze plans, controller executes, workflows records.
"""

import sys

from ai import AI, AIError
from src import controller as controllers
from src import workflows
from src.bridge import Bridge, Outcome

HELP = """\
  <goal>            do it — replays a saved workflow, or plans a new one
  /ask <question>   just ask the model, no actions taken
  /workflows        list what has been recorded
  /forget <name>    delete a recorded workflow
  /controllers      show which controllers work on this machine
  /help             this
  /exit             quit\
"""


def setup_ai() -> AI:
    """Connect to the GX10 and get the model resident before any real work.

    Raises AIError if the box is unreachable or the model is missing, so
    startup fails loudly rather than at the first inference call.
    """
    ai = AI()

    if not ai.is_up():
        raise AIError(
            f"GX10 unreachable at {ai.host}. "
            "Check `tailscale status` — gx10-3d18 should not be offline."
        )
    if not ai.has_model():
        raise AIError(
            f"{ai.model} is not on the GX10. Run: python ai.py --pull"
        )

    # Loading 19.6 GB takes ~30s; do it now so the first request is instant.
    ai.warm()
    return ai


def report(outcome: Outcome) -> None:
    """Print what happened, step by step. Notes were already shown live."""
    if outcome.error:
        print(f"  {outcome.summary()}")
        return

    for result in outcome.results:
        mark = "ok  " if result.ok else "FAIL"
        print(f"  [{mark}] {result.step}")
        for line in str(result.output).splitlines():
            print(f"         {line}")

    print(f"  {outcome.summary()}")


def handle_goal(bridge: Bridge, goal: str, max_corrections: int = 3) -> None:
    """Run a goal, and on failure offer the human-intervention path.

    Capped: if the model cannot get there in a few tries it is not going to,
    and each attempt costs a model call and the operator's attention.
    """
    outcome = bridge.handle(goal)
    report(outcome)

    for attempt in range(max_corrections):
        if outcome.ok:
            return
        try:
            correction = input("\n  what should it have done? (blank to give up) ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not correction:
            return
        outcome = bridge.correct(outcome, correction)
        report(outcome)

    if not outcome.ok:
        print(f"  giving up after {max_corrections} correction(s)")


def main() -> int:
    try:
        ai = setup_ai()
    except AIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    bridge = Bridge(ai=ai, notify=lambda message: print(f"  · {message}"))
    saved = workflows.all_workflows(bridge.workflow_dir)

    print(f"Rote — {ai.model} on {ai.host}")
    print(f"{len(saved)} recorded workflow(s); controllers: {', '.join(bridge.controllers)}")
    print("/help for commands, /exit to quit.\n")

    while True:
        try:
            line = input("rote> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        command, _, argument = line.partition(" ")
        argument = argument.strip()

        if command in ("/exit", "/quit"):
            break

        elif command == "/help":
            print(HELP)

        elif command == "/ask":
            if not argument:
                print("  usage: /ask <question>")
                continue
            try:
                for piece in ai.stream(argument):
                    print(piece, end="", flush=True)
                print()
            except AIError as e:
                print(f"  error: {e}", file=sys.stderr)

        elif command == "/workflows":
            found = workflows.all_workflows(bridge.workflow_dir)
            if not found:
                print("  nothing recorded yet")
            for workflow in found:
                print(f"  {workflow.goal}")
                print(f"    {len(workflow.plan)} step(s) via {workflow.plan.controller}, "
                      f"replayed {workflow.runs}x  [{workflow.path.name}]")

        elif command == "/forget":
            if not argument:
                print("  usage: /forget <name or goal>")
                continue
            target = workflows.find(argument, bridge.workflow_dir)
            if not target or not target.path:
                print(f"  no workflow matching {argument!r}")
                continue
            target.path.unlink()
            print(f"  forgot {target.path.name}")

        elif command == "/controllers":
            for name, ok, reason in controllers.status():
                print(f"  {'ok ' if ok else '-- '} {name}")
                if reason:
                    for detail in reason.splitlines():
                        print(f"        {detail}")

        elif command.startswith("/"):
            print(f"  unknown command {command}. /help for the list.")

        else:
            handle_goal(bridge, line)

    for controller in bridge.controllers.values():
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
