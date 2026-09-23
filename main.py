"""Rote — Record Once Then Execute.

Entry point. Inference runs on the GX10; `ai` below is the ready-to-use client
(see ai.py for the full API: ask, stream, look, conversation, with_tools, embed).
"""

import sys

from ai import AI, AIError


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


def main() -> int:
    try:
        ai = setup_ai()
    except AIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # One conversation, reused every turn — this is what remembers the history.
    # (ai.ask() is the stateless version: it keeps nothing between calls.)
    chat = ai.conversation()

    print(f"{ai.model} ready on {ai.host}")
    print("Ctrl-D or /exit to quit, /history to review, /reset to clear.\n")

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not prompt:
            continue
        if prompt in ("/exit", "/quit"):
            return 0
        if prompt == "/reset":
            chat.reset()
            print("(history cleared)\n")
            continue
        if prompt == "/history":
            for message in chat.messages:
                print(f"  {message['role']:>9}: {message['content']}")
            print()
            continue

        try:
            # stream_say prints as the model generates, and appends both the
            # prompt and the reply to chat.messages.
            for piece in chat.stream_say(prompt):
                print(piece, end="", flush=True)
            print("\n")
        except AIError as e:
            print(f"error: {e}\n", file=sys.stderr)

    # Other things `ai` can do, for when Rote grows past a prompt loop:
    #   ai.look("screenshot.png", "What's on screen?")   # vision
    #   ai.with_tools("...", tools=[my_function])        # model calls your code


if __name__ == "__main__":
    raise SystemExit(main())
