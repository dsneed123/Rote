"""Client for the Qwen3-VL model hosted on the ASUS GX10.

The GX10 runs Ollama; this module talks to it over HTTP (Tailscale by default),
so nothing needs to be installed on the GX10 beyond Ollama itself.

Use it from the project:

    from ai import AI

    ai = AI()
    print(ai.ask("Summarise what this app does."))
    print(ai.ask("What's in this screenshot?", images=["shot.png"]))

    chat = ai.conversation(system="You are terse.")
    chat.say("Remember the number 7.")
    chat.say("What number?")          # history is kept

Or from the shell:

    python ai.py "hello"
    python ai.py -i shot.png "describe this"
    python ai.py --chat
    python ai.py --check
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import sys
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import requests

# --- Configuration ----------------------------------------------------------
# Override any of these with environment variables (a .env loader is not
# assumed; export them or set them in your process).

HOST = os.environ.get("ROTE_AI_HOST", "http://gx10-3d18:11434").rstrip("/")
MODEL = os.environ.get("ROTE_AI_MODEL", "qwen3-vl:30b")
TIMEOUT = float(os.environ.get("ROTE_AI_TIMEOUT", "600"))
# How long Ollama keeps the model in GPU memory after a request. Loading
# 19 GB takes ~30s, so keeping it resident makes follow-up calls instant.
KEEP_ALIVE = os.environ.get("ROTE_AI_KEEP_ALIVE", "30m")

ImageSource = "str | Path | bytes"


class AIError(RuntimeError):
    """Raised when the GX10 is unreachable or returns an error."""


def _encode_image(image: Any) -> str:
    """Turn a path, raw bytes, or an already-base64 string into base64 text."""
    if isinstance(image, bytes):
        return base64.b64encode(image).decode()
    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode()
    if isinstance(image, str):
        path = Path(image)
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        # Assume the caller already base64-encoded it (e.g. from a browser).
        if image.startswith("data:"):
            return image.split(",", 1)[1]
        return image
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _message(role: str, content: str, images: Sequence[Any] | None = None) -> dict:
    msg: dict[str, Any] = {"role": role, "content": content}
    if images:
        msg["images"] = [_encode_image(i) for i in images]
    return msg


_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def as_tool(fn: Callable) -> dict:
    """Build a tool schema from a Python function's signature and docstring.

    The first line of the docstring becomes the tool description, so write it
    for the model to read. Annotate every parameter; defaults mark it optional.

        def get_weather(city: str, units: str = "celsius") -> str:
            '''Look up the current weather for a city.'''
    """
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n\n")[0].strip() or fn.__name__
    hints = typing.get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = hints.get(name, str)
        # Unwrap Optional[X] / X | None down to X.
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if args:
            annotation = args[0]
        properties[name] = {"type": _JSON_TYPES.get(annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


@dataclass
class AI:
    """A thin, dependency-light wrapper around the GX10's Ollama server."""

    host: str = HOST
    model: str = MODEL
    timeout: float = TIMEOUT
    keep_alive: str = KEEP_ALIVE
    # Passed to Ollama as `options`: temperature, num_ctx, top_p, seed, ...
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.host = self.host.rstrip("/")
        self._session = requests.Session()

    # --- Core calls ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        stream: bool = False,
        tools: list[dict] | None = None,
        think: bool | None = None,
        format: Any = None,
        **options: Any,
    ) -> dict | Iterator[dict]:
        """Raw chat call. Returns the final message dict, or a chunk iterator.

        `format` accepts "json" or a JSON Schema dict to force structured output.
        """
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {**self.options, **options},
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        if format is not None:
            payload["format"] = format

        if stream:
            return self._stream("/api/chat", payload)
        return self._post("/api/chat", payload)

    def ask(
        self,
        prompt: str,
        *,
        images: Sequence[Any] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """One-shot question in, plain text out."""
        messages = []
        if system:
            messages.append(_message("system", system))
        messages.append(_message("user", prompt, images))
        result = self.chat(messages, stream=False, **kwargs)
        return result["message"]["content"]

    def stream(
        self,
        prompt: str,
        *,
        images: Sequence[Any] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Same as `ask`, but yields text fragments as they are generated."""
        messages = []
        if system:
            messages.append(_message("system", system))
        messages.append(_message("user", prompt, images))
        for chunk in self.chat(messages, stream=True, **kwargs):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece

    def look(self, image: Any, prompt: str = "Describe this image.", **kwargs: Any) -> str:
        """Convenience wrapper for the vision path."""
        return self.ask(prompt, images=[image], **kwargs)

    def with_tools(
        self,
        prompt: str,
        tools: Sequence[Callable],
        *,
        images: Sequence[Any] | None = None,
        system: str | None = None,
        max_rounds: int = 6,
        on_call: Callable[[str, dict, Any], None] | None = None,
        **kwargs: Any,
    ) -> str:
        """Ask a question, letting the model call your Python functions.

        `tools` are plain functions with annotated parameters and a docstring;
        their schemas are derived automatically. The loop runs until the model
        answers in prose or `max_rounds` is hit. `on_call`, if given, is invoked
        as (name, arguments, result) after each tool runs — useful for logging.
        """
        registry = {fn.__name__: fn for fn in tools}
        schemas = [as_tool(fn) for fn in tools]

        messages = []
        if system:
            messages.append(_message("system", system))
        messages.append(_message("user", prompt, images))

        for _ in range(max_rounds):
            result = self.chat(messages, stream=False, tools=schemas, **kwargs)
            reply = result["message"]
            messages.append(reply)

            calls = reply.get("tool_calls")
            if not calls:
                return reply.get("content", "")

            for call in calls:
                fn_spec = call["function"]
                name = fn_spec["name"]
                arguments = fn_spec.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                fn = registry.get(name)
                if fn is None:
                    output: Any = f"No such tool: {name}"
                else:
                    try:
                        output = fn(**arguments)
                    except Exception as e:  # report failures back to the model
                        output = f"{type(e).__name__}: {e}"

                if on_call:
                    on_call(name, arguments, output)
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": output if isinstance(output, str) else json.dumps(output),
                })

        raise AIError(f"tool loop did not finish within {max_rounds} rounds")

    def embed(
        self,
        text: str | Sequence[str],
        *,
        model: str = "nomic-embed-text",
    ) -> list[list[float]]:
        """Embeddings, for retrieval. Uses a small dedicated model, not the VL one."""
        payload = {"model": model, "input": text, "keep_alive": self.keep_alive}
        return self._post("/api/embed", payload)["embeddings"]

    def conversation(self, system: str | None = None) -> "Conversation":
        return Conversation(self, system=system)

    # --- Housekeeping -------------------------------------------------------

    def is_up(self) -> bool:
        try:
            self._session.get(f"{self.host}/api/version", timeout=5).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def models(self) -> list[str]:
        data = self._session.get(f"{self.host}/api/tags", timeout=15).json()
        return [m["name"] for m in data.get("models", [])]

    def loaded(self) -> list[str]:
        """Models currently resident in GPU memory."""
        data = self._session.get(f"{self.host}/api/ps", timeout=15).json()
        return [m["name"] for m in data.get("models", [])]

    def has_model(self, model: str | None = None) -> bool:
        target = model or self.model
        names = self.models()
        # Ollama reports "qwen3-vl:30b"; accept a bare name as ":latest".
        return target in names or f"{target}:latest" in names

    def pull(self, model: str | None = None, *, progress: bool = True) -> None:
        """Download a model onto the GX10. Safe to call when it already exists."""
        target = model or self.model
        for chunk in self._stream("/api/pull", {"model": target}, timeout=None):
            if "error" in chunk:
                raise AIError(chunk["error"])
            if progress:
                done, total = chunk.get("completed"), chunk.get("total")
                if done and total:
                    pct = 100 * done / total
                    print(f"\r{chunk['status']}: {pct:5.1f}%", end="", file=sys.stderr)
                else:
                    print(f"\r{chunk.get('status', '')}", end="", file=sys.stderr)
        if progress:
            print(file=sys.stderr)

    def warm(self, model: str | None = None) -> None:
        """Load the model into GPU memory so the next real call is fast."""
        self._post("/api/chat", {
            "model": model or self.model,
            "messages": [],
            "keep_alive": self.keep_alive,
            "stream": False,
        })

    def openai_base_url(self) -> str:
        """Ollama speaks the OpenAI API too, if you'd rather use that SDK.

            from openai import OpenAI
            client = OpenAI(base_url=AI().openai_base_url(), api_key="ollama")
        """
        return f"{self.host}/v1"

    # --- Transport ----------------------------------------------------------

    def _post(self, path: str, payload: dict, _attempts: int = 3) -> dict:
        # A request that lands while a 19 GB model is still being paged into
        # GPU memory can come back 5xx. Retry those; 4xx is our own mistake.
        for attempt in range(_attempts):
            try:
                r = self._session.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                raise AIError(f"cannot reach the GX10 at {self.host}: {e}") from e
            if r.status_code >= 500 and attempt < _attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise AIError(f"{r.status_code} from {path}: {r.text[:500]}")
            return r.json()
        raise AIError(f"{path} failed after {_attempts} attempts")

    def _stream(self, path: str, payload: dict, timeout: float | None = ...) -> Iterator[dict]:
        payload = {**payload, "stream": True}
        try:
            r = self._session.post(
                f"{self.host}{path}",
                json=payload,
                stream=True,
                timeout=self.timeout if timeout is ... else timeout,
            )
        except requests.RequestException as e:
            raise AIError(f"cannot reach the GX10 at {self.host}: {e}") from e
        if r.status_code >= 400:
            raise AIError(f"{r.status_code} from {path}: {r.text[:500]}")
        for line in r.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


@dataclass
class Conversation:
    """A chat that remembers what was said."""

    ai: AI
    system: str | None = None
    messages: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.system and not self.messages:
            self.messages.append(_message("system", self.system))

    def say(self, text: str, *, images: Sequence[Any] | None = None, **kwargs: Any) -> str:
        self.messages.append(_message("user", text, images))
        result = self.ai.chat(self.messages, stream=False, **kwargs)
        reply = result["message"]
        self.messages.append({"role": "assistant", "content": reply["content"]})
        return reply["content"]

    def stream_say(
        self, text: str, *, images: Sequence[Any] | None = None, **kwargs: Any
    ) -> Iterator[str]:
        self.messages.append(_message("user", text, images))
        parts: list[str] = []
        for chunk in self.ai.chat(self.messages, stream=True, **kwargs):
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                yield piece
        self.messages.append({"role": "assistant", "content": "".join(parts)})

    def reset(self) -> None:
        self.messages = [_message("system", self.system)] if self.system else []


# A ready-made instance, so callers can just `from ai import ai`.
ai = AI()


# --- CLI --------------------------------------------------------------------

def _check(client: AI) -> int:
    print(f"host    {client.host}")
    if not client.is_up():
        print("status  UNREACHABLE", file=sys.stderr)
        print("\nIs the GX10 awake and on the tailnet? Try: tailscale status", file=sys.stderr)
        return 1
    print("status  up")
    print(f"model   {client.model}" + ("" if client.has_model() else "  (NOT PULLED)"))
    print(f"loaded  {', '.join(client.loaded()) or '-'}")
    print("\navailable:")
    for name in client.models():
        print(f"  {name}")
    if not client.has_model():
        print(f"\nRun `python ai.py --pull` to download {client.model}.", file=sys.stderr)
        return 1
    return 0


def _repl(client: AI) -> int:
    chat = client.conversation()
    print(f"{client.model} on {client.host}. Ctrl-D or /exit to quit, /reset to clear.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/reset":
            chat.reset()
            print("(cleared)")
            continue
        try:
            for piece in chat.stream_say(line):
                print(piece, end="", flush=True)
            print()
        except AIError as e:
            print(f"error: {e}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="ai.py", description="Talk to Qwen3-VL running on the ASUS GX10."
    )
    p.add_argument("prompt", nargs="*", help="the prompt; reads stdin if omitted")
    p.add_argument("-i", "--image", action="append", default=[], help="image file (repeatable)")
    p.add_argument("-s", "--system", help="system prompt")
    p.add_argument("-m", "--model", help=f"model to use (default {MODEL})")
    p.add_argument("--host", help=f"Ollama host (default {HOST})")
    p.add_argument("-t", "--temperature", type=float)
    p.add_argument("--json", action="store_true", help="force valid JSON output")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--chat", action="store_true", help="interactive session")
    p.add_argument("--check", action="store_true", help="show connection and model status")
    p.add_argument("--pull", action="store_true", help="download the model onto the GX10")
    p.add_argument("--warm", action="store_true", help="preload the model into GPU memory")
    args = p.parse_args(argv)

    client = AI(
        host=(args.host or HOST),
        model=(args.model or MODEL),
    )
    if args.temperature is not None:
        client.options["temperature"] = args.temperature

    try:
        if args.check:
            return _check(client)
        if args.pull:
            client.pull()
            return 0
        if args.warm:
            client.warm()
            print(f"{client.model} loaded.")
            return 0
        if args.chat:
            return _repl(client)

        prompt = " ".join(args.prompt) or (None if sys.stdin.isatty() else sys.stdin.read())
        if not prompt:
            p.print_help()
            return 2

        kwargs: dict[str, Any] = {}
        if args.json:
            kwargs["format"] = "json"

        if args.no_stream or args.json:
            print(client.ask(prompt, images=args.image, system=args.system, **kwargs))
        else:
            for piece in client.stream(prompt, images=args.image, system=args.system, **kwargs):
                print(piece, end="", flush=True)
            print()
        return 0
    except AIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
