# Rote
Record Once Then Execute

## AI backend

Inference runs on the ASUS GX10, which hosts Ollama. `ai.py` is the client
library; `main.py` is the entry point and owns startup — it connects, fails
loudly if the GX10 is unreachable or the model is missing, and warms the model
so the first request doesn't pay the ~30s load cost.

    python main.py

The GX10 is reached over Tailscale, so this works from anywhere both machines
are on the tailnet — no port forwarding, no VPN config.

`ai-guide.pdf` is the full guide: Python API, tool calls, commands.

### Maintenance commands

Running `ai.py` directly is for checking the link and one-off questions, not
for running the project:

    python ai.py --check              # connection + model status
    python ai.py "hello"              # one-shot, streams to stdout
    python ai.py -i shot.png "what is this?"
    python ai.py --chat               # interactive session
    python ai.py --pull               # download the model onto the GX10
    python ai.py --warm               # preload into GPU memory (~30s)

In code:

    from ai import ai               # ready-made client, or use the one main.py built

    ai.ask("Summarise this log.", system="Be terse.")
    ai.look("screenshot.png", "What button is highlighted?")
    ai.ask("List 3 colours.", format="json")     # forced valid JSON

    for piece in ai.stream("Write a haiku."):
        print(piece, end="")

    chat = ai.conversation(system="You are terse.")
    chat.say("Remember the number 7.")
    chat.say("What number?")          # history is kept

    ai.embed(["some text"])          # 768-dim, via nomic-embed-text

Tool calling — the model runs your functions and uses the results:

    def get_weather(city: str, units: str = "celsius") -> str:
        """Look up the current weather for a city."""
        return json.dumps({"city": city, "temp": 14, "sky": "raining"})

    ai.with_tools("What's the weather in Boston?", tools=[get_weather])

Schemas are derived from the annotations and the docstring's first line.

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROTE_AI_HOST` | `http://gx10-3d18:11434` | Ollama endpoint on the GX10 |
| `ROTE_AI_MODEL` | `qwen3-vl:30b` | model tag |
| `ROTE_AI_TIMEOUT` | `600` | per-request seconds |
| `ROTE_AI_KEEP_ALIVE` | `30m` | how long the model stays in GPU memory |

### Using the OpenAI SDK instead

Ollama also serves an OpenAI-compatible API, so any library that speaks it can
point at the GX10 directly:

    from openai import OpenAI
    client = OpenAI(base_url="http://gx10-3d18:11434/v1", api_key="ollama")

`AI().openai_base_url()` returns that URL from the configured host.
