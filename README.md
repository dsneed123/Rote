# Rote
Record Once Then Execute

## Running it

    python main.py

Type a goal. Rote replays a recorded workflow if it has one, otherwise the
model plans the steps, you confirm anything that changes state, and a plan that
works is recorded — so the next time is a replay with no model call.

    rote> count the python files in src and write it to count.txt
      · planning with the shell controller
      run run(command="find src -name '*.py' | wc -l > count.txt")?
      [y/N] y
      [ok] ...
      planned 1 step(s); all ok; recorded to count-the-python-files-....yaml

Commands: `/workflows`, `/forget <name>`, `/controllers`, `/ask <question>`,
`/help`, `/exit`.

### Layers

| File | Role |
| --- | --- |
| `main.py` | entry point and prompt loop |
| `src/bridge.py` | replay-or-plan orchestration, human correction |
| `src/analyze.py` | screenshot + goal → validated plan |
| `src/controller.py` | executes steps; shell / desktop / browser |
| `src/plan.py` | the `{tool, args, why}` contract |
| `src/workflows.py` | YAML recordings in `src/workflows/` |

### Controllers

`/controllers` shows which work here. **shell** works out of the box. The other
two need setup:

    sudo apt install ydotool && sudo systemctl enable --now ydotoold   # desktop
    pip install playwright && playwright install chromium              # browser

Desktop input uses ydotool rather than xdotool because this is a Wayland
session, where X11 tools cannot inject into other applications. Screenshots
also need a backend (`sudo apt install gnome-screenshot`); without one, Rote
still plans, just without seeing the screen.

## AI backend

Inference runs on the ASUS GX10, which hosts Ollama. `ai.py` is the client
library; `main.py` owns startup — it connects, fails loudly if the GX10 is
unreachable or the model is missing, and warms the model so the first request
doesn't pay the ~30s load cost.

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
