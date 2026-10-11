# Rote — Record Once, Then Execute

A computer-use system for legacy back-office apps. An LLM works out how to do a
task on a live UI **once**. The run is compiled into a typed, versioned,
reviewable **capability**, and from then on an AI agent invokes that capability
by name. Replay is deterministic: no model in the loop. It classifies runtime
errors as business outcomes, recoverable conditions or failures, and it can hand
the live session to a human when it gets stuck.

```
goal ──▶ discovery (LLM observe→decide→act) ──▶ capability.yaml ──▶ replay(inputs) ──▶ result
                    │                                 ▲                 │
                    └─ human approval / takeover ─────┴── review ───────┴─ escalate to operator
```

Evidence of real runs is in [evidence/](evidence/).

## Setup

Python 3.11+ (developed on 3.14), Linux or macOS.

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    playwright install chromium

### Model (discovery only)

Discovery uses a vision-language model served by [Ollama](https://ollama.com).
The runs in `evidence/` used `qwen3-vl:30b` on a GX10 reached over Tailscale.
Any Ollama host with that model works:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROTE_AI_HOST` | `http://gx10-3d18:11434` | Ollama endpoint |
| `ROTE_AI_MODEL` | `qwen3-vl:30b` | model tag (needs vision + JSON-schema output) |
| `ROTE_AI_TIMEOUT` | `600` | per-request seconds |

`python ai.py --check` reports whether the host is reachable and the model is present.

### Secrets

Application credentials are never written to an artifact, a log or a prompt.
They are resolved from the environment when the sign-on step types them:

    export ROTE_SECRET_CORELINK_USER=teller01
    export ROTE_SECRET_CORELINK_PASSWORD=demo-pass-123

The values above are the bundled mock app's fictional demo credentials. Rote
falls back to them only when the target is `localhost`.

### Without live services

- **Replay needs no model.** With the mock app running, every replay command
  below works offline.
- **The test suite needs no model.** It drives the real mock app in headless
  Chromium, with a scripted model standing in for the LLM in discovery tests:
  `pytest -q` runs 162 tests in about 3 minutes.
- **The real discovery runs are already recorded** in `evidence/`.

## Demo

Run each in its own terminal from the repo root.

**1. Start the target app.** "CoreLink Teller" is a deliberately legacy mock
core-banking app: framesets, table layouts, no ids, and injectable faults.

    python -m rote app --port 8765               # tenant a: First Community FCU
    python -m rote app --port 8766 --tenant b    # tenant b: Lakeshore CU (same product, configured differently)

**2. Discover.** The LLM drives the app, and the run is recorded as a draft capability.

    python -m rote discover \
        --goal "Look up member 12345 and read their current primary savings balance" \
        --name corelink.member.read_savings_balance \
        --title "Read a member's primary savings balance" \
        --param member_id=12345 --pattern 'member_id=\d{4,9}' --headed

This writes `library/capabilities/corelink.member.read_savings_balance/1.0.0.yaml`
plus evidence in `runs/<run-id>/`. Add `--headed` to watch the browser.

**3. Review it.**

    python -m rote show corelink.member.read_savings_balance

**4. Replay it with other inputs.** This is deterministic: no model calls.

    python -m rote replay corelink.member.read_savings_balance --input member_id=20001
    #  → status: succeeded, outputs: {primary_savings_balance: "87012.00"}

**5. Exceptional states.** Use real inputs, or inject faults into the app.

    ... --input member_id=99999                 # business_outcome: member_not_found
    ... --input member_id=77777                 # business_outcome: access_restricted
    ... --input member_id=12a                   # failed: invalid_input (rejected before touching the UI)
    ... --input member_id=12345 --inject notice # succeeded; system notice dismissed (recovered)
    ... --input member_id=12345 --inject expire@submit_member_inquiry   # re-signs on, restarts, succeeds
    ... --input member_id=12345 --inject error  # failed: app_error, retryable, with screenshot + snapshot

**6. Human handoff.** Inject a dialog the app profile doesn't know about:

    python -m rote replay corelink.member.read_savings_balance --input member_id=12345 \
        --inject unknown_dialog --headed

The run escalates and prints the operator console URL (`http://127.0.0.1:8770`).
Open it and click **Take control**. Then, in the browser window (the same live
session), click **Remind Me Later**, and back in the console click **done**. The
run verifies the checkpoint, resumes and returns the balance. Your click is
recorded in the run log. For a hands-free version, run
`python scripts/operator_bot.py` first. It plays the operator, attaching to the
session over CDP.

**7. A second tenant.** Use the base capability plus a tenant overlay:

    python -m rote replay corelink.member.read_savings_balance --input member_id=12345 \
        --base-url http://127.0.0.1:8766                       # fails: drift events, wrong-typed value caught
    python -m rote replay corelink.member.read_savings_balance --input member_id=12345 \
        --base-url http://127.0.0.1:8766 --tenant lakeshore    # succeeds with overlays/lakeshore.yaml

**8. Approve, and the agent-facing catalog.**

    python -m rote approve corelink.member.read_savings_balance --by "A. Reviewer"
    python -m rote catalog        # every capability as a typed tool definition

`scripts/make_evidence.sh` reruns every replay scenario above into `evidence/`.

## Layout

| Path | What |
| --- | --- |
| `rote/schema.py` | Capability, AppProfile, Overlay, and the RunResult contract (pydantic) |
| `rote/surface.py` | the surface seam: WebSurface (Playwright), locator resolution and synthesis, evidence capture |
| `rote/agent.py` | discovery: the LLM observe → decide → act loop |
| `rote/recorder.py` | trace → capability: templating, checkpoints, PII lint |
| `rote/replay.py` | deterministic replay, detectors, recovery, escalation |
| `rote/policy.py` | allowlist, risk classes, redaction |
| `rote/handoff.py`, `rote/console.py` | session control model, escalators, operator console |
| `rote/store.py`, `rote/cli.py` | artifact storage and the command line |
| `library/` | app profiles, policies, capabilities and tenant overlays (the reviewed artifacts) |
| `mockapp/corelink.py` | the target application |
| `ai.py` | Ollama client for the GX10 |
| `tests/` | unit + browser integration tests |

`src/` and `main.py` are an earlier prototype of this project: a one-shot
plan-then-execute REPL over shell, desktop and browser controllers. They're
kept for history. The system described above lives in `rote/`.
