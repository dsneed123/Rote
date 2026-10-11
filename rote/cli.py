"""Command line: run the mock app, discover a capability, replay it, review it.

    python -m rote app [--port 8765] [--tenant a|b]
    python -m rote discover --goal "..." --name corelink.member.read_savings_balance \\
        --param member_id=12345 --pattern 'member_id=\\d{4,9}'
    python -m rote replay corelink.member.read_savings_balance --input member_id=20001
    python -m rote replay ... --inject notice@submit_inquiry     # simulate runtime conditions
    python -m rote show corelink.member.read_savings_balance      # review view
    python -m rote approve corelink.member.read_savings_balance --by "J. Reviewer"
    python -m rote catalog                                        # capabilities as agent tools

Credentials come from ROTE_SECRET_CORELINK_USER / ROTE_SECRET_CORELINK_PASSWORD.
For the bundled mock app the demo values are used if those are unset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .evidence import RUNS, RunLog, new_run_id
from .handoff import NoEscalation, SessionControl, TerminalEscalator
from .policy import Guard, Policy, Redactor
from .schema import Sensitivity, Status, ValueType
from .store import LIBRARY, Store

DEFAULT_BASE = "http://127.0.0.1:8765"
DEMO_SECRETS = {"corelink_user": "teller01", "corelink_password": "demo-pass-123"}


def secrets_source(base_url: str):
    def get(name: str) -> str:
        key = f"ROTE_SECRET_{name.upper()}"
        if key in os.environ:
            return os.environ[key]
        # The fictional mock app's demo credentials, only for a local target.
        if base_url.startswith(("http://127.0.0.1", "http://localhost")) and name in DEMO_SECRETS:
            return DEMO_SECRETS[name]
        raise KeyError(f"secret {name!r} not set; export {key}")
    return get


def _pairs(values: list[str] | None, what: str) -> dict[str, str]:
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"{what} must be NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


class Session:
    """Everything one run needs: surface, log, control, operator."""

    def __init__(self, args, mode: str, kind: str, product: str, tenant: str | None) -> None:
        from .surface import WebSurface

        self.store = Store(args.library)
        self.profile = self.store.profile(product, tenant)
        policy = Policy.load(Path(args.library) / "policies" / f"{product}.yaml").for_base_url(args.base_url)
        self.redactor = Redactor(money=policy.redact_money_in_logs)
        self.guard = Guard(policy, mode=mode)  # type: ignore[arg-type]
        self.policy = policy
        self.control = SessionControl(on_change=lambda old, new, why: self.log.event(
            "control", from_=old.value, to=new.value, why=why))
        self.log = RunLog(new_run_id(kind), self.redactor, root=args.runs,
                          echo=(lambda line: print(f"  · {line}")) if not args.quiet else None)
        self.surface = WebSurface(args.base_url, self.guard, self.redactor, headless=not args.headed,
                                  cdp_port=args.cdp_port if args.operator != "none" else None,
                                  secrets=secrets_source(args.base_url),
                                  on_human_event=self.control.record_human_event)
        self.surface.sensitive_labels = self.profile.sensitive_labels
        self.console = None
        self.args = args

    def escalator(self):
        pump = lambda ms: self.surface.page.wait_for_timeout(ms)   # noqa: E731
        if self.args.operator == "terminal":
            return TerminalEscalator(pump=pump, timeout_s=self.args.operator_timeout)
        if self.args.operator == "console":
            from .console import ConsoleEscalator, OperatorConsole
            self.console = OperatorConsole(self.control, port=self.args.console_port).start()
            print(f"  operator console: {self.console.url}")
            return ConsoleEscalator(self.console, pump=pump, timeout_s=self.args.operator_timeout)
        return NoEscalation()

    def __enter__(self):
        self.surface.start()
        return self

    def __exit__(self, *exc):
        self.surface.close()
        if self.console:
            self.console.stop()
        self.log.close()


# --- commands ---------------------------------------------------------------------


def cmd_app(args) -> int:
    from mockapp.corelink import main
    return main(["--port", str(args.port), "--tenant", args.tenant, "--faults"])


def cmd_discover(args) -> int:
    from ai import AI
    from .agent import Discoverer
    from .recorder import Param, RecordingRejected, compile_capability

    values = _pairs(args.param, "--param")
    types = _pairs(args.type, "--type")
    patterns = _pairs(args.pattern, "--pattern")
    choices = _pairs(args.choices, "--choices")
    params = {
        name: Param(value=value, type=ValueType(types.get(name, "string")),
                    sensitivity=Sensitivity.pii if name in (args.sensitive or []) else Sensitivity.internal,
                    pattern=patterns.get(name), values=choices[name].split("|") if name in choices else None)
        for name, value in values.items()
    }
    ai = AI(model=args.model) if args.model else AI()
    if not ai.is_up():
        print(f"error: model host {ai.host} is unreachable", file=sys.stderr)
        return 2

    with Session(args, "discovery", "discovery", args.product, args.tenant) as s:
        print(f"discovery run {s.log.run_id} with {ai.model} — evidence in {s.log.dir}")
        agent = Discoverer(s.surface, s.profile, s.log, s.redactor, ai, session=s.control,
                           escalator=s.escalator(), max_steps=args.max_steps or s.policy.max_steps)
        result = agent.run(args.goal, params, product=args.product, entry=args.entry, tenant=args.tenant)
        print(f"\ndiscovery: {result.status} after {result.turns} turn(s) — {result.message}")
        if result.status != "recorded":
            return 1
        try:
            existing = s.store.versions(args.name)
            version = "1.0.0"
            if existing:
                latest = s.store.load(args.name)
                major, minor, _ = (int(x) for x in latest.version.split("."))
                version = f"{major}.{minor + 1}.0" if latest.approved else latest.version
            # The run's redactor: it knows the values read during discovery.
            cap, notes = compile_capability(result.recording, args.name, s.profile, s.redactor,
                                            version=version, title=args.title)
        except RecordingRejected as e:
            s.log.event("recording_rejected", reason=str(e))
            print(f"not recorded: {e}", file=sys.stderr)
            return 1
        path = s.store.save(cap)
        s.log.event("recorded", capability=cap.name, version=cap.version, path=str(path),
                    fingerprint=cap.fingerprint(), notes=notes)
        print(f"recorded {cap.name} v{cap.version} (draft) → {path}")
        for note in notes:
            print(f"  review: {note}")
    return 0


def _faults(base_url: str, spec: list[str] | None):
    """--inject notice|error|expire|slow=MS [@step_id] → a before_step hook."""
    import requests

    plan: list[tuple[str | None, dict]] = []
    for item in spec or []:
        what, _, at = item.partition("@")
        name, _, arg = what.partition("=")
        fault = {"notice": {"notice_once": True}, "error": {"error_once": True},
                 "expire": {"expire_session": True}, "slow": {"slow_ms": int(arg or 1500)},
                 "unknown_dialog": {"password_prompt_once": True}}.get(name)
        if fault is None:
            raise SystemExit(f"unknown fault {name!r}; use notice, error, expire, unknown_dialog or slow=MS")
        plan.append((at or None, fault))
    if not plan:
        return None
    for at, fault in plan:   # faults with no step apply from the start
        if at is None:
            requests.post(f"{base_url}/__control/faults", json=fault, timeout=5)
    fired: set[int] = set()

    def before_step(index, step):
        for n, (at, fault) in enumerate(plan):
            if at == step.id and n not in fired:
                fired.add(n)
                requests.post(f"{base_url}/__control/faults", json=fault, timeout=5)
                print(f"  ! injected {fault} before {step.id}")
    return before_step


def cmd_replay(args) -> int:
    from .replay import Replayer

    store = Store(args.library)
    name = args.capability.replace("__", ".")
    capability = store.resolve(name, args.tenant, args.version)
    inputs: dict[str, Any] = _pairs(args.input, "--input")
    with Session(args, "replay", "replay", capability.app.product, args.tenant) as s:
        s.guard.approved = capability.approved
        print(f"replay run {s.log.run_id}: {capability.name} v{capability.version} "
              f"({'approved' if capability.approved else 'DRAFT'}) — evidence in {s.log.dir}")
        replayer = Replayer(s.surface, s.profile, s.log, s.redactor, session=s.control,
                            escalator=s.escalator(), before_step=_faults(args.base_url, args.inject),
                            escalation_timeout_s=args.operator_timeout)
        result = replayer.run(capability, inputs, tenant=args.tenant)
    # stdout is the caller's channel: it gets the real outputs. Evidence doesn't.
    print(json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2))
    return {Status.succeeded: 0, Status.business_outcome: 0}.get(result.status, 1)


def cmd_show(args) -> int:
    store = Store(args.library)
    cap = store.resolve(args.capability.replace("__", "."), args.tenant, args.version)
    status = "APPROVED" if cap.approved else ("draft (edited since approval)" if cap.approval.status == "approved"
                                               else cap.approval.status)
    print(f"{cap.name} v{cap.version} — {cap.title}")
    print(f"  product {cap.app.product} ({cap.app.surface}), entry {cap.app.entry}; side effects: "
          f"{cap.side_effects}; {status}; fingerprint {cap.fingerprint()}")
    if cap.provenance.run_id:
        p = cap.provenance
        print(f"  discovered by {p.model} in run {p.run_id} ({p.recorded_at}); human steps: {p.human_steps}")
    print("  inputs:")
    for n, i in cap.inputs.items():
        print(f"    {n}: {i.type.value}{' ' + i.pattern if i.pattern else ''}"
              f"{' one of ' + str(i.values) if i.values else ''} [{i.sensitivity.value}]")
    print("  outputs:")
    for n, o in cap.outputs.items():
        print(f"    {n}: {o.type.value} [{o.sensitivity.value}]")
    print(f"  business outcomes: {', '.join(o.code for o in cap.outcomes) or 'none'}")
    print("  steps:")
    for i, step in enumerate(cap.steps, 1):
        what = step.target.description if step.target else ""
        val = f" ← {step.value}" if step.value else ""
        out = f" → {step.output}" if step.output else ""
        print(f"   {i}. [{step.risk.value:6}] {step.id}: {step.action.value} {what}{val}{out}")
        if step.target:
            for k, s in enumerate(step.target.strategies):
                fields = {k2: v for k2, v in s.model_dump().items() if k2 != "by"}
                print(f"        {'locate' if k == 0 else 'else  '} {s.by} {fields}"
                      f"{' in frame ' + str(step.target.frame) if k == 0 and step.target.frame else ''}")
        for c in step.expect:
            print(f"        expect {c.kind}: {getattr(c, 'text', None) or getattr(c, 'pattern', None) or getattr(c, 'path_regex', '')}")
    print("  success when: " + "; ".join(getattr(c, "text", None) or c.kind for c in cap.success))
    return 0


def cmd_approve(args) -> int:
    store = Store(args.library)
    cap = store.load(args.capability, args.version)
    cap = store.approve(cap.name, cap.version, by=args.by)
    print(f"approved {cap.name} v{cap.version} by {args.by}; fingerprint {cap.approval.fingerprint}")
    return 0


def cmd_catalog(args) -> int:
    """The agent-facing view: every capability as a callable tool."""
    tools = []
    for cap in Store(args.library).catalog():
        tool = cap.tool_schema()
        tool["x-rote"] = {"capability": cap.name, "version": cap.version, "approved": cap.approved,
                          "side_effects": cap.side_effects,
                          "outputs": {n: o.type.value for n, o in cap.outputs.items()},
                          "business_outcomes": [o.code for o in cap.outcomes]}
        tools.append(tool)
    print(json.dumps(tools, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rote", description="Record once, then execute.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, run: bool = True):
        p.add_argument("--library", default=str(LIBRARY))
        if run:
            p.add_argument("--base-url", default=DEFAULT_BASE, help="the tenant's app instance")
            p.add_argument("--tenant", default=None, help="tenant id for overlays (e.g. lakeshore)")
            p.add_argument("--runs", default=str(RUNS), help="where evidence goes")
            p.add_argument("--headed", action="store_true", help="show the browser window")
            p.add_argument("--operator", choices=["console", "terminal", "none"], default="console")
            p.add_argument("--operator-timeout", type=float, default=900)
            p.add_argument("--console-port", type=int, default=8770)
            p.add_argument("--cdp-port", type=int, default=9222)
            p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("app", help="run the CoreLink mock app")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--tenant", choices=["a", "b"], default="a")
    p.set_defaults(fn=cmd_app)

    p = sub.add_parser("discover", help="LLM-driven discovery run that records a capability")
    common(p)
    p.add_argument("--goal", required=True)
    p.add_argument("--name", required=True, help="capability name, e.g. corelink.member.read_savings_balance")
    p.add_argument("--title")
    p.add_argument("--product", default="corelink-teller")
    p.add_argument("--entry", default="/main")
    p.add_argument("--param", action="append", help="NAME=EXAMPLE_VALUE")
    p.add_argument("--type", action="append", help="NAME=string|integer|money|decimal|date|enum")
    p.add_argument("--pattern", action="append", help="NAME=REGEX")
    p.add_argument("--choices", action="append", help="NAME=a|b|c (enum values)")
    p.add_argument("--sensitive", action="append", help="NAME: treat as PII")
    p.add_argument("--model", help="override ROTE_AI_MODEL")
    p.add_argument("--max-steps", type=int)
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("replay", help="deterministic replay of a capability")
    common(p)
    p.add_argument("capability")
    p.add_argument("--version")
    p.add_argument("--input", action="append", help="NAME=VALUE")
    p.add_argument("--inject", action="append", help="notice|error|expire|unknown_dialog|slow=MS[@step_id]")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("show", help="review a capability")
    common(p, run=False)
    p.add_argument("capability")
    p.add_argument("--version")
    p.add_argument("--tenant")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("approve", help="approve a reviewed capability version")
    common(p, run=False)
    p.add_argument("capability")
    p.add_argument("--version")
    p.add_argument("--by", required=True)
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("catalog", help="list capabilities as agent-callable tools")
    common(p, run=False)
    p.set_defaults(fn=cmd_catalog)
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # for ai.py and mockapp
    args = build_parser().parse_args(argv)
    return args.fn(args)
