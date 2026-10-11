"""Human-in-the-loop: who controls the live session, and how control moves.

The control model is a single owner per live session:

    automation ──(intervention raised)──▶ awaiting_human
    awaiting_human ──(operator takes control)──▶ human
    human ──(operator resolves: retry | done | approve | deny | abort)──▶ automation

Automation checks `assert_automation()` before every action, so the moment a
person takes control the automation cannot act, even if it was mid-loop; the
operator never fights the bot for the mouse. While a human holds control, the
surface's in-page recorder reports what they click and change, and those
events are kept on the intervention (redacted) as part of the evidence.

How the request reaches a person is behind the Escalator interface:
  * TerminalEscalator — prompts on the console (the live browser window is the
    session; the operator uses it directly)
  * the operator console in console.py — a small web page listing open
    interventions with their screenshot, a CDP endpoint to attach to the *same*
    browser, and resolve buttons
  * ScriptedEscalator — for tests and the recorded demo
A real deployment would put a queue and a co-browsing console behind the same
interface; the engine wouldn't change.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal, Protocol

Reason = Literal[
    "approval_required",     # a risky step needs a person's decision
    "stuck",                 # discovery: the agent can't make progress
    "unrecognized_state",    # replay: the screen matches nothing known
    "target_not_found",
    "checkpoint_failed",
    "recovery_exhausted",
]

ResolutionAction = Literal["retry", "done", "approve", "deny", "abort"]


class Owner(str, Enum):
    automation = "automation"
    awaiting_human = "awaiting_human"
    human = "human"


class ControlLost(Exception):
    """Automation tried to act while a human holds the session."""


@dataclass
class Intervention:
    run_id: str
    reason: Reason
    message: str
    capability: str | None = None       # replay
    goal: str | None = None             # discovery
    step_id: str | None = None
    step_index: int | None = None
    step_description: str = ""
    location: dict[str, str] = field(default_factory=dict)
    screenshot: str | None = None
    cdp_endpoint: str | None = None     # attach here to drive the same browser
    options: list[ResolutionAction] = field(default_factory=lambda: ["retry", "done", "abort"])
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Resolution:
    action: ResolutionAction
    by: str = "operator"
    note: str = ""
    human_events: list[dict] = field(default_factory=list)


class Escalator(Protocol):
    def escalate(self, intervention: Intervention, session: "SessionControl") -> Resolution: ...


class SessionControl:
    """The single source of truth for who may act on one live session."""

    def __init__(self, on_change: Callable[[Owner, Owner, str], None] | None = None) -> None:
        self._owner = Owner.automation
        self._lock = threading.Lock()
        self._resolved = threading.Event()
        self._resolution: Resolution | None = None
        self.current: Intervention | None = None
        self.human_events: list[dict] = []
        self.on_change = on_change

    @property
    def owner(self) -> Owner:
        return self._owner

    def _set(self, owner: Owner, why: str) -> None:
        with self._lock:
            old, self._owner = self._owner, owner
        if self.on_change and old != owner:
            self.on_change(old, owner, why)

    def assert_automation(self) -> None:
        if self._owner != Owner.automation:
            raise ControlLost(f"session is held by {self._owner.value}")

    # -- the automation side --

    def raise_intervention(self, intervention: Intervention) -> None:
        self.current = intervention
        self.human_events = []
        self._resolution = None
        self._resolved.clear()
        self._set(Owner.awaiting_human, f"intervention {intervention.id}: {intervention.reason}")

    def wait(self, timeout_s: float) -> Resolution | None:
        """Block until a human resolves the intervention, or time runs out."""
        if not self._resolved.wait(timeout_s):
            return None
        return self._resolution

    # -- the human side --

    def take(self, by: str = "operator") -> None:
        if self._owner == Owner.automation:
            raise RuntimeError("nothing to take: no intervention is open")
        self._set(Owner.human, f"taken by {by}")

    def record_human_event(self, event: dict) -> None:
        # Only while a human holds control; the automation's own clicks fire
        # the same DOM events and must not be mistaken for the operator's.
        if self._owner == Owner.human:
            self.human_events.append({**event, "t": time.time()})

    def resolve(self, resolution: Resolution) -> None:
        if self.current is None:
            raise RuntimeError("no open intervention")
        if resolution.action not in self.current.options:
            raise ValueError(f"{resolution.action} not offered; options: {self.current.options}")
        resolution.human_events = list(self.human_events)
        self._resolution = resolution
        self._set(Owner.automation, f"resolved: {resolution.action} by {resolution.by}")
        self.current = None
        self._resolved.set()


class NoEscalation:
    """Unattended: nobody to ask. Escalations become failures."""

    def escalate(self, intervention: Intervention, session: SessionControl) -> Resolution:
        return Resolution(action="abort", by="system", note="no operator configured")


class ScriptedEscalator:
    """Resolves interventions with a scripted operator. For tests and demos.

    `act(intervention, session)` plays the human: it runs while the session is
    held by the human, can drive the live browser (e.g. over CDP), and returns
    the resolution.
    """

    def __init__(self, act: Callable[[Intervention, SessionControl], Resolution]) -> None:
        self.act = act
        self.seen: list[Intervention] = []

    def escalate(self, intervention: Intervention, session: SessionControl) -> Resolution:
        self.seen.append(intervention)
        session.raise_intervention(intervention)
        session.take("scripted-operator")
        resolution = self.act(intervention, session)
        session.resolve(resolution)
        return session.wait(0) or resolution


class TerminalEscalator:
    """Asks on the terminal. The headed browser window is the live session:
    the operator works in it directly, then answers here."""

    def __init__(self, timeout_s: float = 900) -> None:
        self.timeout_s = timeout_s

    def escalate(self, intervention: Intervention, session: SessionControl) -> Resolution:
        session.raise_intervention(intervention)
        print("\n" + "=" * 70)
        print(f"  INTERVENTION {intervention.id}: {intervention.reason}")
        print(f"  {intervention.message}")
        if intervention.step_id:
            print(f"  step: {intervention.step_id} — {intervention.step_description}")
        if intervention.screenshot:
            print(f"  screenshot: {intervention.screenshot}")
        if intervention.cdp_endpoint:
            print(f"  attach to the live session: {intervention.cdp_endpoint}")
        print("  You now control the browser window. Do what is needed there.")
        session.take("terminal-operator")
        options = "/".join(intervention.options)
        while True:
            try:
                answer = input(f"  then resolve [{options}]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "abort"
            if answer in intervention.options:
                break
        note = ""
        try:
            note = input("  note for the record (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            pass
        session.resolve(Resolution(action=answer, by="terminal-operator", note=note))  # type: ignore[arg-type]
        return session.wait(0) or Resolution(action="abort")
