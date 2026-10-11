"""Discovery: an LLM works out how to reach a goal on a live surface, once.

    observe ─▶ handle known conditions ─▶ ask the model for ONE action ─▶
    validate + policy ─▶ act ─▶ record (verified locators) ─▶ observe ...

The model sees a numbered list of controls (role, name, neighbouring label,
value) plus the visible text of each frame and a masked screenshot. It answers
with one JSON action. It never produces selectors or coordinates: it picks a
number, and the surface turns that control into ranked locators verified
against the live page. So the model's job is judgement — which control, what
value, are we done — and the artifact's robustness doesn't depend on it.

Two things the model does not get to decide:
  * runtime conditions (a system notice, an expired session) are handled by
    the app profile *before* the model sees the screen, and are never recorded
    as steps. Otherwise the capability would only work on days with a notice.
  * risky actions go through the same Guard as replay; a commit during
    discovery is put to a human.

The model also names, each turn, a short heading identifying the screen it is
looking at. That becomes the checkpoint of the step that led here — after the
system verifies it is really on screen and not a value or a name.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence import RunLog
from .handoff import Escalator, Intervention, NoEscalation, SessionControl
from .policy import NeedsHuman, PolicyViolation, Redactor
from .recorder import Param, RecordedStep, Recording
from .replay import SignOnFailed, detect, dismiss, sign_on
from .schema import (Action, AppProfile, Disposition, FrameRef, LabelNearLocator, OutputSpec,
                     Risk, RoleLocator, Sensitivity, Target, TextLocator, ValueType)
from .surface import Observation, Surface, TargetNotFound, render

SYSTEM = """\
You operate a legacy back-office banking application for a bank employee.
Each turn you get the current screen: numbered controls, the visible text of
each frame, and a screenshot. Choose exactly ONE next action toward the goal.

Actions (field "action"):
  click    element=<number>
  fill     element=<number>, value=<text>   use {{inputs.NAME}} to type a parameter
  select   element=<number>, value=<option text exactly as listed>
  press    value=<key, e.g. Enter>
  extract  output=<snake_case name>, output_type=string|money|integer|date,
           value=<REQUIRED: the value copied EXACTLY as shown on screen, e.g. "$1,234.56">
  done     success_text=<short text visible NOW that proves the goal is complete>
  stuck    thought=<why you cannot proceed>; a human operator will take over

Rules:
- Use only element numbers from the CURRENT list. One action per turn.
- Type parameters with {{inputs.NAME}}, never by retyping their values.
- Extract every value the goal asks to read before you say done.
- "screen": copy EXACTLY a short heading that identifies the CURRENT screen
  (e.g. a panel title). Never a person's name, a number, or an amount.
- "step_name": a short snake_case name for this action, e.g. enter_member_number.
- Fill every field; use "" or -1 for fields the action doesn't use.
- Reply with JSON only.
"""

# Every field is required and non-null ("" or -1 when unused). With optional
# fields, the model reliably left `value` out of extract actions and kept the
# value in its prose; forcing the key makes it fill it in.
ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "screen": {"type": "string"},
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["click", "fill", "select", "press", "extract", "done", "stuck"]},
        "element": {"type": "integer"},
        "value": {"type": "string"},
        "output": {"type": "string"},
        "output_type": {"type": "string", "enum": ["", "string", "money", "integer", "decimal", "date"]},
        "success_text": {"type": "string"},
        "step_name": {"type": "string"},
    },
    "required": ["screen", "thought", "action", "element", "value", "output", "output_type",
                 "success_text", "step_name"],
}

Status = Literal["recorded", "unachievable", "max_steps", "stuck", "denied", "failed"]


@dataclass
class DiscoveryResult:
    status: Status
    recording: Recording
    message: str = ""
    outcome_code: str | None = None
    turns: int = 0


@dataclass
class Turn:
    n: int
    summary: str


class Discoverer:
    def __init__(self, surface: Surface, profile: AppProfile, log: RunLog, redactor: Redactor, ai: Any, *,
                 session: SessionControl | None = None, escalator: Escalator | None = None,
                 max_steps: int = 20, max_errors: int = 3, max_duration_s: float = 900) -> None:
        self.surface = surface
        self.profile = profile
        if hasattr(surface, "sensitive_labels"):
            surface.sensitive_labels = list(profile.sensitive_labels)
        self.log = log
        self.redactor = redactor
        self.ai = ai
        self.session = session or SessionControl()
        self.escalator = escalator or NoEscalation()
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.max_duration_s = max_duration_s
        # What the model is shown is redacted too — but not amounts: the model
        # is on-prem and has to read the screen to do its job.
        self.prompt_redactor = Redactor(money=False)

    # -- the loop --

    def run(self, goal: str, params: dict[str, Param], *, product: str, entry: str,
            tenant: str | None = None) -> DiscoveryResult:
        recording = Recording(goal=goal, product=product, entry=entry, params=params,
                              run_id=self.log.run_id, model=getattr(self.ai, "model", None), tenant=tenant)
        for name, p in params.items():
            if p.sensitivity in (Sensitivity.pii, Sensitivity.secret):
                self.redactor.know(p.value, f"{p.sensitivity.value}:{name}")
            if p.sensitivity == Sensitivity.secret:
                self.prompt_redactor.know(p.value, f"secret:{name}")
        self.log.event("discovery_start", goal=goal, product=product, entry=entry, tenant=tenant,
                       params={k: v.value if v.sensitivity in (Sensitivity.public, Sensitivity.internal)
                               else f"<{v.sensitivity.value}:{k}>" for k, v in params.items()},
                       model=recording.model)
        self.history: list[Turn] = []
        self.inputs = {k: p.value for k, p in params.items()}
        started = time.monotonic()
        errors = 0
        last_screen_text = ""

        try:
            self._start(entry)
        except SignOnFailed as e:
            return self._end(DiscoveryResult("failed", recording, f"sign-on failed: {e}"))

        for turn in range(1, self.max_steps + 1):
            if time.monotonic() - started > self.max_duration_s:
                return self._end(DiscoveryResult("failed", recording, "discovery timed out", turns=turn))

            found = self._handle_conditions(recording, entry)
            if found and found.detector.disposition == Disposition.business:
                return self._end(DiscoveryResult(
                    "unachievable", recording, f"{found.detector.code}: {found.message}",
                    outcome_code=found.detector.code, turns=turn))
            if found and found.detector.disposition == Disposition.failure:
                return self._end(DiscoveryResult("failed", recording, f"{found.detector.code}: {found.message}",
                                                 turns=turn))

            shot = self.log.path(f"turn{turn:02d}", ".png")
            obs = self.surface.observe(screenshot=shot)
            before, last_screen_text = last_screen_text, "\n".join(obs.text.values())

            try:
                decision = self._decide(goal, params, obs, recording)
            except Exception as e:
                errors += 1
                self.log.event("model_error", turn=turn, error=str(e)[:300])
                self._attach_checkpoint(recording, obs, before, None)
                if errors >= self.max_errors:
                    ended = self._stuck(recording, f"model failed {errors} times: {e}", turn)
                    if ended:
                        return self._end(ended)
                    errors = 0
                continue
            self._attach_checkpoint(recording, obs, before, decision.get("screen"))

            self.log.event("decision", turn=turn, screen=decision.get("screen"), action=decision.get("action"),
                           element=decision.get("element"), value=decision.get("value"),
                           output=decision.get("output"), thought=decision.get("thought"),
                           screenshot=self.log.relative(shot))

            try:
                outcome = self._apply(decision, obs, recording, turn)
            except _Problem as problem:
                errors += 1
                self.history.append(Turn(turn, f"{decision.get('action')} -> ERROR: {problem}"))
                self.log.event("action_rejected", turn=turn, reason=str(problem))
                if errors >= self.max_errors:
                    ended = self._stuck(recording, f"{errors} consecutive errors; last: {problem}", turn)
                    if ended:
                        return self._end(ended)
                    errors = 0
                continue
            errors = 0
            if isinstance(outcome, DiscoveryResult):
                outcome.turns = turn
                return self._end(outcome)

        return self._end(DiscoveryResult("max_steps", recording, f"goal not reached in {self.max_steps} steps",
                                         turns=self.max_steps))

    # -- start and runtime conditions --

    def _start(self, entry: str) -> None:
        self.surface.navigate(entry)
        for _ in range(3):
            found = detect(self.surface, self.profile.detectors)
            if not found or found.detector.disposition != Disposition.recoverable:
                return
            self._remedy(found, entry)

    def _remedy(self, found, entry: str) -> None:
        self.log.event("detected", code=found.detector.code, disposition="recoverable", message=found.message)
        if found.detector.recover.kind == "reauthenticate":
            sign_on(self.surface, self.profile, self.log)
            self.surface.navigate(entry)
        else:
            dismiss(self.surface, found.detector)
        self.log.event("recovered", code=found.detector.code, via=found.detector.recover.kind)

    def _handle_conditions(self, recording: Recording, entry: str):
        """Clear known interruptions before the model looks. Never recorded."""
        for _ in range(3):
            found = detect(self.surface, self.profile.detectors)
            if not found or found.detector.disposition != Disposition.recoverable:
                return found
            if found.detector.recover.kind == "reauthenticate" and recording.steps:
                # The session restarted at the entry point: what was recorded
                # so far no longer leads here, so start the recording over.
                self.log.event("recording_reset", reason=found.detector.code, dropped=len(recording.steps))
                recording.steps.clear()
                recording.outputs.clear()
                self.history.append(Turn(0, "SESSION RESTARTED: you are back at the start; redo the flow"))
            self._remedy(found, entry)
        return None

    # -- asking the model --

    def _decide(self, goal: str, params: dict[str, Param], obs: Observation, recording: Recording) -> dict:
        prompt = self._prompt(goal, params, obs, recording)
        self.log.event("prompt", chars=len(prompt), elements=len(obs.elements))
        raw = self._ask(prompt, obs)
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"model reply was not JSON: {raw[:200]}") from e
        if decision.get("action") not in ACTION_SCHEMA["properties"]["action"]["enum"]:
            raise ValueError(f"unknown action {decision.get('action')!r}")
        # "" and -1 mean "not used" (see ACTION_SCHEMA).
        return {k: (None if v in ("", -1) else v) for k, v in decision.items()}

    def _ask(self, prompt: str, obs: Observation) -> str:
        """One structured call. Thinking stays on: with think=False and a format
        schema, this Ollama build returns the JSON in `thinking` and leaves
        `content` empty — so read either, content first."""
        if not hasattr(self.ai, "chat"):          # simple test doubles
            return self.ai.ask(prompt, images=[obs.screenshot] if obs.screenshot else None,
                               system=SYSTEM, format=ACTION_SCHEMA, temperature=0)
        from ai import _message
        reply = self.ai.chat([_message("system", SYSTEM),
                              _message("user", prompt, [obs.screenshot] if obs.screenshot else None)],
                             format=ACTION_SCHEMA, temperature=0)
        message = reply.get("message", {})
        content = (message.get("content") or "").strip()
        if not content and (message.get("thinking") or "").strip().startswith("{"):
            content = message["thinking"].strip()
        self.log.event("model_reply", chars=len(content), thinking_chars=len(message.get("thinking") or ""),
                       eval_ms=round(reply.get("total_duration", 0) / 1e6))
        return content

    def _prompt(self, goal: str, params: dict[str, Param], obs: Observation, recording: Recording) -> str:
        r = self.prompt_redactor
        lines = [f"GOAL: {goal}", "", "PARAMETERS (type them as {{inputs.NAME}}):"]
        for name, p in params.items():
            shown = p.value if p.sensitivity != Sensitivity.secret else "<secret>"
            lines.append(f"  {name} = {shown}" + (f"  ({p.description})" if p.description else ""))
        if recording.outputs:
            lines.append(f"ALREADY EXTRACTED: {', '.join(recording.outputs)}")
        lines += ["", "HISTORY:"]
        lines += [f"  {t.n}. {r.text(t.summary)}" for t in self.history[-10:]] or ["  (none yet)"]
        lines += ["", "FRAMES: " + ", ".join(f"{k}={v}" for k, v in obs.frames.items()), "", "CONTROLS:"]
        lines += [f"  {r.text(e.line())}" for e in obs.elements[:80]]
        lines += ["", "VISIBLE TEXT:"]
        for frame, text in obs.text.items():
            lines.append(f"--- frame {frame} ---")
            lines.append(r.text(text[:1500]))
        return "\n".join(lines)

    # -- applying a decision --

    def _apply(self, d: dict, obs: Observation, recording: Recording, turn: int):
        action = d["action"]
        if action == "stuck":
            return self._stuck(recording, d.get("thought") or "the model is stuck", turn)
        if action == "done":
            return self._done(d, obs, recording)
        if action == "extract":
            return self._extract(d, recording, turn)

        act = Action(action)
        target = None
        control = d.get("value") or ""
        if act in (Action.click, Action.fill, Action.select):
            element = obs.element(d.get("element") or -1)
            if element is None:
                raise _Problem(f"element {d.get('element')} is not on the current screen")
            try:
                target, control = self.surface.locators_for(element.id)
            except TargetNotFound as e:
                raise _Problem(f"element {element.id} cannot be located reliably: {e}")
        value = d.get("value")
        if act in (Action.fill, Action.select, Action.press) and value is None:
            raise _Problem(f"{action} needs a value")
        if act == Action.select and target is not None:
            options = obs.element(d["element"]).options
            rendered = render(value, self.inputs)
            if options and rendered not in options:
                raise _Problem(f"{rendered!r} is not one of the options {options}")
        try:
            self.session.assert_automation()
            resolved = self.surface.resolve(target, 3_000) if target else None
            try:
                risk = self.surface.perform(act, resolved, value, None, d.get("step_name") or "", inputs=self.inputs)
            except NeedsHuman as e:
                if not self._approve(d, target, str(e), recording, turn):
                    return DiscoveryResult("denied", recording, f"operator denied: {e}")
                risk = self.surface.perform(act, resolved, value, None, d.get("step_name") or "",
                                            inputs=self.inputs)
        except PolicyViolation as e:
            raise _Problem(f"not allowed: {e}")
        except KeyError as e:
            raise _Problem(f"bad template: {e}")
        except TargetNotFound as e:
            raise _Problem(str(e))

        recording.steps.append(RecordedStep(
            action=act, target=target, value=value, risk=risk, note=d.get("thought", "")[:200],
            name=d.get("step_name") or ""))
        self.history.append(Turn(turn, f"{action} {target.description if target else ''} "
                                       f"{('value=' + value) if value else ''} -> ok".strip()))
        self.log.event("acted", turn=turn, action=action, control=control, risk=risk.value, value=value,
                       locators=[s.by for s in target.strategies] if target else None)
        return None

    def _extract(self, d: dict, recording: Recording, turn: int):
        name = re.sub(r"[^a-z0-9_]", "_", (d.get("output") or "").lower()).strip("_")
        shown = (d.get("value") or "").strip()
        if not name:
            raise _Problem('extract needs "output": a snake_case name for the value')
        if not shown:
            shown = self._value_from_thought(d.get("thought") or "", d.get("output_type"), recording)
            if not shown:
                raise _Problem('extract needs "value": copy the value exactly as it appears on screen, '
                               'e.g. "value": "$1,234.56"')
            self.log.event("value_inferred", turn=turn, output=name,
                           reason="model omitted value; one candidate in its thought is shown exactly once")
        try:
            type_ = ValueType(d.get("output_type") or "string")
        except ValueError:
            type_ = ValueType.string
        try:
            target, _, pattern = self.surface.locators_for_value(shown)
            read = self.surface.read(self.surface.resolve(target, 2_000))
        except TargetNotFound as e:
            raise _Problem(f"cannot locate {shown!r} on screen: {e}")
        spec = OutputSpec(type=type_, description=f"read during discovery for: {name}",
                          sensitivity=Sensitivity.pii, pattern=pattern)
        try:
            spec.parse(read)
        except ValueError as e:
            raise _Problem(f"{read!r} is not a {type_.value}: {e}")
        target = target.model_copy(update={"description": f"value for {name}"})
        recording.outputs[name] = spec
        self.redactor.know(read, f"pii:{name}")
        recording.steps.append(RecordedStep(action=Action.extract, target=target, output=name,
                                            note=d.get("thought", "")[:200], name=f"read_{name}"))
        self.history.append(Turn(turn, f"extract {name} -> ok"))
        self.log.event("extracted", turn=turn, output=name, type=type_.value,
                       locators=[s.by for s in target.strategies])
        return None

    def _value_from_thought(self, thought: str, output_type: str | None, recording: Recording) -> str:
        """A guarded fallback for a model that names the value in prose but
        leaves `value` empty: accept a candidate only if it fits the declared
        type, isn't one of our own parameters, and is the *only* such token in
        the thought that appears exactly once on screen."""
        patterns = {"money": r"\$\s?[\d,]+(?:\.\d{2})?", "date": r"\b\d{4}-\d{2}-\d{2}\b",
                    "integer": r"\b\d[\d,]*\b", "decimal": r"\b\d[\d,]*\.\d+\b"}
        pattern = patterns.get(output_type or "", "|".join(patterns.values()))
        own = {p.value for p in recording.params.values()}
        tokens = [t.strip() for t in re.findall(pattern, thought) if t.strip() not in own]
        located = []
        for token in dict.fromkeys(tokens):
            try:
                self.surface.locators_for_value(token)
                located.append(token)
            except TargetNotFound:
                continue
        return located[0] if len(located) == 1 else ""

    def _done(self, d: dict, obs: Observation, recording: Recording):
        # The model proposes; the system checks. Take the first candidate that
        # is on screen and is a heading rather than data: the model's
        # success_text, then its screen heading, then a heading found on screen.
        candidates = [(d.get("success_text") or "").strip(), (d.get("screen") or "").strip()]
        candidates += [text for text, _ in self._headings(obs)]
        rejected = []
        for text in candidates:
            if not text:
                continue
            frame = self._frame_with(text, obs)
            if frame is False:
                rejected.append(f"{text!r} is not on screen")
            elif self.redactor.leaks(text) or re.search(r"\$|\d{3,}", text) or len(text) > 60:
                rejected.append(f"{text!r} is data, not a heading")
            else:
                break
        else:
            raise _Problem("no usable success_text: " + "; ".join(rejected)
                           + ". Give a short heading visible on this screen")
        if rejected:
            self.log.event("success_text_substituted", used=text, rejected=rejected)
        recording.success.append((text, frame))
        # If an identifying parameter is shown on the final screen, assert it
        # too: we reached *this* member's page, not just a member page.
        for name, p in recording.params.items():
            if p.sensitivity in (Sensitivity.public, Sensitivity.internal) and len(p.value) >= 3:
                found = self._frame_with(p.value, obs)
                if found is not False:
                    recording.success.append((f"{{{{inputs.{name}}}}}", found))
        self.log.event("done", success=[t for t, _ in recording.success])
        return DiscoveryResult("recorded", recording, "goal reached")

    def _frame_with(self, text: str, obs: Observation) -> FrameRef | None | Literal[False]:
        """The frame showing `text` (None = top document), or False if none does."""
        for label, body in obs.text.items():
            if text and text in body:
                return None if label == "top" else FrameRef(name=label)
        return False

    def _attach_checkpoint(self, recording: Recording, obs: Observation, before: str,
                           proposed: str | None) -> None:
        """Give the previous step a checkpoint: a heading of the screen it led to.

        The model proposes one (its "screen" field); the system accepts it only
        if it is really on screen, not sensitive, not a number — and preferably
        new since the step, so it proves the step changed something. Otherwise
        a capitalised panel title is found heuristically.
        """
        if not recording.steps:
            return
        last = recording.steps[-1]
        if last.screen_after or last.action not in (Action.click, Action.press):
            return
        candidates = self._headings(obs)
        proposed = (proposed or "").strip()
        frame = self._frame_with(proposed, obs) if proposed else False
        if (frame is not False and 4 <= len(proposed) <= 60 and not re.search(r"\d{3,}|\$", proposed)
                and not self.redactor.leaks(proposed)):
            candidates.insert(0, (proposed, frame))
        fresh = [c for c in candidates if c[0] not in before]
        pick = (fresh or candidates or [None])[0]
        if pick:
            last.screen_after, last.screen_frame = pick
            self.log.event("checkpoint_recorded", step=last.name, text=pick[0], fresh=bool(fresh))

    def _headings(self, obs: Observation) -> list[tuple[str, FrameRef | None]]:
        """Short, stable-looking lines: capitalised panel titles and the like."""
        found = []
        for label, body in obs.text.items():
            if label in ("banner", "nav"):
                continue
            for line in body.splitlines():
                line = line.strip()
                if (4 <= len(line) <= 48 and line.upper() == line and re.search(r"[A-Z]{3}", line)
                        and not re.search(r"\d{3,}|\$", line) and not self.redactor.leaks(line)):
                    found.append((line, None if label == "top" else FrameRef(name=label)))
        return found

    # -- humans --

    def _approve(self, d: dict, target: Target | None, message: str, recording: Recording, turn: int) -> bool:
        evidence = self.log.capture(self.surface, f"approval-turn{turn}")
        intervention = Intervention(
            run_id=self.log.run_id, reason="approval_required", message=message, goal=recording.goal,
            step_id=d.get("step_name"), step_description=target.description if target else d["action"],
            location=self.surface.location(), screenshot=evidence[0] if evidence else None,
            cdp_endpoint=getattr(self.surface, "cdp_endpoint", None), options=["approve", "deny"])
        self.log.event("escalated", reason="approval_required", step=d.get("step_name"), message=message)
        resolution = self.escalator.escalate(intervention, self.session)
        self.log.event("handoff_resolved", action=resolution.action if resolution else None,
                       by=resolution.by if resolution else None)
        if resolution and resolution.action == "approve":
            self.surface.guard.granted.add(d.get("step_name") or "")
            return True
        return False

    def _stuck(self, recording: Recording, why: str, turn: int) -> DiscoveryResult | None:
        """Hand the live session to a human. Their actions become recorded steps."""
        evidence = self.log.capture(self.surface, f"stuck-turn{turn}", aria=True)
        intervention = Intervention(
            run_id=self.log.run_id, reason="stuck", message=why, goal=recording.goal,
            location=self.surface.location(), screenshot=evidence[0] if evidence else None,
            cdp_endpoint=getattr(self.surface, "cdp_endpoint", None), options=["done", "abort"])
        self.log.event("escalated", reason="stuck", message=why, screenshot=intervention.screenshot)
        resolution = self.escalator.escalate(intervention, self.session)
        if resolution is None or resolution.action == "abort":
            self.log.event("handoff_resolved", action="abort", by=resolution.by if resolution else None)
            return DiscoveryResult("stuck", recording, why, turns=turn)
        human_steps = [s for s in (self._from_human(e) for e in resolution.human_events) if s]
        recording.steps.extend(human_steps)
        self.history.append(Turn(turn, f"a human operator took over and did {len(human_steps)} action(s): "
                                       + "; ".join(f"{s.action.value} {s.target.description}" for s in human_steps)))
        self.log.event("handoff_resolved", action=resolution.action, by=resolution.by, note=resolution.note,
                       human_events=resolution.human_events, recorded_steps=len(human_steps))
        return None

    def _from_human(self, event: dict) -> RecordedStep | None:
        """A captured operator action as a step, located the way they'd describe it."""
        strategies: list = []
        if event.get("name") and event.get("role") not in ("generic", "cell"):
            strategies.append(RoleLocator(role=event["role"], name=event["name"]))
        if event.get("label"):
            strategies.append(LabelNearLocator(label=event["label"], role=event.get("role") or "textbox"))
        if event.get("kind") == "click" and event.get("role") == "link" and event.get("name"):
            strategies.append(TextLocator(text=event["name"]))
        if not strategies:
            return None
        frame = FrameRef(name=event["frame"]) if event.get("frame") not in (None, "", "top") else None
        target = Target(description=f"{event.get('role')} {event.get('name') or event.get('label')!r}",
                        frame=frame, strategies=strategies)
        if event.get("kind") == "change":
            return RecordedStep(action=Action.fill, target=target, value=event.get("value", ""),
                                risk=Risk.input, by_human=True, name=f"enter_{event.get('label') or 'value'}")
        return RecordedStep(action=Action.click, target=target, by_human=True,
                            risk=self.surface.guard.classify(Action.click, event.get("name") or ""),
                            name=f"click_{event.get('name') or 'control'}")

    def _end(self, result: DiscoveryResult) -> DiscoveryResult:
        self.log.capture(self.surface, f"discovery-{result.status}", aria=True)
        self.log.event("discovery_end", status=result.status, message=result.message, turns=result.turns,
                       steps=len(result.recording.steps), outputs=list(result.recording.outputs))
        return result


class _Problem(Exception):
    """The model's chosen action can't be carried out; told to it next turn."""
