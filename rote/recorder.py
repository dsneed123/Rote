"""Compile a discovery trace into a capability artifact.

The discovery run produces a Recording: what was done, to which verified
targets, what each screen turned out to be, which values were read. This
module turns it into a Capability — and that is where the run's specifics are
removed:

  * values typed from parameters become {{inputs.name}} templates
  * checkpoints that happen to contain a parameter value are templated too
  * everything is linted for PII and secrets before it is allowed to be saved
  * the model transcript stays in the run evidence; the artifact only links to
    it by run id (provenance), so the capability is reviewable on its own

It is a pure function of the Recording, so it is tested without a browser
or a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .policy import Redactor
from .schema import (Action, AppProfile, AppRef, Capability, Disposition, FrameRef, InputSpec,
                     OutcomeSpec, OutputSpec, Provenance, Risk, Sensitivity, Step, Target,
                     TextCheck, ValueType)


@dataclass
class Param:
    """A discovery-time parameter: an example value plus its contract."""

    value: str
    type: ValueType = ValueType.string
    sensitivity: Sensitivity = Sensitivity.internal
    description: str = ""
    pattern: str | None = None
    values: list[str] | None = None


@dataclass
class RecordedStep:
    action: Action
    target: Target | None = None
    value: str | None = None            # as typed by the agent; may already be a template
    risk: Risk = Risk.read
    note: str = ""
    name: str = ""                      # proposed step id
    output: str | None = None
    by_human: bool = False
    screen_after: str | None = None     # verified text identifying the screen this step led to
    screen_frame: FrameRef | None = None


@dataclass
class Recording:
    goal: str
    product: str
    entry: str
    params: dict[str, Param]
    steps: list[RecordedStep] = field(default_factory=list)
    outputs: dict[str, OutputSpec] = field(default_factory=dict)
    success: list[tuple[str, FrameRef | None]] = field(default_factory=list)
    run_id: str | None = None
    model: str | None = None
    tenant: str | None = None


class RecordingRejected(ValueError):
    """The recording would put sensitive data into an artifact."""


def snake(text: str, fallback: str = "step") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"{fallback}_{slug}".strip("_")
    return slug[:48]


def templatize(text: str, params: dict[str, Param]) -> str:
    """Replace parameter example values with {{inputs.name}}. Longest first,
    so a value that contains another isn't half-replaced."""
    for name, param in sorted(params.items(), key=lambda kv: len(kv[1].value), reverse=True):
        if param.value and len(param.value) >= 2:
            text = text.replace(param.value, f"{{{{inputs.{name}}}}}")
    return text


def _step_id(step: RecordedStep, taken: set[str]) -> str:
    base = snake(step.name) if step.name else ""
    if not base:
        control = step.target.description if step.target else step.value or ""
        base = snake(f"{step.action.value}_{control}")
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    taken.add(candidate)
    return candidate


def compile_capability(recording: Recording, name: str, profile: AppProfile, redactor: Redactor,
                       version: str = "1.0.0", title: str | None = None) -> tuple[Capability, list[str]]:
    """Build the capability, and the review notes a human should read first."""
    notes: list[str] = []
    params = recording.params
    # Lint with the example values known: if one survives templating into an
    # artifact, it is reported by name.
    for pname, param in params.items():
        if param.sensitivity in (Sensitivity.pii, Sensitivity.secret):
            redactor.know(param.value, f"{param.sensitivity.value}:{pname}")

    steps: list[Step] = []
    taken: set[str] = set()
    for recorded in recording.steps:
        value = templatize(recorded.value, params) if recorded.value is not None else None
        expect = []
        if recorded.screen_after:
            expect.append(TextCheck(text=templatize(recorded.screen_after, params),
                                    frame=recorded.screen_frame))
        note = recorded.note
        if recorded.by_human:
            note = f"performed by an operator during discovery; verify on first replay. {note}".strip()
            notes.append(f"step {len(steps) + 1} was done by a human; its locator is unverified")
        if recorded.target and recorded.target.strategies[0].by == "css":
            notes.append(f"step {len(steps) + 1} relies on a css selector; consider a better anchor")
        if value is not None and recorded.action in (Action.fill, Action.select) and "{{" not in value:
            notes.append(f"step {len(steps) + 1} types a literal {value!r}; should it be an input?")
        steps.append(Step(
            id=_step_id(recorded, taken), action=recorded.action, target=recorded.target,
            value=value, output=recorded.output, risk=recorded.risk, expect=expect, note=note,
        ))

    success = [TextCheck(text=templatize(text, params), frame=frame) for text, frame in recording.success]
    if not success:
        raise RecordingRejected("no success checkpoint was verified; refusing to record")

    inputs = {
        pname: InputSpec(type=p.type, description=p.description, pattern=p.pattern, values=p.values,
                         sensitivity=p.sensitivity,
                         example=p.value if p.sensitivity in (Sensitivity.public, Sensitivity.internal) else None)
        for pname, p in params.items()
    }
    # Only parameters the flow actually uses become inputs.
    used = set(re.findall(r"inputs\.([a-z_][a-z0-9_]*)",
                          " ".join([s.value or "" for s in steps]
                                   + [c.text for s in steps for c in s.expect if isinstance(c, TextCheck)]
                                   + [c.text for c in success])))
    for unused in sorted(set(inputs) - used):
        notes.append(f"parameter {unused!r} was never used by the flow; dropped")
        inputs.pop(unused)

    commits = any(s.risk == Risk.commit for s in steps)
    capability = Capability(
        name=name, version=version, title=title or recording.goal[:80],
        description=f"Recorded from the goal: {recording.goal}",
        app=AppRef(product=recording.product, entry=recording.entry),
        side_effects="irreversible" if commits else "none",
        inputs=inputs, outputs=recording.outputs,
        # Every business outcome the product can produce is declared; the
        # reviewer prunes the ones that can't happen on this flow. Declaring
        # too many costs a line of YAML; too few surprises a caller.
        outcomes=[OutcomeSpec(code=d.code, description=d.description)
                  for d in profile.detectors if d.disposition == Disposition.business],
        steps=steps, success=success,
        provenance=Provenance(
            run_id=recording.run_id, model=recording.model, tenant=recording.tenant, goal=recording.goal,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            human_steps=sum(1 for s in recording.steps if s.by_human)),
    )

    leaks = _lint(capability, redactor)
    if leaks:
        raise RecordingRejected("sensitive data would be stored in the artifact: " + "; ".join(leaks))
    return capability, notes


def _lint(capability: Capability, redactor: Redactor) -> list[str]:
    """Everything a human can read in the artifact, checked for PII and secrets."""
    texts: list[str] = [capability.title, capability.description]
    for step in capability.steps:
        texts += [step.value or "", step.note, step.target.description if step.target else ""]
        if step.target:
            texts += [str(s.model_dump()) for s in step.target.strategies]
        texts += [c.text for c in step.expect if isinstance(c, TextCheck)]
    texts += [c.text for c in capability.success if isinstance(c, TextCheck)]
    texts += [s.example or "" for s in capability.inputs.values()]
    problems = []
    for text in texts:
        problems += [f"{label} in {redactor.text(text)[:60]!r}" for label in redactor.leaks(text)]
    return sorted(set(problems))
