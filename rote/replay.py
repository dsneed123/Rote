"""Deterministic replay: the production path an agent invokes.

No model is consulted. Given a capability, an app profile and typed inputs,
the engine walks the steps and, at every point where the screen could have
gone somewhere unexpected, asks the profile's detectors what it is looking at:

    resolve target ─▶ act (via policy) ─▶ await checkpoint ─▶ next step
          │                                      │
          └──── no match / timeout ──────────────┴──▶ detect()
                                                        │
                    business ──▶ return business_outcome (not an error)
                    recoverable ──▶ remedy, then resume the same phase
                    failure ──▶ stop: typed failure + evidence
                    nothing known ──▶ escalate to a human, or fail

Two phases matter for recovery. If a condition shows up *before* a step acts
(the notice was already on screen), the step is retried whole. If it shows up
*after* (we clicked Inquire, a notice appeared instead of the detail page),
only the checkpoint is awaited again — repeating the click would be wrong.
Re-authentication restarts the flow from the top, and is refused once a commit
step has run: replaying an irreversible action because the session timed out
is precisely the bug this layer exists to prevent.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from .evidence import RunLog
from .handoff import (ControlLost, Escalator, Intervention, NoEscalation, Resolution,
                      SessionControl)
from .policy import NeedsHuman, PolicyViolation, Redactor
from .schema import (Action, AppProfile, Capability, Check, Detector, Disposition, Event,
                     Failure, FailureKind, OutcomeResult, Risk, RunResult, Sensitivity, Status,
                     Step, coerce)
from .surface import Interrupted, Surface, TargetNotFound


def env_secrets(name: str) -> str:
    """Secrets come from the environment: ROTE_SECRET_<NAME>."""
    key = f"ROTE_SECRET_{name.upper()}"
    value = os.environ.get(key)
    if value is None:
        raise KeyError(f"secret {name!r} not set; export {key}")
    return value


@dataclass
class Detected:
    detector: Detector
    message: str

    def __str__(self) -> str:
        return f"{self.detector.code} ({self.detector.disposition.value})"


class Stop(Exception):
    """Ends the run with this result."""

    def __init__(self, status: Status, *, failure: Failure | None = None,
                 outcome: OutcomeResult | None = None) -> None:
        super().__init__(status.value)
        self.status, self.failure, self.outcome = status, failure, outcome


class Restart(Exception):
    """Re-authenticated; run the flow again from the first step."""


class CheckpointTimeout(Exception):
    def __init__(self, checks: list[Check], observed: str) -> None:
        super().__init__("checkpoint not reached")
        self.checks, self.observed = checks, observed


def detect(surface: Surface, detectors: list[Detector]) -> Detected | None:
    """The first detector whose checks all hold on the current screen."""
    for detector in detectors:
        try:
            if all(surface.holds(check) for check in detector.when):
                message = ""
                if detector.message_regex:
                    match = re.search(detector.message_regex, surface.visible_text())
                    message = (match.group(1) if match and match.groups() else match.group(0) if match else "")
                return Detected(detector, " ".join(message.split()) or detector.description)
        except Exception:
            continue
    return None


def describe_checks(checks: list[Check]) -> str:
    parts = []
    for check in checks:
        data = check.model_dump(exclude_none=True, exclude_defaults=True)
        parts.append(", ".join(f"{k}={v}" for k, v in data.items()) or check.kind)
    return "; ".join(f"[{p}]" for p in parts)


class SignOnFailed(Exception):
    def __init__(self, failure: Failure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def sign_on(surface: Surface, profile: AppProfile, log: RunLog) -> None:
    """Run the product's sign-on steps. Shared by replay and discovery.

    Credentials are {{secrets.x}} templates in the profile, resolved by the
    surface at the moment of typing; they are never seen here.
    """
    log.event("sign_on", steps=len(profile.sign_on))
    for step in profile.sign_on:
        resolved = surface.resolve(step.target, step.timeout_ms) if step.target else None
        surface.perform(step.action, resolved, step.value, step.risk, step.id)
        if step.expect:
            deadline = time.monotonic() + step.timeout_ms / 1000
            while not all(surface.holds(c) for c in step.expect):
                if time.monotonic() > deadline:
                    raise SignOnFailed(Failure(
                        kind=FailureKind.checkpoint_failed, step_id=f"sign_on.{step.id}",
                        message="sign-on did not complete", expected=describe_checks(step.expect),
                        evidence=log.capture(surface, "sign-on-failed", aria=True)))
                time.sleep(0.2)
    if profile.signed_in and not all(surface.holds(c) for c in profile.signed_in):
        found = detect(surface, profile.detectors)
        raise SignOnFailed(Failure(
            kind=FailureKind.checkpoint_failed, step_id="sign_on",
            message=f"not signed in after sign-on ({found or 'unknown screen'})",
            evidence=log.capture(surface, "sign-on-failed", aria=True)))


def dismiss(surface: Surface, detector: Detector) -> None:
    """Apply a click or wait remedy. (Re-authentication is the caller's: what to
    do afterwards — restart a flow, reset a recording — depends on who's asking.)"""
    recovery = detector.recover
    if recovery.kind == "click":
        resolved = surface.resolve(recovery.target, 5_000)
        surface.perform(Action.click, resolved, None, Risk.submit, f"recover_{detector.code}")
    elif recovery.kind == "wait_retry":
        time.sleep(recovery.backoff_ms / 1000)


class Replayer:
    def __init__(self, surface: Surface, profile: AppProfile, log: RunLog, redactor: Redactor, *,
                 session: SessionControl | None = None, escalator: Escalator | None = None,
                 escalate_on: tuple[str, ...] = ("unrecognized_state", "target_not_found",
                                                 "checkpoint_failed", "recovery_exhausted"),
                 escalation_timeout_s: float = 900, max_duration_s: float = 300,
                 before_step: Callable[[int, Step], None] | None = None,
                 max_interventions_per_step: int = 2) -> None:
        self.surface = surface
        self.profile = profile
        self.log = log
        self.redactor = redactor
        self.session = session or SessionControl()
        self.escalator = escalator or NoEscalation()
        self.escalate_on = escalate_on
        self.escalation_timeout_s = escalation_timeout_s
        self.max_duration_s = max_duration_s
        # Test and demo hook: runs before each step (used to inject faults mid-run).
        self.before_step = before_step
        self.max_interventions_per_step = max_interventions_per_step

    # -- public --

    def run(self, capability: Capability, inputs: dict[str, Any], tenant: str | None = None) -> RunResult:
        started = time.monotonic()
        self.events: list[Event] = []
        self.outputs: dict[str, Any] = {}
        self.attempts: dict[str, int] = {}
        self.escalations: dict[str, int] = {}
        self.committed = False
        self.capability = capability
        self.done_steps = 0
        self._deadline = started + self.max_duration_s

        result = RunResult(run_id=self.log.run_id, capability=capability.name,
                           version=capability.version, tenant=tenant, status=Status.failed,
                           evidence_dir=self.log.relative(self.log.dir))
        self.log.event("run_start", capability=capability.name, version=capability.version,
                       fingerprint=capability.fingerprint(), approved=capability.approved,
                       tenant=tenant, inputs=self._loggable_inputs(capability, inputs))
        try:
            try:
                self.inputs = capability.validate_inputs(inputs)
            except ValueError as e:
                raise Stop(Status.failed, failure=Failure(kind=FailureKind.invalid_input, message=str(e)))
            for name, spec in capability.inputs.items():
                if spec.sensitivity in (Sensitivity.pii, Sensitivity.secret) and name in self.inputs:
                    self.redactor.know(self.inputs[name], f"{spec.sensitivity.value}:{name}")
            self._run_flow(capability)
            result.status = Status.succeeded
            result.outputs = dict(self.outputs)
        except Stop as stop:
            result.status, result.failure, result.outcome = stop.status, stop.failure, stop.outcome
        except Exception as e:   # the surface itself broke; still a structured result
            result.failure = Failure(kind=FailureKind.surface_error, message=f"{type(e).__name__}: {e}",
                                     evidence=self.log.capture(self.surface, "surface-error", aria=True))
        result.events = self.events
        result.steps_completed = self.done_steps
        result.duration_ms = int((time.monotonic() - started) * 1000)
        self._finish(capability, result)
        return result

    # -- the flow --

    def _run_flow(self, capability: Capability) -> None:
        restarts = 0
        while True:
            try:
                self.outputs.clear()
                self.surface.navigate(capability.app.entry)
                self.log.event("navigate", path=capability.app.entry, location=self.surface.location())
                self._clear_entry_conditions()
                for index, step in enumerate(capability.steps):
                    self._step(index, step)
                    self.done_steps = index + 1
                self._verify_success(capability)
                return
            except Restart:
                restarts += 1
                if restarts > 2:
                    raise Stop(Status.failed, failure=Failure(
                        kind=FailureKind.recovery_exhausted, message="re-authenticated repeatedly; giving up"))

    def _clear_entry_conditions(self) -> None:
        """On arrival, handle what's in the way (signed out, a notice) before step 1."""
        for _ in range(3):
            found = detect(self.surface, self.profile.detectors)
            if not found or found.detector.disposition != Disposition.recoverable:
                return
            self._recover(found, step=None, phase="entry")

    def _step(self, index: int, step: Step) -> None:
        if self.before_step:
            self.before_step(index, step)
        self.log.event("step_start", index=index, step=step.id, action=step.action.value,
                       target=step.target.description if step.target else None, note=step.note)
        phase = "act"
        while True:
            self._check_deadline(step)
            try:
                if phase == "act":
                    self._act(index, step)
                    phase = "expect"
                self._await(step.expect, step.timeout_ms, step)
                self.log.event("step_ok", index=index, step=step.id)
                return
            except Interrupted as stop:
                self._recover(stop.detected, step, phase)
            except TargetNotFound as e:
                self._unexpected(index, step, FailureKind.target_not_found, str(e),
                                 expected=step.target.description if step.target else "", phase=phase)
                phase = self._after_human(step)
            except CheckpointTimeout as e:
                self._unexpected(index, step, FailureKind.checkpoint_failed,
                                 f"after {step.action.value}, the expected screen did not appear",
                                 expected=describe_checks(e.checks), observed=e.observed, phase=phase)
                phase = self._after_human(step)

    def _act(self, index: int, step: Step) -> None:
        self.session.assert_automation()
        resolved = None
        if step.target:
            resolved = self.surface.resolve(step.target, step.timeout_ms, interrupt=self._interrupt)
            self.log.event("resolved", step=step.id, strategy=resolved.strategy.by,
                           strategy_index=resolved.strategy_index, control=resolved.control)
            if resolved.strategy_index > 0:
                detail = (f"primary locator failed; resolved with fallback #{resolved.strategy_index} "
                          f"({resolved.strategy.by})")
                self.events.append(Event(kind="drift", step_id=step.id, detail=detail))
                self.log.event("drift", step=step.id, detail=detail)

        if step.action == Action.extract:
            self._extract(step, resolved)
            return

        try:
            risk = self.surface.perform(step.action, resolved, step.value, step.risk, step.id,
                                        inputs=self.inputs)
        except NeedsHuman as e:
            self._approval(index, step, str(e))
            risk = self.surface.perform(step.action, resolved, step.value, step.risk, step.id,
                                        inputs=self.inputs)
        except PolicyViolation as e:
            raise Stop(Status.failed, failure=Failure(
                kind=FailureKind.policy_violation, message=str(e), step_id=step.id, step_index=index,
                evidence=self.log.capture(self.surface, f"policy-{step.id}")))
        if risk == Risk.commit:
            self.committed = True
        self.log.event("acted", step=step.id, action=step.action.value, risk=risk.value,
                       value=self._loggable_value(step.value))

    def _extract(self, step: Step, resolved) -> None:
        spec = self.capability.outputs[step.output]
        raw = self.surface.read(resolved)
        try:
            value = spec.parse(raw)
        except ValueError as e:
            raise Stop(Status.failed, failure=Failure(
                kind=FailureKind.extraction_failed, message=str(e), step_id=step.id,
                expected=spec.type.value, observed=self.redactor.text(raw),
                evidence=self.log.capture(self.surface, f"extract-{step.id}", aria=True)))
        self.outputs[step.output] = value
        if spec.sensitivity in (Sensitivity.pii, Sensitivity.secret):
            self.redactor.know(raw, f"{spec.sensitivity.value}:{step.output}")
            self.redactor.know(value, f"{spec.sensitivity.value}:{step.output}")
        self.log.event("extracted", step=step.id, output=step.output, type=spec.type.value,
                       value=f"<{spec.sensitivity.value}:{step.output}>"
                       if spec.sensitivity in (Sensitivity.pii, Sensitivity.secret) else value)

    def _await(self, checks: list[Check], timeout_ms: int, step: Step | None) -> None:
        """Poll until every check holds, a known condition appears, or time is up."""
        if not checks:
            found = detect(self.surface, self.profile.detectors)
            if found:
                raise Interrupted(found)
            return
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            if all(self.surface.holds(c, self.inputs) for c in checks):
                self.log.event("checkpoint_ok", step=step.id if step else "success",
                               checks=describe_checks(checks))
                return
            found = self._interrupt()
            if found:
                raise Interrupted(found)
            if time.monotonic() >= deadline:
                observed = self.redactor.text(" | ".join(
                    f"{k}={v}" for k, v in self.surface.location().items()))
                raise CheckpointTimeout(checks, observed)
            time.sleep(0.2)

    def _interrupt(self) -> Detected | None:
        return detect(self.surface, self.profile.detectors)

    def _verify_success(self, capability: Capability) -> None:
        try:
            self._await(capability.success, 5_000, None)
        except CheckpointTimeout as e:
            raise Stop(Status.failed, failure=Failure(
                kind=FailureKind.checkpoint_failed, message="final success checkpoint did not hold",
                step_id="success", expected=describe_checks(e.checks), observed=e.observed,
                evidence=self.log.capture(self.surface, "success-failed", aria=True)))
        except Interrupted as stop:
            self._recover(stop.detected, None, "success")
        missing = [name for name in capability.outputs if name not in self.outputs]
        if missing:
            raise Stop(Status.failed, failure=Failure(
                kind=FailureKind.extraction_failed, message=f"outputs not produced: {missing}"))

    # -- runtime conditions --

    def _recover(self, found: Detected, step: Step | None, phase: str) -> None:
        detector = found.detector
        where = step.id if step else phase
        self.log.event("detected", code=detector.code, disposition=detector.disposition.value,
                       step=where, message=found.message)

        if detector.disposition == Disposition.business:
            declared = {o.code for o in self.capability.outcomes}
            if detector.code not in declared:
                self.events.append(Event(kind="drift", step_id=where,
                                         detail=f"business outcome {detector.code} not declared by capability"))
            evidence = self.log.capture(self.surface, f"outcome-{detector.code}")
            self.log.event("business_outcome", code=detector.code, evidence=evidence)
            raise Stop(Status.business_outcome, outcome=OutcomeResult(
                code=detector.code, message=self.redactor.text(found.message), step_id=where))

        if detector.disposition == Disposition.failure:
            raise Stop(Status.failed, failure=Failure(
                kind=FailureKind.app_error, message=f"{detector.code}: {self.redactor.text(found.message)}",
                step_id=where, retryable=detector.retryable,
                observed=self.redactor.text(found.message),
                evidence=self.log.capture(self.surface, f"app-error-{detector.code}", aria=True)))

        # recoverable
        recovery = detector.recover
        count = self.attempts.get(detector.code, 0) + 1
        self.attempts[detector.code] = count
        if count > recovery.max_attempts:
            self._unexpected(None, step, FailureKind.recovery_exhausted,
                             f"{detector.code} recurred {count} times", expected="", phase=phase)
            return

        if recovery.kind in ("click", "wait_retry"):
            dismiss(self.surface, detector)
        elif recovery.kind == "reauthenticate":
            if self.committed:
                raise Stop(Status.failed, failure=Failure(
                    kind=FailureKind.recovery_exhausted, step_id=where,
                    message=f"{detector.code} after a commit step; restarting could repeat it",
                    evidence=self.log.capture(self.surface, "reauth-refused")))
            self._sign_on()
            if phase == "entry":
                self.surface.navigate(self.capability.app.entry)
        # Signing on at the start of a run is establishing the session, not
        # recovering from anything; only report it when it interrupted a flow.
        if not (phase == "entry" and recovery.kind == "reauthenticate"):
            self.events.append(Event(kind="recovered", step_id=where,
                                     detail=f"{detector.code} via {recovery.kind}"))
        self.log.event("recovered", code=detector.code, via=recovery.kind, step=where, attempt=count)
        if recovery.kind == "reauthenticate" and phase != "entry":
            raise Restart()

    def _sign_on(self) -> None:
        try:
            sign_on(self.surface, self.profile, self.log)
        except SignOnFailed as e:
            raise Stop(Status.failed, failure=e.failure)

    # -- the unexpected: escalate or fail --

    def _unexpected(self, index: int | None, step: Step | None, kind: FailureKind, message: str,
                    expected: str = "", observed: str | None = None, phase: str = "") -> None:
        """Nothing known explains the screen. Ask a human, or stop with evidence."""
        evidence = self.log.capture(self.surface, f"{kind.value}-{step.id if step else 'run'}", aria=True)
        observed = observed or self.redactor.text(" | ".join(
            f"{k}={v}" for k, v in self.surface.location().items()))
        failure = Failure(kind=kind, message=message, step_id=step.id if step else None,
                          step_index=index, expected=expected, observed=observed, evidence=evidence)
        self.log.event("unexpected", kind=kind.value, step=step.id if step else None,
                       message=message, expected=expected, observed=observed, evidence=evidence)
        if kind.value not in self.escalate_on or step is None:
            raise Stop(Status.failed, failure=failure)
        # A person said "done" or "retry" and the step still doesn't verify:
        # asking again forever helps no one. Two interventions per step, then stop.
        asked = self.escalations.get(step.id, 0)
        if asked >= self.max_interventions_per_step:
            raise Stop(Status.escalated, failure=failure.model_copy(update={
                "message": f"{message} (still failing after {asked} intervention(s))"}))
        self.escalations[step.id] = asked + 1
        resolution = self._escalate(Intervention(
            run_id=self.log.run_id, reason=kind.value, message=message,  # type: ignore[arg-type]
            capability=self.capability.name, step_id=step.id, step_index=index,
            step_description=step.target.description if step.target else step.action.value,
            location=self.surface.location(), screenshot=evidence[0] if evidence else None,
            cdp_endpoint=getattr(self.surface, "cdp_endpoint", None),
            options=["retry", "done", "abort"]), failure)
        self._resolution = resolution

    def _after_human(self, step: Step) -> str:
        """Where to pick up after an intervention resolved with retry/done."""
        resolution = getattr(self, "_resolution", None)
        self._resolution = None
        if resolution and resolution.action == "done":
            # The human performed the step. Extraction still has to read the
            # value; anything else only needs its checkpoint verified.
            return "act" if step.action == Action.extract else "expect"
        return "act"

    def _approval(self, index: int, step: Step, message: str) -> None:
        evidence = self.log.capture(self.surface, f"approval-{step.id}")
        failure = Failure(kind=FailureKind.not_approved, message=message, step_id=step.id,
                          step_index=index, evidence=evidence)
        resolution = self._escalate(Intervention(
            run_id=self.log.run_id, reason="approval_required", message=message,
            capability=self.capability.name, step_id=step.id, step_index=index,
            step_description=step.target.description if step.target else step.action.value,
            location=self.surface.location(), screenshot=evidence[0] if evidence else None,
            options=["approve", "deny"]), failure)
        if resolution.action != "approve":
            raise Stop(Status.failed, failure=failure.model_copy(update={
                "kind": FailureKind.human_aborted, "message": f"denied by {resolution.by}: {message}"}))
        self.surface.guard.granted.add(step.id)

    def _escalate(self, intervention: Intervention, failure: Failure) -> Resolution:
        self.log.event("escalated", intervention=intervention.id, reason=intervention.reason,
                       step=intervention.step_id, message=intervention.message,
                       screenshot=intervention.screenshot)
        started = time.monotonic()
        resolution = self.escalator.escalate(intervention, self.session)
        waited = round(time.monotonic() - started, 1)
        if resolution is None:
            self.log.event("handoff_timeout", intervention=intervention.id, waited_s=waited)
            raise Stop(Status.escalated, failure=failure.model_copy(update={
                "kind": FailureKind.escalation_timeout,
                "message": f"no operator resolved intervention {intervention.id} in {waited}s"}))
        self.events.append(Event(kind="handoff", step_id=intervention.step_id,
                                 detail=f"{intervention.reason} -> {resolution.action} by {resolution.by} "
                                        f"after {waited}s, {len(resolution.human_events)} human action(s)"))
        self.log.event("handoff_resolved", intervention=intervention.id, action=resolution.action,
                       by=resolution.by, note=resolution.note, waited_s=waited,
                       human_events=resolution.human_events)
        if resolution.by == "system" and resolution.action == "abort":
            raise Stop(Status.failed, failure=failure)
        if resolution.action == "abort":
            raise Stop(Status.escalated, failure=failure.model_copy(update={
                "kind": FailureKind.human_aborted, "message": f"aborted by {resolution.by}: {failure.message}"}))
        return resolution

    def _check_deadline(self, step: Step) -> None:
        if time.monotonic() > self._deadline:
            raise Stop(Status.failed, failure=Failure(
                kind=FailureKind.timeout, message=f"run exceeded {self.max_duration_s}s", step_id=step.id,
                evidence=self.log.capture(self.surface, "timeout")))

    # -- logging helpers --

    def _loggable_inputs(self, capability: Capability, inputs: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for name, value in inputs.items():
            spec = capability.inputs.get(name)
            if spec and spec.sensitivity in (Sensitivity.pii, Sensitivity.secret):
                out[name] = f"<{spec.sensitivity.value}:{name}>"
            else:
                out[name] = value
        return out

    @staticmethod
    def _loggable_value(value: str | None) -> str | None:
        # Templates are logged, not their values: "{{inputs.member_id}}" says
        # which input went in, and the value is in run_start if it's loggable.
        return value

    def _finish(self, capability: Capability, result: RunResult) -> None:
        if result.status == Status.succeeded:
            self.log.capture(self.surface, "final")
        stored = result.model_dump(mode="json")
        for name, spec in capability.outputs.items():
            if name in stored["outputs"] and spec.sensitivity in (Sensitivity.pii, Sensitivity.secret):
                stored["outputs"][name] = f"<{spec.sensitivity.value}:{spec.type.value}>"
        self.log.write_json("result.json", stored)
        self.log.event("run_end", status=result.status.value,
                       outcome=result.outcome.code if result.outcome else None,
                       failure=result.failure.kind.value if result.failure else None,
                       events=len(result.events), duration_ms=result.duration_ms)
