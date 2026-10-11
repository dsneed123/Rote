"""Replay against the live mock app: every branch of the result contract."""

from pathlib import Path

import pytest
import requests

from mockapp.corelink import serve_in_thread
from rote.evidence import RunLog
from rote.handoff import Resolution, ScriptedEscalator
from rote.policy import Guard, Policy, Redactor
from rote.replay import Replayer
from rote.schema import FailureKind, Overlay, Status, apply_overlay
from rote.store import LIBRARY, Store
from rote.surface import WebSurface

FIX = Path(__file__).parent / "fixtures"
SECRETS = {"corelink_user": "teller01", "corelink_password": "demo-pass-123"}


@pytest.fixture(scope="module")
def app():
    server, url = serve_in_thread(port=0, faults=True)
    yield url
    server.shutdown()


@pytest.fixture(autouse=True)
def reset(app):
    requests.post(f"{app}/__control/faults", json={"reset": True})


def fault(app, **kw):
    requests.post(f"{app}/__control/faults", json=kw)


@pytest.fixture
def run(app, tmp_path):
    """run(capability, inputs, escalator=..., approved=..., before_step=...) -> (result, log dir)"""
    surfaces = []

    def go(cap, inputs, *, escalator=None, approved=None, before_step=None, profile=None, base=None):
        policy = Policy.load(LIBRARY / "policies" / "corelink-teller.yaml").for_base_url(base or app)
        redactor = Redactor()
        guard = Guard(policy, mode="replay", approved=cap.approved if approved is None else approved)
        surface = WebSurface(base or app, guard, redactor, secrets=SECRETS.__getitem__)
        surface.start()
        surfaces.append(surface)
        log = RunLog("t", redactor, root=tmp_path / str(len(surfaces)))
        replayer = Replayer(surface, profile or Store().profile("corelink-teller"), log, redactor,
                            escalator=escalator, before_step=before_step)
        try:
            result = replayer.run(cap, inputs)
        finally:
            log.close()
            surface.close()     # one sync Playwright at a time
        return result, log.dir

    return go


@pytest.fixture
def balance():
    return Store.load_path(FIX / "read_balance.yaml")


@pytest.fixture
def open_share():
    return Store.load_path(FIX / "open_share.yaml")


def test_happy_path(run, balance):
    result, logdir = run(balance, {"member_id": "12345"})
    assert result.status == Status.succeeded, result.failure
    assert result.outputs == {"savings_balance": "1234.56"}
    stored = (logdir / "result.json").read_text() + (logdir / "log.jsonl").read_text()
    assert "1234.56" not in stored and "1,234.56" not in stored      # evidence never has the value
    assert "demo-pass-123" not in stored


@pytest.mark.parametrize("member,code", [("99999", "member_not_found"), ("77777", "access_restricted")])
def test_business_outcomes(run, balance, member, code):
    result, _ = run(balance, {"member_id": member})
    assert result.status == Status.business_outcome
    assert result.outcome.code == code and result.outcome.message
    assert result.failure is None


def test_invalid_input_never_touches_ui(run, balance):
    result, logdir = run(balance, {"member_id": "12a"})
    assert result.failure.kind == FailureKind.invalid_input
    assert "navigate" not in (logdir / "log.jsonl").read_text()


def test_notice_is_dismissed(run, balance, app):
    fault(app, notice_once=True)
    result, _ = run(balance, {"member_id": "12345"})
    assert result.ok
    assert [e.detail for e in result.events] == ["system_notice via click"]


def test_session_expiry_mid_run_reauthenticates(run, balance, app):
    fired = []

    def expire(index, step):
        if step.id == "submit_inquiry" and not fired:   # once; the restart passes here again
            fired.append(step.id)
            fault(app, expire_session=True)
    result, _ = run(balance, {"member_id": "12345"}, before_step=expire)
    assert result.ok and result.outputs["savings_balance"] == "1234.56"
    assert any("session_expired via reauthenticate" in e.detail for e in result.events)


def test_slow_app_is_waited_for(run, balance, app):
    fault(app, slow_ms=700)
    assert run(balance, {"member_id": "12345"})[0].ok


def test_host_error_is_a_retryable_failure(run, balance, app):
    fault(app, error_once=True)
    result, _ = run(balance, {"member_id": "12345"})
    assert result.status == Status.failed
    assert result.failure.kind == FailureKind.app_error and result.failure.retryable
    assert result.failure.step_id == "submit_inquiry" and result.failure.evidence


def test_fallback_locator_is_reported_as_drift(run, balance):
    broken = apply_overlay(balance, Overlay(tenant="x", applies_to=balance.name, steps={
        "submit_inquiry": {"target": {"strategies": [
            {"by": "role", "role": "button", "name": "Lookup"},
            {"by": "role", "role": "button", "name": "Inquire"}]}}}))
    result, _ = run(broken, {"member_id": "12345"})
    assert result.ok and [e.kind for e in result.events] == ["drift"]


def test_unrecognized_screen_fails_with_evidence(run, balance):
    wrong = apply_overlay(balance, Overlay(tenant="x", applies_to=balance.name, steps={
        "submit_inquiry": {"timeout_ms": 1500, "expect": [{"kind": "text", "text": "NEVER SHOWN"}]}}))
    result, logdir = run(wrong, {"member_id": "12345"})
    f = result.failure
    assert f.kind == FailureKind.checkpoint_failed and f.step_id == "submit_inquiry"
    assert "NEVER SHOWN" in f.expected and "main=/member" in f.observed
    assert any(p.endswith(".png") for p in f.evidence) and any(p.endswith(".aria.txt") for p in f.evidence)


def test_unrecognized_screen_escalates_and_human_says_done(run, balance):
    wrong = apply_overlay(balance, Overlay(tenant="x", applies_to=balance.name, steps={
        "submit_inquiry": {"timeout_ms": 1000, "expect": [{"kind": "text", "text": "DETAIL"}],
                           "target": {"strategies": [{"by": "role", "role": "button", "name": "Lookup"}]}}}))
    clicked = []

    def human(intervention, session):
        # The operator finds the button themselves and presses it.
        clicked.append(intervention.step_id)
        return Resolution(action="done", by="tester", note="pressed Inquire myself")

    result, _ = run(wrong, {"member_id": "12345"}, escalator=ScriptedEscalator(human))
    # The human said "done" but never actually clicked: the step's checkpoint
    # is still verified, so it is not blindly trusted.
    assert clicked == ["submit_inquiry", "submit_inquiry"]      # asked twice, then gave up
    assert result.status == Status.escalated
    assert "after 2 intervention" in result.failure.message
    assert [e.kind for e in result.events] == ["handoff", "handoff"]


def test_human_completes_step_over_cdp_and_run_resumes(app, tmp_path):
    """The operator attaches to the *same* browser over CDP, does the step,
    and the run picks up from there; their clicks are captured."""
    from playwright.sync_api import sync_playwright
    import threading

    cap = apply_overlay(Store.load_path(FIX / "read_balance.yaml"), Overlay(
        tenant="x", applies_to="corelink.member.read_savings_balance", steps={
            "submit_inquiry": {"timeout_ms": 800, "target": {"strategies": [
                {"by": "role", "role": "button", "name": "Lookup"}]}}}))
    policy = Policy.load(LIBRARY / "policies" / "corelink-teller.yaml").for_base_url(app)
    redactor = Redactor()
    session_holder = {}

    def operator(intervention, session):
        # Run the human in another thread with its own Playwright: a separate
        # client attached to the live browser, as a real operator tool would be.
        def drive():
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(intervention.cdp_endpoint)
                page = browser.contexts[0].pages[0]
                main = next(f for f in page.frames if f.name == "main")
                main.get_by_role("button", name="Inquire").click()
                main.wait_for_url("**/member*")
        t = threading.Thread(target=drive)
        t.start()
        # The automation thread keeps pumping its own Playwright so the
        # capture binding can deliver the operator's click.
        while t.is_alive():
            surface.page.wait_for_timeout(100)
        return Resolution(action="done", by="cdp-operator")

    from rote.handoff import SessionControl
    session = SessionControl()
    surface = WebSurface(app, Guard(policy, mode="replay"), redactor, cdp_port=9333,
                         secrets=SECRETS.__getitem__, on_human_event=session.record_human_event)
    surface.start()
    try:
        log = RunLog("t", redactor, root=tmp_path)
        result = Replayer(surface, Store().profile("corelink-teller"), log, redactor,
                          session=session, escalator=ScriptedEscalator(operator)).run(cap, {"member_id": "12345"})
    finally:
        surface.close()
    assert result.ok, result.failure
    assert result.outputs == {"savings_balance": "1234.56"}
    handoff = next(e for e in result.events if e.kind == "handoff")
    assert "1 human action" in handoff.detail
    assert '"name": "Inquire"' in (log.dir / "log.jsonl").read_text()


def test_commit_in_draft_needs_approval(run, open_share):
    inputs = {"member_id": "12345", "share_type": "SAVINGS CLUB", "opening_deposit": "25"}
    denied, _ = run(open_share, inputs, escalator=ScriptedEscalator(
        lambda i, s: Resolution(action="deny", by="supervisor")))
    assert denied.failure.kind == FailureKind.human_aborted and denied.failure.step_id == "post_share"

    asked = []
    approved, _ = run(open_share, inputs, escalator=ScriptedEscalator(
        lambda i, s: asked.append(i.reason) or Resolution(action="approve", by="supervisor")))
    assert asked == ["approval_required"]
    assert approved.ok and approved.outputs["confirmation_number"].isdigit()


def test_commit_in_draft_without_operator_fails(run, open_share):
    result, _ = run(open_share, {"member_id": "12345", "share_type": "SAVINGS CLUB",
                                 "opening_deposit": "25"})
    assert result.failure.kind == FailureKind.not_approved


def test_approved_commit_runs_unattended(run, open_share):
    result, _ = run(open_share, {"member_id": "12345", "share_type": "SAVINGS CLUB",
                                 "opening_deposit": "25"}, approved=True)
    assert result.ok


def test_host_validation_is_a_business_outcome(run, open_share):
    result, _ = run(open_share, {"member_id": "12345", "share_type": "MONEY MARKET",
                                 "opening_deposit": "10"}, approved=True)
    assert result.status == Status.business_outcome and result.outcome.code == "validation_error"
    assert "MINIMUM OPENING DEPOSIT" in result.outcome.message

