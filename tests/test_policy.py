"""Guardrails: the URL allowlist, risk levels that need a human or approval,
and redaction of sensitive values before anything is logged."""

import pytest

from rote.policy import Guard, NeedsHuman, Policy, PolicyViolation, Redactor
from rote.schema import Action, Risk
from rote.store import LIBRARY

BASE = "http://127.0.0.1:8765"


@pytest.fixture
def policy():
    return Policy.load(LIBRARY / "policies" / "corelink-teller.yaml").for_base_url(BASE)


def guard(policy, mode="replay", approved=False):
    return Guard(policy, mode=mode, approved=approved)


def test_allowlist(policy):
    g = guard(policy)
    assert g.url_allowed(f"{BASE}/member?m=1")
    assert not g.url_allowed("https://evil.example/main")
    assert not g.url_allowed(f"{BASE}/__control/faults")      # blocked path
    assert not g.url_allowed(f"{BASE}/admin")                  # not on allowed list
    with pytest.raises(PolicyViolation):
        g.check_url("http://127.0.0.1:9999/main")              # other port, other origin


@pytest.mark.parametrize("action,control,want", [
    (Action.click, "Post", Risk.commit),
    (Action.click, "Inquire", Risk.submit),
    (Action.click, "Member Inquiry", Risk.read),
    (Action.click, "Open Share", Risk.read),
    (Action.fill, "Member Number", Risk.input),
    (Action.press, "Enter", Risk.submit),
    (Action.extract, "Balance", Risk.read),
])
def test_classify(policy, action, control, want):
    assert guard(policy).classify(action, control) == want


def test_declared_risk_cannot_downgrade(policy):
    assert guard(policy).classify(Action.click, "Post", declared=Risk.read) == Risk.commit


def test_commit_gate(policy):
    with pytest.raises(NeedsHuman):
        guard(policy, mode="discovery").check(Action.click, "Post")
    with pytest.raises(NeedsHuman):
        guard(policy, approved=False).check(Action.click, "Post")
    assert guard(policy, approved=True).check(Action.click, "Post") == Risk.commit
    g = guard(policy, mode="discovery")
    g.granted.add("post_share")
    assert g.check(Action.click, "Post", step_id="post_share") == Risk.commit


def test_blocked_controls_and_actions(policy):
    with pytest.raises(PolicyViolation):
        guard(policy).check(Action.click, "Close Account")
    narrow = policy.model_copy(update={"allowed_actions": [Action.click]})
    with pytest.raises(PolicyViolation):
        guard(narrow).check(Action.fill, "x")


def test_redaction():
    r = Redactor()
    r.know("hunter2-pass", "secret:password")
    out = r.text("SSN 900-12-3456 pw hunter2-pass bal $1,234.56 dob 1984-03-02 "
                 "card 4111 1111 1111 1111 id 1234567890123 jo@x.org")
    assert "900-12-3456" not in out and "<ssn>" in out
    assert "hunter2-pass" not in out and "<secret:password>" in out
    assert "<amount>" in out and "<date>" in out and "<card>" in out and "<email>" in out
    assert "1234567890123" in out          # not Luhn-valid: not a card, left alone


def test_redact_nested():
    r = Redactor(money=False)
    assert r.obj({"a": ["900-12-3456"], "b": 3}) == {"a": ["<ssn>"], "b": 3}


def test_leaks_for_lint():
    r = Redactor()
    r.know("JANE Q SAMPLE", "pii:name")
    assert r.leaks("checkpoint: JANE Q SAMPLE") == ["pii:name"]
    assert r.leaks("MEMBER INQUIRY - DETAIL") == []
