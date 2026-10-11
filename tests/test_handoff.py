"""Handoff: one owner per live session, and control moving between automation
and the operator through the console's HTTP endpoints."""

import threading
import time

import pytest
import requests

from rote.console import ConsoleEscalator, OperatorConsole
from rote.handoff import ControlLost, Intervention, Owner, Resolution, SessionControl


def test_control_model():
    s = SessionControl()
    s.assert_automation()
    with pytest.raises(RuntimeError):
        s.take()                                   # nothing to take
    s.raise_intervention(Intervention(run_id="r", reason="stuck", message="m"))
    with pytest.raises(ControlLost):
        s.assert_automation()                      # automation may not act now
    s.record_human_event({"kind": "click"})        # not human yet: ignored
    s.take("op")
    s.record_human_event({"kind": "click", "name": "Inquire"})
    with pytest.raises(ValueError):
        s.resolve(Resolution(action="approve"))    # not offered
    s.resolve(Resolution(action="done", by="op"))
    assert s.owner == Owner.automation
    assert [e["name"] for e in s.wait(0).human_events] == ["Inquire"]


def test_console_round_trip():
    session = SessionControl()
    console = OperatorConsole(session, port=0).start()
    escalator = ConsoleEscalator(console, timeout_s=10, notify=lambda m: None)
    out = {}
    worker = threading.Thread(target=lambda: out.update(r=escalator.escalate(
        Intervention(run_id="r", reason="checkpoint_failed", message="no detail page",
                     step_id="submit_inquiry"), session)))
    worker.start()
    try:
        for _ in range(50):
            state = requests.get(f"{console.url}/api/state").json()
            if state["open"]:
                break
            time.sleep(0.05)
        assert state["owner"] == "awaiting_human" and state["open"]["step_id"] == "submit_inquiry"
        assert requests.post(f"{console.url}/resolve", json={"action": "done"}).status_code == 409
        requests.post(f"{console.url}/take")
        session.record_human_event({"kind": "click", "name": "Inquire"})
        requests.post(f"{console.url}/resolve", json={"action": "done", "note": "clicked it"})
        worker.join(5)
        assert out["r"].action == "done" and out["r"].by == "console-operator"
        assert len(out["r"].human_events) == 1
        assert requests.get(f"{console.url}/api/state").json()["history"][0]["human_events"] == 1
        assert "Rote operator console" in requests.get(console.url).text
    finally:
        console.stop()


def test_console_times_out_to_none():
    session = SessionControl()
    console = OperatorConsole(session, port=0).start()
    try:
        r = ConsoleEscalator(console, timeout_s=0.3, notify=lambda m: None).escalate(
            Intervention(run_id="r", reason="stuck", message="m"), session)
        assert r is None
    finally:
        console.stop()
