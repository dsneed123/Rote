"""Discovery → record → replay, with a scripted model standing in for the LLM.

The scripted model reads the same prompt the real one gets and picks element
numbers from it, so the whole pipeline — observation, verified locators,
checkpoints, templating, linting, replay with *other* inputs — runs for real.
"""

import json
import re

import pytest
import requests

from mockapp.corelink import serve_in_thread
from rote.agent import Discoverer
from rote.evidence import RunLog
from rote.handoff import Resolution, ScriptedEscalator
from rote.policy import Guard, Policy, Redactor
from rote.recorder import Param, compile_capability
from rote.replay import Replayer
from rote.schema import Status, ValueType
from rote.store import LIBRARY, Store
from rote.surface import WebSurface

SECRETS = {"corelink_user": "teller01", "corelink_password": "demo-pass-123"}


class ScriptedModel:
    """Plays a competent model: finds the control it wants in the prompt."""

    model = "scripted"

    def __init__(self, plan):
        self.plan = list(plan)
        self.prompts = []

    @staticmethod
    def element(prompt, pattern):
        match = re.search(r"\[(\d+)\] " + pattern, prompt)
        assert match, f"{pattern!r} not in prompt:\n{prompt}"
        return int(match.group(1))

    def ask(self, prompt, **kwargs):
        self.prompts.append(prompt)
        step = self.plan.pop(0)
        decision = step(prompt) if callable(step) else step
        return json.dumps({"thought": "scripted", "screen": "", **decision})


BALANCE_PLAN = [
    lambda p: {"action": "click", "element": ScriptedModel.element(p, 'link "Member Inquiry"'),
               "step_name": "open_member_inquiry", "screen": "TELLER HOME"},
    lambda p: {"action": "fill", "element": ScriptedModel.element(p, r'textbox label="Member Number:"'),
               "value": "{{inputs.member_id}}", "step_name": "enter_member_number", "screen": "MEMBER INQUIRY"},
    lambda p: {"action": "click", "element": ScriptedModel.element(p, 'button "Inquire"'),
               "step_name": "submit_inquiry", "screen": "MEMBER INQUIRY"},
    {"action": "extract", "output": "savings_balance", "output_type": "money", "value": "$1,234.56",
     "screen": "MEMBER INQUIRY - DETAIL"},
    {"action": "done", "success_text": "MEMBER INQUIRY - DETAIL", "screen": "MEMBER INQUIRY - DETAIL"},
]


@pytest.fixture(scope="module")
def app():
    server, url = serve_in_thread(port=0, faults=True)
    yield url
    server.shutdown()


@pytest.fixture(autouse=True)
def reset(app):
    requests.post(f"{app}/__control/faults", json={"reset": True})


def discover(app, tmp_path, model, params, goal="read the savings balance", escalator=None):
    policy = Policy.load(LIBRARY / "policies" / "corelink-teller.yaml").for_base_url(app)
    redactor = Redactor()
    surface = WebSurface(app, Guard(policy, mode="discovery"), redactor, secrets=SECRETS.__getitem__)
    surface.start()
    try:
        log = RunLog("disc", redactor, root=tmp_path)
        result = Discoverer(surface, Store().profile("corelink-teller"), log, redactor, model,
                            escalator=escalator).run(goal, params, product="corelink-teller", entry="/main")
        log.close()
        return result, log.dir, redactor
    finally:
        surface.close()


def replay(app, tmp_path, cap, inputs):
    policy = Policy.load(LIBRARY / "policies" / "corelink-teller.yaml").for_base_url(app)
    redactor = Redactor()
    surface = WebSurface(app, Guard(policy, mode="replay"), redactor, secrets=SECRETS.__getitem__)
    surface.start()
    try:
        log = RunLog("replay", redactor, root=tmp_path)
        return Replayer(surface, Store().profile("corelink-teller"), log, redactor).run(cap, inputs)
    finally:
        surface.close()


def test_discover_record_replay(app, tmp_path):
    model = ScriptedModel(BALANCE_PLAN)
    result, logdir, redactor = discover(app, tmp_path, model,
                                        {"member_id": Param("12345", pattern=r"\d{4,9}")})
    assert result.status == "recorded", result.message

    cap, notes = compile_capability(result.recording, "corelink.member.read_savings_balance",
                                    Store().profile("corelink-teller"), Redactor())
    steps = {s.id: s for s in cap.steps}
    assert list(steps) == ["open_member_inquiry", "enter_member_number", "submit_inquiry",
                           "read_savings_balance"]
    assert steps["enter_member_number"].value == "{{inputs.member_id}}"
    assert steps["enter_member_number"].target.strategies[0].by == "label_near"
    assert steps["submit_inquiry"].expect[0].text == "MEMBER INQUIRY - DETAIL"
    read = steps["read_savings_balance"].target.strategies[0]
    assert (read.by, read.row_value, read.column) == ("table_cell", "PRIMARY SAVINGS", "Balance")
    assert {c.text for c in cap.success} == {"MEMBER INQUIRY - DETAIL", "{{inputs.member_id}}"}
    assert cap.outputs["savings_balance"].type == ValueType.money
    assert "member_not_found" in {o.code for o in cap.outcomes}
    # Nothing from the run leaks into the artifact.
    dumped = json.dumps(cap.model_dump(mode="json"))
    assert "1,234.56" not in dumped and "12345" not in dumped.replace('"12345"', "")  # example only

    # Replay with a *different* member: the artifact generalised.
    other = replay(app, tmp_path, cap, {"member_id": "20001"})
    assert other.status == Status.succeeded and other.outputs == {"savings_balance": "87012.00"}
    missing = replay(app, tmp_path, cap, {"member_id": "99999"})
    assert missing.status == Status.business_outcome and missing.outcome.code == "member_not_found"


def test_notice_during_discovery_is_not_recorded(app, tmp_path):
    requests.post(f"{app}/__control/faults", json={"notice_once": True})
    result, logdir, _ = discover(app, tmp_path, ScriptedModel(BALANCE_PLAN),
                                 {"member_id": Param("12345")})
    assert result.status == "recorded"
    assert all("Acknowledge" not in (s.target.description if s.target else "") for s in result.recording.steps)
    assert '"code": "system_notice"' in (logdir / "log.jsonl").read_text()


def test_business_outcome_ends_discovery(app, tmp_path):
    result, _, _ = discover(app, tmp_path, ScriptedModel(BALANCE_PLAN[:3]), {"member_id": Param("99999")})
    assert result.status == "unachievable" and result.outcome_code == "member_not_found"


def test_bad_element_is_fed_back_then_stuck_goes_to_human(app, tmp_path):
    asked = []
    escalator = ScriptedEscalator(lambda i, s: asked.append(i.reason) or Resolution(action="abort"))
    model = ScriptedModel([{"action": "click", "element": 999}] * 3)
    result, _, _ = discover(app, tmp_path, model, {"member_id": Param("12345")}, escalator=escalator)
    assert "element 999 is not on the current screen" in model.prompts[1]
    assert asked == ["stuck"] and result.status == "stuck"


def test_commit_in_discovery_asks_a_human(app, tmp_path):
    plan = BALANCE_PLAN[:3] + [
        lambda p: {"action": "click", "element": ScriptedModel.element(p, 'link "Open Share"'),
                   "step_name": "open_share_form"},
        lambda p: {"action": "select", "element": ScriptedModel.element(p, r'combobox label="Share Type"'),
                   "value": "{{inputs.share_type}}", "step_name": "choose_share_type"},
        lambda p: {"action": "fill", "element": ScriptedModel.element(p, r'textbox label="Opening Deposit"'),
                   "value": "{{inputs.deposit}}", "step_name": "enter_deposit"},
        lambda p: {"action": "click", "element": ScriptedModel.element(p, 'button "Continue"'),
                   "step_name": "continue_to_review"},
        lambda p: {"action": "click", "element": ScriptedModel.element(p, 'button "Post"'),
                   "step_name": "post_share"},
        {"action": "extract", "output": "confirmation", "output_type": "string",
         "value": "SHARE S01 OPENED. CONFIRMATION 48211"},
        {"action": "done", "success_text": "SHARE MAINTENANCE - CONFIRMATION"},
    ]
    asked = []
    escalator = ScriptedEscalator(lambda i, s: asked.append((i.reason, i.step_id)) or Resolution(action="approve"))
    result, _, _ = discover(app, tmp_path, ScriptedModel(plan), {
        "member_id": Param("12345"), "share_type": Param("SAVINGS CLUB"), "deposit": Param("25.00")},
        escalator=escalator)
    assert asked == [("approval_required", "post_share")]
    assert result.status == "recorded"
    cap, _ = compile_capability(result.recording, "corelink.share.open_share",
                                Store().profile("corelink-teller"), Redactor())
    assert cap.side_effects == "irreversible"
    assert {s.id: s.risk.value for s in cap.steps}["post_share"] == "commit"
