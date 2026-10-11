"""WebSurface against the real mock app in headless Chromium."""

import pytest

from mockapp.corelink import serve_in_thread
from rote.policy import Guard, Policy, PolicyViolation, Redactor
from rote.schema import (Action, CssLocator, FrameRef, LabelNearLocator, RoleLocator,
                         TableCellLocator, Target, TextCheck)
from rote.surface import TargetNotFound, WebSurface

SECRETS = {"corelink_user": "teller01", "corelink_password": "demo-pass-123"}
MAIN = FrameRef(name="main")


@pytest.fixture(scope="module")
def app():
    server, url = serve_in_thread(port=0, faults=True)
    yield url
    server.shutdown()


@pytest.fixture
def surface(app, tmp_path):
    policy = Policy(allowed_paths=[r"^/(?!__control)"]).for_base_url(app)
    s = WebSurface(app, Guard(policy, mode="replay"), Redactor(), secrets=SECRETS.__getitem__)
    s.start()
    yield s
    s.close()


def sign_on(s):
    s.navigate("/login")
    s.perform(Action.fill, s.resolve(Target(strategies=[LabelNearLocator(label="Operator ID")])),
              "{{secrets.corelink_user}}")
    s.perform(Action.fill, s.resolve(Target(strategies=[LabelNearLocator(label="Password")])),
              "{{secrets.corelink_password}}")
    s.perform(Action.click, s.resolve(Target(strategies=[RoleLocator(role="button", name="Sign On")])), None)


def to_member(s, number="12345"):
    sign_on(s)
    s.perform(Action.click, s.resolve(Target(frame=FrameRef(name="nav"),
              strategies=[RoleLocator(role="link", name="Member Inquiry")])), None)
    s.perform(Action.fill, s.resolve(Target(frame=MAIN, strategies=[
        LabelNearLocator(label="Member Number:")])), "{{inputs.m}}", inputs={"m": number})
    s.perform(Action.click, s.resolve(Target(frame=MAIN, strategies=[
        RoleLocator(role="button", name="Inquire")])), None)


def test_sign_on_and_frames(surface):
    sign_on(surface)
    assert surface.location()["main"] == "/home"
    assert surface.holds(TextCheck(text="TELLER HOME", frame=MAIN))


def test_label_near_and_table_cell(surface):
    to_member(surface)
    assert surface.holds(TextCheck(text="MEMBER INQUIRY - DETAIL", frame=MAIN))
    cell = surface.resolve(Target(frame=MAIN, strategies=[TableCellLocator(
        row_column="Description", row_value="PRIMARY SAVINGS", column="Balance")]))
    assert surface.read(cell) == "$1,234.56"


def test_ambiguity_falls_through_and_is_reported(surface):
    to_member(surface, "20001")   # two SAVINGS rows
    target = Target(frame=MAIN, strategies=[
        TableCellLocator(row_column="Type", row_value="SAVINGS", column="Balance"),   # ambiguous
        TableCellLocator(row_column="Description", row_value="HOLIDAY CLUB", column="Balance"),
    ])
    resolved = surface.resolve(target, timeout_ms=1000)
    assert resolved.strategy_index == 1 and surface.read(resolved) == "$450.00"


def test_not_found_lists_what_was_tried(surface):
    sign_on(surface)
    with pytest.raises(TargetNotFound, match="role: 0 matches"):
        surface.resolve(Target(strategies=[RoleLocator(role="button", name="Nope")]), timeout_ms=300)


def test_interrupt_stops_waiting(surface):
    from rote.surface import Interrupted
    sign_on(surface)
    with pytest.raises(Interrupted):
        surface.resolve(Target(strategies=[RoleLocator(role="button", name="Nope")]),
                        timeout_ms=5000, interrupt=lambda: "detected")


def test_network_allowlist_blocks(surface):
    sign_on(surface)
    with pytest.raises(PolicyViolation):
        surface.navigate("/__control/faults")
    surface.page.evaluate("() => { window.frames['main'].location = '/__control/faults' }")
    with pytest.raises(PolicyViolation):
        surface.settle()


def test_observe_and_verified_locators(surface):
    sign_on(surface)
    surface.perform(Action.click, surface.resolve(Target(frame=FrameRef(name="nav"),
                    strategies=[RoleLocator(role="link", name="Member Inquiry")])), None)
    obs = surface.observe()
    field = next(e for e in obs.elements if e.role == "textbox" and e.frame == "main")
    assert field.label == "Member Number:"
    target, _ = surface.locators_for(field.id)
    kinds = [s.by for s in target.strategies]
    assert kinds[0] == "label_near" and "css" in kinds and target.frame.name == "main"


def test_locators_for_value_prefers_structure(surface):
    to_member(surface)
    target, _ = surface.locators_for_value("$1,234.56")
    first = target.strategies[0]
    assert first.by == "table_cell" and first.row_value == "PRIMARY SAVINGS" and first.column == "Balance"


def test_screenshot_masks_and_snapshot_redacts(surface, tmp_path):
    to_member(surface)
    path = surface.screenshot(tmp_path / "shot.png")
    assert path.stat().st_size > 1000
    snap = surface.accessibility_snapshot()
    assert "900-12-3456" not in snap and "<ssn>" in snap


def test_profile_labels_redact_names(surface):
    surface.sensitive_labels = ["Name", "SSN"]
    to_member(surface)
    snap = surface.accessibility_snapshot()
    assert "JANE Q SAMPLE" not in snap and "<pii:name>" in snap
    assert surface.redactor.text("thought: JANE Q SAMPLE has $5") == "thought: <pii:name> has <amount>"
