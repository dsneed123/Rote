"""The capability artifact: strict validation, approval bound to content,
tenant overlays, the agent-facing tool schema, and value coercion."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rote.schema import Capability, Overlay, apply_overlay, coerce, ValueType
from rote.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "read_balance.yaml"


@pytest.fixture
def raw():
    return yaml.safe_load(FIXTURE.read_text())


@pytest.fixture
def cap(raw):
    return Capability.model_validate(raw)


def test_fixture_round_trips_through_yaml(cap):
    again = Capability.model_validate(yaml.safe_load(yaml.safe_dump(cap.model_dump(mode="json"))))
    assert again == cap


def test_unknown_keys_fail_loudly(raw):
    raw["steps"][0]["target"]["stratgies"] = []
    with pytest.raises(ValidationError):
        Capability.model_validate(raw)


def test_undeclared_input_in_template_rejected(raw):
    raw["steps"][1]["value"] = "{{inputs.nope}}"
    with pytest.raises(ValidationError, match="undeclared input"):
        Capability.model_validate(raw)


def test_output_must_be_extracted(raw):
    raw["outputs"]["other"] = {"type": "string"}
    with pytest.raises(ValidationError, match="never extracted"):
        Capability.model_validate(raw)


def test_commit_step_requires_irreversible_declaration(raw):
    raw["steps"][2]["risk"] = "commit"
    with pytest.raises(ValidationError, match="irreversible"):
        Capability.model_validate(raw)


def test_inputs_validated_before_ui(cap):
    assert cap.validate_inputs({"member_id": "12345"}) == {"member_id": "12345"}
    with pytest.raises(ValueError, match="does not match"):
        cap.validate_inputs({"member_id": "12a"})
    with pytest.raises(ValueError, match="missing"):
        cap.validate_inputs({})
    with pytest.raises(ValueError, match="unknown"):
        cap.validate_inputs({"member_id": "12345", "ssn": "x"})


@pytest.mark.parametrize("raw,type_,want", [
    ("$1,234.56", ValueType.money, "1234.56"),
    ("($10.00)", ValueType.money, "-10.00"),
    ("1,200", ValueType.integer, 1200),
    ("2024-01-02", ValueType.date, "2024-01-02"),
])
def test_coerce(raw, type_, want):
    assert coerce(raw, type_) == want


def test_coerce_rejects_garbage():
    with pytest.raises(ValueError):
        coerce("N/A", ValueType.money)


def test_approval_is_bound_to_content(cap, tmp_path):
    store = Store(tmp_path)
    store.save(cap)
    approved = store.approve(cap.name, cap.version, by="reviewer")
    assert approved.approved
    approved.steps[0].note = "edited after approval"
    assert not approved.approved          # fingerprint changed → back to draft
    with pytest.raises(FileExistsError):
        store.save(store.load(cap.name))  # approved versions are immutable


def test_overlay_patches_by_step_id(cap):
    overlay = Overlay(tenant="b", applies_to=cap.name, steps={
        "enter_member_number": {"target": {"strategies": [
            {"by": "label_near", "label": "Acct Holder No.:", "role": "textbox"}]}},
    })
    merged = apply_overlay(cap, overlay)
    assert merged.steps[1].target.strategies[0].label == "Acct Holder No.:"
    assert merged.steps[1].target.frame.name == "main"     # untouched keys kept
    assert merged.fingerprint() != cap.fingerprint()


def test_overlay_cannot_invent_steps(cap):
    with pytest.raises(ValueError, match="unknown steps"):
        apply_overlay(cap, Overlay(tenant="b", applies_to=cap.name, steps={"nope": {}}))


def test_tool_schema(cap):
    tool = cap.tool_schema()
    assert tool["name"] == "corelink__member__read_savings_balance"
    assert tool["input_schema"]["required"] == ["member_id"]


def test_shipped_profile_loads():
    assert Store().profile("corelink-teller").detectors
