"""Compiling a discovery recording into a capability: typed parameter values
become templates, and a recording with PII in a checkpoint or no success
checkpoint is rejected."""

import pytest

from rote.policy import Redactor
from rote.recorder import Param, RecordedStep, Recording, RecordingRejected, compile_capability, templatize
from rote.schema import Action, FrameRef, LabelNearLocator, Sensitivity, Target
from rote.store import Store

PROFILE = Store().profile("corelink-teller")
FIELD = Target(description="field", frame=FrameRef(name="main"),
               strategies=[LabelNearLocator(label="Member Number:")])


def recording(**kw):
    base = dict(goal="g", product="corelink-teller", entry="/main",
                params={"member_id": Param("12345")},
                steps=[RecordedStep(Action.fill, FIELD, value="12345", name="enter member number")],
                success=[("MEMBER INQUIRY - DETAIL", FrameRef(name="main"))])
    base.update(kw)
    return Recording(**base)


def test_literal_param_values_are_templated():
    cap, _ = compile_capability(recording(), "corelink.x.y", PROFILE, Redactor())
    assert cap.steps[0].value == "{{inputs.member_id}}" and cap.steps[0].id == "enter_member_number"


def test_templatize_longest_first():
    params = {"a": Param("123"), "b": Param("12345")}
    assert templatize("12345 123", params) == "{{inputs.b}} {{inputs.a}}"


def test_pii_in_checkpoint_is_rejected():
    rec = recording(params={"member_id": Param("12345"), "name": Param("JANE Q SAMPLE", sensitivity=Sensitivity.pii)},
                    success=[("SSN 900-12-3456", None)])
    with pytest.raises(RecordingRejected, match="ssn"):
        compile_capability(rec, "corelink.x.y", PROFILE, Redactor())


def test_unused_params_are_dropped_and_noted():
    rec = recording(params={"member_id": Param("12345"), "unused": Param("zzz")})
    cap, notes = compile_capability(rec, "corelink.x.y", PROFILE, Redactor())
    assert "unused" not in cap.inputs and any("unused" in n for n in notes)


def test_no_success_checkpoint_no_artifact():
    with pytest.raises(RecordingRejected):
        compile_capability(recording(success=[]), "corelink.x.y", PROFILE, Redactor())


def test_human_steps_are_flagged():
    rec = recording(steps=[RecordedStep(Action.fill, FIELD, value="12345", by_human=True)])
    cap, notes = compile_capability(rec, "corelink.x.y", PROFILE, Redactor())
    assert cap.provenance.human_steps == 1 and "operator" in cap.steps[0].note
    assert any("human" in n for n in notes)


def test_notes_are_templated_and_redacted():
    rec = recording(steps=[RecordedStep(Action.fill, FIELD, value="12345",
                                        note="member 12345 has a balance of $1,234.56")])
    cap, _ = compile_capability(rec, "corelink.x.y", PROFILE, Redactor())
    assert cap.steps[0].note == "member {{inputs.member_id}} has a balance of <amount>"
