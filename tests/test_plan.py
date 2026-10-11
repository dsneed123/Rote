"""The analyze/controller contract: a step is {tool, args, why}."""

import pytest

from src.plan import Plan, Step, StepResult


class TestStep:
    def test_round_trips_through_dict(self):
        step = Step("run", {"command": "ls"}, "list files")
        assert Step.from_dict(step.to_dict()) == step

    def test_args_default_to_empty(self):
        assert Step.from_dict({"tool": "read_page"}).args == {}

    def test_why_is_optional(self):
        assert Step.from_dict({"tool": "run", "args": {}}).why == ""

    @pytest.mark.parametrize("raw, message", [
        ({"args": {}}, "missing 'tool'"),
        ({"tool": ""}, "missing 'tool'"),
        ({"tool": "run", "args": ["not", "a", "dict"]}, "must be a mapping"),
        ("not a dict", "must be a mapping"),
    ])
    def test_rejects_malformed_input(self, raw, message):
        # The model writes these, so malformed input is expected, not exceptional.
        with pytest.raises(ValueError, match=message):
            Step.from_dict(raw)

    def test_str_is_readable_for_confirmation_prompts(self):
        # This string is what the operator sees before approving a step.
        assert str(Step("run", {"command": "rm -rf /"})) == "run(command='rm -rf /')"


class TestPlan:
    def test_round_trips_through_dict(self, simple_plan):
        assert Plan.from_dict(simple_plan.to_dict()) == simple_plan

    def test_len_counts_steps(self, simple_plan):
        assert len(simple_plan) == 1

    def test_requires_a_steps_list(self):
        with pytest.raises(ValueError, match="missing a 'steps' list"):
            Plan.from_dict({"controller": "shell"})

    def test_empty_step_list_is_valid(self):
        # A model declining a goal returns no steps; that is data, not an error.
        assert len(Plan.from_dict({"controller": "shell", "steps": []})) == 0

    def test_one_bad_step_rejects_the_whole_plan(self):
        with pytest.raises(ValueError):
            Plan.from_dict({"controller": "shell", "steps": [
                {"tool": "run", "args": {}},
                {"args": {}},                      # no tool
            ]})


class TestStepResult:
    def test_str_marks_failure_visibly(self):
        step = Step("run", {"command": "false"})
        assert "FAILED" in str(StepResult(step, False, "exit 1"))
        assert "ok" in str(StepResult(step, True, ""))
