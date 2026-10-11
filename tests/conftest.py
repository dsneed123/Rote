"""Shared fixtures.

The default suite never touches the GX10: a 30B model is slow and its output
varies between runs, neither of which belongs in a unit test. FakeAI stands in
for it and returns exactly what a test wants to exercise. The handful of tests
that do need the real thing live in test_integration.py behind a marker.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controller import ShellController, allow_all  # noqa: E402
from src.plan import Plan, Step  # noqa: E402


class FakeAI:
    """Stands in for ai.AI, returning scripted replies and recording calls."""

    def __init__(self, replies=None, host="http://fake", model="fake-model"):
        # replies: a list consumed in order, or a single value reused forever.
        self.replies = replies if isinstance(replies, list) else [replies]
        self.calls = []
        self.host = host
        self.model = model

    def ask(self, prompt, images=None, system=None, **kwargs):
        self.calls.append({
            "prompt": prompt, "images": images, "system": system, **kwargs,
        })
        if not self.replies:
            raise AssertionError("FakeAI ran out of scripted replies")
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def stream(self, prompt, **kwargs):
        yield self.ask(prompt, **kwargs)

    def is_up(self):
        return True

    def has_model(self):
        return True

    def warm(self):
        pass


def plan_json(controller="shell", steps=(("run", {"command": "echo hi"}),), goal="do it"):
    """Build the JSON string a model would return for a plan."""
    return json.dumps({
        "goal": goal,
        "controller": controller,
        "steps": [{"tool": t, "args": a, "why": ""} for t, a in steps],
    })


@pytest.fixture
def workdir(tmp_path):
    """An isolated directory for a ShellController to operate in."""
    (tmp_path / "sample.txt").write_text("hello\n")
    (tmp_path / "sub").mkdir()
    return tmp_path


@pytest.fixture
def shell(workdir):
    """A ShellController that never prompts, rooted in a temp directory."""
    return ShellController(root=workdir, confirm=allow_all)


@pytest.fixture
def wf_dir(tmp_path):
    directory = tmp_path / "workflows"
    directory.mkdir()
    return directory


@pytest.fixture
def simple_plan():
    return Plan(
        controller="shell",
        goal="say hello",
        steps=[Step("run", {"command": "echo hello"}, "greet")],
    )
