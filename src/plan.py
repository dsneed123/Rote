"""The contract between the analyze layer and the controller layer.

analyze.py produces a Plan; controller.py executes it; workflows.py records it
so it can be replayed without asking the model again. Every step is one tool
call, flat and serialisable, so a plan round-trips through YAML unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    """One tool call.

    tool  name of a tool the chosen controller exposes, e.g. "run", "click"
    args  keyword arguments for it; must match that tool's signature
    why   the model's stated reason, kept for debugging and for the recording
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "Step":
        if not isinstance(raw, dict):
            raise ValueError(f"step must be a mapping, got {type(raw).__name__}")
        if not raw.get("tool"):
            raise ValueError(f"step is missing 'tool': {raw!r}")
        args = raw.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"step 'args' must be a mapping: {raw!r}")
        return cls(tool=str(raw["tool"]), args=args, why=str(raw.get("why", "")))

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.tool}({rendered})"


@dataclass
class Plan:
    """An ordered list of steps, plus the controller expected to run them."""

    controller: str
    steps: list[Step] = field(default_factory=list)
    goal: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "Plan":
        if not isinstance(raw, dict):
            raise ValueError(f"plan must be a mapping, got {type(raw).__name__}")
        steps = raw.get("steps")
        if not isinstance(steps, list):
            raise ValueError("plan is missing a 'steps' list")
        return cls(
            controller=str(raw.get("controller", "")),
            steps=[Step.from_dict(s) for s in steps],
            goal=str(raw.get("goal", "")),
        )

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "controller": self.controller,
            "steps": [s.to_dict() for s in self.steps],
        }

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class StepResult:
    step: Step
    ok: bool
    output: str = ""

    def __str__(self) -> str:
        return f"[{'ok' if self.ok else 'FAILED'}] {self.step} -> {self.output}"


# Handed to the model as `format=` so the reply is a Plan-shaped object rather
# than prose that needs parsing. See analyze.py.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "controller": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                    "why": {"type": "string"},
                },
                "required": ["tool", "args"],
            },
        },
    },
    "required": ["controller", "steps"],
}
