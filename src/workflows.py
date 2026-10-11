"""Recorded workflows: the "record once" half of Record Once Then Execute.

A workflow is a Plan that worked, saved to src/workflows/<slug>.yaml so the
same goal can be replayed later without asking the model to plan it again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from .plan import Plan

WORKFLOW_DIR = Path(__file__).parent / "workflows"


def slugify(goal: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    return (slug or "workflow")[:60]


@dataclass
class Workflow:
    goal: str
    plan: Plan
    created: str = ""
    runs: int = 0
    path: Path | None = None

    def to_dict(self) -> dict:
        # The plan carries its own `goal` (the model's phrasing of the task).
        # The workflow's goal is what the operator actually asked for, and that
        # is what find() matches on, so it must not be overwritten by the merge.
        body = self.plan.to_dict()
        body.pop("goal", None)
        return {
            "goal": self.goal,
            "created": self.created or datetime.now().isoformat(timespec="seconds"),
            "runs": self.runs,
            **body,
        }

    @classmethod
    def from_dict(cls, raw: dict, path: Path | None = None) -> "Workflow":
        return cls(
            goal=raw.get("goal", ""),
            plan=Plan.from_dict(raw),
            created=raw.get("created", ""),
            runs=int(raw.get("runs", 0)),
            path=path,
        )


def save(plan: Plan, goal: str | None = None, directory: Path = WORKFLOW_DIR) -> Path:
    """Record a plan that worked. Returns the file it was written to."""
    goal = goal or plan.goal
    workflow = Workflow(goal=goal, plan=plan)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slugify(goal)}.yaml"
    path.write_text(yaml.safe_dump(workflow.to_dict(), sort_keys=False))
    return path


def load(path: Path) -> Workflow:
    raw = yaml.safe_load(path.read_text()) or {}
    if not raw:
        raise ValueError(f"{path.name} is empty")
    return Workflow.from_dict(raw, path=path)


def all_workflows(directory: Path = WORKFLOW_DIR) -> list[Workflow]:
    """Every saved workflow. Unreadable or empty files are skipped, not fatal."""
    found = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            found.append(load(path))
        except (ValueError, yaml.YAMLError):
            continue
    return found


def find(goal: str, directory: Path = WORKFLOW_DIR) -> Workflow | None:
    """Look for a saved workflow for this goal.

    Exact slug match first, then a word-overlap score. This is deliberately
    literal: a wrong match replays the wrong actions, so it prefers to find
    nothing and let the model plan fresh. Swapping in embeddings here
    (ai.embed) would make matching fuzzier — and riskier.
    """
    candidates = all_workflows(directory)
    if not candidates:
        return None

    slug = slugify(goal)
    for workflow in candidates:
        if slugify(workflow.goal) == slug:
            return workflow

    wanted = set(slug.split("-"))
    best, best_score = None, 0.0
    for workflow in candidates:
        words = set(slugify(workflow.goal).split("-"))
        overlap = len(wanted & words) / max(len(wanted | words), 1)
        if overlap > best_score:
            best, best_score = workflow, overlap
    return best if best_score >= 0.7 else None


def record_run(workflow: Workflow) -> None:
    """Bump the run counter on a workflow that was replayed."""
    if not workflow.path:
        return
    workflow.runs += 1
    workflow.path.write_text(yaml.safe_dump(workflow.to_dict(), sort_keys=False))
