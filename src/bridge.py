"""The orchestration layer between main.py and the controllers.

`Bridge.handle(goal)` is the whole loop: replay a saved workflow if one matches
the goal; otherwise take a screenshot, ask the model which controller fits,
have analyze.py plan the steps, run them, and record the plan if every step
succeeded. When a run fails, `Bridge.correct()` re-plans with the operator's
clarification, so the same goal does not fail the same way twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ai import AI, AIError

from . import analyze as analyze_mod
from . import workflows
from .controller import Controller, available, ask_terminal
from .plan import Plan, Step, StepResult


@dataclass
class Outcome:
    """What handle() did, so the caller can report or react without guessing."""

    goal: str
    plan: Plan | None = None
    results: list[StepResult] = field(default_factory=list)
    replayed: bool = False          # came from a saved workflow
    saved_to: Path | None = None    # newly recorded
    error: str = ""
    note: str = ""                  # non-fatal remark, e.g. no screenshot available

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.results) and all(r.ok for r in self.results)

    def summary(self) -> str:
        if self.error:
            return f"failed: {self.error}"
        source = "replayed" if self.replayed else "planned"
        line = f"{source} {len(self.results)} step(s); " + ("all ok" if self.ok else "had failures")
        if self.saved_to:
            line += f"; recorded to {self.saved_to.name}"
        return line


class Bridge:
    def __init__(
        self,
        ai: AI | None = None,
        controllers: dict[str, Controller] | None = None,
        confirm: Callable[[Step], bool] = ask_terminal,
        workflow_dir: Path = workflows.WORKFLOW_DIR,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.ai = ai or AI()
        # Called as things happen, so a caller can show progress during a run
        # rather than only in the summary afterwards.
        self.notify = notify or (lambda message: None)
        # only_usable: a controller whose binaries are missing is left out, so
        # the model is never offered one whose every step would fail.
        self.controllers = controllers or available(only_usable=True, confirm=confirm)
        self.workflow_dir = workflow_dir
        self._warned_no_screen = False
        if not self.controllers:                      # shell has no prerequisites
            self.controllers = available(confirm=confirm)

    # -- choosing a controller ------------------------------------------------

    @property
    def default_controller(self) -> Controller:
        """Shell where possible — it is the one with no prerequisites."""
        return self.controllers.get("shell") or next(iter(self.controllers.values()))

    def choose_controller(self, goal: str, screenshot: Path | None = None) -> Controller:
        """Decide which controller fits, from the goal and what is on screen.

        Asks the model to pick, and falls back to shell if the answer is not a
        controller we have.
        """
        menu = "\n".join(f"  {c.name}: {c.description}" for c in self.controllers.values())
        question = (
            f"Which controller should handle this goal?\n\nGoal: {goal}\n\n"
            f"Controllers:\n{menu}\n\nReply with the controller name only."
        )
        try:
            images = [screenshot] if screenshot else None
            answer = self.ai.ask(question, images=images, temperature=0).strip().lower()
        except AIError:
            return self.default_controller

        for name, controller in self.controllers.items():
            if name in answer:
                return controller
        return self.default_controller

    # -- the main loop --------------------------------------------------------

    def handle(
        self,
        goal: str,
        *,
        screenshot: str | Path | None = None,
        use_saved: bool = True,
        record: bool = True,
    ) -> Outcome:
        """Solve a goal: replay a saved workflow if we have one, else plan it."""
        outcome = Outcome(goal=goal)

        # A saved set of steps beats planning: no model call, and it is known to work.
        if use_saved:
            saved = workflows.find(goal, self.workflow_dir)
            if saved:
                controller = self.controllers.get(saved.plan.controller)
                if controller:
                    self.notify(f"replaying {saved.path.name if saved.path else 'a recording'}")
                    outcome.plan = saved.plan
                    outcome.replayed = True
                    outcome.results = controller.run_plan(saved.plan)
                    if outcome.ok:
                        workflows.record_run(saved)
                        return outcome
                    # Drift: the recording no longer fits. Fall through and re-plan.
                    outcome.results = []
                    outcome.replayed = False

        # Best effort: a goal that never needed the screen (a shell command)
        # should not fail because no screenshot backend is installed.
        shot = Path(screenshot) if screenshot else None
        if shot is None:
            try:
                shot = analyze_mod.capture_screen()
            except RuntimeError as e:
                outcome.note = f"planning without a screenshot ({e.args[0].splitlines()[0]})"
                if not self._warned_no_screen:   # true for every goal; say it once
                    self.notify(outcome.note)
                    self._warned_no_screen = True

        controller = self.choose_controller(goal, shot)
        self.notify(f"planning with the {controller.name} controller")

        try:
            plan = analyze_mod.analyze(goal, controller, screenshot=shot, ai=self.ai)
        except AIError as e:
            outcome.error = str(e)
            return outcome

        outcome.plan = plan
        if not plan.steps:
            outcome.error = plan.goal or "the model produced no steps"
            return outcome

        self.notify(f"{len(plan)} step(s) planned")
        outcome.results = controller.run_plan(plan)

        if outcome.ok and record:
            outcome.saved_to = workflows.save(plan, goal, self.workflow_dir)

        return outcome

    # -- human intervention ---------------------------------------------------

    def correct(self, outcome: Outcome, correction: str) -> Outcome:
        """Re-plan after a failure, telling the model what went wrong.

        This is the "human intervention" path: the operator says what should
        have happened, and a plan that then succeeds is recorded, so the same
        goal does not fail the same way twice.
        """
        controller = (
            self.controllers.get(outcome.plan.controller, self.default_controller)
            if outcome.plan else self.default_controller
        )

        # Two ways the last attempt can have failed: steps ran and errored, or
        # the model declined to produce any. Say which, so the retry is aimed.
        failures = "\n".join(str(r) for r in outcome.results if not r.ok)
        if failures:
            went_wrong = f"These steps failed:\n{failures}"
        else:
            went_wrong = (
                "No steps were produced. The reason given was: "
                f"{outcome.error or 'none'}"
            )

        revised_goal = (
            f"{outcome.goal}\n\n{went_wrong}\n\n"
            f"The operator has since clarified: {correction}"
        )
        # The planner is otherwise told it may return an empty plan when a goal
        # looks impossible — which is exactly the conclusion being overridden
        # here, so that escape route is closed for the retry.
        override = (
            "\nThis is a RETRY after a failure. The operator's clarification is "
            "authoritative and describes something that IS possible with these "
            "tools. Do not return an empty step list, and do not repeat the "
            "earlier conclusion that it cannot be done — follow the clarification.\n"
        )

        self.notify("re-planning with your correction")
        try:
            plan = analyze_mod.analyze(
                revised_goal, controller, ai=self.ai, extra_instructions=override
            )
        except (AIError, RuntimeError) as e:
            return Outcome(goal=outcome.goal, error=str(e))

        if not plan.steps:
            return Outcome(
                goal=outcome.goal, plan=plan,
                error=plan.goal or "the model still produced no steps",
            )

        result = Outcome(goal=outcome.goal, plan=plan)
        result.results = controller.run_plan(plan)
        if result.ok:
            result.saved_to = workflows.save(plan, outcome.goal, self.workflow_dir)
        return result
