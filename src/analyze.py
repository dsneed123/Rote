"""
Analyze the screenshot and prompt to determine a course of action.
This should return a list of tool call instructions, which will be executed by the controller layer.
 The instructions should be in the form of a list of dictionaries, where each dictionary
 contains the following keys:

    tool  the name of a tool the chosen controller exposes  (str, required)
    args  keyword arguments for that tool                   (dict, required)
    why   the model's reason for the step                   (str, optional)

See plan.py for the dataclasses; PLAN_SCHEMA constrains the model's reply to
this shape, so the result is parsed rather than scraped out of prose.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ai import AI, AIError

from .controller import Controller
from .plan import PLAN_SCHEMA, Plan

PLANNER_SYSTEM = """\
You plan UI and system automation. You are given a screenshot and a goal.

Reply with a plan: an ordered list of steps that achieves the goal using ONLY
the tools listed below. Never invent a tool or an argument name.

Rules:
- Each step is one tool call. Use the exact tool name and argument names given.
- Prefer the fewest steps that work. Do not add verification steps.
- If the screenshot shows the goal is already met, return an empty step list.
- If the goal cannot be done with these tools, return an empty step list and
  say why in `goal`.
"""


def describe_tools(controller: Controller) -> str:
    """Render a controller's tools as a signature list for the prompt."""
    import inspect
    import typing

    lines = []
    for fn in controller.tools():
        signature = inspect.signature(fn)
        # get_type_hints, not p.annotation: controller.py uses postponed
        # annotations, so p.annotation is the string "int", not the type.
        hints = typing.get_type_hints(fn)
        params = ", ".join(
            f"{name}: {getattr(hints.get(name, str), '__name__', 'str')}"
            + ("" if p.default is inspect.Parameter.empty else f" = {p.default!r}")
            for name, p in signature.parameters.items()
        )
        doc = (inspect.getdoc(fn) or "").split("\n")[0]
        lines.append(f"  {fn.__name__}({params})\n      {doc}")
    return "\n".join(lines)


def capture_screen(path: str | Path | None = None) -> Path:
    """Take a screenshot and return the file path.

    Tries the Wayland-native tools first, then X11, then PIL. On a bare Wayland
    session none may be present, so the error names what to install rather than
    failing with something opaque.
    """
    target = Path(path) if path else Path(tempfile.mkstemp(suffix=".png")[1])

    backends = [
        ("grim", [str(target)]),                                  # wlroots
        ("gnome-screenshot", ["-f", str(target)]),                # GNOME
        ("spectacle", ["-b", "-n", "-o", str(target)]),           # KDE
        ("scrot", ["-o", str(target)]),                           # X11
        ("import", ["-window", "root", str(target)]),             # ImageMagick
    ]
    for binary, args in backends:
        if shutil.which(binary):
            proc = subprocess.run([binary, *args], capture_output=True, text=True)
            if proc.returncode == 0 and target.exists() and target.stat().st_size:
                return target

    try:  # last resort; on Wayland this usually returns black or fails
        from PIL import ImageGrab

        image = ImageGrab.grab()
        image.save(target)
        return target
    except Exception:
        pass

    raise RuntimeError(
        "no working screenshot backend. On Wayland install one of:\n"
        "  sudo apt install grim            # wlroots compositors\n"
        "  sudo apt install gnome-screenshot # GNOME"
    )


def analyze(
    prompt: str,
    controller: Controller,
    *,
    screenshot: str | Path | None = None,
    ai: AI | None = None,
    extra_instructions: str = "",
) -> Plan:
    """Turn a goal plus what is on screen into an executable Plan.

    `screenshot` is a path to an image, or None to plan without one. Capturing
    is the caller's job (bridge.py), because a failed capture should not stop a
    goal that never needed the screen — a shell command, say.
    """
    ai = ai or AI()
    image = Path(screenshot) if screenshot else None

    instructions = (
        f"{PLANNER_SYSTEM}\n"
        f"Controller: {controller.name} — {controller.description}\n"
        f"Available tools:\n{describe_tools(controller)}\n"
        f"{extra_instructions}"
    )

    raw = ai.ask(
        f"Goal: {prompt}",
        images=[image] if image else None,
        system=instructions,
        format=PLAN_SCHEMA,
        temperature=0,          # planning should be repeatable
    )

    try:
        plan = Plan.from_dict(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as e:
        raise AIError(f"could not parse a plan from the model: {e}\n{raw[:400]}") from e

    # The controller is already decided — bridge chose it, and the tools shown
    # to the model came from it. The model sometimes fills this field with a
    # tool name instead, which would break workflow replay, so it is not
    # consulted: always record the controller the plan was actually built for.
    plan.controller = controller.name
    plan.goal = plan.goal or prompt
    validate(plan, controller)
    return plan


def validate(plan: Plan, controller: Controller) -> None:
    """Reject a plan that calls tools the controller does not have.

    Cheaper to catch here than halfway through executing it.
    """
    known = controller.registry()
    unknown = [step.tool for step in plan.steps if step.tool not in known]
    if unknown:
        raise AIError(
            f"plan calls tools {controller.name} does not have: {', '.join(unknown)}"
        )
