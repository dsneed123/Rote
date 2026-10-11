"""Guardrails: what the automation may touch, how risky an action is, and
what must never be written down.

Every action — whether the model chose it during discovery, a replay is
executing it, or a recovery handler fired it — passes through Guard.check()
inside the surface layer. There is no other path to the browser, so the model
cannot talk its way around the policy: the allowlist is enforced on the action
and again on the network (see surface.WebSurface, which aborts navigations to
origins not on the list).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import urlparse

import yaml
from pydantic import Field

from .schema import Action, Model, Risk

# --- policy -----------------------------------------------------------------------


class RiskRules(Model):
    """How each risk class is treated.

    allow     go ahead
    approved  allowed in replay only if the capability is approved (reviewed by
              a human, fingerprint intact); during discovery, ask a human
    confirm   ask a human every time, even in an approved replay
    block     never
    """

    read: Literal["allow", "approved", "confirm", "block"] = "allow"
    input: Literal["allow", "approved", "confirm", "block"] = "allow"
    submit: Literal["allow", "approved", "confirm", "block"] = "allow"
    commit: Literal["allow", "approved", "confirm", "block"] = "approved"


class Policy(Model):
    name: str = "default"
    allowed_origins: list[str] = Field(default_factory=list)   # scheme://host:port
    allowed_paths: list[str] = Field(default_factory=lambda: [".*"])   # regexes
    blocked_paths: list[str] = Field(default_factory=list)
    allowed_actions: list[Action] = Field(default_factory=lambda: list(Action))
    blocked_controls: list[str] = Field(default_factory=list)  # regexes on control names
    # Words on a control that make clicking it a commit. Deliberately broad: a
    # false positive costs a confirmation; a false negative posts a transaction.
    commit_words: list[str] = Field(default_factory=lambda: [
        "post", "commit", "transfer", "pay", "delete", "remove", "close account",
        "approve", "disburse", "withdraw", "reverse", "void", "charge off",
        "finali[sz]e", "authori[sz]e", "confirm", "send", "apply",
    ])
    submit_words: list[str] = Field(default_factory=lambda: [
        "submit", "search", "inquire", "continue", "next", "go", "find", "sign on",
        "log ?in", "acknowledge", "ok",
    ])
    risk: RiskRules = Field(default_factory=RiskRules)
    max_steps: int = 25                 # discovery budget
    redact_money_in_logs: bool = True

    @classmethod
    def load(cls, path: Path | str) -> "Policy":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def for_base_url(self, base_url: str) -> "Policy":
        """This policy with the tenant's own origin allowed (and nothing else added)."""
        origin = _origin(base_url)
        if origin in self.allowed_origins:
            return self
        return self.model_copy(update={"allowed_origins": [*self.allowed_origins, origin]})


def _origin(url: str) -> str:
    parts = urlparse(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme, 0)
    return f"{parts.scheme}://{parts.hostname}:{port}"


class PolicyViolation(Exception):
    def __init__(self, message: str, rule: str) -> None:
        super().__init__(message)
        self.rule = rule


class NeedsHuman(Exception):
    """The action is permitted only with a person's say-so."""

    def __init__(self, message: str, risk: Risk) -> None:
        super().__init__(message)
        self.risk = risk


@dataclass
class Guard:
    """Applies a Policy to concrete actions. One per run."""

    policy: Policy
    mode: Literal["discovery", "replay"]
    approved: bool = False              # replay: is the capability approved?
    # Commit approvals granted by a human during this run, keyed by step.
    granted: set[str] = field(default_factory=set)

    # -- urls --

    def url_allowed(self, url: str) -> bool:
        if url.startswith(("about:", "data:")):
            return url == "about:blank"
        if _origin(url) not in self.policy.allowed_origins:
            return False
        path = urlparse(url).path or "/"
        if any(re.search(p, path) for p in self.policy.blocked_paths):
            return False
        return any(re.search(p, path) for p in self.policy.allowed_paths)

    def check_url(self, url: str) -> None:
        if not self.url_allowed(url):
            raise PolicyViolation(f"{url} is outside the allowlist", "allowlist")

    # -- actions --

    def classify(self, action: Action, control: str = "", declared: Risk | None = None) -> Risk:
        """Risk of an action on a control with this visible name.

        The higher of the declared risk (from a reviewed artifact) and the
        inferred one wins, so a mislabelled artifact cannot downgrade a Post
        button to a read.
        """
        inferred = Risk.read
        if action in (Action.fill, Action.select):
            inferred = Risk.input
        elif action in (Action.click, Action.press):
            name = control.lower()
            if any(re.search(rf"\b{w}\b", name) for w in self.policy.commit_words):
                inferred = Risk.commit
            elif action == Action.press and "enter" in name:
                inferred = Risk.submit
            elif any(re.search(rf"\b{w}\b", name) for w in self.policy.submit_words):
                inferred = Risk.submit
        order = list(Risk)
        if declared and order.index(declared) > order.index(inferred):
            return declared
        return inferred

    def check(self, action: Action, control: str = "", declared: Risk | None = None,
              step_id: str = "") -> Risk:
        """Raise unless the action may proceed now. Returns its risk class."""
        if action not in self.policy.allowed_actions:
            raise PolicyViolation(f"action {action.value} is not allowed", "actions")
        for pattern in self.policy.blocked_controls:
            if control and re.search(pattern, control, re.IGNORECASE):
                raise PolicyViolation(f"control {control!r} is blocked by policy", "blocked_controls")

        risk = self.classify(action, control, declared)
        rule = getattr(self.policy.risk, risk.value)
        if rule == "allow" or step_id in self.granted:
            return risk
        if rule == "block":
            raise PolicyViolation(f"{risk.value} actions are blocked ({control!r})", f"risk.{risk.value}")
        if rule == "approved" and self.mode == "replay" and self.approved:
            return risk
        raise NeedsHuman(
            f"{risk.value} action on {control!r} needs a human decision", risk)


# --- redaction --------------------------------------------------------------------

SSN = r"\b\d{3}-\d{2}-\d{4}\b"
CARD = r"\b(?:\d[ -]?){13,19}\b"
EMAIL = r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"
PHONE = r"\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b"
MONEY = r"\$\s?\d[\d,]*(?:\.\d{2})?"
DATE = r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"


def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


class Redactor:
    """Masks sensitive data in anything about to be logged or stored.

    Two sources: patterns (SSNs, card numbers, emails, phones, dates, money)
    and known values (this run's secrets and PII inputs, which may look like
    anything). Known values are replaced by name, so the log still says
    *which* input went where: "filled <member_id>" not "filled 12345".
    """

    def __init__(self, money: bool = True) -> None:
        self.money = money
        self._known: dict[str, str] = {}

    def know(self, value: Any, label: str) -> None:
        text = str(value)
        if len(text) >= 3:   # don't mask "1" everywhere
            self._known[text] = label

    def forget_all(self) -> None:
        self._known.clear()

    @property
    def screen_patterns(self) -> list[str]:
        """Patterns whose on-screen matches get masked in screenshots."""
        return [SSN, DATE] + ([MONEY] if self.money else [])

    def text(self, value: str) -> str:
        if not value:
            return value
        # Longest first, so "12345" inside "123456" is not half-masked.
        for known in sorted(self._known, key=len, reverse=True):
            value = value.replace(known, f"<{self._known[known]}>")
        value = re.sub(SSN, "<ssn>", value)
        value = re.sub(CARD, lambda m: "<card>" if _luhn(re.sub(r"\D", "", m.group())) else m.group(), value)
        value = re.sub(EMAIL, "<email>", value)
        value = re.sub(PHONE, "<phone>", value)
        value = re.sub(DATE, "<date>", value)
        if self.money:
            value = re.sub(MONEY, "<amount>", value)
        return value

    def obj(self, value: Any) -> Any:
        """Recursively redact strings in dicts/lists."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.obj(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.obj(v) for v in value]
        return value

    def leaks(self, text: str) -> list[str]:
        """What in `text` would be redacted. Used to lint artifacts before saving."""
        found = [label for known, label in self._known.items() if known in text]
        checks = [("ssn", SSN), ("email", EMAIL), ("date", DATE)] + ([("amount", MONEY)] if self.money else [])
        for name, pattern in checks:
            if re.search(pattern, text):
                found.append(name)
        for match in re.finditer(CARD, text):
            if _luhn(re.sub(r"\D", "", match.group())):
                found.append("card")
        return found


def lint_text(texts: Iterable[str], redactor: Redactor) -> list[str]:
    problems = []
    for text in texts:
        problems += [f"{label} in {text[:60]!r}" for label in redactor.leaks(text)]
    return problems
