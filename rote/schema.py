"""The capability artifact and the result contract.

A *capability* is what a discovery run leaves behind: a typed, versioned,
reviewable description of one flow through one application, decoupled from
the model transcript that produced it. An AI agent invokes it by name with
typed inputs and gets back typed outputs, a declared business outcome, or a
debuggable failure.

Three documents, three owners:

  Capability   one flow ("read a member's savings balance"). Written by the
               recorder from a discovery run, reviewed and approved by a human.
  AppProfile   one vendor product ("corelink-teller"). How to sign on, and the
               runtime conditions the product can throw at any flow: record
               not found, session expired, system notices. Shared by every
               capability and every tenant running that product.
  Overlay      one tenant's differences from the base capability, addressed
               by step id ("on Lakeshore the field is labelled Acct Holder No.").

Everything here is plain data (pydantic → YAML); nothing references a browser,
so the same schema serves a desktop surface. Surface-specific detail lives in
the locator strategies, which are tried in order and degrade gracefully.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "rote.capability/v1"

TEMPLATE = re.compile(r"\{\{\s*(inputs|secrets)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
STEP_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class Model(BaseModel):
    # Unknown keys are an error: a typo in a reviewed artifact ("stratgies")
    # must fail loudly at load time, not silently drop a fallback locator.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --- locators -----------------------------------------------------------------
#
# A target is identified by an ordered list of strategies, most robust first.
# Replay takes the first strategy that resolves to exactly one visible element;
# zero or several matches moves on to the next. Using anything but the first is
# reported as drift, so a slowly changing UI is noticed before it breaks.
#
# The ranking (role > label_near > text > table_cell > css) follows what a
# human operator would use to find the control, because that is what survives
# vendor upgrades and tenant branding: the words on screen and what kind of
# control it is, not the markup underneath.


class RoleLocator(Model):
    """Accessible role + accessible name: the accessibility tree's view.

    Works the same on desktop (UIA / AT-SPI expose role and name), so it is the
    preferred strategy wherever the surface computes a name at all.
    """

    by: Literal["role"] = "role"
    role: str
    name: str
    exact: bool = True


class LabelNearLocator(Model):
    """A control identified by the text label beside it.

    Legacy forms put "Member Number:" in one table cell and the input in the
    next, with no <label for>, so the control has no accessible name. This is
    how a human finds it, and it survives meaningless field names like f_0012.
    """

    by: Literal["label_near"] = "label_near"
    label: str
    role: str = "textbox"              # textbox | combobox | checkbox | button


class TextLocator(Model):
    """Visible text. For links and buttons whose text is their identity."""

    by: Literal["text"] = "text"
    text: str
    exact: bool = True


class TableCellLocator(Model):
    """A cell found by row key and column header, not by position.

    "The Balance column of the row whose Description is PRIMARY SAVINGS"
    survives column reordering, extra rows and extra columns — exactly how two
    tenants' configurations of the same product differ.
    """

    by: Literal["table_cell"] = "table_cell"
    row_column: str                    # header of the column that keys the row
    row_value: str                     # value in that column
    column: str                        # header of the column to read


class CssLocator(Model):
    """Structural fallback. Brittle; recorded only as a last resort."""

    by: Literal["css"] = "css"
    css: str


Locator = Annotated[
    Union[RoleLocator, LabelNearLocator, TextLocator, TableCellLocator, CssLocator],
    Field(discriminator="by"),
]

# Lower is more robust. The recorder orders strategies by this, and a capability
# whose best strategy for some step is a css selector gets a review warning.
LOCATOR_RANK = {"role": 0, "label_near": 1, "text": 2, "table_cell": 2, "css": 9}


class FrameRef(Model):
    """Which frame a target lives in. Framesets name their frames; fall back to
    the frame's URL path when they don't."""

    name: str | None = None
    url_path: str | None = None

    def __str__(self) -> str:
        return self.name or self.url_path or "top"


class Target(Model):
    description: str = ""
    frame: FrameRef | None = None
    strategies: list[Locator] = Field(min_length=1)


# --- checks (checkpoints and detector conditions) ------------------------------


class TextCheck(Model):
    kind: Literal["text"] = "text"
    text: str                          # visible substring; may use {{inputs.x}}
    frame: FrameRef | None = None
    negate: bool = False


class RegexCheck(Model):
    kind: Literal["regex"] = "regex"
    pattern: str
    frame: FrameRef | None = None
    negate: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        re.compile(value)
        return value


class ElementCheck(Model):
    kind: Literal["element"] = "element"
    target: Target
    negate: bool = False


class UrlCheck(Model):
    kind: Literal["url"] = "url"
    path_regex: str                    # matched against the frame's URL path
    frame: FrameRef | None = None
    negate: bool = False


Check = Annotated[Union[TextCheck, RegexCheck, ElementCheck, UrlCheck], Field(discriminator="kind")]


# --- typed inputs and outputs -------------------------------------------------


class ValueType(str, Enum):
    string = "string"
    integer = "integer"
    decimal = "decimal"
    money = "money"                    # returned as a decimal string, "1234.56"
    date = "date"                      # ISO 8601
    enum = "enum"
    boolean = "boolean"


class Sensitivity(str, Enum):
    """How a value may be handled outside the live session.

    public     can be logged as-is
    internal   logged as-is, but not sent to third parties (member numbers)
    pii        redacted in logs and evidence; returned to the caller only
    secret     never logged, never returned, never stored (credentials)
    """

    public = "public"
    internal = "internal"
    pii = "pii"
    secret = "secret"


class InputSpec(Model):
    type: ValueType = ValueType.string
    description: str = ""
    required: bool = True
    pattern: str | None = None         # validated before any UI is touched
    values: list[str] | None = None    # for enum
    sensitivity: Sensitivity = Sensitivity.internal
    example: str | None = None         # fictional; documentation only

    @model_validator(mode="after")
    def _enum_has_values(self) -> "InputSpec":
        if self.type == ValueType.enum and not self.values:
            raise ValueError("an enum input needs `values`")
        return self


class OutputSpec(Model):
    type: ValueType = ValueType.string
    description: str = ""
    sensitivity: Sensitivity = Sensitivity.pii
    # Optional regex with one group, applied to the element's text before
    # coercion: "CONFIRMATION (\d+)" out of "SHARE S01 OPENED. CONFIRMATION 48211".
    pattern: str | None = None

    def parse(self, raw: str) -> Any:
        if self.pattern:
            match = re.search(self.pattern, raw)
            if not match:
                raise ValueError(f"{raw!r} does not match {self.pattern}")
            raw = match.group(1) if match.groups() else match.group(0)
        return coerce(raw, self.type)


def coerce(value: Any, type_: ValueType) -> Any:
    """Parse a raw string (typed by a caller or read off a screen) into a value.

    Raises ValueError with a message fit for a caller. Money and decimals come
    back as strings so they survive JSON without float rounding.
    """
    text = str(value).strip()
    if type_ in (ValueType.string, ValueType.enum):
        return text
    if type_ == ValueType.integer:
        return int(text.replace(",", ""))
    if type_ in (ValueType.decimal, ValueType.money):
        cleaned = text.replace("$", "").replace(",", "").replace(" ", "")
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        try:
            number = Decimal(cleaned)
        except InvalidOperation:
            raise ValueError(f"not a {type_.value}: {text!r}") from None
        number = -number if negative else number
        return f"{number:.2f}" if type_ == ValueType.money else str(number)
    if type_ == ValueType.date:
        return date.fromisoformat(text).isoformat()
    if type_ == ValueType.boolean:
        if text.lower() in ("true", "yes", "y", "1"):
            return True
        if text.lower() in ("false", "no", "n", "0"):
            return False
        raise ValueError(f"not a boolean: {text!r}")
    raise ValueError(f"unknown type {type_}")


# --- steps ----------------------------------------------------------------------


class Risk(str, Enum):
    """What a step does to the system of record. Drives policy (see policy.py).

    read     navigation and reading; no state changes anywhere
    input    fills a field locally; nothing is sent until a submit
    submit   sends a request that does not change records (search, continue)
    commit   changes the system of record and cannot be undone (post, transfer)
    """

    read = "read"
    input = "input"
    submit = "submit"
    commit = "commit"


class Action(str, Enum):
    navigate = "navigate"              # value is a path relative to the tenant base URL
    click = "click"
    fill = "fill"
    select = "select"
    press = "press"
    extract = "extract"                # read the target's text into `output`
    wait_for = "wait_for"              # only `expect`; for slow screens


class Step(Model):
    id: str                            # stable, semantic; overlays address steps by it
    action: Action
    target: Target | None = None
    value: str | None = None           # literal or {{inputs.x}} / {{secrets.x}}
    output: str | None = None          # for extract: which declared output
    risk: Risk = Risk.read
    expect: list[Check] = Field(default_factory=list)   # checkpoint after the step
    timeout_ms: int = 10_000
    note: str = ""                     # why this step exists; from discovery, reviewed

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not STEP_ID.match(value):
            raise ValueError(f"step id must be snake_case: {value!r}")
        return value

    @model_validator(mode="after")
    def _action_needs(self) -> "Step":
        if self.action in (Action.click, Action.fill, Action.select, Action.extract) and not self.target:
            raise ValueError(f"step {self.id}: {self.action.value} needs a target")
        if self.action in (Action.fill, Action.select, Action.navigate, Action.press) and self.value is None:
            raise ValueError(f"step {self.id}: {self.action.value} needs a value")
        if self.action == Action.extract and not self.output:
            raise ValueError(f"step {self.id}: extract needs an output name")
        return self


# --- the capability -----------------------------------------------------------


class AppRef(Model):
    product: str                       # vendor product; shared across tenants
    versions: str = "*"                # product versions this was verified against
    surface: Literal["web", "desktop"] = "web"
    entry: str = "/"                   # where a run starts, relative to the tenant base URL


class OutcomeSpec(Model):
    """A business outcome the caller must be prepared for. Not an error."""

    code: str
    description: str = ""


class Approval(Model):
    status: Literal["draft", "approved", "retired"] = "draft"
    by: str | None = None
    at: str | None = None
    fingerprint: str | None = None     # the content that was approved


class Provenance(Model):
    run_id: str | None = None
    model: str | None = None
    recorded_at: str | None = None
    tenant: str | None = None
    goal: str | None = None            # the natural-language goal, as given
    human_steps: int = 0               # steps a human performed during discovery


class Capability(Model):
    schema_version: Literal["rote.capability/v1"] = SCHEMA_VERSION
    name: str                          # dotted, e.g. corelink.member.read_savings_balance
    version: str = "1.0.0"
    title: str
    description: str = ""
    app: AppRef
    side_effects: Literal["none", "reversible", "irreversible"] = "none"
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    steps: list[Step] = Field(min_length=1)
    success: list[Check] = Field(min_length=1)   # final checkpoint
    approval: Approval = Field(default_factory=Approval)
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("name")
    @classmethod
    def _name_shape(cls, value: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$", value):
            raise ValueError(f"capability name must be dotted snake_case: {value!r}")
        return value

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        if not re.match(r"^\d+\.\d+\.\d+$", value):
            raise ValueError(f"version must be semver: {value!r}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "Capability":
        ids = [s.id for s in self.steps]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate step ids: {sorted(dupes)}")
        for step in self.steps:
            for kind, name in TEMPLATE.findall(step.value or ""):
                if kind == "inputs" and name not in self.inputs:
                    raise ValueError(f"step {step.id} uses undeclared input {name!r}")
            if step.output and step.output not in self.outputs:
                raise ValueError(f"step {step.id} extracts undeclared output {step.output!r}")
        produced = {s.output for s in self.steps if s.output}
        missing = set(self.outputs) - produced
        if missing:
            raise ValueError(f"outputs never extracted: {sorted(missing)}")
        commits = any(s.risk == Risk.commit for s in self.steps)
        if commits and self.side_effects != "irreversible":
            raise ValueError("a capability with a commit step must declare side_effects: irreversible")
        return self

    def fingerprint(self) -> str:
        """Hash of what the capability *does* — not who approved it or when.

        Approval is bound to this, so editing a step after approval puts the
        capability back in draft without anyone having to remember to.
        """
        body = self.model_dump(mode="json", exclude={"approval", "provenance"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def approved(self) -> bool:
        return self.approval.status == "approved" and self.approval.fingerprint == self.fingerprint()

    def validate_inputs(self, given: dict[str, Any]) -> dict[str, Any]:
        """Check and coerce caller inputs before any UI is touched."""
        unknown = set(given) - set(self.inputs)
        if unknown:
            raise ValueError(f"unknown inputs: {sorted(unknown)}")
        clean: dict[str, Any] = {}
        for name, spec in self.inputs.items():
            if name not in given or given[name] in (None, ""):
                if spec.required:
                    raise ValueError(f"missing required input {name!r}")
                continue
            value = coerce(given[name], spec.type)
            if spec.pattern and not re.fullmatch(spec.pattern, str(value)):
                raise ValueError(f"input {name!r} does not match {spec.pattern}")
            if spec.values and str(value) not in spec.values:
                raise ValueError(f"input {name!r} must be one of {spec.values}")
            clean[name] = value
        return clean

    def tool_schema(self) -> dict:
        """This capability as a function-calling tool definition, for an agent."""
        types = {ValueType.integer: "integer", ValueType.boolean: "boolean"}
        properties = {}
        for name, spec in self.inputs.items():
            prop: dict[str, Any] = {"type": types.get(spec.type, "string"),
                                    "description": spec.description}
            if spec.pattern:
                prop["pattern"] = spec.pattern
            if spec.values:
                prop["enum"] = spec.values
            properties[name] = prop
        return {
            "name": self.name.replace(".", "__"),
            "description": f"{self.title}. {self.description}".strip(),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": [n for n, s in self.inputs.items() if s.required],
            },
        }


# --- the app profile ------------------------------------------------------------


class Disposition(str, Enum):
    """How a detected runtime condition is treated. The heart of the error model.

    business     a legitimate answer the caller needs ("no such member").
                 The run ends with status=business_outcome, not failed.
    recoverable  a known interruption with a known remedy (dismiss a notice,
                 wait out slowness, sign on again). Handled, logged, continued.
    failure      cannot proceed. Stop and report with evidence.
    """

    business = "business"
    recoverable = "recoverable"
    failure = "failure"


class Recovery(Model):
    kind: Literal["click", "reauthenticate", "wait_retry"]
    target: Target | None = None       # for click
    max_attempts: int = 2
    backoff_ms: int = 1_000


class Detector(Model):
    code: str                          # stable, e.g. member_not_found
    disposition: Disposition
    when: list[Check] = Field(min_length=1)   # all must hold
    message_regex: str | None = None   # pull a human-readable message off the page
    recover: Recovery | None = None
    retryable: bool = False            # for failures: may the caller simply retry?
    description: str = ""

    @model_validator(mode="after")
    def _recoverable_has_recovery(self) -> "Detector":
        if self.disposition == Disposition.recoverable and not self.recover:
            raise ValueError(f"detector {self.code}: recoverable needs `recover`")
        return self


class AppProfile(Model):
    """Everything about a vendor product that is not one specific flow."""

    product: str
    description: str = ""
    sign_on: list[Step] = Field(default_factory=list)
    signed_in: list[Check] = Field(default_factory=list)   # how to tell we are in
    detectors: list[Detector] = Field(default_factory=list)
    # On-screen labels whose neighbouring value is PII ("Name", "SSN"). Names
    # and addresses have no pattern a regex could catch, so the product says
    # where they are; the values found there are redacted from all evidence.
    sensitive_labels: list[str] = Field(default_factory=list)


# --- tenant overlays --------------------------------------------------------------


class Overlay(Model):
    """One tenant's differences from a base capability or profile.

    `patch` is a JSON merge patch (RFC 7386) keyed by step id for steps, so an
    overlay can only touch steps that exist and names them. Lists inside a step
    (strategies, expect) are replaced wholesale — a tenant's locators are its
    own, not an interleaving with the base's.
    """

    tenant: str
    applies_to: str                    # capability name or profile product
    base_versions: str = "*"           # which base versions this was checked against
    steps: dict[str, dict[str, Any]] = Field(default_factory=dict)
    success: list[Check] | None = None
    detectors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    note: str = ""


def merge_patch(base: Any, patch: Any) -> Any:
    """RFC 7386 JSON merge patch."""
    if not isinstance(patch, dict):
        return patch
    result = dict(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def apply_overlay(capability: Capability, overlay: Overlay | None) -> Capability:
    if overlay is None:
        return capability
    if overlay.applies_to != capability.name:
        raise ValueError(f"overlay for {overlay.applies_to} applied to {capability.name}")
    raw = capability.model_dump(mode="json")
    known = {s["id"] for s in raw["steps"]}
    unknown = set(overlay.steps) - known
    if unknown:
        raise ValueError(f"overlay {overlay.tenant} patches unknown steps: {sorted(unknown)}")
    raw["steps"] = [merge_patch(s, overlay.steps.get(s["id"], {})) for s in raw["steps"]]
    if overlay.success is not None:
        raw["success"] = [c.model_dump(mode="json") for c in overlay.success]
    # Approval was for the base; the merged artifact is re-validated, and its
    # fingerprint differs, so approve the overlay explicitly (see store.py).
    return Capability.model_validate(raw)


def apply_profile_overlay(profile: AppProfile, overlay: Overlay | None) -> AppProfile:
    if overlay is None:
        return profile
    raw = profile.model_dump(mode="json")
    raw["sign_on"] = [merge_patch(s, overlay.steps.get(s["id"], {})) for s in raw["sign_on"]]
    raw["detectors"] = [merge_patch(d, overlay.detectors.get(d["code"], {})) for d in raw["detectors"]]
    return AppProfile.model_validate(raw)


# --- the result contract ----------------------------------------------------------


class Status(str, Enum):
    succeeded = "succeeded"            # checkpoint verified, outputs returned
    business_outcome = "business_outcome"   # a declared, legitimate non-success answer
    failed = "failed"                  # could not complete; see failure
    escalated = "escalated"            # waiting on / abandoned by a human


class FailureKind(str, Enum):
    invalid_input = "invalid_input"            # rejected before touching the UI
    not_approved = "not_approved"              # policy: unattended run of a draft
    policy_violation = "policy_violation"      # allowlist or risk gate
    target_not_found = "target_not_found"      # no strategy resolved
    checkpoint_failed = "checkpoint_failed"    # acted, but the expected state didn't appear
    app_error = "app_error"                    # the application reported an error
    unrecognized_state = "unrecognized_state"  # screen matches nothing known
    timeout = "timeout"
    recovery_exhausted = "recovery_exhausted"  # a known condition kept recurring
    human_aborted = "human_aborted"
    escalation_timeout = "escalation_timeout"
    extraction_failed = "extraction_failed"    # found the value but it didn't parse
    surface_error = "surface_error"            # browser crashed, etc.


class Failure(Model):
    kind: FailureKind
    message: str
    step_id: str | None = None
    step_index: int | None = None
    expected: str | None = None
    observed: str | None = None        # redacted
    retryable: bool = False
    evidence: list[str] = Field(default_factory=list)   # paths


class OutcomeResult(Model):
    code: str
    message: str = ""
    step_id: str | None = None


class Event(Model):
    """Something handled along the way: a recovery, a drift, a handoff."""

    kind: Literal["recovered", "drift", "handoff", "retry"]
    step_id: str | None = None
    detail: str = ""


class RunResult(Model):
    run_id: str
    capability: str
    version: str
    tenant: str | None = None
    status: Status
    outputs: dict[str, Any] = Field(default_factory=dict)
    outcome: OutcomeResult | None = None
    failure: Failure | None = None
    events: list[Event] = Field(default_factory=list)
    steps_completed: int = 0
    duration_ms: int = 0
    evidence_dir: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == Status.succeeded
