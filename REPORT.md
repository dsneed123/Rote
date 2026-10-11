# Rote — Design Report

## 1. Architecture

Rote is a single-process Python library and CLI. It has two paths that share
one surface layer and one policy guard:

```
              ┌──────────── discovery (once) ────────────┐   ┌──── replay (production path) ────┐
 goal+params ─▶ agent.py: observe → LLM picks ONE action ─▶ recorder.py ─▶ capability.yaml ─▶ replay.py ─▶ RunResult
              │   known conditions handled by profile    │   (template,      (draft → review    │ detectors classify
              │   risky steps → human approval           │    lint, verify)   → approved)       │ every surprise
              └──────────────────┬───────────────────────┘                                    └──────┬───────────
                                 ▼                                                                   ▼
           surface.py (WebSurface/Playwright) ── every action through policy.Guard ── evidence.py (redacted)
                                 ▲
           handoff.py / console.py: SessionControl (one owner), escalators, operator console, CDP attach
```

Key decisions:

- **Target:** a local mock core-banking app built to be hostile (`mockapp/corelink.py`):
  framesets, table layouts, field names like `f_0012`, no ids, and labels in
  neighbouring cells instead of `<label for>`. Its faults can be injected:
  session expiry, a system-notice interstitial, an unknown dialog, slowness, a
  host error. A second tenant configuration has different labels, column order
  and branding. A public site can't produce "record not found" or "session
  expired" on demand, and the brief weights runtime errors over drift.
- **Automation:** Playwright with Chromium. It reaches into every frame, exposes
  accessibility-oriented locators, masks screenshots, and lets a second client
  attach to the *same* browser over CDP. That last capability is the human
  handoff.
- **Model:** `qwen3-vl:30b` (vision), served on-premises with Ollama on a GX10.
  For regulated data that is a feature: screens never leave the institution.
  The trade-off is a weaker model than a frontier CUA, so the design asks the
  model only for judgement and verifies everything it says (see §2 and §3).
- **The model never writes selectors or coordinates.** It sees numbered
  controls (role, name, neighbouring label, value), each frame's text and a
  masked screenshot, and answers with one schema-constrained JSON action. The
  surface turns "element 7" into ranked locators, and keeps only those that
  resolve uniquely to that same element on the live page.
- **Not built:** services and queues. The seams where they would go are the
  `Surface` protocol, the `Escalator` interface and `Store`.

Live-model findings shaped the agent. Ollama returned JSON in the `thinking`
field when thinking was off. The model left `value` empty on extract until
every schema field was required. And it put data in its "success text". Each
is handled by verifying, then substituting a checked alternative, rather than
by trusting the model.

## 2. Artifact schema

The artifact is three documents with three owners (`rote/schema.py`):

- **Capability:** one flow, recorded by discovery and approved by a human. It
  has a dotted name, a semver version, typed `inputs` (type, pattern, enum,
  **sensitivity**), typed `outputs` (type, optional extraction regex,
  sensitivity), declared business `outcomes`, ordered `steps` and a `success`
  checkpoint. It also carries `approval` and `provenance` (model, run id,
  goal, number of human-performed steps).
- **AppProfile:** one vendor product, shared by every capability and tenant.
  It holds the sign-on steps (credentials as `{{secrets.x}}`), the runtime
  **detectors**, and `sensitive_labels` (where on screen the PII sits).
- **Overlay:** one tenant's differences, as a JSON merge patch keyed by
  **stable step id**.

A step is `{id, action, target, value, output, risk, expect[], note}`. A
target is a frame reference plus an **ordered list of locator strategies**:
`role`+name, `label_near` (the text in the neighbouring cell), `text`,
`table_cell` (row key + column header) and `css` as a last resort. Values are
templates (`{{inputs.member_id}}`), never literals from the discovery run.
Checkpoints are typed checks (text, regex, element, url) and can be negated or
templated. For example, the success check requires the *requested* member
number on screen, not just any member page.

Why this shape:

- **"No such member" is a fact about the product, not about one flow.** Putting
  detectors in the profile means 20 capabilities don't each relearn session
  expiry.
- **Tenants differ by configuration, not by flow.** Small overlays addressed by
  step id beat hundreds of copies.
- **Approval is bound to a content fingerprint** (sha256 of everything but
  approval and provenance). Edit a step and the capability is a draft again.
  Approved versions are immutable on disk. An overlaid capability has its own
  fingerprint, so a tenant specialisation needs its own review.
- **Strict parsing.** Unknown keys fail loudly, so a typo can't silently drop a
  fallback locator. Every template must reference a declared input. Every
  declared output must be extracted. A commit step forces
  `side_effects: irreversible`.
- **Reviewable and callable.** `rote show` renders it for a human, and
  `rote catalog` emits each capability as a function-calling tool with its
  input schema, output types and possible business outcomes.
- **Provenance only.** The model transcript stays in the run evidence; the
  artifact links to it by run id.

The real artifact is in
[library/capabilities/…/1.0.0.yaml](library/capabilities/corelink.member.read_savings_balance/1.0.0.yaml).

## 3. Determinism & error handling

**Targeting.** Strategies are tried in order. The first one that resolves to
**exactly one visible element** wins. Zero matches, or several, falls through
to the next strategy, because ambiguity is never resolved by "take the first".
If any strategy other than the first is used, a `drift` event goes into the
result. Locators are ranked by what a human operator would use, since that is
what survives upgrades and rebranding. The role/name/label logic is a single
JS library used at record time and at replay time, so what was recorded is
what gets matched.

**Waiting.** After an action, Rote waits for the network to go quiet, then for
every frame's load event. A click returns before its navigation starts, so a
plain "wait for load" passed too early. Checkpoints are polled with timeouts,
and there are no fixed sleeps.

**Error taxonomy.** Every place the screen could have gone somewhere
unexpected — resolving a target, waiting for a checkpoint, the final success
check — asks the profile's detectors what is on screen. They are polled while
waiting, so a "not found" page ends the wait immediately. Each detector has
one disposition:

| Disposition | Examples | Response | Result |
| --- | --- | --- | --- |
| business | member_not_found, access_restricted, validation_error | stop, return code + on-screen message | `business_outcome` |
| recoverable | system_notice, session_expired, signed_out | declared remedy (click / reauthenticate / wait), bounded attempts | continues; `recovered` event |
| failure | host_error (retryable), posting_rejected | stop with evidence | `failed: app_error` |
| *(none match)* | unknown dialog, wrong screen | escalate to a human, else fail | `failed`/`escalated` |

Recovery depends on *when* the condition appeared. Before the step acted, the
whole step is retried. After it acted, only the checkpoint is awaited again,
because clicking Inquire twice is wrong. Re-authentication restarts the flow
from the entry point, and is **refused once a commit step has run**: repeating
an irreversible action because a session timed out is exactly the bug this
layer exists to prevent. Host errors are reported as retryable but are never
auto-retried, since whether a submit was posted is the caller's call.

Failures carry `kind`, step id and index, `expected` vs `observed` (a redacted
excerpt of the screen), `retryable`, and evidence paths: a masked screenshot
plus a redacted accessibility snapshot of every frame. Inputs are validated
against their types and patterns before any UI is touched.

**Typed outputs are a correctness check.** In `evidence/replay-10`, tenant B
without its overlay resolved the balance cell through a position-based
fallback and got `S00`. The `money` type rejected it, so the run reported
`extraction_failed` instead of returning a wrong number. That label-based
fallback has since been removed for table cells.

## 4. Heterogeneity & multi-tenant

**Surfaces.** The flow says *what* to do in surface-neutral terms. The
`Surface` protocol says *how*: resolve, perform, read, holds, observe,
locators_for, screenshot, accessibility_snapshot. Nothing above `surface.py`
imports Playwright.

- **Legacy web:** the same WebSurface. The mock app *is* legacy web
  (framesets, no ids), which is why `label_near` and `table_cell` exist.
- **Desktop:** a `DesktopSurface` over UI Automation (Windows) or AT-SPI
  (Linux). Role and name, neighbouring labels and grid patterns are native
  there, so `role`, `label_near` and `table_cell` carry over directly. `css`
  has no equivalent; a coordinates/image strategy would take its place, ranked
  last and flagged in review exactly as `css` is now. Runtime detectors are
  checks over visible text and element presence, which a desktop tree also
  provides.
- **Unchanged:** the schema, the replay engine, the policy guard and the
  handoff model.

**Tenants.** A capability is recorded once against a vendor *product*
(`app.product`, `app.versions`) with tenant-relative paths (`entry: /main`).
Per tenant, Rote replays the base plus an optional overlay. The demo shows
`lakeshore.yaml` patching three steps, and inherits every other step,
checkpoint, output, outcome and the shared profile. Locators were chosen to
absorb the commonest tenant differences: `table_cell` keys by header, not
position, which absorbs column reordering.

**Drift detection and management:**

- **Fallback usage per (tenant, step)** is the early-warning signal. Tenant B
  without an overlay shows `drift` on three steps before anything fails.
- **Typed outputs and checkpoints** turn silent wrong answers into explicit
  failures.
- **An undeclared business outcome** returns correctly but adds a drift event,
  so the contract gets updated.
- **Next:** aggregate drift events per tenant and product version in a store,
  open a review task when a step's primary locator stops matching on a tenant,
  and let a bounded, policy-checked LLM pass propose the overlay patch, as a
  draft needing approval.
- **Product versions:** a capability declares the versions it was verified
  against. A tenant upgrading the vendor product triggers a canary replay
  before unattended runs continue.

## 5. Escalation & handoff

**Detecting "stuck".** Escalation is raised in these cases:

- **Discovery:** the model says `stuck`, there are 3 consecutive rejected or
  failed actions, or a model error.
- **Replay:** `target_not_found`, `checkpoint_failed`, or a recoverable
  condition exceeding its attempts — anything no detector explains.
- **Risky steps:** `approval_required` (see §6).

An intervention carries the run id, the capability or goal, the step id,
description and index, the location of every frame, a masked screenshot, the
CDP endpoint of the live browser, and the allowed resolutions.

**Control model** (`handoff.py`). Each live session has **one owner**:
`automation → awaiting_human → human → automation`. The engine asserts it owns
the session before every action, so it cannot act while a person holds it.
Every transition is logged with who made it and why.

**Taking over the same session.** The browser is launched with a
remote-debugging port. A headed run's window *is* the session. From another
machine, an operator attaches over CDP (`chrome://inspect`, or any CDP
client). Rote's test does exactly that from a separate Playwright client and
verifies the run resumes on the same cookies and page.

An init script in every frame reports the operator's clicks and changes
(redacted; passwords never). They are recorded **only while the owner is the
human**, so the automation's own events aren't misattributed. One automated
fill used to fire its change event on the operator's first click, which is now
fixed. While waiting, the automation thread keeps pumping the browser's event
loop so those events are delivered before control returns.

**Handing back.** The operator resolves with `retry` (re-run the step), `done`
(I did it), `approve`/`deny`, or `abort`. "Done" is **not trusted blindly**:
the stalled step's checkpoint is verified, and extraction still reads the value
itself. Escalation is limited to two interventions per step, then the run ends
`escalated`. No answer within the timeout gives `escalation_timeout`. During
discovery, the human's actions become recorded steps, flagged "verify on first
replay" and counted in provenance.

**Mocked:** the operator console (`console.py`) is a minimal web page: open
intervention, screenshot, Take control, Resolve, history. The control transfer
behind it is real. `scripts/operator_bot.py` plays the operator for the
recorded demo through the same console endpoints and CDP session a person
would use (`evidence/replay-09`). A production console would add operator
identity and authorisation, routing and queues, notifications, and an embedded
co-browsing view (CDP screencast). It would sit behind the same `Escalator`
interface.

## 6. Safety

- **Allowlist, enforced twice.** Every action passes `Guard.check()` inside the
  surface. There is no other path to the UI, whoever chose the action. Origins
  and path regexes are also enforced at the **network layer** with route
  interception, so a click that would leave the allowlist is aborted by the
  browser. The fault-injection route is explicitly blocked as an example of a
  dangerous admin path. Action types and named controls ("close account") can
  be blocked outright.
- **Risk classes.** Actions are `read`, `input`, `submit` or `commit`. The
  class is inferred from the action and the **resolved control's** visible
  name: post, transfer, delete, approve, confirm and so on make it a commit.
  The higher of the inferred and declared risk wins, so a mislabelled artifact
  can't downgrade "Post".
- **Treatment of commits** is configurable per class: allow, approved, confirm
  or block. The default for commit is *approved*. In discovery, a human must
  approve that specific step. In replay, a commit runs unattended only if the
  capability is approved **and its fingerprint is intact**; otherwise it
  escalates for approval, or fails `not_approved` with no operator. A tenant
  can tighten this to *confirm* (a person every time) or *block*. I chose
  approval-gating over blocking because "open a sub-account" is a legitimate
  capability. The invariant is that no unreviewed irreversible action runs
  unattended.
- **Secrets** are referenced as `{{secrets.x}}` and resolved from the
  environment at the moment of typing. They are never in an artifact, a log or
  a prompt.
- **Redaction.** One Redactor masks SSNs, Luhn-valid card numbers, emails,
  phones, dates and amounts. It also masks **known values** by name: secrets,
  PII inputs, extracted outputs (`<pii:primary_savings_balance>`), and values
  harvested from beside the profile's `sensitive_labels` (Name, SSN, Birth
  Date). That last source exists because names have no pattern; it closed a
  real leak found in the first evidence snapshot.
  - **Screenshots** mask those elements before the file is written. The model
    is sent the masked image.
  - **Run results on disk** replace output values with their type. Only the
    caller's return value has them.
  - **Playwright traces** (raw DOM) are off by default.
- **Artifact lint.** The recorder refuses to save a capability whose values,
  checkpoints, descriptions, locators or notes contain anything the redactor
  flags. The model's notes are templated and redacted, after review of the
  first real artifact found an amount quoted in one.

**Limits.**

- PII in free text with no label and no pattern (memo fields) still gets
  through. The fix is NER, or allowlist redaction, where evidence keeps labels
  and structure and masks every value.
- Screenshot masking depends on finding the text in the DOM, so canvas or
  image-rendered text isn't masked.
- Risk inference is keyword-based. Controls named with icons only need a
  declared risk in the profile.
- The policy trusts the browser's view of the origin.
- The operator console has no authentication.

## 7. Cuts

**Cut deliberately:**

- A desktop surface (designed in §4, not built).
- Multi-tenant storage, queues and services: files on disk, single process.
- A real co-browsing console (CDP plus a minimal page instead).
- Operator identity and authorisation.
- Automated drift aggregation.
- Auto-generating overlays.
- A bounded LLM recovery step during replay.
- Video recording of runs.
- Model-transcript replay or evaluation harnesses.
- Parallel runs: one browser per run, which is fine at this scale.

**Built beyond the core:**

- The agent-facing catalog (`rote catalog`, tool schemas with outputs and
  outcomes).
- Draft → approved gating bound to a content fingerprint.
- One real cross-tenant demonstration: a base capability, a failing replay on
  the variant, and a passing overlay.

**Next, in order:**

1. A replay stability job: N canary replays per tenant after each vendor
   upgrade, with per-step drift and flakiness scores feeding approval.
2. Bounded assisted recovery: on `target_not_found`, one policy-checked LLM
   proposal for that single step, recorded as an overlay *draft*, never applied
   unattended.
3. A DesktopSurface over UIA.
4. NER-based redaction for free text.
5. A console with operator auth and a CDP screencast.
6. Detector learning: when an operator resolves an unknown dialog, offer to add
   a recoverable detector to the profile, so the next run handles it alone.
