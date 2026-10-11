# Evidence

These are real runs against the CoreLink mock app. Discovery used `qwen3-vl:30b`
on-prem via Ollama, and replay used no model at all. Each run directory
(`<timestamp>-<kind>-<id>/`) holds:

- **`log.jsonl`**: one redacted event per line, covering what was done, why, and
  where. For discovery, the model's decisions are logged with its reasoning.
- **`NN-*.png`**: masked screenshots. Names, SSNs, dates and amounts are blacked
  out before the file is written. Discovery takes one per turn; replay takes
  them at failures, outcomes, approvals and the end.
- **`NN-*.aria.txt`**: a redacted accessibility snapshot of every frame, captured
  at failures and at the end of discovery.
- **`result.json`**: the run result. Output values are replaced by their type.

Each scenario also has a `*.console.txt`, the command's stdout. Stdout is the
caller's channel, so it **does** contain output values. Those values are
fictional, and they are the only place outputs appear: Rote's own logs, results
and screenshots never contain them.

## Saved artifacts

[artifacts/](artifacts/) holds:

- **The two discovered capabilities, as recorded.**
  - [read_savings_balance](artifacts/corelink.member.read_savings_balance-1.0.0.yaml)
  - [open_savings_club](artifacts/corelink.share.open_savings_club-1.0.0.yaml). This
    copy includes the approval record from `rote approve` (run 13), bound to
    its fingerprint.
- **The Lakeshore tenant overlay.**
- **`rote show` review renderings.**
- **The agent-facing tool catalog** (`catalog.json`).

## Discovery runs (LLM-driven)

| Scenario | Run | What happened |
| --- | --- | --- |
| Read savings balance | [discovery-read-balance](discovery-read-balance.console.txt), `20261011T023321Z-discovery-3c83bb/` | Sign-on was handled by the profile. Then 5 model turns: menu → fill member number as `{{inputs.member_id}}` → Inquire → extract the balance → done. Every locator was verified on the live page. The model's success text contained data, so the system substituted the verified heading. |
| Open a share (irreversible) | [discovery-open-share](discovery-open-share.console.txt), `20261011T025202Z-discovery-a817bf/`, [operator](discovery-open-share.operator.txt) | 12 turns through a 3-screen form. Clicking **Post** was classified `commit` and escalated `approval_required` to the operator console. The operator (bot) approved, the run continued, and the confirmation number was extracted out of a sentence with a derived regex. The recorder flagged a literal value for review. |

## Replay runs (deterministic, no model)

| # | Scenario | Result |
| --- | --- | --- |
| 01 | member 20001, a different input from discovery | `succeeded`, balance 87012.00 |
| 02 | member 99999 | `business_outcome: member_not_found` + host message |
| 03 | member `12a` | `failed: invalid_input`; rejected before the browser navigated anywhere |
| 04 | member 77777 (employee account) | `business_outcome: access_restricted` |
| 05 | system-notice interstitial injected | `succeeded`; `recovered: system_notice via click` |
| 06 | session expired before Inquire | `succeeded`; `recovered: session_expired via reauthenticate` (flow restarted) |
| 07 | host system error injected | `failed: app_error`, `retryable: true`, screenshot + ARIA snapshot |
| 08 | 1.5 s latency on every request | `succeeded`; no fixed sleeps, checkpoints polled |
| 09 | an **unknown** dialog the profile can't classify | `checkpoint_failed` → escalated to the console → operator took control, attached over CDP to the same browser, clicked "Remind Me Later" (captured) → handed back → checkpoint verified → `succeeded`. See [operator transcript](replay-09-handoff-unknown-dialog.operator.txt) and the `control` events in the log. |
| 10 | tenant B (Lakeshore) with the base capability only | 3 `drift` events (fallback locators used), then `failed: extraction_failed`: a position-based fallback found `S00`, and the `money` type refused it |
| 11 | tenant B + `overlays/lakeshore.yaml` | `succeeded` |
| 12 | open share, **draft**, unattended | `failed: not_approved` at `post_new_share` (nothing posted) |
| 13 | open share after `rote approve` | `succeeded`, confirmation number returned |
| 14 | open share with a 1.00 deposit | `business_outcome: validation_error`, "MINIMUM OPENING DEPOSIT FOR SAVINGS CLUB IS &lt;amount&gt;". Host messages are redacted before they're returned, which is conservative: here it hides a harmless minimum. |

`scripts/make_evidence.sh` reproduces the replay runs.
