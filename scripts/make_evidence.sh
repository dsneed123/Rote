#!/usr/bin/env bash
# Reproduce the replay evidence in evidence/. Needs the mock app running:
#   python -m rote app --port 8765            (tenant a)
#   python -m rote app --port 8766 --tenant b (tenant b, for the overlay demo)
# Discovery (needs the model) is a separate step; see README.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CAP=corelink.member.read_savings_balance
OUT=evidence
run() {  # name, args...
  local name=$1; shift
  echo "== $name"
  $PY -m rote replay "$CAP" --runs "$OUT" "$@" > "$OUT/$name.console.txt" 2>&1
  echo "   exit $? — $(grep -m1 '"status"' "$OUT/$name.console.txt")"
}
curl -s -XPOST localhost:8765/__control/faults -d '{"reset": true}' >/dev/null

run replay-01-success            --operator none --input member_id=20001
run replay-02-member-not-found   --operator none --input member_id=99999
run replay-03-invalid-input      --operator none --input member_id=12a
run replay-04-access-restricted  --operator none --input member_id=77777
run replay-05-notice-recovered   --operator none --input member_id=12345 --inject notice
run replay-06-session-expired    --operator none --input member_id=12345 --inject expire@submit_member_inquiry
run replay-07-host-error         --operator none --input member_id=12345 --inject error
run replay-08-slow-host          --operator none --input member_id=30003 --inject slow=1500
curl -s -XPOST localhost:8765/__control/faults -d '{"reset": true}' >/dev/null

# Handoff: an unknown dialog the profile can't classify -> escalate to the
# operator console -> the operator (bot) takes the live session over CDP.
echo "== replay-09-handoff-unknown-dialog"
$PY scripts/operator_bot.py --console http://127.0.0.1:8770 --policy dismiss \
    > "$OUT/replay-09-handoff-unknown-dialog.operator.txt" 2>&1 &
BOT=$!
$PY -m rote replay "$CAP" --runs "$OUT" --operator console --input member_id=12345 \
    --inject unknown_dialog > "$OUT/replay-09-handoff-unknown-dialog.console.txt" 2>&1
echo "   exit $? — $(grep -m1 '"status"' "$OUT/replay-09-handoff-unknown-dialog.console.txt")"
kill $BOT 2>/dev/null; wait $BOT 2>/dev/null

# Tenant b: same vendor product, different configuration.
run replay-10-tenant-b-base-only --operator none --input member_id=12345 --base-url http://127.0.0.1:8766
run replay-11-tenant-b-overlay   --operator none --input member_id=12345 --base-url http://127.0.0.1:8766 --tenant lakeshore

# The irreversible capability: a draft may not commit unattended; approved, it may.
OPEN=corelink.share.open_savings_club
echo "== replay-12-open-share-draft-unattended"
$PY -m rote replay $OPEN --runs "$OUT" --operator none --input member_id=20001 --input nickname=XMAS \
    --input deposit=40.00 > "$OUT/replay-12-open-share-draft-unattended.console.txt" 2>&1
echo "   exit $? — $(grep -m1 '"kind"' "$OUT/replay-12-open-share-draft-unattended.console.txt")"
$PY -m rote approve $OPEN --by "demo-reviewer" > "$OUT/approve-open-share.txt" 2>&1
echo "== replay-13-open-share-approved"
$PY -m rote replay $OPEN --runs "$OUT" --operator none --input member_id=20001 --input nickname=XMAS \
    --input deposit=40.00 > "$OUT/replay-13-open-share-approved.console.txt" 2>&1
echo "   exit $? — $(grep -m1 '"status"' "$OUT/replay-13-open-share-approved.console.txt")"
echo "== replay-14-open-share-below-minimum"
$PY -m rote replay $OPEN --runs "$OUT" --operator none --input member_id=30003 --input nickname=TINY \
    --input deposit=1.00 > "$OUT/replay-14-open-share-below-minimum.console.txt" 2>&1
echo "   exit $? — $(grep -m1 '"code"' "$OUT/replay-14-open-share-below-minimum.console.txt")"
