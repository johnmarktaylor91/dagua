#!/usr/bin/env bash
# complete-session.sh -- Mandatory gate for autonomous session completion.
#
# Reads .project-context/autonomous_gate.json, checks that ALL exit
# criteria have status "pass". Refuses completion if any are "fail"
# or "untested".
#
# This script is the ONLY legal path to declaring an autonomous session
# complete. Writing summaries, updating batons, or presenting "done"
# to the user without running this script first is a process violation.
#
# Usage:
#   scripts/complete-session.sh
#
# Exit codes:
#   0 -- all criteria PASS, session may be completed
#   1 -- one or more criteria FAIL or UNTESTED
#   2 -- no gate file found

set -euo pipefail

GATE=".project-context/autonomous_gate.json"

if [ ! -f "$GATE" ]; then
    echo "ERROR: No autonomous gate file at $GATE"
    echo "Not in autonomous mode, or gate was never created."
    exit 2
fi

echo "=== Autonomous Session Gate Check ==="
echo ""

# Use python to parse JSON and check criteria
python3 - "$GATE" << 'PYEOF'
import json, sys

gate_path = sys.argv[1]
with open(gate_path) as f:
    gate = json.load(f)

print(f"Mode: {gate.get('mode', 'unknown')}")
print(f"Started: {gate.get('started', 'unknown')}")
if 'user_directive' in gate:
    print(f"Directive: {gate['user_directive']}")
print()

criteria = gate.get("exit_criteria", [])
if not criteria:
    print("WARNING: No exit criteria in gate file.")
    print("SESSION NOT COMPLETE: Add measurable exit criteria.")
    sys.exit(1)

total = len(criteria)
passed = 0
failed = 0
blocked = 0
untested = 0

for c in criteria:
    name = c.get("criterion", "unnamed")
    status = c.get("status", "untested").lower()
    measured = c.get("measured_value", "not measured")
    target = c.get("target", "unspecified")
    source = c.get("measurement_source", "self-reported")

    if status == "pass":
        passed += 1
        icon = "PASS"
    elif status == "fail":
        failed += 1
        icon = "FAIL"
    elif status == "blocked":
        blocked += 1
        icon = "BLOCKED"
    else:
        untested += 1
        icon = "UNTESTED"

    print(f"  {icon}: {name}")
    print(f"         target={target}  measured={measured}  source={source}")

print()
print(f"--- Results ---")
print(f"Total: {total}  Passed: {passed}  Failed: {failed}  Blocked: {blocked}  Untested: {untested}")
print()

if failed > 0 or untested > 0 or blocked > 0:
    print("SESSION NOT COMPLETE")
    print()
    print(f"You have {failed} failing, {blocked} blocked, and {untested} untested criteria.")
    print("Return to the defect queue and fix the next item.")
    print("Do NOT write summaries, update batons, or declare done.")
    sys.exit(1)

print("ALL CRITERIA PASS")
print()
print("Session may be completed. You may now:")
print("  - Write the summary document")
print("  - Update the baton")
print("  - Present results to the user")

# Write proof artifact
import datetime
proof = {
    "completed_at": datetime.datetime.now().isoformat(),
    "criteria": criteria,
    "all_pass": True,
}
proof_path = gate_path.replace("autonomous_gate", "completion_proof")
with open(proof_path, "w") as f:
    json.dump(proof, f, indent=2)
print(f"\nProof artifact written to: {proof_path}")
PYEOF

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Gate file retained for audit: $GATE"
fi

exit $exit_code
