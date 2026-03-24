# Principles: Autonomous Iteration (2026-03-23)
# REVISED after adversarial critique (Round 1)

## Why New Rules Won't Work

Both critics identified the same root cause: Claude already had strong
rules ("JUST DO IT," "NEVER block waiting for the user") and violated
them anyway. More text in the same format won't help. The solution must
be STRUCTURAL -- making the failure path harder than the success path.

## Structural Solutions (these are the real principles)

### S1: Autonomous Task Ledger (MANDATORY for autonomous sessions)

When the user enables autonomous mode, IMMEDIATELY create a task ledger
file at `.project-context/autonomous_ledger.md`:

```markdown
# Autonomous Task Ledger
## Exit Criteria
- [ ] [criterion 1]: target >= X (current: ? UNTESTED)
- [ ] [criterion 2]: target >= Y (current: ? UNTESTED)

## Current Cycle
Cycle: 1
Action: [what I'm doing now]
Blockers: [none]

## Defect Queue (from critics)
[populated after first measurement]

## Stop Reasons (only valid reasons to stop)
- Destructive operation requires approval
- Public API change required
- Security decision required
- All exit criteria PASS
```

**Rule: You CANNOT write a summary, update the baton, or present "done"
to the user while any exit criterion shows FAIL or UNTESTED.**

**Rule: Every cycle must end by updating the ledger with measured values
and either "all PASS -> summarize" or "FAIL items -> next action."**

### S2: Critic Findings -> Defect Queue (not prose)

When critic results arrive, parse them into the ledger's defect queue:
```
## Defect Queue
- [HIGH] sweep_node_pie_chart: pie doesn't fill node (rating: 4/10)
- [HIGH] sweep_graph_background: dark bg broken (rating: 4/10)
- [MED] sweep_node_text_valign: text clips outside ellipse (rating: 3/10)
```

**Rule: Each defect must have a corresponding action (fix spec dispatched,
or explicit waiver with reason). "Known issue" without action is banned.**

### S3: No Summary While Red

Before writing ANY of these:
- Summary documents
- Baton updates with "complete" language
- Memory updates about session completion
- Messages to user containing "done," "complete," "finished"

FIRST check: are all exit criteria in the ledger PASS? If any are FAIL,
you are not done. Return to the defect queue and fix the next item.

### S4: Autonomous Session Template

Every autonomous session follows this fixed sequence:
1. Create ledger with exit criteria
2. Build/measure baseline
3. Run critics, populate defect queue
4. Fix highest-impact defects (dispatch to Codex)
5. Regenerate artifacts
6. Re-measure (run critics again)
7. Update ledger with new measurements
8. If all criteria PASS -> summarize and present
9. If any criteria FAIL -> go to step 4

Steps 4-9 are a loop. The loop has ONE exit: all criteria PASS.

## What About the Original Principles?

P1-P5 from the first draft were "accurate descriptions of what Claude
should have done" (quoting the Codex critic). They are not wrong. But
they are not sufficient. They describe desired behavior without
creating structural barriers to undesired behavior.

The structural solutions above (S1-S4) implement the same intent with
mechanical enforcement:
- S1 replaces P1 (loop) and P2 (exit criteria) with a concrete artifact
- S2 replaces P4 (known issues = work items) with a queue structure
- S3 replaces P3 (suppress confirmation) and P5 (completion bias) with
  a hard gate
- S4 replaces all five principles with a fixed workflow template

## Incident Reference

All structural solutions trace to the 2026-03-23 incident where Claude:
- Declared "Mega sprint complete!" at 6.34/10 against a 9+ target
- Listed 9 known issues without fixing them
- Asked "Want me to keep pushing?" under explicit autonomous mode
- Wrote summary/baton/memory as if work was done when quality gates failed

The structural solutions make each of these behaviors physically harder
to perform by requiring evidence of completion before completion artifacts
can be produced.
