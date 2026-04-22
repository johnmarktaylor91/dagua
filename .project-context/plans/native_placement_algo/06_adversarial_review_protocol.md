# Adversarial Review Protocol

Adversarial Codex review is a gate between sprints AND a mid-sprint checkpoint,
never a final garnish. This file defines the cadence, templates, escalation,
and verdict handling.

## Cadence

1. **Plan-level (twice, before Sprint 0 begins).** Focus areas:
   - "Attack the scaling assumptions and memory budgets."
   - "Attack the evaluation rubric for overfit risk."
2. **Pre-sprint (once, after sprint file drafted).** Focus: "Attack the sprint
   exit criteria for hidden dependencies or unverifiable claims."
3. **Mid-sprint (once, for any sprint longer than one clock day).**
   Focus: whatever the highest-risk component of that sprint is (see sprint
   file "adversarial review plan" section).
4. **Sprint exit (once, always).** Focus: "Did the sprint meet its exit
   criteria? Did it regress prior gains? Any un-approved scope creep?"

## Dispatch mechanics

- Command: `/codex:adversarial-review --background`
- Invocation: follow CLAUDE.md; use direct `codex exec` only if the plugin
  fails. Verify with `pgrep -fl "codex exec"` within 8s.
- Cwd: `/home/jtaylor/projects/dagua`.
- Output: JSON blob with findings. Use `/codex:result <job-id>` after
  notification.

## Prompt templates

### Plan-level template (scaling + memory)

```
<task>
Adversarial review of the Dagua native placement algorithm meta-plan.
Repo at /home/jtaylor/projects/dagua. Plan lives at
.project-context/plans/native_placement_algo/. Read 00_overview.md,
02_sprint_map.md, 03_test_matrix.md.

Focus: attack scaling assumptions and memory budgets.

Specifically investigate:
- Can the V-cycle coarsening plan actually hit 10M nodes in <=45 minutes on
  GPU within 120 GB RAM? Sketch the autograd memory multiplier by stage.
- Does the multilevel threshold analysis account for the 3-4x autograd budget
  noted in scaling_principles.md?
- Is there any sprint where we assume a runtime that a cost model would reject?
- Is there a graph family at a specific tier we likely cannot reach with the
  stated budget?
- Does the edge routing sprint stack on top of node layout memory rather than
  replacing it?

Output: JSON with verdict (BLOCK / CHANGES_REQUIRED / APPROVE) plus findings
each with severity (CRITICAL/HIGH/MEDIUM/LOW/NOTE), file, concern, and
recommended change.
</task>
<verification_loop>
Cite specific file and line in the plan when flagging a concern.
If a finding is speculative, mark confidence as LOW.
Do not propose reimplementation; propose plan-level changes only.
</verification_loop>
<missing_context_gating>
If a plan section appears missing, reply with a gap flag; do not invent content.
</missing_context_gating>
```

### Plan-level template (overfit + evaluation)

```
<task>
Adversarial review of the Dagua native placement algorithm meta-plan.
Repo at /home/jtaylor/projects/dagua. Read 00_overview.md,
03_test_matrix.md, 04_evaluation_rubric.md.

Focus: attack the evaluation rubric for overfit risk.

Specifically investigate:
- Does the iteration/held-out/rolling split actually prevent overfit?
- Can a clever engineer tune the rubric coefficients to win without improving
  real layout quality?
- Is the HJ protocol rate-limited but not starved? Could we miss a visual
  regression?
- Are composite weights proportional to real perceptual importance, or are
  they arbitrary?
- Is the anti-overfit gap check at Sprint 9 stringent enough?

Output: JSON verdict + findings, same format as the scaling review.
</task>
<verification_loop>
Cite specific lines. Provide one concrete counterexample per CRITICAL finding.
</verification_loop>
<missing_context_gating>
Flag gaps rather than invent.
</missing_context_gating>
```

### Plan-level template (Frankenstein risk)

This one is optional for Sprint 0-1; mandatory before Sprint 3.

```
<task>
Adversarial review of the Dagua native placement algorithm plan.
Focus: attack the hybrid differentiable / classical integration for
Frankenstein risk. Read 13_sprint_hybrid_classical.md (when created) plus
the architecture doc.

Specifically investigate:
- Will the addition of layer-sweep and Brandes-Kopf ops force us to break
  the SolveState invariant or add new typed fields?
- Is the warm-start path a true warm-start, or are we discarding its positions?
- Does the classical polish step undo differentiable gains on the held-out set?
- How does the flex/pin system survive the hybrid sequence?

Output: JSON verdict + findings.
</task>
<verification_loop>
Cite lines. Provide one concrete failure scenario per HIGH finding.
</verification_loop>
```

### Pre-sprint template

```
<task>
Adversarial review of Sprint N spec at
.project-context/plans/native_placement_algo/<file>.md.

Focus: attack entry/exit criteria for hidden dependencies, unverifiable claims,
or scope creep.

Specifically investigate:
- Does every exit criterion have a runnable command or assertion?
- Does the test plan actually test what the goal promises?
- Can rollback actually restore Sprint N-1 state?
- Is there a failure mode not mentioned that is >10% likely?

Output: JSON verdict + findings.
</task>
<verification_loop>
Cite lines. Suggest a concrete test or assertion for each gap.
</verification_loop>
```

### Sprint-exit template

```
<task>
Adversarial review of Sprint N completion. Repo at /home/jtaylor/projects/dagua.
Read .project-context/plans/native_placement_algo/sprint_<N>_exit_note.md,
eval_output/native_algo/sprint_<N>_exit/metrics.json, and the diff of all
commits on branch feat/native-algo-sprint-<N>.

Focus: did the sprint meet its declared exit criteria without unapproved
scope creep or smuggled assumptions?

Specifically investigate:
- Every exit criterion: is it verifiable from the committed artifacts? Verify
  at least three.
- Were any prior sprint gains regressed? Cross-reference
  eval_output/native_algo/sprint_<N-1>_exit/metrics.json.
- Any file-level change outside the sprint's declared scope? If so, is it
  explained?
- Any test turned red and silently disabled?
- Any configuration change that weakens the evaluation?

Output: JSON verdict (BLOCK / CHANGES_REQUIRED / APPROVE) plus findings.
</task>
<verification_loop>
Cite file and line for every finding. Cross-check numerical claims against
metrics.json.
</verification_loop>
<missing_context_gating>
If metrics.json appears incomplete, flag as gap; do not rely on best-case
interpolation.
</missing_context_gating>
```

## Verdict handling

| Verdict | Sprint action |
|---------|---------------|
| APPROVE | Sprint exit confirmed. Proceed to next sprint. |
| CHANGES_REQUIRED | Fix CRITICAL and HIGH findings, re-run exit review. One retry allowed. |
| BLOCK | Sprint is not exited. Retrospective required. Escalate to user. |

Two consecutive BLOCK verdicts across any sprint boundary trigger a forced
pause and re-plan.

## Escalation path

1. Codex implementation and Codex adversarial review disagree:
   - Fresh Codex thread with both positions pasted.
2. That third thread cannot arbitrate:
   - User is notified via iMessage with the three positions and asked to
     break the tie.
3. Claude NEVER arbitrates silently. If Claude has a view on the disagreement,
   state it in the escalation message but do not make the decision.

## Archival

All adversarial review outputs archived at
`.project-context/plans/native_placement_algo/reviews/<sprint>_<focus>_<timestamp>.json`.
The sprint-exit note cites the review file(s) by filename.

## Reviewer diversity (from 2026-04-22 adversarial finding)

Both LLM reviewers today share a model family, which creates rubric
circularity: the same agent class defining the rubric is grading it.

Mitigations, in order:
1. Different harnesses: Codex (OpenAI) for one, Claude subagent for the other.
   Already done in protocol above.
2. Divergent prompt framing: one reviewer attacks "as a paying user who
   hates this layout"; the other rates against Purchase 1997 with citations.
   Prompts documented in each sprint file under "Adversarial review plan."
3. Blind human spot-checks every two sprints on random non-flagship graphs
   (see 04_evaluation_rubric.md HJ protocol additions).
4. If agreement rate exceeds 80% across a full sprint (suspiciously
   agreeable), force a Level-3 user review on a stratified sample.

## Plan-level review log (2026-04-22)

Two adversarial reviews completed at plan drafting.

- **plan_scaling_20260422_130212.json** -- verdict BLOCK.
  2 CRITICAL, 3 HIGH, 2 MEDIUM, 1 NOTE. Fixes incorporated into 01, 02, 03,
  08, 09 in this revision pass.
- **plan_overfit_20260422_130235.json** -- verdict BLOCK.
  1 CRITICAL, 6 HIGH, 3 MEDIUM. Fixes incorporated into 03, 04, 06, 08, 09.

Findings not yet addressed (parked, lower priority):
- L1 LLM reviewer drift on slow aesthetic regressions (R7 mitigation is
  "ping every 2 sprints"; implemented in 04).
- Family weights may still suppress hard cases even with veto bars; monitor
  in Sprint 9. Can add per-family dashboards if trouble emerges.
- Absolute P1 / P2 tuning bars; Sprint 9 decides.

## What adversarial reviews are NOT

- Not a quality score. Not a coverage metric. Not a formal proof.
- Not a replacement for unit or integration tests.
- Not a judgment of "the design is bad" -- they flag risks, not aesthetics.
- Not optional. Skipping any listed review is a process failure.
