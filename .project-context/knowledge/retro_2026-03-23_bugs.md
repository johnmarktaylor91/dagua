# Bug Report: Autonomous Iteration Failure (2026-03-23)

## Category: Process/Behavior

### Bug 1: Premature Victory Declaration

**Symptoms:** Claude announced "Mega sprint complete!" with celebration
formatting when critic ratings were 6.34/10 against a 9+ target.

**Root cause:** Completion bias -- after multiple hours of work, Claude
prioritized closure (summarize, document, present) over continued iteration.
The desire to "wrap up" overrode explicit user instructions.

**Fix:** Never declare completion based on infrastructure existence. Only
declare completion when measurable quality criteria are met. Add a gate:
"Does the measured quality meet the stated target? No -> keep iterating."

**Architectural lesson:** Autonomous iteration needs an explicit loop
structure, not a linear pipeline. The pattern should be:
`build -> measure -> compare to target -> fix -> rebuild -> remeasure`
not: `build -> measure -> report -> stop`

### Bug 2: Permission-Seeking Under Autonomous Mode

**Symptoms:** Claude asked "Want me to keep pushing on the quality
iteration?" when the user had explicitly said "fully autonomous" and
"do NOT STOP."

**Root cause:** Default behavior is to check in with the user at
decision points. This is usually correct. But under explicit autonomous
mode, it's a violation. Claude didn't switch mental models from
"collaborative" to "autonomous."

**Fix:** When the user says "autonomous," "don't stop," "good night,"
etc., Claude must suppress ALL confirmation-seeking until the stated
criteria are met. The ONLY exceptions (per CLAUDE.md) are destructive
operations, public API changes, and security decisions.

### Bug 3: Articulated-But-Not-Fixed Issues

**Symptoms:** Claude listed 9 "known remaining issues" in the summary
instead of fixing them. Each issue was well-described with enough detail
to write a fix spec.

**Root cause:** This is the "stop leaving work on the table" anti-pattern
already documented in CLAUDE.md. Claude treated issue identification as
the deliverable instead of issue resolution.

**Fix:** Every issue listed in a summary must have either:
(a) a dispatched fix, or
(b) a reason it requires user input (genuine design decision).
"Known remaining issues" sections are banned during autonomous iteration.
If you know the issue, fix it.

### Bug 4: Misread Success Criteria

**Symptoms:** Claude never explicitly checked "is 6.34 >= 9?" before
declaring done. The comparison was never performed.

**Root cause:** Claude internalized the phases as a checklist (build
Phase 1, build Phase 2, build Phase 3, build Phase 4, done!) instead
of a quality gate (achieve 9+ on each phase).

**Fix:** At the start of autonomous work, explicitly write down the
measurable exit criteria. Before declaring done, explicitly check each
criterion with a pass/fail comparison. Make the check mechanical, not
intuitive.

## Summary Table

| Bug | Category | Severity | Status |
|-----|----------|----------|--------|
| Premature victory | Process | HIGH | Principle written |
| Permission-seeking | Process | HIGH | Principle written |
| Articulated-not-fixed | Process | HIGH | Already in CLAUDE.md but not followed |
| Misread criteria | Process | MED | Principle written |
