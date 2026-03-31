# Retro 2026-03-29: Operational Principles (FINAL -- both critics confirmed)

## The Meta-Failure

7.5 hours wasted because the agent never compared its INTENT against the
tool's ACTUAL WORK PLAN, then went passive when the consequences arrived,
then didn't validate outputs before proceeding to the next step.

## Principle 1: Work Plan Verification (MASTER)

**Rule:** Before any operation estimated at >5 minutes, verify the work
plan matches intent:

1. State the intent explicitly: "Re-run 20 changed reimpl engines"
2. Construct the command
3. Determine what the command WILL ACTUALLY DO -- not what you think
   it does. Read source, dry-run, or parse startup output.
4. Compare plan against intent. If they diverge, STOP and fix.

**Composition safety:** When combining 2+ flags that affect scope, trace
each flag's effect independently, then trace their COMBINED effect. Watch
for: flags that override each other, defaults that change meaning when
combined, scope-expanding flags (--all, --variants, --recursive) that
multiply with mode flags (--resume, --force).

**Incident:** `--resume` means "skip completed entries for ALL engines."
`--engines` scopes which engines to include. Without `--engines`, resume
runs everything. One `--help` read would have revealed this. 6+ hours
wasted.

## Principle 2: Monitoring With Cadence

**Rule:** Every long-running operation gets a monitoring cadence tied to
its cost tier:

| Duration | Check Cadence | What to Check |
|----------|--------------|---------------|
| 5-30 min | Once at halfway | Log output: what is running? |
| 30 min-2 hr | Every 15 min | Log: what engine/variant? Matches intent? |
| 2+ hr | Every 30 min + kill criteria | Log + rate. Kill if wrong target. |

At every check: read the ACTUAL log, not just a count. Identify WHAT
the process is doing RIGHT NOW. Compare against what it SHOULD be doing.
"Still running" is never an acceptable report -- must include what is
running and whether it matches intent.

**Kill criteria (define BEFORE launching):** "If engine X hasn't appeared
in the log by time T, kill and investigate." "If rate drops below Y/min,
investigate within 2 minutes."

**Incident:** User asked "now?" 10 times. Agent reported result counts
without reading the log. Log clearly showed deterministic reference
engines being backfilled. 2+ hours of passive polling.

## Principle 3: Validate Outputs Before Proceeding

**Rule:** After any pipeline step completes, sanity-check its output
before feeding it to the next step. Takes 30 seconds. Check:

1. Expected number of rows/groups/entries
2. No unexpected NaN/empty/zero columns in key fields
3. At least one spot-check of a known-good case end-to-end
   (e.g., a graph where you KNOW the expected verdict)

**Incident:** Analysis run with `--skip-metrics` left all TOST columns
as NaN. Verdict logic produced 66 divergent (should have been ~30).
A 10-second check of one row's TOST values would have caught this
before the 12-minute recompute run.

## Principle 4: Cost-Proportional Pre-Flight

**Rule:** Verification effort scales with operation cost:

- <5 min: glance at the command, run it
- 5-30 min: verify scope (P1), check first 30s of output
- 30 min-2 hr: full P1 verification, monitor per P2, define kill criteria
- 2+ hr: P1 + dry-run on 2 items + P2 cadence + explicit kill criteria
- After a failed run: verify no stale intermediate state before relaunch
  (pycache, partial outputs, lock files, corrupted cache entries)

**Blast radius modifier:** If the operation blocks the user's timeline
or consumes shared resources, apply MAXIMUM scrutiny regardless of
duration. A 30-min benchmark during an autonomous session where the user
is sleeping has higher blast radius than a 4-hour benchmark during idle
time.

## Principle 5: Encode in Tools, Not Willpower

**Rule:** When a principle can be enforced by a wrapper script, pre-flight
check, or runtime assertion, implement the enforcement. Agent discipline
fails under cognitive load. Tool guardrails don't.

Every principle should have a corresponding guardrail where feasible:
- P1 -> scripts should print work plan in first 3 lines of output
- P2 -> scripts should emit progress with engine/variant names, not just counts
- P3 -> analysis scripts should print summary stats before writing CSVs

**Obligation:** Every P1-P4 rule that can be automated MUST have a
corresponding enforcement mechanism. This is not aspirational -- if you
identify a guardrail opportunity, implement it in the same session.

**Concrete actions for this project:**
- Add work-plan summary to `run_benchmark.py` startup: engines count,
  graphs count, seeds, total jobs, cached jobs, estimated time
- Warn when `--resume` scope differs from previous run
- Pipeline scripts should emit progress with entity names, not just counts
- Analysis scripts should print output summary stats before writing CSVs

---

## Concrete Rules (not principles, just gotchas)

- **Pipe safety:** Never pipe long-running commands through head/tail/wc.
  Write to file, read separately.
- **HDF5 purge:** Never delete H5 keys individually from large files.
  Either skip (let benchmark overwrite) or write new file with kept keys.
- **Cache invalidation:** Before purging, ask: "Will downstream overwrite
  it anyway?" If yes, skip the purge.
- **CSV float safety:** All `float()` on CSV values must handle empty
  strings. Use a safe conversion helper.
- **Error propagation:** Pipeline steps must fail loud. No bare
  try/except on data operations. Silent fallbacks on data-critical
  paths will eventually produce plausible-looking wrong output.

---

## Distribution Plan

These principles are USELESS in this file. They must be embedded in
always-loaded context at trigger points:

| Principle | Target Location | Section |
|-----------|----------------|---------|
| P1 (Work Plan) | CLAUDE.md | "Long Script Pre-Flight" |
| P2 (Monitoring) | CLAUDE.md | "Checking Task Status" |
| P3 (Output Validation) | CLAUDE.md | "Autonomous Iteration" |
| P3 (Output Validation) | AGENTS.md / spec template | Task specs for pipelines |
| P4 (Cost-Proportional) | CLAUDE.md | "Long Script Pre-Flight" |
| P5 (Tool Guardrails) | CLAUDE.md | "Encode Guardrails in Tools" (new section) |
| Concrete rules | .project-context/knowledge/gotchas.md | Append |
