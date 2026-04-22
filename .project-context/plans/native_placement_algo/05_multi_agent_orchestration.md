# Multi-Agent Orchestration

When to delegate. When to dispatch. When to parallelize. When to stop.

## Agent roster

| Agent | Role | Cost profile |
|-------|------|--------------|
| Claude (architect) | Plan, review, integrate | Moderate |
| Codex (implementation) | Write code, one spec at a time | High if overused |
| Codex (adversarial review) | Attack the plan or code | Moderate |
| Claude subagent (general-purpose) | Research, literature, code-reading | Low |
| Claude subagent (Explore) | Codebase searches | Very low |
| Codex spark | Light-weight lint/rename/docstring fixups | Low |
| User (HJ via iMessage) | Final aesthetic judgment | Rare and rate-limited |

## Dispatch rules

### Codex implementation (`/codex:rescue --background --write`)

- 1-3 files per spec, as per CLAUDE.md.
- Spec must include the XML blocks from `gpt-5-4-prompting`:
  task, completeness_contract, verification_loop, missing_context_gating,
  action_safety, default_follow_through_policy.
- Every spec names:
  * Exact file paths
  * Function signatures
  * Expected behavior
  * Edge cases
  * Test commands (pytest invocations)
  * Success criteria (exit criteria of the sprint task)
  * Non-regression tests to run post-change
  * Link to the sprint file
- Max 2 parallel Codex writes, strictly. Greater parallelism risks merge
  conflicts. Greater-than-one requires the pre-dispatch checklist
  (shared files, behavioral coupling, known-red tests, model mix).
- The dispatch invocation must follow CLAUDE.md exactly:
  `codex exec --skip-git-repo-check --sandbox danger-full-access
   --cd /home/jtaylor/projects/dagua "<prompt>"` via `run_in_background=true`.
- Post-dispatch FIRST action: verify with `pgrep -fl "codex exec"` within 8s.

### Codex adversarial review (`/codex:adversarial-review --background`)

- Every sprint exit. Two review focuses, minimum.
- Long sprints (>1 clock day): additional mid-sprint check.
- Prompt templates in 06_adversarial_review_protocol.md.
- Review output is parsed for CRITICAL / HIGH / MEDIUM / LOW / NOTE findings.
- STOP after presenting findings. DO NOT auto-apply fixes. Ask the user.

### Claude subagents (general-purpose)

For each of these, dispatch one or more subagents in parallel (single response,
multiple tool calls):

- **Literature scan** (Sprint 1, 3, 6, 8). Dispatch 2 parallel subagents with
  different reading lists. See 07.
- **Competitor code reading** (Sprint 2, 3). Dispatch per-library subagents
  (OGDF, Gephi, Graphviz sfdp, NetworkX, igraph). 5 parallel max.
- **Failure analysis** (any sprint with unexpected metric regression). One
  subagent with explicit hypothesis, one without, compare conclusions.

### Claude subagent (Explore)

Use this for ALL 3+-query codebase exploration. Never Bash-grep for exploration.
Thoroughness level:
- "quick" for one-shot lookups
- "medium" for sprint-level audits
- "very thorough" for Sprint 0 or cross-cutting deep reads

## Parallelism patterns

### 2-agent parallel Codex implementation

Example: Sprint 1 -- one Codex writes `initializers/spectral.py`, another
writes `initializers/warm_sgd2.py`. Pre-dispatch checklist must confirm:
- Distinct file paths. No shared module imports on the write path.
- Distinct test paths.
- Distinct dispatch IDs.
- Known-red tests captured and forwarded to both.
- Model mix: at least one Claude-authored spec using different phrasing than
  the other (anti-groupthink).

### 2-parallel adversarial reviews at plan-level

Both run on the meta-plan itself at the end of its drafting phase. One focus
"scaling and memory"; the other "overfit and evaluation." Both complete before
we declare the plan ready. See 06 and 08.

### N-parallel subagent research

Literature + competitor reading dispatched as 2-5 subagents in a single
response. Each returns <=400 words. Claude synthesizes.

## Escalation rules

If Codex implementation fails QA:
- One retry with a tightened prompt.
- If a second failure, STOP. Retrospective. Do not try a third Codex run
  on the same spec.

If Codex adversarial review and implementation Codex disagree:
- Spawn a fresh Codex thread with both briefs pasted. This is the arbitration
  thread.
- If arbitration cannot reconcile, escalate to user with both positions.

If Claude and Codex disagree on a design decision:
- Claude MUST NOT unilaterally override Codex on a point Codex raised in
  adversarial review. The disagreement surfaces to user.

## Handoffs between sprints

Every sprint exit produces:
- A `sprint_<N>_exit_note.md` in `.project-context/plans/native_placement_algo/`
  with: what was done, open items, tolerated regressions, pointer to artifacts.
- Baseline metrics at `eval_output/native_algo/sprint_<N>_exit/metrics.json`.
- Adversarial findings archived.

The next sprint reads ONLY the exit note and its own sprint file. It does NOT
need to re-read prior sprint files. If it does, that sprint file is underspecified.

## Tool hygiene

- Never Bash-grep. Use Grep tool.
- Never read full files when a slice is enough. Use offset+limit.
- Never read all memory files at once. Check MEMORY.md index, read on demand.
- Background Codex means you get notified on completion. Do NOT poll or sleep.
- Use `dispatch.sh` for non-Codex background tasks (benchmark runs, pipelines).

## Budget discipline

- Per sprint: max 6 Codex runs (implementation) + 2-4 adversarial reviews.
  Exceeding this means scope creep; replan.
- Subagent research: per sprint max 8 subagents total.
- HJ pings: max 1 per sprint unless emergency.

## Communication

- Claude writes user-facing updates in the conversation. Brief.
- Codex writes code + review JSON. Claude synthesizes, never pastes raw.
- Sprint exit note is the one place Claude writes a multi-paragraph summary.
  Everywhere else: terse.
