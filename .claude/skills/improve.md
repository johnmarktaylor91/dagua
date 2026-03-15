---
name: improve
description: Multi-agent codebase improvement pipeline. Dispatches 4 review agents (2 Claude + 2 Codex), synthesizes findings, writes specs, dispatches execution.
user_invocable: true
---

# /improve — Multi-Agent Codebase Improvement Pipeline

Usage: `/improve <focus area>`
Example: `/improve performance in the layout engine at 1B scale`
Example: `/improve error handling in graph module`
Example: `/improve` (will ask for focus area)

## Pipeline

### Phase 1: Planning
If `$ARGUMENTS` is empty, ask the user: "What should I focus on?"
Otherwise, use `$ARGUMENTS` as the focus area.

Write a planning memo (internal, not a file) that specifies:
- Focus area and goals
- Which modules to examine
- What counts as a useful finding
- Any known issues to skip (read gotchas.md)

### Phase 2: Diverse Review (5 agents — 4 parallel + self)
Launch all 4 subagents simultaneously AND do your own exploration:

**Agent 0 — Prime Claude (you):**
While the 4 subagents run in background, read the relevant code yourself.
Form your own opinions BEFORE seeing their results. You have conversation
context they lack — use it. Focus on whatever angle seems most productive
given what you and the user have been discussing. Write your findings
mentally (don't need a file) so you can weigh them against the 4 reports.

**Subagents (launched in parallel):**

**Agent A — Claude Big-Picture:**
```
Agent(subagent_type="general-purpose", run_in_background=True, prompt="""
You are the BIG-PICTURE reviewer. Focus: architecture, abstractions, design smells, algorithmic alternatives.

PLANNING MEMO:
{planning_memo}

INSTRUCTIONS:
- Read .project-context/knowledge/gotchas.md and AGENTS.md first
- Focus on {modules}
- For each finding, report: SEVERITY (critical/high/medium/low), FILE, DESCRIPTION, SUGGESTED FIX
- Think about: Could CUDA help? Are there structural shortcuts? Is the overall approach right?
- DO NOT write code. Return findings sorted by severity.
""")
```

**Agent B — Claude Detail:**
```
Agent(subagent_type="general-purpose", run_in_background=True, prompt="""
You are the DETAIL reviewer. Focus: bugs, edge cases, missing error handling, type issues, memory leaks, cache-hostile patterns.

PLANNING MEMO:
{planning_memo}

INSTRUCTIONS:
- Read .project-context/knowledge/gotchas.md and AGENTS.md first
- Focus on {modules}
- For each finding, report: SEVERITY (critical/high/medium/low), FILE:LINE, DESCRIPTION, SUGGESTED FIX
- Check: redundant allocations, dtype waste, Python loops that could be vectorized, unnecessary copies
- DO NOT write code. Return findings sorted by severity.
""")
```

**Agent C — Codex Big-Picture:**
```
dispatch.sh review-improve-bp codex exec --full-auto --ephemeral "
You are the BIG-PICTURE reviewer. {planning_memo}
Read AGENTS.md and .project-context/knowledge/gotchas.md first.
Focus on {modules}. Report findings as: SEVERITY, FILE, DESCRIPTION, SUGGESTED FIX.
Think about architecture, dead code, API inconsistencies, structural issues.
Write report to .project-context/tasks/review-improve-bp.report.md
DO NOT edit any source files.
"
```

**Agent D — Codex Detail:**
```
dispatch.sh review-improve-detail codex exec --full-auto --ephemeral "
You are the DETAIL reviewer. {planning_memo}
Read AGENTS.md and .project-context/knowledge/gotchas.md first.
Focus on {modules}. Report findings as: SEVERITY, FILE:LINE, DESCRIPTION, SUGGESTED FIX.
Check: off-by-ones, race conditions, missing tests, perf bottlenecks, memory waste.
Write report to .project-context/tasks/review-improve-detail.report.md
DO NOT edit any source files.
"
```

### Phase 3: Synthesis
When all 4 subagents return, combine their findings with your own:
1. Read all findings
2. Deduplicate (same issue flagged by multiple agents = higher confidence)
3. Rank by severity and cross-agent agreement
4. Group by module (for non-overlapping parallel execution)
5. Discard noise (known issues, trivial style nits)
6. Write a DRAFT execution plan (not final yet — goes to adversary first)

### Phase 3.5: Adversarial Review
Dispatch a Codex agent in read-only mode to attack the draft plan:

```
dispatch.sh review-adversary codex exec --full-auto --ephemeral "
You are the ADVERSARIAL REVIEWER. Your job is to ATTACK this plan and
find problems. Be harsh, skeptical, and thorough. You are not here to
be agreeable.

DRAFT EXECUTION PLAN:
{draft_plan}

ORIGINAL REVIEW REPORTS (check the plan against these):
{all_four_reports}

Your tasks:
1. OVERLOOKED FINDINGS: Identify findings from the four reports that the
   plan overlooked or underweighted. Why were they dropped? Were they
   actually important?
2. HIDDEN DEPENDENCIES: Find dependencies between changes the plan treats
   as independent. Will change A break if change B isn't done first?
   Will parallel execution cause merge conflicts?
3. RISK ASSESSMENT: For each proposed change, what could go wrong? What
   hasn't been considered? What's the blast radius if it fails?
4. IS IT WORTH IT: Question whether each change justifies its risk and
   complexity. Some 'optimizations' make code harder to maintain for
   marginal gains.
5. 80/20 PLAN: Propose the simplest subset of the plan that captures
   80% of the total value. What can be cut without losing much?

Be specific. Cite findings by number. Name files and line numbers.
Don't just say 'this seems risky' — say WHY and WHAT could happen.

Write your critique to .project-context/tasks/review-adversary.report.md
DO NOT edit any source files.
"
```

After the adversary reports back:
1. Read the critique carefully
2. Accept valid critiques — update the plan
3. Reject invalid critiques — note WHY in the final plan
4. Document: "Adversary raised X. Accepted/rejected because Y."
5. The adversary's 80/20 plan is especially valuable — seriously consider it

### Phase 4: Final Plan + Execution
After incorporating adversary feedback:
1. If there are genuinely controversial tradeoffs (reasonable people would
   disagree), surface them to the user. Otherwise, proceed autonomously.
2. Decompose into independent task specs grouped by module (non-overlapping files)
3. Dispatch each to Codex in parallel
4. Each spec includes quality gates: `ruff check . --fix && pytest tests/ -x --tb=short -q`
5. Monitor completion, commit passing changes

### Phase 5: Post-Execution Quality Review
After all execution Codexes finish, before considering the work done:

1. **Prime Claude (you) reviews all diffs holistically:**
   - Do the changes work together coherently?
   - Did anything get missed?
   - Any regressions introduced?
   - Read the actual code changes, not just the diff stats

2. **Dispatch a Codex quality reviewer in parallel (read-only):**
   ```
   dispatch.sh review-quality codex exec --full-auto --ephemeral "
   Review the recent changes (git diff HEAD~N..HEAD) for bugs, edge cases,
   missed tests, style issues, anything the implementers got wrong.
   Read AGENTS.md for quality standards. Check type hints, docstrings,
   error handling. Write findings to .project-context/tasks/review-quality.report.md
   DO NOT edit any source files.
   "
   ```

3. **Run full quality gates:**
   ```
   ruff check . --fix
   mypy --follow-imports=silent dagua/cli.py
   pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
   ```

4. **Compare findings from both reviews.** If either catches real issues,
   dispatch fixes. If both are clean, the work is done.

5. **Notify user with summary:**
   - What was built (high-level)
   - What the reviewers caught
   - What was fixed
   - Final test results
   - Only surface controversial issues — everything else, just report

## Key Rules
- All review agents get the same planning memo + gotchas.md context
- Codex review agents are READ-ONLY (no edits)
- Execution Codex agents get only their spec (standard pattern)
- Use claude-opus-4-6 at max effort for Claude subagents
- Non-overlapping file groups for parallel execution
- Notify via Pushover at Phase 2 complete and Phase 4 complete

## Tier Selection
- Major focus area (whole module, optimization pass): full 4-agent sweep
- Narrow focus (single file, specific bug class): 2 agents (1 Claude + 1 Codex)
- User can override: "/improve --quick ..." for 2-agent, "/improve --full ..." for 4-agent
