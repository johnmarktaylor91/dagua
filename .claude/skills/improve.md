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
6. Present unified execution plan to user
7. Surface controversial or high-stakes changes for discussion
8. Send Pushover: "Phase 2 complete. {N} findings across {M} modules. Ready for review."

### Phase 4: Execution
After user approval:
1. Decompose into independent task specs grouped by module (non-overlapping files)
2. Dispatch each to Codex in parallel
3. Each spec includes quality gates: `ruff check . --fix && pytest tests/ -x --tb=short -q`
4. Monitor completion, commit passing changes
5. Send Pushover: "Phase 4 complete. {N} changes committed."

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
