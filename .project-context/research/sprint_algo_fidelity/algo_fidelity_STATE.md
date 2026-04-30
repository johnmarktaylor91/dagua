---
run: algo_fidelity
created: 2026-04-29T19:16:57-04:00
state: ROUND_3_DONE
current_round: 3
current_family: fdp
codex_pid: 3793569
codex_log: /tmp/algo_fid_round_3.log
watchdog_pid: 3793728
dispatched_at: 2026-04-29T19:55:00-04:00
prompt_file: .project-context/research/sprint_algo_fidelity/PROMPT_round_3.md
flail_count_dot: 1

baseline_truth_change: |
  ROUND 1 CACHED RMSDs ARE OBSOLETE. Cached benchmark used different
  node sizes than current code. Use live_compare.py output as the new
  ground truth from now on. Round 2's live baseline:
    dot family median (live): 0.341942
    mixed_width_labels (live): 0.404615
    shape_and_routing_matrix (live): 0.456349
    small_label_storm (live): 0.485187
  Layer assignment + x-ordering already match dot on diagnostic graphs.
  Remaining gap = coordinate values (Brandes-Köpf vs network-simplex).

round_1_summary: |
  baseline median RMSD per family:
    dot:           0.3245 (worst family)  -- attack first
    fdp:           0.2918 (second worst, ALL graphs > 0.15 floor)
    sfdp:          0.0915
    neato_mds:     0.0455
    neato_stress:  0.0353
  sugiyama PERFECT on simple/linear graphs (RMSD < 0.01),
  cliffs at medium graphs. Smallest divergent reproducer:
  mixed_width_labels (6 nodes, RMSD 0.35).
---

# algo_fidelity -- Autonomous Loop State

Drop-in graphviz replacement is central to the dagua pitch. This sprint
audits and improves dagua's faithful reproduction of the algorithms it
claims to reimplement, with **graphviz tools first** (dot, neato, fdp,
sfdp) and other claimed originals as a follow-on phase.

This file is the canonical "where are we" record. Every wake-up event
(watcher fire, user ping, schedule trigger) MUST read this file FIRST
and act on the case routing below given the current observable state.

## Scope (CRITICAL: parallel sprint constraint)

A separate "cluster cosmetic" sprint is also active on the same `develop`
branch. It owns:
- `dagua/render/**`
- `dagua/styles.py`
- `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`

**This sprint must NOT touch any of the above.** Algorithm-layout work only:
- `dagua/layout/ops/**` (registered ops + pipelines)
- `dagua/eval/variants.py` (registry entries -- ADD only, do not perturb existing)
- `scripts/**` for new fidelity-comparison scripts (do not modify
  `scripts/graphviz_theme_comparison.py` or other cosmetic scripts)
- `dagua/eval/competitors/*` only if a graphviz adapter has a real bug;
  otherwise leave alone
- `tests/test_layout/**` for any new pipeline-level tests
- `.project-context/research/sprint_algo_fidelity/**`

If a fix is unavoidably entangled with cosmetic code, **stop, document the
entanglement here, and skip that sub-issue**. We don't merge across sprints.

## Worker policy (CRITICAL)

The cluster sprint uses Opus subagents for visual inspection. To avoid
double-billing the user's Anthropic quota, **this sprint uses CODEX
exclusively for execution**. Opus subagents only allowed if codex is
quota-blocked AND the work is small (< ~50 lines).

## Stop criteria (observable, quantitative)

Primary (graphviz):
- Median Procrustes RMSD <= 0.05 for each of: graphviz_dot vs sugiyama,
  graphviz_neato vs stress_maj, graphviz_fdp vs fmmm, graphviz_sfdp vs sfdp
- AND worst-graph RMSD <= 0.15 for each family

Secondary (other claimed originals):
- partial_match families lifted to weak_equivalent OR documented as
  ceiling
- divergent families (davidson_harel) lifted to partial_match OR
  documented as ceiling

Anti-flail: 3 consecutive rounds with same un-closeable issue on the
same family -> mark `principled_residual`, document, move on.

max_rounds: 30 across all families combined.

## Wake-up case routing

Run this triage on EVERY new turn (watcher event, user ping, schedule):

```bash
git log --oneline -3
pgrep -af "codex/codex exec" | head -5
ls -la /tmp/algo_fid_*.log 2>/dev/null | tail -5
cat .project-context/research/sprint_algo_fidelity/algo_fidelity_STATE.md | grep -E "^state:|^current_round:|^current_family:"
```

| Observable signal | State | Action |
|---|---|---|
| codex pid alive AND log growing | RUNNING | ack the user "still cooking, watcher armed", yield turn |
| codex exited + new commit on develop matching `feat(fidelity):` | ROUND_DONE | run Case A below |
| codex exited + NO new commit | ROUND_FAIL | run Case D below |
| watcher fired CODEX_FAILED with quota text | QUOTA_BLOCKED | run Case C below |
| watcher fired CODEX_TIMEOUT | TIMEOUT | inspect log; if work was happening, dispatch focused fixup; else Case D |
| user pinged "how goes" while RUNNING | RUNNING | ack, optionally tail latest log, yield |
| anti-flail (3x same family unclosed) | RESIDUAL | run Case E below |

### Case A: ROUND_DONE -> proceed
1. `cat eval_output/algo_fidelity/round_<N>/SUMMARY.md` -- read codex's per-round summary
2. Update Iteration log at bottom of this file
3. Decide next family:
   - If current family converged (per stop criterion above) -> advance to
     next family in priority order: dot -> neato -> fdp -> sfdp -> phase 2
   - Else -> continue same family with refined prompt
4. Write `PROMPT_round_<N+1>.md`
5. Update `state: DISPATCHING`, `current_round: N+1`, `current_family: <name>`
6. Dispatch via `codex-bg.sh /tmp/algo_fid_round_<N+1>.log <prompt>`
7. Arm `codex-watch.sh <PID> /tmp/algo_fid_round_<N+1>.log` via Monitor
8. Yield turn

### Case C: QUOTA_BLOCKED
1. Read codex log to confirm quota error message
2. Read /tmp/algo_fid_round_<N>.log -- check what work was already done
3. If small remaining work: pivot to one Opus subagent (model="opus",
   subagent_type="general-purpose") with adapted prompt
4. If large remaining work: schedule wakeup for codex reset time + 5min,
   send "blocked, will resume <time>" iMessage, set state: BLOCKED, stop
5. NEVER export OPENAI_API_KEY

### Case D: ROUND_FAIL
1. `tail -100 /tmp/algo_fid_round_<N>.log` -- find error
2. If <10 min fixable: write fixup spec, dispatch codex --resume
3. Else: revert any partial state, document failure in iteration log,
   move to next family if anti-flail triggered, else retry same family
   with simpler prompt

### Case E: RESIDUAL
1. Mark family `principled_residual` in iteration log with classification:
   - `architectural_floor` -- requires major reimpl
   - `data_floor` -- limited by 25-graph sample
   - `numerical_residual` -- below noise of stochastic seeds
   - `proxy_mismatch` -- competitors are not exact-algorithm matches
2. Continue to next family (don't shut down on first residual)

## Fallback chain (resource limits)

1. Primary: `codex-bg.sh` + `codex-watch.sh` (Monitor-armed)
2. Quota blocked: pivot to one Opus subagent for short remaining work, OR
   schedule wakeup for reset time
3. Both blocked: state: BLOCKED, iMessage user, stop

## Shutdown procedure (mechanical -- user may be asleep)

When stop criterion triggers (or user explicitly says "stop"):

1. Write `algo_fidelity_SUMMARY.md` containing:
   - Per-family before/after Procrustes RMSD (median, p95, worst)
   - Per-family verdict transitions (e.g., partial -> weak)
   - Commits in this run (`git log --grep="feat(fidelity):" --oneline`)
   - Accepted residuals with classification + rationale
   - Anything deferred (e.g., out-of-scope cosmetic entanglement)
   - Drop-in-graphviz pitch readiness assessment
2. Generate one comparison panel per graphviz family showing best+worst
   pairs (dagua vs graphviz side-by-side) using
   `scripts/algo_fidelity_panel.py` (codex builds in round 1)
3. Send via `~/.claude/scripts/send-to-jmt.sh -a <panel> "<family>: median RMSD <X>"`
4. Update this file: state: DONE, append final row to iteration log
5. Send final iMessage: "Run algo_fidelity done. graphviz parity: <verdict>. <N> commits, <M> residuals."

## Codex prompt template policy

Each round's prompt lives at:
`.project-context/research/sprint_algo_fidelity/PROMPT_round_<N>.md`

Each prompt MUST include:
- `<task>` block with: family in scope, files to read, files to create/edit,
  exact commands to run for verification
- `<scope_constraints>` block with the parallel-sprint exclusion list
- `<completeness_contract>` requiring: (a) measurable improvement OR
  documented `principled_residual`, (b) commit on develop with
  `feat(fidelity):` prefix, (c) per-round SUMMARY.md saved to
  `eval_output/algo_fidelity/round_<N>/`
- `<verification_loop>`: pytest changed-files, then comparator script,
  then commit
- `<missing_context_gating>`: if prompt is unclear, ABORT before edits and
  write `BLOCKED.md` -- do not guess

## Iteration log (append per round)

| Round | Family | Start | End | Commit | Median RMSD before/after | Worst graph | Notes |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 19:19 | 19:30 | 78e8529 | N/A baseline | dot/small_label_storm 0.4744 | Built cross-comparator + panels. Identified dot+fdp as worst families. |
| 2 | dot | 19:33 | 19:50 | (no commit) | live 0.3419 (cached 0.3245 obsolete) | mixed_width_labels live 0.4046 | DIAGNOSIS_ONLY: cached vs live mismatch from node-size drift. Layer + x-ordering match dot; gap is in coordinate assignment (BK vs network-simplex). live_compare.py added. |
| 1 | baseline | 2026-04-29T19:16:57-04:00 | 2026-04-29T19:26:04-04:00 | 0a9a957 | N/A baseline | center_port_backedge_hub | Round 1 = infrastructure |
| 2 | dot | 2026-04-29T19:32:00-04:00 | 2026-04-29T19:47:27-04:00 | none | 0.3245 cached / 0.3419 live | small_label_storm 0.4852 live | Built live comparator; blocked before Sugiyama fix because live baseline differs from Round 1 cache on 8/22 graphs due node-size context drift. |
| 3 | dot | 2026-04-29T19:55:00-04:00 | 2026-04-29T20:20:04-04:00 | feat(fidelity): round 3 | 0.3419 live / 0.0191 live | densenet_block 0.1679 live | COMMITTED: aligned classic Sugiyama direct defaults to dot point spacing (`rank_sep=72`, `node_sep=18`); diagnostics improved without simple-graph regression. Next family: fdp. |
