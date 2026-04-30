# Cluster Sprint — Autonomous State

Started: 2026-04-29
Goal: implement Phases 1-4 of the cluster sprint (DESIGN.md). Phase 5 (Sugiyama+clusters) deferred to a separate sprint.

## Architectural decisions (resolved)

1. `cluster_aware=True` default-on after Phase 4 verifies
2. Cluster placeholder anchor: leaves-centroid
3. Cross-cluster edges in inner placement: ignored (first sprint)
4. Default `label_position`: theme-conditional (top-center for graphviz themes)
5. External-edge clearance: ~10pt (dot's range)
6. Phase 5 deferred to separate sprint
7. Keep legacy params with deprecation warnings
8. Boolean API surface (`cluster_aware=True`)

## Current state
- Round: **Phase 1 in flight**
- Most recent commit: b00f434 (graphviz parity B4)
- Watcher event expected: `CODEX_PHASE_1_DONE` or `CODEX_PHASE_1_FAILED`

## ✅ PAUSE CLEARED 2026-04-29 — user resumed; Phase 3 in flight

## Wake-up protocol (FOLLOW EXACTLY)

On any new turn, FIRST: check git log -3 + pgrep -x codex + this file's "Current state".

### Case A: codex committed AND not running (good path)
1. Read REPORT_phase_<N>.md and verify tests passed
2. **If pause directive is active (see above): document the result in this STATE.md, end turn, do NOT dispatch next phase.**
3. If next phase exists AND no pause → write PROMPT_phase_<N+1>.md, dispatch codex w/ PID watcher, update Current state
4. If at end of sprint (post-Phase 4) → run cosmetic audit subagent, then if all green run "Final shutdown"

### Case B: codex still running
Acknowledge briefly, end turn. Watcher will fire.

### Case C: codex quota hit (`CODEX_*_FAILED` with usage limit error)
Switch to Opus subagent on same prompt. If Opus also blocked, write blocked-status, send user a "blocked, resume manually" text, stop.

### Case D: tests fail post-codex
1. Read failure output
2. If fixable (<10 min): commit fix, continue
3. Else: revert codex's commit (`git revert HEAD`), document, skip phase
4. Continue to next phase if applicable

### Case E: anti-flail
3 consecutive phases with same un-closeable issue → mark deferred, declare done with current state.

### Final shutdown
After Phase 4 lands AND post-Phase-4 visual audit passes:
1. Write SUMMARY.md (per-phase outcomes, before/after metrics, accepted residuals)
2. Send iMessage via `~/.claude/scripts/send-to-jmt.sh` with summary
3. Render fresh comparison gallery + send 4-5 representative cluster panel side-by-sides
4. Update Current state → DONE

## Stop criteria

- All 4 phases committed + phase-4 visual audit passes (zero `real_cosmetic_gap + fixable` findings on cluster fixtures), OR
- Anti-flail (3 phases with same un-closeable issue), OR
- Codex AND Opus both quota-blocked, OR
- Catastrophic test failure that can't be recovered

## Iteration log

- DESIGN.md investigation: complete (Opus). 6 phases proposed; user-architect dispatched Phases 1-4 only. Phase 5 (Sugiyama+clusters) deferred. Phase 6 (cleanup/docs) folded into Phases 1-4 incrementally.
- Phase 1: COMMITTED `d46cdaf`. Pure refactor, pixel L1 = 0 on all 3 cluster panels.
- Phase 2: COMMITTED `aed468a`. ClusterAwareDriver landed for FR/KK/FA2/SFDP. dagua_native fell back (recursive subproblems hit layered ordering precondition — gated off with warning, deferred). 14 cluster tests pass. 233 layout tests pass. Parity metric stable at 99.27%. transformer_block visually clear; nested_clusters/cluster_showcase still show render-side overlap because graphviz_theme_comparison.py uses dot positions, bypassing Phase 2's placement (Phase 3 render parity work).
- Phase 3: COMMITTED `2d7cb4b`. graphviz themes now use top-center labels w/ opaque @background masks. Universal mask rendering (any theme can opt in). Z-order fix so masks actually cover borders. 158 render/style tests pass. Parity metric stable at 99.27%.
- Phase 4: COMMITTED `394c67d`. Edge clipping at cluster perimeter for cosmetic edges. 4 new tests + 395 existing pass.
- Post-Phase-4 audit: FAIL. 5 HIGH-severity defects (cluster rectangles MISSING on nested_clusters/cluster_showcase, label mask too narrow, edge clipping not visibly engaging, sibling overlap, harness uses dot positions hiding Phase 2). Plus instrument gap: declarative metric blind to rectangle presence. AUDIT_post_phase_4.md.
- Phase 5: COMMITTED `e5d5e26`. F1/F2/F3/F5/F6 PASS. F4 partial (bypass edges still need work). 395 tests pass.
- Post-Phase-5 audit: PARTIAL/CONTINUE. G1 (deep_nesting_4 concentric collapse regression) + G2 (edge bodies clipped to stubs) + G3 (label z-order vs nodes) + G4 (bypass edges) + G5 (re-run gallery with --use-dagua-placement). AUDIT_post_phase_5.md.
- Phase 6: COMMITTED `9e7a06e`. G1/G2/G3/G5 PASS. G4 partial (bypass edges improved, still on watch). 1h 24m.
- Post-Phase-6 audit: PARTIAL/CONTINUE. H1 (top edges missing on 5 panels — bbox cap collapsing top to zero), H2 (Outer Group label still fragmented despite Phase 6 z-order claim), H3 (cluster-aware placement collapses directed-flow graphs — DEFERRED to cluster-aware Sugiyama sprint per DEFERRED.md).
- Phase 7: COMMITTED `82eb897`. H1 (top edges) + H2 (label fragmentation) both fixed.
- Final audit: PASS / STOP. All H1/H2 closed, no regressions. cluster_rect_missing 41/41. AUDIT_DECLARED_FINAL.md.
- **Round: DONE 2026-04-29 evening.** SUMMARY.md + 5 cluster panel images sent to JMT. Cluster sprint complete; cluster-aware Sugiyama deferred to separate sprint per DEFERRED.md.
