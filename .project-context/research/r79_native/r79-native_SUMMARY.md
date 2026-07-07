# r79 Native Placement Algorithm Sprint -- SUMMARY

Status as of 2026-07-07. Branch: r79/native (worktree ~/.claude/worktrees/dagua-native).
Durable mirror: ~/.claude/research/dagua/r79-native/. This file is the cold-start entry
point for anyone (esp. a future Fable session) resuming the sprint.

## Mandate (JMT, 2026-07-05)
Make dagua's OWN native placement algorithm as good as possible across the widest range of
graphs: directed/undirected/weighted/nested-clusters/multi-component/massive-scale, plus
placement of edges/labels/cluster-boxes where they affect node placement. Cosmetic-only work
out of scope. Decomposable-ops philosophy mandatory. A tunable time-vs-quality knob with a
sensible default. Graph-type-aware routing encouraged. Learn from the 23 reimplemented
reference algorithms. Target aspiration: "works amazingly well for as broad a range as
possible", ~95% best-or-tied.

## Headline outcome (honest)
- Honest best-or-tied on the frozen 108-graph corpus: ~64/108 (legacy 56W/8T/29L +
  extended 8W/2T/5L). This is DOWN from a pre-sprint 87/12/9 ONLY because we fixed an oracle
  bug (see P3a) that had been handing ~44 free composite points to layered engines on
  undirected graphs. The old number was partly fiction; this one is real.
- The 95% aspiration is NOT met. The single reason: the bet that undirected graphs
  (social/community/small-world/mesh) would flip to a stress core FAILED after 3 honest
  attempts (only grids flip). That is the sprint's headline RESIDUAL and the main subject of
  the follow-up.
- Real wins banked: better layered-DAG quality (jitter-stable, visually verified), a ~100x
  scale-ceiling raise (opt-in multilevel path), a working quality knob, honest measurement
  infrastructure, cluster + SCC + semantic-direction infrastructure.

## What is MERGED into r79/native
| Round | Commit(s) | What | Default-path effect |
|---|---|---|---|
| P1 | 11beeec (via bb608b0) | 3 broken algo registry entries fixed, PARAM_REGISTRY drift synced, config-propagation tests | baseline-neutral (proven bit-identical) |
| P0/P0b | c80e970, 672a941 | frozen 8-engine baseline store + hardened harness (subprocess isolation, jsonl resume, atomic swap) | eval infra only |
| P2c | e1a8344, 4fbabbd | cluster-aware x-compaction, cluster-contiguous ordering, rank-row snapping, component tiling scorer, multiedge cap | dependency_graph_100 flipped to WIN; dependency_500 +3.9; nested_clusters -> tie; 0 regressions; jitter-stable; Opus-visually-verified |
| P2c-fix | e2a9f8b | compaction spacing guard (min intra-row spacing) after the Opus audit flagged node-bar collisions | sweep-neutral defensive guard |
| P3a | 2c19c45 | ORACLE FIX: is_semantically_directed() -- 39 graphs retagged undirected, frozen store rescored (no re-layout) | corrected scoring; this is why W/T/L "dropped" to honest numbers |
| P3c | 6f4b246 (via d68bf6c) | hybrid v2: SCC-condensation pipeline + ops | ROUTED OFF (forced route regressed SCC targets 58->49; needs quality iteration) |
| P4/P4b | c0a266d, 1cf774c (via f654d39) | native_stress_ml multilevel scale path (coarsen/BH/sampled-coarsest) | OPT-IN only; 20K in seconds at <1GB where default times out/eats 13.5GB |
| P3d | 208342a | quality knob (draft/balanced/high/max or 0-1 float) + time_budget_s, wired to all 3 native cores; balanced==today PROVEN | sweep-identical at default; also fixed ~6 pre-existing stale tests |

Consolidated sweep on merged head: legacy 56/8/29, extended 8/2/5 (unchanged -- everything
merged is inert-by-default or baseline-neutral).

## What is BUILT but NOT merged (held / residual)
- **native_stress core** (in r79/native since P2): PivotMDS init -> annealed stress-SGD ->
  SMACOF polish -> size-aware overlap projection, weighted-Dijkstra targets. Reachable via
  algorithm="native_stress" / force_pipeline="stress". ROUTED OFF as a default because the
  route-flip failed (P3b/P3b2/P3b3).
- **P5 clusters** (9eb3d2c on r79/p5-clusters, HELD FOR REVIEW): native_stress gets REAL
  recursive cluster placement + a ClusterTree.from_flat_membership containment bug fix
  (0 violations / 27 clustered graphs, sweep-neutral). Full recursive LAYERED clustering
  regressed the sweep to 49/9/35 so it was correctly left disabled (residual). OPEN DECISION
  BEFORE MERGE: P5 removed the "falling back to flat" warning for ALL native clustered graphs,
  but layered clusters are STILL flat underneath -> the warning was silenced while still true.
  Recommend: keep the warning for the layered path (honest), keep native_stress recursion.

## Key evidence artifacts (all in .project-context/research/r79_native/, mirrored durable)
- r79_DESIGN.md -- the architecture (sketch->route->core->refine->polish) and adopted SOTA ideas.
- P2b_ELK_GAP_DOSSIER.md -- per-metric + positional forensics on all 10 original loss graphs.
- P2C_VISUAL_AUDIT.md -- Opus adversarial before/after audit of the layered flips.
- P3B2_STRESS_FORENSICS.md -- THE key doc for the follow-up: stage-by-stage diagnosis of why
  the stress core loses on undirected graphs; the sfdp+our-overlap-cleanup headroom proof.
- P3B3_EVIDENCE.md -- the 3rd stress-fix attempt (gates failed 54/8/31); what was tried.
- P3D_EVIDENCE.md -- quality knob mapping table + wall-time samples.
- P4_EVIDENCE.md -- scale ladder (20K/100K/1M time/RSS/quality) + contraction profiles.
- P5_EVIDENCE.md -- cluster containment + the layered-recursion regression numbers.
- KNOWN_RED_TESTS.md -- pre-existing stale test failures (avoid the whack-a-mole trap).
- gallery_p2c/, gallery_p2cfix/, gallery_p5/, stress_forensics/ -- rendered PNGs.

## Oracle / correctness bugs found this sprint (all real, worth remembering)
1. is_semantically_directed defaulted everything to directed -> undirected graphs scored with
   ~44 directed points (P3a fix).
2. Overlap projector projection.py:222-246 -- torch advanced-index += does not accumulate
   repeated indices, so dense overlap cliques never converge (found P3b2; fix attempted P3b3,
   did not convert to wins but the bug is real).
3. 3 native pipeline registry entries raised AttributeError on dispatch (P1 fix).
4. PARAM_REGISTRY defaults drifted from LayoutConfig dataclass (P1 fix).
5. TorchLens 2.28 API drift broke tl_* corpus graph construction (P3d fix).

## Operational incidents (and the guardrails added)
- Codex exec-mode sessions die if they emit a no-tool-call message while idling -> all briefs
  now carry <operational_rules>.
- Disk-full (/tmp wave scratch) silently killed rounds with rc=0 -> ledger issue + df-guard
  suggestion.
- Cross-session watcher pkill hygiene kills other sessions' monitors -> ledger issue.
- Full-suite `pytest tests/ -x` gates whack-a-mole for hours on pre-existing stale failures
  (P5 burned 11h) -> KNOWN_RED_TESTS.md + ledger issue; future gates scope to touched files.
</content>
