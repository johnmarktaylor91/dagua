# P12: r80-S9 squeeze round -- clustered-undirected candidate + weighted semantics

Branch: r80/squeeze (worktree dagua-native-p1, forked from r79/native @ 6bccd08).
Commit: 5293938 "feat(r80-S9): clustered-undirected + weighted-similarity portfolio candidates".

## VERDICT: gate PASS

Gate sweep: W=74 T=16 L=18, best-or-tied=90/108 (`scripts/r79_baseline.py
--dagua-only --fresh`, wall time 1911.7s, log preserved at
`/tmp/r80_s9_sweep.log`).

- Legacy: W=64 T=14 L=15 (best-or-tied 78) -- **exactly** matches the cited
  89-state legacy split.
- Extended: W=10 T=2 L=3 (best-or-tied 12) -- +1 best-or-tied over the cited
  89-state extended split (9/2/4, best-or-tied 11).
- Zero WIN->LOSS or TIE->LOSS regressions across all 108 graphs (full
  per-graph verdict diff below).
- Net >= 0: satisfied amply (net score delta from S9's own candidates,
  isolated from unrelated already-merged work, is +37.19 summed across 4
  graphs, zero negatives anywhere).
- At least +1 best-or-tied: satisfied -- `r79_undirected_sbm_high_mix_3x30`
  flips LOSS->WIN, attributable to Deliverable 1 alone (see attribution
  below).

## Important correction to the brief's stated baseline

The committed trunk store (`eval_output/r79_baseline/results.json` at
r79/native head 6bccd08) is **stale relative to the current code** -- it
still reflects the pre-r80/projector 87-state (legacy 63/14/16, extended
8/2/5), not the cited 89-state (legacy 64/14/15, extended 9/2/4) the merged
head actually produces. This matches the brief's own caveat ("the committed
trunk store still shows the older 87-state -- your gate sweep will certify
the merged head").

Because of this, a naive before/after diff against the committed store
conflates **two different sources of improvement**: (1) already-merged
r80/projector code that was never re-swept into results.json, and (2) this
round's S9 candidates. Three graphs changed WIN/LOSS verdict in the full
sweep; only one is attributable to S9:

| graph | stale committed | true pre-S9 (fresh, unmodified r79/native) | after S9 | attributable to |
|---|---:|---:|---:|---|
| planar_60 | 57.86 (LOSS) | 77.72 (WIN) | 77.72 (WIN) | already-merged (r80/projector); **zero S9 involvement**, verified by running pre-S9 code directly (identical to after-S9 value) |
| r79_weighted_community_4x18 | 46.21 (LOSS) | 58.37 (WIN) | 58.37 (WIN) | already-merged; **zero S9 involvement** by the same direct check -- S9's cluster_sfdp/weighted_similarity candidates run but do not win this graph's contest |
| r79_undirected_sbm_high_mix_3x30 | 36.54 (LOSS) | 36.54 (LOSS) | 46.88 (WIN) | **S9 Deliverable 1** (cluster-aware sfdp driver candidate) |

"True pre-S9" values were produced by running `dagua.layout()` directly in
the untouched trunk worktree (`/home/jtaylor/.claude/worktrees/dagua-native`,
pinned at r79/native @ 6bccd08, r80/squeeze's exact parent commit) on the
same graph objects and scoring with the same honest composite -- not by
another full corpus sweep (time/disk budget). This isolates exactly what
this round's candidates changed.

## Deliverable 1: clustered-undirected portfolio access

**Status: implemented, gate expectations fully met.**

### Root-cause correction

The brief's cited diagnosis ("the engine's cluster driver preempts
routing") was verified **stale** for the current `dagua_native`/default
algorithm. Empirical trace (see Appendix A): `engine._layout_cluster_aware_
pipeline` DOES run first for clustered graphs, but `_build_cluster_inner_
pipeline("dagua_native", config)` returns `None` (that algorithm name isn't
in its supported set: fr/kk/fa2/sfdp/native_stress), so the cluster driver
returns `None` and the code falls through to the normal `dagua_native`
dispatch -- which DOES reach `undirected_portfolio` for declared-undirected
clustered graphs (`_choose_native_pipeline` never checks `problem.clusters`
at all). The real gap was narrower than the brief assumed: the contest DOES
run for these graphs today, but its incumbent and flat-sfdp/neato
challengers never structurally PLACE cluster hierarchy levels -- they rely
entirely on the composite's cluster-separation term to reward containment
after the fact.

### Fix

Added Candidate D: `_cluster_aware_sfdp_candidate` in
`dagua/layout/ops/pipelines/native_undirected.py`, fired only when
`problem.clusters` is truthy. Reuses the EXISTING recursive
`ClusterAwareDriver` (`dagua/layout/ops/cluster_driver.py`) with an sfdp
inner pipeline built via the existing `engine._build_cluster_inner_
pipeline("sfdp", config)` -- no new placement machinery, just a new
candidate wiring that lets the referee compare it. Wired into the contest
via the existing `_add_challenger` helper (both cleanup-projector variants,
degeneracy guard), so it can only ever help or match, never regress.

### Gate results

| graph | before (true) | after | delta | vs best external | verdict change |
|---|---:|---:|---:|---|---|
| r79_undirected_sbm_high_mix_3x30 | 36.54 | 46.88 | +10.34 | elk_layered 40.38 (+6.50 margin) | **LOSS -> WIN** |
| r79_undirected_sbm_low_mix_4x25 | 52.78 | 62.39 | +9.61 | elk_layered 50.12 (+12.27 margin) | WIN -> WIN (bigger margin) |
| r79_undirected_sbm_mid_mix_5x20 | 49.61 | 55.98 | +6.37 | elk_layered 37.83 (+18.15 margin) | WIN -> WIN (bigger margin) |

Matches the S4-era probe number exactly (46.88 for high_mix, cited in
P8_PORTFOLIO_EVIDENCE.md's "Caps and residuals" section as "flat sfdp+proj
would score 46.88") -- confirming the probe's original diagnosis was
correct about the SCORE achievable, even though the routing-preemption
mechanism it named turned out to be stale.

### Regression check (brief's explicit expectations)

- `r79_undirected_sbm_high_mix_3x30`: LOSS -> WIN. Met.
- low/mid: stay WINs (bigger margins). Met.
- 3 `r79_nested_clusters_*` graphs + `clustered_medium_5x20`: **verified
  unreachable by the new code**, not merely "unregressed." Direct
  `classify_graph` + `_choose_native_pipeline` trace (Appendix B) shows all
  4 route to `"layered_dag"`, never `"undirected_portfolio"` --
  `clustered_medium_5x20` is heuristically inferred undirected but fails
  the second routing gate (no explicit declaration, reciprocal-edge-ratio
  too low); the 3 `nested_clusters` graphs classify directed outright. The
  new candidate function is never called for any of the 4. Zero risk by
  construction, confirmed identical scores in the sweep (no row in the
  regression table above).

## Deliverable 2: weighted-similarity semantics

**Status: implemented, contest-protected, no flips on its own but a real
per-graph improvement (evidence a future round could build on).**

### Mini-probe (3 graphs, picking 1/w vs 1/sqrt(w))

Candidates: `r79_weighted_small_world_120` (brief's named biggest loss),
`r79_weighted_community_4x18` (P3B2_STRESS_FORENSICS.md Ranked Fix 4's
named graph), `real_lesmis_77` (real-world weighted-community LOSS). Ran
the native-stress core directly with three weight treatments (raw-as-
distance = today's default; `1/w`; `1/sqrt(w)`), scored raw, legacy-
projected, and convergent-projected positions with the honest composite:

| graph | raw dist | 1/w raw | 1/sqrt(w) raw | 1/w proj | 1/sqrt(w) proj | proj_conv (all transforms) |
|---|---:|---:|---:|---:|---:|---:|
| r79_weighted_small_world_120 | 19.84 | 22.61 | **23.88** | 25.14 | **26.31** | 45.28 (identical) |
| r79_weighted_community_4x18 | 19.20 | **27.64** | 21.85 | **23.96** | 21.29 | 48.98 (identical) |
| real_lesmis_77 | 21.58 | **21.66** | 17.95 | **27.49** | 24.69 | 43.68 (identical) |

`1/w` (`weight_transform="inverse"`) wins 2 of 3 graphs at both the raw and
legacy-projected tiers (community_4x18, real_lesmis_77); `1/sqrt(w)` wins
small_world_120 by 1.0-1.3 points. The convergent-projector tier is
IDENTICAL across all three weight treatments per graph -- 200 damped
overlap-resolution passes converge to the same arrangement regardless of
the small-scale stress differences between transforms, so it carries no
signal for this decision. Chose `1/w` (2-of-3 majority, and it is the
transform `preprocess.py`'s `BuildAdjacencyConfig.weight_transform` already
implements, so no new transform code was needed -- only threading a new
`NativeStressConfig.weight_transform` field through to it, default
`"none"`, preserving today's behavior everywhere else).

### Fix

Added Candidate E: `_weighted_similarity_candidate` in
`native_undirected.py`, fired only when `problem.edge_weights is not None`.
Reruns the native-stress core (`layout_native_stress_pipeline`) with
`NativeStressConfig(weight_transform="inverse")`. Threaded a new
`weight_transform` field (default `"none"`) through `NativeStressConfig`,
`_resolve_native_stress_config`, `_config_from_public`, and
`build_native_stress_pipeline`'s `BuildAdjacencyConfig` call in
`dagua/layout/ops/pipelines/native_stress.py`. Wired into the contest via
the same `_add_challenger` helper (both cleanup variants, degeneracy
guard).

### Gate results (all 10 weighted-undirected graphs in the corpus)

| graph | before (true pre-S9) | after | delta | verdict |
|---|---:|---:|---:|---|
| r79_weighted_small_world_120 | 34.41 | 45.28 | **+10.87** | LOSS -> LOSS (margin -17.28 -> -6.41) |
| r79_weighted_community_4x18 | 58.37 | 58.37 | 0.00 | WIN -> WIN (candidate ran, did not win the contest here -- already won by pre-existing code) |
| heavy_tail_weights_50 | 67.47 | 67.47 | 0.00 | WIN -> WIN |
| real_karate_34 | 68.79 | 68.79 | 0.00 | WIN -> WIN |
| real_lesmis_77 | 50.62 | 50.62 | 0.00 | LOSS -> LOSS (margin unchanged, -2.19) |
| r79_weighted_mesh_10x12 | 87.84 | 87.84 | 0.00 | LOSS -> LOSS (margin unchanged, -4.68) |
| r79_weighted_ladder_40 | 94.70 | 94.70 | 0.00 | WIN -> WIN |
| r79_weighted_bipartite_16x24 | 64.71 | 64.71 | 0.00 | WIN -> WIN |
| weighted_clusters_3x10 | 68.05 | 68.05 | 0.00 | WIN -> WIN |
| weighted_karate_34 | 69.55 | 69.55 | 0.00 | WIN -> WIN |

No flips are attributable to Deliverable 2 alone in this round -- the
`weighted_similarity` candidate materially improves
`r79_weighted_small_world_120` (+10.87, narrowing the gap to best-external
by 63%) but does not clear the remaining bar there (elk/kk-style community
detection still wins by -6.41); on the other 9 weighted graphs it competes
in the contest and loses to the incumbent or existing sfdp/neato
challengers, with zero regression. This is an honest, contest-protected
result consistent with the brief's framing ("the referee decides per
graph") -- not every predicate-gated candidate needs to win to be worth
adding.

## Gates

1. **Scoped tests**: 25 new/extended tests (`tests/test_native_stress_
   weight_transform.py`, 6 new; `tests/test_native_undirected_portfolio.py`,
   9 new tests appended) all pass. Broader scoped run: `pytest tests/ -q -k
   "cluster or undirected or native_stress or portfolio"` with the standard
   KNOWN_RED_TESTS.md deselects -- 249 passed, 2 skipped, 0 failed.
2. **Full gate sweep**: see VERDICT above. Log at `/tmp/r80_s9_sweep.log`
   (wall time 1911.7s / 31.9 min).
3. **ruff**: clean on all 4 touched files (`native_stress.py`,
   `native_undirected.py`, both test files) -- verified before commit and
   via the pre-commit hook at commit time.
4. Sweep launched via `nohup ... > /tmp/r80_s9_sweep.log 2>&1 &`, PID
   captured and polled to completion (no external supervisor).

## Candidate reachability (blast-radius audit)

Of the full 108-graph corpus, exactly 13 graphs route to
`undirected_portfolio` AND carry clusters or weights (the only graphs where
either new candidate can fire): the 3 `r79_undirected_sbm_*` (clusters),
and 10 weighted graphs (`heavy_tail_weights_50`, `real_karate_34`,
`real_lesmis_77`, `r79_weighted_community_4x18`, `r79_weighted_mesh_10x12`,
`r79_weighted_small_world_120`, `r79_weighted_ladder_40`,
`r79_weighted_bipartite_16x24`, `weighted_clusters_3x10`,
`weighted_karate_34`). All 13 were directly verified above (score
unchanged or improved, never regressed). The other 95 corpus graphs are
provably unaffected by this round's code -- `_choose_native_pipeline`
routing logic was not touched, so any graph that didn't reach
`undirected_portfolio` before still doesn't, and any graph that reaches it
without clusters or weights never calls either new candidate function.

## Appendix A: routing trace confirming Deliverable 1's corrected root cause

```
is_semantically_directed: False
clusters: True 3
num_nodes: 90
>>> _layout_cluster_aware_pipeline called, algorithm= dagua_native
>>> _layout_cluster_aware_pipeline result is None? True
>>> UNDIRECTED PORTFOLIO ENTERED
```
(`r79_undirected_sbm_high_mix_3x30` via `dagua.layout()`, r79/native head,
before any S9 code.)

## Appendix B: nested-cluster / clustered_medium routing (no-regression proof)

```
clustered_medium_5x20: is_sem_directed=False route=layered_dag
r79_nested_clusters_2x3x12: is_sem_directed=True route=layered_dag
r79_nested_clusters_3x2x10: is_sem_directed=True route=layered_dag
r79_nested_clusters_4x2x8: is_sem_directed=True route=layered_dag
```
None reach `"undirected_portfolio"`, so `_cluster_aware_sfdp_candidate` and
`_weighted_similarity_candidate` are never invoked for them.

## Commits

- 5293938 feat(r80-S9): clustered-undirected + weighted-similarity portfolio candidates

## Concerns / open questions

- The `_cluster_aware_sfdp_candidate` uses raw `config.cluster_side_
  padding_pt`/`cluster_label_band_pt`/`cluster_external_clearance_pt`/
  `w_cluster` values rather than the graph-style-merged values
  `engine._effective_cluster_side_padding` would produce for the top-level
  cluster driver (no `DaguaGraph` is available inside the headless
  contest). Documented as a known limitation in the candidate's docstring;
  did not affect any gate graph (none of the 3 SBM graphs set a custom
  `cluster_style.padding` override).
  `r79_weighted_community_4x18` is BOTH clustered and weighted -- both new
  candidates fire on it, but neither wins its contest (the pre-existing
  merged code already wins with 58.37). Not a bug, just worth flagging: the
  evidence doc's "gate results" tables for D1/D2 both list this graph with
  a 0.00 S9-attributable delta.
- Did not attempt to trace which SPECIFIC named candidate (`cluster_sfdp`
  vs `cluster_sfdp_convergent`, `weighted_similarity` vs
  `weighted_similarity_convergent`) won each flipped/improved graph in the
  full sweep -- `results.json` rows don't carry per-candidate provenance.
  The standalone probes in Deliverables 1/2 above establish which
  mechanism is responsible (cluster-aware sfdp driver for D1; native-stress
  core with `weight_transform="inverse"` for D2), which is sufficient to
  explain the score deltas, but a future round wanting exact win-rate
  attribution would need to add instrumentation (out of scope here per the
  "do not touch metrics composite functions / eval adapters" hard rule --
  results.json's row schema is produced by `scripts/r79_baseline.py`,
  which this round does not touch).
- P3B2_STRESS_FORENSICS.md and P8_PORTFOLIO_EVIDENCE.md were both located
  and read in full (the former only existed under
  `~/.claude/research/dagua/r79-native/`, not in either worktree's
  `.project-context/research/r79_native/` -- flagging this in case that
  durable-notes mirror should also exist in-repo for future sprints).
