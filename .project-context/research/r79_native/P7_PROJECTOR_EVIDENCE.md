# P7 Evidence: Convergent overlap projector + metric-gated acceptance (r80/projector)

Branch: `r80/projector` (based at r79/native head ef4eef5; trunk 897dbe3
merged for S2b).
Brief: `.project-context/research/r79_native/briefs/r80_s2_projector_fix.md`
(S2), superseded by the architect's S2b salvage directive.

## S2b sweep FAIL bisect (2026-07-09): the referee is honest -- S2b REPLACED the winning cleanup variant instead of contesting it

S2b gate sweep verdict (architect-evaluated): 61/13/19 + 9/2/4 = 85/108,
down from trunk's 87; net composite +9.90 over 15 movers. Large gains
(planar_60 +19.9, regular_4_40 +15.4, random_bipartite_60 +13.8,
r79_weighted_community +12.2, er_500 +4.9) but three S4 flagship wins
collapsed: petersen_10 79.0->58.3, weighted_karate_34 69.5->50.1,
weighted_clusters_3x10 68.1->53.5.

Instrumented end-to-end decomposition (`scripts/r80_probe_petersen.py`,
run on all three collapsed graphs) rules out every scoring-divergence
hypothesis:

| check | petersen_10 | weighted_karate_34 | weighted_clusters_3x10 |
|---|---:|---:|---:|
| frame gap (contest score vs benchmark score, SAME positions) | +0.000 | +0.000 | +0.000 |
| post-selection gap (benchmark: winner-as-selected vs final returned) | +0.000 | +0.000 | +0.000 |
| final positions identical to winner-as-selected | yes | yes | yes |

The contest scorer and the benchmark scorer agree term-for-term
(node_sizes identical, direction identical, undirected flavor both sides,
per-term tables all +0.0000), and NO post-selection stage (aspect_fit,
packing, anything) touches the positions after scoring. The referee
scores exactly what the user gets.

The actual cause -- cleanup-variant replacement (challenger cleanup
scores; contest frame == benchmark frame on every row):

| graph / challenger | raw | legacy-cleaned | convergent-cleaned |
|---|---:|---:|---:|
| petersen_10 / sfdp | 33.4 | 54.4 | 52.7 |
| petersen_10 / neato | 59.4 | **79.0** | 58.3 (selected) |
| weighted_karate_34 / sfdp | 28.2 | 28.4 | 26.1 |
| weighted_karate_34 / neato | 49.3 | **69.5** | 46.1 (lost to incumbent 50.1) |
| weighted_clusters_3x10 / sfdp | 23.2 | 28.5 | **49.8** (+21.4!) |
| weighted_clusters_3x10 / neato | 48.0 | **68.1** | 53.5 (selected) |

The trunk's three flagship scores (79.0 / 69.5 / 68.1) are EXACTLY the
legacy-cleaned neato candidates. S2b's `_project_candidate` change swapped
the cleanup from legacy to convergent, so those candidates ceased to
exist; the contest then honestly selected the best of what remained.
Meanwhile the wclusters sfdp row (+21.4 for convergent) shows the inverse
case -- the source of S2b's five big sweep gains. NEITHER cleanup variant
dominates.

Named cause: **candidate-pool regression, not referee dishonesty.** The
fix that preserves both win classes: have each challenger contribute BOTH
cleanup variants to the contest (equivalently: per challenger, clean both
ways, score both with the same honest composite, keep the better -- the
contest is argmax so both formulations select identically). Cost: one
extra projection + one extra `full()` scoring call per challenger, bounded
by the existing MAX_CONTEST_NODES cap. Expected effect: keeps the trunk's
87 (legacy-cleaned candidates return to the pool) plus the S2b gains
(convergent-cleaned candidates stay); the re-sweep verdict should be a
strict superset of both.

Status: reported to architect BEFORE any code change, per directive.

## S2b salvage (2026-07-09): convergent projector is now OPT-IN, wired into portfolio challengers only

Architect decision after the S2 sweep FAIL below: salvage, not rework.
Commits (each stage separate):

1. Merge trunk (897dbe3: S4 undirected portfolio + d665600 declared-
   undirectedness fix + drawing-metrics) into r80/projector -- `0a71fce`.
2. `191b209` -- default-path rewiring REVERTED. `_project_exact` is now a
   dispatcher: default `convergent=False` runs the restored pre-r80 legacy
   pass bit-for-bit (pinned by
   `test_default_path_preserves_legacy_trajectory`, which also documents
   the legacy dense-clique stall as intended default behavior);
   `convergent=True` selects the accumulate+damp+deadlock-re-lay projector.
   `native_stress.py` restored bit-identical to the r79/native base
   (verified `git diff ef4eef5 -- ...native_stress.py` empty before the
   trunk merge). `project_overlaps` gains a `convergent` passthrough.
   `OverlapProjectionGated` stays registered+tested but is wired nowhere.
3. `897c60f` -- portfolio challenger cleanup (`_project_candidate` in
   `native_undirected.py`) now uses `convergent=True` with a 200-pass
   early-exit ceiling. Rationale: challengers (sfdp/neato candidates)
   arrive with dense overlap fields the legacy projector stalls on; the
   honest-composite referee + degeneracy guard make a bad convergent
   trajectory unshippable on this path (it simply loses the contest),
   which is exactly the protection the default path lacks.

### r80-S2 gate blind spot: BISECTED (task 4)

Question: on rgg_500 the gated acceptance saw 0 rejections yet the final
composite dropped enough to flip WIN->LOSS -- where exactly did the gate
lose sight of the damage?

Answer (instrumented call-site attribution, `scripts/r80_probe_callsites.py`
on the pre-merge S2 code): **the gate was never in rgg_500's path at all.**
Wrapping `project_overlaps` with a stack-sampling counter during a full
dagua engine run shows:

| graph | periodic_overlap_projection (ungated) | plain overlap_projection (ungated) | gated calls |
|---|---:|---:|---:|
| rgg_500 | 40 | 1 | 0 |
| r79_weighted_hub_spoke_4x18 | 7 | 1 | 0 |

Neither regressing graph routes through `native_stress`'s final projection
(the only call site S2 gated). Their trajectories diverged across the MANY
ungated `PeriodicOverlapProjection` invocations during optimization (plus
one ungated final `OverlapProjection`); each pass reshapes the layout
slightly differently under the convergent update, and the differences
compound. So: NOT a proxy-vs-sweep terms/params/seed mismatch, NOT a
later-stage (aspect_fit) undo -- a pure gate-coverage gap. "0 rejections"
was vacuous.

Mechanism per graph (`scripts/r80_probe_regression.py`):
- r79_weighted_hub_spoke_4x18: the grid-spread deadlock valve fired once
  with a 66-of-72-node residual set -- re-laying nearly the whole graph
  onto a grid mid-optimization (depth rho -0.28, stress +0.088).
- rgg_500: grid-spread never fired; 41 damped-accumulation calls compound
  into CV +0.091 / crossings +0.009; runtime also doubled (294s -> 590s in
  the probe) because the convergent exact pass is costlier at N=500.

Implication for any future default-path fix: gating a single final
projection cannot protect the default path; either every periodic
projection call must be trajectory-safe by construction, or the gate must
wrap the whole optimization stage. Recorded on `OverlapProjectionGated`'s
docstring as well.

## Historical: S2 sweep verdict (superseded by S2b salvage above)

SWEEP GATE FAILED -- net -13.459, one W->L flip (rgg_500); stopped per
architect decision tree.

UPDATE (architect override of the earlier gate-1 stop): the corpus sweep
(gate 3) was run as the arbiter. It FAILED acceptance: net composite delta
-13.459 across 108 compared graphs and one WIN->LOSS flip (rgg_500). Per
the decision tree, the quality-knob smoke test was NOT touched and no
further fixes were attempted. Full per-graph table:
`P7_SWEEP_DELTA_TABLE.md` (same directory). Details in "Sweep gate
(gate 3)" below. The earlier gate-1 stop analysis is retained further down
for the record.

### Sweep gate (gate 3) results

- Pre-change baseline: dagua rows produced at 10f01af (= branch base
  ef4eef5 + one docs-only commit; zero `dagua/` changes -- verified with
  `git log ef4eef5..10f01af -- dagua/`), snapshotted from the dagua-native
  worktree store into
  `eval_output/r79_baseline/results.pre_r80_at_base.json`.
- Post-change: `scripts/r79_baseline.py --dagua-only` on r80/projector in
  this worktree (112-graph corpus; 108 in common with the pre snapshot;
  the 4 `tl_*` graphs are corpus drift, excluded from the math). Both
  sides scored against the same frozen external rows; sprint TIE_BAND=0.5.
- Comparison tool: `scripts/r80_projector_delta.py`.

| summary | value |
|---|---|
| graphs compared | 108 |
| net composite delta | **-13.459** |
| pre W/T/L | 64/10/34 |
| post W/T/L | 63/10/35 |
| WIN->LOSS flips | 1 (rgg_500) |
| graphs changed at all | 2 of 108 (all other dagua rows bit-identical) |

The entire net regression comes from two graphs; 106/108 composites are
identical to the pre-change baseline (their final projections either found
zero overlaps at entry and early-exited, or their routes do not reach the
exact projector), so the change's blast radius is narrow but negative.

Per-term deltas for the two regressing graphs (pre -> post):

rgg_500 (500 nodes, exact-path boundary; composite 53.486 -> 48.094,
delta -5.392; W -> L; runtime 294s -> 454s):

| term | pre | post | delta |
|---|---:|---:|---:|
| overlap_count | 0 | 0 | 0 |
| crossing_rate | 0.0306 | 0.0393 | +0.0087 |
| edge_length_cv | 0.6361 | 0.7274 | +0.0912 |
| sampled_stress | 0.8595 | 0.8332 | -0.0264 |
| angular_res_mean_deg | 0.2241 | 0.1697 | -0.0543 |

r79_weighted_hub_spoke_4x18 (composite 75.497 -> 67.486, delta -8.011;
stays W; runtime 21s -> 15s):

| term | pre | post | delta |
|---|---:|---:|---:|
| overlap_count | 0 | 0 | 0 |
| crossing_rate | 0.0222 | 0.0385 | +0.0162 |
| edge_length_cv | 0.3711 | 0.4456 | +0.0745 |
| sampled_stress | 0.6599 | 0.7475 | +0.0876 |
| depth_spearman_rho | 0.6516 | 0.3757 | -0.2760 |
| dag_consistency | 1.0000 | 0.9744 | -0.0256 |

Diagnosis: on both graphs the OLD projector already reached zero overlaps;
the new accumulate+damp projector reaches zero too but along a worse
geometric trajectory, inflating CV/crossings (rgg_500) and
stress/depth-correlation (hub_spoke). The overlap term itself is a wash --
the change traded geometry quality for nothing on these graphs. Why the
acceptance gate did not save us: the gate's proxy compares
before-projection vs after-projection (its job is to reject projections
that hurt); when the pre-projection state has overlaps, the +20 no-overlap
term dominates the proxy and the projection is (correctly, by its own
rule) accepted -- the gate cannot see that a DIFFERENT projector would
have cleared the same overlaps more cheaply. The regression is in the
projector trajectory, not the gate logic. (Mechanism probe below.)

Fix directions for the next round (NOT taken, per stop rule):
1. Restrict the grid-spread deadlock valve to genuinely dense small
   residual sets (e.g., residual pair density above a threshold, or
   residual set below ~50 nodes); on a 500-node near-converged layout a
   grid re-lay is far too blunt.
2. Consider damping only when a node participates in multiple pairs
   (counts-aware scaling), so sparse-overlap graphs keep the old
   trajectory exactly.
3. Cap the exact-path iteration ceiling by N (200 passes at N=500 is
   O(N^2)-per-pass expensive: rgg_500 runtime +160s).
4. Gate refinement: also require per-term non-regression caps (CV,
   crossings) when the overlap term is what flips the proxy.

## Original gate-1 stop record (superseded by the sweep run above)

Parts 1 and 2 are implemented, unit-proven, and lint-clean. One pre-existing
smoke test in the gate-1 scope (`tests/test_layout/test_quality_knob.py::
test_quality_high_smoke_spends_more_and_scores_near_draft`) fails with this
diff and passed on the base commit (verified via `git stash` A/B). One fix
attempt was made (directed-aware gate proxy, kept -- it closes a real blind
spot); the test still failed. The sweep verdict above supersedes the "smoke
test jitter is orthogonal" hypothesis: the corpus shows the same signature
(CV/crossings/stress inflation from the changed projector trajectory), so
the smoke failure was an early true positive, not noise. Diagnosis below.

## What changed

### Part 1: convergent exact projector (`dagua/layout/projection.py`)

- Root cause fixed: `_project_exact` used torch advanced-index in-place adds
  (`pos[x_r, 0] += ...`). With repeated node indices these are
  last-write-wins, NOT accumulating, so on dense overlap cliques (many pairs
  sharing a node) most of the intended displacement was silently discarded
  and the projector never converged (P3B2 forensics, ranked fix item 1).
- Pushes are now accumulated per node into a delta tensor with `index_add_`
  over ALL overlapping pairs, then applied once per pass scaled by a
  `damping` parameter (default 0.7, new `_project_exact` keyword; module
  constant `_EXACT_PROJECTION_DEFAULT_DAMPING`).
- Termination: zero overlaps OR no-progress OR max iterations. Progress is
  measured as total penetration depth (sum of each overlapping pair's
  overlap along its push axis), not the raw pair count: count misfires on
  healthy single-pair convergence (holds count=1 for many passes while depth
  decays ~35%/pass), while depth cleanly separates converging cases from
  deadlocks. A pass counts as stagnant when depth shrinks by less than 1%
  (`_EXACT_PROJECTION_STAGNATION_MIN_PROGRESS_RATIO`); 3 consecutive
  stagnant passes (`_EXACT_PROJECTION_STAGNATION_WINDOW`) trigger the
  fallback below, and a second stagnation after the fallback stops the loop.
- Deadlock escape valve: accumulated damped pushes are a Jacobi-style
  simultaneous update, and dense cliques can reach a fixed point where
  per-node corrections cancel while pairwise overlaps remain (observed:
  the 30-node clique plateaued at 29-55 residual overlaps indefinitely; 3000
  passes did not clear it). On first stagnation, the still-overlapping node
  subset is re-laid once onto a deterministic grid
  (`_grid_spread_residual_overlaps`, no RNG, cell = max box + padding with a
  1.01 float32 safety margin), then normal passes resume. Pure torch,
  deterministic, no RNG anywhere.
- Public signatures preserved (`project_overlaps` unchanged; `_project_exact`
  gained only a defaulted keyword).
- `native_stress` `overlap_iterations` default raised 10 -> 200 (the exact
  path now exits as soon as it converges, so the ceiling is rarely consumed).

### Part 2: metric-gated acceptance (`dagua/layout/ops/project.py`,
`dagua/layout/ops/pipelines/native_stress.py`)

- New registered op `overlap_projection_gated` (`OverlapProjectionGated` +
  frozen config), wired into the native_stress pipeline's final projection
  stage (the only projector call site in native pipelines; verified via grep
  -- native_planar/native_stress_ml/dagua_flat call sites were left alone per
  the touched-file scope).
- Proxy composite before/after projection using REAL `dagua.metrics`
  functions only (no reimplemented formulas): `count_overlaps_detailed`,
  `sampled_crossing_rate` (fixed project metric seed 0, matching
  `metrics.full()`'s pinned-seed convention; 200k samples),
  `edge_length_cv`, combined via `composite_auto`. On semantically-directed
  graphs the proxy also includes `dag_consistency` (O(|E|), cheap) because
  projection can push nodes against the layout direction, which the three
  undirected terms cannot see; direction hint lookup mirrors
  `sugiyama._directed_igraph_fidelity_gate` (problem hint -> structure hint
  -> conservative directed default).
- Keep projected result iff proxy(after) >= proxy(before); else return input
  positions unchanged and emit a `log.debug` line with both scores.

### Tests

- `tests/test_layout/test_projection.py`: new `TestExactProjectorConvergence`
  -- 30-node dense clique (near-coincident nodes, real 60x20 label boxes,
  435/435 pairs overlapping) must reach 0 overlaps; single-seed and
  10-seed variants. Plus an iteration bump (4 -> 8) in one existing test
  whose exact budget assumed the undamped convergence rate.
- `tests/test_ops_project.py`: gated-op accept test (overlapping pair gets
  resolved and kept) and reject test (monkeypatched destructive projection
  on an already-clean layout must be reverted bit-for-bit). Plus an
  iteration bump (20 -> 40) in one existing exact-tolerance test, same
  damping reason (commented inline).

## Convergence proof (gate 2)

30-node dense overlap clique, all 435 pairs overlapping at start
(60x20 label boxes, padding 2.0):

| seed | initial overlaps | final overlaps | passes to zero |
|-----:|-----------------:|---------------:|---------------:|
| 0..9 | 435 | 0 | 18 (all ten seeds) |

(Minimum iteration cap found by bisection; the trajectory is ~4 damped
passes -> stagnation detected -> one grid re-lay -> mop-up passes.)
Old code on the same seed-0 case: 55 overlaps remaining after 200 passes,
29 after 3000 -- structurally stuck, matching the sbm_4x30 forensics (37+
overlaps after 50 iterations).

Also verified at larger N (throwaway scale probe, pre-grid-spread bisection
budget): n=100/300/500 all reach 0 overlaps within the 200-pass ceiling.

## Gate results

1. Scoped tests -- PARTIAL FAIL (stopped here after two failures):
   - `pytest tests -k "projection or overlap"`: 58 passed, 7 skipped.
   - native_stress-touching files (`test_pipeline_native_stress_ml.py`,
     `test_layout/test_quality_knob.py`, `test_pipeline_registry.py`,
     `test_ops_distance.py`, `test_layout/test_cluster_driver.py`):
     77 passed, 1 FAILED:
     `test_quality_high_smoke_spends_more_and_scores_near_draft`.
2. Unit proof of convergence -- PASS (see table above; added as pytest).
3. Full dagua-only sweep -- NOT RUN (stop rule).
4. `ruff check` on the five touched files -- PASS ("All checks passed!").

## The failing gate-1 test: diagnosis

The test lays a fixed 6-node/9-edge DAG at quality 0.1 and 0.9 for three
seeds and asserts score(high) >= score(low) - 0.5 with the full directed
composite. With this diff, seed 11 gives low=51.32 / high=49.93
(diff -1.39; seeds 17/23 pass the tolerance). Per-term deltas
(high - low), seed 11:

| term | low | high | delta | composite impact |
|------|----:|-----:|------:|-----------------:|
| dag_consistency | 0.556 | 0.667 | +0.111 | +2.4 (better) |
| edge_length_cv | 0.448 | 0.423 | -0.025 | +0.4 (better) |
| depth_spearman_rho | 0.143 | -0.029 | -0.171 | -1.9 (worse) |
| overlap_count | 0 | 0 | 0 | 0 |
| edge_straightness_mean_deg | 18.86 | 16.66 | -2.20 | +0.4 (better) |
| crossing_rate | 0.375 | 0.313 | -0.063 | +0.6 (better) |
| sampled_stress | 0.296 | 0.311 | +0.015 | -0.2 |
| angular_res_mean_deg | 32.28 | 10.61 | -21.68 | -2.7 (worse) |

Both layouts are overlap-free; the high-quality layout is better on every
term the projector influences (overlaps, crossings, CV, straightness, even
dag_consistency). The loss is entirely angular resolution + depth
correlation -- position-jitter side effects on a 6-node graph, not a
projection-quality regression. The gate never rejected in any of these runs
(debug logging enabled; zero rejection lines), so gate asymmetry is ruled
out; the damped accumulate projector simply lands final positions a few
units away from where the old last-write-wins projector did, and this
particular smoke assertion is sensitive to that jitter at the 1-point scale.

Assessment: the assertion failure is real but marginal (0.89 over the 0.5
tolerance) and orthogonal to what this change fixes. Options for the next
round (NOT taken here, per stop rule): (a) accept as known-flaky-boundary
and loosen the tolerance for this 6-node smoke; (b) trace which stage's
jitter flips the angular term; (c) evaluate whether the corpus sweep (gate
3) shows the same pattern -- if the sweep is net-positive with no W->L
flips, this single 6-node smoke tolerance is likely the thing to adjust.

## Rejected-gate statistics

- The sweep run was not instrumented for gate decisions (`log.debug` is not
  captured by the baseline script), so no corpus-level rejection count.
  Observable proxy: 106/108 dagua rows are bit-identical to pre-change,
  which means their final projections were either no-ops (zero overlaps at
  entry) or their routes never reach the gated native_stress call site; the
  2 changed graphs were accepted projections (both post rows have
  overlap_count 0, which rejected/unprojected positions would not
  guarantee).
- In all instrumented runs (quality-knob debug: 7 native_stress layout
  calls with gate debug logging): 0 rejections.
- The reject path is exercised and proven by
  `test_overlap_projection_gated_reverts_when_proxy_regresses` (destructive
  fake projection reverted bit-for-bit).

## W/T/L

Measured by the gate-3 sweep (see "Sweep gate (gate 3) results" above):
pre 64W/10T/34L -> post 63W/10T/35L; one WIN->LOSS flip (rgg_500); net
composite delta -13.459 over 108 compared graphs. ACCEPTANCE: FAIL.
Recommendation: do NOT merge r80/projector as-is; rework the projector
trajectory per the fix-directions list, using the two named graphs plus
the quality-knob smoke as fast regression probes.
