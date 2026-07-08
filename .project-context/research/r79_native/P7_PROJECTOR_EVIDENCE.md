# P7 Evidence: Convergent overlap projector + metric-gated acceptance (r80/projector)

Branch: `r80/projector` (based at r79/native head ef4eef5).
Brief: `.project-context/research/r79_native/briefs/r80_s2_projector_fix.md`.

## Status: STOPPED AT GATE 1 PER BRIEF RULE (gate failed twice)

Parts 1 and 2 are implemented, unit-proven, and lint-clean. One pre-existing
smoke test in the gate-1 scope (`tests/test_layout/test_quality_knob.py::
test_quality_high_smoke_spends_more_and_scores_near_draft`) fails with this
diff and passed on the base commit (verified via `git stash` A/B). One fix
attempt was made (directed-aware gate proxy, kept -- it closes a real blind
spot); the test still fails, which is the second failure, so per the brief
("STOP and report if a gate fails twice") the corpus sweep (gate 3) was NOT
run and no further fixes were attempted. Diagnosis below.

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

- Corpus sweep not run (stop rule), so no corpus-level rejection rate.
- In all observed pipeline runs (quality-knob debug: 7 native_stress layout
  calls with gate debug logging): 0 rejections -- the convergent projector
  plus directed-aware proxy accepted every projection.
- The reject path is exercised and proven by
  `test_overlap_projection_gated_reverts_when_proxy_regresses` (destructive
  fake projection reverted bit-for-bit).

## W/T/L

Not measured -- the gate-3 sweep was not run because gate 1 failed twice
(see stop rule). Pre-change baseline W/T/L for r79/native remains
64W/10T/34L (74/108 best-or-tied) per r79 SUMMARY.
