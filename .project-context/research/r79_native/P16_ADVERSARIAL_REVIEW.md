# r80 Sprint -- Adversarial Merge-Gate Review

Scope: `git diff ef4eef5..HEAD` on worktree dagua-native (r79/native trunk after r80
sprint). Read-only. 8,820 insertions / 50 files. Target: shared `develop`.

## VERDICT: SAFE WITH FIXES

One must-fix (HIGH, test-only) + one robustness regression worth guarding (MEDIUM).
No CRITICAL default-path safety hole found: the r80 mis-routing failure class
(outerplanar_dag_20) is closed and does not recur elsewhere on the corpus.

---

## FINDINGS

### HIGH-1 -- Newly RED test: stale monkeypatch stub after `_run_projection_impl` signature change
`tests/test_layout/test_projection.py::TestProjectOverlaps::test_projection_logs_oom_cpu_fallback`

The S2 projection work added a `convergent: bool = False` parameter to
`dagua/layout/projection.py::_run_projection_impl`, and `project_overlaps` now calls it
`_run_projection_impl(..., convergent=convergent)` (projection.py ~lines 465/476/483).
The test's monkeypatch double `_oom_once_then_delegate` (test_projection.py:125) still
has the OLD signature `(positions, node_sizes, padding, iterations, layer_index)`, so the
production call raises:

```
TypeError: ..._oom_once_then_delegate() got an unexpected keyword argument 'convergent'
```

Confirmed: passes on base ef4eef5 (no `convergent` param there), fails on HEAD. NOT in
KNOWN_RED_TESTS.md. Production code is correct; only the test double is stale.

Repro: `pytest tests/test_layout/test_projection.py::TestProjectOverlaps::test_projection_logs_oom_cpu_fallback`
Fix: add `convergent: bool = False,` to the `_oom_once_then_delegate` stub signature (and
forward it to `original_run_projection_impl`). One-line, test-only. A merge gate should
not ship a newly-red test on a touched module.

### MEDIUM-1 -- New crash surface: non-finite positions crash the default edge router
`dagua/edges.py::route_edges` -> `_build_node_grid` (line ~1654) and
`_local_density_spread_scales` (line ~1732): `int(math.floor(x / cell_size))`.

If any node position is NaN or +/-inf, `int(math.floor(NaN))` raises
`ValueError: cannot convert float NaN to integer` (inf -> `OverflowError`). The S7
spatial grid is built unconditionally whenever `num_nodes > 0`, so this fires before any
per-edge logic.

Confirmed NEW: base ef4eef5 `route_edges` had zero `_build_node_grid`/`math.floor`
occurrences; pre-r80 non-finite positions flowed into float math and produced a garbage
curve, not a crash. Now the default render path (`dagua.draw`) raises, and so does the
benchmark drawing-metrics pass (`benchmark.py::_drawing_metrics` runs dagua's
`route_edges` over EVERY engine's positions -- an external adapter emitting a NaN would
error that competitor row).

Repro (verified):
```python
pos = torch.tensor([[0.,0.],[float('nan'),1.],[2.,2.],[3.,3.]])
route_edges(pos, torch.tensor([[0,1,2],[1,2,3]]), torch.tensor([[30.,20.]]*4))
# ValueError: cannot convert float NaN to integer
```
Only triggers on already-broken (divergent) layouts, so severity is MEDIUM not HIGH.
Fix: guard the grid build -- if `not torch.isfinite(pos).all()`, skip node-avoidance /
port-spread (return early to the pre-S7 routing), or clamp/sanitize coords before
`math.floor`. Cheap and localized.

Other edge crash surfaces tested and HELD: zero-length (coincident non-self) edges,
self-loops, single-node self-loop, two-node graphs (avoidance gated on `num_nodes>2`),
all-zero node sizes, fully-coincident 6-node clique -- all return curves without error.

### LOW-1 -- composite_large_undirected / composite_large_auto added but not wired
`dagua/metrics.py` adds `composite_large_undirected` (65/35) and `composite_large_auto`,
but the only active large-tier scoring call (`benchmark.py:1855`, `_run_salt_derived_suite`
holdout path) still calls `composite_large(m)` unconditionally. So undirected N>2000
graphs are still scored with the DIRECTED large composite (30 pts of dead
`dag_consistency`). This is the exact S1 MEDIUM-1 landmine the P6 batch claimed to fix; the
weights are sane but the fix is inert until wired. No bias introduced (it simply isn't
called), so not a merge blocker -- flag for the scale rounds.

### LOW-2 -- `_reciprocal_edge_ratio` on the default classify hot path
`graph_classify.py::_infer_semantically_directed` (line ~523) now calls
`_reciprocal_edge_ratio` (a Python `set` over all edge pairs, O(E) mem+time) with no size
guard, for sparse graphs up to 10M nodes. The structure-field copy at line 820 IS guarded
(`<=100_000`), but the inference call is not. Mostly avoided in practice (dense-large hits
the fast path -> `True`; trees/forests return early before the reciprocity check), so real
exposure is large sparse non-tree forests. Bounded and correctness-neutral; note for
million-node runs.

### LOW-3 -- `_grid_candidates` iteration order relies on set-of-ints ordering
`edges.py::_grid_candidates` returns `list(seen)` where `seen: Set[int]`. Node-deflection
results depend on candidate processing order (cp1/cp2 mutate as each blocker clears).
`hash(int)==int` makes this stable within an interpreter (so benchmark determinism holds),
but it is not sorted -> only environment-stable, not guaranteed cross-version. Sort for
belt-and-suspenders determinism.

---

## WHAT I CHECKED THAT HELD UP

**Surface 1 -- default-path safety (PRIMARY CONCERN): SAFE.**
Ran a full-corpus route probe (`/tmp/r80_probe_route.py`, 122 graphs) comparing each
graph's benchmark direction tag vs the actual portfolio-routing predicate:
`DANGER (tag-directed but routes undirected): []` -- ZERO mis-routes.
- Structural safety argument (verified): the `reciprocal_edge_ratio > 0.3` trigger cannot
  hit a hierarchical DAG. Reciprocal edges (u,v)+(v,u) are 2-cycles -> graph is not acyclic
  -> `is_directed_acyclic=False` -> family already becomes FORCE_DIRECTED (classify line
  ~780, `is_semantically_directed is False`) in the BASELINE. So the incumbent for any
  reciprocity-triggered graph is already the direction-agnostic force-directed layout; the
  portfolio only adds sfdp/neato challengers scored on the same undirected composite. No
  hierarchy is lost that wasn't already lost pre-r80.
- Highest directed-graph reciprocity in the corpus is 0.17 (recurrent_feedback_cell,
  disconnected_label_cycle_collage) -- comfortable margin below 0.3.
- The deep-layering-inference-only path (the outerplanar_dag_20 / recurrent_feedback_cell
  regression) is correctly excluded: `_choose_native_pipeline` requires
  `direction_is_declared OR reciprocal_edge_ratio>0.3` in addition to
  `is_semantically_directed is False` (dagua_native.py routing block). Verified in code and
  by the probe (both graphs show `is_semantically_directed=False` but do NOT route).
- Corpus declaration and benchmark scoring share ONE oracle
  (`graphs.py::is_semantically_directed`, tag-based); `_declare_semantic_direction` only
  sets `False` for undirected-tagged graphs, leaving directed as `None`. Declaration and
  scoring cannot disagree for declared graphs.
- Multigraph/citation edge cases considered: double-edges can inflate `_reciprocal_edge_ratio`,
  but inflation still requires genuine 2-cycles -> non-acyclic -> already force-directed
  (same argument). Residual (LOW): a truly-hierarchical DAG can reach the portfolio ONLY via
  an explicit `is_semantically_directed=False` user declaration, which is by-design.

**Surface 2 -- contest integrity: SAFE.** `native_undirected.py`: candidate positions are
fresh tensors (`_project_candidate` -> `detach().clone()`); the scored object IS the
returned object (`positions[best_name]`), no post-scoring mutation. Degeneracy guard applies
to challengers only (`_add_challenger`), incumbent always eligible; if all challengers are
degenerate the sane incumbent wins (test:
`test_collapsed_challenger_loses_to_sane_incumbent`). Selection is load-independent:
`time_budget_s is not None` -> return incumbent; `n > MAX_CONTEST_NODES` -> return
incumbent; seeds fixed; `metrics.full` self-deterministic. Both cleanup variants contest
(no replacement) -- test `test_contest_registers_both_cleanup_variants` passes.

**Surface 3 -- convergent projector opt-in: SAFE.** `_project_exact(convergent=False)` ->
`_project_exact_legacy` bit-for-bit; `project_overlaps` default `convergent=False`;
`native_stress` `overlap_iterations` reverted to 10 (diff shows only the inert
`weight_transform` addition). `grep -rn "convergent=True"` finds it ONLY at
native_undirected.py:314 (referee-protected challenger cleanup). `OverlapProjectionGated` is
registered but wired into NO pipeline (grep of pipelines/ empty). Default path untouched.

**Surface 4 -- routing determinism/bounds: HELD** (except MEDIUM-1). Deflection is bounded
(`max_attempts=4` + chord-cap saturation `break`); chord `< 1e-6` guard; self-loops and
`num_nodes<=2` handled; crossing-aware referee accumulates in edge-index order
(deterministic). 131 existing edge tests + 113 new edge/drawing/project tests pass.

**Surface 5 -- metrics: SAFE.** `_is_degenerate_scale` threshold 0.25 is conservative (a
non-degenerate compact layout has edges >~ node size); guard only removes vacuous CV +
crossing credit, and conservatively no-fires when `edge_length_mean`/`node_diag_mean` are
absent (both are populated by `edge_length_cv`/full-quick). Applied symmetrically to dagua
and externals via `composite_auto`. Tests `test_metrics_degeneracy_guard` +
`test_metrics_composite_large` pass; state doc reports blast radius 9/972, 0 verdict flips.

**Surface 6 -- eval fairness: SAFE.** `size_policy._SIZE_AWARE_EXTERNALS` is consulted only
by graphviz/elk/dagre adapters (grep), making externals size-AWARE = STRONGER competition
for dagua (correct direction; drove the honest 90->74 re-freeze). Never touches dagua rows;
confined to `dagua/eval/`. `overlap=prism` likewise external-only. No leak into layout code.

**Surface 7 -- config/API surface: SAFE.** No `dagua/config.py` changes. New public field
`EdgeStyle.avoid_nodes=True` has dedicated tests (`test_edge_routing_avoidance.py`).
`NativeStressConfig.weight_transform` is validated (raises on invalid) and tested. The
default `draw()` edge-refinement path (`maybe_refine_routes` at balanced quality 0.5) is
logic-equivalent to the prior Sprint-6 adaptive-skip; the forced fuller-weight pass only
fires at `quality>=0.75` (opt-in). `_ForcedQualityEdgeConfig.__getattr__` wrapper is correct
(no recursion; overrides only w_edge_* weights).

## Test runs
- New suites: `test_native_undirected_portfolio, test_metrics_degeneracy_guard,
  test_classify_undirected, test_native_stress_weight_transform, test_layout/test_projection`
  -> 53 passed, **1 failed** (HIGH-1).
- `test_edge_routing_avoidance, test_edge_route_quality_gate, test_drawing_metrics,
  test_drawing_capture, test_metrics_composite_large, test_eval/test_size_aware_externals,
  test_ops_project` -> 113 passed.
- Existing edge suites (`test_edge_optimization, test_edge_routing_config, test_custom_edges,
  test_edges_rectilinear_optimization, test_ops_edge_route, test_taxi_routing,
  test_cosmetic_edge_features`) -> 131 passed. No S7 breakage.

## Recommendation
Fix HIGH-1 (stale test stub, ~1 line) before merge. Strongly recommend the MEDIUM-1
non-finite guard in `route_edges` in the same pass (cheap, closes a real default-path crash).
LOW items are follow-ups, not blockers.
