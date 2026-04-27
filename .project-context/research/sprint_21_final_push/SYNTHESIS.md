# Sprint 21 Synthesis — All 12 Reports

12 reports across 6 areas (A/B/C/D/E/F), each with one Claude + one Codex.

## What everyone agrees on

### 1. F: One-line component-fix bug — UNANIMOUS, IMPLEMENT FIRST

Both F Claude and F Codex independently identified `dagua/layout/ops/pipelines/dagua_native.py:807-808`:

```python
if _selected_force_pipeline(child_config) is None:
    child_config.force_pipeline = "legacy_monolith"
```

This forces every per-component child solve to use `legacy_monolith` regardless of the child's topology. Tree-shaped children can never re-classify into `native_tree`. **Costs `multi_component_80` -13.37.**

Fix: allow re-classification when the child component is structurally a tree/chain. Keep the `legacy_monolith` fallback for cyclic / multi-component children to preserve sprint-19d packing wins.

**Effort: ~5 LOC. Predicted impact: +13 on multi_component_80, possibly more on other multi-component graphs.**

### 2. E: New polish primitives — measured +48.97 net

E Claude empirically tested 13 candidate primitives on 81 graphs. Two new candidates work:

- `ee + y_layer_snap(0.5)`: snap each y-band to its median y. **6 wins, ZERO regressions** across 81 graphs. Biggest: wide_single_layer_1_50_1 +9.77, wide_3_50_3 +6.86.
- `ee + ortho_align(10, 0.1)`: per-edge nudge toward dominant cardinal axis. **14 wins**. Biggest: multiscale_skip_cascade +7.73, weighted_clusters_3x10 +4.18.

E Codex adds `aspect_preserving_equalize` (locks bounding box during projection): expected +0.5..+1.5 on `dependency_500` which currently rejects polish.

**Key insight (E Claude):** composite weights are y-dominated (dag_consistency 25 + depth_spearman 15 = 40/100), so primitives that affect y discontinuously are much more leveraged. Existing `_equalize_edges` is x/y symmetric and misses this asymmetry.

**Effort: ~50 LOC. Measured impact: +48.97 net composite, 0 regressions (picker margin gate).**

### 3. A: Lattice synthesis — convergent, both confident

Both A Claude and A Codex propose: synthesize the canonical hex/tri embedding directly from graph topology (BFS-layer index + within-layer x-rank), feed as a picker candidate. The 0.5-margin filters bad detections.

Magnitudes differ but direction agrees:
- hex_lattice_42: A Claude +5..+8, A Codex +3.5..+5.5
- triangular_lattice_36: A Claude +3..+6, A Codex +0.8..+1.8

Codex applies stricter gate (max_degree ≤ 3, 1.15 ≤ E/N ≤ 1.45, lattice_like tag) which is the right engineering call.

**Effort: ~100 LOC. Predicted impact: +3.5..+8 hex, +1..+6 tri.**

### 4. C+B: Scored crossing/order polish — converges across 3 reports

B Codex, C Claude, and C Codex all independently propose adding a discrete crossing-reduction polish candidate.

C Codex notes there's an EXISTING `dagua/layout/ops/crossing_swap.py` Sugiyama adjacent-swap op that may not be wired into the post-polish path — reuse before reinventing.

Targets: weighted_clusters_3x10, triangular_lattice_36, multi_component_80, densenet_block, parallel_cycles_4x5.

**Effort: ~50-100 LOC if reusing existing op. Predicted impact: +6..+12 across 4-5 graphs.**

### 5. C: Component depth-rank restacking — clean one-graph flip

C Claude and C Codex agree: `disconnected_label_cycle_collage` loses -2.89 weighted points on `depth_spearman_rho` because component tiling is row-major and scrambles global y ordering. Fix: sort components by max_depth (or mean depth) before tiling.

**Effort: ~20 LOC change in `_tile_component_positions`. Predicted impact: +2..+3 on disconnected_label, possibly +0.5..+1 on multi_component_80.**

### 6. D: Deterministic picker scoring — meta-improvement (D Codex unique)

D Codex points out the picker uses `composite(full(...))` with stochastic sampling for crossings/angular. Near the 0.5 margin, RNG noise can reject a good candidate. Replace with exact pair-wise scoring at small N (e.g. E ≤ 1000), deterministic stratified sampling above.

**Effort: ~80-180 LOC. Predicted impact: +0.5..+2 suite-wide via fewer false-rejects.**

## What everyone agrees NOT to do

- All gradient weight tuning is saturated (already verified in CONTEXT.md)
- Bundling-aware metrics (off-axis vs composite)
- GNN/learned layouts (high effort, uncertain payoff, risk to runtime determinism)
- Pipeline-level all-six tournament (F Claude probe: 93% of graphs already optimal under current dispatcher)
- Force_directed / planar in any auto-routing shortlist (lose almost everywhere)
- petersen-specific brute-force / SAT (B Claude says it's already a win at HEAD)

## Recommended implementation order

| # | Change | Effort | Expected Δ | Risk |
|---|---|---|---|---|
| 1 | **F: Component-local tree re-classification** (1 line) | 5 LOC | +13 on multi_component_80 | low |
| 2 | **C: Component depth-rank restacking** (sort tiles by depth) | 20 LOC | +2..+3 on disconnected_label_cycle_collage | low |
| 3 | **E: y_layer_snap + ortho_align polish primitives** | 50 LOC | **+48.97 measured net** | zero (picker margin) |
| 4 | **E: aspect_preserving_equalize for large DAGs** | 30 LOC | +0.5..+1.5 on dependency_500 | low |
| 5 | **A: Lattice synthesis (hex first, then tri)** | 100 LOC | +3.5..+8 hex, +1..+6 tri | low (gated) |
| 6 | **C/B: Scored crossing-reduction polish** | 50-100 LOC | +6..+12 across 4-5 graphs | low |
| 7 | **D: Deterministic picker scoring** | 80-180 LOC | +0.5..+2 suite-wide | low |
| 8 | **F: Two-candidate metric-aware guard** (LAST, smallest marginal) | 50 LOC | small | medium runtime |

Estimated cumulative composite gain if all land: **+70 to +110 net** across the suite, with 8-12 graphs flipping from non-dominate to dominate.

## Risk-management

The picker's existing 0.5-margin gate (sprint-20k) is the key safety net. Every polish primitive added is **strict-upside** because the picker reverts to baseline when no candidate beats it by ≥0.5. Items 1, 2, 5 don't go through the picker so they need their own validation.

Sprint-19d component packing is a protected behavior. F's tree re-classification must be gated to "child component IS a tree (acyclic, connected, |E|=N-1)" so cyclic/general components keep `legacy_monolith`.

Lattice synthesis (item 5) needs strict structural gates per A Codex (max_degree, E/N, lattice_like tag, planar) plus metric guards (no overlap creation, no DAG drop, depth drop ≤ 0.02).

## Petersen note

B Claude flagged that petersen_10 may already be a win at HEAD (+3.42 fresh measure vs the -2.72 in CONTEXT.md). Verify before deciding whether B's brute-force-N≤12 proposal is needed.

## Open questions

1. Re-verify the 5 close-loss + 3 moderate-loss bucket members at HEAD with deterministic scoring before sprint-21 starts — CONTEXT.md numbers may be stale (especially petersen).
2. Does C's crossing-reduction op overlap with item 6 from C Codex (existing `crossing_swap.py`)? Read that file first.
3. Should A's lattice synthesis be opt-in via config flag for the first sprint, then made default in sprint-22 once measured?
