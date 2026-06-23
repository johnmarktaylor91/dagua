# r73 Thread 4 Findings: fmmm Divergence Bucket

**Bucket:** classic_fmmm_steps10, classic_fmmm_steps100, classic_fmmm_steps200, classic_fmmm_graphviz_fdp_fidelity
**Divergent combos:** 88 (Mode-A)
**Insufficient_data:** 12 (all classic_fmmm_graphviz_fdp_fidelity, large graphs)
**Source data:** eval_output/fidelity_definitive_r72/per_combo.json (final_rung field)

---

## SUB-BUCKET BREAKDOWN

| Mechanism | Combos | Status |
|-----------|--------|--------|
| M1: OGDF component packing algorithm mismatch | 15 | **FIXABLE** |
| M2: FP-chaos at steps10 iteration floor | 21 | FLOOR |
| M3: FP-chaos converging but not yet converged | 24 | FLOOR |
| M4: Compound graph connected FP-floor | 14 | FLOOR |
| M5a: FDP emulator FP divergence (disconnected + small) | 10 | FLOOR |
| M5b: Multi-edge aggregation in FDP emulator | 1 | **FIXABLE** |
| M6: Isolated stochastic border cases | 3 | FLOOR |
| **TOTAL** | **88** | 2 FIXABLE mechanisms (~16 combos addressable) |

---

## M1: OGDF COMPONENT PACKING ALGORITHM MISMATCH

**Status: FIXABLE**
**Affected combos: 15 (steps10/100/200 variants only; excludes fdp_fidelity)**

### Affected graphs and observed disp

| graph | steps10 | steps100 | steps200 |
|-------|---------|---------|---------|
| random_dag_50 | 1.238 | 1.196 | 1.193 |
| multi_component_80 | 0.851 | 0.818 | 0.815 |
| disconnected_encoder_residual | 0.734 | 0.729 | 0.730 |
| disconnected_label_cycle_collage | - | 0.699 | 0.691 |
| parallel_cycles_4x5 | - | 0.671 | 0.667 |
| kitchen_sink_platform_graph | 1.119 | - | - |
| random_dag_200 | 2.149 | - | - |

### Root cause

`fmmm.py:1484` in `_layout_ogdf_fmmm_component_fidelity` calls
`_graphviz_tile_pack_offsets(component_boxes)`, which is Graphviz's polyomino tile
packer (sorts by **perimeter** descending). OGDF's actual component packer is
`TileToRowsCCPacker` (places components in rows, sorts by **square-aspect-ratio-adjusted
area** descending). The two algorithms produce fundamentally different final positions.

### Evidence: constant disp across all step counts

The signature of a packing mismatch (vs FP-chaos) is that disp is CONSTANT regardless of
iteration count. FP-chaos converges to 0 as iters increase; packing mismatch does not:

- random_dag_50 (52 components: 1 large + 51 singletons):
  - steps10 disp=1.238, steps100 disp=1.196, steps200 disp=1.193 -- essentially constant
  - fdp_fidelity disp=0.971 (also constant; but fdp uses correct packer for Graphviz FDP reference)

- multi_component_80 (7 unequal components: 40, 20, 10, 5, 3, 1, 1 nodes):
  - steps10 disp=0.851, steps100 disp=0.818, steps200 disp=0.815 -- constant

- parallel_cycles_4x5 (4 equal components, 5 nodes each):
  - steps100 disp=0.671, steps200 disp=0.667 -- constant

Contrast: connected graphs with FP-chaos show DECREASING disp (e.g., deep_chain_20:
steps10=2.140, steps100=0.255, steps200=0.078).

### Why fdp_fidelity for disconnected graphs is NOT M1

`_layout_fmmm_fidelity_components` (the fdp path for disconnected graphs) also calls
`_graphviz_tile_pack_offsets` -- but Graphviz FDP itself uses this same packer. So the
fdp_fidelity path correctly uses Graphviz's packer to match the Graphviz FDP reference.
The fdp_fidelity divergences for disconnected graphs are FP-chaos in the spring force
iterations (classified as M5a below).

### Fix spec

**File:** `dagua/layout/ops/pipelines/fmmm.py`
**Function:** `_layout_ogdf_fmmm_component_fidelity` (line 1398)
**Line to change:** 1484

**Change:** Replace `_graphviz_tile_pack_offsets(component_boxes)` with a new function
`_ogdf_tile_to_rows_packer(component_boxes)` that implements OGDF's TileToRowsCCPacker:

```python
def _ogdf_tile_to_rows_packer(
    boxes: list[tuple[float, float, float, float]],
    margin: float = 1.0,
) -> list[tuple[float, float]]:
    """Pack component boxes matching OGDF TileToRowsCCPacker (pageRatio=1).

    Sort by square-aspect-ratio area desc, then place in rows:
    - Row width = first (largest) component's width
    - Each component added to current row if it fits; else new row
    - Row height = max height of components in that row
    """
    if not boxes:
        return []
    # Compute (width, height) with margin on each side
    rects = [(box[2] - box[0] + 2*margin, box[3] - box[1] + 2*margin) for box in boxes]
    # Sort by OGDF square-aspect-ratio area desc, preserving original index
    order = sorted(range(len(boxes)),
                   key=lambda i: -_ogdf_fmmm_square_aspect_area(rects[i][0], rects[i][1]))

    row_width = rects[order[0]][0]  # First (largest) component sets row width
    offsets = [(0.0, 0.0)] * len(boxes)

    cur_x = 0.0
    cur_y = 0.0
    row_h = 0.0
    for i, idx in enumerate(order):
        w, h = rects[idx]
        if i == 0:
            offsets[idx] = (0.0, 0.0)
            cur_x = w
            row_h = h
        elif cur_x + w <= row_width + 1e-6:
            offsets[idx] = (cur_x, cur_y)
            cur_x += w
            row_h = max(row_h, h)
        else:
            cur_y += row_h
            cur_x = w
            row_h = h
            offsets[idx] = (0.0, cur_y)

    # Adjust from center-of-box to match _translate_packed_components_to_origin
    return [(offsets[i][0] + margin - boxes[i][0],
             offsets[i][1] + margin - boxes[i][1]) for i in range(len(boxes))]
```

**NOTE:** The exact OGDF TileToRowsCCPacker logic (row_width = first component width, row
placement rules) must be verified against OGDF source `TileToRowsCCPacker.cpp`. The
`_ogdf_fmmm_square_aspect_area` function (already at fmmm.py:925) provides the correct
sort key (same formula OGDF uses internally). The implementer must also verify the margin
convention matches OGDF's `getMinDistCC()` value.

### Verification

Run the 7 disconnected graphs above at steps100/200 after the fix. Expected:
- disp drops from 0.67-1.20 to near 0 (the internal layouts match; only packing was wrong)
- If disp drops but doesn't reach rung=1/2, the row-placement rule still differs from OGDF

### Expected impact

Resolving 15 divergent combos (all steps10/100/200 variants for the 7 graphs above).
Additionally may promote some fdp_fidelity combos for the same graphs, though those
have additional spring-force FP issues.

---

## M2: FP-CHAOS AT STEPS10 ITERATION FLOOR

**Status: FLOOR**
**Affected combos: 21 (steps10-only, connected graphs)**

### Affected graphs (all diverge at steps10, all pass at steps100/200)

asymmetric_hourglass_hub, broken_symmetry_residual_pair, clustered_longlabel_handoffs,
grid_50x50 (N=2500), grid_5x5, grid_rect_6x8, hexagonal_lattice_42, kitchen_sink_hybrid_net,
linear_3layer_mlp, multiscale_skip_cascade, nested_cluster_label_stack, nested_shallow_enc_dec,
residual_block, rgg_2000 (N=2000), sierpinski_42, small_world_2000 (N=2000),
tl_mlp_3layer, tl_resnet_2block, transformer_full_4h_2l, transformer_layer, triangular_lattice_36

### Root cause

For small graphs (N <= 50): `_layout_ogdf_fmmm_small_fidelity` runs
`max(100, 10 * int(steps))` iterations. At steps=10 this is 100 iterations.

For large graphs (N > 500): `_layout_ogdf_fmmm_multilevel_fidelity` uses
`_ogdf_fmmm_max_mult_iter` for each level. At level 0 (finest level) with
`fixed_iterations=10`, the formula yields exactly 10 iterations.

At 100 iterations (small graphs) or 10 iterations (large graphs), differences between
Python `math.sqrt` / `torch.*` operations and C++ `std::sqrt` / `libm` accumulate
into different attractor basins. The Lyapunov exponent of FMMM's force computation
is positive at low iteration counts.

### FLOOR evidence

The same graphs at steps100 (10-100x more iterations) all pass (rung=1/2). This proves
the FP differences can be overcome with enough iterations -- but the steps10 variant
genuinely falls into a different basin with only 100/10 iters. Examples:

- sierpinski_42: steps10 disp=11.324, steps100=rung2, steps200=rung2
- grid_5x5: steps10 disp=4.897, steps100=rung3Q, steps200=rung3Q
- rgg_2000: steps10 disp=2.287, steps100=rung2, steps200=rung3Q

Bit-exact emulation of C++ libm (libstdc++ `__ieee754_sqrt`) in Python is not feasible.

---

## M3: FP-CHAOS CONVERGING BUT NOT YET CONVERGED

**Status: FLOOR**
**Affected combos: 24 (steps100 and/or steps200 divergent, disp decreasing with more iters)**

### Affected graphs and disp pattern

| graph | N | steps10 | steps100 | steps200 |
|-------|---|---------|---------|---------|
| deep_chain_20 | 22 | rung4 (2.140) | rung4 (0.255) | rung4 (0.078) |
| weighted_chain_20 | 20 | rung4 (2.252) | rung4 (0.227) | rung4 (0.081) |
| sparse_pair_50 | 50 | rung4 (1.736) | rung4 (0.302) | rung4 (0.180) |
| small_world_100 | 100 | rung4 (2.783) | rung4 (0.271) | rung4 (0.179) |
| small_world_500 | 500 | rung4 (3.517) | rung4 (0.230) | rung4 (0.195) |
| long_range_residual_ladder | 38 | rung4 (1.785) | rung4 (0.422) | rung4 (0.315) |
| ragged_feature_pyramid | 12 | rung4 (1.597) | rung4 (0.386) | rung4 (0.383) |
| heavy_tail_weights_50 | 50 | rung4 (1.510) | rung4 (0.706) | rung4 (0.661) |
| org_chart_deep (M6) | 79 | rung4 | rung2 | rung4 (1.062) |
| protein_ppi_200 (M6) | 200 | rung2 | rung2 | rung4 (0.856) |

### Root cause and FLOOR evidence

For converging cases (deep_chain_20, weighted_chain_20, sparse_pair_50): disp is
monotonically decreasing but not yet at rung=1/2 at steps200. These would pass at
steps400-1000. But the evaluation only benchmarks steps10/100/200. These are
NOT structural -- they are FP-chaos basins that WIDEN at low iters and SHRINK as
iters converge to the energy minimum.

For plateau cases (ragged_feature_pyramid steps100=0.386, steps200=0.383): disp
barely changes from 100->200. N=12, so both are single-level. At this graph size
and topology, both dagua and OGDF reference are stuck in "different random attractor
basins" and 2000 iterations doesn't help. The plateau disp (~0.383) represents the
average Procrustes distance between draws from two different energy minima.

For heavy_tail_weights_50 (N=50, steps100=0.706, steps200=0.661): both dagua and
OGDF reference ignore edge weights (confirmed from ogdf_runner.cpp which creates
GraphAttributes without edgeDoubleWeight). The persistent divergence is pure FP-chaos
in the spring force accumulation.

OGDF itself uses libm `sqrt`, `exp`, `sin` etc. Matching their exact floating-point
trajectory requires bit-level emulation. FLOOR.

---

## M4: COMPOUND GRAPH CONNECTED FP-FLOOR

**Status: FLOOR**
**Affected combos: 14 (compound clustered graphs, connected)**

### Affected combos

| graph | N | steps10 | steps100 | steps200 |
|-------|---|---------|---------|---------|
| compound_10x20 | 200 | rung4 (1.641) | rung4 (0.351) | rung4 (0.204) |
| compound_dag_5x30 | 150 | rung4 (1.470) | rung4 (0.528) | rung4 (0.427) |
| resnet_stack_4x16 | 30 | rung4 (1.662) | rung4 (0.864) | rung4 (0.864) |
| tl_cnn_small | 10 | rung4 (3.775) | rung3Q | rung4 (0.303) |
| tl_transformer_1layer | 38 | rung4 (2.428) | rung4 (0.635) | rung4 (0.637) |

### Root cause

r72 (commit b0fc1e8) fixed the primary compound graph issue: dagua was incorrectly
routing clustered graphs through the cluster-force path even for OGDF fidelity. The fix
made OGDF fidelity (fidelity_mode=True) ignore cluster info entirely, passing graphs as
plain graphs.

The remaining 14 combos are pure FP-chaos in the plain-graph FMMM iterations:

- All 5 graphs are CONNECTED (single component, confirmed by _ogdf_fmmm_connected_components)
- No packing mismatch applies
- fidelity_mode=True does NOT pass cluster info (confirmed in _quick_classic: clusters only
  forwarded when fidelity_mode == "graphviz_fdp")
- resnet_stack_4x16 (N=30) and tl_transformer_1layer (N=38): disp CONSTANT at steps100/200
  (0.864/0.864 and 0.635/0.637). This is the "random attractor basin floor" -- both layouts
  are drawn from different basins, and median Procrustes between random basins is 0.64-0.86.
  2000 iters cannot distinguish them (same N, same force model, just FP differences).
- compound_10x20 (N=200): large multilevel graph, disp decreasing (1.641->0.351->0.204)
  -- converging but slow (same as M3 but for compound topology)

FLOOR for same reason as M3: bit-level libm emulation required.

---

## M5a: FDP EMULATOR DIVERGENCE -- DISCONNECTED AND SMALL GRAPHS

**Status: FLOOR**
**Affected combos: 10**

### Sub-cases

**Disconnected graphs at fdp_fidelity (5 combos):**
random_dag_50 fdp (disp=0.971), multi_component_80 fdp (0.816), disconnected_encoder_residual fdp (0.838), disconnected_label_cycle_collage fdp (0.885), parallel_cycles_4x5 fdp (0.821)

These use `_layout_fmmm_fidelity_components` -> `_graphviz_fdp_component_layout` for each
component, then `_graphviz_tile_pack_offsets` (correct for Graphviz FDP reference).
The packing is not the issue here. FP divergence in the spring force iterations inside
`_graphviz_fdp_component_layout` (pure Python vs Graphviz C) is the cause.

**Small/dense graphs at fdp_fidelity only, pass at all ogdf variants (5 combos):**
rgg_100 fdp (disp=0.405, e_rel=1.124), sbm_4x30 fdp (0.843), shape_and_routing_matrix fdp (1.118), extreme_mixed_width_transformer fdp (0.904), mixed_width_labels fdp (1.234)

These pass the OGDF FMMM reference at all step counts but fail the fdp_fidelity reference.
The OGDF FMMM algorithm and Graphviz FDP algorithm are fundamentally different
(multilevel FM^3 vs simple spring). The fdp emulator replicates Graphviz FDP's spring
forces in Python -- FP differences accumulate over iterations.

FLOOR: matching C libm sqrt/exp in Python spring iterations is bit-level emulation.

---

## M5b: MULTI-EDGE AGGREGATION IN FDP EMULATOR

**Status: FIXABLE**
**Affected combos: 1 (parallel_multiedge_bundle fdp)**

### Observed data

parallel_multiedge_bundle: N=3, E=6 (3 unique edges, each with 2 parallel copies).
fdp rung=4, disp=1.221, **e_rel=5.151** (dagua energy is 5x reference energy!)

The passes at OGDF all variants (rung2 at steps10/100/200). The fdp_fidelity path uses
`_graphviz_fdp_component_layout` which processes each edge in `edge_index`.

### Root cause hypothesis

Graphviz FDP aggregates parallel edges (multi-edges) into a single spring before computing
forces. Dagua's `_graphviz_fdp_component_layout` processes each edge in edge_index separately,
treating 2 parallel edges as 2 independent springs -- yielding 2x the attraction force.

e_rel=5.15 (dagua stress is 5x reference) is consistent with 2x spring force (force scales
quadratically with energy when springs are doubled up).

### Fix spec

**File:** `dagua/layout/ops/pipelines/fmmm.py`
**Function:** `_graphviz_fdp_component_layout` (find by name; search for `def _graphviz_fdp_component_layout`)

**Change:** Before the spring force computation loop, deduplicate parallel edges by aggregating
their weights (or counting multiplicity). If Graphviz FDP treats multi-edges as single edges
with unit weight, replace the raw edge_index with a deduplicated version. If Graphviz FDP
sums weights of parallel edges, multiply the single edge's weight by the multiplicity.

**Verify against Graphviz source:** Check `lib/fdp/fdp.c` or `lib/fdp/spring.c` for how
`fdp_tLayout` handles parallel edges in the spring force computation.

### Expected impact

1 divergent combo (parallel_multiedge_bundle fdp). Also may correct the fdp path for any
other multi-edge graph that crosses this code path.

---

## M6: ISOLATED STOCHASTIC BORDER CASES

**Status: FLOOR**
**Affected combos: 3**

### Cases

**org_chart_deep (N=79):** steps10=rung4 (disp=1.226), steps100=rung2 (PASSES!), steps200=rung4 (disp=1.062).
Alternating pass/fail across step counts with no monotone trend. This is a stochastic
border case where the energy landscape has two nearby minima; 1000 iters (steps100)
lands in the correct basin but 2000 iters (steps200) wanders to the wrong one.

**protein_ppi_200 (N=200):** steps10=rung2, steps100=rung2, steps200=rung4 (disp=0.856).
Same pattern: passes at 10 and 100, diverges at 200. N=200, multilevel path. At 2000
iters the reference and dagua settle into different coarse-level energy minima.

These are genuine stochastic FP-floor cases. No consistent signal to exploit. FLOOR.

---

## 12 INSUFFICIENT_DATA CASES: FDP EMULATOR PERFORMANCE

**All:** classic_fmmm_graphviz_fdp_fidelity, large graphs (N=200-500+)
**Reason:** `matched_seeds_lt_30` -- fewer than 30 seeds complete before timeout

The full list: ba_500, citation_dag_300, er_500, grid_20x20, hub_spoke_10x20,
hub_spoke_5x50, powerlaw_500, protein_ppi_200, random_dag_200, rgg_500, sbm_5x50,
small_world_500

### Root cause

The `_graphviz_fdp_component_layout` function is O(N^2) per iteration (all-pairs
repulsion). At N=200-500 with 200 iterations = 200*N^2 force evaluations in pure Python.
This is ~100x slower than needed for 30 seeds within benchmark timeout.

### Recommendation: Barnes-Hut approximation

Implement theta=1.0 Barnes-Hut (BH) approximation for the repulsion term in
`_graphviz_fdp_component_layout`. This reduces complexity to O(N log N) per iteration.

**Alignment with OGDF FMMM:** OGDF's FM^3 itself uses a multipole approximation (not
exact N^2 forces) for the repulsion term. A BH approximation at theta=1.0 would be BOTH
faster AND more faithful to the spirit of FMMM's force model.

**Fix scope:** Only the `_graphviz_fdp_component_layout` function in fmmm.py (the fdp
emulator's inner repulsion loop). The existing spring (attraction) term is already O(E).

**Expected gain:** ~50-200x speedup for N=200-500; should unblock 30-seed collection.
Note: this changes the fdp emulator's force trajectory, so it may change which combos
are divergent/passing. The newly unblocked large-graph fdp combos may land anywhere on
the rung scale.

---

## SUMMARY TABLE

| Mechanism | Combos | Status | Fix Location |
|-----------|--------|--------|-------------|
| M1: OGDF packer (TileToRows vs polyomino) | 15 | FIXABLE | fmmm.py:1484 in _layout_ogdf_fmmm_component_fidelity |
| M2: FP-chaos steps10 floor | 21 | FLOOR | -- |
| M3: FP-chaos converging slowly | 24 | FLOOR | -- |
| M4: Compound connected FP-floor | 14 | FLOOR | -- |
| M5a: FDP spring FP divergence | 10 | FLOOR | -- |
| M5b: Multi-edge aggregation FDP | 1 | FIXABLE | fmmm.py: _graphviz_fdp_component_layout |
| M6: Stochastic border | 3 | FLOOR | -- |
| **TOTAL** | **88** | **16 combos addressable** | |
| 12x INSUFFICIENT: FDP O(N^2) timeout | 12 | PERF FIX | Barnes-Hut in _graphviz_fdp_component_layout |

**Highest-value fix: M1 (15 combos)**. The OGDF packer change is a single-function
replacement at line 1484, using an existing helper (`_ogdf_fmmm_square_aspect_area`
at line 925 already implements the OGDF sort key). The row-placement algorithm needs
OGDF source verification for exact row_width and placement rules.

**Codex implementer must verify:** OGDF `TileToRowsCCPacker.cpp` for exact pageRatio=1
row-width rule and row height tracking. Do NOT implement from the paper; match the code.
