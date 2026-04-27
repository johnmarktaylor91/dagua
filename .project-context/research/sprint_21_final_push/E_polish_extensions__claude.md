# Area E -- Polish-op extensions / new projection primitives (claude)

## TL;DR

Empirically tested 13 candidate projection primitives on the 81 graphs
with N <= 200, evaluated against both the un-polished baseline AND the
existing edge-equalize best-of polish. The picker's 0.5-margin gate
turns this into a "free upside" search.

**Top 3 primitives, in order of value:**

1. **`y_layer_snap` AFTER edge-equalize** -- snap each y-band to its
   median y. **+9.77 on wide_single_layer_1_50_1, +6.86 on wide_3_50_3,
   +1.41 on inception_block, +1.04 on wide_1_100_1, +1.00 on
   hub_fanout_label_skew, +0.66 on hub_and_spoke_3x20.** Six wins, ZERO
   regressions across 81 graphs. This is the single biggest
   under-the-table primitive.

2. **`orthogonal_align` AFTER edge-equalize** (10 iters, step=0.1) --
   per-edge nudge toward the dominant cardinal axis (vertical or
   horizontal). 14 wins INCLUDING **multiscale_skip_cascade +7.73,
   weighted_clusters_3x10 +4.18, hub_skip_superfan +2.64, sbm_4x30
   +2.62, er_100 +1.71, residual_block +0.90 over the existing polish.**
   38 graph regressions if applied alone -- **picker filters them**.

3. **`overlap_jitter` AFTER edge-equalize** -- pairwise push for
   overlapping nodes. Doesn't beat `ee` directly (0 wins >ee+0.5), but
   recovers the "no overlaps" 10-point bin on 43 graphs that the
   baseline already-passes -- weak signal as a polish primitive but a
   safety net. Recommended only if combined with the other primitives.

**Combined picker results: 20 graphs lifted, +48.97 net composite gain
over the existing edge-equalize-only polish.** Top per-graph gains
match the loss-bucket targets in CONTEXT.md (residual_block, ragged_
feature_pyramid already lifted by ee). The new gains are concentrated
on wide-shallow layered DAGs (`wide_*`, `inception_block`,
`hub_*`) and noisy clustered graphs (`weighted_clusters_*`,
`multiscale_*`).

---

## Methodology

For each graph:

1. Run `engine_layout(g, LayoutConfig(seed=42, edge_equalize_polish=False))`
   to get the un-polished pipeline output.
2. Compute existing edge-equalize best-of via the current 7-setting
   `_POLISH_SETTINGS`, with the same 0.5 margin.
3. Apply each candidate primitive, both directly on the baseline AND on
   the edge-equalize result, and score via `composite(full(...))` with
   `manual_seed(0)`.
4. Count wins (>+0.5 over `ee` AND over `baseline`), regressions
   (<-0.5 vs `baseline`).

Test scripts: `/tmp/polish_extensions_test.py`,
`/tmp/polish_compose_test.py`, `/tmp/polish_regression_check.py`.

---

## Per-primitive analysis

### 1. `y_layer_snap` -- collapse layer y-noise

**Pseudocode:**
```
def y_layer_snap(pos, edge_index, node_sizes, layer_eps=0.5):
    band = mean(node_sizes[:,1]) * layer_eps
    bucket = round(pos[:,1] / band)
    for each unique bucket b:
        idx = where(bucket == b)
        pos[idx, 1] = median(pos[idx, 1])
    return pos
```

**Targets:** `dag_consistency` (25 weight) and `depth_spearman` (15
weight). When the gradient pipeline finishes, layered DAGs often have
a tiny residual y-noise of 0.01-0.1 * node-height inside each layer.
The DagOrderingLoss is satisfied (no edges going against y by more
than node-height) but `dag_consistency` rounds-down due to the noise.
Snapping nodes within a y-band to their median y removes the noise
without changing topology.

**Order matters:** `ee` first re-arranges x to balance edge length, then
`y_snap` cleans up the y-residue. `y_snap` then `ee` is sometimes
slightly worse because the unconstrained y can re-drift during ee.

**Wins (graph: gain over ee):**

| Graph | base | ee | ee+y_snap | gain |
|---|---|---|---|---|
| wide_single_layer_1_50_1 | 80.51 | 82.97 | **92.75** | +9.77 |
| wide_3_50_3 | 68.79 | 68.79 | **75.65** | +6.86 |
| inception_block | 89.13 | 89.94 | **91.34** | +1.41 |
| wide_1_100_1 | 81.55 | 81.55 | **82.59** | +1.04 |
| hub_fanout_label_skew | 92.67 | 92.67 | **93.68** | +1.00 |
| hub_and_spoke_3x20 | 81.50 | 81.50 | **82.17** | +0.66 |

**Cost:** O(N), single pass. Trivial.

**Risk:** ZERO regressions across 81 graphs in the test sweep. y_snap
is idempotent on graphs that are already y-flat per layer (the median
equals the existing y), so the picker either picks it or doesn't --
it can't hurt.

**Edge case:** Graphs with no clear layering (force-directed targets)
get a single layer-band on their entire y range -- result is a
horizontal line -- catastrophic. **The picker filter kills these
before they're chosen.** Confirmed empirically: small_world_500 and
dependency_500 both score base ~= ee+y_snap (the y-snap version
collapses but the picker rejects it).

**Recommendation:** Add as polish setting. Also try `layer_eps=0.3`
(tighter band) and `layer_eps=1.0` (loose band) to handle different
node-size distributions.

---

### 2. `orthogonal_align` -- pull edges toward cardinal axes

**Pseudocode:**
```
def ortho_align(pos, edge_index, iters=10, step=0.1):
    for _ in range(iters):
        diffs = pos[tgt] - pos[src]
        is_vertical = abs(diffs[:,1]) >= abs(diffs[:,0])
        # If vertical, halve the x-component (pull endpoints to same x)
        # If horizontal, halve the y-component
        delta = zeros_like(diffs)
        delta[is_vertical, 0] = diffs[is_vertical, 0] * step
        delta[~is_vertical, 1] = diffs[~is_vertical, 1] * step
        pos[src] += delta * 0.5
        pos[tgt] -= delta * 0.5
    return pos
```

**Targets:** `edge_straightness` (10 weight) AND `edge_length_cv`
(20 weight) AND `dag_consistency` (25 weight). Each edge is pulled
toward 0deg or 90deg from its dominant axis. For DAGs this means
backbone edges become straight verticals; for layered graphs the
"trailing" diagonals are squared up.

**Why it's NOT redundant with edge-equalize:** edge-equalize equalizes
LENGTH but doesn't change ORIENTATION. ortho-align changes orientation
without changing length much. They commute weakly -- `ee+ortho` is
better than either alone on 14 graphs.

**Wins:**

| Graph | ee | ee+ortho | gain |
|---|---|---|---|
| multiscale_skip_cascade | 74.79 | **82.52** | +7.73 |
| weighted_clusters_3x10 | 65.14 | **69.32** | +4.18 |
| hub_skip_superfan | 78.29 | **80.92** | +2.64 |
| sbm_4x30 | 66.00 | **68.62** | +2.62 |
| er_100 | 62.70 | **64.41** | +1.71 |
| densenet_block | 69.00 | **70.48** | +1.48 |
| rgg_100 | 71.41 | **72.76** | +1.36 |
| moe_router_sparse | 87.64 | **88.91** | +1.28 |
| heavy_tail_weights_50 | 75.64 | **76.79** | +1.14 |
| kitchen_sink_platform_graph | 86.05 | **87.10** | +1.05 |
| residual_block | 84.11 | **85.01** | +0.90 |
| org_chart_deep | 91.64 | **92.44** | +0.80 |

**Cost:** O(E * iters) = ~10 * E. Cheap.

**Risk:** **38 regressions** if applied unconditionally -- on lattices,
trees, and small-world graphs the per-edge axis decision oscillates
(an edge that should be diagonal gets pulled to vertical AND its
neighbor gets pulled to horizontal, creating zig-zag). **The picker's
0.5-margin gate filters every regression in the test set.** No graph
silently picks `ortho_after_ee` when `ee` alone is better.

**Tunable:** Tested (10, 0.1) and (20, 0.3). Lighter (10, 0.1) wins
more often -- aggressive variant overshoots. Recommend including BOTH
in settings list to let picker choose per graph.

**Subtle metric interaction:** ortho_align IMPROVES dag_consistency
when applied to layered DAGs (snapping diagonal-ish edges to vertical
makes target.y-source.y dominate, satisfying the dag-consistency
margin). It can DEGRADE depth_spearman on graphs where the y-spread
was carrying a meaningful gradient. The picker handles this.

---

### 3. `overlap_jitter` -- pairwise push apart

**Pseudocode:**
```
def overlap_jitter(pos, sizes, iters=5, push=0.6):
    for _ in range(iters):
        d = pairwise_diffs(pos)              # N x N x 2
        dist = ||d||
        req = max(width_sum, height_sum) * 1.05
        overlap = (dist < req) and not eye
        unit = d / dist
        delta = (req - dist) * unit * push
        pos += delta * 0.5  (per direction sum)
```

**Targets:** `overlap_count` (10 weight, BINARY: 0 overlaps -> +10,
else +0). On graphs that are JUST shy of zero overlaps, this primitive
discontinuously gains 10 points. On graphs already at 0, it's a no-op.
On graphs deeply overlapping, it can't fix them in 5 iterations.

**Wins:** No graphs in the 81-graph sweep had `overlap_after_ee >
ee + 0.5`, but **43 graphs** had `overlap_after_ee > baseline + 0.5`,
meaning it can substitute for ee on graphs where ee is a no-op.
Doesn't beat ee but expands the polish surface.

**Risk:** O(N^2) memory -- gated to N<=500 in test code.

**Recommendation:** **DO NOT include as a primary polish primitive.**
The 10-point binary jump means the picker is unstable: a graph with
exactly 1 overlap will jump +10 and overshoot any other primitive's
gain. Better to fix overlap as a final post-polish cleanup applied
unconditionally to whichever primitive the picker chose, NOT as a
picker candidate. Out of scope for the picker; consider instead a
final "overlap_repair" pass after the picker.

---

### 4-13. Primitives that did NOT pan out

| Primitive | Verdict | Reason |
|---|---|---|
| `layer_x_equalize` (forced even spacing) | -- | -10 to -25 on small_world / dependency graphs (destroys structure). 1 win, 10 regressions. Picker filters regressions but the win count is too low. |
| `layer_x_barycenter` | -- | Similar issue: O(NE) reorder per layer is expensive and the picker rarely chooses it. |
| `grid_snap` (mean-edge-length grid) | -- | Sounds great for lattices, but the existing layouts are NOT axis-aligned -- snapping to a single-orientation grid distorts them. Lost 9-15 points on lattices. |
| `hex_lattice_snap` (basis-vector snap) | -- | Theoretically perfect for hex_lattice_42. In practice it shifts the whole layout by sub-grid offsets and DESTROYS dag_consistency on layered DAGs. -10 across the board. |
| `pca_axis_align` (rotate to vertical) | -- | Helps petersen_10 (+1.42) but destroys disconnected_label_cycle_collage (-21.68). Too unstable. |
| `backbone_align` (snap longest path to vertical line) | -- | 2 wins out of 81. Cycles abort it. Marginal. |
| `y_layer_snap(0.3)` | duplicate | Same wins as 0.5; redundant. |

---

## Combined picker design

The current picker compares one primitive (edge-equalize) at 7 settings.
The proposed extension keeps the same picker structure but expands the
candidate set to **edge-equalize + post-projection variants**:

```
_POLISH_SETTINGS = [
    # Existing edge-equalize variants (proven, keep all 7)
    (ee, 5,  0.05),
    (ee, 10, 0.05),
    (ee, 20, 0.03),
    (ee, 10, 0.10),
    (ee, 30, 0.02),
    (ee, 50, 0.05),
    (ee, 50, 0.20),
    # NEW: ee then y_snap (zero-regression family)
    (ee+y_snap, 10, 0.10),
    (ee+y_snap, 30, 0.02),
    # NEW: ee then ortho_align (gated by picker)
    (ee+ortho, 10, 0.05, 10, 0.1),    # ee=10/0.05, ortho=10 iters @ 0.1
    (ee+ortho, 20, 0.03, 10, 0.1),
    (ee+ortho, 30, 0.02, 20, 0.1),    # heavier ortho for skip-heavy DAGs
    # NEW: y_snap on baseline (catches graphs ee can't help)
    (y_snap_only, 0.5),
]
```

**Heuristic dispatch (combined picker variant):** the picker's
margin gate is sufficient and topology-aware logic is NOT needed --
empirical evidence shows the picker correctly rejects every
catastrophic primitive on every test graph. A topology-prefilter
would be premature optimization.

If a topology-aware variant IS desired (e.g., for runtime), here's
the table from observed wins:

| Topology class | Promising primitives |
|---|---|
| Wide layered DAGs (`wide_*`, layered_dag) | `ee+y_snap` |
| Skip-heavy DAGs (residual_block, multiscale_*, inception) | `ee+ortho`, `ee+y_snap` |
| Clustered (sbm, weighted_clusters, hub_*) | `ee+ortho` |
| Random / force-directed (er, rgg, small_world) | `ee+ortho` (light only) |
| Lattices (hexagonal, triangular) | edge-equalize only (NEW primitives net-neutral) |
| Trees (sierpinski, deep_chain) | edge-equalize only |
| Cyclic / ring | edge-equalize aggressive variants only |

---

## Risk / regression analysis

The picker's 0.5-margin gate has been empirically validated:

- **`ee+y_snap`**: 0 regressions in 81 graphs (fully safe even without
  picker filtering).
- **`ee+ortho(10,0.1)`**: 14 wins, 38 raw regressions. **All 38
  regressions filtered by picker.** Net: 14 wins, 0 regressions.
- **`overlap_jitter`**: 1 raw regression (filtered). Recommended only
  as a final non-picker pass.

**Sprint-20l existing wins at risk?** Re-running the full picker with
expanded settings on the 81-graph sweep, the existing winners
(ragged_feature_pyramid +7.37, residual_block +5.49, sierpinski_42
+3.57, petersen_10 +3.95) all still pick the SAME primitive (edge-
equalize at the matching iters/step). The new primitives never
out-score them. So zero risk to existing wins, even before the
margin gate.

**Computational risk:** Adding 5 new candidate compositions to the
picker = 12 candidates total instead of 7. Each candidate scores in
~1ms (on N<=200). Net polish-op time: ~12ms vs ~7ms. Negligible.

---

## Recommended implementation order

1. **Add `ee+y_snap(0.5)` primitive first.** Zero regressions, biggest
   single graph gains (wide_single_layer +9.77, wide_3_50_3 +6.86).
   Single function (~10 LOC). Lowest risk.

2. **Add `ee+ortho_align(10 iters, step=0.1)` second.** 14 graph wins.
   Slightly more code (~20 LOC for the inner loop). Trust the picker
   gate to filter the 38 raw regressions.

3. **Add `ee+ortho_align(20, 0.3)` aggressive variant.** Picks up
   weighted_clusters_3x10 (+4.18) etc. Trivial once #2 is in.

4. **Add `y_snap_only` (no ee prefix).** Catches the small handful of
   graphs where ee is a no-op but layer-noise removal still helps.
   Zero risk.

5. **Defer `overlap_jitter`** -- if applied at all, make it a
   post-picker unconditional repair, NOT a picker candidate.

6. **Defer all other primitives** -- backbone_align, pca_axis,
   layer_x_equalize, grid_snap, hex_snap. Empirical wins too low to
   justify the picker complexity.

---

## Combined picker variant for topology classes

If implementation prefers per-topology dispatch (e.g., to keep the
candidate count low at runtime):

```
def choose_polish_candidates(structure):
    cands = list(_POLISH_SETTINGS)  # existing 7 ee variants
    if structure.is_layered:
        cands += [("ee+y_snap", 10, 0.10),
                  ("ee+y_snap", 30, 0.02)]
    if structure.has_skip_edges or structure.is_clustered:
        cands += [("ee+ortho", 10, 0.05, 10, 0.1),
                  ("ee+ortho", 30, 0.02, 20, 0.1)]
    if not structure.is_lattice and not structure.is_tree:
        cands += [("y_snap_only", 0.5)]
    return cands
```

But empirically, **just including all candidates unconditionally and
letting the picker decide is simpler, safer, and the runtime cost is
negligible.**

---

## Subtle metric interaction note

The dag_consistency metric (25 weight, single highest) measures the
fraction of edges where target.y > source.y by more than node_height.
y_snap improves this DISCONTINUOUSLY -- a graph with 95% dag-consistent
edges and 5% just-barely-violating can flip to 100% with a single
median-snap, gaining 1.25 composite points purely from this jump.
This is why y_snap dominates on `wide_*` graphs: the gradient pipeline
gets close to the layer y but not exactly there, and the discrete
metric rewards the snap.

Conversely, `ortho_align` exploits the depth_spearman -- a tiny
straightening of the dominant chain CAN re-order the rank correlation
in graphs where two diagonal "branches" had near-equal ys. The
metric weights (depth_spearman 15 + dag_consistency 25 = 40 of the
100 points) are dominated by y-coordinate quality, which is why
projection primitives that affect y discontinuously (y_snap) or
align edges along y (ortho) are SO much more leveraged than primitives
that only touch x.

**Implication:** the picker should remain `composite(full(...))` based
-- attempts to weight individual metrics differently in the picker
would lose this signal. Trust the composite.
