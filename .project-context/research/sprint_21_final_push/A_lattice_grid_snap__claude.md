# Area A — Lattice grid-snap (claude sibling)

## TL;DR

- **The right framing is not "grid snap" — it's "axis-aligned coord
  quantization."** dot's hex_lattice_42 layout has 18 unique x and 12
  unique y values for 42 nodes (`eval_output/variant_bench_full/
  positions/hexagonal_lattice_42__graphviz_dot.pt`); edge lengths range
  72.0 to 101.8 (CV=0.099). It's *not* a perfect hex grid (a true hex
  grid would have CV=0). dot wins by snapping nodes to a small ladder
  of x-coordinates within each layer while preserving rank order. That
  is reproducible from dagua's polished output in O(N+E) without an
  expensive grid-fit search.
- **Don't chase pure CV — ogdf_stress hits CV=0.027 on hex_lattice and
  scores 55.45 vs dot's 88.99.** Dropping CV alone is not the win;
  preserving dag_consistency=1.00, depth_spearman>=0.8, zero overlaps
  AND straightness>=0.4 *while* shaving CV from 0.51 to ~0.10 is. This
  rules out KK-style "true unit graph" approaches.
- **Highest-leverage primitive: per-layer x-quantization on lattice-
  tagged graphs.** Bin x within each y-rank to the nearest of K
  cluster centers (1-D K-means with K = round(median_layer_width)).
  Closes most of the CV gap without touching dag/depth/overlap.
  Projected delta on hex_lattice_42: ~+5 to +6 composite. Expected
  after-polish dagua score: ~91-92, vs dot's 88.99.
- **Triangular_lattice_36 and sierpinski_42 may already be tied at
  HEAD.** Measured competitor scores (from cached positions, my
  `/tmp/lattice_score.py`) on triangular_lattice_36 are
  ogdf_sugiyama=87.16 (best), dot=87.09, ogdf_pivot_mds=86.45. Stale
  cached `dagua.pt` scores 86.78 — within 0.4 of best. Even if HEAD
  drifted to 85, a grid-snap targets at most ~+1 here.
- **Petersen_10 is structurally unfit for grid-snap.** N=10, 3-regular
  non-planar, dot itself only scores 72.07 (loses to igraph_sugiyama
  77.36). The win lies elsewhere; defer to area B.
- **Recommended scope: hex_lattice + (maybe) triangular_lattice only,
  gated on a strict classifier signal, executed as a `_best_of_polish`
  candidate** — same picker safety net as sprint-20k.

## What I measured

Scripts: `/tmp/lattice_inspect.py` and `/tmp/lattice_score.py`.
Composite scoring uses the same `composite(full(pos, ei, ns))` path
the benchmark uses with `torch.manual_seed(0)`, mirroring the seeded
buckets script (`/tmp/h2h_buckets_seeded.py:34-36`).

### hex_lattice_42 — competitor geometry

| engine | composite | CV | dag | depth | ovl | strt | uniq_x | uniq_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| graphviz_dot | **88.99** | 0.099 | 1.00 | 0.82 | 0 | 0.45 | 18 | 12 |
| dagre | 82.88 | 0.317 | 1.00 | 0.82 | 0 | 0.47 | 21 | 12 |
| igraph_sugiyama | 82.19 | 0.505 | 1.00 | 1.00 | 0 | 0.49 | 14 | 12 |
| ogdf_sugiyama | 78.72 | 0.076 | 1.00 | 0.82 | 1 | 0.21 | 31 | 12 |
| elk_layered | 76.71 | 0.576 | 1.00 | 0.82 | 0 | 0.49 | 23 | 12 |
| ogdf_pivot_mds | 76.77 | 0.264 | 0.85 | 0.89 | 0 | 0.00 | 42 | 42 |
| graphviz_neato | 47.43 | 0.027 | 0.02 | -0.66 | 0 | 0.30 | 42 | 41 |
| ogdf_stress | 55.45 | 0.027 | 0.34 | -0.83 | 0 | 0.00 | 42 | 42 |

Three observations:

1. **All five DAG-aware engines (dot, dagre, both sugiyamas, elk)
   collapse y to ~12 layers** — that's the rank-snap of layered
   layouts. dagua already does this (depth_spearman=1.00 in the
   prompt evidence).
2. **dot stands alone in *also* collapsing x to 18 discrete values.**
   The other layered engines have 21-31 unique x's, which is what
   keeps their CV at 0.31-0.58. ogdf_sugiyama tries 31 unique x and
   gets CV=0.076 (the best layered CV) but loses 5 points on overlap +
   straightness.
3. **dot's straightness=0.45 vs sugiyama's 0.21** is a second secret:
   the coarse x-grid forces edges into a small set of slope bins,
   which the straightness metric rewards.

The implication is precise. **dagua's polished hex_lattice output
already has 1.00 depth_spearman, 0 overlaps, AND 1.00 straightness
(better than dot at 0.45) — only CV is bad at ~0.51.** Quantizing
dagua's x within each layer should drop CV without harming the rest,
because the y-coordinates are already on the discrete layer ladder.

### triangular_lattice_36

| engine | composite | CV | strt | uniq_x | uniq_y |
|---|---:|---:|---:|---:|---:|
| ogdf_sugiyama | **87.16** | 0.140 | 0.32 | 10 | 11 |
| graphviz_dot | 87.09 | 0.233 | 0.32 | 11 | 11 |
| ogdf_pivot_mds | 86.45 | 0.061 | 0.29 | 19 | 19 |
| elk_layered | 85.08 | 0.302 | 0.27 | 15 | 11 |

The triangular gap is small. Stale cached `dagua.pt` scores 86.78 —
within 0.4 of best.

### sierpinski_42

| engine | composite | CV | strt | uniq_x | uniq_y |
|---|---:|---:|---:|---:|---:|
| graphviz_dot | **84.29** | 0.353 | 0.41 | 36 | 23 |
| elk_layered | 84.26 | 0.362 | 0.40 | 15 | 23 |
| ogdf_sugiyama | 79.87 | 0.524 | 0.30 | 37 | 23 |

Sierpinski is fractal — dot does NOT grid-snap it (36 unique x of 42
nodes). dot wins on dag+depth+overlap, not CV. **Grid-snap will not
help sierpinski.** The lattice classifier should explicitly NOT fire
on it.

### petersen_10

dot=72.07, dot CV=0.456, only 7 unique x and 6 unique y. dot is not
the best (igraph_sugiyama=77.36). Any "snap to grid" approach is the
wrong tool — Petersen is structurally non-planar 3-regular; the right
answer is a circular/cage layout, not a grid. Defer to area B.

## Algorithm sketch

A post-pipeline projection added as a new `_POLISH_SETTINGS`-style
candidate, scored against the un-polished + edge-equalize candidates,
and accepted only if it beats them by `margin >= 0.5`.

```
def _layered_x_quantize(pos, edge_index, layer_assignments,
                        k_per_layer=None, axis_lock=True):
    """Quantize x within each layer to the nearest of K cluster centers.

    Preconditions checked by caller (lattice classifier gate):
      - depth_spearman >= 0.95 on input pos (rank-aligned with y)
      - num_layers >= 5
      - layer_width_cv <= 0.30 (dense regular layers)
      - 1.2 <= edge_to_node_ratio <= 2.0 (planar regime)

    Algorithm (O(N log N) per pass):
      1. Group node indices by layer_assignments[i].
      2. For each layer L with width w_L:
           K_L = min(w_L, k_per_layer or round(median_layer_width))
           Run 1-D K-means on pos[L, 0] with K_L centers, max 8 iters
           (K-means in 1-D collapses to sort + Otsu-style optimal cuts;
            scipy.cluster.vq.kmeans2 is fine for K <= 64).
           Snap each node's x to its assigned center.
      3. Compute global lattice step h = median over edges of
         |pos[u].x - pos[v].x| over non-vertical edges.
      4. Optional second pass (`axis_lock=True`): pick a global phase
         offset phi in [0, h) that minimizes total node displacement
         when each layer's center set is snapped to {phi + k*h}. This
         is a 1-D periodic alignment problem — sweep 60 phi candidates,
         O(N * 60).
      5. y is left untouched (already on the discrete layer ladder).

    K estimation:
      median_w = median(layer_widths)
      K = round(median_w)               # one bin per typical-layer node
      For hex with stagger, prefer K = round(median_w * 1.5), since
      adjacent rows are offset by half-a-cell.
    """

def _equalize_edges_with_axis_lock(pos, edge_index, iters, step,
                                   layers, lock_x_to_grid_every=5):
    """Variant of _equalize_edges that re-quantizes x onto the layered
    grid every `lock_x_to_grid_every` iterations. Keeps the projection
    from drifting off the grid while still equalizing edge lengths.
    """
```

Both go into `_LATTICE_POLISH_SETTINGS` (a parallel tuple to
`_POLISH_SETTINGS` at `dagua_native.py:376`), invoked only when the
classifier fires. The `_best_of_polish` picker
(`dagua_native.py:436-488`) provides the 0.5-composite-margin safety
net automatically.

### Why K-means and not "snap to integer multiples of h"

dot's x-set on hex_lattice_42 is `{27, 81, 117, 135, 153, 207, 225,
243, 261, 279, 297, 315, 333, 351, 387, 405, 441, 459}`. Successive
deltas are mostly 18 or 36 (one or two grid steps), but not strictly
periodic — some layers are offset by half a step (the honeycomb
stagger). A rigid integer grid would over-constrain. K-means per
layer respects the structure dagua's gradient pipeline already
discovered while collapsing within-layer micro-jitter.

## Expected composite delta

Computed by holding all metrics except `edge_length_cv` and
`edge_straightness_below_15` constant, replacing CV with an estimated
post-snap value, and recomputing composite weights from
`dagua/metrics.py:1147` (CV worth 20pts, straightness worth 10).

| Graph | dagua HEAD | best comp | gap | est. post-snap | est. delta |
|---|---:|---:|---:|---:|---:|
| hexagonal_lattice_42 | 86.46 | 88.99 (dot) | -2.52 | ~91-92 | **+4.5 to +5.5** |
| triangular_lattice_36 | ~85 | 87.16 (sugi) | ~-2.0 | ~86-87 | +1 to +2 |
| sierpinski_42 | ~78-80 | 84.29 (dot) | -4 to -6 | unchanged | 0 (gate skips) |
| petersen_10 | 74.64 | 77.36 (sugi) | -2.72 | unchanged | 0 (gate skips) |

Net suite delta: **+5 to +7 composite** concentrated on hex_lattice +
triangular_lattice. Worth one engineering session.

## Risk / regression analysis

**At-risk wins:**

1. **grid_20x20** (square-grid graph in the suite). Any lattice-snap
   that fires here must land on a clean square grid — if the
   classifier mis-fires and K-means over-collapses, this risks the
   existing strong win. Mitigation: gate uses `is_planar_hint AND 2
   <= max_degree <= 4 AND e/n in [1.6, 2.0]` for square; keep a
   tighter band for hex/triangular.
2. **dependency_graph_100, er_100 wins from sprint-20j.** Dense DAGs
   that share `is_planar_hint`. Today they get tagged `dense_dag` not
   `lattice_like` (graph_classify.py:579-597 show the bands are
   disjoint by `max_degree` and `layer_width_cv`). Low risk if the
   gate adds an additional tightening: `layer_width_cv <= 0.30`
   (current `lattice_like` is 0.45 — too loose for our purposes).
3. **Tree pipeline outputs.** RT already produces discrete x.
   Lattice snap fires only via the layered_dag/hybrid path.
4. **disconnected_label_cycle_collage / petersen_10 sprint-20l polish
   wins.** Both accepted by `_best_of_polish` because aggressive
   variants (50 iters) help. The lattice candidate must not displace
   them — picker margin selection makes this safe automatically.

**Sprint-19/20 invariants preserved:**

- aspect_target=0.05 for `lattice_like` (`resolve.py:149-150`).
  Lattice snap runs *after* the layout pipeline so this is unaffected.
- The s20i stress route at the top of dispatch (small_world_100). The
  stress route doesn't share layer_assignments structure; lattice
  snap should not fire on its outputs because `num_layers >= 5` will
  be false in flat-layered topology.
- s20j w_straightness=0.5. Independent of post-processing.

**Failure modes to test:**

- **K-means collapsing two adjacent nodes on top of each other** — CV
  drops but overlap increases. Guard: reject candidate if any pair
  within node_size after snap.
- **Layer assignments missing.** layered_dag pipeline supplies them,
  but force_directed/legacy_monolith may not. Guard: only fire when
  `layer_assignments is not None and num_layers >= 5`.
- **Off-by-half-step on hex stagger.** If K-means lands on K=18 with
  the wrong phase, edge CV could regress. Mitigation: compute CV
  pre/post and route through the picker — automatic safety.

## Recommended gate

```python
def should_lattice_snap(structure, layer_assignments, num_nodes,
                       polished_pos, edge_index):
    if structure is None or layer_assignments is None:
        return False
    tags = set(structure.topology_tags)
    if "lattice_like" not in tags:
        return False
    # Tighter band than the existing lattice_like tag.
    if not (1.2 <= structure.edge_to_node_ratio <= 2.0):
        return False
    if structure.layer_width_cv > 0.30:
        return False
    if structure.num_layers < 5 or structure.num_layers > 60:
        return False
    if num_nodes < 24 or num_nodes > 400:
        return False
    # Empirical: only fire when CV is the bottleneck. If the polished
    # output already has CV < 0.20 the snap can't gain enough.
    cv_now = compute_edge_cv(polished_pos, edge_index)
    if cv_now < 0.20:
        return False
    return True
```

Empirical e/n values (from cached graphs): hex_lattice_42 = 53/42 =
1.26; triangular_lattice_36 = 85/36 = 2.36 (above the 2.0 cap, so it
falls outside this gate by default). To bring triangular in, raise
the upper bound to 2.5 — but that risks pulling in
dependency_graph_100. **First implementation: ship the hex-only band
(1.2-2.0). Triangular's ceiling is +2 anyway; not worth the
classification headache for sprint 21.**

## Open questions

1. **Does sierpinski's `lattice_like` tag fire today?** Classifier
   uses `1.0 <= e/n <= 2.2` and sierpinski has e/n=81/42=1.93 —
   probably yes. The new `should_lattice_snap` adds
   `layer_width_cv <= 0.30` which sierpinski's irregular branching
   should fail; verify with a one-line probe before relying on it.

2. **Is the layered_dag pipeline the only entry point?** Per
   `dagua_native.py`, hex_lattice goes through `layered_dag` via
   `_choose_native_pipeline` (planar+layered). Need to verify
   triangular_lattice doesn't get diverted to force_directed (its e/n
   = 2.36 may push it there). Cross-check by reading
   `_choose_native_pipeline` for the four targets.

3. **Should the snap be a separate `_best_of_polish` candidate or
   replace the current ones for lattice graphs?** *Additional* is
   right. The gradient pipeline output is already a strong baseline;
   the picker's margin enforcement is the safety net.

4. **K-means on x within layer is O(N log N). Phase-sweep is O(N *
   60). Total cost on hex_lattice_42:** ~10K ops, negligible.

## Implementation order

1. **Add the K-means-per-layer primitive as a new ops module
   `dagua/layout/ops/lattice_snap.py`** with `LatticeSnapConfig`
   dataclass. Register the op. Unit tests on a synthetic 6x7 hex
   pattern proving CV drops to <0.05 without overlap.
2. **Wire it into `_best_of_polish` as one new candidate** behind
   `should_lattice_snap`. Existing picker margin handles regressions.
3. **Run h2h_buckets_seeded.py** before/after, attending to
   hex_lattice_42, triangular_lattice_36, sierpinski_42, petersen_10,
   grid_20x20 (sanity check), and sprint-20j/k protected wins
   (dependency_graph_100, er_100,
   disconnected_label_cycle_collage).
4. **Iterate on K estimation** if hex_lattice gain < +3. Likely fix:
   bias K toward `round(median_layer_width * 1.5)` for hex stagger.
5. **Only after hex gain >= +3** consider extending to triangular
   (raise e/n cap and re-run the protected-wins sanity check). Skip
   sierpinski and petersen entirely from this work item.

## Why not the obvious alternatives

- **Lloyd relaxation on a perfect hex lattice** would force CV → 0
  but destroy depth_spearman. Net negative.
- **Replace the layered_dag pipeline with a Sugiyama-mimic for
  lattice_like** — too invasive; ogdf_sugiyama only scores 78.72
  (worse than dagua HEAD's 86.46), so this regresses.
- **Add an `EdgeLengthVarianceLoss` schedule that ramps up at
  convergence** — already tried and ruled out (CONTEXT.md:97-99: the
  loss is plumbed; gradient saturated). The polish picker exists
  precisely because gradient methods can't finish the CV job.
- **PCA-rotate then snap to global square grid** works on
  square_lattice/grid_20x20 but breaks on hex (hex needs 60-deg axes
  not 90-deg). The per-layer K-means is implicitly axis-agnostic.

The K-means-per-layer + picker-gated polish candidate is the path
with the highest expected value, the smallest blast radius, and the
existing safety infrastructure to catch regressions. Net projected
contribution to sprint 21: **+5 to +7 composite** on the deterministic
93-graph suite, almost entirely from hex_lattice_42 going from -2.52
delta to +2 to +3 delta vs graphviz_dot.
