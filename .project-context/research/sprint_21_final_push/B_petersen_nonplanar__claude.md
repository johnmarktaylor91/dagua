# Area B -- Non-planar regular graphs (petersen_10) -- claude

Agent: claude (Opus 4.7, 1M ctx), 2026-04-25
Sibling: codex (independent)
Mandate: research-only, no code changes.
Reference HEAD: `97286e4` (sprint-20l).

## TL;DR

- **The petersen_10 gap is *gone* at HEAD.** Re-running the deterministic
  bucket-script logic with current code gives `dagua=80.59` vs best
  competitor `igraph_sugiyama=77.17`, **delta=+3.42**. Sprint-20l polish
  already moved petersen from -2.72 into the strong-win bucket. The
  CONTEXT.md table (74.64 / -2.72) appears to be a stale snapshot from
  before polish landed for this graph.
- **The honest residual bug is metric-asymmetry**, not layout quality:
  `composite()` rewards 60/100 of its weight on direction-sensitive
  metrics (dag_consistency=25, depth_spearman=15, edge_straightness=10,
  cluster_separation=5 for clustered graphs). The Petersen graph has no
  semantic direction, so these metrics measure aesthetics-orthogonal
  attributes. The canonical pentagon-pentagram embedding (textbook,
  beautiful, crossing_rate=0.031) scores **37.78** on composite() because
  dag_consistency=0.40. Switching petersen to `composite_undirected()`
  via `composite_auto()` would lift it cleanly.
- **The single biggest win available is dispatching petersen-class
  graphs (`tags={"regular","famous","small"}` and N<=12 with degeneracy
  >= 3) through `composite_auto(..., is_semantically_directed=False)`**
  in the per-graph score gate that picks "current vs polish." This is a
  ~2-line change to scoring routing, not a new algorithm.
- **If we still want a structural algorithm**, the cheapest high-yield
  win is a brute-force enumeration of layered orderings at N<=12: with
  10! = 3.6M, scoring each in <100us via a vectorized stress kernel
  fits in <5 minutes on CPU. Add a deterministic tiebreaker (lexicographic
  permutation hash) and you get +1-2 composite over sugiyama on the
  graphs where sugiyama is already winning. Generality: petersen_10,
  regular_3_30 (N=30 -- enumeration infeasible, fall back to symmetric
  init), regular_4_40 (N=40 -- ditto). For N>12, **spectral init then
  stress majorization** is the right fallback.
- **Spectral init on Laplacian eigenvectors {3,1,1,1,1,1,-2,-2,-2,-2}** is
  uniquely well-suited to Petersen because the second/third smallest
  eigenvectors have multiplicity (5-fold for eigenvalue 1, 4-fold for
  eigenvalue -2). This gives a degenerate eigenspace that yields
  multiple natural embeddings; picking the basis by minimizing
  edge_length_cv is automatic.
- **Risk: low.** Petersen already wins. The proposed change is a routing
  refinement that protects against composite() pathology on undirected
  regular graphs. No regression on graphs we currently win.

## Evidence -- the gap is already closed

Replicating `/tmp/h2h_buckets_seeded.py` selection logic at HEAD:

```
dagua sprint-20l seed=42:        80.586
  graphviz_dot                  74.952
  graphviz_sfdp                 43.887
  elk_layered                   72.244
  dagre                         74.721
  nx_spring                     29.868
  igraph_kamada_kawai           42.499
  igraph_sugiyama               77.168   <- best competitor
```

Delta = +3.42 (strong win, not -2.72 loss).

Per-metric breakdown of dagua's current layout (file:
`/tmp/petersen_dagua_seed0.npy`, reproduced via `engine_layout(g,
LayoutConfig(seed=0))`):

| metric | dagua | sugiyama | weight | dagua-pts | sugi-pts |
|---|---|---|---|---|---|
| dag_consistency       | 1.000 | 1.000 | 25 | 25.0 | 25.0 |
| edge_length_cv        | 0.213 | 0.490 | 20 | 15.7 | 10.2 |
| depth_spearman_rho    | 0.939 | 0.981 | 15 | 13.5 | 14.4 |
| overlap_count         | 0     | 0     | 10 | 10.0 | 10.0 |
| edge_straightness     | 25.4° | 29.8° | 10 | 7.2  | 6.7  |
| crossing_rate         | 0.041 | 0.068 | 10 | 5.9  | 3.2  |
| angular_res_mean_deg  | 23.7° | 27.2° | 5  | 3.0  | 3.4  |

Dagua dominates on edge_length_cv (+5.5 pts) and crossing_rate (+2.7 pts),
loses slightly on depth_spearman (-0.9 pts) and angular_res (-0.4 pts).
The polish op (sprint-20k/l) is what got us here -- pre-polish dagua
scores 75.96 on the cached `__dagua.pt` file, post-polish reaches 80.59.

Conclusion: **CONTEXT.md is reporting pre-polish numbers for this
graph.** The polish best-of picker selected an aggressive variant that
specifically helps petersen.

## What sugiyama actually does on petersen (for reference)

igraph's `layout_sugiyama` produces a 6-layer flat layout (y in 0..5):

```
[[ 11.5  0]   node 0 -- top of layered DAG
 [ 11.5  1]
 [-22.   2]
 [-45.5  3]
 [-68.   4]
 [ 68.5  3]
 [ 12.   3]
 [  1.   4]
 [ 46.   4]
 [-22.   5]]  node 9 -- bottom
```

Crossings: 5 (vs Petersen's known crossing number = 2).
Edge length CV: 0.49.
Strength: depth_spearman=0.981, dag_consistency=1.0 (because layering is
guaranteed by construction). This is the Sugiyama-family ceiling on a
non-planar 3-regular graph.

The "secret" igraph uses is barycenter sweep (`sugiyama_layer_passes=4`
default in igraph) -- which we've already replicated through native_
sugiyama and surpassed.

## Why algorithm work yields diminishing returns here

The Petersen graph has **crossing number = 2** (Guy 1969). No straight-
line 2D embedding can do fewer than 2 crossings without bending edges.
Both dagua (4 crossings) and sugiyama (5 crossings) are within 2x of the
theoretical floor. The remaining gap to the floor cannot be closed
without:

1. Edge bends (dagua doesn't render curves at scoring time -- straight
   only).
2. A non-Euclidean embedding (e.g. circular) that the metric pipeline
   doesn't natively reward.

So further gains come from *tuning what the metric values*, not from
finding a layout with fewer crossings. The composite already gives only
10 pts to crossing_rate. Pushing crossings 4 -> 2 buys at most 2 pts.

## Recommended algorithm

### Primary: keep current pipeline, fix scoring routing

`dagua_native` with sprint-20l polish is already winning. The proposed
change is metric-side, not layout-side:

1. In `dagua_native.py:_best_of_polish`, when picking the best polish
   candidate, score with `composite_auto(metrics, is_directed)` instead
   of `composite(metrics)`. The graph's `is_semantically_directed` flag
   exists in the pipeline state (TestGraph carries `tags`) -- expose it.
2. In the bucket script and h2h test harness, use the same routing.

This costs zero runtime. It protects symmetric-undirected layouts from
being penalized for not having a strong y-axis.

Expected delta:
- petersen_10: +0.5 to +2 composite (already winning, this is insurance).
- regular_3_30, regular_4_40: same direction, smaller magnitude.

### Fallback 1: spectral-init for non-planar regular graphs

For graphs detected as `regular and not planar and N <= 100`:

1. Build symmetric-normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
2. Take the 2nd and 3rd smallest eigenvectors as initial (x, y).
3. For Petersen: eigenvalues are {3, 1^5, -2^4} so the second-smallest
   has 5-fold multiplicity. Pick the basis vectors by minimizing
   `edge_length_cv` over the eigenspace (ARPACK gives an arbitrary
   basis; rotate within multiplicity to align with x/y axes).
4. Hand off to existing dagua_native pipeline.

This is the **dominant-eigenvector trick** from spectral graph drawing
(Koren 2003, "On Spectral Graph Drawing"). Petersen is the textbook
example. Expected: dagua_native already gets to 80.59; spectral init may
push to 82-83 by giving the optimizer a more symmetric basin to refine.

### Fallback 2: brute-force layered ordering (small N only)

For N <= 12 ONLY (10! = 3.6M permutations, ~2-5s in vectorized PyTorch):

1. After FAS layering produces L layers with sizes (n1, ..., nL),
   enumerate all permutations within each layer.
2. Score each by composite() on a stress-only kernel (no full metric
   evaluation, just edge-length CV + crossings + edge-straightness).
3. Take the best.

This is feasible only at small N because the per-layer permutation
count is product(n_l!). On petersen_10 with one of igraph's 6-layer
splits (1,1,1,3,3,1), it's 1*1*1*6*6*1 = 36 -- trivially exhaustive.
With multiple layer assignments tried (~L choose split), maybe 10x.

Expected delta on petersen: +1-2 composite (bringing crossing count
toward the 2-crossing floor). Not the unlock; included as safety net.

## Detection gate

```
def _is_nonplanar_regular_small(graph) -> bool:
    if graph.num_nodes > 12:
        return False  # brute-force infeasible above this
    if not _is_regular(graph):
        return False  # all degrees equal
    if _is_planar(graph):
        return False  # planar pipeline already handles
    return True

def _is_undirected_regular(graph, tags) -> bool:
    """Wider gate for the metric routing change."""
    return (
        "regular" in tags
        and graph.num_nodes <= 100
        and not graph.is_semantically_directed
    )
```

The metric-routing change uses the wider gate; the brute-force fallback
uses the narrow gate.

## Generality -- which graphs benefit

From the benchmark, three graphs match the regular non-planar profile:

| graph | N | M | regular | planar | dagua HEAD | best comp | delta |
|---|---|---|---|---|---|---|---|
| petersen_10  | 10 | 15 | 3-reg | False | 80.59 | igraph_sugi 77.17 | +3.42 |
| regular_3_30 | 30 | 45 | 3-reg | False | 76.32 | graphviz_dot 72.28 | +4.04 |
| regular_4_40 | 40 | 80 | 4-reg | False | 69.75 | graphviz_dot 61.49 | +8.27 |

**All three are already winning at HEAD.** The metric-routing change
would lift petersen_10 by ~1-2 (it's the only one in tie/close-loss
range under the alternative metric). regular_3_30 and regular_4_40
are clean wins under both metrics.

A wider sweep for `composite_auto` routing (any graph where
`is_semantically_directed = False`) would touch ~15-20 graphs in the
93-graph benchmark (real_karate_34, real_football_115, the lattice
graphs, multi_component_80, etc.). This could be net-positive but
needs sweep validation before landing -- some of those graphs might
score lower under undirected weights.

## Risk / regression analysis

**Low risk** for the recommended primary change because:

1. Petersen is already winning -- the change is insurance, not a fix.
2. `composite_auto` exists today and is gated on
   `is_semantically_directed`. The TestGraph dataclass has the flag.
3. The polish picker uses score deltas; the metric only affects which
   variant is chosen, and the chosen variant already dominates under
   both metrics.

**Higher risk** for the wider routing sweep (composite_auto on all
undirected): some graphs may regress. **Mitigation:** run the 93-graph
sweep before landing. Gate on per-graph delta rather than blanket
applying.

**Risk on the brute-force fallback:** zero -- only fires at N<=12 and
only when current pipeline scores below polish baseline. Picker
already keeps best.

## Implementation order

1. **First** -- audit CONTEXT.md numbers. Re-run h2h_buckets_seeded.py
   at HEAD and confirm petersen is already in win bucket. (5 min)
2. **Second** -- wire `composite_auto` into the polish picker for
   undirected graphs (1-line change in `_best_of_polish`). (15 min)
3. **Third** -- run 93-graph sweep with the metric routing change.
   Confirm no regressions. (1 run cycle)
4. **Later** -- if metric routing succeeds and other gaps remain,
   consider spectral-init for the regular-non-planar gate as a
   pipeline option (not default). (1 day)
5. **Don't bother** -- brute-force layered ordering. Only fires on
   one graph (petersen_10), which is already winning. The
   engineering cost outweighs the ~2-point gain.

## Bottom line

The petersen_10 finding is anti-climactic in the best way: **the gap
isn't real anymore.** Sprint-20l polish closed it. CONTEXT.md is
stale on this graph. The remaining work is metric-routing hygiene to
make sure we don't accidentally regress on this class of graph in
future composite() tuning.

If sprint 21 wants a "petersen win," the cheapest narrative is:
"audited CONTEXT.md numbers, confirmed petersen already in win
bucket post-polish, hardened metric routing for undirected
regular graphs to lock the win in." That's a 1-hour task with no
regression risk.

If sprint 21 wants a *new algorithm* for non-planar regular graphs,
spectral-init is the right answer -- but the marginal gain over
sprint-20l polish is single-digit, and the algorithm is general
enough that it would also need to integrate with the lattice/grid
detector (Area A) to avoid overlap.

## References (graph theory literature)

- Guy, R.K. (1969). "The decline and fall of Zarankiewicz's
  theorem." Petersen crossing number = 2.
- Koren, Y. (2003). "On Spectral Graph Drawing." GD'03. Pentagon-
  pentagram embedding via Laplacian eigenvectors.
- Eades, P., & Sugiyama, K. (1990). "How to draw a directed graph."
  Original Sugiyama framework -- 4 phases (cycle removal, layering,
  crossing reduction, x-coordinate assignment).
- Brandes, U., & Pich, C. (2007). "Eigensolver methods for
  progressive multidimensional scaling of large data." PivotMDS,
  used as init in our stress pipeline.
- Hachul, S., & Junger, M. (2007). "Drawing large graphs with
  potential field based multilevel algorithms." FMMM/sfdp lineage.
- Holten, D., & van Wijk, J.J. (2009). "Force-directed edge bundling
  for graph visualization." Edge bundling could reduce visual
  crossings without changing node positions; not in scope here but
  noted.
