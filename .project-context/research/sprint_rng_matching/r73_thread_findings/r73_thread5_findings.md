# r73 Thread 5 -- UMAP (26 divergent, Mode-A) Findings

## Summary

All 26 divergent umap combos split into two root-cause mechanisms:

- **Mechanism A (24 combos): Parallel-edge distance-matrix mismatch** -- FIXABLE
- **Mechanism B (2 combos): Multi-component spectral-init FP drift** -- FLOOR (narrow)

---

## 1. Sub-Bucket Breakdown

| Mechanism | Combos | Root Cause | Fixable? |
|-----------|--------|------------|----------|
| A: Parallel-edge CSR weight accumulation | 24 | dagua deduplicates parallel edges; reference CSR sums them, inflating shortest-path distances | YES |
| B: Multi-component spectral init FP drift | 2 | random_dag_50::nn5, random_dag_200::nn5 -- identical distance matrices but tiny init divergence (Procrustes 0.023) from multi-component meta-embedding differences | FLOOR (narrow) |

Graphs with parallel edges (mechanism A):
- `parallel_multiedge_bundle`: edges `(0,1)` x3, `(1,2)` x2 -> 5 of 6 variants diverge
- `kitchen_sink_hybrid_net`: edge `(4,7)` x2 (+ self-loop at 13) -> 6/6 variants diverge
- `clustered_longlabel_handoffs`: edge `(4,6)` x2 -> 5/6 variants diverge
- `nested_cluster_label_stack`: edge `(4,5)` x2 -> 6/6 variants diverge
- `dense_pair_50`: 7 parallel edge pairs -> only `mindist05` variant fails (others pass at rung 2-3)

Note: `citation_dag_300` also has 1 parallel edge `(33,54)` x2 and has max dist diff 1.0, but
ALL its umap variants PASS (rung=2 with W_D ~1.09-1.13). The energy-test
`dist_equivalent` still passes because the distance difference is small relative to the
distributional spread of those large graphs. This is consistent -- the parallel-edge effect
is proportionally larger for the smaller graphs.

---

## 2. Mechanism A: Parallel-Edge Distance-Matrix Mismatch -- FIXABLE

### Evidence

**Reference path** (`umap_competitor.py::_distance_matrix`, lines 77-91):
```python
rows = np.concatenate([edge_index[0], edge_index[1]])
cols = np.concatenate([edge_index[1], edge_index[0]])
data = np.ones(rows.shape[0], dtype=np.float32)
adjacency = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
distances = shortest_path(adjacency, directed=False)
```

When edge `(4,7)` appears twice in `edge_index`, both `(4,7)` and `(7,4)` are added twice each.
SciPy's CSR `csr_matrix` with duplicate (row,col) entries **sums the data values** by default
(standard COO->CSR conversion). So the (4,7) entry gets weight 2.0, and `shortest_path`
treats it as a path cost of 2.0 instead of 1.0.

**Empirical verification** (`kitchen_sink_hybrid_net`, edge `(4,7)` x2):
- Reference dist[4,7] = 2.0; dagua dist[4,7] = 1.0 (diff=1.0)
- All distances through node 7 shift by +1 in the reference vs dagua
- `adj_ref[4,7] = 2.0` confirmed via `adj_ref[7].toarray()`

**Dagua path** (`_build_undirected_adjacency`, umap.py lines 148-160):
```python
adjacency_sets[source].add(target)
adjacency_sets[target].add(source)
```
Python `set.add()` deduplicates: the parallel edge is silently dropped, giving weight 1.0.

**Impact of distance mismatch**: Different kNN neighborhoods -> different fuzzy simplicial sets
-> different spectral init -> different SGD trajectory. All 24 combos for the 5 affected graphs diverge.

### Fix Spec

**File**: `/home/jtaylor/projects/dagua/dagua/layout/ops/umap.py`
**Function**: `_build_undirected_adjacency` (line 123)

The fix is to **sum duplicate edge weights** (matching SciPy CSR behavior) instead of deduplicating.

For the **unweighted** path (lines 148-160): change from `set` to accumulate counts, or simply use
the count of parallel edges as the "weight" -- matching SciPy's behavior. The simplest correct
approach is to count duplicate directed edges and add their sum:

```python
# CURRENT (lines 148-160) -- deduplicates:
adjacency_sets: list[set[int]] = [set() for _ in range(num_nodes)]
...
adjacency_sets[source].add(target)
adjacency_sets[target].add(source)
return [[(neighbor, 1.0) for neighbor in sorted(neighbors)] for neighbors in adjacency_sets]

# FIX -- accumulate parallel-edge counts as weights (match SciPy CSR sum):
adjacency_maps: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
    if source == target:
        continue
    adjacency_maps[source][target] = adjacency_maps[source].get(target, 0.0) + 1.0
    adjacency_maps[target][source] = adjacency_maps[target].get(source, 0.0) + 1.0
return [sorted(neighbors.items()) for neighbors in adjacency_maps]
```

This unifies the unweighted path with the weighted path (which already uses dicts and sums, lines 162-179).

**Exact change**: Replace the `if edge_weights is None:` branch (lines 148-160) with the dict-based
accumulator pattern already used in the weighted branch (lines 162-179), using `cost=1.0` per
unweighted edge.

**Self-loop handling**: The current code correctly skips self-loops (`if source == target: continue`).
No change needed for self-loops -- `shortest_path` ignores self-loops for path computation, dagua
already matches this behavior.

**Verification**: After the fix, run:
```bash
python3 -c "
import os; os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/dagua-numba-cache')
import sys; sys.path.insert(0, '.')
import numpy as np
from dagua.eval.benchmark import _named_graphs
from dagua.eval.competitors.umap_competitor import _distance_matrix
from dagua.layout.ops.umap import _build_undirected_adjacency, _all_pairs_shortest_paths

ng = _named_graphs()
for gname in ['kitchen_sink_hybrid_net', 'clustered_longlabel_handoffs', 'parallel_multiedge_bundle', 'nested_cluster_label_stack', 'dense_pair_50']:
    tg = ng[gname]
    dg = tg.graph
    ref_dists = _distance_matrix(dg)
    adj = _build_undirected_adjacency(dg.edge_index, dg.num_nodes)
    dagua_dists = _all_pairs_shortest_paths(adj).numpy()
    diff = np.abs(ref_dists - dagua_dists).max()
    print(f'{gname}: max_diff={diff:.6f}')  # should be 0.0 after fix
"
```

Expected: max_diff=0.0 for all 5 graphs after fix.

---

## 3. Mechanism B: Multi-Component Spectral Init FP Drift -- FLOOR

### Evidence

**Graphs**: `random_dag_50` (N=97, 52 components) and `random_dag_200` (N=383, 202 components)
**Variants**: only `nn5` -- both graphs only have `nn5` and `nn30` variants in the divergent bucket,
and `nn30` is `INSUFFICIENT_DATA` (reference returns few seeded runs for nn30 at N<<nn30).

**Distance matrices are bit-identical**: Verified empirically -- `max dist diff = 0.0` for both.
kNN indices also identical at nn=5.

**Initial embedding diverges** (random_dag_50, seed=42, nn5):
- Procrustes RMSD at init: 0.023 (slight mismatch)
- Procrustes RMSD at final: 0.011 (SGD slightly reduces it -- not chaotic amplification)
- Example: node 1 -- ref y=3.340, dagua y=3.385 (diff ~0.045)

**Root cause of init mismatch**: Both implementations call multi-component spectral embedding
(52/202 components >> 2*dim=4 threshold triggers `component_layout`). The reference uses
`umap.spectral.component_layout` -> `SpectralEmbedding(affinity='precomputed')` from sklearn,
while dagua uses `_component_meta_embedding` -> `_connected_spectral_embedding` (scipy `eigsh`).
These are **different solvers** for the meta-embedding -- sklearn's SpectralEmbedding vs scipy's
`eigsh`. They can produce eigenvectors with different signs and orderings, leading to small
coordinate differences.

Concretely, sklearn's `SpectralEmbedding` uses `arpack` internally but applies different
normalization and output conventions than dagua's raw `eigsh` call.

**Why this is FLOOR (not FIXABLE for r73)**:

1. The residual RMSD (0.011-0.023) is already small. The W_D at 0.17-0.19 fails the dist_equivalent
   test, but only barely -- these are highly dispersed 500-epoch layouts where small init differences
   matter.

2. To exactly match: dagua would need to replicate sklearn's `SpectralEmbedding` including its
   internal normalization, random initialization, and ARPACK settings -- a different solver stack
   that would be brittle across sklearn versions. This is NOT "bit-level libm emulation," but it
   IS solver-implementation replication that provides no user-visible quality benefit.

3. These are large disconnected graphs where the meta-embedding is a secondary aesthetic choice.
   Quality metrics (stress, crossings, kNN-neighborhood) are likely comparable.

4. **Better path**: Check if `n_neighbors=5` on a 97-node graph with 52 components is even a valid
   configuration -- the reference's `n_neighbors = min(5, N-1) = 5`, but with 52 components many
   nodes have only 1-2 graph neighbors. This may be an INVALID-COMPARISON case where the parameter
   config produces a degenerate input.

**Recommended action**: Classify these 2 combos as QUALITY-IDENTICAL (run 3Q battery to verify)
rather than attempting solver-level replication.

---

## 4. Expected Impact

| Fix | Combos resolved | New rung |
|-----|-----------------|----------|
| Mechanism A: parallel-edge weight accumulation | 22-24 | 1 (bit-exact or near) to 2 (dist-equiv) |
| Mechanism B: none / 3Q reclassification | 2 | 3Q if quality passes |

The 24 Mechanism A combos should become bit-exact (RNG stream already matched from r72; UMAP
graph-construction kernel verified bit-exact in r71) once the distance matrices match.

The 2 Mechanism B combos (`random_dag_50::nn5`, `random_dag_200::nn5`) should be checked with
the 3Q quality battery -- they may qualify as QUALITY-IDENTICAL since the Procrustes RMSD is
small and the SGD doesn't amplify the divergence.

---

## 5. Residual (Post-Fix)

After the parallel-edge fix, the residual is 2 combos (`random_dag_*::nn5`). Evidence this is floor:
- Different meta-embedding solver (sklearn SpectralEmbedding vs scipy eigsh)
- RMSD is small (0.011-0.023 Procrustes); no chaotic amplification -- SGD actually reduces the gap
- The 3Q battery may clear these as QUALITY-IDENTICAL

No "bit-level transcendental emulation" is needed -- the residuals here are genuinely from a
different solver choice for the multi-component meta-layout, not from FP chaos.

---

## 6. Fix Spec (Codex-Ready)

**Single-file fix**, `/home/jtaylor/projects/dagua/dagua/layout/ops/umap.py`:

**Target**: `_build_undirected_adjacency` function, line ~123.

**Change**: Replace the `if edge_weights is None:` branch (currently uses `set` which deduplicates)
with a `dict`-based accumulator that sums parallel-edge counts (matching SciPy CSR behavior).

**Before** (lines ~148-160):
```python
if edge_weights is None:
    adjacency_sets: list[set[int]] = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source == target:
            continue
        adjacency_sets[source].add(target)
        adjacency_sets[target].add(source)
    return [[(neighbor, 1.0) for neighbor in sorted(neighbors)] for neighbors in adjacency_sets]
```

**After**:
```python
if edge_weights is None:
    adjacency_maps_uw: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source == target:
            continue
        prev_st = adjacency_maps_uw[source].get(target)
        adjacency_maps_uw[source][target] = (prev_st + 1.0) if prev_st is not None else 1.0
        prev_ts = adjacency_maps_uw[target].get(source)
        adjacency_maps_uw[target][source] = (prev_ts + 1.0) if prev_ts is not None else 1.0
    return [sorted(neighbors.items()) for neighbors in adjacency_maps_uw]
```

**Tests to run after fix**:
```bash
cd /home/jtaylor/projects/dagua
python3 -c "
import os; os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/dagua-numba-cache')
import sys; sys.path.insert(0, '.')
import numpy as np
from dagua.eval.benchmark import _named_graphs
from dagua.eval.competitors.umap_competitor import _distance_matrix
from dagua.layout.ops.umap import _build_undirected_adjacency, _all_pairs_shortest_paths

ng = _named_graphs()
for gname in ['kitchen_sink_hybrid_net', 'clustered_longlabel_handoffs', 'parallel_multiedge_bundle', 'nested_cluster_label_stack', 'dense_pair_50']:
    tg = ng[gname]
    dg = tg.graph
    ref_dists = _distance_matrix(dg)
    adj = _build_undirected_adjacency(dg.edge_index, dg.num_nodes)
    dagua_dists = _all_pairs_shortest_paths(adj).numpy()
    diff = np.abs(ref_dists - dagua_dists).max()
    print(f'{gname}: max_diff={diff:.6f} (expected 0.0)')
"
pytest tests/ -x --tb=short -q
```

**Note**: Do NOT change the weighted path (already uses dict accumulation with `+= cost`).
The fix is unweighted-path only.
