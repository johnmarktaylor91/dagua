# Area C Bug Hunt -- Empirical Correctness Audit

## TL;DR

- I verified 11 concrete correctness bugs with runnable reproducers. The highest-impact ones are in `dagua/layout/cycle.py` and `dagua/metrics.py`.
- Two metric bugs materially distort benchmark scoring: `segments_intersect()` misses valid crossings when the determinant is negative, and `count_overlaps_detailed()` misses overlaps across adjacent hash cells for `N > 2000`.
- Large-graph initialization still has two latent issues: the vectorized x-centering is off by half a slot, and the spectral initializer is non-deterministic despite the module claiming deterministic topology-based init.

## Findings

### 1. `make_acyclic_robust()` can return a still-cyclic graph on self-loops

- Severity: `high`
- File: `dagua/layout/cycle.py:130-193`, `dagua/layout/cycle.py:196-221`

**Reproducer**

```python
import torch
from dagua.layout.cycle import make_acyclic_robust, _is_acyclic
ei = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
acyclic, mask = make_acyclic_robust(ei, 2)
print(acyclic.tolist(), mask.tolist(), _is_acyclic(acyclic, 2))
```

**Actual**

- `acyclic == [[0, 0, 0], [0, 1, 1]]`
- `mask == [False, False, False]`
- `_is_acyclic(acyclic, 2) == False`

**Expected**

- A function named `make_acyclic_robust()` should never return a cyclic result.
- At minimum, self-loops must be filtered or separately marked before the greedy fallback runs.

**Why it happens**

- `detect_back_edges()` correctly marks self-loops as back edges, but `make_acyclic()` cannot repair a self-loop because swapping `(u, u)` is a no-op.
- `_greedy_fas()` then computes a fresh reversal mask on the already-modified edge list. For self-loops, `order_index[source] > order_index[target]` is always `False`, so the returned mask forgets the earlier reversal intent and the final graph remains cyclic.

**Proposed fix**

- Strip self-loops before both DFS and greedy FAS passes when the caller needs a DAG for layering/ordering.
- If the API must preserve input edge cardinality, return a separate `self_loop_mask` or explicitly OR the final reversal mask with the original `back_edge_mask` for non-removable edges.
- Add a postcondition check so `make_acyclic_robust()` cannot silently return a cyclic graph.

**Blast radius**

- Any cyclic-preprocessing path that relies on this helper for DAG constraints.
- Graphs with self-loops can get incorrect `back_edge_mask` metadata, wrong DAG metrics, and broken cycle handling in downstream layout stages.

### 2. Large-graph vectorized initialization is shifted left by half a slot

- Severity: `medium`
- File: `dagua/layout/init_placement.py:294-303`

**Reproducer**

```python
import torch
from dagua.layout.init_placement import init_positions
for n in (100, 101):
    pos = init_positions(torch.empty((2, 0), dtype=torch.long), n, torch.ones((n, 2)))
    print(n, round(pos[:, 0].mean().item(), 4))
```

**Actual**

- `100 -> 0.0`
- `101 -> -13.0` with default spacing (`-5.5` when `node_sep=10`)

**Expected**

- Both paths should center the layer at `x == 0`.

**Why it happens**

- The vectorized path uses `x = (order - layer_width / 2) * spacing`.
- For centered ordinal coordinates, the correct midpoint is `(layer_width - 1) / 2`.
- Current code shifts every large-layer result by `-0.5 * spacing`.

**Proposed fix**

- Change the centering term to `(node_layer_width - 1.0) / 2.0`.
- Add a regression test that compares `init_positions(..., n=100)` and `n=101` centroids for an edgeless graph.

**Blast radius**

- All `num_nodes > 100` calls that take the vectorized initializer.
- Most visible on wide, lightly constrained, or sparse large graphs where initialization bias survives into the solve.

### 3. `_spectral_order()` is non-deterministic despite deterministic-init contract

- Severity: `medium`
- File: `dagua/layout/init_placement.py:371-380`

**Reproducer**

```python
import torch
from dagua.layout.init_placement import _spectral_order
ei = torch.tensor([[i for i in range(19)], [i + 1 for i in range(19)]], dtype=torch.long)
a = _spectral_order(ei, 20, "cpu"); b = _spectral_order(ei, 20, "cpu")
print(torch.allclose(a, b), float((a - b).abs().max()))
```

**Actual**

- `False 0.0007777512073516846`

**Expected**

- Same topology should yield the same ordering basis unless the caller explicitly asked for randomness.

**Why it happens**

- `_spectral_order()` seeds `lobpcg` with `torch.randn(N, 2)` every call.
- `init_placement.py` explicitly documents deterministic topology-based initialization, so this violates module semantics on the large-graph spectral path.

**Proposed fix**

- Use a deterministic initial subspace, matching the newer pattern already present in `dagua/layout/ops/ordering.py`.
- If randomness is desired, thread an explicit seed instead of using global RNG state implicitly.

**Blast radius**

- Large sparse graphs on the spectral-init path (`N > 10_000` in `init_positions()`).
- Benchmark reproducibility, layout caching, and regression pinning for large DAGs.

### 4. `segments_intersect()` has a sign bug for negative determinants

- Severity: `high`
- File: `dagua/metrics.py:146-161`

**Reproducer**

```python
import torch
from dagua.metrics import segments_intersect
p1 = torch.tensor([[1., 0.]]); p2 = torch.tensor([[0., 1.]])
p3 = torch.tensor([[0., 0.]]); p4 = torch.tensor([[1., 1.]])
print(segments_intersect(p1, p2, p3, p4).item())
```

**Actual**

- `False`

**Expected**

- `True` because the two segments cross at `(0.5, 0.5)`.

**Why it happens**

- Both `t` and `u` divide by `cross.clamp(min=1e-10)`.
- When `cross` is negative, the clamp flips the denominator sign instead of preserving it, so valid intersections on one orientation branch become false negatives.

**Proposed fix**

- Preserve the sign of `cross`, e.g. divide by `torch.where(parallel, torch.ones_like(cross), cross)`.
- Keep the `parallel` mask for the near-zero case instead of clamping all negative values upward.

**Blast radius**

- `sampled_crossing_rate()`
- Any composite score or benchmark comparison that uses crossing metrics
- Any future edge-crossing tooling built on this helper

### 5. `count_overlaps_detailed()` misses overlaps that span adjacent hash cells

- Severity: `high`
- File: `dagua/metrics.py:387-435`

**Reproducer**

```python
import torch
from dagua.metrics import count_overlaps_detailed
pos = torch.arange(2001, dtype=torch.float32).unsqueeze(1).repeat(1, 2) * 100
pos[0, 0], pos[1, 0] = 10.9, 11.1
print(count_overlaps_detailed(pos, torch.ones((2001, 2)) * 10))
```

**Actual**

- `{'overlap_count': 0}`

**Expected**

- `{'overlap_count': 1}` because the two boxes overlap heavily on x and y.

**Why it happens**

- For `N > 2000`, the function only checks node pairs inside the exact same spatial-hash cell.
- With `cell_size = max_size + 1`, two overlapping boxes can easily land in neighboring cells near a boundary.

**Proposed fix**

- Check the 3x3 neighborhood of adjacent cells, not just same-cell pairs.
- Keep the per-cell cap if needed, but move it to the neighborhood candidate set rather than dropping cross-cell overlaps entirely.

**Blast radius**

- Large-graph overlap auditing
- Binary `overlap_count == 0` bonus inside `composite()`
- Any benchmark decision that treats overlap-free layouts as a hard pass/fail

### 6. `sampled_crossing_rate()` overestimates `crossing_estimated_total`

- Severity: `medium`
- File: `dagua/metrics.py:625-663`

**Reproducer**

```python
import torch
from dagua.metrics import sampled_crossing_rate
pos = torch.tensor([[0., 0.], [1., 1.], [1., 0.], [0., 1.]])
ei = torch.tensor([[0, 2, 0], [1, 3, 2]], dtype=torch.long)
print(sampled_crossing_rate(pos, ei, n_samples=10000, seed=0))
```

**Actual**

- `{'crossing_rate': 1.0, ..., 'crossing_estimated_total': 3, 'crossing_n_samples': 1}`

**Expected**

- The graph has exactly one valid non-incident edge pair, so the estimated total should be `1`, not `3`.

**Why it happens**

- The rate is computed only over valid non-incident sampled pairs.
- `crossing_estimated_total` is then scaled by `E * (E - 1) / 2`, which includes invalid same-node pairs that were excluded from the estimator.

**Proposed fix**

- Scale by the number of valid unordered non-incident pairs, not all unordered pairs.
- If computing that denominator exactly is too expensive at scale, rename the field to make the approximation explicit.

**Blast radius**

- Dashboards and reports that consume `crossing_estimated_total`
- Any tuning logic that ranks layouts by estimated absolute crossing count rather than raw rate

### 7. `depth_position_correlation()` returns NaNs on constant inputs

- Severity: `medium`
- File: `dagua/metrics.py:325-338`

**Reproducer**

```python
import torch
from dagua.metrics import depth_position_correlation
pos = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
depth = torch.tensor([0, 0, 0])
print(depth_position_correlation(pos, depth))
```

**Actual**

- `{'depth_spearman_rho': nan, 'depth_spearman_pval': nan}`

**Expected**

- A stable numeric result for degenerate cases, typically `0.0` or `1.0` by policy.

**Why it happens**

- `scipy.stats.spearmanr()` returns NaN when either input is constant.
- The function forwards NaNs directly into the metric surface.

**Proposed fix**

- Detect constant-input cases before calling `spearmanr()`.
- Return a documented fallback, e.g. `1.0` when both rankings are identical constants, else `0.0`.

**Blast radius**

- Small trivial graphs
- Flat or cycle-collapsed layerings
- Composite scoring and CSV/report pipelines that assume finite floats

### 8. `composite()` can exceed its documented 0-100 range

- Severity: `medium`
- File: `dagua/metrics.py:1147-1206`

**Reproducer**

```python
from dagua.metrics import composite
metrics = {'dag_consistency':1.0,'edge_length_cv':0.0,'depth_spearman_rho':1.0,
           'overlap_count':0,'edge_straightness_mean_deg':0.0,'crossing_rate':0.0,
           'angular_res_mean_deg':90.0,'cluster_mean_sep_ratio':10.0,
           'edge_node_crossing_rate':0.0,'label_overlaps':0,'label_node_overlaps':0}
print(composite(metrics))
```

**Actual**

- `105.0`

**Expected**

- The docstring says `Range 0-100`.

**Why it happens**

- `composite()` adds optional edge-aware bonuses (+3 and +2) on top of a base scale that is already documented as 100.

**Proposed fix**

- Either clamp `composite()` to 100 or renormalize weights so optional metrics fit within the declared budget.

**Blast radius**

- Benchmark dashboards, tuning sweeps, and regression comparisons
- Any caller that assumes `composite_score` is on a stable 0-100 scale

### 9. `full()` is non-deterministic by default and gives no way to seed sampled metrics

- Severity: `medium`
- File: `dagua/metrics.py:1425-1438`

**Reproducer**

```python
import torch
from dagua.metrics import full
pos = torch.cat([torch.stack([torch.zeros(20), torch.arange(20.)], 1),
                 torch.stack([torch.ones(20), torch.arange(19, -1, -1.)], 1)])
ei = torch.tensor([(i, 20 + j) for i in range(20) for j in range(20)], dtype=torch.long).T
print(full(pos, ei, crossing_samples=2000)['crossing_rate'], full(pos, ei, crossing_samples=2000)['crossing_rate'])
```

**Actual**

- Example run: `0.24008934199810028 0.24777282774448395`

**Expected**

- Research and benchmark helpers should be reproducible on identical inputs unless randomness is explicitly requested.

**Why it happens**

- `full()` calls `sampled_crossing_rate()`, `neighborhood_preservation()`, and `angular_resolution()` without a caller-visible seed.
- `sampled_crossing_rate()` already supports a seed, but `full()` drops that control surface entirely.

**Proposed fix**

- Add `seed: Optional[int]` to `full()` and thread it into every stochastic helper.
- Prefer deterministic sampling defaults for benchmark code paths.

**Blast radius**

- Benchmark reproducibility
- CI/regression pinning for metrics
- Any tuner comparing close variants on sampled metric deltas

### 10. `compare()` swallows shape and input errors instead of surfacing them

- Severity: `low`
- File: `dagua/metrics.py:1494-1498`

**Reproducer**

```python
import torch
from dagua.metrics import compare
print(compare(torch.zeros((2, 2)), torch.zeros((3, 2))))
```

**Actual**

- `{'procrustes_disparity': 1.0}`

**Expected**

- A shape mismatch should raise a clear error; returning a valid-looking scalar hides the real problem.

**Why it happens**

- The function catches `Exception` broadly around `scipy.spatial.procrustes()` and substitutes a hard-coded worst score.
- That makes caller bugs indistinguishable from a genuine, valid high-disparity comparison.

**Proposed fix**

- Only catch known numerical failure modes if there are any.
- Let structural input errors propagate with their original exception.

**Blast radius**

- Evaluation scripts comparing layouts from differently sized node sets
- Debugging of report-generation code that accidentally misaligns inputs

### 11. `layer_uniformity()` reports one layer for an empty graph

- Severity: `low`
- File: `dagua/metrics.py:864-872`

**Reproducer**

```python
import torch
from dagua.metrics import layer_uniformity
print(layer_uniformity(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long)))
```

**Actual**

- `{'layer_spacing_cv': 0.0, ..., 'n_layers': 1}`

**Expected**

- An empty graph has `0` layers, not `1`.

**Why it happens**

- The early return for `unique_depths.numel() < 2` hard-codes `n_layers: 1` for both singleton-layer and empty inputs.

**Proposed fix**

- Return `int(unique_depths.numel())` instead of the constant `1` in that branch.

**Blast radius**

- Empty-graph metric summaries
- Any dashboard or schema validator that assumes `n_layers` is structurally accurate

## Runtime / Review Notes

- I read the required context and core files first, then spot-checked the requested ops families: barycenter ordering, crossing reduction, coordinate assignment, normalization/postprocess, loss wrappers, ordering, and Sugiyama helpers.
- Reviewed ops included `barycenter.py`, `crossing_swap.py`, `coordinate.py`, `loss_engine.py`, `postprocess.py`, `ordering.py`, `sugiyama.py`, `preprocess.py`, `project.py`, and the native pipeline wiring. I did not find additional empirically provable sign/comparison inversions there beyond the initialization issues above.
- I deliberately did **not** report speculative items that I could not force with a minimal reproducer.
- I also sanity-checked the recent `_greedy_fas` fix context against nearby cycle code paths. The main adjacent failure mode was not another sign inversion inside `_greedy_fas` itself after the fix, but the self-loop / stale-mask interaction in `make_acyclic_robust()` that bypasses the intended DAG guarantee.
- No dead-code removals are proposed here; this was a read-only audit and I did not identify newly unreachable production code within the reviewed scope.

## Recommended Action Queue

1. Fix `segments_intersect()` and `count_overlaps_detailed()` first. Those directly corrupt benchmark metrics and can invert optimization conclusions.
2. Fix `make_acyclic_robust()` self-loop handling next. It violates its own contract and is the closest analogue to the recently discovered `_greedy_fas` sign bug.
3. Fix the vectorized init centering formula and spectral-order determinism together. Both affect large-graph quality and reproducibility.
4. Repair metric-surface stability next: `depth_position_correlation()`, `full(seed=...)`, and `composite()` range enforcement.
5. Treat `crossing_estimated_total` and `compare()` error swallowing as reporting-layer bugs unless downstream automation already consumes them directly.
