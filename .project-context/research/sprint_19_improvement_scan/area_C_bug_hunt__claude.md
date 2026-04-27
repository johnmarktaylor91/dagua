# Area C -- Bug Hunt (Claude)

Audit focus: find latent bugs of the same class as the `_greedy_fas` score
inversion that triggered this investigation. For each finding, I ran a
reproducer against the live checkout at
`/home/jtaylor/projects/dagua`.

The `_greedy_fas` fix at `dagua/layout/cycle.py:173` verifies correctly
(reproducer in Section 4) -- the `(out-in, -idx)` ordering now puts
source-like nodes first and a 0->1->2 linear chain comes back with
`reversed_mask = [False, False]`. Use that as a sanity-check baseline for
the rest of this report.

---

## 1. TL;DR -- Top 5 Most Critical

1. **HIGH -- `edge_direction_straightness` treats zero-length edges
   asymmetrically for TB vs LR.** In TB a coincident edge reports 0
   degrees (perfectly straight); in LR the same edge reports 45 degrees
   (maximum deviation). Every LR graph with a zero-length or very-short
   edge loses up to 10/100 composite points vs the TB counterpart.
   `dagua/metrics.py:480-487`.

2. **HIGH -- `dag_consistency` uses strict `>` so any edge with
   y_src == y_tgt (TB) / x_src == x_tgt (LR) counts as a violation.**
   Self-loops, zero-length edges, or nodes that happen to land on the
   same y-coordinate from barycenter ties all get flagged. Worth 25/100
   points. `dagua/metrics.py:291,295,299,303`.

3. **HIGH -- `angular_resolution` is non-deterministic even with a
   global `torch.manual_seed`.** It uses `torch.randperm` on the global
   generator and accepts no `seed` kwarg, so the metric that contributes
   5/100 to the composite score varies between runs on the same
   positions. This introduces variance into every iteration loop and
   hides small improvements in the noise floor. `dagua/metrics.py:757`.

4. **MEDIUM -- `composite()` relies on CPython's `max()` short-circuit
   to silently turn NaN-valued metrics into 0.** `max(0.0, nan)` returns
   0.0 today, but the semantics are implementation-defined. If a future
   numpy/torch update changes the comparison order, composite scores
   flip to NaN and the entire benchmark pipeline breaks.
   `dagua/metrics.py:1166,1169,1176`.

5. **MEDIUM -- `CrossingSwapPolish` uses a too-aggressive edge filter
   when measuring before/after crossings.** Edges that do not touch the
   swapped pair are dropped, so crossings between swapped-edge and
   unrelated-edge pairs are invisible to the swap decision. The op is
   opt-in (`enabled=False`) but is shipped as infrastructure and will
   mis-score swaps if anyone turns it on. `dagua/layout/ops/crossing_swap.py:176-194`.

---

## 2. Verified Bugs

### 2.1 HIGH -- `edge_direction_straightness` asymmetric degenerate handling

- File: `dagua/metrics.py:465-492`
- Class: sign/asymmetry plus degenerate-case handling.

The function clamps `dy` to `1e-6` unconditionally, but clamps `dx` to
`1e-6` only in LR mode:

```python
src, tgt = edge_index[0], edge_index[1]
dx = (pos[tgt, 0] - pos[src, 0]).abs()
dy = (pos[tgt, 1] - pos[src, 1]).abs().clamp(min=1e-6)

if direction in ("LR", "RL"):
    dx = dx.clamp(min=1e-6)
    angles = torch.atan2(dy, dx) * 180 / torch.pi
else:
    angles = torch.atan2(dx, dy) * 180 / torch.pi
```

**Reproducer:**

```python
# CUDA_VISIBLE_DEVICES="" python -c "..."
import torch
from dagua.metrics import edge_direction_straightness
pos = torch.tensor([[0., 0.], [0., 0.]])
ei = torch.tensor([[0], [1]], dtype=torch.long)
print(edge_direction_straightness(pos, ei, direction='TB'))
# {'edge_straightness_mean_deg': 0.0, 'edge_straightness_below_15': 1.0}
print(edge_direction_straightness(pos, ei, direction='LR'))
# {'edge_straightness_mean_deg': 45.0, 'edge_straightness_below_15': 0.0}
```

**Expected:** Both directions report the same value for a degenerate
(zero-length) edge. Either both 0 (treat it as neutral) or both some
NaN-sentinel that is then excluded from the mean.

**Actual:** TB reports the edge as perfectly straight (0 deg). LR
reports it as 45 deg, the worst possible non-axis-aligned value.

**Blast radius:** Composite `edge_straightness` term is worth 10/100.
Any LR graph with at least one near-zero edge loses up to that 10. TB
graphs are artificially flattered. Also skews head-to-head comparisons
whenever a competitor produces a tighter layout with small edges.

**Proposed fix:** Clamp both axes symmetrically before `atan2`:

```python
dx = (pos[tgt, 0] - pos[src, 0]).abs().clamp(min=1e-6)
dy = (pos[tgt, 1] - pos[src, 1]).abs().clamp(min=1e-6)
if direction in ("LR", "RL"):
    angles = torch.atan2(dy, dx) * 180 / torch.pi
else:
    angles = torch.atan2(dx, dy) * 180 / torch.pi
```

Or gate zero-length edges out of the aggregate entirely.

---

### 2.2 HIGH -- `dag_consistency` strict inequality punishes ties

- File: `dagua/metrics.py:291,295,299,303`
- Class: sign/inequality error at degenerate boundary.

All four direction branches use `y_tgt > y_src` (or the mirror). When
y_tgt == y_src, the edge is neither going forward nor backward but it
counts as a violation. This hurts:

- Self-loops, unless excluded via `back_edge_mask`.
- Tied-barycenter nodes that land on identical y (happens often with
  small graphs).
- Graphs where the pipeline sets y=0 for everything (the single-layer
  cyclic case that `Force2DInitIfFlat` is supposed to rescue; but
  `dag_consistency` is measured on the final pos, which may still
  collapse if Force2DInit is bypassed).

**Reproducer:**

```python
import torch
from dagua.metrics import dag_consistency

# Self-loop: TB with edges 0->0 and 0->1
pos = torch.tensor([[0., 0.], [0., 10.]])
ei = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
print(dag_consistency(pos, ei, direction='TB'))
# {'dag_consistency': 0.5, 'dag_num_violations': 1, ...}  <-- self-loop counted as violation

# Same x in LR with a chain
pos = torch.tensor([[5., 0.], [5., 1.], [5., 2.]])
ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
print(dag_consistency(pos, ei, direction='LR'))
# {'dag_consistency': 0.0, ...}
```

**Expected:** Either self-loops are excluded by default, or equality
is treated as "neither violation nor satisfaction" and only truly
backward edges count as violations.

**Actual:** Both equality and backward count as violations. Up to 25/100
composite points at risk.

**Blast radius:** Every graph with self-loops, pinned layers, or
pathological ties. The benchmark likely hides this via `back_edge_mask`
for cycle-containing graphs, but self-loops are independent.

**Proposed fix:** Exclude self-loops (src == tgt) from the edge index
before computing `correct`, and consider using `>=` with a secondary
magnitude check. A simpler patch: strict `>=` with documented
"equality counts as OK":

```python
correct = y_tgt > y_src  # strict
# becomes:
self_loop = (src == tgt)
correct = (y_tgt > y_src) | self_loop  # self-loops neutral
```

Or drop self-loops from `forward_ei` up-front.

---

### 2.3 HIGH -- `angular_resolution` non-deterministic sampling

- File: `dagua/metrics.py:757`
- Class: seed non-determinism when seed is supposed to give determinism.

Other sampled metrics (`count_overlaps_detailed`, `sampled_crossing_rate`,
`sampled_stress`) accept a `seed=` kwarg and build a local
`torch.Generator`. `angular_resolution` does not -- it calls
`torch.randperm(candidates.numel())[...]` on the global generator.

**Reproducer:**

```python
import torch
from dagua.metrics import angular_resolution
N = 200
torch.manual_seed(0)
src = torch.randint(0, N, (500,))
tgt = torch.randint(0, N, (500,))
ei = torch.stack([src, tgt])
pos = torch.randn(N, 2)

torch.manual_seed(999)
r1 = angular_resolution(pos, ei, n_samples=50)
torch.manual_seed(123)
r2 = angular_resolution(pos, ei, n_samples=50)
print(r1['angular_res_mean_deg'], r2['angular_res_mean_deg'])
# 9.72... vs 13.97... on the SAME positions
```

**Expected:** Same positions -> same metric value, or accept a `seed=`
kwarg like sibling metrics.

**Actual:** Variance in the 4-degree range with a default n_samples,
much larger when `n_samples < candidates.numel()`.

**Blast radius:** Composite angular_resolution term is 5/100. Injects
run-to-run variance into every benchmark number. Especially corrosive
to small-sprint iterations (`sprint-18k` +0.03 composite improvements
can be swamped by this noise).

**Proposed fix:** Add `seed: Optional[int] = None` and thread a local
generator through the `randperm`:

```python
def angular_resolution(..., *, seed: Optional[int] = None):
    ...
    gen = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(candidates.numel(), generator=gen)
    sample = candidates[perm[: min(n_samples, candidates.numel())]]
```

And plumb `seed` through `quick()` / `full()` the same way it is for
`count_overlaps_detailed`.

---

### 2.4 MEDIUM -- `composite()` NaN handling is CPython-implementation-specific

- File: `dagua/metrics.py:1166-1206`
- Class: numerical instability / implicit assumption.

`composite()` propagates `depth_spearman_rho` via `max(0.0, rho)`. When
rho is NaN (constant topo_depth, which happens on cyclic graphs that
end up with all-in-one-layer), this expression depends on CPython's
`max` short-circuit:

```python
>>> max(0.0, float('nan'))
0.0          # first arg wins because nan is not-less-than anything
>>> max(float('nan'), 0.0)
nan          # first arg wins, also not-less-than anything
```

The code relies on the "happens to work" branch.

**Reproducer:**

```python
from dagua.metrics import composite
m = dict(
    dag_consistency=1.0,
    edge_length_cv=0.3,
    depth_spearman_rho=float('nan'),
    overlap_count=0,
    edge_straightness_mean_deg=10.0,
    crossing_rate=0.0,
    angular_res_mean_deg=30.0,
)
print(composite(m))  # 73.03 -- composes with depth term silently zero'd

m['edge_length_cv'] = float('nan')
print(composite(m))  # 72.53 -- cv term silently zero'd (but could flip to NaN on other platforms)
```

**Expected:** Explicit NaN sanitization. Either
`rho = 0.0 if not isfinite else rho` up-front, or a single
`_safe_score(metric, key, default)` helper.

**Actual:** Relies on undocumented CPython behavior. A numpy scalar
(vs a Python float) in the `metrics` dict is already enough to break
this -- `numpy.float64.max` follows numpy NaN semantics.

**Blast radius:** All composite scoring. If this ever flips to
NaN-propagating, `benchmark_full` leader boards turn into NaN and the
whole evaluation pipeline is unusable until someone traces the
regression.

**Proposed fix:** Add a clamp helper at the top of `composite()`:

```python
def _safe(val, default):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) or math.isinf(v) else v
```

and use it on every `metrics.get(...)` call. Same for `composite_large`
and `composite_strict`.

---

### 2.5 MEDIUM -- `CrossingSwapPolish` under-counts crossings

- File: `dagua/layout/ops/crossing_swap.py:175-194`
- Class: wrong-set / scoring error. Op is `enabled=False` by default,
  but the bug is real if anyone turns it on.

The swap decision computes `before`/`after` crossings by filtering
edges touching the swapped pair `{a, b}`:

```python
mask = (t == a) | (t == b) | (s == a) | (s == b)
before += _count_crossings_for_layer_pair(x_view, s[mask], t[mask])
```

Only crossings *between filtered edges* are counted. But crossings
between a filtered edge (touching `a` or `b`) and an *unfiltered* edge
(touching neither) also change when we swap x_a and x_b -- the
filtered edge's endpoint just moved. Those are missed in both
`before` and `after`.

**Analysis:** Consider edges A=a->c (x_a=0, x_c=0), B=b->d (x_b=10,
x_d=10), C=e->f (x_e=5, x_f=5), all in one layer-pair. Before swap,
no crossings. After swap (x_a <-> x_b), A=a->c now runs (10,0)->(0,0)
and crosses C=e->f. The filter drops C from both before and after, so
the algorithm sees before=0, after=0, and keeps the swap.

**Expected:** For each swap, compare the full cross-count over all
edges in the layer-pair.

**Actual:** Only edges touching `{a, b}` are counted, hiding the
externally-induced crossings.

**Blast radius:** Config ships with `enabled=False`, so no current
user impact. If turned on, users will see net *regression* in
crossings on anything denser than a bipartite dominator pattern.

**Proposed fix:** Drop the filter and recount the full layer-pair,
or expand the filter to "edges whose endpoint x changed OR any edge
that could cross one whose endpoint changed" (which is essentially
the full set):

```python
# drop the per-edge filter:
before += _count_crossings_for_layer_pair(x_view, s, t)
```

Or do it incrementally: only edges touching `a` or `b` changed
x-values, so `crossings_new = crossings_old + delta` where `delta` is
`(sum over touched edge E, over all edges F != E) sign_change` --
still O(d_a + d_b) per swap but now correct.

---

### 2.6 LOW -- `longest_path_layering` pure-cycle collapse

- File: `dagua/utils.py:1358-1377`
- Class: degenerate-case handling. Already partly mitigated by
  `init_placement.py:74-112`, but worth documenting.

When the graph has **no source node** (pure cycle), Kahn's queue is
empty from the start. `layers` stays `[-1, -1, ...]` and then the
unresolved-fallback code sets every node to `fill_layer = 0`.

**Reproducer:**

```python
import torch
from dagua.utils import longest_path_layering
ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
print(longest_path_layering(ei, 3))
# [0, 0, 0]   <-- everybody in layer 0
```

**Expected:** Either return a non-degenerate layering via cycle
removal (what `init_positions` already does as a second pass), or
explicitly communicate "all cycle" so downstream code can pick its
2D-init branch without parsing counts.

**Actual:** Silently returns all-zero. Works today because
`init_positions` re-runs after `make_acyclic_robust`, and because
`Force2DInitIfFlat` detects the single-layer case. If any caller uses
`longest_path_layering` directly, they hit this pitfall.

**Blast radius:** Most callers go through `init_positions`, so small.
But the metric suite imports `longest_path_layering` directly
(`dagua/metrics.py:1349-1351`) for `quick()` when `topo_depth=None`.
On a pure-cycle graph that path hands `depth_position_correlation`
a constant vector, and `scipy.stats.spearmanr` returns NaN, which
then flows into composite (see 2.4). See Section 3.1.

**Proposed fix:** In `longest_path_layering`, detect "no source"
up-front and either raise or run `make_acyclic_robust` internally.
At minimum, add a docstring note pointing callers at
`init_positions` for the cycle-safe path.

---

### 2.7 LOW -- `detect_back_edges` visit order treats first-cycle asymmetrically

- File: `dagua/layout/cycle.py:44-46`
- Class: ordering / non-uniqueness.

The DFS starts from in-degree-0 nodes first, then "remaining". For a
pure cycle 0->1->2->0 there are no in-degree-0 nodes, so we enter by
iteration order (node 0). Back edge becomes 2->0. This is fine for a
single cycle, but for two disjoint cycles the choice of back edge is
an implementation detail of the node ordering. Confirmed behavior:

```python
from dagua.layout.cycle import detect_back_edges
import torch
ei = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
print(detect_back_edges(ei, 3).tolist())  # [False, False, True]
```

The reversed edge is the last one in the cycle. This is fine; calling
out as a low-severity *convention* bug because re-running with edges
in a different order may pick a different back-edge and change
downstream layouts, affecting reproducibility guarantees.

**Proposed fix:** Document it. No action required unless benchmarks
hit reproducibility drift.

---

## 3. Suspicious Patterns -- Worth Investigating

### 3.1 `depth_position_correlation` returns NaN on constant depth

`dagua/metrics.py:325-338`. `scipy.stats.spearmanr` emits a
`ConstantInputWarning` and returns NaN when either input is
constant. Verified:

```python
from dagua.metrics import depth_position_correlation
import torch
pos = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
depth = torch.tensor([0, 0, 0])
print(depth_position_correlation(pos, depth))
# {'depth_spearman_rho': nan, 'depth_spearman_pval': nan}
```

Combined with `composite()`'s CPython-specific NaN handling (2.4),
this is a potential benchmark killer for any single-layer graph.
Suggest clamping rho to 0.0 when constant inputs are detected and
returning 1.0 or 0.0 explicitly with a sentinel key (e.g.
`depth_spearman_defined: bool`).

### 3.2 `_spread_fanout_children` in-place mutation of `positions`

`dagua/layout/init_placement.py:571-622`. Docstring says "Modifies
positions in-place" and the call sites in `init_positions` pass the
locally-constructed positions tensor, so this is intentional. But:
the tensor is constructed with `torch.zeros(num_nodes, 2, device=device)`
and never `.clone()`'d before `_spread_fanout_children`. If
`positions` is ever passed in from a caller (it is not currently),
that caller loses their original. Low-severity, but adding
`.clone()` before the post-pass would make the function safe under
future reuse.

### 3.3 `_count_local_crossings` O(n) `sorted_layers.index()` in hot loop

`dagua/layout/init_placement.py:535`. The transpose heuristic calls
`sorted_layers.index(current_layer)` for every node-pair in every
layer, per pass. On `num_nodes=2000` graphs with 8 passes this is a
measurable O(N * layers^2) drag. Not a correctness bug, but worth
caching a dict `{layer: pos}` once at the top. See sprint 19 runtime
area.

### 3.4 `_greedy_fas` Python-loop O(N^2) per FAS call

`dagua/layout/cycle.py:162-177`. For each of N iterations we rescan
all N nodes, computing degrees by re-iterating full adjacency lists.
That's O(N * (N + E)) per call. Fine at N<=500 (the benchmark cap)
but will fail the scaling story. Priority queue + incremental updates
are the classical implementation.

### 3.5 `BarycenterReorder` sort-tie ordering can shift layouts between
torch versions

`dagua/layout/ops/barycenter.py:203,208`: `torch.argsort(..., stable=True)`
combined with `torch.sort(current_x)` (not marked stable). `torch.sort`
stability is platform-dependent on CUDA. If a layer has two nodes
with identical barycenter AND identical current x, the assignment
order can flip. Benchmark determinism may drift under CUDA builds.

### 3.6 `CenterPositions` / `NormalizePositions` do not re-check after
`AspectRatioFit`

`dagua/layout/ops/postprocess.py:542-650`. `AspectRatioFit` uses the
position bbox ignoring node sizes, so the final node-padded bbox can
still be off-target after the op runs. The `aspect_ratio` metric
(line 439-462 of metrics.py) includes node sizes. This explains some
of the "target=0.25 but still off" observations in the sprint-18h
sweep.

### 3.7 `Force2DInitIfFlat.extent_factor=1.0` may be too small

`dagua/layout/ops/force_2d_init.py:60`. When a cyclic graph is
collapsed to a single layer, x-extent is determined by
`init_positions` using `node_sep * (N-1)` spacing, so extent scales
with N. 1.0 extent_factor yields a square layout, which may be too
cramped for spring losses to unfold. Empirical sweep recommended for
small_world/cyclic families (area D overlap).

---

## 4. Sanity Checks That Passed

### 4.1 `_greedy_fas` fix

- Linear chain 0->1->2 returns `reversed_mask = [False, False]`.
  Verified.
- 3-cycle 0->1->2->0 reverses exactly one edge (edge 2, i.e. 2->0).
  Verified.
- 4-node DAG with tied out-in degree scores returns all
  `reversed_mask = [False] * 4`. Verified.

### 4.2 `detect_back_edges` self-loop handling

- Input `[[0,0,1],[0,1,2]]` (edge 0 is self-loop). Output
  `[True, False, False]`. Self-loops correctly flagged as back edges.

### 4.3 `longest_path_layering` basic cases

- Disconnected 2-component `[[0,2],[1,3]]` -> `[0,1,0,1]`. Correct.
- Isolated node + chain `[[1,2],[2,3]]` on 4 nodes -> `[0,0,1,2]`.
  Isolated 0 lands at layer 0 alongside the chain source, which is
  the documented behavior.
- Empty graph -> `[0,0,0,0,0]`. Correct.

### 4.4 `edge_length_cv` degenerate cases

- Single edge -> CV=0 (std clamp to 0 when n<=1). Correct.
- Zero-length edges (pos overlap) -> CV=0 (guarded by
  `mean_len > 1e-8`). Correct.

### 4.5 `count_overlaps_detailed` capped-cell determinism

- N=3000 all-zero positions with seed=42 produces identical counts
  across two calls. Correct.
- Without seed, global torch state controls the subsample, but with
  identical positions overlaps are identical. Determinism under
  `seed=` kwarg confirmed.

### 4.6 `_spectral_order` exception fallback

- `except Exception: return None` (line 380) is deliberate and
  well-scoped; lobpcg is known to fail on disconnected graphs.
  Callers of `_init_positions_vectorized` handle `None` by falling
  back to barycenter ordering. Correct.

### 4.7 `dag_consistency` forward direction logic

- Correct TB layout (y_tgt > y_src) reports consistency 1.0.
- Reversed TB layout reports 0.0. Correct.
- LR directions behave symmetrically to TB. Verified.

### 4.8 `BarycenterReorder` sweep direction coverage

- Upsweep iterates `range(1, num_layers)` (skips layer 0).
- Downsweep iterates `range(num_layers-1, -1, -1)` (skips
  num_layers-1 via the adj_layer_idx guard).
- Net: every layer visited in every iteration, alternating reference
  layers. Correct.

### 4.9 `make_acyclic` `.clone()` discipline

- Returns `edge_index.clone()` before in-place swap. No tensor
  aliasing. Correct.

### 4.10 `Force2DInitIfFlat` determinism via seeded generator

- Uses `torch.Generator(device="cpu").manual_seed(int(problem.seed))`.
  Reproducible across calls with the same `problem.seed`. Correct.

---

## 5. Recommended Action Queue (by impact/effort)

| Rank | Bug | Effort | Expected composite delta |
|------|-----|--------|--------------------------|
| 1 | 2.1 edge_straightness LR/TB symmetry | 1 LOC | up to +1.0 on LR-heavy graphs |
| 2 | 2.3 angular_resolution seed plumbing | 5 LOC | noise floor drop of ~0.2 |
| 3 | 2.2 dag_consistency self-loop exclusion | 3 LOC | +0.5 on graphs with self-loops |
| 4 | 2.4 composite NaN guards | 10 LOC | prevents catastrophic regression when rho becomes NaN |
| 5 | 3.1 depth_spearman NaN guard | 3 LOC | same as (4), upstream fix |
| 6 | 2.6 longest_path_layering pure-cycle warn | doc-only | none now; future-proofing |
| 7 | 2.5 CrossingSwapPolish counting correctness | ~20 LOC | unlocks opt-in use |

Items 1-3 are one-to-three line fixes that directly reclaim composite
points with no downstream risk. Items 4-5 harden the benchmark against
silent breakage. Item 7 only matters if the opt-in op is promoted to
default.

---

## 6. File References

All file paths absolute:

- `/home/jtaylor/projects/dagua/dagua/layout/cycle.py`
- `/home/jtaylor/projects/dagua/dagua/layout/init_placement.py`
- `/home/jtaylor/projects/dagua/dagua/utils.py` (L1297 and
  L1358-1377)
- `/home/jtaylor/projects/dagua/dagua/layout/engine.py`
- `/home/jtaylor/projects/dagua/dagua/metrics.py` (L206-492,
  L1147-1281)
- `/home/jtaylor/projects/dagua/dagua/layout/ops/barycenter.py`
- `/home/jtaylor/projects/dagua/dagua/layout/ops/crossing_swap.py`
- `/home/jtaylor/projects/dagua/dagua/layout/ops/force_2d_init.py`
- `/home/jtaylor/projects/dagua/dagua/layout/ops/ordering.py`
  (barycenter/median helpers at L381-449)
- `/home/jtaylor/projects/dagua/dagua/layout/ops/postprocess.py`
  (AspectRatioFit at L542-650)
