# Median + Transpose in `dagua_native`

## Section 1 -- Design

### Recommended ordering

Use the following discrete crossing-reduction block in the default native pipeline:

1. `BarycenterReorder(iterations=8)`
2. `MedianSweep(passes=4)`
3. `TransposeHeuristic(passes=8)` with early stop when a full pass makes no swap
4. Preserve the refined ordering for downstream coordinate assignment

I recommend this sequence over an alternating barycenter/median loop because current HEAD already has a production-tuned positional barycenter polish in [dagua/layout/ops/barycenter.py:40-69] and [dagua/layout/ops/barycenter.py:115-213]. It is directly wired into `dagua_native` at [dagua/layout/ops/pipelines/dagua_native.py:421-427], and Sprint 18k already tuned its pass count from 4 to 8. The least risky change is therefore to keep that pass exactly as-is and add a new ordering phase after it, not replace it.

The important nuance is that `BarycenterReorder` is a positional op and `MedianSweep` / `TransposeHeuristic` are ordering ops. On current HEAD, `BarycenterReorder` mutates `state.pos`, while `MedianSweep` and `TransposeHeuristic` only write `state.ordering` in [dagua/layout/ops/ordering.py:678-683], [dagua/layout/ops/ordering.py:757-762], and [dagua/layout/ops/ordering.py:832-837]. Nothing later in the native pipeline consumes `state.ordering`. So inserting those ops blindly after barycenter would compile but would not change final positions. The patch therefore needs one bridge:

- seed the ordering ops from the current per-layer x order produced by `BarycenterReorder`
- project the refined ordering back onto the existing per-layer x coordinates after each ordering op, or at least after transpose

That keeps the new phase in the same semantic style as `BarycenterReorder`: preserve the per-layer x set, only change which node gets which slot. It also avoids reintroducing the overlap regressions documented in [dagua/layout/ops/crossing_swap.py:55-64], where direct swap polishing was disabled by default.

### DAG gate

Gate the new median+transpose phase on acyclicity:

- run it only when `structure.is_acyclic` is true
- skip it entirely on cyclic graphs

The native resolve path already stores the classification on config in [dagua/layout/resolve.py:342-348], and the same structure is already used to suppress `DagOrderingLoss` on cyclic graphs in [dagua/layout/resolve.py:379-388]. Reuse that exact signal.

Reason for the gate: transpose is a local crossing optimizer over the current layer assignment. For cyclic graphs, the layer assignment is already an arbitrary artifact of cycle breaking / longest-path fallback, and the native pipeline explicitly compensates for cyclic collapse with `Force2DInitIfFlat` in [dagua/layout/ops/pipelines/dagua_native.py:386-395]. Running a greedy adjacent-swap heuristic against that artificial layering can lock in a visually worse order even when the sampled crossing count improves. The smallest regression-safe rule is therefore:

- acyclic graph: run barycenter -> median -> transpose
- cyclic graph: keep existing behavior

That matches the deliverable requirement and preserves `recurrent_feedback_cell`-style layouts.

### Dummy-node integration and fallback

The Wave 2 prompt says `MedianSweep` and `TransposeHeuristic` read `state.extras.expanded_graph`. Current HEAD does not match that statement. On HEAD:

- `InsertDummyNodes` writes `state.extras["expanded_graph"]` in [dagua/layout/ops/layering.py:630-686]
- `MedianSweep` reads `state.layers` and `state.adjacency` in [dagua/layout/ops/ordering.py:687-762]
- `TransposeHeuristic` reads `state.layers`, `state.ordering`, and `problem.edge_index` in [dagua/layout/ops/ordering.py:766-837]

So the dummy-aware behavior must be added explicitly.

The right fallback is:

- if `state.extras["expanded_graph"]` exists and the active solve state has been expanded to the same node count, use `expanded_graph.edge_index` and `expanded_graph.layers`
- otherwise, fall back to `problem.edge_index` and `state.layers`

The extra guard on node-count parity matters. In current `dagua_native`, `state.pos` and `state.layers` are original-node sized. If another concurrent wave has inserted dummy nodes only as metadata, but has not expanded `state.pos` and `state.layers`, the ordering ops cannot safely operate on the dummy graph. In that case they must fall back to the original graph instead of producing a mismatched `state.ordering`.

This is also why I do not recommend solving this by inserting `BuildAdjacency` into `dagua_native`. `BuildAdjacency` resolves the active edge tensor from `state.extras["preprocess_edge_index"]` or `problem.edge_index` in [dagua/layout/ops/preprocess.py:34-52], but it still sizes adjacency from `problem.num_nodes` in [dagua/layout/ops/preprocess.py:858-868]. That is correct for original graphs and incorrect for dummy-expanded graphs. The safer patch is to make the ordering ops themselves resolve the active layered graph, not to thread `expanded_graph` through the generic adjacency builder.

## Section 2 -- Exact code patches

### 2a) `build_dagua_pipeline` in `dagua/layout/ops/pipelines/dagua_native.py`

Current insertion point is the block at [dagua/layout/ops/pipelines/dagua_native.py:421-427]. Keep that barycenter call unchanged and append a DAG-gated median+transpose block after it.

Patch shape:

```python
# new imports near the top of the file
from dagua.layout.ops.ordering import (
    MedianSweep,
    MedianSweepConfig,
    TransposeHeuristic,
    TransposeHeuristicConfig,
)
```

```python
# inside build_dagua_pipeline(), before the return Pipeline(...)
structure = getattr(config, "_dagua_native_structure", None) or getattr(config, "structure", None)
is_acyclic = bool(getattr(structure, "is_acyclic", True)) if structure is not None else True
enable_native_median_transpose = bool(getattr(config, "use_native_median_transpose", True))
native_median_passes = int(getattr(config, "native_median_passes", 4))
native_transpose_passes = int(getattr(config, "native_transpose_passes", 8))

crossing_reduction_ops = [
    BarycenterReorder(BarycenterReorderConfig()),
]
if enable_native_median_transpose and is_acyclic:
    crossing_reduction_ops.extend(
        [
            MedianSweep(MedianSweepConfig(passes=native_median_passes)),
            TransposeHeuristic(TransposeHeuristicConfig(passes=native_transpose_passes)),
        ]
    )
```

```python
# in the final pipeline composition, replace the single barycenter op
*crossing_reduction_ops,
```

That patch is deliberately narrow:

- it preserves the 8-pass barycenter default
- it keeps cyclic behavior unchanged
- it adds a config-controlled rollback flag
- it does not couple this wave to Brandes-Koepf landing order

### 2b) Parameter tweaks

Recommended defaults:

- `BarycenterReorder(iterations=8)`: unchanged; keep [dagua/layout/ops/barycenter.py:67]
- `MedianSweep(passes=4)`: enough to clean up small local inconsistencies after an 8-pass positional barycenter
- `TransposeHeuristic(passes=8)`: matches the current `TransposeHeuristicConfig` default in [dagua/layout/ops/ordering.py:570-580], but actual runtime is bounded by the op's own early-stop behavior through `_transpose_heuristic(..., num_passes=...)` in [dagua/layout/ops/ordering.py:823-829]

Why `passes=4` for median instead of 24, which is the default in [dagua/layout/ops/ordering.py:557-568]?

- in `dagua_native`, median is no longer the first discrete ordering phase
- it is running after an already-tuned 8-pass barycenter positional polish
- the target is a cheap residual cleanup, not a full Sugiyama solve

So the production default should be shorter than the standalone ordering-op default.

### 2c) Patch `MedianSweep` / `TransposeHeuristic` to support either expanded or original graph

This is the critical functional patch. Without it, the new pipeline block is either a no-op or dummy-blind.

I would patch [dagua/layout/ops/ordering.py:16-124] with three helpers:

1. `_resolve_active_layered_graph(problem, state) -> tuple[torch.Tensor, torch.Tensor, int]`
2. `_resolve_initial_ordering(layers_cpu, state, num_nodes) -> Optional[torch.Tensor]`
3. `_apply_ordering_to_positions(state, layers_cpu, ordering_cpu, num_nodes) -> None`

Behavior:

```python
def _resolve_active_layered_graph(
    problem: LayoutProblem,
    state: SolveState,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the layered graph that ordering ops should use.

    Prefer ``state.extras["expanded_graph"]`` only when the solve state already
    carries tensors at the expanded node count. Otherwise fall back to the
    original graph.
    """
```

Resolution rule:

- read `expanded_graph = state.extras.get("expanded_graph")`
- if it has `edge_index`, `layers`, and `num_nodes`
- and `state.pos` or `state.ordering` is already sized to `expanded_graph.num_nodes`
- then use `expanded_graph.edge_index`, `expanded_graph.layers`, `expanded_graph.num_nodes`
- else use `problem.edge_index`, `state.layers`, `problem.num_nodes`

The second helper seeds median / transpose from current x order:

```python
def _resolve_initial_ordering(
    layers_cpu: torch.Tensor,
    state: SolveState,
    num_nodes: int,
) -> Optional[torch.Tensor]:
    """Use existing ordering when present, otherwise derive it from current x."""
```

Order of precedence:

- if `state.ordering` exists and matches `num_nodes`, use it
- else if `state.pos` exists and matches `num_nodes`, derive per-layer rank from `x`
- else return `None` and let `_initial_ordered_layers()` fall back to stable node-id order

That fixes the main native-pipeline bug: `MedianSweep` currently starts from `_initial_ordered_layers(layers_cpu)` in [dagua/layout/ops/ordering.py:737], which ignores the `BarycenterReorder` result completely.

The third helper projects ordering back onto positions:

```python
def _apply_ordering_to_positions(
    state: SolveState,
    layers_cpu: torch.Tensor,
    ordering_cpu: torch.Tensor,
    num_nodes: int,
) -> None:
    """Permute each layer's existing x-values to match ``ordering_cpu``."""
```

Implementation rule:

- for each layer, collect member nodes
- sort member nodes by `ordering_cpu`
- sort the current x-values for that layer
- assign the sorted x-values to nodes in the new order

That is the same "preserve x set, change assignment" strategy already used in [dagua/layout/ops/barycenter.py:205-210].

Then patch `MedianSweep.apply()` at [dagua/layout/ops/ordering.py:706-762]:

- resolve the active graph with `_resolve_active_layered_graph()`
- initialize `ordered_layers` from `_initial_ordered_layers(layers_cpu, ordering=initial_ordering)`
- when `state.adjacency` is absent or incompatible, derive `parents` / `children` from `_layered_neighbors_from_edges()` instead of failing
- after `state.ordering` is written, call `_apply_ordering_to_positions()` when `state.pos` matches the active node count

And patch `TransposeHeuristic.apply()` at [dagua/layout/ops/ordering.py:785-837]:

- resolve `edge_index` and `layers` from the active graph, not always from `problem.edge_index`
- if `state.ordering` is missing, seed from current x order
- after writing refined `state.ordering`, project it back to `state.pos`

That gives `dagua_native` a visible post-barycenter refinement without needing a new op file.

### 2d) Rollback flag in `dagua/config.py`

Add three public fields in [dagua/config.py:124-140], right after `use_tree_fast_path`:

```python
use_native_median_transpose: bool = True
native_median_passes: int = 4
native_transpose_passes: int = 8
```

That satisfies the explicit rollback requirement. If a graph family regresses, the user can disable the entire phase with:

```python
LayoutConfig(use_native_median_transpose=False)
```

## Section 3 -- Interaction with other wave-2 patches

### #1 Brandes-Koepf

Brandes-Koepf belongs after crossing reduction, not before it. The discrete ordering phase defines the permutation; BK converts that permutation into x coordinates. That is already how the dedicated Sugiyama path is structured in [dagua/layout/ops/pipelines/sugiyama.py:63-72] and [dagua/layout/ops/sugiyama.py:1617-1730] followed by [dagua/layout/ops/sugiyama.py:1733-1804].

The order should be:

1. reduce crossings
2. assign x from that order with BK
3. aspect-fit / cluster arrange

For this specific patch, I would not hard-require BK to land simultaneously, because the helper above makes median and transpose visible immediately by re-permuting the existing x slots. But once Wave 2 patch #1 is merged, BK should consume the refined `state.ordering` instead of the pre-median order.

### #2 Dummy nodes

If the dummy-node wave lands first and expands the active solve state, median and transpose should operate on `expanded_graph.edge_index` and `expanded_graph.layers`. That improves crossing reduction exactly where the scan says Dagua is weak: long edges that only become orderable once they are dummy-split.

If dummy metadata exists but the active state is still original-node sized, fall back to original edges. Do not try to synthesize a partial expanded ordering. That would be fragile and would silently mismatch node counts.

### #4 Per-component decomposition

This patch composes cleanly with per-component layout. The ordering block should run inside each component subpipeline, not after packing. Component decomposition changes the problem boundary; median and transpose should simply see the smaller per-component `problem.edge_index` / `state.layers` and run normally.

That is actually safer than a global ordering pass because disconnected components should never influence each other's left-right order.

### #5 Aspect ratio

Aspect-ratio fitting is downstream geometry rescaling. It does not change per-layer order. So this patch is orthogonal to Wave 2 aspect-ratio work. The only order constraint is to keep `AspectRatioFit` after crossing reduction, exactly where it already sits in [dagua/layout/ops/pipelines/dagua_native.py:434-439].

## Section 4 -- Regression safety

### Top dagua wins

The new phase should be net-positive or neutral on the graphs we do not want to regress. Current cached `dagua` crossing-rate baselines from `eval_output/variant_bench_full/positions/*__dagua.pt`, recomputed with `sampled_crossing_rate()` in [dagua/metrics.py:593-669], are:

| Graph | Current `crossing_rate` | Read |
| --- | ---: | --- |
| `random_dag_200` | `0.233238` | already crossing-heavy; likely improves |
| `org_chart_deep` | `0.021854` | near-zero; should be neutral or slightly better |
| `random_dag_50` | `0.129560` | already better than dot/dagre/ELK on this metric; must not regress |
| `hub_fanout_label_skew` | `0.000000` | exact no-op target |
| `org_chart_1_5_4_8` | `0.000000` | exact no-op target |

Why this stays safe:

- median is seeded from current x order, not from raw node-id order
- transpose is strict-improvement only
- zero-crossing layers are stable fixed points
- the whole phase is disabled on cyclic graphs

The one existing op that already tried direct swap polish, `CrossingSwapPolish`, was left disabled because it created overlap regressions unless followed by a heavier projection pass; see [dagua/layout/ops/crossing_swap.py:55-64]. That is a useful warning: do not revive positional greedy swapping. Use the ordering ops plus order-to-x projection instead.

### Cyclic graphs

The explicit regression target is `recurrent_feedback_cell`. Current cached crossing rates are:

- `dagua`: `0.000000`
- `graphviz_dot`: `0.000000`
- `dagre`: `0.000000`
- `elk_layered`: `0.142857`

This is a graph where there is nothing to gain from transpose and real risk in perturbing the current layout. The reason to skip transpose on cyclic graphs is not merely performance. It is that the objective is wrong there. The ordering heuristic assumes layer adjacency encodes meaningful flow. In cyclic graphs it often does not.

The clean rule is therefore:

- if `structure.is_acyclic` is false, skip the new median+transpose block entirely

That keeps the cyclic composite effectively unchanged.

## Section 5 -- Tests

I would cover this with four tests, reusing existing ordering fixtures where possible.

### 1. Unit: `MedianSweep` reduces crossings on a 2-layer bipartite

File: [tests/test_ops_ordering.py](/home/jtaylor/projects/dagua/tests/test_ops_ordering.py)

Add a test adjacent to the existing barycenter reduction test at [tests/test_ops_ordering.py:212-232]. Use the same known-crossing two-layer DAG:

```python
edge_index = [(0, 3), (1, 2), (0, 5), (1, 4)]
layers = [0, 0, 1, 1, 1, 1]
```

Assertions:

- `MedianSweep(...).apply(...)` decreases `_crossings_between_adjacent_layers(...)`
- when `state.pos` is provided, the op also permutes x in place rather than only writing `state.ordering`

That second assertion is the regression guard against the current no-op integration bug.

### 2. Unit: `TransposeHeuristic` finds the known optimal swap

File: [tests/test_ops_ordering.py](/home/jtaylor/projects/dagua/tests/test_ops_ordering.py)

Extend the existing transpose test at [tests/test_ops_ordering.py:163-190] into an explicit 3-node-per-layer example. The current fixture already verifies that the heuristic lowers crossings and preserves per-layer permutations. Add:

- a known initial ordering
- the expected final ordering after the beneficial adjacent swap
- optional `state.pos` so the test also verifies x-slot projection

This test should assert the exact swapped middle-layer order, not just "crossings decreased."

### 3. Integration: default native pipeline improves `dense_pair_50`

Best home is [tests/test_layout_default_dispatch.py](/home/jtaylor/projects/dagua/tests/test_layout_default_dispatch.py), because that file already validates `algorithm=None` and explicit `algorithm="dagua_native"` routing.

Test recipe:

- build `dense_pair_50` from `make_sparse_dense_pair(n=50, seed=42)`
- run `layout(..., LayoutConfig(algorithm="dagua_native", seed=42, use_native_median_transpose=False, steps=<small fixed budget>))`
- run again with `use_native_median_transpose=True`
- compute `sampled_crossing_rate(..., seed=42)`
- assert the enabled run is strictly lower or at minimum not higher if stochastic tolerance is needed

This is the end-to-end guard that the new phase is actually active in the default pipeline.

### 4. Regression: cyclic graph unchanged

Use `recurrent_feedback_cell`, either in the same dispatch test file or a new native-pipeline regression file.

Assertions:

- positions are identical or numerically equivalent between flag off and on
- `crossing_rate` is unchanged
- composite is unchanged within a tiny tolerance

Because the DAG gate skips the phase, this should be a true no-op test, not just a "did not get worse" test.

### Optional fifth test worth adding

Patch-level dummy-node safety deserves one small unit test in `tests/test_ops_ordering.py`:

- provide `state.extras["expanded_graph"]`
- ensure the op uses it only when state tensors match the expanded node count
- otherwise verify it falls back to the original graph

That prevents subtle mismatched-node-count failures when Wave 2 patches merge in a different order.

## Section 6 -- Expected impact

`crossing_rate` is scored in `composite()` as:

```python
crossing_score = max(0.0, 1.0 - metrics.get("crossing_rate", 0.5) * 10)
```

at [dagua/metrics.py:1185]. That means every `0.01` reduction in crossing rate is worth about `0.1` raw score and `1.0` weighted composite point on this metric, until the rate reaches zero. Because the four target graphs are all acyclic and crossing-sensitive, this patch has direct upside.

Empirical current baselines from cached `dagua` positions:

| Graph | Current composite | Current `crossing_rate` | Best competitor `crossing_rate` seen | Projected `crossing_rate` after patch | Projected composite |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_pair_50` | `71.81` | `0.034610` | `0.016548` (`graphviz_dot`) | `0.022 - 0.026` | `72.8 - 74.0` |
| `extreme_mixed_width_transformer` | `73.82` | `0.102564` | `0.000000` (`graphviz_dot`), `0.025641` (`dagre`) | `0.026 - 0.051` | `75.5 - 77.0` |
| `hexagonal_lattice_42` | `82.42` | `0.013302` | `0.000000` (`dot`, `dagre`) | `0.004 - 0.008` | `83.5 - 84.8` |
| `sierpinski_42` | `78.35` | `0.010526` | `0.000000` (`dot`, `elk`) | `0.003 - 0.006` | `79.5 - 80.8` |

Interpretation:

- `dense_pair_50` is the cleanest native win. Dagua is already close to the competitor floor. Median+transpose should recover about one third to one half of the remaining crossing gap without needing dummy nodes.
- `extreme_mixed_width_transformer` has only 39 valid sampled pairs, so each removed crossing pair moves the metric materially. This is exactly the kind of small, branchy DAG where a post-barycenter transpose pass pays off.
- `hexagonal_lattice_42` and `sierpinski_42` are already close to planar-zero. The likely gain is smaller in absolute composite terms, but these are very low-risk improvements because the good target is simply "remove the last few crossings."

I would describe the standalone effect of this patch as:

- likely `+1` to `+3` composite on the stated crossing-heavy DAGs
- potentially higher on very small DAGs like `extreme_mixed_width_transformer` because the crossing metric there is quantized and currently clipped to zero score

If the dummy-node wave lands first, the top end of those ranges becomes more credible, because transpose will then be operating on the correct local segments rather than on long unsplit edges.

## Section 7 -- Rollback

Expose one public kill switch:

```python
LayoutConfig(use_native_median_transpose=False)
```

And two tuning knobs for debugging:

```python
LayoutConfig(native_median_passes=0)
LayoutConfig(native_transpose_passes=0)
```

Rollback order:

1. set `native_transpose_passes=0` if a regression appears only on a narrow DAG family
2. set `use_native_median_transpose=False` if the whole phase must be disabled quickly

That gives a fast bisect path without touching the already-adopted 8-pass barycenter polish.

## Bottom line

The repo already has the right primitives. The missing work is composition and state-bridging:

- keep the current 8-pass `BarycenterReorder`
- seed median from the current x order
- let transpose refine that order
- project the refined order back onto positions
- prefer `expanded_graph` when the solve state is actually expanded, otherwise fall back safely
- skip the entire phase on cyclic graphs

That is the smallest patch that makes `MedianSweep` and `TransposeHeuristic` real in `dagua_native` instead of decorative imports, and it lines up cleanly with the parallel Wave 2 BK and dummy-node work.
