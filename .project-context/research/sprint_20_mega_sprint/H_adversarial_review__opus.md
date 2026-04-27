# Sprint-20 H -- Adversarial review of sprint-19 commits (Opus 4.7)

Read-only review of `33868c2` (19d per-component), `d1ba9ef` (19e topology
aspect), `092ee79` (19f median+transpose), `519f58e` (19g BK), `ec7d4db`
(19h dummy nodes). The premise of the review is harsh: "what would break
this in production?" Findings cite file:line at the current HEAD
`ec7d4db`.

## TL;DR -- top 5 risks ranked

1. **MedianSweep collapses every dense-DAG x to a single column**
   (`dagua/layout/ops/ordering.py:310-317`). When every layer has <=1
   node, the op overwrites all x with `torch.median(pos[:,0])`. That
   intentionally rescues `dense_pair_50`, but it will detonate any
   single-chain or longest-path DAG where the gradient core actually
   produced sane x spread. No gate, no minimum N, no opt-out.

2. **Per-component decomposition leaks the parent's gappy layer numbering
   into each child** (`dagua/layout/ops/pipelines/dagua_native.py:744-748`).
   Children inherit the parent's longest-path layer ids sliced by node
   index, never relabeled to start at 0. `build_layer_index`
   (`dagua/layout/layers.py:168`) sets `num_layers = max(layers)+1`, so a
   child whose nodes happen to land at parent layers `{5,7,9}` allocates
   ten layer slots, seven of them empty. Nothing crashes, but every
   layer-aware op (Force2DInitIfFlat, BK gate, BarycenterReorder,
   MedianSweep) silently sees a different topology than the child
   actually has.

3. **AspectRatioFit runs twice in the per-component path with conflicting
   targets** (`pipelines/dagua_native.py:1162-1166` inside the per-child
   pipeline, then `pipelines/dagua_native.py:1324-1328` on the tiled
   parent). The inner call uses the topology-aware target resolved from
   each child's structure; the outer call hard-codes
   `AspectRatioFitConfig()` (= None target = pure 0.25 fallback). For
   `disconnected_label_cycle_collage` -- the very graph that motivated
   sprint-19d -- this means topology-aware aspect is silently overridden
   by 0.25 after tiling. Wide-layered children get squashed, lattice
   children get stretched.

4. **Dummy-node insertion mutates `state.layers` to expanded layer ids
   but never restores them on the per-component recursion path**. When a
   child component triggers dummy insertion, the gradient core,
   barycenter, median+transpose, and BK all run on the expanded graph.
   `StripDummyNodes` (`dagua/layout/ops/postprocess.py:1144-1148`) only
   restores `state.layers` from `extras["original_layers"]` if that key
   exists -- but `extras["expanded_graph"]` is **never cleared** after
   stripping. On the parent's outer code path that's harmless. Inside
   `_run_native_problem`, the result is sliced via
   `result[: problem.num_nodes]` (`pipelines/dagua_native.py:504`), but
   any debugger / op that holds a reference to `state` and inspects
   `state.extras["expanded_graph"]` after the fact will think the
   problem is still expanded.

5. **`_should_apply_brandes_koepf_refine`'s component check uses an
   exact list match** (`dagua/layout/ops/coordinate.py:991-993`):
   `if component_sizes not in ([num_nodes], [num_nodes - 1, 1])`. After
   sprint-19d, the BK op is invoked inside per-component children; so
   the gate is satisfied for clean per-child cases. But on graphs where
   decomposition is **skipped** (clusters, pins, dominant component +
   3+ singletons), the BK op sees the parent and bails out for any
   shape outside that exact pattern -- including the very common
   `[N-2, 1, 1]` ("two satellites") case. Silent skip, no telemetry.

## Per-commit findings

### 33868c2 -- sprint-19d per-component decomposition

#### F1 (high): child layers inherit parent gaps

`_extract_component_problem` at `pipelines/dagua_native.py:743-747` does

```
sub_layer_assignments = layer_assignments.to(...)[parent_indices].clone()
```

with no remap. `layer_assignments` here is whatever
`_dagua_native_layer_assignments` was on the parent -- post sprint-19h,
that's a longest-path layering of the **whole** parent graph. Two
isolated subgraphs in the parent are independently layered, so node id
12 in component A might get layer 5 while node id 13 in component B is
in layer 7, etc.

Downstream consequences:
- `build_layer_index(layers)` in `dagua/layout/layers.py:168` allocates
  `max(layers)+1` slots. Child with layers `{5,7,9}` gets a 10-slot
  layer index where only 3 slots are used.
- `_initial_ordered_layers` in `dagua/layout/ops/ordering.py:435-436`
  does the same -- iterates 10 empty/non-empty layer buckets.
- `Force2DInitIfFlat` in `dagua/layout/ops/force_2d_init.py:97-98`
  reads `layer_index.num_layers`. A child with 3 actual layers but
  `num_layers=10` is treated as "deep enough" and 2D-init is skipped
  even if the 3 actual layers all collapse to flat geometry.

`classify_graph` -- the *only* downstream op that does the right thing
-- counts nonempty layers via `nonempty_counts` at
`graph_classify.py:310-311`, so the structure tags are correct.
Inconsistent.

**Repro**: 4-component graph where parent longest-path puts components
at non-contiguous layer ranges. Decomposition fires, child sees
`max(layers)=large`, layer index has many empty layers. Verify with
`build_layer_index([5,7,9]).num_layers == 10`.

**Proposed fix**: in `_extract_component_problem`, relabel child
layers via `unique = torch.unique(sub_layer_assignments,
sorted=True); sub_layer_assignments = torch.searchsorted(unique,
sub_layer_assignments)`. One-line. Or pass `layer_assignments=None` to
the child config so it recomputes longest-path on the local edge_index
(safer, slightly slower).

#### F2 (high): outer AspectRatioFit on tiled output ignores topology

`pipelines/dagua_native.py:1324-1328`:

```
outer_state = AspectRatioFit(AspectRatioFitConfig()).apply(
    problem, SolveState(pos=tiled_positions), ctx,
)
```

No `target_aspect`. `AspectRatioFit` falls back to 0.25 (the doc-spec
default per `postprocess.py:518-523`). Inside each child the BK-aware
target was applied. So a parent whose children are e.g.
`bipartite_dag` -> target 0.85 each, then the *tiled* result gets
flattened to 0.25. The tiling itself uses
`_COMPONENT_PACK_TARGET_ASPECT = 1.0` for the packing aesthetic
(`pipelines/dagua_native.py:78`), then immediately gets reshaped.
Three different aspect targets fighting in three different stages.

**Repro**: any disconnected graph with at least one wide-layered child.
`disconnected_label_cycle_collage` is the obvious candidate -- it was
the win target -- check its h2h: did the +12.33 it scored actually
come from decomposition, or from the +12 alone offsetting the
post-tile aspect squash?

**Proposed fix**: pass `target_aspect=getattr(prepared_config,
"_dagua_native_target_aspect", None)` to the outer call, or skip the
outer AspectRatioFit entirely (the children already self-fit and the
tiling preserves intra-child geometry).

#### F3 (med): _COMPONENT_TILE_PAD_FACTOR is global, not per-child

`_tile_component_positions` uses `gap = max(node_sep *
_COMPONENT_TILE_PAD_FACTOR, 1.0)` at `pipelines/dagua_native.py:889`,
where `node_sep` is the **parent**'s resolved node_sep. If a parent
has e.g. `adaptive_spacing=True` and `num_nodes` is large, the parent
node_sep is shrunk (per `dagua/layout/resolve.py:adaptive_spacing`).
Children then get their own (much smaller) node_sep. So the inter-
child gap can end up tighter than intra-child gap on hub-and-spoke
graphs. Visual collisions between component bboxes.

**Proposed fix**: gap should be `max(child_node_sep) *
_COMPONENT_TILE_PAD_FACTOR`, taking max over per-child resolved
node_sep. Or use the box-bbox padding that's already in
`_COMPONENT_TILE_PAD_FACTOR` and add a per-child geometric buffer.

#### F4 (med): _has_cross_component_flex check tolerates global flex

`_has_cross_component_flex` at `pipelines/dagua_native.py:626-651` is
documented as "global spacing flex is allowed because it is re-applied
inside each child solve". True for `flex_node_sep`, but
`_subset_flex` at `pipelines/dagua_native.py:613-684` rebuilds the
align_groups from scratch and drops any group with <2 in-component
members. A user who provided an alignment group spanning the whole
graph (e.g. "all root nodes left-align") will silently lose that
constraint when each component sees only a fragment of the group.

**Repro**: `g.align(["a", "b", "c"], axis="x")` where a, b, c are in
three different components. The check at `pipelines/dagua_native.py:670`
(`if unique_members.numel() >= 2`) drops the group entirely from each
child. The user's align constraint is dropped without warning.

**Proposed fix**: either disable decomposition when align_groups span
multiple components (make `_has_cross_component_flex` return True for
this case -- the doc text says "global spacing flex is allowed" but
align groups aren't spacing) or surface a warning.

#### F5 (low): dominance gate uses largest/total, not largest/(largest+rest)

`pipelines/dagua_native.py:586-590`:

```
if largest_component / max(problem.num_nodes, 1) >= 0.85:
    return False
```

Combined with the singleton-skip at L591-593. For a graph with 10 nodes
where one component has 8 and two singletons exist, fraction = 0.8 < 0.85,
so we DO decompose. For 100 nodes / 86+singletons, fraction = 0.86 >=
0.85 so we skip. The threshold of 0.85 is arbitrary. Worse: for graphs
where the "dominant" component is itself uninteresting (e.g. a long
chain of 100 nodes with 5 singletons attached), per-component
decomposition would actually help. Magic number with no measurement.

### d1ba9ef -- sprint-19e topology-aware aspect

#### F6 (high): rank_sep_multiplier is dead code today

`dagua/layout/resolve.py:131-156` returns `(target_aspect,
rank_sep_multiplier)` but every branch returns `1.0` for the
multiplier. Then `resolve.py:351` does `resolved_rank_sep *=
rank_sep_multiplier` (no-op) and stores both
`_dagua_native_rank_sep` and `_dagua_native_rank_sep_multiplier`.
Nothing else reads `_dagua_native_rank_sep_multiplier`. The plan
mentioned multipliers but the implementation never used them. Either
fold the multiplier into rank_sep at the call site (delete the
attribute) or wire it up.

#### F7 (med): topology tags ignore the directed/undirected gap

`_derive_topology_tags` at `graph_classify.py:340-398` early-returns
empty tags if `not is_directed_acyclic` (line 374). So **every cyclic
graph gets the default 0.25 target**. For `small_world_100` (cyclic,
already losing -8.51 to igraph_sugiyama), there's no path to a
different aspect via tags. Sprint-19e cannot help cyclic small-world,
yet this graph is on the loss-target list. This commit's contract
specifically says "targets families that aren't trees", but cyclic
graphs are in scope and get nothing.

#### F8 (med): is_lattice_like rule mis-tags Sierpinski-like graphs

`graph_classify.py:377-384`: lattice_like fires when
`is_planar_hint` AND `2 <= max_degree <= 6` AND `1.0 <=
edge_to_node_ratio <= 2.2` AND `num_layers >= 5` AND `layer_width_cv
<= 0.45`. The CV threshold is the killer: triangular/hex lattices
have very uniform layer widths, but a Sierpinski (sprint-19f win
graph) has highly variable layer widths -- so it's NOT lattice_like
under this rule. Yet the commit message claims sprint-19e helped
sierpinski_42 (+1.18). That win must be from another tag (probably
the planar_dag fallback at line 388, which fires for any planar DAG
with max_degree<=4 + num_layers>=5 NOT in lattice_like). Mostly OK,
but the tag classification is fragile -- one outlier layer (e.g. a
big middle layer) flips lattice -> planar -> different aspect (0.05
-> 0.08), and the classification result is silent.

#### F9 (low): `_count_components` no longer detects acyclicity

The 19e refactor split `_count_components_and_acyclic` into
`_count_components` and `_is_undirected_acyclic`. The former at
`graph_classify.py:96-145` short-circuits when `num_edges > num_nodes
- 1` and returns `1`. That's WRONG for any disconnected dense graph
(num_edges > N-1 doesn't imply 1 component). It used to be only
"conservative for the tree-shortcut", but now `_count_components` is
called standalone at `graph_classify.py:471`. For a graph with two
dense components of 5 nodes each (10 edges total > 9 = N-1), the
function now returns `1` instead of `2`. This is silently wrong and
breaks the per-component decomposition gate (which uses
`structure.num_components`).

**Repro**:
```
import torch
from dagua.layout.graph_classify import classify_graph
ei = torch.tensor([[0,0,1,1,2,5,5,6,6,7],
                   [1,2,2,3,3,6,7,7,8,8]], dtype=torch.long)
print(classify_graph(ei, 10).num_components)  # outputs 1, should be 2
```

This is the most concrete bug in 19e. It silently disables
sprint-19d decomposition on dense-disconnected graphs.

### 092ee79 -- sprint-19f median + transpose

#### F10 (high): single-node-per-layer collapse to median is a footgun

`ordering.py:310-317`:

```
if ordered_layers and all(len(layer_nodes) <= 1 for layer_nodes in ordered_layers):
    pos_new[:, 0] = torch.median(pos_new[:, 0])
```

This destroys all x information when the layering happens to put each
node in its own layer. The condition fires for:

- `dense_pair_50` (the win target -- intentional)
- Any chain (every node at unique depth, including chains_50, chain
  family graphs)
- Any DAG where longest-path puts nodes at distinct depths -- e.g. a
  totally-ordered diamond, a thin tree without siblings, ascending
  random_dag where every node has its own depth

The MedianSweep op writes `state.pos` directly (`ordering.py:316`).
The optimizer is already done by this point so there's no parameter
graph to break, but the resulting layout is pathological for chains:
all nodes get x = median, then OverlapProjection has to push them
apart from a degenerate start. Even with strong projection, the
post-median geometry will be a tight zigzag instead of any
meaningful arrangement.

**Repro**: layout a chain of 30 nodes through `algorithm="dagua"`,
inspect x-spread before/after MedianSweep. Pre-19f: x spread is
solver output. Post-19f: x is collapsed to one column.

**Proposed fix**: gate the collapse on `num_nodes > N_min` AND
`max_layer_width > 1` somewhere in the graph (i.e. only collapse if
the layering has at least one layer with multiple nodes). A pure
chain has no inter-node ordering ambiguity to fix; leave its x alone.

#### F11 (high): seeding ordering from current x and re-sorting reads/writes circularly

`ordering.py:322-327`:

```
sorted_x, _ = torch.sort(pos_new[members, 0])
pos_new[ordered_members, 0] = sorted_x
```

This permutes x-coordinates to match the new ordering. Sounds right --
but `sorted_x` is **all the x values from the layer, sorted ascending**.
If the gradient solver placed nodes with x=[10, 20, 30] and the new
ordering is [b, a, c], the result is x=[10, 20, 30] applied to [b, a, c]
-> b=10, a=20, c=30. The geometric spacing is preserved, but the
**identity** of which node had which x is lost. For a graph where the
solver had placed certain nodes at meaningful x positions (e.g.
matching cluster centroid, or matching pin proximity), that identity
is destroyed.

This isn't strictly a bug -- it's the whole point of
"reorder-by-permutation" -- but it interacts badly with downstream pin
loss / align loss. They run during `gradient_core`, but the post-loop
median+transpose+BK can rearrange away from where the optimizer
converged. The pipeline has no "post-rearrangement pin reapply" step
either via projection or loss.

**Proposed fix**: when align_groups or soft pins are active, run a
final HardPinProjection / soft-pin re-snap *after* BK and before
AspectRatioFit. Or skip median+transpose+BK entirely when pin/align
flex is present. Currently no such interaction is guarded.

#### F12 (med): _solve_state_matches_node_count picks wrong path on size-1 collisions

`ordering.py:175-189`:

```
return (
    state.pos is not None and ... shape[0] == num_nodes
)
```

If `pos` and `expanded_num_nodes` happen to coincide (e.g. a graph
with 0 long edges where dummies aren't added but the state was
re-used from a previous run that did expand), the check passes and we
use the stale expanded layers. Look at this path: it depends on
whether `extras["expanded_graph"]` is ever non-None when the active
state isn't actually expanded. Sprint-19h ALWAYS leaves
`extras["expanded_graph"]` set after StripDummyNodes runs (no
clearing). So if the same SolveState is reused (which it shouldn't
be in production but happens in tests / batched calls), this can
mis-fire.

**Proposed fix**: StripDummyNodes (`postprocess.py:1131-1153`) should
clear `state.extras["expanded_graph"]`, `extras["original_layers"]`,
`extras["original_layer_index"]` after restoring.

#### F13 (med): native_median_passes=4, native_transpose_passes=8 -- N^2 cost not gated

`config.py:139-140` defaults. Each pass iterates every node in every
layer. For a 500-node graph with 50 layers averaging 10 nodes, each
median pass is `O(L * avg_layer_size * neighbor_count)` -- order of
magnitude tens of thousands of ops. 4 + 8 = 12 sweeps on CPU through
Python loops. For dependency_500 (the 19h win target with 500 nodes),
this could add hundreds of milliseconds. There's no scale gate. The
plan said "4-8 passes converge for typical inputs" -- typical at what
scale?

**Repro**: time `dagua.layout(dependency_500_graph)` with
`use_native_median_transpose=True` vs False. The h2h shows +4.28 on
dependency_500 from 19h alone, not 19f, so the cost-benefit on 500-
node graphs hasn't been measured directly.

### 519f58e -- sprint-19g Brandes-Köpf

#### F14 (high): _ordering_from_current_x uses Python for-loop with tolist()

`coordinate.py:892-905`:

```
for layer_index in range(max_layer + 1):
    layer_nodes = torch.where(layers == layer_index)[0].tolist()
    layer_nodes.sort(key=lambda node_idx: (float(pos_cpu[node_idx, 0].item()), node_idx))
    for position, node_idx in enumerate(layer_nodes):
        ordering[node_idx] = position
```

For each layer this does a `torch.where` (O(N) scan), then a Python
sort with `.item()` calls (slow). On a graph with many layers (e.g.
deep DAGs that 19g specifically targets, like
`extreme_mixed_width_transformer`) this is expensive. Combined with
the existing `_ordering_from_current_x` work in `MedianSweep` and
`TransposeHeuristic` (which already build ordering from x), this is
duplicative work. BK reads x from state.pos; median/transpose just
wrote ordering+pos; BK could trust state.ordering instead of
re-deriving from x.

**Proposed fix**: use `state.ordering` if present and node-count
matches, else derive from x. Saves a redundant scan on every BK call.

#### F15 (high): `_should_apply_brandes_koepf_refine` exact-match component check

`coordinate.py:991-993`:

```
component_sizes = _weak_component_sizes(...)
if component_sizes not in ([num_nodes], [num_nodes - 1, 1]):
    return False
```

Allows exactly: one component, OR one big + one singleton. For graphs
that should fall to the parent path (decomposition skipped because of
clusters or pins) AND have the very common `[N-2, 1, 1]` (two
satellites) shape, BK silently skips. The doc-comment at
`coordinate.py:984-987` says this is intentional, but real graphs
have arbitrary tail topology. A 500-node DAG with 3 isolated nodes
should still benefit from BK on the 497-node main component -- but
since BK runs on the full problem (not per-component when
decomposition is skipped), it bails out entirely.

**Proposed fix**: if dominant component fraction >= 0.95, run BK on
the dominant component only and leave satellites alone (they have no
edges so x doesn't matter).

#### F16 (med): BK skipped on `lattice_like` -- but sprint-19h enabled it after dummies

`coordinate.py:980`: `if "lattice_like" in getattr(structure,
"topology_tags", ()): return False`. After dummy nodes are inserted,
the expanded graph isn't strictly "lattice_like" anymore (it's been
augmented with chains). But the gate uses the **original** structure
(via `self.config.structure or problem.structure`), so BK is skipped
on the expanded graph too. We may be leaving BK quality on the table
for hex/lattice graphs that get dummy nodes via 19h.

**Repro**: `hexagonal_lattice_42`. 19h reports +1.597 from dummies, but
BK is skipped. Run with BK forcibly enabled and measure delta.

#### F17 (low): structure caching -- BrandesKoepfHorizontalRefineConfig is frozen, structure is mutable

`coordinate.py:1418-1421` declares `structure:
Optional[TopologyGraphStructure] = None` on a `@dataclass(frozen=True)`
config. `TopologyGraphStructure` itself is a regular dataclass (not
frozen) with mutable tuple fields. Storing the whole structure in the
config, then sharing the config across recursive component calls, means
mutation of `topology_tags` on one call could affect others. Practically
unlikely to fire but a latent footgun.

### ec7d4db -- sprint-19h dummy nodes

#### F18 (high): `_dagua_native_layer_assignments` is overwritten in `_prepare_native_config`

`pipelines/dagua_native.py:430-436`:

```
resolved_layers = _resolve_native_layer_assignments(...)
if resolved_layers is not None:
    setattr(prepared_config, "_dagua_native_layer_assignments", resolved_layers)
```

But `prepare_pipeline_config` at `resolve.py:361` already set
`_dagua_native_layer_assignments` to the **original input**
`layer_assignments` (potentially None, potentially user-provided).
Then `_prepare_native_config` overwrites this with a freshly-computed
longest-path layering whenever the input was None or whenever the
user provided their own.

Wait -- `_resolve_native_layer_assignments` returns the input if
provided (`pipelines/dagua_native.py:115-116`), else longest-path.
And `_prepare_native_config` overwrites with the result. So if the
user passes their own `layer_assignments`, it's preserved. If they
pass None, longest-path is computed and stored. **But the user might
explicitly want layer_assignments=None to mean "don't use a layered
approach" for a non-DAG**. Now their cyclic graph gets longest-path
layers stuffed into it anyway, which `Force2DInitIfFlat` would then
miss-classify.

Actually checking again: `_resolve_native_layer_assignments` already
checks `edge_index.numel() == 0` and returns None for empty graphs.
Cyclic graphs go through `longest_path_layering` -- which what? For
a cyclic graph, longest_path is undefined; it likely produces some
arbitrary layering. Then we'd treat the cyclic graph as if it had
real layers. The downstream `_should_use_native_dummy_nodes` gate at
line 200 catches this: `if not is_directed_acyclic: return False`.
But the layer assignments are still stuffed into the config and used
by NativeEngineInit. Was previous behavior to compute layers
internally only when needed?

**Repro**: a small cyclic graph -- `small_world_100`. Trace whether
`_dagua_native_layer_assignments` was None pre-19h vs non-None
post-19h. Confirm whether NativeEngineInit's behavior changes.

#### F19 (high): dummy-graph node sizes default to (0, 0) -- repulsion sees zero-area dummies

`InsertDummyNodesConfig.dummy_width / dummy_height` default to 0. The
`OverlapProjection` and `RepulsionLoss` ops get the **original**
node_sizes via `_visible_original_pos` / `_active_node_sizes`, so
they're correctly restricted to the original-node block. But
`BarycenterReorder`, `MedianSweep`, `TransposeHeuristic`, and BK use
the **expanded** node_sizes. Zero-size dummies mean BK treats them as
points. That's correct geometrically -- a dummy is a routing pivot --
but it means the BK output places dummies at positions where any
real node could collide with a dummy at the same x. The
`OverlapProjection` step that follows runs only on original nodes,
so dummy-vs-original collisions in x are not enforced. After
`StripDummyNodes`, the original nodes might end up with reasonable
spacing but the implied edge route through the (stripped) dummy
would have crossed neighbouring nodes. Edge crossings could go up.

**Repro**: take a layered DAG with one long edge crossing several
densely-packed layers. Compare crossing_rate with
`insert_dummy_nodes=True` vs False.

#### F20 (high): expanded_graph state leaked across StripDummyNodes

`postprocess.py:1131-1153` -- the strip op truncates pos/layers/
ordering but does NOT delete `extras["expanded_graph"]`,
`extras["original_layers"]`, or `extras["original_layer_index"]`. Any
op downstream of StripDummyNodes that calls
`_visible_original_positions` or similar would still see expanded_
graph and either go into the "expanded" branch (if the size happens
to match) or into the wrong branch (if it doesn't).

In practice the next ops are AspectRatioFit and ClusterGridArrange,
neither of which inspect `extras`, so the leak is currently latent.
But it's a bug waiting to surface as soon as someone adds a
post-strip op that checks for expanded state.

**Proposed fix**: clear all three keys at the end of
`StripDummyNodes.apply`.

#### F21 (med): `_seed_expanded_positions` has O(E) Python loop with tensor indexing

`layering.py:617-628`:

```
for path in edge_paths:
    if len(path) <= 2:
        continue
    start = pos[path[0]]
    end = pos[path[-1]]
    for step, node in enumerate(path[1:-1], start=1):
        ...
        expanded[node] = start + ((end - start) * alpha)
```

For dependency_500 with say 200 long edges of average length 5 (~ 600
dummies), this is a Python loop of 200 outer + 600 inner ops with
tensor indexing each iteration. Hundreds of milliseconds easily. The
plan claims "+4.28 composite on dependency_500" -- but at what
runtime cost?

**Proposed fix**: vectorize with `torch.lerp` or pre-built segment
tensors. Or alternatively, just init dummies to the midpoint of
their layer (cheaper, less precise initialization but the gradient
loop converges anyway).

#### F22 (med): `_DUMMY_NODE_MIN_NODES = 20` is the only scale gate

`pipelines/dagua_native.py:88`: `_DUMMY_NODE_MIN_NODES = 20`. This is
the only thing that protects small DAGs from dummy-node noise. But
a 30-node DAG with one long edge that spans 15 layers gets dummy-
expanded -- and 15 dummies on a 30-node graph means 50% of the
problem is now noise. Expected to make crossings/aesthetics worse
for the original 30 nodes.

**Proposed fix**: gate on `num_dummies / num_original_nodes` ratio,
not just node count.

#### F23 (med): BK output dimension mismatch potential

`coordinate.py:1571-1579` calls `_brandes_koepf_x_positions` with
`num_nodes=active_num_nodes` and `num_original_nodes=problem.num_nodes`.
The function returns a list of length `num_nodes` (expanded). Then
`refined_pos[:, 0] = torch.tensor(x_coordinates, ...)` at
`coordinate.py:1582-1586`. `refined_pos` came from `state.pos` which
is also expanded (shape `[N_expanded, 2]`). OK, dimensions match.

But: `_should_apply_brandes_koepf_refine` was called with
`num_nodes=active_num_nodes` (expanded count). The min_layers gate at
`coordinate.py:996` uses expanded layer count. So a graph with 3
original layers but 8 expanded layers would now pass min_layers=6.
Original-only BK would have skipped this graph; expanded BK fires.
Possibly a feature, possibly a bug. Untested.

#### F24 (low): is_directed_acyclic gate falls back to is_acyclic

`pipelines/dagua_native.py:198`:

```
if not bool(getattr(structure, "is_directed_acyclic", getattr(structure, "is_acyclic", True))):
```

The fallback to `is_acyclic` (undirected) is intentional for
backwards compat with old structures that don't have the new field,
but it inverts the semantics: an undirectedly-acyclic graph (a tree
or forest) returns True for both, whereas a DAG with one undirected
cycle (e.g. converging branches) is `is_directed_acyclic=True` but
`is_acyclic=False`. Caller intent is unambiguous (DAG-only), so the
fallback could mask the field's absence. Low impact today; high
impact when older callers stop refreshing structures.

## Cross-commit interaction risks

### CC1: 19f median collapse + 19h dummy nodes

When dummies are inserted (19h), every original long edge becomes a
chain of length-1 edges. Most layers in the expanded graph will
contain BOTH original nodes AND dummy nodes, so `all(len(layer)<=1)`
in the median collapse check (`ordering.py:310`) is FALSE for most
expanded graphs. The median-collapse path effectively turns off
when 19h is active. Fine for the common case, but the very chains
that 19f's collapse was designed to "fix" (`dense_pair_50`) might
not benefit from the collapse if 19h decides to expand long edges.
Did the +8.62 win on dense_pair_50 happen pre-19h or post-19h?
Worth re-measuring -- the wins might cancel each other out.

### CC2: 19d per-component + 19g BK

When decomposition fires, BK runs once per child (inside the per-
child build_dagua_pipeline call) with the child's structure. The
child's classification is computed fresh from local edges + local
(gappy) layers. `_should_apply_brandes_koepf_refine`'s component
check sees `[child_num_nodes]` -- one component -- always passes.
But the BK layered-forward check (`_has_strict_forward_layering`)
uses the gappy layer ids; an edge spanning original layers 5 -> 9
in the child has `layer_delta = 4 > 0`, so passes. So BK sees the
child as if it had 10 layers (max+1) when really it has 3. The
`min_layers >= 6` gate fires falsely. A child with 3 actual layers
gets BK applied as if it were a 10-layer DAG. Probably benign but
not what the gate intended.

### CC3: 19h dummies + 19d decomposition + 19e topology aspect

Worst-case: a disconnected graph triggers 19d, each child component
is independently classified, gets a topology-aware aspect target
(via `_dagua_native_target_aspect`), gets dummy-expanded if it's a
clean DAG, runs BK on the expanded graph, has overlap projection on
original-only positions, strips dummies, applies child aspect fit,
then the parent runs ANOTHER aspect fit (with default 0.25, ignoring
all the topology work from below). Net effect: child topology
optimization is partially undone by parent default aspect.

Also: `_dagua_native_use_dummy_nodes` is set in `_prepare_native_config`
based on the structure that was set via `prepare_pipeline_config`.
For the per-component path, each child re-prepares, so each child
independently decides whether to use dummies. Children that ARE
dummy-expanded run the gradient core on the expanded graph; children
that AREN'T run on the original. After tiling, the result tensors
have heterogeneous histories. If two children happened to differ in
their use_dummy decision, the per-child geometry would differ in
quality, and the tiled result would look uneven.

### CC4: gappy layers (19d) + dense_dag tag (19e)

`is_dense_dag` requires `num_layers >= int(0.6 * num_nodes)`. For a
child whose parent-sliced layers are e.g. [5, 7, 9, 11, 12] (gappy),
`num_layers` from `_analyze_layers` is 5 (only counts nonempty
layers). With 5 nodes, 0.6*5 = 3, so 5 >= 3 -> is_dense_dag tag
fires. Then `resolve_topology_aware_aspect` returns `(0.05, 1.0)`
and this child gets a near-zero (extremely tall, very thin) aspect.
For a child that was just 5 nodes scattered across 5 layers, this
is wrong-headed -- a vertical column of 5 dots.

## Latent silent-degradation paths

### LSD1: GPU paths silently fall to CPU

`graph_classify.py:277` (sprint-19e): `prefer_device = "cuda" if
torch.cuda.is_available() and torch.cuda.device_count() > 0 else
"cpu"`. The `device_count() > 0` extra check was added 19e. Same
pattern at `dagua/utils.py:1472-1473`. Both are silent fallbacks --
no log when CUDA is "available" but no device exists (e.g.
CUDA_VISIBLE_DEVICES=""). The benchmarks in
`eval_output/native_algo/holdout_v1/` were likely run with
`CUDA_VISIBLE_DEVICES=""` -- so they hit CPU paths. Performance
numbers won't transfer to a GPU-enabled deployment without
re-measurement.

### LSD2: median/transpose `cpu()` conversions inside the gradient hot path

`ordering.py:323-325`:

```
order_values = ordering_cpu[members.cpu()]
ordered_members_cpu = members.cpu()[torch.argsort(order_values, stable=True)]
ordered_members = ordered_members_cpu.to(device=pos_new.device)
```

These `.cpu()` / `.to(device=...)` round-trips happen per layer per
pass per op. For a CUDA run with 20 layers and 4 median + 8 transpose
passes, that's 20 * 12 = 240 cpu/gpu round trips on the median path
alone. Synchronous, blocks the CUDA stream.

### LSD3: `_weak_component_sizes` allocates a Python adjacency list

`coordinate.py:826-867` (sprint-19g): builds `adjacency: List[List[int]]
= [[] for _ in range(num_nodes)]` and fills it via `for source,
target in edge_index.t().tolist()`. For dependency_500 with ~600
expanded edges, the `.tolist()` materializes a 1200-int Python list
and the appends are Python-loop. Plus a stack-based DFS afterward.
All to compute one boolean. Cached nowhere.

### LSD4: `_should_use_native_dummy_nodes` recomputes structure twice per call

19h's `_prepare_native_config` reads `structure = getattr(
prepared_config, "_dagua_native_structure", None)` (already computed
by `prepare_pipeline_config`). Good. But then
`_should_use_native_dummy_nodes` calls back into stuff that re-checks
structure. And per-component recursion re-classifies children twice:
once in `_extract_component_problem` (line 749) and once in
`_prepare_native_config` -> `prepare_pipeline_config`. Duplicate work.

## Tests we should add to lock in the wins

### T1: Regression test for child layer relabeling
After F1 fix, assert that a 3-component graph with parent layers
{5, 7, 9} has each child see layers starting at 0 with `num_layers`
matching nonempty layer count.

### T2: Outer aspect fit shouldn't override inner
For a graph with 2 disconnected wide-layered components (target
aspect 0.85), assert the final aspect is closer to 0.85 than to 0.25
after tiling. Currently fails with the F2 design flaw.

### T3: `_count_components` correctness on dense disconnected graphs
Add the F9 reproducer as a unit test. `classify_graph` on a graph
with two dense components must report `num_components=2`.

### T4: MedianSweep doesn't collapse chains
Layout a chain of 30 nodes through `algorithm="dagua"`. Assert that
final x-spread is non-trivial (e.g. `pos[:,0].std() > 0.1`). Will
fail today because of F10.

### T5: StripDummyNodes clears expanded_graph extras
After running the pipeline, assert
`state.extras.get("expanded_graph") is None` post-strip.

### T6: BK gate on `[N-2, 1, 1]` shape
F15: a graph with one big component and two satellites should still
run BK on the dominant component. Today it skips BK entirely.

### T7: Pin / align preservation through median+transpose+BK
A graph with 3 hard pins -- verify pins are still respected after
the rearrangement ops. F11 says they may not be.

### T8: Sufficient timing test
Time the full native pipeline on dependency_500, hexagonal_lattice_42,
and `extreme_mixed_width_transformer` with all 19f-h flags off vs
all on. The h2h composite gains were measured but not the runtime
cost. Establish a budget (e.g. <10% wall-clock regression for
500-node DAG).

### T9: Per-component decomposition with dropped align groups
Test that a `g.align(["a","b","c"])` group spanning components either
(a) blocks decomposition, or (b) emits a warning when dropped.
Documents F4 behavior.

### T10: cyclic graph with `_dagua_native_layer_assignments` not None
F18: trace whether `prepare_pipeline_config` + `_prepare_native_config`
ever stuffs a longest-path layering into a cyclic graph's config and
whether anything downstream uses it inappropriately.

## Implementation order (what to fix first)

1. **F9** (`_count_components` returns 1 for dense disconnected) --
   this silently disables sprint-19d for dense disconnected DAGs.
   One-liner fix: don't short-circuit the union-find on dense graphs.

2. **F10** (median collapse on chains) -- one-line gate to skip
   collapse when graph is a single chain. Direct regression
   protection on chain-family graphs (chain_25k, chain test
   suite).

3. **F1 / F18 / CC4** (gappy layer relabeling) -- relabel child
   layers in `_extract_component_problem`. Three observed bugs
   collapse to one fix.

4. **F2** (outer aspect fit override) -- pass `target_aspect` from
   `prepared_config` to the outer AspectRatioFit, or skip it
   entirely.

5. **F20 / F12** (extras leak after strip) -- clear extras in
   StripDummyNodes; this also defuses F12.

6. **F15** (BK gate exact-match) -- relax to "if dominant component
   >= 0.95, run BK on dominant component".

7. **F4** (align groups dropped silently) -- add guard or warning.

8. **F19 / F23** (dummy-graph node sizes + BK on expanded) -- needs
   measurement first; fix may mean expanding overlap projection to
   include dummy positions (which is what BK needs to be effective
   for routing).

9. Performance hygiene (F11/LSD2/LSD3/F21/F14) -- batch as one
   sprint after the correctness fixes land.

10. F6 (dead rank_sep_multiplier) -- delete or wire up. Cosmetic.

The biggest threat to "don't regress strengths" is F1 + CC2 + CC4.
Those three together produce wrong-but-not-crashing behavior on the
disconnected graphs that 19d was designed to fix. The h2h numbers in
the commit messages may not be reliable until F1 is fixed and re-
benchmarked, because the test fixtures might have happened to land in
configurations where the gappy layers didn't matter.

The biggest threat to future ambition is F11 -- the median+transpose+BK
discrete polish phase happily runs after the gradient solver, and if a
user adds new soft constraints (pins, alignment, custom losses), those
constraints may be silently overridden by the discrete polish.
Sprint-20 should establish a "discrete polish respects soft
constraints" contract as a precondition for any further expansion of
the polish phase.
