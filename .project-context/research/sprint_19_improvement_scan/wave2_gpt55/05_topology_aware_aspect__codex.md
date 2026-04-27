# Wave 2 Plan: Topology-aware Aspect Ratio and Spacing for `dagua_native`

## Section 1 -- Design

### Recommendation

Keep the Sprint 18h default target aspect ratio of `0.25` as the baseline and
only override it for a narrow, topology-verified subset:

| Proposed topology tag | Trigger | Target aspect | Optional `rank_sep` multiplier |
| --- | --- | --- | --- |
| `DEFAULT_KEEP` | Everything not explicitly widened | `0.25` | `1.00` |
| `PLANAR_DAG` | Planar-ish DAGs that are not regular meshes | `0.45` | `1.00` |
| `LATTICE_LIKE` | Grid / lattice / regular planar mesh | `0.60` | `0.75` |
| `WIDE_LAYERED` | Two-layer / very wide layered DAGs | `0.85` | `1.00` |
| `DENSE_DAG` | Deep, narrow, high-density DAGs like `dense_pair_50` | `0.45` | `0.90` in sweep only |

This is intentionally conservative. The key safety property is unchanged:
`0.25` remains the default for the DAG families that currently generate the
largest mean-composite win, including `random_dag_200`, `random_dag_50`,
`org_chart_deep`, `hub_fanout_label_skew`, and `org_chart_1_5_4_8`.

### Why the current classifier is not yet sufficient

The task says to use `classify_graph()`, but the current implementation in
`dagua/layout/graph_classify.py:210-301` does not expose the topology slices
needed for this policy:

1. The enum at `dagua/layout/graph_classify.py:19-29` only provides
   `GENERAL`, `TREE`, `FOREST`, `CHAIN`, `BIPARTITE_DAG`, `WIDE_LAYERED`, and
   `GRID`. There is no `PLANAR`, `LATTICE`, or `DENSE_DAG` family.
2. The current `GraphStructure` dataclass at
   `dagua/layout/graph_classify.py:31-42` does not expose `max_layer_width`,
   `edge_density`, or any tag set.
3. `_count_components_and_acyclic()` at
   `dagua/layout/graph_classify.py:91-145` treats undirected cycles as
   `is_acyclic=False`. That is too coarse for this use case. Many true DAGs
   and mesh-like DAGs come back as `is_acyclic=False`, so a naive gate would
   misclassify the exact graphs we care about.

Empirical outputs from the live code confirm the problem:

| Graph | Current `family` | `num_components` | `num_layers` | `avg_layer_width` | `is_planar_hint` | `is_acyclic` | Proposed policy bucket | Proposed target |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| `hexagonal_lattice_42` | `GENERAL` | 1 | 12 | 3.50 | `True` | `False` | `LATTICE_LIKE` | `0.60` |
| `sierpinski_42` | `GENERAL` | 1 | 23 | 1.83 | `True` | `False` | `PLANAR_DAG` | `0.45` |
| `random_bipartite_60` | `BIPARTITE_DAG` | 1 | 2 | 30.00 | `True` | `False` | `WIDE_LAYERED` | `0.85` |
| `complete_bipartite_8x12` | `BIPARTITE_DAG` | 1 | 2 | 10.00 | `False` | `False` | `WIDE_LAYERED` | `0.85` |
| `dense_pair_50` | `GENERAL` | 1 | 50 | 1.00 | `False` | `False` | `DENSE_DAG` | `0.45` |
| `random_dag_200` | `GENERAL` | 202 | 10 | 38.30 | `True` | `False` | `DEFAULT_KEEP` | `0.25` |
| `random_dag_50` | `GENERAL` | 52 | 12 | 8.08 | `True` | `False` | `DEFAULT_KEEP` | `0.25` |
| `org_chart_deep` | `TREE` | 1 | 6 | 13.17 | `True` | `True` | `DEFAULT_KEEP` | `0.25` |
| `hub_fanout_label_skew` | `GENERAL` | 1 | 5 | 2.00 | `True` | `False` | `DEFAULT_KEEP` | `0.25` |
| `wide_1_100_1` | `GENERAL` | 1 | 3 | 34.00 | `True` | `False` | `WIDE_LAYERED` | `0.85` |
| `grid_20x20` | `GENERAL` | 1 | 39 | 10.26 | `True` | `False` | `LATTICE_LIKE` | `0.60` |
| `small_world_100` | `WIDE_LAYERED` | 1 | 1 | 100.00 | `True` | `False` | `DEFAULT_KEEP (cyclic guard)` | `0.25` |

Three regression guards fall directly out of that table:

1. Do not widen on raw `is_planar_hint=True`. `random_dag_200` is a top win
   and currently comes back planar-hinted.
2. Do not widen on raw `family == WIDE_LAYERED`. `small_world_100` is cyclic
   and would be a false positive.
3. Do not widen on raw width metrics alone. `hub_fanout_label_skew` is already
   a top-5 dagua win and should remain on `0.25`.

### Required classifier extension

The smallest safe change is not to replace `classify_graph()`, but to extend it
with enough data to derive topology-aware aspect policy without changing the
existing fast-path families. I recommend the following additions in
`dagua/layout/graph_classify.py`:

1. Add a directed acyclicity signal that is separate from the current
   undirected-cycle test.
   Location: replace or split `_count_components_and_acyclic()` at
   `dagua/layout/graph_classify.py:91-145`.
   New helpers:
   - `_count_components(edge_index: torch.Tensor, num_nodes: int) -> int`
   - `_is_directed_acyclic(edge_index: torch.Tensor, num_nodes: int) -> bool`

2. Extend `GraphStructure` at `dagua/layout/graph_classify.py:31-42` with:
   - `max_layer_width: int`
   - `layer_width_cv: float`
   - `edge_to_node_ratio: float`
   - `is_directed_acyclic: bool`
   - `topology_tags: tuple[str, ...]`

3. Compute tags in `classify_graph()` at `dagua/layout/graph_classify.py:210-301`.
   Proposed tags:
   - `bipartite_dag`
   - `wide_layered`
   - `planar_dag`
   - `lattice_like`
   - `dense_dag`

### Exact trigger rules

The goal is not a clever formula. The goal is a conservative switch that only
widens graphs that are currently harmed by the hard-coded `0.25`.

Recommended tag rules:

1. `wide_layered`
   Trigger:
   - `structure.is_directed_acyclic`
   - and either `structure.family == GraphFamily.BIPARTITE_DAG`
   - or `structure.family == GraphFamily.WIDE_LAYERED`
   - or `structure.num_layers <= 3 and structure.max_layer_width >= 24`

2. `lattice_like`
   Trigger:
   - `structure.is_directed_acyclic`
   - `structure.is_planar_hint`
   - `2 <= structure.max_degree <= 6`
   - `1.0 <= structure.edge_to_node_ratio <= 2.2`
   - `structure.num_layers >= 5`
   - `structure.layer_width_cv <= 0.45`

   This is intentionally strict. It catches the regular mesh family
   (`hexagonal_lattice_42`, `triangular_lattice_36`, `grid_*`) and avoids
   sparse random DAGs.

3. `planar_dag`
   Trigger:
   - `structure.is_directed_acyclic`
   - `structure.is_planar_hint`
   - `structure.max_degree <= 4`
   - `structure.num_layers >= 5`
   - not `lattice_like`

   This gives `sierpinski_42`, `outerplanar_dag_20`, and `planar_60` a modest
   widening without pushing them all the way to the mesh target.

4. `dense_dag`
   Trigger:
   - `structure.is_directed_acyclic`
   - `structure.edge_to_node_ratio >= 3.5`
   - `structure.max_layer_width <= 3`
   - `structure.num_layers >= int(0.6 * num_nodes)`

   That is intentionally narrow. In the first rollout it is mainly a
   `dense_pair_50` fix. This should not become a broad bucket without a sweep.

### Why the aspect target should be the primary lever

There are two candidate mechanisms:

1. Change `AspectRatioFit` target.
2. Change spacing (`rank_sep` / `node_sep`) so the optimizer lands wider
   before postprocess.

The safer first cut is `AspectRatioFit` target, because it is:

- topology-gated
- post-hoc
- trivially reversible
- less entangled with the optimization landscape

The current spacing ratio is set at `node_sep=70.0` and `rank_sep=240.0` in
`dagua/config.py:64-65`. The adaptive dispatch in
`dagua/layout/resolve.py:311-318` carries that ratio almost unchanged for
benchmark-scale graphs. That 3.43x vertical bias is part of the problem, but it
is also part of the reason the mean improved after Sprint 18h. I would not
retune spacing globally.

I do recommend one optional spacing gate:

- for `lattice_like` only, apply `rank_sep *= 0.75` after adaptive spacing
  and before the pipeline is built

That moves the rank/node ratio from `240/70 = 3.43` to `180/70 = 2.57` on the
default case, which is much closer to the square-ish geometry these graphs want.
I would not enable the `dense_dag` spacing multiplier in the first patch; keep
it in the sweep plan only.

## Section 2 -- Exact code patches

### Patch A: classifier metadata

File: `dagua/layout/graph_classify.py`

Patch points:

- `19-29`: keep the existing `GraphFamily` enum intact; do not overload it with
  too many new structural meanings.
- `31-42`: extend `GraphStructure`.
- `91-145`: split undirected connectivity from directed acyclicity.
- `183-207`: upgrade `_analyze_layers()` to also return `max_layer_width` and
  `layer_width_cv`.
- `210-301`: derive `topology_tags`.

Recommended shape:

```python
@dataclass(frozen=True)
class GraphStructure:
    family: GraphFamily
    num_components: int
    max_degree: int
    num_layers: int
    avg_layer_width: float
    is_planar_hint: bool
    is_acyclic: bool = True
    max_layer_width: int = 0
    layer_width_cv: float = 0.0
    edge_to_node_ratio: float = 0.0
    is_directed_acyclic: bool = True
    topology_tags: tuple[str, ...] = ()
```

This preserves existing call sites while giving `resolve.py` enough signal to
make a conservative decision.

### Patch B: resolve aspect target and optional spacing upstream

File: `dagua/layout/resolve.py`

Patch points:

- add a helper immediately after `adaptive_spacing()` at
  `dagua/layout/resolve.py:112-126`
- call it in `prepare_pipeline_config()` after the current spacing resolution at
  `dagua/layout/resolve.py:311-318`

Recommended helper:

```python
def resolve_topology_aware_aspect(
    structure: Optional[GraphStructure],
) -> tuple[float, float]:
    """Return ``(target_aspect, rank_sep_multiplier)`` for one graph."""
    if structure is None:
        return 0.25, 1.0

    tags = set(structure.topology_tags)
    if "lattice_like" in tags:
        return 0.60, 0.75
    if "planar_dag" in tags:
        return 0.45, 1.0
    if "wide_layered" in tags or structure.family == GraphFamily.BIPARTITE_DAG:
        return 0.85, 1.0
    if "dense_dag" in tags:
        return 0.45, 1.0
    return 0.25, 1.0
```

Then in `prepare_pipeline_config()`:

1. Run `adaptive_spacing()` as today.
2. Resolve topology-aware aspect from the classified structure.
3. Apply the rank multiplier to `resolved_rank_sep`.
4. Stash both on the config:
   - `_dagua_native_target_aspect`
   - `_dagua_native_rank_sep_multiplier`

This keeps the policy in one place and avoids teaching `AspectRatioFit` how to
classify graphs.

### Patch C: thread the resolved target into the pipeline

File: `dagua/layout/ops/pipelines/dagua_native.py`

Patch points:

- pipeline composition at `dagua/layout/ops/pipelines/dagua_native.py:427-447`
- `LayoutProblem` construction at `dagua/layout/ops/pipelines/dagua_native.py:511-520`

Changes:

1. Replace the current bare instantiation at line `439`:

```python
AspectRatioFit(AspectRatioFitConfig()),
```

with:

```python
AspectRatioFit(
    AspectRatioFitConfig(
        target_aspect=getattr(config, "_dagua_native_target_aspect", None),
    )
),
```

2. Pass the already-classified structure into `LayoutProblem` so downstream ops
   can read it later if needed:

```python
problem = LayoutProblem(
    ...,
    structure=getattr(prepared_config, "_dagua_native_structure", None),
    ...
)
```

That second change is worth doing even if the first rollout resolves aspect in
`resolve.py`. It removes a current information gap. `LayoutProblem` already has
the field at `dagua/layout/ops/state.py:134-150`; it is just not populated by
the native pipeline today.

### Patch D: keep `AspectRatioFit` simple, but fix the stale docs

File: `dagua/layout/ops/postprocess.py`

Patch points:

- `AspectRatioFitConfig` docstring at `dagua/layout/ops/postprocess.py:514-530`
- `AspectRatioFit.apply()` implementation at `dagua/layout/ops/postprocess.py:565-650`

The implementation at `621-623` currently does:

```python
target = self.config.target_aspect
if target is None:
    target = 0.25
```

That behavior is fine for the first rollout. The important patch here is the
docstring: lines `519-524` still describe an old "sqrt(N), clamped to
`[0.6, 2.5]`" default that no longer exists. That comment is stale and should
be updated to say:

- `target_aspect=None` falls back to the resolved upstream default
- the native pipeline usually passes an explicit target
- the pure fallback remains `0.25`

I would not add callback logic to `AspectRatioFit` in the first cut. Upstream
resolution is easier to test and easier to rollback.

### Optional Patch E: lattice-only spacing gate

File: `dagua/layout/resolve.py`

Patch point:

- same block as Patch B, after the existing adaptive spacing at `311-318`

Rule:

- only if `lattice_like` is present
- only after `adaptive_spacing`
- multiply `resolved_rank_sep` by `0.75`

I would not change `node_sep` in the first rollout. The issue is not that mesh
graphs need denser horizontal packing; it is that they are being stretched too
hard vertically.

## Section 3 -- Interaction with other wave-2 patches

### #1 BK coordinate assignment

If Wave 2 patch #1 wires in Brandes-Kopf from
`dagua/layout/ops/coordinate.py:1249-1307`, it should run before
`AspectRatioFit`. The sequencing I would want is:

`BarycenterReorder -> TransposeHeuristic -> BrandesKopf4Pass -> OverlapProjection -> AspectRatioFit`

Rationale:

- BK is the discrete x-coordinate solver.
- Aspect fit is only a last-mile bbox correction.
- If BK already lands near the topology target, the current `0.55` tolerance in
  `dagua/layout/ops/postprocess.py:533-537` means the aspect op will often
  no-op.

That is the correct relationship. BK should own local x structure; aspect fit
should only own coarse global anisotropy.

### #2 Dummy nodes

If Wave 2 patch #2 inserts dummy nodes, they must be stripped before aspect
fit. The relevant stripping op already exists as `StripDummyNodes` in
`dagua/layout/ops/postprocess.py:1092+`.

Required order:

`InsertDummyNodes -> ordering/coordinate work on expanded graph -> StripDummyNodes -> OverlapProjection -> AspectRatioFit`

Reason:

- dummy chains change the bbox
- they are routing aids, not user-visible nodes
- the aspect target should be computed from the real-node layout

### #3 Median / transpose

Median or transpose-based ordering passes should run before aspect fit for the
same reason as BK: they change local x ordering, and aspect fit should not
fight that work. It should see the final discrete order and only widen or
heighten if the entire solved layout is still outside the tolerated band.

### #4 Per-component layout

If per-component layout lands in Wave 2, aspect handling should happen twice:

1. Component-local solve:
   - classify each component
   - resolve a component-local aspect target
   - apply component-local aspect fit after component-local overlap projection

2. Tiled-bbox global solve:
   - after the components are placed into their tile grid, run one more global
     `AspectRatioFit`
   - but use a softened target:
     `clamp(weighted_mean(component_targets), 0.25, 0.60)`
   - and a looser tolerance, e.g. `0.70`

This prevents a single lattice component from forcing a whole disconnected
collage into a square-ish global frame, while still allowing an obviously
vertical or obviously horizontal component tiling to be trimmed.

## Section 4 -- Regression safety

### Assumption about the benchmark roster

The task text says "93 benchmark graphs". The current local cache under
`eval_output/variant_bench_full/positions/` exposes `97` graphs with
`N <= 500`. I therefore use the local 97-graph roster below and treat the
prompt's 93-graph count as stale. If the active benchmark list is trimmed back
to 93, the policy itself does not change; only four rows disappear.

### The non-negotiable keep list

These must remain on `0.25` in the first rollout:

| Graph | Reason |
| --- | --- |
| `random_dag_200` | Biggest win; current classifier is a false positive on `planar_hint` |
| `org_chart_deep` | Tree family already stable |
| `random_dag_50` | Large win; no mesh / bipartite evidence |
| `hub_fanout_label_skew` | Large win; wide-looking but not a target family |
| `org_chart_1_5_4_8` | Tree family already stable |

### Full local roster assignment table

| Graph | Proposed bucket | Target |
| --- | --- | ---: |
| asymmetric_hourglass_hub | DEFAULT_KEEP | 0.25 |
| ba_500 | DEFAULT_KEEP | 0.25 |
| binary_tree | DEFAULT_KEEP | 0.25 |
| bipartite_4_3_4 | WIDE_LAYERED | 0.85 |
| braided_feedback_tails | DEFAULT_KEEP | 0.25 |
| broken_symmetry_residual_pair | DEFAULT_KEEP | 0.25 |
| center_port_backedge_hub | DEFAULT_KEEP | 0.25 |
| chung_lu_150 | DEFAULT_KEEP | 0.25 |
| citation_dag_300 | DEFAULT_KEEP | 0.25 |
| cluster_member_style_stress | DEFAULT_KEEP | 0.25 |
| clustered_longlabel_handoffs | DEFAULT_KEEP | 0.25 |
| clustered_medium_5x20 | DEFAULT_KEEP | 0.25 |
| complete_bipartite_8x12 | WIDE_LAYERED | 0.85 |
| compound_10x20 | DEFAULT_KEEP | 0.25 |
| compound_dag_5x30 | DEFAULT_KEEP | 0.25 |
| deep_chain_20 | DEFAULT_KEEP | 0.25 |
| dense_pair_50 | DENSE_DAG | 0.45 |
| densenet_block | DEFAULT_KEEP | 0.25 |
| dependency_500 | DEFAULT_KEEP | 0.25 |
| dependency_graph_100 | DEFAULT_KEEP | 0.25 |
| disconnected_encoder_residual | DEFAULT_KEEP | 0.25 |
| disconnected_label_cycle_collage | DEFAULT_KEEP | 0.25 |
| edge_label_braid | DEFAULT_KEEP | 0.25 |
| er_100 | DEFAULT_KEEP | 0.25 |
| er_500 | DEFAULT_KEEP | 0.25 |
| extreme_mixed_width_transformer | DEFAULT_KEEP | 0.25 |
| grid_20x20 | LATTICE_LIKE | 0.60 |
| grid_5x5 | LATTICE_LIKE | 0.60 |
| grid_rect_6x8 | LATTICE_LIKE | 0.60 |
| heavy_tail_weights_50 | DEFAULT_KEEP | 0.25 |
| hexagonal_lattice_42 | LATTICE_LIKE | 0.60 |
| hierarchical_residual_stage | DEFAULT_KEEP | 0.25 |
| hub_and_spoke_3x20 | DEFAULT_KEEP | 0.25 |
| hub_fanout_label_skew | DEFAULT_KEEP | 0.25 |
| hub_skip_superfan | DEFAULT_KEEP | 0.25 |
| hub_spoke_10x20 | DEFAULT_KEEP | 0.25 |
| hub_spoke_5x50 | DEFAULT_KEEP | 0.25 |
| inception_block | DEFAULT_KEEP | 0.25 |
| interleaved_cluster_crosstalk | DEFAULT_KEEP | 0.25 |
| kitchen_sink_hybrid_net | DEFAULT_KEEP | 0.25 |
| kitchen_sink_platform_graph | DEFAULT_KEEP | 0.25 |
| linear_3layer_mlp | DEFAULT_KEEP | 0.25 |
| long_range_residual_ladder | DEFAULT_KEEP | 0.25 |
| long_skip_only_24 | DEFAULT_KEEP | 0.25 |
| mixed_width_labels | DEFAULT_KEEP | 0.25 |
| moe_router_sparse | DEFAULT_KEEP | 0.25 |
| multi_component_80 | DEFAULT_KEEP | 0.25 |
| multiscale_skip_cascade | DEFAULT_KEEP | 0.25 |
| nested_cluster_label_stack | DEFAULT_KEEP | 0.25 |
| nested_shallow_enc_dec | DEFAULT_KEEP | 0.25 |
| org_chart_1_5_4_8 | DEFAULT_KEEP | 0.25 |
| org_chart_deep | DEFAULT_KEEP | 0.25 |
| outerplanar_dag_20 | PLANAR_DAG | 0.45 |
| parallel_cycles_4x5 | DEFAULT_KEEP | 0.25 |
| parallel_multiedge_bundle | DEFAULT_KEEP | 0.25 |
| petersen_10 | DEFAULT_KEEP | 0.25 |
| planar_60 | PLANAR_DAG | 0.45 |
| powerlaw_500 | DEFAULT_KEEP | 0.25 |
| protein_ppi_200 | DEFAULT_KEEP | 0.25 |
| ragged_feature_pyramid | DEFAULT_KEEP | 0.25 |
| random_bipartite_60 | WIDE_LAYERED | 0.85 |
| random_dag_200 | DEFAULT_KEEP | 0.25 |
| random_dag_50 | DEFAULT_KEEP | 0.25 |
| real_football_115 | DEFAULT_KEEP | 0.25 |
| real_karate_34 | DEFAULT_KEEP | 0.25 |
| real_lesmis_77 | DEFAULT_KEEP | 0.25 |
| recurrent_feedback_cell | DEFAULT_KEEP | 0.25 |
| regular_3_30 | DEFAULT_KEEP | 0.25 |
| regular_4_40 | DEFAULT_KEEP | 0.25 |
| residual_block | DEFAULT_KEEP | 0.25 |
| resnet_stack_4x16 | DEFAULT_KEEP | 0.25 |
| rgg_100 | DEFAULT_KEEP | 0.25 |
| rgg_500 | DEFAULT_KEEP | 0.25 |
| sbm_4x30 | DEFAULT_KEEP | 0.25 |
| sbm_5x50 | DEFAULT_KEEP | 0.25 |
| scale_free_ba_120 | DEFAULT_KEEP | 0.25 |
| shape_and_routing_matrix | DEFAULT_KEEP | 0.25 |
| sierpinski_42 | PLANAR_DAG | 0.45 |
| small_label_storm | DEFAULT_KEEP | 0.25 |
| small_world_100 | DEFAULT_KEEP | 0.25 |
| small_world_500 | DEFAULT_KEEP | 0.25 |
| sparse_pair_50 | DEFAULT_KEEP | 0.25 |
| tl_cnn_small | DEFAULT_KEEP | 0.25 |
| tl_mlp_3layer | DEFAULT_KEEP | 0.25 |
| tl_resnet_2block | DEFAULT_KEEP | 0.25 |
| tl_transformer_1layer | DEFAULT_KEEP | 0.25 |
| transformer_full_4h_2l | DEFAULT_KEEP | 0.25 |
| transformer_layer | DEFAULT_KEEP | 0.25 |
| triangular_lattice_36 | LATTICE_LIKE | 0.60 |
| unet_small | DEFAULT_KEEP | 0.25 |
| weighted_chain_20 | DEFAULT_KEEP | 0.25 |
| weighted_clusters_3x10 | DEFAULT_KEEP | 0.25 |
| weighted_karate_34 | DEFAULT_KEEP | 0.25 |
| wide_1_100_1 | WIDE_LAYERED | 0.85 |
| wide_3_50_3 | WIDE_LAYERED | 0.85 |
| wide_single_layer_1_50_1 | WIDE_LAYERED | 0.85 |
| width_skew_late_merge | DEFAULT_KEEP | 0.25 |

This table is intentionally narrow:

- only 11 of 97 local graphs get a wider target
- only 5 of 97 get the stronger lattice target
- only 1 of 97 gets the dense-DAG target

That is the right rollout profile if the goal is to preserve the mean-composite
win.

## Section 5 -- Tests

### Unit tests

1. `tests/test_layout/test_engine.py`
   Add a classifier regression near the current classification block at
   `tests/test_layout/test_engine.py:1208-1261`.

   Proposed assertions:
   - `hexagonal_lattice_42` gets `topology_tags` containing `lattice_like`
   - `sierpinski_42` gets `planar_dag` but not `lattice_like`
   - `small_world_100` does not get an aspect-widening tag because
     `is_directed_acyclic=False`

2. `tests/test_ops_postprocess.py`
   Add a focused test for `AspectRatioFit` target selection:
   - resolved target `0.60` widens a tall layout
   - resolved target `0.25` leaves the same layout unchanged

3. `tests/test_layout/test_engine.py` or a new
   `tests/test_layout/test_resolve_aspect_policy.py`
   Add `prepare_pipeline_config()` tests:
   - `hexagonal_lattice_42` resolves `_dagua_native_target_aspect == 0.60`
   - `random_dag_200` resolves `_dagua_native_target_aspect == 0.25`
   - lattice path applies `rank_sep *= 0.75` if the optional spacing gate is enabled

### Integration tests

1. `tests/test_layout/test_engine.py`
   Build `hexagonal_lattice_42`, run the native pipeline, and assert the final
   bbox aspect is approximately the intended band:
   - target `0.60`
   - accepted final ratio in `[0.40, 0.85]`

2. `tests/test_layout/test_engine.py`
   Build `random_dag_200`, run the same pipeline, and assert that the resolved
   target remains `0.25`. This is the important regression test, not the exact
   measured bbox.

3. If the rank-separation gate ships, add a direct test that the lattice path
   reduces `resolved_rank_sep` while the random-DAG path does not.

### Commands

If this plan is implemented, the mandatory validation path should stay aligned
with the project guide:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

## Section 6 -- Expected impact

### Target graphs

I would budget the expected improvement as follows:

| Graph | Current score | Expected score after patch | Notes |
| --- | ---: | ---: | --- |
| `hexagonal_lattice_42` | 82.42 | `85.5` to `87.0` | biggest direct beneficiary; target `0.60` plus optional lower `rank_sep` |
| `sierpinski_42` | 78.35 | `80.5` to `82.0` | modest widen only; `0.45` is safer than mesh target |
| `dense_pair_50` | 71.81 | `73.5` to `75.0` | milder gain; aspect target helps more than spacing here |

### Why the gain should be neutral-to-positive on mean

The mean-composite risk is low because:

1. The default remains `0.25`.
2. The keep-list winners remain on `0.25`.
3. The strongest target (`0.85`) is only applied to obvious wide-layer cases.
4. The lattice-only spacing change is optional and local.

I would expect the overall suite mean to land somewhere between flat and
`+0.2` to `+0.5`, with the more important outcome being that the planar-mesh
tail gets noticeably shorter without undoing the Sprint 18h win.

## Section 7 -- Rollback and sweep plan

### Rollback order

If the suite mean regresses, rollback should happen in this order:

1. Disable the lattice `rank_sep *= 0.75` gate, keep the topology-aware aspect
   target.
2. Remove `dense_dag` from the widening set and keep only
   `lattice_like`, `planar_dag`, and `wide_layered`.
3. Remove `planar_dag` and keep only explicit lattice + explicit wide-layered.
4. Revert fully to `0.25` global target.

That rollback ordering matters. The postprocess target change is much safer than
spacing retuning, and the dense-DAG bucket is the most speculative slice.

### Sweep protocol

Automated sweep grid:

- `lattice_target in {0.50, 0.60, 0.70}`
- `planar_target in {0.40, 0.45, 0.50}`
- `wide_target in {0.70, 0.85, 1.00}`
- `lattice_rank_sep_multiplier in {1.00, 0.85, 0.75}`
- optional `dense_target in {0.35, 0.45, 0.55}`

Acceptance criteria:

1. Suite mean composite must be `>=` current baseline.
2. No regression larger than `-0.5` on any of the top-5 dagua wins.
3. `hexagonal_lattice_42` and `sierpinski_42` together should gain at least
   `+4` composite.
4. `random_dag_200` target resolution must remain `0.25`.
5. `small_world_100` and `small_world_500` must stay on the default target.

### Final recommendation

Implement the first cut as:

1. classifier metadata extension
2. resolve-time target selection
3. explicit target threading into `AspectRatioFit`
4. optional lattice-only `rank_sep` reduction behind the same tag gate

Do not broaden the widened set beyond the 11 rows in the table until the sweep
shows that the current mean-composite win survives.
