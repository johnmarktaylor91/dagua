# Area A -- Lattice Grid Snap (codex)

## TL;DR

- **Single biggest call:** implement lattice snap as an additional
  post-pipeline, metric-scored polish candidate, not as a replacement for the
  native layered/hybrid pipeline. The existing `_best_of_polish` pattern is the
  right safety model.
- The winning primitive is **layered oblique-grid projection**: preserve the
  current topological layer order, fit a small set of grid phases/steps from the
  polished positions, snap to the nearest collision-free grid sites, then accept
  only if `composite(full(...))` improves by at least the current 0.5 margin.
- Do **not** gate on `lattice_like` alone. At HEAD, `hexagonal_lattice_42` is
  tagged `lattice_like`, but `petersen_10` is also tagged because the cheap
  planar hint passes even though exact planarity is false. Require exact
  `structure.is_planar is True`.
- Scope the first implementation to `hexagonal_lattice_42`. It has the cleanest
  headroom: current dagua CV `0.456` vs dot `0.099`, while dagua already has
  `dag_consistency=1.0`, `depth=0.995`, and no overlaps.
- Treat `triangular_lattice_36` as phase two. It is exactly planar and
  high-degree lattice-like, but it is not currently tagged `lattice_like`
  (`E/N=2.361`, `layer_width_cv=0.489`), so pulling it in requires a separate
  high-density lattice gate.
- Skip `sierpinski_42` by default. Current HEAD measured `dagua=85.43` vs
  `graphviz_dot=84.29`; dagua already has better CV (`0.206` vs `0.353`).
  A lattice snap risks trading away a protected TIE-or-better case.

## Findings

### 1. High -- the gap is a discrete coordinate assignment gap, not another loss-weight gap

Evidence: `CONTEXT.md` already rules out `w_length_variance`, `w_attract`,
`w_repel`, `multi_start_k`, more steps, and wider polish sweeps. Code backs
that up: `dagua/layout/ops/pipelines/dagua_native.py` runs the sprint-20k/l
direct edge-equalize picker after the selected pipeline, scoring candidates via
`composite(full(...))`. The composite gives edge-length CV 20 points and
straightness 10 points in `dagua/metrics.py`.

Measured at HEAD with `LayoutConfig(seed=42)` and seeded metric scoring:

| graph | dagua | best/reference | current diagnosis |
|---|---:|---:|---|
| `hexagonal_lattice_42` | 86.46 | dot 88.99 | CV `0.456` vs `0.099`; dagua wins depth/straightness |
| `triangular_lattice_36` | 85.48 | ogdf_sugiyama 87.16, dot 87.09 | CV `0.256` vs `0.140`/`0.233`; smaller remaining gap |
| `sierpinski_42` | 85.43 | dot 84.29 | already ahead; dot's grid-ish y ranks do not win |

The hex target can afford to trade some straightness for edge uniformity. If
hex CV moves from `0.456` to dot-like `0.10`, the CV term alone is worth about
`+7.1` composite. If straightness degrades from dagua's `4.8 deg` to dot's
`17.4 deg`, that costs about `-2.8`. Keeping dagua's depth correlation instead
of dot's `0.823` preserves another `~+2.6` relative to dot. That is why a
post-pipeline snap can plausibly land around `+3.5..+5.5` on hex.

### 2. High -- the gate must use exact planarity and metric bottleneck checks

The current classifier's `lattice_like` tag is useful but not safe enough for a
projection that moves nodes. From `classify_graph`:

- `hexagonal_lattice_42`: `N=42`, `E=53`, `max_degree=3`, `E/N=1.262`,
  `layer_width_cv=0.396`, `is_planar=True`, tags `("lattice_like",)`.
- `triangular_lattice_36`: `N=36`, `E=85`, `max_degree=6`, `E/N=2.361`,
  `layer_width_cv=0.489`, `is_planar=True`, tags `()`.
- `sierpinski_42`: `N=42`, `E=81`, `max_degree=4`, `E/N=1.929`,
  `layer_width_cv=0.475`, `is_planar=True`, tags `("planar_dag",)`.
- `petersen_10`: `N=10`, `E=15`, `max_degree=3`, `E/N=1.500`,
  `layer_width_cv=0.283`, `is_planar=False`, tags `("lattice_like",)`.

So the right gate is not just "planar hint + uniform degree + planar bound".
The cheap planar bound admits Petersen. The recommended gate is:

```python
def should_lattice_snap(structure, pos, edge_index, node_sizes, selected):
    if selected not in {"layered_dag", "hybrid"}:
        return False
    if structure is None or structure.is_planar is not True:
        return False
    if not structure.is_directed_acyclic:
        return False
    if structure.num_components != 1:
        return False
    if structure.num_layers < 5 or structure.num_layers > 64:
        return False
    if structure.max_degree < 3 or structure.max_degree > 6:
        return False

    cv = edge_length_cv(pos, edge_index)
    if cv < 0.20:
        return False

    hex_band = (
        "lattice_like" in structure.topology_tags
        and structure.max_degree <= 3
        and 1.15 <= structure.edge_to_node_ratio <= 1.45
        and structure.layer_width_cv <= 0.42
    )
    tri_band = (
        structure.max_degree >= 5
        and 2.10 <= structure.edge_to_node_ratio <= 2.55
        and structure.layer_width_cv <= 0.55
    )
    return hex_band or tri_band
```

For sprint 21, ship `hex_band` first and keep `tri_band` disabled or hidden
behind a second candidate flag until the 93-graph sweep proves it does not
touch protected wins.

### 3. Medium -- fit an oblique lattice, but preserve layer/order invariants

The snap should be a projection from the polished layout, not a fresh layout.
The projection should never reorder nodes within a topological layer unless a
candidate is later rejected by the score picker.

Pseudocode:

```python
def lattice_snap_candidate(pos, edge_index, layers, structure, node_sizes):
    base_score = score(pos)
    groups = group_nodes_by_layer(layers, order_by=pos[:, 0])
    y_rank = fit_monotone_layer_ladder(pos[:, 1], groups)

    candidates = []
    for family in allowed_families(structure):
        # family = "hex_sparse" first; "tri_dense" later.
        basis_set = fit_basis_candidates(pos, edge_index, family)
        for basis in basis_set:
            for phase in small_phase_sweep(basis):
                grid = build_layered_grid(groups, y_rank, basis, phase)
                assignment = min_displacement_monotone_assignment(
                    pos=pos,
                    groups=groups,
                    grid=grid,
                    keep_layer_order=True,
                    forbid_collisions=True,
                )
                cand = materialize_positions(assignment)
                cand = optional_axis_locked_equalize(
                    cand,
                    edge_index,
                    basis=basis,
                    iters=5,
                    step=0.03,
                )
                if violates_hard_guards(cand, pos, edge_index, node_sizes):
                    continue
                candidates.append(cand)

    return best_by_composite(candidates, baseline=pos, margin=0.5)
```

Basis fitting details:

- `hex_sparse`: keep topological y-ranks, fit one x step from non-vertical edge
  projections, and allow alternating half-step phase per layer. This matches
  dot's observed behavior better than a mathematically perfect hex lattice:
  dot has 18 unique x values and 12 unique y values, but edge lengths still
  range enough for CV `0.099`, not zero.
- `tri_dense`: use a 60-degree oblique basis. Fit two dominant edge-vector
  modes after normalizing polished edge vectors, then solve scale/phase by a
  small sweep. This should be opt-in later because triangular's current gap is
  modest and the gate is broader.
- Assignment: for each layer, use sorted-node-to-sorted-site matching. This is
  O(N log N), deterministic, and preserves crossing behavior better than nearest
  free-site matching that may swap neighbors.
- Guards before scoring: finite coordinates, no new overlaps, no worse
  `dag_consistency`, no crossing-rate increase above a tiny epsilon, and no
  depth Spearman drop larger than `0.02`.

Do not pursue unconstrained Lloyd relaxation on a full 2-D grid. It optimizes
displacement/CV but has no reason to preserve the rank semantics that carry 25
points of DAG consistency and 15 points of depth correlation.

## Expected Composite Delta

| graph | current | expected after snap | delta | rationale |
|---|---:|---:|---:|---|
| `hexagonal_lattice_42` | 86.46 | 90.0-92.0 | `+3.5..+5.5` | CV can plausibly drop from `0.456` toward dot's `0.099`; straightness can degrade and still net positive |
| `triangular_lattice_36` | 85.48 | 86.3-87.3 | `+0.8..+1.8` | Smaller CV headroom; too much straightness loss erases the gain |
| `sierpinski_42` | 85.43 | 85.43 | `0` | Already TIE-or-better; CV is not the losing metric |

If the implementation accidentally lets hex depth fall to dot's `0.823`, the
hex gain falls by about `2.6` points. That is still possibly useful, but it is
unnecessary: a projection from dagua's existing layered result should keep
depth near `0.995`.

## Risk / Regression Analysis

- **Sprint-20k/l polish wins:** petersen and disconnected/collage gains depend
  on the aggressive edge-equalize variants. Keep lattice snap as an additional
  picker candidate so existing candidates can still win. Exact planarity also
  excludes Petersen.
- **Sprint-19 aspect-target tuning:** `lattice_like` currently feeds aspect
  policy. The snap runs after layout and must not change the `aspect_target`
  default or the classifier tag definition during the first pass.
- **Square grids (`grid_5x5`, `grid_rect_6x8`, `grid_20x20`):** these are
  exactly planar DAGs but currently tagged `planar_dag`, not `lattice_like`,
  with layer-width CV around `0.47..0.55`. They are already competitive; do not
  include a square-grid snap in the first implementation.
- **Large DAG losses (`dependency_500`):** `dependency_500` has `max_degree=53`,
  `E/N=2.94`, `layer_width_cv=0.922`, and exact `is_planar=False`; it should
  never enter this path.
- **Crossing and overlap regressions:** snapping can create collisions or swap
  near-neighbors. Use sorted per-layer assignment, overlap guards, and the final
  composite picker. Reject if `overlap_count` becomes nonzero or crossing rate
  increases.

## Recommended Gate

For the first implementation:

1. `selected in {"layered_dag", "hybrid"}`.
2. `structure.is_planar is True`, not merely `is_planar_hint`.
3. Connected, directed-acyclic, `5 <= num_layers <= 64`.
4. Current polished CV `>= 0.20`; otherwise there is not enough 20-point metric
   headroom.
5. Hex-only structural band: `max_degree <= 3`, `1.15 <= E/N <= 1.45`,
   `layer_width_cv <= 0.42`, and `lattice_like` tag present.
6. Metric guards on candidate: finite, no overlap, no lower DAG consistency,
   depth drop <= `0.02`, crossing-rate increase <= epsilon.

After the hex result is verified, add a second triangular band:
`max_degree >= 5`, `2.10 <= E/N <= 2.55`, `layer_width_cv <= 0.55`, exact
planarity, and current CV `>= 0.20`. That band should be separately swept
because it is broader than the current classifier tag.

## Implementation Order

1. Add a private lattice candidate near `_best_of_polish` rather than a new
   top-level pipeline. It can reuse the existing score picker and margin.
2. Implement hex sparse projection only: y-rank ladder, per-layer sorted
   x-grid with alternating half-step phase sweep, monotone assignment, guards.
3. Verify targeted graphs: `hexagonal_lattice_42`, `triangular_lattice_36`,
   `sierpinski_42`, `petersen_10`, `grid_5x5`, `grid_rect_6x8`, `grid_20x20`,
   `dependency_500`, `disconnected_label_cycle_collage`.
4. Run the deterministic 93-graph h2h once. The acceptable first-pass outcome
   is hex `+3` or better with zero regressions.
5. Only then enable the triangular dense candidate and repeat the same sweep.

## Open Questions

- Does axis-locked edge equalization after snap add value, or does the initial
  snap already give the full CV gain? Keep it as a candidate variant, not a
  required step.
- Should candidate selection optimize the exact benchmark composite or a
  stricter internal score with hard metric floors? I recommend both: hard floors
  first, then composite picker.
- Is triangular worth the extra gate complexity for sprint 21? The quantified
  upside is only `+0.8..+1.8`, so it should follow the hex proof rather than
  block it.
