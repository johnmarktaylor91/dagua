# Area E -- Polish-op extensions / new projection primitives (codex)

## TL;DR

The next polish picker should stay exactly in the sprint-20k shape: generate
cheap detached candidates, score each with `composite(full(...))`, and keep the
baseline unless a candidate clears the 0.5 margin. That shape is already in
`dagua/layout/ops/pipelines/dagua_native.py`: `_POLISH_SETTINGS` enumerates
seven edge-equalize variants, `_equalize_edges` only moves edge endpoints
toward the current mean edge length, and `_best_of_polish` scores candidates
against the unpolished baseline before accepting them.

The three most promising additions are:

1. **Layer-y snap / layer flatten**. Snap near-equal y bands to their median y.
   It directly targets `dag_consistency`, `depth_spearman_rho`, and
   `edge_straightness_mean_deg`, not just `edge_length_cv`. Expected: +6 to
   +10 on wide single-layer graphs, +0.5 to +1.5 on `inception_block`,
   `hub_fanout_label_skew`, and similar shallow DAGs. It will not fix
   `dependency_500`, but it can flip ties and close-loss shallow DAGs.

2. **Orthogonal dominant-axis align**. For each edge, damp the minor coordinate
   delta, choosing vertical or horizontal per edge. This is the Manhattan-axis
   snap idea in a soft projection form. Expected: +2 to +8 on skip-heavy and
   clustered DAGs (`multiscale_skip_cascade`, `hub_skip_superfan`,
   `weighted_clusters_3x10`), +0.5 to +1.5 on `residual_block`-like graphs.

3. **Topology-specific lattice basis snap**. Do not do naive integer
   grid-snap. Fit two local basis vectors from existing short edges, then snap
   only lattice-tagged/low-degree near-regular graphs by blending to the nearest
   axial or triangular lattice coordinate. Expected: +1.0 to +2.5 on
   `hexagonal_lattice_42` and `triangular_lattice_36`; low value elsewhere.

Secondary candidates worth testing after those three are a crossing-aware
adjacent swap pass and an aspect-preserving residual edge polish for
`dependency_500`. I would not put FR/SFDP residual steps in the first picker
extension because they are not direct projections and they consume more budget
than the direct candidates.

## Evidence anchor

Current sprint context says sprint-20k/20l polish is already a low-risk lever:
edge-equalize added +94.20 net composite over 45 wins and 0 regressions, then
the aggressive variants lifted `petersen_10` and
`disconnected_label_cycle_collage`. The remaining moderate losses are
`dependency_500` (-2.90), `petersen_10` (-2.72), and
`hexagonal_lattice_42` (-2.52). Close losses include
`triangular_lattice_36`, `transformer_layer`, and `small_world_500`.

Metric leverage is clear from `dagua/metrics.py:1171`: composite weights are
25 for DAG consistency, 20 for edge-length CV, 15 for depth correlation, 10
for overlaps, 10 for edge straightness, 10 for crossings, and 5 each for
angular resolution and cluster separation. The existing `_equalize_edges`
projection at `dagua_native.py:391` addresses mostly the 20-point CV term.
It can incidentally improve crossings or straightness, but it has no direct
notion of layer flatness, node ordering, lattice basis, or overlap repair.

## Primitive 1: Layer-y snap / layer flatten

**Pseudocode**

```python
def layer_y_snap(pos, node_sizes, band_factor=0.5, min_layer_size=2):
    out = pos.clone()
    band = max(median(node_sizes[:, 1]) * band_factor, 1.0)
    bucket = round(out[:, 1] / band)
    for b in unique(bucket):
        idx = where(bucket == b)
        if len(idx) >= min_layer_size:
            out[idx, 1] = median(out[idx, 1])
    return out
```

A slightly safer variant computes buckets from the existing y gaps: sort nodes
by y, split a new layer when the next gap exceeds `0.75 * median_node_height`,
then replace each band by its median y. That avoids accidentally collapsing a
force-directed cloud into a line when node sizes are large relative to layout
span.

**Targets and expected delta**

This primitive targets the 25-point DAG term, 15-point depth term, and
10-point straightness term. It is most useful when gradient optimization
already found the correct layer but left small y noise. Existing context
already identifies `wide_1_100_1`, `wide_single_layer_1_50_1`, and
`wide_3_50_3` as graphs where Dagua historically lost depth/ordering score
despite an obvious layered structure. Expected deltas:

- `wide_single_layer_1_50_1`: +6 to +10 composite.
- `wide_3_50_3`: +4 to +7.
- `inception_block`, `hub_fanout_label_skew`, `hub_and_spoke_3x20`: +0.5 to
  +1.5.
- `transformer_layer`: low to moderate, +0.0 to +0.8, because its known loss
  is more crossing/CV than pure layer noise.

**Cost**

One pass over nodes plus sorting if the gap-based variant is used: O(N log N)
or O(N). No edge-pair work, no metric precomputation.

**Risk**

Applied blindly, it can destroy non-hierarchical layouts such as
`small_world_500` by collapsing y variation. In the picker, that is acceptable:
`_best_of_polish` preserves the baseline unless the candidate beats it by 0.5.
For runtime cleanliness, restrict this candidate to `layered_dag`, `tree`, and
`hybrid`, or require at least one y band with two or more nodes and a
layer-count-to-node-count ratio below about 0.6. Existing wins from
edge-equalize should be safe because the candidate is compared, not applied
unconditionally.

## Primitive 2: Orthogonal dominant-axis align

**Pseudocode**

```python
def orthogonal_align(pos, edge_index, iters=10, step=0.1):
    out = pos.clone()
    src, tgt = edge_index
    src, tgt = drop_self_loops(src, tgt)
    for _ in range(iters):
        d = out[tgt] - out[src]
        vertical = abs(d[:, 1]) >= abs(d[:, 0])
        delta = zeros_like(d)
        delta[vertical, 0] = d[vertical, 0] * step
        delta[~vertical, 1] = d[~vertical, 1] * step
        out.index_add_(0, src, 0.5 * delta)
        out.index_add_(0, tgt, -0.5 * delta)
    return out
```

This is a soft Manhattan-axis snap. Unlike edge-equalize, it does not care
whether an edge is too long or too short; it reduces the minor axis component
of each edge. For top-to-bottom DAGs, most edges become more vertical. For
local horizontal cross-links, it preserves horizontalness instead of forcing
everything into a vertical stack.

**Targets and expected delta**

Primary target is `edge_straightness_mean_deg` (10 points), with secondary
effects on `dag_consistency`, `depth_spearman_rho`, and crossings. It should
be especially useful on skip-heavy DAGs where edge-equalize can make lengths
more uniform but cannot square up slanted bridges. Expected deltas:

- `multiscale_skip_cascade`: +5 to +8 composite.
- `weighted_clusters_3x10`, `sbm_4x30`: +2 to +4.
- `hub_skip_superfan`, `densenet_block`, `moe_router_sparse`: +1 to +3.
- `residual_block`: +0.5 to +1.0 on top of the existing edge-equalize win.
- `hexagonal_lattice_42` and `triangular_lattice_36`: likely rejected by the
  picker; true lattice edges should remain diagonal.

**Cost**

O(E * iters), with 10 to 20 iterations enough. It uses the same `index_add_`
pattern as `_equalize_edges`, so implementation risk is low and GPU behavior
is predictable.

**Risk**

Raw regressions are likely on lattices, trees, and non-hierarchical graphs:
regular diagonal structure becomes stair-stepped, and some cyclic graphs may
oscillate as adjacent edges choose different axes. The mitigation is to add it
as a candidate after the existing edge-equalize variants, not as a replacement.
Use a light setting first (`iters=10`, `step=0.1`); an aggressive setting can be
included later only if measurements show unique wins.

## Primitive 3: Lattice basis snap

**Pseudocode**

```python
def lattice_basis_snap(pos, edge_index, blend=0.35):
    short_edges = shortest_quantile_edges(pos, edge_index, q=0.65)
    basis = fit_two_basis_vectors_from_edge_angles(short_edges)
    if basis.condition_bad:
        return pos
    uv = solve_lattice_coordinates(pos - median(pos), basis)
    snapped_uv = round_to_axial_or_triangular_grid(uv)
    snapped = median(pos) + snapped_uv @ basis
    return (1.0 - blend) * pos + blend * snapped
```

This is deliberately not "snap x and y to the nearest integer grid." The
remaining lattice losses are hexagonal/triangular, so the useful grid is
oblique. Fit basis vectors from the current layout's own short-edge angle
modes, then blend toward the nearest lattice coordinate. Try blends
`0.25`, `0.35`, and `0.50`.

**Targets and expected delta**

Primary target is `edge_length_cv` (20 points). The sprint context says
`hexagonal_lattice_42` is at CV 0.43 versus graphviz_dot 0.10, with current
polish already lifting it by +1.25 but leaving a -2.52 loss. A basis snap
should not need to close the whole CV gap; a CV move from 0.43 to ~0.30 is
worth about +2.6 raw composite before straightness/depth tradeoffs. Expected
net:

- `hexagonal_lattice_42`: +1.0 to +2.5 if blend is modest.
- `triangular_lattice_36`: +0.8 to +2.0.
- `sierpinski_42`: +0.0 to +0.7; fractal structure is not a uniform lattice.
- `grid_20x20` / rectangular grids if present in picker path: +0.5 to +2.0
  only with a rectangular basis mode.

**Cost**

O(E) for edge vectors, O(N) for coordinate projection. Fitting angle modes can
be done with a small fixed histogram, not clustering.

**Risk**

High if applied to non-lattices. It can break the straightness and depth terms
that keep Dagua competitive on layered DAGs. Gate by topology before scoring:
low degree CV, degree mostly 2-4 or 3-6, low component count, and no strong
semantic DAG signal. Even then, rely on the picker. Do not use this candidate
for `dependency_500`, `transformer_layer`, or `small_world_500`.

## Primitive 4: Layer-internal x equalize

**Pseudocode**

```python
def layer_x_equalize(pos, node_sizes):
    out = pos.clone()
    for layer in y_bands(out, node_sizes):
        order = argsort(out[layer, 0])
        widths = node_sizes[layer[order], 0]
        gap = median(widths) * 1.5
        centered = cumulative_width_positions(widths, gap)
        out[layer[order], 0] = centered - mean(centered) + mean(out[layer, 0])
    return out
```

**Targets and expected delta**

This targets `crossing_rate` and `edge_length_cv` together by preserving
within-layer order while making x spacing uniform. It is plausible on
`transformer_layer`, `ragged_feature_pyramid`, `wide_3_50_3`, and dependency
subgraphs with broad layers. Expected +0.5 to +1.5 on shallow/wide DAG ties.
For `dependency_500`, expect only +0.0 to +0.8 because the graph's loss is
large-DAG CV and angular resolution; uniform x spacing may lengthen some skip
edges.

**Cost**

O(N log N) from per-layer sorting.

**Risk**

This can erase meaningful barycenter spacing and increase crossings when a
layer has interleaved parents. I would not include a forced x-equalize-only
candidate until after y-snap and orthogonal-align are measured. Safer variant:
blend x only 25-50% toward equal spacing and score both blends.

## Primitive 5: Crossing-aware adjacent swap

**Pseudocode**

```python
def adjacent_crossing_swap(pos, edge_index, node_sizes, max_passes=2):
    out = pos.clone()
    layers = y_bands(out, node_sizes)
    for _ in range(max_passes):
        for layer in layers:
            order = argsort(out[layer, 0])
            for a, b in adjacent_pairs(order):
                before = local_crossing_count(a, b, out, edge_index)
                swap_x(out, a, b)
                after = local_crossing_count(a, b, out, edge_index)
                if after >= before:
                    swap_x(out, a, b)
    return out
```

**Targets and expected delta**

This directly targets `crossing_rate` (10 points) and often helps angular
resolution. Expected:

- `transformer_layer`: +0.8 to +1.8, because context calls it a close loss
  and crossing/CV dominated.
- `ragged_feature_pyramid`: +0.5 to +1.5 after edge-equalize.
- `regular_3_30` / `planar_60`-like graphs: +0.5 to +2.0 if still close.
- `dependency_500`: probably too expensive globally; use sampled/local swaps
  only, expected +0.0 to +0.7.

**Cost**

Naive crossing recount is O(E^2), which is too high for every candidate. Keep
it local: for swapped nodes, inspect only incident edges and edges between
neighboring y bands. Two passes should be enough.

**Risk**

Swapping can improve crossings while hurting edge length CV and straightness,
especially on lattices. Picker scoring is mandatory. This primitive also has
the highest implementation complexity among the direct projections, so it
should come after the simpler geometry projections.

## Primitive 6: Aspect-preserving residual edge polish

**Pseudocode**

```python
def aspect_preserving_equalize(pos, edge_index, iters=10, step=0.05):
    target_box = bbox(pos)
    target_aspect = width(target_box) / height(target_box)
    out = equalize_edges(pos, edge_index, iters, step)
    out = recenter(out, center(pos))
    out = rescale_to_aspect_and_area(out, target_aspect, area(target_box))
    return out
```

**Targets and expected delta**

This is for `dependency_500`: current polish reportedly regresses, likely
because equalizing edges drifts the aspect/area that the large DAG needs for
readable separation. Locking the box gives the edge-length projection less
room to damage depth and straightness. Expected:

- `dependency_500`: +0.5 to +1.5 if the current rejection is aspect drift.
- `transformer_layer`: +0.3 to +1.0.
- Existing edge-equalize wins: usually neutral; it may underperform plain
  equalize on `petersen_10` and `disconnected_label_cycle_collage`.

**Cost**

O(E * iters) plus O(N) rescale. Very cheap.

**Risk**

If the bad candidate is not aspect drift but local crossing/ordering damage,
this will not help. Still worth testing because it reuses existing
`_equalize_edges` semantics and preserves the sprint-20k mental model.

## Deferred candidates

**Force-directed-on-residual** is likely useful for `small_world_500`, but it
is not really a direct projection. It also overlaps with the existing stress
route work from sprint-20i. If included, make it a separate topology-dispatch
candidate for dense cyclic/flat-layer graphs, not a general polish setting.
Expected +1 to +3 on `small_world_500`; runtime risk is much higher than the
direct projections.

**Backbone-then-leaf smoothing** sounds attractive for trees and chains, but
sprint-20l edge-equalize already helps tree/fractal cases such as
`sierpinski_42`. Longest-path alignment can damage angular resolution around
hubs. I would only test it if tree close losses remain after y-snap and
orthogonal-align.

**Overlap jitter** targets the binary 10-point overlap term, but as a picker
candidate it can dominate by accident. Better design: after the best polish
candidate is chosen, run a tiny final overlap repair and re-score once.

## Recommended order to add to the polish settings list

1. Keep all seven existing edge-equalize variants first.
2. Add `edge_equalize -> layer_y_snap` for two proven equalize prefixes:
   `(10, 0.10)` and `(30, 0.02)`.
3. Add `layer_y_snap` directly on `base_pos` for graphs where equalize is a
   no-op but layer noise is the only problem.
4. Add `edge_equalize -> orthogonal_align(10, 0.1)` using light equalize
   prefixes `(10, 0.05)` and `(20, 0.03)`.
5. Add `aspect_preserving_equalize(10, 0.05)` and `(20, 0.03)` for the
   `dependency_500`/large-DAG bucket.
6. Add lattice-basis snap only behind a lattice-like topology predicate.
7. Add crossing-aware adjacent swap after the simple candidates are measured.

## Combined topology-aware picker

```python
def polish_candidates(base_pos, edge_index, node_sizes, structure):
    yield from existing_edge_equalize_candidates(base_pos)

    if structure.family in {"layered_dag", "tree", "hybrid"}:
        yield y_snap(base_pos)
        for ee in selected_equalize_prefixes(base_pos):
            yield y_snap(ee)
            yield orthogonal_align(ee, iters=10, step=0.1)
            yield aspect_preserving_equalize(base_pos, edge_index)

    if is_lattice_like(structure, edge_index):
        for blend in (0.25, 0.35, 0.50):
            yield lattice_basis_snap(base_pos, edge_index, blend=blend)

    if is_large_dag_or_wide_layered(structure):
        yield layer_x_equalize_blend(base_pos, blend=0.35)
        yield adjacent_crossing_swap(base_pos, edge_index, node_sizes)

    if is_multi_component(structure):
        # Use only shape-preserving candidates after tiled placement.
        yield y_snap(base_pos)
        yield aspect_preserving_equalize(base_pos, edge_index)
```

This keeps candidate count small where runtime matters, but the important
safety property remains the same: every candidate is scored by the same
`composite(full(...))` gate used today. The only candidates I would
topology-gate before scoring are lattice-basis snap and any force/stress
residual candidate; the others are cheap enough to let the picker reject.

## Assumptions

Expected deltas are projections from the sprint-21 context, the metric weights,
and previously recorded per-graph diagnoses, not fresh exhaustive measurements.
I attempted the local `/tmp/score_breakdown.py` helper, but it produced no
output before the session ended; I did not use it as evidence. The
implementation pass should verify the ordered candidate list with the same
deterministic 93-graph seeded h2h used for sprint-20l.
