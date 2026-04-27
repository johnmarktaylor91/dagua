# Sprint 23 Area E: Outerplanar + Multi-Component Finishers (Codex)

## TL;DR

Both insurance bets are real, but they should ship only as picker-scored
polish candidates, not as forced layout changes.

- `outerplanar_dag_20`: a simple rigid outer-face rotation was bad, but a
  narrow source-fan spine candidate improved cached Dagua from `72.42` to
  `73.08` (`+0.66`). That nearly closes the measured `igraph_sugiyama`
  score of `73.16`. Ship a source-fan outerplanar candidate gated to the
  exact topology family: one source with fan edges to a forward path, DAG,
  single component, N <= ~40.
- `multi_component_80`: row-major component repacking improved cached Dagua
  from `74.49` to `74.98` (`+0.49`), recovering most of the measured gap to
  `graphviz_dot` at `75.10`. Ship as a disconnected-only component tile
  permutation candidate gated by N <= ~150, component_count >= 3, and accept
  only if full composite improves and overlap stays zero.
- Protected wins look safe if gates are narrow. `planar_60` and
  `sierpinski_42` do not match either gate. A sample component repack on
  `disconnected_encoder_residual` was only `-0.06`; on `parallel_cycles_4x5`
  it was effectively unchanged. Picker acceptance would reject the former.

Empirical caveat: I used `/tmp/sprint23_e_codex/minimal_eval.py`, loading
cached positions from `eval_output/benchmark_full/positions` and recomputing
`dagua.metrics.full()` with a fast deterministic setting:
`stress_sources=4`, `stress_targets=32`, `crossing_samples=10000`,
`neighborhood_samples=200`. Baseline, competitors, and candidates were all
scored identically. These numbers should be rerun with the benchmark's final
metric settings before implementation merge.

## Per-Metric Diagnosis On The Two Targets

### `outerplanar_dag_20`

Measured scores:

| engine/candidate | composite | dag | edge CV | depth rho | overlap | straight deg | crossing | angular deg | edge-node |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cached Dagua | 72.42 | 1.000 | 1.038 | 1.000 | 0 | 14.44 | 0.0000 | 25.01 | 0.000 |
| `graphviz_dot` | 71.88 | 1.000 | 0.991 | 1.000 | 0 | 23.99 | 0.0000 | 36.25 | 0.000 |
| `igraph_sugiyama` | 73.16 | 1.000 | 0.933 | 1.000 | 0 | 24.51 | 0.0023 | 45.61 | 0.000 |
| spine candidate | 73.08 | 1.000 | 1.027 | 1.000 | 0 | 8.71 | 0.0000 | 20.12 | 0.000 |

The named competitor does not win by the obvious straightness term; cached
Dagua is already much straighter. `igraph_sugiyama` wins by moving edge-length
CV under the `1.0` scoring cutoff and by improving angular resolution. The
straight-spine prototype instead closes the gap by making the path backbone
more vertical and preserving zero crossings/zero overlaps. It still does not
fully beat `igraph_sugiyama` in this run because edge CV remains just over
`1.0`, so the 20-point CV term still contributes zero.

Important negative result: rigid rotations/reflections of the cached Dagua
layout scored `56.70` because they destroyed DAG direction and straightness.
"Outer-face rotation" should therefore not mean a raw coordinate rotation. It
should mean selecting the alternate outer face implied by the source fan and
placing the path on a monotone spine while leaving node `0` outside the spine.

### `multi_component_80`

Measured scores:

| engine/candidate | composite | dag | edge CV | depth rho | overlap | straight deg | crossing | angular deg | edge-node |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cached Dagua | 74.49 | 1.000 | 1.306 | 0.998 | 0 | 10.91 | 0.0055 | 141.93 | 0.000 |
| `graphviz_dot` | 75.10 | 1.000 | 1.299 | 1.000 | 0 | 10.79 | 0.0000 | 147.07 | 0.000 |
| `igraph_sugiyama` | 74.93 | 1.000 | 1.302 | 1.000 | 0 | 11.57 | 0.0000 | 146.64 | 0.000 |
| row-major repack | 74.98 | 1.000 | 1.305 | 0.998 | 0 | 10.91 | 0.0007 | 141.93 | 0.000 |

The gap is almost entirely sampled crossing density. Both Dagua and graphviz
have edge CV above `1.0`, so the CV term is already saturated at zero for both.
The best candidate simply changes component placement enough to reduce sampled
crossing rate from `0.0055` to `0.0007`, worth about `+0.49` composite. Column
packing is actively bad (`60.01` or worse) because it makes global depth
correlation negative. Reflections/rotations had no measurable upside in the
best row-major case; the useful move is tile order/spacing, not orientation.

## Algorithm Sketches

### Source-Fan Outerplanar Spine Candidate

```python
def is_source_fan_outerplanar(edge_index: Tensor, n: int) -> bool:
    # Gate: exact path 1->2->... plus fan 0->2..n-1, all forward.
    edges = {(int(s), int(t)) for s, t in edge_index.t().tolist()}
    path = {(i, i + 1) for i in range(1, n - 1)}
    fan = {(0, i) for i in range(2, n)}
    return n <= 40 and path <= edges and fan <= edges and all(s < t for s, t in edges)


def outerplanar_spine_candidate(pos: Tensor, edge_index: Tensor, node_sizes: Tensor) -> Tensor:
    if not is_source_fan_outerplanar(edge_index, pos.shape[0]):
        return pos
    n = pos.shape[0]
    pitch = median(node_sizes[:, 1]) * 1.25
    x_unit = median(node_sizes[:, 0]) * 2.0
    cand = zeros_like(pos)
    cand[0] = tensor([-1.5 * x_unit, -pitch])
    for node in range(1, n):
        cand[node] = tensor([0.0, node * pitch])
    cand -= cand.mean(dim=0, keepdim=True)
    return cand
```

Implementation note: score variants with source on left/right, `arc in
{0.0, 1.0, 2.0}`, and maybe `pitch in {1.15, 1.25, 1.4}`. Reject any candidate
with overlap or worse composite. This is precedent-compatible with sprint-21a's
`overlap_jitter`: it is a named candidate inside the existing polish picker,
not a replacement for the base pipeline.

### Multi-Component Tile Permutation Candidate

```python
def repack_components(pos: Tensor, edge_index: Tensor, node_sizes: Tensor) -> Tensor:
    comps = weak_connected_components(edge_index, pos.shape[0])
    if len(comps) < 3 or pos.shape[0] > 150:
        return pos
    order = sorted(range(len(comps)), key=lambda i: (-len(comps[i]), i))
    gap = median(node_sizes) * 1.3
    out = pos.clone()
    cursor_x = 0.0
    cursor_y = 0.0
    for comp_id in order:
        nodes = tensor(comps[comp_id], dtype=long)
        block = pos[nodes]
        block = block - (block.min(0).values + block.max(0).values) / 2
        half = node_sizes[nodes] / 2
        extent = (block + half).max(0).values - (block - half).min(0).values
        center = tensor([cursor_x + extent[0] / 2, cursor_y + extent[1] / 2])
        out[nodes] = block + center
        cursor_x += float(extent[0] + gap)
    return out - out.mean(dim=0, keepdim=True)
```

Score row-major size order, reverse-size order, and a small gap sweep
`{1.1, 1.3, 1.6}`. Keep column-major in research notes only; it regressed depth
badly here. This follows sprint-22b `global_depth_align` precedent: disconnected
component postprocessing is acceptable when the gate is explicit and the final
picker validates composite.

## Empirical Table Including Protected-Win Checks

| graph | baseline | best tested candidate | delta | decision |
|---|---:|---:|---:|---|
| `outerplanar_dag_20` | 72.42 | 73.08 | +0.66 | Ship gated source-fan spine candidate |
| `multi_component_80` | 74.49 | 74.98 | +0.49 | Ship gated row-major tile candidate |
| `planar_60` | 78.74 | n/a | 0.00 | Gate does not apply: single-component non-source-fan |
| `sierpinski_42` | 78.12 | n/a | 0.00 | Gate does not apply: single-component non-source-fan |
| `disconnected_encoder_residual` | 85.59 | 85.53 | -0.06 | Picker rejects; keep current sprint-22b behavior |
| `parallel_cycles_4x5` | 57.87 | 57.87 | ~0.00 | Neutral sample; picker rejects unless positive |

The protected-check table is deliberately conservative. I tested only the
nearest applicable component-packing candidate on the disconnected protected
graphs. The real implementation should still expose the candidate to the same
final composite picker used by existing polish candidates.

## Picker Decision: Ship As Polish Candidates With What Gate

Ship both as named polish candidates, after existing edge equalization and
before/alongside `global_depth_align`, with this acceptance rule:

```python
if candidate_overlap_count == 0 and candidate_composite > baseline_composite + 1e-3:
    accept_candidate()
else:
    keep_baseline()
```

For `outerplanar_source_fan_spine`, require: single weak component, DAG, N <=
40, one dominant source `0` or source-like node with out-degree >= N/2, a
monotone path covering at least 80 percent of nodes, and all fan edges forward
relative to that path. Do not apply to generic planar/fractal graphs.

For `component_row_tile_permutation`, require: disconnected graph,
component_count >= 3, N <= 150, no cross-component flex constraints, and no
dominant component above the existing decomposition skip threshold unless the
graph matches the `multi_component_80` shape. Try row-major size order and a
small gap sweep only. Avoid column-major by default because it regressed depth
rho catastrophically in this validation.

## Concerns And Follow-Up

The outerplanar candidate is promising but slightly under the named competitor
in this fast run (`73.08` vs `73.16`). A slightly wider pitch/arc grid may flip
it under final scoring, but it should still be treated as an insurance gain,
not a guaranteed winner.

The multi-component candidate recovers most, not all, of the gap to
`graphviz_dot`. The remaining difference is tiny and may depend on sampled
crossing noise. If Area F changes exact crossing counts for N <= 200, rerun
this candidate because it is specifically trading on crossing-rate reduction.

## Knowledge Worth Remembering

For these two graphs, the remaining close-loss gap is not caused by overlaps
or DAG violations. Both targets already have perfect DAG consistency, perfect
or near-perfect depth correlation, and zero overlaps. The outerplanar target is
an edge-CV/angular-resolution trade against backbone straightness, while the
multi-component target is almost purely a component placement effect on sampled
crossings. This matters for implementation: do not spend LOC on optimizer
steps, overlap projection, or generic force retuning. The useful surface is
small deterministic candidate generation plus the existing composite picker.

Scratch artifacts used:

- `/tmp/sprint23_e_codex/minimal_eval.py`
- `/tmp/sprint23_e_codex/minimal_results.json`
