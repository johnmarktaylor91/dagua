# Sprint 24 Area C: Hex Layer Centering -- Codex Report

## TL;DR

Do **not** ship either requested variant as written. Full-sample
`dagua.metrics.full()` + `dagua.metrics.composite()` scoring says both miss the
strict hex target:

- Current live Dagua on `hexagonal_lattice_42`: `88.355`, delta `-0.632` vs
  graphviz_dot `88.986`.
- Variant 1, honeycomb half-pitch stagger: `82.751`, delta `-6.235`.
- Variant 2, additive per-layer median centering: `86.216`, delta `-2.771`.
- Strict threshold is `88.49`; neither requested candidate reaches it.

The useful finding is a nearby third candidate: **uniform centered layer slots
at `0.75 * LP pitch`**. It scores `89.114` on `hexagonal_lattice_42`, delta
`+0.127` vs dot and `+0.759` vs current live Dagua. That clears the strict
success criterion and the `0.1` picker margin. It also improves
`triangular_lattice_36` from `86.607` to `87.058` and keeps it tied with dot
(`-0.028`). It would regress current `grid_5x5` if forced (`94.136 -> 93.056`),
so it must be a scored polish candidate or use a narrow honeycomb/triangular
lattice gate, not a forced replacement.

Recommended ship decision: **ship neither Variant 1 nor Variant 2**. If sprint
24 wants an Area C implementation, ship the third candidate as
`_lattice_uniform_centered_slots` inside the existing polish picker, gated to
small lattice-like DAGs and rejected for Sierpinski, planar nested cycles,
parallel multiedges, and dependency DAGs. This is outside the prompt's two named
variants, but it is the only tested candidate here that reaches the target.

Scratch artifacts:

- `/tmp/sprint24_c_codex/hex_layer_centering_research.py`
- `/tmp/sprint24_c_codex/score_selected.py`
- `/tmp/sprint24_c_codex/results_selected_full.json`

Scoring used `stress_sources=200`, `stress_targets=1000`,
`crossing_samples=1_000_000`, and `neighborhood_samples=5000`.

## Algorithm sketch for both variants

Variant 1, hex-staggered LP:

```python
def looks_like_honeycomb(edge_index, n):
    if n < 12 or edge_index.numel() == 0:
        return False
    e = edge_index.shape[1]
    if not 1.20 <= e / n <= 1.35:
        return False
    deg = undirected_degree(edge_index, n)
    if deg.max() > 3:
        return False
    return mean(deg == 3) >= 0.25


def hex_staggered_lp(lp_pos, edge_index):
    if not looks_like_honeycomb(edge_index, lp_pos.shape[0]):
        return lp_pos
    layers = group_by_equal_y(lp_pos)
    pitch = median_positive_adjacent_x_gap(lp_pos, layers)
    out = center_each_layer_midrange_on_global_midrange(lp_pos)
    for layer_index, layer in enumerate(layers):
        direction = -0.5 if layer_index % 2 == 0 else 0.5
        out[layer, 0] += direction * pitch
    return out - out.mean(dim=0)
```

This matches the prompt's "stagger even/odd rows by half-pitch" idea. It is
intentionally narrow: the gate accepts `hexagonal_lattice_42` and rejects the
non-honeycomb protected graphs. Empirically it fails because the stagger makes
many diagonals too uneven and raises straightness deviation to `31.61` degrees.

Variant 2, additive lattice layer-centering:

```python
def layer_median_center(lp_pos):
    out = lp_pos.clone()
    layers = group_by_equal_y(out)
    global_median = median(out[:, 0])
    for layer in layers:
        layer_median = median(out[layer, 0])
        out[layer, 0] += global_median - layer_median
    return out - out.mean(dim=0)
```

This is the exact prompt variant: preserve every layer's internal LP spacing and
apply only one additive x-shift per layer. It lowers hex CV only marginally
(`0.1660 -> 0.1627`) and worsens straightness (`14.99 -> 24.15`), so it loses
composite.

The successful probe:

```python
def uniform_centered_slots(lp_pos, pitch_scale=0.75):
    out = lp_pos.clone()
    layers = group_by_equal_y(out)
    pitch = median_positive_adjacent_x_gap(out, layers) * pitch_scale
    axis = median(out[:, 0])
    for layer in layers:
        order = sort(layer, key=lambda v: out[v, 0])
        count = len(order)
        slots = axis + (arange(count) - (count - 1) / 2) * pitch
        out[order, 0] = slots
    return out - out.mean(dim=0)
```

This is not just centering; it replaces each layer's x coordinates with
uniformly spaced centered slots while preserving within-layer order and all y
coordinates. It is closer to a small Brandes-Koepf coordinate rewrite than to an
additive shift.

## Empirical table

Scores are full composite scores. `dBase` is against current live Dagua for
small graphs; for `dependency_500` I used the stored Dagua benchmark tensor
because live layout is slow and the candidate gate should reject it anyway.
`dDot` is against stored graphviz_dot.

| Graph | Current/stored Dagua | LP baseline | V1 hex stagger | V2 median center | Uniform 0.75 probe | graphviz_dot |
|---|---:|---:|---:|---:|---:|---:|
| `hexagonal_lattice_42` | 88.355 | 88.187 | 82.751 | 86.216 | **89.114** | 88.986 |
| `triangular_lattice_36` | 86.607 | 86.607 | 86.607 | 83.022 | **87.058** | 87.086 |
| `grid_5x5` | **94.136** | 89.265 | 89.265 | 89.265 | 93.056 | 91.597 |
| `sierpinski_42` | **85.576** | 83.776 | 83.776 | 76.306 | rejected: 85.576 | 84.290 |
| `planar_60` | **80.089** | 76.045 | 76.045 | 78.738 | rejected: 80.089 | 75.115 |
| `parallel_multiedge_bundle` | 85.500 | 67.500 | 67.500 | 67.500 | rejected: 85.500 | **85.501** |
| `dependency_500` | 58.210 | n/a unsafe zero fallback | rejected | rejected | rejected: 58.210 | 54.244 |

Important guardrail notes:

- `grid_5x5` remains a Dagua win only if the new candidate is scored by the
  picker and rejected. Forced replacement would be a `-1.081` regression from
  current Dagua.
- `sierpinski_42` and `planar_60` prove that broad additive centering is unsafe.
  Variant 2 regresses Sierpinski by `-9.270` and planar by `-1.351` from current
  Dagua.
- `parallel_multiedge_bundle` and `dependency_500` should not call the LP-derived
  transform. The private LP candidate returns the seed tensor when its gate
  rejects, which is an invalid all-zero layout; production wiring must only
  score transformed candidates when their own gate accepts.

## Hex per-metric breakdown

| Variant | Composite | dDot | edge_length_cv | depth_spearman_rho | straight deg | crossing_rate | angular deg | overlaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current live Dagua | 88.355 | -0.632 | 0.4197 | 0.9953 | 3.07 | 0.00000 | 51.49 | 0 |
| LP baseline | 88.187 | -0.800 | 0.1660 | 0.8225 | 14.99 | 0.00000 | 75.90 | 0 |
| V1 hex stagger | 82.751 | -6.235 | 0.2532 | 0.8225 | 31.61 | 0.00000 | 55.37 | 0 |
| V2 median center | 86.216 | -2.771 | 0.1627 | 0.8225 | 24.15 | 0.00000 | 68.18 | 0 |
| Uniform 0.75 probe | **89.114** | **+0.127** | **0.0472** | 0.8225 | 21.51 | 0.00000 | 68.20 | 0 |
| graphviz_dot | 88.986 | 0.000 | 0.0991 | 0.8225 | 17.42 | 0.00000 | 78.83 | 0 |

The gap is indeed mostly edge-length regularity, not crossings. Current live
Dagua wins straightness and depth correlation, but its edge-length CV is high.
The LP baseline matches dot's depth correlation and crossing behavior but is
slightly worse on CV and straightness. Additive centering does not repair that
tradeoff. Uniform centered slots overcorrect CV in a good way (`0.0472`, better
than dot) while losing only enough straightness to remain net positive.

## Recommended implementation

Slot in `dagua/layout/ops/pipelines/dagua_native.py`, adjacent to
`_dot_lattice_lp`, as a new polish candidate rather than a replacement.
Estimated LOC:

- Honeycomb/tri/grid-ish gate: 35-55 LOC.
- Layer grouping + pitch extraction: 25-35 LOC, reusable with existing LP helper
  code if factored locally.
- Uniform centered slot transform: 25-35 LOC.
- Picker wiring and candidate naming: 10-20 LOC.
- Tests: 60-90 LOC focused on gate behavior and no forced `grid_5x5` regression.

Production gate recommendation:

```python
def _should_lattice_uniform_center(edge_index, num_nodes, lp_pos):
    if num_nodes < 12 or num_nodes > 200:
        return False
    if not _should_dot_lattice_lp(edge_index, num_nodes):
        return False
    layers = group_by_equal_y(lp_pos)
    widths = sorted(len(layer) for layer in layers)
    if len(layers) < 5 or max(widths) < 4:
        return False
    deg = undirected_degree(edge_index, num_nodes)
    if deg.max() > 6:
        return False
    # Reject fractal/nested planar shapes: too many singleton/tiny layers.
    if sum(width <= 2 for width in widths) / len(widths) > 0.45:
        return False
    return True
```

Then score `_lattice_uniform_centered_slots(_dot_lattice_lp(...), 0.75)` inside
the existing `_best_of_polish` picker. The picker is essential: it accepts the
hex win, likely accepts the tri improvement, and rejects the `grid_5x5`
regression against current live Dagua. If implementation wants a safer first
ship, gate to honeycomb only; that still closes the sprint-24 blocker.

Controversial choice: I would not claim the requested "lattice BK
layer-centering" bet succeeded. The successful candidate is a coordinate-slot
rewrite derived during the probe sweep. It is small enough for the same LOC
budget, but it is semantically different from an additive per-layer shift.

Concerns:

- The `0.75` pitch scale is empirical. It should be guarded by picker scoring,
  and a tiny local search over `{0.7, 0.75, 0.8}` may be more robust if runtime
  remains acceptable.
- `dependency_500` revealed a private-LP failure mode: when the LP gate rejects,
  the helper returns the input seed. Production code must not transform or score
  that zero fallback as if it were a real LP layout.
- The new candidate should not be forced on `grid_5x5`; Dagua's current live
  layout is still better there.
