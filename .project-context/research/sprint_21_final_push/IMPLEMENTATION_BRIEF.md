# Sprint-21 Implementation Brief

## Context

Research reports synthesized from 6 areas (A/B/C/D/E/F). Implementation
follows the convergence: 4 polish-op extensions + 1 per-component
pipeline fix.

Reports to read for context (in `.project-context/research/sprint_21_final_push/`):
- `E_polish_extensions__claude.md` — empirical: 3 new primitives,
  **+48.97 net composite, 0 regressions** across 81 graphs.
- `C_close_loss_lift__claude.md` — per-graph recommendations including
  2-opt anti-crossing polish (cluster 1, 5 graphs).
- `A_lattice_grid_snap__claude.md` — 1-D K-means per-layer x for
  hex_lattice. Reframed: dot is per-layer x-quantization, not grid-snap.
- `F_metric_aware_routing__claude.md` — multi_component_80 hidden -13.37
  miss because per-component path forces `legacy_monolith`.

## Build target

Add 4 new polish primitives to `_POLISH_SETTINGS`/`_best_of_polish` in
`dagua/layout/ops/pipelines/dagua_native.py`. Each is wrapped as a
candidate that the picker (margin >= 0.5) will only adopt if it
improves composite. **Strictly additive — picker filters losses.**

### Primitive 1: y_layer_snap (highest leverage)

Pseudocode (from E_polish_extensions__claude.md L60-95):
```
def y_layer_snap(pos, edge_index, node_sizes, layer_eps=0.5):
    band = mean(node_sizes[:, 1]) * layer_eps
    bucket = round(pos[:, 1] / band)
    for each unique bucket b:
        idx = where(bucket == b)
        pos[idx, 1] = median(pos[idx, 1])
    return pos
```

After edge-equalize. Compose: equalize first (any of the existing
settings), then snap-y. The picker tries the composed candidate.

Expected wins: wide_single_layer_1_50_1 +9.77, wide_3_50_3 +6.86,
inception_block +1.41, wide_1_100_1 +1.04, hub_fanout_label_skew +1.00,
hub_and_spoke_3x20 +0.66.

### Primitive 2: orthogonal_align (after edge-equalize)

Pseudocode: per-edge nudge toward the dominant cardinal axis (vertical
or horizontal). 10 iters, step=0.1.

```
def orthogonal_align(pos, edge_index, iters=10, step=0.1):
    pos = pos.detach().clone()
    src, tgt = edge_index[0], edge_index[1]
    for _ in range(iters):
        diffs = pos[tgt] - pos[src]
        # for each edge, decide vertical or horizontal based on |dx| vs |dy|
        is_vert = diffs[:, 1].abs() > diffs[:, 0].abs()
        # nudge x toward equal for vertical edges; nudge y toward equal for
        # horizontal edges
        delta = torch.zeros_like(diffs)
        delta[is_vert, 0] = -diffs[is_vert, 0] * step
        delta[~is_vert, 1] = -diffs[~is_vert, 1] * step
        pos.index_add_(0, src, delta * 0.5)
        pos.index_add_(0, tgt, -delta * 0.5)
    return pos
```

Expected wins: multiscale_skip_cascade +7.73, weighted_clusters_3x10
+4.18, hub_skip_superfan +2.64, sbm_4x30 +2.62, er_100 +1.71,
residual_block +0.90 (over existing polish). 14 graph wins total.

### Primitive 3: overlap_jitter (recovery primitive)

Pairwise push for overlapping nodes. Helps recover the "no overlaps"
10-pt bin on 43 graphs already-passing the baseline. Doesn't beat
edge_equalize directly, but is a safety net when other primitives push
nodes too close.

```
def overlap_jitter(pos, node_sizes, padding=2.0, iters=5, step=0.5):
    n = pos.shape[0]
    for _ in range(iters):
        diffs = pos.unsqueeze(0) - pos.unsqueeze(1)  # [n, n, 2]
        dx = diffs[..., 0].abs()
        dy = diffs[..., 1].abs()
        half_w = (node_sizes[:, 0:1] + node_sizes[:, 0:1].T) / 2 + padding
        half_h = (node_sizes[:, 1:2] + node_sizes[:, 1:2].T) / 2 + padding
        ix = (half_w - dx).clamp(min=0)
        iy = (half_h - dy).clamp(min=0)
        # push nodes apart in direction of less overlap
        push = ix.unsqueeze(-1) * torch.sign(diffs)
        push[range(n), range(n)] = 0  # no self
        pos = pos + push.sum(dim=0) * step
    return pos
```

### Primitive 4: 2-opt anti-crossing swap (Cluster 1 from Area C)

For graphs n<=200 with edges<=400 and 1+ crossings, try swapping x of
adjacent rank-pairs that participate in a crossing. Score under
composite. Stop after 50 swaps.

```
def swap_2opt_anti_crossing(pos, edge_index, max_swaps=50):
    pos = pos.detach().clone()
    # detect adjacent rank pairs (same y-band) that cross
    # for each, try swapping their x
    # if composite improves, accept; else revert
    ...
```

Expected wins: weighted_clusters_3x10 +5, densenet_block flips tie to
win, multi_component_80 +1.

### Primitive 5: per-layer-x K-means (Area A, lattice)

For graphs that pass `layer_width_cv <= 0.30 AND 1.2 <= e/n <= 2.0 AND
24 <= num_nodes <= 400`, run 1-D K-means per layer on x with K =
round(median_layer_width). Replace x-coords with cluster centroids.

Expected wins: hexagonal_lattice_42 from -2.52 to ~+2.

## Constraints

- All 5 primitives go into `_POLISH_SETTINGS` as candidates of
  `_best_of_polish`. Picker handles regressions (margin 0.5).
- For 4 (anti-crossing), gate on `n <= 200` to bound runtime.
- For 5 (K-means), gate on the topology classifier check; only fire on
  lattice-like inputs.
- Don't change the existing 7 edge-equalize settings.
- Tests:
  - All existing 225 layout tests continue passing.
  - Add a regression test that asserts hex_lattice score > 88 (with the
    K-means primitive enabled) and < 100 (sanity).

## Out of scope for this implementation

- Per-component pipeline picker (Area F's multi_component_80 fix). That's
  a bigger architectural change; do as a separate sprint-21b.
- composite_auto rewiring for petersen (Area B). Metric change requires
  benchmark recalibration.
- Modern literature recommendations (Area D). Larger time investment.
