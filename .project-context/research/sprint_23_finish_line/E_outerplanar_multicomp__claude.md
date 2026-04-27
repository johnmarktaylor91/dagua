# Sprint-23 Area E: Outerplanar + Multi-Component Finishers (Claude)

## TL;DR

Two candidate polishes were prototyped and empirically scored at HEAD `d27fced`
against six graphs (the two targets plus four protected-win regression
checks). Headline:

| graph | baseline | best polish | delta | competitor target |
|---|---:|---:|---:|---:|
| outerplanar_dag_20            | 72.42 | **72.64** | +0.22 | 73.16 (igraph_sugiyama) |
| multi_component_80            | 74.46 | 74.46     | +0.00 | 75.10 (graphviz_dot) |
| planar_60                     | 80.09 | 80.09     | +0.00 | (protected) |
| sierpinski_42                 | 85.43 | 85.43     | +0.00 | (protected) |
| disconnected_encoder_residual | 86.19 | 86.19     | +0.00 | (protected) |
| parallel_cycles_4x5           | 65.36 | 65.36     | +0.00 | (protected) |

**Recommendation: ship `outer_face_rotation` as a polish candidate (gated
on small-N planar / outerplanar topology, ~80 LOC).** It closes ~30% of
the outerplanar_dag_20 gap with zero regression elsewhere. Skip
`multi_component_arrange` and `multi_component_scale_norm` -- the
existing sprint-22b `_global_depth_align` plus the row-major component
tile already saturate this design space; aggressive scale renormalization
breaks the layout (-17.5 on multi_component_80).

The honest takeaway is that Area E was correctly classified as
**low-risk insurance** in the sprint-23 plan -- the lift is real but
small, and the multi_component_80 finisher requires a deeper structural
change (per-component edge-length harmonization that is aware of
node-size constraints) than fits a sub-100-LOC polish budget.

## Per-metric diagnosis

Both targets are scored under `composite()` (deterministic seed=0,
sprint-22b `da58b14`). Per-metric breakdown for the dagua_native
baseline at HEAD:

### outerplanar_dag_20 (N=20, E=37) -- baseline 72.42

| metric                       | value  | weight | points |
|---|---:|---:|---:|
| dag_consistency              | 1.0000 | 25 | 25.00 |
| edge_length_cv               | 1.0377 | 20 |  0.00 (>1.0 caps) |
| depth_spearman_rho           | 1.0000 | 15 | 15.00 |
| overlap_count                | 0      | 10 | 10.00 |
| edge_straightness_mean_deg   | 14.44  | 10 |  6.79 |
| crossing_rate                | 0.0000 | 10 | 10.00 |
| angular_res_mean_deg         | 25.0   |  5 |  3.13 |
| cluster_mean_sep_ratio       | (no clusters) | 5 | 2.50 |

Sum = 72.42. The single 0/20 slot from `edge_length_cv = 1.04` is the
loss. The fan edges from node 0 (lengths spanning depth 1 to depth 19)
mix with the path edges (uniform unit length) to produce a coefficient
of variation slightly above 1.0. igraph_sugiyama at 73.16 either keeps
edge_length_cv just under 1.0 (recovering ~1 point in the slot) and/or
trades small `edge_straightness` improvement.

### multi_component_80 (N=80, E=81) -- baseline 74.46

| metric                       | value  | weight | points |
|---|---:|---:|---:|
| dag_consistency              | 1.0000 | 25 | 25.00 |
| edge_length_cv               | 1.3121 | 20 |  0.00 (>1.0 caps) |
| depth_spearman_rho           | 0.9980 | 15 | 14.97 |
| overlap_count                | 0      | 10 | 10.00 |
| edge_straightness_mean_deg   | 11.35  | 10 |  7.48 |
| crossing_rate                | 0.0049 | 10 |  9.51 |
| angular_res_mean_deg         | 140.0  |  5 |  5.00 (clipped) |
| cluster_mean_sep_ratio       | (no clusters) | 5 | 2.50 |

Sum ~= 74.46. Same story: `edge_length_cv = 1.31` zeroes the 20-pt
slot. The gap to graphviz_dot (75.10) is a credibility-margin's-worth
of `edge_length_cv` improvement -- enough to recover ~1 point if cv
drops below 1.0. Multi-component edge-length variance comes from the
40-node path having internal edges of one scale while the 10-node
star and the 5-node chain run at different per-component scales.

The `_global_depth_align` op (sprint-22b, da58b14) already aligns
component y-rows to global longest-path-layering, so the y-component
of edge-length variance is already minimized. The residual variance
is in the x-component: each component is laid out by the gradient
pipeline at its own natural x-scale, then row-tiled. Components with
fewer edges relax to wider x-spreads.

## Algorithm sketches

### Polish 1: outer_face_rotation (~70 LOC)

```python
def _outer_face_rotation(pos, edge_index, node_sizes,
                         trials=12, score_fn=None):
    """Try K rigid rotations + 3 reflections about the centroid;
    pick the best by composite.

    For outerplanar / small planar graphs, the gradient pipeline
    locks an arbitrary outer-face orientation. Composite metrics
    (edge_length_cv, edge_straightness_mean_deg) are NOT
    rotation-invariant once axis-aligned node boxes are involved --
    a 30deg rotation can shift apparent edge-length distribution.
    """
    pos = pos.detach()
    n = pos.shape[0]
    if n < 4 or n > 200:                  # gate: small graphs only
        return pos
    if not _is_outerplanar_or_small_planar(edge_index, n):
        return pos
    centroid = pos.mean(dim=0, keepdim=True)
    centered = pos - centroid
    best, best_score = pos, score_fn(pos)
    # Rotations
    for k in range(1, trials):
        theta = math.pi * k / trials      # half-circle suffices
        c, s = math.cos(theta), math.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=pos.dtype)
        cand = centered @ R.t() + centroid
        sc = score_fn(cand)
        if sc > best_score:
            best_score, best = sc, cand
    # Three axis-aligned reflections
    for fx, fy in [(-1, 1), (1, -1), (-1, -1)]:
        R = torch.tensor([[fx, 0.0], [0.0, fy]], dtype=pos.dtype)
        cand = centered @ R.t() + centroid
        sc = score_fn(cand)
        if sc > best_score:
            best_score, best = sc, cand
    return best
```

**Picker gate:** `_is_outerplanar_or_small_planar(edge_index, n)`:
- `n <= 100` (avoid expensive networkx planarity check on big graphs)
- `nx.check_planarity(undirected(G))[0]` is True
- Optionally restrict further: outerplanar test (every face is a
  triangle except the outer face, or equivalently no K_4 / K_{2,3}
  minor). The empirical envelope showed planar_60 (planar but NOT
  outerplanar) is no-op-safe at the existing `+margin` threshold,
  so a strict outerplanar test is not required.

This op is added to `_best_of_polish` in `dagua_native.py` as a
named candidate. The existing margin gate (default 0.5) already
prevents adoption when the rotation lift is below the
metric-noise floor, so regression risk is bounded by the gate.

### Polish 2: multi_component_arrange (~80 LOC)

```python
def _multi_component_arrange(pos, edge_index, node_sizes,
                             score_fn=None):
    """Try inter-component reflection / ordering / row-major-vs-
    column-major tile; pick best.

    Search space (bounded): 2 orderings (size-desc, size-asc) x
    2 major axes (row, col) x 4 column counts (sqrt(K), K/2, 1, K)
    x 4 component-reflections (refl_x x refl_y) = 64 candidates.
    Each candidate rebuilds the tile and runs composite() once;
    cost is ~64 metric calls -> ~1-2s on N <= 200.
    """
    comps = _connected_components(edge_index, n)
    if len(comps) < 3:                    # gate: 3+ components
        return pos
    # Pre-extract per-component bbox + AABB-normalized local coords
    bbox_w, bbox_h, local = [], [], []
    for c in comps:
        idx = torch.tensor(c, dtype=torch.long)
        cp = pos[idx].clone()
        cp -= cp.amin(dim=0, keepdim=True)
        local.append((idx, cp))
        bbox_w.append(float(cp[:, 0].max()) + EPS)
        bbox_h.append(float(cp[:, 1].max()) + EPS)
    gap = max(node_sizes[:, 0].mean() * 1.5,
              node_sizes[:, 1].mean() * 1.5,
              max(max(bbox_w), max(bbox_h)) * 0.05)
    best, best_score = pos, score_fn(pos)
    base_order = sorted(range(len(comps)), key=lambda i: -len(comps[i]))
    for order in [base_order, base_order[::-1]]:
        for major in ("row", "col"):
            for cols in (sqrt_K, K // 2, 1, K):
                for rx, ry in product([False, True], repeat=2):
                    cand = _retile_with_reflections(
                        pos, local, bbox_w, bbox_h, gap,
                        order, major, cols, rx, ry)
                    sc = score_fn(cand)
                    if sc > best_score:
                        best_score, best = sc, cand
    return best
```

**Picker gate:** `len(_connected_components(edge_index, n)) >= 3`.

The two-component case is already covered by `_global_depth_align`
(sprint-22b), which forces y-row alignment and is strictly better
than what the row-major arrange could produce.

## Empirical results

Run with:
- `dagua` from local checkout at `d27fced`
- `dagua.layout(g)` (default pipeline = dagua_native)
- `composite(full(pos, edge_index, node_sizes=node_sizes))` with
  `torch.manual_seed(0)` reset before each `full()` call

| graph                          | baseline | rot_12  | depth_x | scale_norm | arrange  | winner   | regression-ok |
|---|---:|---:|---:|---:|---:|---|---|
| outerplanar_dag_20             | 72.417   | **72.639** | 33.625 | 72.417 | 72.417 | rot_12 (+0.22) | n/a |
| multi_component_80             | 74.461   | 74.463  | 33.293 | 56.964 | 74.461 | BASELINE       | n/a |
| planar_60                      | 80.089   | 80.089  | --     | --     | --      | BASELINE       | yes |
| sierpinski_42                  | 85.426   | 85.426  | --     | --     | --      | BASELINE       | yes |
| disconnected_encoder_residual  | 86.186   | 86.186  | --     | --     | 86.186  | BASELINE       | yes |
| parallel_cycles_4x5            | 65.356   | 65.356  | --     | --     | 65.356  | BASELINE       | yes |

Notes:
- **outer_face_rotation_12** delivers a real but modest +0.22 on the
  outerplanar target. It is no-op on the four protected wins (rotation
  was tried but fell within the 0.01 acceptance threshold). The gap to
  igraph_sugiyama at 73.16 narrows from -0.74 to -0.52; we close ~30%
  of the gap.
- **outerplanar_depth_x** (a hard re-placement using
  longest-path-layering for y and current-x sort tie-break) regresses
  catastrophically (-38 on outerplanar, -41 on multi_component). The
  gradient pipeline output already has well-tuned y placement; hard
  override discards the carefully optimized within-layer order.
- **multi_component_scale_norm** (rescale each component's bbox so its
  median internal edge length matches the global median, then retile)
  regresses by -17.5 on multi_component_80. The rescale shrinks small
  components below the node-size floor, which inflates overlap_count
  and tanks the layout. A node-size-aware version that respects per-
  component minimum spacing would be required, and that is no longer
  an ~80 LOC polish.
- **multi_component_arrange_v2** (64-candidate search over component
  permutations / reflections / tile geometry) is no-op on every graph
  tested. The existing `_tile_component_positions` row-major default
  plus sprint-22b's `_global_depth_align` is already at the
  arrangement optimum.

## Picker decision

**Ship: `_outer_face_rotation` op as a polish candidate in
`_best_of_polish`** (between `overlap_jitter` and
`gap_validated_layer_swaps`).

Gate function:

```python
def _should_outer_face_rotate(edge_index, num_nodes):
    if num_nodes < 4 or num_nodes > 100:
        return False
    if edge_index.numel() == 0:
        return False
    G = _build_undirected_nx(edge_index, num_nodes)
    if not nx.is_connected(G):
        return False
    is_planar, _ = nx.check_planarity(G, counterexample=False)
    return is_planar
```

Margin gate: the existing `_best_of_polish` `margin=0.5` already
prevents adoption when the rotation lift is below noise. The
empirical +0.22 on outerplanar_dag_20 falls below 0.5, so this
candidate would NOT actually win against the baseline under the
current 0.5-margin policy. **To realize the lift, drop the margin
to 0.1 specifically for the rotation candidate** (sprint-22's
margin-tightening pattern from `_dot_lattice_lp`), or accept that
the candidate is a no-op for its primary target until the margin
is loosened.

A safer alternative: add the rotation as a *prefix* applied to
every polish candidate output (rotate-then-score), letting the
existing 0.5 margin pick the rotation+other combination only when
it clears 0.5. This adds 12*K candidate evaluations per graph
(small N, fast), but keeps the outer margin gate intact.

**Skip: `_multi_component_arrange` and `_multi_component_scale_norm`.**
- `arrange` is no-op against the existing tile + `_global_depth_align`.
- `scale_norm` regresses the layout because shrinking small
  components below their node-size minimum violates overlap
  constraints. A node-size-aware scale harmonization is feasible
  but outside the LOC budget.

## Picker integration sketch

```python
# In _best_of_polish, in the polish_candidates list:
(
    "outer_face_rotation",
    lambda pos, edges, sizes: _outer_face_rotation(
        best_edge_pos,            # apply on top of edge_equalize seed
        edges,
        sizes,
        trials=12,
        score_fn=score,
    ),
),
```

Where `best_edge_pos` is the existing variable holding the best
post-edge-equalize seed. The rotation candidate runs after the
edge-equalize sweep and competes under the existing
`+margin` gate. With margin=0.5 it is a documented no-op; with
margin=0.1 it lifts outerplanar_dag_20 by +0.22 toward igraph.

## LOC estimate

- `_outer_face_rotation`: ~30 LOC (rotation loop + reflection sweep)
- `_should_outer_face_rotate`: ~20 LOC (gate)
- Wiring into `_best_of_polish`: ~5 LOC
- Tests (rotation invariance / gate boundary / regression-set
  protection): ~30 LOC
- **Total: ~85 LOC**, all in `dagua_native.py` and a single test file.

For `_multi_component_arrange` (NOT recommended): ~80 LOC for
the search loop + retile helper, but with ZERO empirical lift it
should not be merged.

## Risks and notes

1. **Margin gate.** The empirical +0.22 lift on outerplanar_dag_20
   falls below the default `margin=0.5` floor in `_best_of_polish`.
   Without lowering the margin, this candidate never actually
   wins. The honest scope of Area E is "+0.22 IF margin gate is
   tuned." Sprint-23 PROMPT_F (metric audit + picker) is the
   right venue to revisit margin policy.

2. **Composite metric noise.** The +0.22 lift is barely above the
   sampled-crossing-rate variance bound estimated in sprint-22e.
   Re-measuring after PROMPT_F's metric tightening (5M sample
   crossings or exact crossing count for N <= 200) may either
   amplify the lift or absorb it into noise. Either way, the
   candidate is regression-safe on the four protected wins.

3. **Outerplanarity test.** `nx.check_planarity` runs in O(N) but
   constructs a full embedding object. For N <= 100 this is sub-
   millisecond; for N > 200 it becomes a measurable fraction of
   the polish budget. The gate `n <= 100` is generous: outerplanar
   graphs in the benchmark cap at N=20.

4. **Scale-norm potential.** A node-size-aware version of
   `multi_component_scale_norm` (rescale each component's median
   edge length to a global target subject to per-node minimum
   spacing constraints) might still close the multi_component_80
   gap. This requires a per-component LP or barrier-function
   solve and is firmly outside the ~80-LOC budget. Recommended as
   a sprint-24 candidate if the +0.64 gap remains the highest
   close-loss after sprint-23.

5. **Why outerplanar_depth_x failed.** Hard-overwriting positions
   from longest-path-layering destroys the gradient pipeline's
   carefully-tuned within-layer ordering and absolute spacing.
   The composite penalty is dominated by `edge_length_cv` blowing
   up: when nodes are placed on a strict integer-pitch grid with
   unit-x spacing, the diagonal back-edges from node 0 (depths 2-
   19) span lengths ranging from 1.4 to 19.0, producing a CV well
   above 1.0. The gradient pipeline's continuous relaxation is
   strictly better at edge-length homogenization than any depth-
   driven re-placement, which is why igraph_sugiyama -- whose
   placement IS depth-driven -- only beats dagua_native by 0.74.

## Conclusion

Area E delivers one shippable polish candidate (`_outer_face_rotation`,
~85 LOC, +0.22 on outerplanar_dag_20, regression-clean on four
protected wins). The multi_component_80 gap requires deeper
machinery than fits the insurance-bet budget; recommend punting to
a follow-on sprint with explicit node-size-aware constraints. The
empirical exercise validates the sprint-22 architectural choices
(`_global_depth_align`, `_tile_component_positions` defaults) as
already at-or-near saturation on the multi-component design space.

The 94% best-or-tied -> 96% target from the sprint-23 plan does
NOT achieve the +1 composite per insurance bet that the original
prediction assumed (+0.5..+1 each). Realistic delta from Area E
is **+0.22 on outerplanar_dag_20** (with margin gate tuning) and
**0 on multi_component_80**. Sprint-23's success criteria should
weight Areas A (petersen) and C (dense-DAG ordering) more heavily
than this area's two graphs to meet the >= 96% goal.

## Files

- Scratch harness: `/tmp/sprint23_e_claude/polish_candidates.py`,
  `/tmp/sprint23_e_claude/polish_v2.py`, `/tmp/sprint23_e_claude/quick.py`
- Logs: `/tmp/sprint23_e_claude/polish_run3.log`,
  `/tmp/sprint23_e_claude/v2.log`
- Reference precedent ops: `_overlap_jitter` (line 545),
  `_global_depth_align` (line 1212), `_tutte_cyclic_planar`
  (line 1557), `_best_of_polish` (line 1955), all in
  `/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py`.
