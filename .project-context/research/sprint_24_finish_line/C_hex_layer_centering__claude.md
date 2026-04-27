# Sprint 24 Area C -- Lattice layer-centering for hexagonal_lattice_42 (Claude)

## TL;DR

**Do not ship either variant. The +0.13 gap on hexagonal_lattice_42 is
not closable by post-processing layer-shift transforms; it is a metric-
definition floor.** Empirical results across both proposed variants and
six adjacent perturbations all regress hex_42 by 1.6 to 13 points.

The reverse-engineering check uncovered a different diagnosis from the
sprint-23 area B handoff. The "inter-layer centering" framing was
correct *for the LP candidate* (sprint-22c LP zigzags layer midpoints
between -100 and +97), but it was wrong about why dagua-HEAD loses to
graphviz_dot. The actual gap on `hexagonal_lattice_42` at HEAD is
**aspect ratio**: dagua's pipeline produces a 477x8997 column-stack
(aspect 0.05), graphviz_dot produces a 432x792 honeycomb (aspect 0.55).
Per-metric, dagua wins on edge_straightness (3.07 deg vs 17.42),
angular resolution (median 6.4 deg vs 53.1 deg), and layer_spacing_cv
(0.28 vs 0.0) -- because every node sits on its own y-row in dagua's
output and edges run nearly vertically. dagua loses on edge_length_cv
(0.42 vs 0.10), neighborhood_mean (0.41 vs 0.75), and stress (0.81 vs
0.83) -- the vertical column makes diagonal hex edges very long.

The composite weights these in dagua's favor by +14 points on
straightness/angular and dot's favor by +24 points on
length/neighborhood/stress, netting a -0.63 dagua loss. **Any layer-
shift on top of dagua-HEAD positions trades straightness for
length-CV at a 5-15 point per-1-point penalty**, so every perturbation
makes the composite worse.

The sprint-22c LP candidate, which DOES have layered structure
suitable for centering, scores 88.187 by itself -- already below
dagua-HEAD's 88.355 and the 88.49 tie threshold. Centering the LP
layers (V2) drops it to 86.22; staggering alternate rows (V1) drops it
to 83.00; the monotonic shear that mimics dot's actual parallelogram
drift drops it to 80-85 depending on shear amount.

The smallest gap of the three sprint-24 blockers is, paradoxically,
the structurally hardest. It needs either (1) a different metric that
penalises aspect-ratio extremes more aggressively, or (2) an entirely
new layout candidate (LP plus a hex-aware coordinate solver) that
beats dagua-HEAD at composite, not a post-pass on top of it.

## Diagnosis: it is aspect ratio, not layer centering

### Per-metric breakdown for hexagonal_lattice_42 (HEAD seed=42)

| metric | dagua | graphviz_dot | diff (dagua - dot) |
|---|---:|---:|---:|
| composite_score | 88.3545 | 88.9864 | -0.6319 |
| aspect_ratio | 0.0531 | 0.5763 | -0.5232 |
| bbox_width | 477.5 | 476.0 | +1.5 |
| bbox_height | 8997.4 | 826.0 | +8171 |
| edge_length_cv | 0.4197 | 0.0991 | +0.3206 |
| edge_straightness_mean_deg | 3.07 | 17.42 | -14.35 |
| angular_res_median_deg | 6.44 | 53.13 | -46.69 |
| angular_res_below_10deg | 0.65 | 0.00 | +0.65 |
| crossing_rate | 0.000 | 0.000 | 0.000 |
| sampled_stress | 0.806 | 0.833 | -0.027 |
| depth_spearman_rho | 0.995 | 0.823 | +0.173 |
| layer_spacing_cv | 0.282 | 0.000 | +0.282 |
| neighborhood_mean | 0.412 | 0.752 | -0.341 |

dagua's bbox is **10.9x taller than dot's** while having essentially
the same width. Edges that should be ~170 long (one hex pitch) are
running 230-1925 long with mean 1034. Dot's edges are ~72 long with
much tighter CV. The "+0.13 to tie" gap is the residual after
straightness and angular dominate the favorable side.

### Why this is not a layer-centering gap

The sprint-23 area B research correctly identified that the **LP-22c
candidate** has a left-aligned layer pattern versus dot's centered
layers. That observation was right for the LP candidate considered in
isolation (88.187 LP-22c vs 88.986 dot, gap of 0.80). But by the time
the picker chooses, dagua-HEAD has selected a **different candidate
entirely** -- one where there are no horizontal layers because every
node is on its own y-row at a stretched scale. There is nothing to
center.

I verified this by grouping HEAD positions by y at multiple tolerances
(1.0, 10.0, 50.0 units) -- only at >=900-unit tolerance do nodes
collapse into 12 layers, and the layer median shifts at that
granularity are sub-pixel.

### What graphviz_dot actually does on hex_42

Comparing layer midpoints sorted by y:

```
LP layer midpoints:  97 | 64 | 31 | -1 | 31 | -34 | -1 | -34 | -34 | -100 | -67 | -67
dot layer midpoints: 117 | 117 | 171 | 171 | 216 | 207 | 279 | 279 | 315 | 279 | 315 | 315
```

Dot's midpoints **drift monotonically rightward by ~200 over 12
layers** -- this is a sheared parallelogram, the classic offset-
coordinate hex rendering, not a centered layout. The "alt-row half-
pitch stagger" hypothesis from B-claude's fallback was qualitatively
wrong: dot uses a per-layer monotonic shift, not a parity-toggled one.
A faithful "hex shear" candidate is therefore Variant 1' below.

## Algorithm sketches

### Variant 1': hex monotonic shear on LP candidate

```python
def hex_monotonic_shear(pos, ei, n, frac=0.5):
    """Add per-rank monotonic x-shift of frac * pitch to LP output."""
    if not is_honeycomb(ei, n):              # gate (~25 LOC)
        return pos
    out = pos.detach().clone()
    layers = group_by_y(out, tol=1.0)
    if len(layers) < 4: return out
    # Median pitch from intra-layer min-gaps
    gaps = [b - a
            for layer in layers if len(layer) >= 2
            for a, b in zip(sorted(out[i, 0].item() for i in layer),
                            sorted(out[i, 0].item() for i in layer)[1:])
            if b - a > 1e-6]
    if not gaps: return out
    pitch = sorted(gaps)[len(gaps) // 2]
    layers_sorted = sorted(layers, key=lambda lyr: out[lyr[0], 1].item())
    for r, layer in enumerate(layers_sorted):
        for i in layer:
            out[i, 0] += r * pitch * frac
    return out - out.mean(dim=0, keepdim=True)
```

`is_honeycomb`: degrees subset of {1,2,3}, no triangles, no 4-cycles
(Pajek-style girth >= 6 check). ~25 LOC. Total ~55 LOC.

### Variant 2: lattice BK layer-center on LP candidate

```python
def lattice_bk_layer_center(pos, ei, n):
    """Per-layer additive shift so each layer's median x = global median."""
    if not _should_dot_lattice_lp(ei, n): return pos
    out = pos.detach().clone()
    layers = group_by_y(out, tol=1.0)
    if len(layers) < 3: return out
    g = float(out[:, 0].median())
    for layer in layers:
        m = float(out[layer, 0].median())
        for i in layer:
            out[i, 0] += (g - m)
    return out - out.mean(dim=0, keepdim=True)
```

~40 LOC including helper. Wider gate via existing
`_should_dot_lattice_lp`.

### Group-by-y helper (shared)

```python
def group_by_y(pos, tol=1.0):
    ys = pos[:, 1].tolist()
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    layers, cur, cur_y = [], [], None
    for i in order:
        if cur_y is None or abs(ys[i] - cur_y) <= tol:
            cur.append(i); cur_y = ys[i] if cur_y is None else cur_y
        else:
            layers.append(cur); cur = [i]; cur_y = ys[i]
    if cur: layers.append(cur)
    return layers
```

## Empirical table

Methodology: dagua score = `composite(full(engine_layout(g,
LayoutConfig(seed=42)), ei, node_sizes=g.node_sizes))`. Competitors
loaded from `eval_output/variant_bench_full/positions/`. Each variant
applied to either dagua-HEAD positions (`*_pipe`) or to the LP-22c
candidate output (`lp_*`).

### Hex_42 candidate ladder

| Source | Variant | Composite | delta vs dot (88.99) | delta vs HEAD (88.355) |
|---|---|---:|---:|---:|
| dagua-HEAD pipeline | base | 88.355 | -0.632 | 0.000 |
| dagua-HEAD pipeline | V1 hex_alt_stagger | 88.355 | -0.632 | 0.000 |
| dagua-HEAD pipeline | V2 layer_median_center | 78.889 | -10.097 | -9.466 |
| dagua-HEAD pipeline | V2b layer_mean_center | 78.099 | -10.887 | -10.256 |
| dagua-HEAD pipeline | V2c layer_midspan_center | 78.889 | -10.097 | -9.466 |
| dagua-HEAD pipeline | y_compress 0.95 | 78.340 | -10.646 | -10.015 |
| dagua-HEAD pipeline | y_compress 0.50 | 78.118 | -10.868 | -10.237 |
| dagua-HEAD pipeline | y_compress 0.10 | 75.924 | -13.062 | -12.431 |
| LP-22c | base | 88.187 | -0.800 | -0.168 |
| LP-22c | V1 hex_alt_stagger | 82.999 | -5.987 | -5.355 |
| LP-22c | V1' shear -0.10 | 87.676 | -1.310 | -0.679 |
| LP-22c | V1' shear +0.05 | 87.617 | -1.370 | -0.738 |
| LP-22c | V1' shear +0.25 | 85.391 | -3.595 | -2.963 |
| LP-22c | V1' shear +0.50 | 82.999 | -5.987 | -5.355 |
| LP-22c | V1' shear +1.00 | 80.011 | -8.975 | -8.343 |
| LP-22c | V2 layer_median_center | 86.216 | -2.770 | -2.139 |

**Best non-base candidate: 88.355 (V1 on dagua-HEAD, equivalent to
no-op because hex_alt_stagger collapses on a y-unique layout).**
Nothing exceeds 88.49 tie threshold. Nothing exceeds the picker margin
of +0.1.

### Cross-graph sanity

For each protected graph, "winner" is the highest-scoring candidate
across {pipeline base, LP, LP+shear x{0.25,0.5,0.75,1.0}, LP alt-
stagger, LP layer-median}.

| Graph | n | base | best_competitor | best_eng | winner_variant | winner_score | shipped delta |
|---|---:|---:|---:|---|---|---:|---:|
| hexagonal_lattice_42 | 42 | 88.355 | 88.986 | graphviz_dot | base | 88.355 | 0.000 |
| triangular_lattice_36 | 36 | 86.607 | 87.086 | graphviz_dot | base | 86.607 | 0.000 |
| grid_5x5 | 25 | 94.136 | 91.597 | graphviz_dot | base | 94.136 | 0.000 |
| sierpinski_42 | 42 | 85.576 | 84.290 | graphviz_dot | base | 85.576 | 0.000 |
| planar_60 | 60 | 80.089 | 75.115 | graphviz_dot | base | 80.089 | 0.000 |
| parallel_multiedge_bundle | 3 | 85.500 | 85.501 | graphviz_dot | base | 85.500 | 0.000 |
| dependency_500 | 500 | 57.884 | 58.189 | elk_layered | base | 57.884 | 0.000 |

Across the seven evaluated graphs, **the picker would select the base
pipeline output every time**. No variant fires. No regression risk
either. But also no lift.

## Decision: ship neither variant

The +0.13 lift required to flip hex_42 to tied is **not achievable
through any layer-shift transform** on top of either dagua-HEAD or
LP-22c candidates. I tried:

1. V1 alt-row half-pitch stagger (PROMPT C variant 1).
2. V2 per-layer median center (PROMPT C variant 2).
3. V2b per-layer mean center.
4. V2c per-layer midspan center.
5. V1' monotonic shear (the actual transform dot uses, with frac in
   {-1.0, -0.75, -0.5, -0.25, -0.1, +0.05, +0.10, +0.15, +0.20, +0.25,
   +0.50, +0.75, +1.00}).
6. y-axis compression to multiple aspect targets.
7. y-compression composed with V2 layer center.

None reach 88.49. Most regress by 1.5-13 points on hex_42, and several
also regress on triangular_lattice_36, sierpinski_42, and grid_5x5.

The honest finding is:

- **The LP candidate IS layered**, has midpoints near the global
  median (zigzagging in [-100, +97]), and reaches 88.187. Centering
  shifts the median-zero zigzag into a nearly-flat midpoint sequence,
  which gains ~0 edge_length_cv but costs ~2 points on straightness
  because the diagonal edges become shorter and steeper.
- **The dagua-HEAD candidate is NOT layered** (each node has a unique
  y), so layer-centering is a no-op. Y-compression to dot's aspect
  costs more straightness than it gains in CV.
- **Dot's hex layout uses a sheared parallelogram** (monotonic
  midpoint drift, not parity stagger). Replicating this on LP-22c at
  any shear fraction tested still loses 1.3-9 points.

This is **the metric-definition floor B-claude warned about in the
sprint-23 dispatch**. The composite weighting between
straightness/angular and length-CV/neighborhood is exactly tuned such
that dagua's tall-stack layout and dot's compact-hex layout score
within 0.63 of each other -- and any geometric perturbation breaks
the balance in the wrong direction.

### What would actually close the gap

To flip hex_42 to tied, you need one of:

1. **A new candidate** that scores >88.49 in its own right, not a
   transform of an existing one. The most promising would be a
   hex-aware coordinate solver: detect honeycomb, lay out in offset
   coordinates with explicit pitch and stagger from the start (not as
   a post-pass), and let the picker choose. Estimate: 250-400 LOC, on
   the same order as Bet A, not a 40-60 LOC quick fix.
2. **A composite re-weighting** that pushes edge_length_cv and
   neighborhood weight higher, or aspect_ratio penalty higher. This is
   sprint-23 area F territory and would touch many graphs.
3. **Different baseline scoring**: if the metric used `quick(...)`
   instead of `full(...)`, or if `crossing_samples` were lowered, the
   absolute composite values would shift and the deltas might flip.
   Worth probing as a sanity check for sprint-24 area F.

If sprint-24 must ship something on Area C, my recommendation is to
ship **nothing for Area C** and accept hex_42 at 90/93 best-or-tied
plus this graph as a documented metric-definition edge case. The +0.13
gap is genuinely below the noise floor of these aesthetic
loss functions.

## LOC estimate (for the record, not recommended)

If the architect insists on shipping despite the negative empirical
evidence:

- **Variant 2 (lattice BK layer-center)**: ~50 LOC including gate
  helper and the `group_by_y` utility. Slot in
  `dagua/layout/ops/pipelines/dagua_native.py` after `_dot_lattice_lp`
  (around line 1210). Wire into `_polish_candidates` near line 2474
  alongside the existing LP candidate. Add a 3-4 entry unit test in
  `tests/layout/ops/pipelines/test_dot_lattice_lp.py`.

- **Variant 1' (hex monotonic shear)**: ~80 LOC including the
  `is_honeycomb` topology gate (girth check), the `group_by_y`
  utility, and the shear routine. Slot in same file, same wiring.

Combined picker fall-through: ~110 LOC total. **Predicted impact
based on empirical results: 0 graphs flipped. Net delta on the picker
gate: zero (variants never exceed +0.1 margin on any of the seven
tested graphs).**

## Methodology / artifacts

- `/tmp/sprint24_c_claude/score_full_pipeline.py` -- full-pipeline
  scoring with V1, V2, V2b, V2c, combined.
- `/tmp/sprint24_c_claude/test_aspect_rescale.py` -- aspect ratio
  rescale probe (showed compression hurts).
- `/tmp/sprint24_c_claude/test_hex_shear.py` -- monotonic shear at
  multiple fractions on top of LP and base pipeline.
- `/tmp/sprint24_c_claude/results_pipeline.json`,
  `aspect_results.json`, `shear_results.json` -- raw scoring output.

All variants implemented purely in user-space scripts; `dagua/`
unchanged. Scoring uses `dagua.metrics.composite(full(...))` matching
`/tmp/h2h_buckets.py` (the production picker scoring path) and
`graph.compute_node_sizes()` per-graph defaults rather than the prompt
spec's blanket [40, 20] -- the blanket sizes give a different
composite (LP=84.56, base=85.45 with [40,20]) that does not match the
dagua=88.35 target reported in CONTEXT.md.

## Final verdict

**Ship neither V1 nor V2.** Document hex_42 as a metric-definition
floor: dagua-HEAD's 88.355 is within 0.63 of graphviz_dot's 88.986,
and the gap is dominated by aspect-ratio effects that no layer-shift
transform can address. Spend the sprint-24 LOC budget on Bets A
(petersen, +2.22 needed) and B (clustered_medium_5x20, +0.91 needed),
where the structural fix is real.

If the architect accepts 92/93 (98.9%) best-or-tied with hex_42 as the
documented exception, this is achievable. If the architect insists on
93/93 = 100%, the path is a **new hex-aware candidate** (250-400 LOC,
sprint-25 territory), not a 40-60 LOC layer-shift on top of existing
candidates.
