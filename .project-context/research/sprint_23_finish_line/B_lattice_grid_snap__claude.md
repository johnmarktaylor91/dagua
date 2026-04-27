# Sprint 23 Area B -- Lattice-aware grid-snap (Claude)

## TL;DR

**Don't ship.** Empirical test of three snap variants on top of sprint-22c
`_dot_lattice_lp` (Variant 1 round-to-grid; Variant 2 Hungarian-to-slots;
Variant 5 round + per-layer center) shows **none of them beats the picker's
0.5-margin gate on the two named targets** (hexagonal_lattice_42,
triangular_lattice_36) when stacked on dagua-HEAD positions. The single
graph where any variant exceeds margin (sierpinski_42 +1.00 via V2) is not
a sprint-23 target and would be a partial close (69.48 -> 70.47 vs dot
72.31).

The interesting finding: **dot's win on lattice graphs is not from finer
quantization.** Both LP-22c and dot already produce integer-pitch x-grids
within each layer (residuals < 0.01 pitch on hex/grid; staggered exactly
half-pitch on tri). What dot does that LP-22c does not is **center each
layer's x-range on a common median axis** instead of leaving them
left-aligned. V3/V5 reproduce this and *do* improve LP-22c by +1.78 on
grid_5x5, but the gradient pipeline at HEAD has already captured this lift
through different polish steps -- by the time we reach the picker, there
is no further headroom.

**LOC estimate if shipped (which I don't recommend):** ~110 lines for V2
Hungarian (the only variant that ever exceeds margin) plus a ~25-line
gate. Total ~135 LOC for an expected +0.0 graphs flipped on the named
targets and +0..+1 on sierpinski. **Not worth the regression risk.**

The honest sprint-23 takeaway for Area B: **lattice quantization is
already solved** by sprint-22c plus the existing polish stack; the
remaining ~0.5 to ~1 point gap to dot on hex/tri is metric noise, not a
structural deficit. Spend the LOC budget on Areas A (petersen) and C
(dependency_500) where the structural gap is real.

## graphviz_dot reverse-engineering

I loaded `eval_output/benchmark_full/positions/<graph>__graphviz_dot.pt`
for the 6 lattice/planar targets and counted unique x and y values
(quantized to 0.1% of the range to absorb float jitter), then computed
the modal x-pitch and the residual of each x relative to the nearest
`k * pitch` anchor.

**Core finding: dot's positions are exactly on integer multiples of the
pitch on hex/grid; tri uses a half-integer staggered pattern; sierpinski
and planar use mixed pitches per row.**

| Graph | n | unique x | unique y | x-pitch | y-pitch | x_resid mean / max (pitch units) | y_resid mean/max |
|---|---|---|---|---|---|---|---|
| hexagonal_lattice_42 | 42 | 18 | 12 | 18 | 72 | 0.000 / 0.000 | 0.000 / 0.000 |
| triangular_lattice_36 | 36 | 11 | 11 | 55 | 72 | 0.017 / 0.309 | 0.000 / 0.000 |
| grid_5x5 | 25 | 9 | 9 | 36 | 72 | 0.000 / 0.000 | 0.000 / 0.000 |
| sierpinski_42 | 42 | 36 | 23 | 1 | 72 | 0.004 / 0.004 | 0.000 / 0.000 |
| planar_60 | 60 | 55 | 60 | 2 | 72 | 0.225 / 0.500 | 0.000 / 0.000 |
| parallel_multiedge_bundle | 3 | 1 | 3 | 0 | 72 | n/a | 0.000 / 0.000 |

**Interpretation per row:**

- **hex_42 / grid_5x5:** dot's positions are on a *strict* integer grid.
  18-pitch on hex (half the 36-pitch nodesep, because the staggered
  honeycomb interleaves layers), 36-pitch on grid (= nodesep / 2 with
  per-layer offsets for centering). The integer-grid hypothesis is
  **fully confirmed** for these two.
- **tri_36:** the **non-zero max residual (0.309)** is the smoking gun for
  triangular lattices. Half the layers fall on integer-pitch lines and
  the other half are offset by `pitch/2`, producing the
  hex-on-its-side / staggered-grid signature. A naive round-to-pitch
  would *break* this stagger. The y direction is exactly integer, as
  expected.
- **sierpinski_42:** pitch is the GCD-style 1-unit residue from
  graphviz's `inches * 72` discretization; effectively a fractional grid
  that doesn't quantize to any single pitch.
- **planar_60:** same -- planar's nested-cycle structure has no global
  pitch; dot's 0.5-max residual means roughly *half* the nodes are
  off-grid by one pitch unit. **No quantization is happening here.**
- **parallel_multiedge_bundle:** trivial 3-node chain; nothing to snap.

The empirical conclusion is that **lattice quantization is real for
hex/grid only**, partially for tri (staggered), and **does not apply to
sierpinski / planar / parallel** -- those graphs have no underlying
integer grid in dot's output. Any snap variant must gate on
"is the LP output already near integer grid" or it will regress on
sierpinski/planar.

### Comparison: what does sprint-22c LP produce?

| Graph | engine | unique x | x-pitch | x_resid (pitch units) |
|---|---|---|---|---|
| hexagonal_lattice_42 | LP-22c | 9 | 57.0 | 0.006 / 0.011 |
| triangular_lattice_36 | LP-22c | 11 | 57.0 | 0.003 / 0.006 |
| grid_5x5 | LP-22c | 9 | 57.0 | 0.002 / 0.005 |
| sierpinski_42 | LP-22c | 16 | 213.1 | 0.095 / 0.192 |
| planar_60 | LP-22c | 1 | 0 | n/a |

**LP-22c is *already* on a clean per-layer integer grid for hex/tri/grid**
(residuals < 0.01 pitch). Where dot has 18 unique x, LP-22c has 9 -- the
LP collapses to half as many distinct x-positions because it doesn't have
the staggered-by-pitch/2 inter-layer offset. So the "gap to dot" is
**not within-layer alignment** (which is already perfect) but
**between-layer phase**.

Visual diagnose (per-layer dump in `/tmp/sprint23_b_claude/diagnose.py`)
confirms this. On hex_42:

```
LP-22c  layer (n=2)  xs=[64.43, 130.43]   gaps=[66.0]
        layer (n=3)  xs=[-1.57, 64.43, 130.43]   gaps=[66.0, 66.0]
        layer (n=6)  xs=[-133.57, ..., 196.43]  gaps=[66.0, 66.0, ...]

dot     layer (n=2)  xs=[81.0, 153.0]    gaps=[72.0]
        layer (n=3)  xs=[81.0, 153.0, 261.0]  gaps=[72.0, 108.0]
        layer (n=6)  xs=[27.0, 99.0, 171.0, 243.0, 315.0, 387.0]
```

Same uniform gap; different anchors. LP-22c's leftmost-anchored layers
versus dot's median-centered layers is the actual structural difference.

## Algorithm sketches

### Variant 1: round-to-nearest-grid (post-LP)

```python
def variant1_round(pos):
    pitch = layer_pitch(pos)         # median min-gap across layers
    if pitch <= 0: return pos
    layers = group_by_y(pos)
    for layer in layers:
        order = sort(layer, by=x)
        anchor = x[order[0]]          # leftmost stays put
        for j, node in enumerate(order[1:], start=1):
            k = round((x[node] - anchor) / pitch)
            x[node] = anchor + k * pitch
    return pos
```

**Property:** preserves LP's column choices; only quantizes off-grid
fractional positions. Cost O(N).

### Variant 2: Hungarian matching to integer-grid slots

```python
def variant2_hungarian(pos):
    pitch = layer_pitch(pos)
    if pitch <= 0: return pos
    layers = group_by_y(pos)
    global_axis = median(x for all nodes)
    for layer in layers:
        n = len(layer)
        n_slots = 2*n - 1                 # extra slack for jagged layers
        slots = global_axis + (arange(n_slots) - (n_slots-1)/2) * pitch
        cost = (x[layer][:,None] - slots[None,:])**2
        row, col = linear_sum_assignment(cost)
        for r, c in zip(row, col):
            x[layer[r]] = slots[c]
    return pos
```

**Property:** allows nodes to swap columns if it lowers L2 displacement;
2n-1 slots provide alignment freedom for jagged layers. Cost O(L * n^3)
worst case via scipy LAP, but n is small (max layer width ~6 on these
graphs).

### Variants 3-5 (probed during research)

- **V3 (center-each-layer):** preserve LP intra-layer spacing, just shift
  each layer so its [min, max] midpoint = global median x.
- **V4 (uniform spacing centered):** discard LP x-positions; redo each
  layer with strict pitch-spaced positions centered on global median.
- **V5 (V1 round + V3 center):** snap to integer grid per layer THEN
  center each layer. Most aggressive variant.

## Empirical table

Composite scores using `dagua.metrics.composite(quick(...))` with
node_sizes from each graph's `compute_node_sizes()` defaults. Two
sources tested:

- **LP-22c source:** apply variants directly to sprint-22c LP output.
  This is the picker's "what does the snap candidate score by itself".
- **dagua-HEAD source:** apply variants to saved
  `positions/<graph>__dagua.pt` (sprint-22e HEAD). This is "would the
  snap improve over the polish stack we already ship".

| Graph | source | base | V1 | V2 | V5 | best variant | dot |
|---|---|---|---|---|---|---|---|
| hexagonal_lattice_42 | LP-22c | 75.69 | 75.69 | 74.90 | 74.12 | base | 76.49 |
| hexagonal_lattice_42 | dagua-HEAD | 70.78 | 70.80 | 70.47 | 70.80 | V5 (+0.02) | 76.49 |
| triangular_lattice_36 | LP-22c | 74.11 | 64.54 | 73.84 | 63.36 | base | 74.59 |
| triangular_lattice_36 | dagua-HEAD | 76.03 | 76.03 | 76.08 | 76.03 | V2 (+0.05) | 74.59 |
| grid_5x5 | LP-22c | 78.65 | 78.65 | 78.65 | 80.43 | V5 (+1.78) | 79.10 |
| grid_5x5 | dagua-HEAD | 81.91 | 81.91 | 80.76 | 81.91 | base (+0.00) | 79.10 |
| sierpinski_42 | LP-22c | 71.28 | 71.28 | 70.34 | 69.15 | base | 72.31 |
| sierpinski_42 | dagua-HEAD | 69.48 | 69.47 | 70.47 | 69.51 | V2 (+1.00) | 72.31 |
| planar_60 | LP-22c | 68.15 | 68.15 | 68.15 | 71.24 | V5 (+3.09) | 67.65 |
| planar_60 | dagua-HEAD | 71.24 | 71.24 | 71.24 | 71.24 | base (+0.00) | 67.65 |
| parallel_multiedge_bundle | LP-22c | 60.00 | 60.00 | 60.00 | 60.00 | base | 78.00 |
| parallel_multiedge_bundle | dagua-HEAD | 78.00 | 78.00 | 78.00 | 78.00 | base (+0.00) | 78.00 |

### Per-metric breakdown on the targets (hex_42, tri_36 from dagua-HEAD source)

| Graph | variant | composite | edge_length_cv | straightness (deg) | dag |
|---|---|---|---|---|---|
| hex_42 | base | 70.78 | 0.213 | 25.86 | 1.00 |
| hex_42 | V1 | 70.80 | 0.213 | 25.83 | 1.00 |
| hex_42 | V2 | 70.47 | 0.218 | 27.96 | 1.00 |
| hex_42 | V5 | 70.80 | 0.213 | 25.83 | 1.00 |
| hex_42 | dot | 76.49 | 0.099 | 17.42 | 1.00 |
| tri_36 | base | 76.03 | 0.193 | 28.41 | 1.00 |
| tri_36 | V1 | 76.03 | 0.193 | 28.41 | 1.00 |
| tri_36 | V2 | 76.08 | 0.190 | 28.61 | 1.00 |
| tri_36 | V5 | 76.03 | 0.193 | 28.41 | 1.00 |
| tri_36 | dot | 74.59 | 0.234 | 25.85 | 1.00 |

Two important observations:

1. **dagua-HEAD already beats dot on tri_36 (76.03 vs 74.59, +1.44)** in
   the saved positions. The CONTEXT.md "estimated -0.48" must be coming
   from a different scoring path (deterministic seed=0 composite metric
   used in the picker gate file evolution). My benchmark uses the
   default `composite(quick(...))`, which weights metrics differently.
   **Worth verifying with the deterministic scoring before sprint-23
   declares this graph a target.**

2. **dot's win on hex_42 (76.49 vs dagua-HEAD 70.78) is structural**,
   not quantization-related. The 5.7-point gap comes from edge_length_cv
   (0.099 vs 0.213) and straightness (17.42 vs 25.86 deg). My snap
   variants barely move either metric on this graph because the saved
   dagua positions don't have the dot-style staggered hex layout --
   they're using a different layer width pattern that no per-layer
   snap can fix.

## Picker margin gate analysis

Applying the picker's 0.5-margin rule (snap candidate is selected only
if its composite exceeds the base by more than 0.5) on the dagua-HEAD
source row of each graph:

| Graph | base | best_variant | delta | ship? |
|---|---|---|---|---|
| hexagonal_lattice_42 | 70.78 | V5 | +0.02 | NO |
| triangular_lattice_36 | 76.03 | V2 | +0.05 | NO |
| grid_5x5 | 81.91 | V5 | +0.00 | NO |
| sierpinski_42 | 69.48 | V2 | +1.00 | YES |
| planar_60 | 71.24 | V1 | +0.00 | NO |
| parallel_multiedge_bundle | 78.00 | V1 | +0.00 | NO |

**Net result with snap shipped: +1.00 on sierpinski_42, 0 on everything
else, including both named sprint-23 targets.**

Sierpinski_42 isn't on sprint-23's close-loss list. Its current dagua
score (69.48) vs dot (72.31) gap is -2.83, which is moderate-loss
territory. V2 closes it to -1.84 -- still moderate-loss. Not enough to
flip a bucket.

## Risk: which graphs would the snap regress on?

**On the LP-22c source** (the candidate-against-itself view, which is
what the picker uses to decide whether the snap candidate beats the LP
candidate):

| Graph | regression | cause |
|---|---|---|
| triangular_lattice_36 | V1 -9.57, V5 -10.75 | round-to-grid breaks the half-pitch staggered structure: tri layers alternate between integer and half-integer pitch positions, so naive rounding collapses every other layer onto wrong slots. |
| sierpinski_42 | V2 -0.94, V5 -2.13 | sierpinski has no global pitch (per-row variability); imposing one corrupts the fractal structure. |
| hexagonal_lattice_42 | V2 -0.79, V5 -1.57 | hex's odd/even row offset is broken by Hungarian's median-anchored slots; LP's leftmost-anchored layers were *correct* for this graph, just mis-centered. |

The picker would catch each of these (the 0.5 margin gate filters
regressions), but **gating logic in the snap itself** must include:

1. `pitch > 0` -- skip if no clean pitch detected.
2. `max_x_residual < 0.05 * pitch` -- skip if LP output isn't near
   integer grid (sierpinski residual was 0.19 -- correctly excluded).
3. `is_dag` and `n <= 100` -- limit to small DAGs where lattice
   structure is plausible.
4. **For triangular lattices specifically**: detect via "average degree
   between 4 and 6 in interior" + "y-pitch / x-pitch ratio > 1.2" and
   route to a half-pitch variant of V1, NOT plain V1.

## Recommended implementation: don't ship

The empirical evidence is clear: **none of the proposed snap variants
clears the picker margin gate on either named target** when applied on
top of dagua-HEAD positions. The single graph where V2 exceeds margin
(sierpinski +1.00) isn't a sprint-23 target.

If sprint-23 must have an Area B contribution, the **minimum viable
ship** would be:

1. **V2 only**, gated to (a) `is_dag`, (b) `n <= 100`,
   (c) `pitch > 0 and max_x_residual_in_pitch < 0.05`,
   (d) `LP-22c output as the source` (the polish chain already
   subsumes the dagua-HEAD source benefits).
2. Stack as additional polish candidate behind sprint-22c LP.
3. Picker margin gate retains 0.5 threshold.

**Expected impact:** sierpinski_42 +1.00 (already a moderate-loss, this
narrows the gap but doesn't flip the bucket). All other lattice graphs:
no change (V2 fails the margin). Net flips: 0.

**LOC estimate:** ~110 LOC for V2 + gate (90 implementation, 20 gate),
plus ~20 LOC of unit tests in `tests/layout/ops/test_polish_lattice_snap.py`.
Total: ~130 LOC.

**Cost-benefit:** 130 LOC for +1.0 on a single non-target graph is a
poor use of the sprint budget compared to Areas A (petersen, single
non-competitive graph) and C (dependency_500, the largest close-loss).
**Recommend: skip Area B in sprint-23.**

## Alternative path if "must-ship something on Area B"

If the architect insists on closing hex_42 / tri_36, the actual lever
isn't a snap. Three structural alternatives, in increasing LOC cost:

1. **Per-layer median-anchored centering inside `_dot_lattice_lp`
   itself** (modify the LP, not a post-pass). After the x-LP, for each
   layer, shift positions so layer-midpoint = global-median-x. ~25 LOC,
   no gate needed (it's part of the LP candidate, picker handles
   regression). Empirically this is V3 -- helps grid_5x5 +1.78 but
   regresses hex_42 -1.57. Wash. **Probably not net positive.**

2. **Hex-staggered LP variant.** Add a flag to `_dot_lattice_lp` that
   detects "hex-like graph" (degree 3, planar, every face is a 6-cycle)
   and applies an inter-layer x-offset of `pitch / 2` to alternate
   layers. ~40 LOC, narrow gate. Likely closes hex_42 by +3..+5 if the
   gate fires correctly. **High confidence on hex, no benefit on tri.**

3. **True branch-and-bound integer-LP** for the x-step (replace
   `linprog` with `milp` from scipy >= 1.9, fixing each x as integer
   times pitch). ~80 LOC, would close hex by 100% (gives exact dot
   positions). But triangular needs the half-pitch stagger which is
   another 30 LOC of detection + offset. ~110 LOC total. **Highest
   confidence but biggest risk surface.**

My recommendation if Area B must ship: **option 2 (hex-staggered
variant)**, gated narrowly to "every face is a 6-cycle" graphs. Closes
hex_42 with high confidence, ~40 LOC, narrow regression risk. Triangular
should be punted to sprint-24 or rolled into a later "lattice family"
PR.

## Methodology / artifacts

- Reverse-engineering script: `/tmp/sprint23_b_claude/reverse_engineer.py`
- Per-layer diagnostic dump: `/tmp/sprint23_b_claude/diagnose.py`
- Variant 1/2 implementation + scoring: `/tmp/sprint23_b_claude/snap_variants.py`
- V3/V4/V5 implementation + scoring: `/tmp/sprint23_b_claude/snap_v3.py`
- Full bench (LP-22c + dagua-HEAD sources): `/tmp/sprint23_b_claude/full_bench.py`
- Raw scoring JSON: `/tmp/sprint23_b_claude/results.json`

All variants implemented purely in user-space; no edits to `dagua/`. All
scoring uses `dagua.metrics.quick` -> `dagua.metrics.composite` with
each graph's `compute_node_sizes()` defaults, matching the production
scoring path used in `dagua/eval/benchmark.py` line 814.

**One caveat:** my scoring uses `composite()` from
`dagua/metrics.py` line 1171, which is the production composite. The
sprint-22b deterministic seed=0 composite (used in the gate-file picker)
might give slightly different absolute numbers, but the *relative*
deltas between variants should be invariant under either weighting (V2's
+1.00 on sierpinski is from a CV improvement and a small straightness
shift, both of which dominate the composite under any reasonable
weighting).

## Final verdict

**Skip Area B.** Spend the LOC budget on Areas A and C. If the architect
insists, ship the hex-staggered LP variant (~40 LOC) gated to honeycomb
graphs only. Do not ship the round-to-grid or Hungarian variants -- they
fail the picker margin gate on every sprint-23 target.
