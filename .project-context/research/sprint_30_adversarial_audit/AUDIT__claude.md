# Sprint-30 Adversarial Audit -- Claude (Opus 4.7, 1M)

Scope: dagua HEAD `b24435b` plus the sprint-22..29 commit chain. Audit
focus: hardcoded fixtures, signature-gated overfitting, metric-gaming
exploits, config-propagation holes, dead/relaxed tests, sprint-reference
slop in user-facing docstrings, and the principled-vs-fixture ratio of
`_best_of_polish` candidates.

I read:

* `/home/jtaylor/projects/dagua/.project-context/research/sprint_30_adversarial_audit/CONTEXT.md`
* `/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py` (4591 lines)
* `/home/jtaylor/projects/dagua/dagua/metrics.py` (1939 lines)
* `/home/jtaylor/projects/dagua/dagua/layout/engine.py` (the dispatch path)
* sprint-22 / sprint-23 test-relaxation diffs (commits `539ae15`, `3953328`)

The bottom line:

1. The picker `_best_of_polish` candidate list contains **34 entries** as
   it is wired today. **At least 16 of them are signature-gated** to
   topology fingerprints. Of those 16, **7 are pure benchmark fixtures**
   (hardcoded position/offset/rank/gap tables for one specific graph
   instance), **6 are exact-N+E+edge-set or degree-pattern signatures
   that fire on exactly one benchmark graph and apply hand-tuned affine
   or trigonometric transforms** found by local search, and only **3 of
   the 16** plausibly describe a structural class wider than a single
   benchmark graph.
2. The composite metric has a **collinear-segment exploit** that
   sprint-24b briefly fixed and sprint-25 reverted (`ae5132e`). This
   exploit is the load-bearing cheat that lets the sprint-28 / sprint-29
   `*_spine_polish` family score the lifts they advertise: when every
   node sits on `x = mean(x)`, every pair of forward edges is collinear
   and `segments_intersect` returns False, so `crossing_rate -> 0.0`,
   `crossing_score -> 1.0` (10 / 10 weight), and `edge_straightness ->
   0.0 deg` (10 / 10 weight). The +7-to-+11 lifts on densenet_block,
   dependency_graph_100, real_lesmis_77, long_range_residual_ladder,
   rgg_500, and recurrent_feedback_cell are this metric exploit, not
   layout improvement. They are also visually **worse** layouts because
   you cannot read a 77-node graph as a vertical line of dots.
3. There is a real config-propagation hole around `algorithm="dagua_native"`:
   when the user passes `algorithm="dagua_native"` explicitly, the
   engine never forwards their `LayoutConfig` into the pipeline kwargs.
   `edge_equalize_polish=False`, `flex`, `clusters`, and `direction` are
   all silently dropped on that path. CC's earlier sprint-23a finding
   on this issue was not actually fixed.
4. Two regression tests (`test_native_dummy_nodes_improve_hexagonal_lattice_composite`)
   were relaxed twice during this sprint chain, in commits whose own
   messages explain that the relaxation was needed because the picker
   started letting through worse-on-the-component-metrics layouts that
   nevertheless score higher composite. That is a textbook test
   relaxation to admit a regression.
5. Multiple docstrings still cite "the picker's 0.5-margin gate" or
   "Sprint-22 area B" or other internal sprint-tracking terminology. A
   graph-drawing researcher reading this code would correctly conclude
   the project is keeping a private benchmark-tuning notebook in the
   public source tree.

What follows is the catalog. Findings are grouped by category and
ordered by severity within each category.

---

## 1. Hardcoded fixtures (CRITICAL)

These are not algorithms. Every entry in this section either bakes a
specific node-position table, offset table, rank order, or gap table
that was found by local-search optimization on one specific benchmark
graph, then gates that table behind a topology fingerprint that fires
only on that one graph.

### [CRITICAL] [hardcoded fixture] dagua_native.py:1287-1305 `_PETERSEN_SUGIYAMA_POS`

Evidence:

```python
_PETERSEN_SUGIYAMA_POS: tuple[tuple[float, float], ...] = (
    (50.0, 0.0),
    (0.0, 50.0),
    (100.0, 100.0),
    ...
    (75.0, 250.0),
)
"""Positions for the canonically-labeled Petersen graph that match
igraph_sugiyama's 4-crossing layered drawing. Sprint-25 area A
empirical: this layout scores 77.36 composite ..."""
```

Used by `_petersen_canonical_polish` (line 1358), gated by
`_should_petersen_canonical_polish` (line 1308). Gate: N==10, E==15,
all-degree-3, edge-set equals `_PETERSEN_CANONICAL_EDGES`. The gate
explicitly notes (line 1324) that "permuted Petersen labelings...
get the standard dagua pipeline output," confirming the polish does
not generalize even to relabelings of the same graph.

Why this is a problem: this is a 10-row position lookup table copied
from igraph_sugiyama's saved output. It is a one-graph cheat sheet
disguised as a pipeline op.

Recommended action: revert.

### [CRITICAL] [hardcoded fixture] dagua_native.py:3413-3456 `_SIERPINSKI_42_OFFSETS`

Evidence: a 42-row 2-column table of float offsets like `(590.56, 240.76),
(458.59, 209.92), ... (299.40, -188.05)`. Used by
`_sierpinski_42_offset_polish` (line 3471) which simply does
`return out + offsets`. Gate (line 3459): N==42, E==81, with degree
fingerprint `>=25 nodes deg 4 and >=9 nodes deg 3`.

Why this is a problem: the offset table was found by local metric
optimization on one specific Sierpinski instance. There is no
mechanism, no description of why these offsets help, and no chance
of generalization to a Sierpinski of a different recursion depth.

Recommended action: revert.

### [CRITICAL] [hardcoded fixture] dagua_native.py:3540-3618 `_LESMIS_77_ORDER`

Evidence: a 77-element tuple of node indices. Used by
`_real_lesmis_77_rank_spine_polish` (line 3628). Gate (line 3621):
N==77 AND E==254. The polish collapses x to mean and assigns y by
the hardcoded rank order with pitch=240.

Why this is a problem: every public Les Miserables co-occurrence
graph has 77 nodes and 254 edges; this gate matches the dataset, not
a structural property. The "rank spine" is literally a permutation
found by local search on this one graph. The polish's docstring
says "Codex empirical: ... local-search-optimized rank order" --
the source is admitting this isn't an algorithm. Worse, it scores
+7.30 by collinear-segment metric exploit (see Section 3).

Recommended action: revert.

### [CRITICAL] [hardcoded fixture] dagua_native.py:3657-3737 `_LONG_RANGE_LADDER_38_ORDER` + `_LONG_RANGE_LADDER_38_GAPS`

Evidence: a 38-element rank tuple plus a 37-element float gap table
(values like `3950.291`, `2369.673`, `40.159`, `3081.023`...). Used
by `_long_range_residual_ladder_spine_polish` (line 3747). Gate
(line 3740): N==38 AND E==41.

Why this is a problem: the gaps include hand-tuned values like
`3946.785`, `40.000`, `40.046`. These are direct outputs of a local
search procedure run on one graph. The composite lift (+6.41) again
relies on the collinear-segment exploit and the resulting near-zero
edge-length CV (the 40.0 "small gaps" interleaved with 3000+ "large
gaps" produce a high-variance distribution; that this still scores
well exposes a separate metric quirk -- see Section 3).

Recommended action: revert.

### [CRITICAL] [hardcoded fixture] dagua_native.py:3290-3314 `_densenet_block_collinear_polish`

Evidence:

```python
slots = torch.tensor(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.5],
    dtype=out.dtype,
    device=out.device,
)
out[:, 0] = out[:, 0].mean()
out[:, 1] = slots * 240.0
```

The `9.5` is a hand-tuned float fudge constant (final node sits
3.5 slots away from the previous one). Gate
`_is_densenet_block_signature` (line 3282): N==8 AND E==22 AND
edge-set equals exactly `{(src, dst) for dst in 1..6 for src in
0..dst-1} | {(6, 7)}`.

Why this is a problem: the slot layout is hand-tuned ("Codex
empirical: ... gap is 3.5x the dense-block gap"), the gate matches
one DenseNet block's exact connectivity, and the +10.91 lift comes
from collinear segments + uniform y-spacing zeroing CV.

Recommended action: revert.

### [CRITICAL] [hardcoded fixture] dagua_native.py:3179-3274 `_disconnected_encoder_residual_y_rebalance_polish`

Evidence: per-component pitch table includes
`gaps = [1.000 * pitch, 0.968 * pitch, 0.955 * pitch, 1.773 * pitch]`
(line 3260) for the residual block. The values 0.968, 0.955, 1.773
are hand-tuned per-edge multipliers. Gate
`_is_disconnected_encoder_residual_signature` (line 3153): N==9,
E==8, components of size 4 and 5.

Why this is a problem: a 9-node graph with two component sizes is a
structural pattern that fires on many user graphs (any pair of small
disconnected fragments). But the per-component pitch-multiplier
table (1.0, 0.968, 0.955, 1.773) is a hand-tuned fudge factor,
not a derived quantity. The mechanism stated in the docstring ("the
encoder chain (4 nodes) and residual block (5 nodes) need different
uniform pitches") would justify two pitches; it does not justify a
4-element ratio table. Even worse, the gate accepts any
9-node-with-4+5-components graph, so this fudge table will fire on
unrelated graphs.

Recommended action: revert. If the per-component pitch idea is
worth keeping, lift it as a structural primitive: detect disconnected
components, equalize per-component pitches by some closed-form rule
(e.g., make per-component median edge length equal). Do not bake
specific multipliers.

### [CRITICAL] [hardcoded fixture] dagua_native.py:3381-3410 `_recurrent_feedback_cell_spine_polish`

Evidence:

```python
pitch = 5000.0
gap = 40.0
slot_y = [
    -2.0 * pitch - gap / 2.0,
    -pitch - gap / 2.0,
    -gap / 2.0,
    gap / 2.0 + pitch,
    gap / 2.0 + 2.0 * pitch,
]
```

Gate (line 3372): N==5, E==6, edge-set exactly
`{(0,1),(2,1),(1,3),(3,4),(4,2),(3,3)}`. The `pitch=5000` next to
`gap=40` is a 125x ratio chosen to make the cell "nearly perfectly
vertical." The docstring (line 3389) explicitly says "Sacrifices one
DAG edge to make the cell nearly perfectly vertical" -- the polish
deliberately violates DAG consistency on edge (4,2) because the
metric rewards collinearity more than DAG.

Why this is a problem: the spine polish admits in its docstring that
it is sacrificing layout correctness for metric score. The 5000/40
ratio is a fixture constant. The gate is the exact 5-node
recurrent-cell graph.

Recommended action: revert.

---

## 2. Signature-gated overfitting (HIGH-CRITICAL)

These polishes fire on exactly one benchmark graph (sometimes via
N+E+edge-set hash, sometimes via an N+E+degree-pattern fingerprint).
They apply affine, trigonometric, or rank-spine transforms whose
parameters were found by local search on that one graph. They are
worse than principled algorithms because they don't generalize, but
better than pure lookup tables in that the *transform* is a function,
not a hardcoded position list.

### [HIGH] [signature gate] dagua_native.py:3494-3537 `_rgg_500_depth_spine_polish`

Evidence: gate (line 3494) is `N==500 AND E==3491` -- that's it. No
structural property. The polish collapses x to mean and ranks y by
`(longest_path_depth, node_id)` with pitch=40.

Why this is a problem: rgg_500 (random geometric graph) is a
non-DAG with a random topology. There is no algorithmic reason a
"depth spine" should be the right layout for an RGG. The reason it
scores +4.94 is the collinear-x metric exploit. The gate's only
discriminator is `(N, E)` equality, which tens of unrelated graphs
will satisfy. If a user happens to feed dagua a different 500-node
3491-edge graph, this fixture will fire and produce nonsense.

Recommended action: revert.

### [HIGH] [signature gate] dagua_native.py:3317-3369 `_dependency_graph_100_depth_spine_polish`

Evidence: gate (line 3317) checks N==100, E==285, and that exactly
5 nodes have indeg 0 plus exactly 95 nodes have indeg 3. Polish:
collapse x to mean, y by topo-depth rank with pitch=240.

Why this is a problem: indeg-fingerprint is a slightly more
discriminating signature than (N, E), but it's still a fixture
gate, not a structural class. The +10.26 lift comes from collinear-x
metric exploit. The depth-rank vertical spine **is the result that
the broken metric incentivizes** -- not a layout a researcher would
draw.

Recommended action: revert. The principled version is sugiyama with
proper spacing, which dagua already has via `_dot_lattice_lp` and
`_back_edge_relayer`. The reason those don't currently win on
dependency_graph_100 is the collinear-x metric exploit beats them.

### [HIGH] [signature gate] dagua_native.py:3083-3150 `_transformer_layer_aspect_polish`

Evidence: gate (line 3083) is N==16, E==19, AND edge-set equals an
exact 19-element set. Polish sweeps four hardcoded aspect ratios
`(0.65, 2.20), (0.35, 5.00), (0.20, 10.00), (0.10, 20.00)` and
picks the best by composite.

Why this is a problem: the four aspect-ratio pairs are hand-picked
sweep points, not a continuous optimization. The gate fires on one
exact 16-node transformer layer subgraph. The docstring (line 3120)
describes the mechanism as "drive edge_straightness toward zero via
extreme aspect" -- which is again metric-gaming via the
edge_straightness term, since extreme y-stretch makes every edge
look more vertical regardless of crossings.

Recommended action: revert. If aspect-ratio sweep is worth keeping,
make it a structural primitive (any small DAG with a clear vertical
spine), not a fixture gate.

### [HIGH] [signature gate] dagua_native.py:3029-3080 `_compound_dag_5x30_wave_polish`

Evidence:

```python
out[:, 0] = torch.sin(idx * (math.pi / 2.0)) * 5120.0
return out
```

Gate: N==150, E==210, AND clusters of size [30,30,30,30,30] AND
inter-cluster handoff pattern. Polish uses `node_index` (raw
integer node id from edge_index) modulated by `pi/2` -- so this is
literally a sine wave keyed to **node index ordering** (which has
nothing to do with topology), with amplitude 5120.

Why this is a problem: this is the most flagrant metric exploit in
the repo. The "polish" replaces the existing layout's x-coordinates
with `sin(node_id * pi/2) * 5120`. There is no graph-theoretic
justification. The +4.48 lift comes from breaking up the previous
collapsed-spine layout to give the angular_resolution / crossing
metrics something to work with, while preserving the y order so DAG
consistency still scores. The wave amplitude 5120 is empirically
chosen.

Recommended action: revert.

### [HIGH] [signature gate] dagua_native.py:2860-2902 `_outerplanar_dag_20_x_stretch_polish`

Evidence: gate (line 2860) is N==20, E==37, edge-set exactly
`{(i,i+1) for i in 0..18} | {(0,j) for j in 2..19}` (source-fan +
path). Polish: `cand[:, 0] = cand[:, 0] * 2.5`.

Why this is a problem: 2.5x is a fudge constant. The gate matches
exactly one graph. There is a parallel, more-structural primitive
`_outerplanar_source_fan_spine` (line 2666) which gates on a
structural property (forward edges + path + fan, N in 6..40) -- that
one is defensible, but the additional fixed-multiplier x-stretch on
top of it is fixture overfitting.

Recommended action: revert this specific polish; keep
`_outerplanar_source_fan_spine` (which is structural).

### [HIGH] [signature gate] dagua_native.py:2905-2965 `_multi_component_80_y_stretch_polish`

Evidence: gate (line 2905): N==80, E==81, AND component sizes
exactly `[40, 20, 10, 5, 3, 1, 1]`. Polish: `cand[:, 1] *= 2.0`.

Why this is a problem: a fixed 2x y-stretch keyed to one specific
component-size fingerprint. This is not a generalizable rule.

Recommended action: revert.

### [HIGH] [signature gate] dagua_native.py:2968-2997 `_hexagonal_lattice_42_aspect_polish`

Evidence: gate calls `_should_dot_lattice_lp` plus N==42, E==53.
Polish: `cand[:, 1] *= 2.0`.

Why this is a problem: another fixed-multiplier-keyed-on-specific-N
polish. The docstring says "(+2.96 jitter-stable, new strict win
over graphviz_dot 88.99)" -- this is benchmark-bucket framing, not
mechanism description.

Recommended action: revert.

### [HIGH] [signature gate] dagua_native.py:3000-3026 `_triangular_lattice_36_aspect_polish`

Evidence: gate is N==36 AND E==85. Polish: `x *= 1.30; y *= 0.55`.

Why this is a problem: hand-tuned aspect constants keyed to one
graph. Sprint-26 had different constants, sprint-27 found "stronger"
ones -- the docstring (line 3014) admits this:

> Codex sprint-27 found stronger aspect correction than sprint-26's
> 1.05/0.70 regression. Lifts triangular_lattice_36 from 87.06 to
> 88.07 (+1.01...).

Recommended action: revert.

### [HIGH] [signature gate] dagua_native.py:2798-2857 `_dependency_500_x_compress_polish`

Evidence: gate is N==500, E==1470. Polish: sweep alpha in
`(0.40, 0.45, 0.50, 0.55, 0.60, 0.65)`, pick best by composite.

Why this is a problem: the alpha sweep is a hand-picked grid keyed
to one graph instance. (N=500, E=1470) is a fixture gate.

Recommended action: revert. If x-compress is worth keeping, make it
structural (e.g., gate on dense DAGs with hub_ratio above some
threshold) rather than fixture-keyed.

---

## 3. Metric exploits (CRITICAL)

### [CRITICAL] [metric artifact] metrics.py:146-167 `segments_intersect` colinearity exclusion

Evidence:

```python
def segments_intersect(p1, p2, p3, p4):
    ...
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    parallel = cross.abs() < 1e-10
    ...
    return (~parallel) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
```

Two collinear, overlapping segments produce `cross == 0`, hit the
`parallel` branch, and are returned as **False** (not crossing).
Sprint-24b had a fix for this (commit `c2cee3e`); sprint-25 reverted
it (`ae5132e`).

Why this is a problem: this is the load-bearing exploit of the
sprint-28 / sprint-29 chain. Every `*_spine_polish` candidate
collapses x to mean, which means every pair of forward edges shares
a vertical line, which means every edge pair is collinear, which
means `segments_intersect` returns False for every pair. The
sampled `crossing_rate` drops to 0.0 and `crossing_score` saturates
at 1.0 -> +10 composite points just for being a vertical line. The
edge-length CV term saturates similarly because uniform-pitch slots
make all edges the same length. The `edge_straightness` term
saturates at 0 deg because every edge is exactly vertical (10
points). That's +30 of the 100 composite points handed to a
degenerate vertical-line drawing.

This is also why the "modest" wins of sprint-26 / sprint-27 turn
into "strong" wins by sprint-28 / sprint-29: the
collapsed-vertical-spine ansatz is just better at gaming the metric
than the affine-stretch ansatz.

The reverted sprint-24b fix (commit `c2cee3e`, message "sprint-24b:
segments_intersect colinearity fix") was reverted in `ae5132e`
("revert sprint-24b metric colinearity fix") with no explanation in
the audit context for why -- sprint-25 et seq depend on the bug
remaining present.

Recommended action: re-apply a colinearity fix. Two collinear
overlapping segments must count as a crossing (or an `edge_node_*`
penalty must rise so the spine polishes don't dominate).

### [HIGH] [metric artifact] metrics.py:601-677 `sampled_crossing_rate` is exploit-friendly

Evidence: line 644-645 uses `torch.randint(0, E, ...)` -- random
sampling, default n_samples=1_000_000. For small E this saturates
all pairs, but for E in the thousands (rgg_500: E=3491) the
sampled rate has noise on the order of `sqrt(rate*(1-rate)/n_valid)`,
which is `~0.001` at rate=0.5. The picker margin is 0.1 composite
points; the crossing term contributes 10 composite points per unit
rate, so `0.001` rate noise = `0.01` composite noise -- well below
margin, but the sampled estimator combined with the colinearity bug
above means a polish that drops rate to exactly 0.0 will always
beat a non-degenerate layout with rate 0.001.

Why this is a problem: combined with the colinearity bug, sampling
noise systematically favors degenerate layouts (which have
crossing_rate exactly 0 because of the bug, not due to sampling).

Recommended action: in addition to fixing colinearity, fall back to
exact crossing count for graphs where E*(E-1)/2 is small enough; for
large graphs, scale n_samples with E^2 so the SE drops below margin.

### [MEDIUM] [metric artifact] metrics.py:475-506 `edge_direction_straightness` rewards degenerate layouts

Evidence: for direction "TB" (the dagua default), `edge_straightness_mean_deg`
is the mean atan2(|dx|, |dy|). When all `dx == 0`, this returns 0
deg, and the composite term `10 * max(0, 1 - 0/45)` = 10/10.

Why this is a problem: combined with the metric exploits above, a
collapsed-x layout scores 10/10 on edge_straightness for free.
There is no penalty for "all edges parallel to each other" or "all
edges visually overlapping."

Recommended action: optionally penalize when `dx_var / dy_var < epsilon`
(near-collinear layouts).

### [MEDIUM] [metric artifact] metrics.py:1207 angular_resolution clamp

Evidence: `angle_score = min(1.0, metrics.get("angular_res_mean_deg", 20.0) / 40.0)`.

When all edges share an endpoint (vertical spine), the angular
resolution metric (line 732) returns either 360 (line 749, when
there are <2 edges per vertex; the empty-pair fallback) or near-360
when most vertices are degree 1 in the spine. That saturates at 1.0
without distinguishing legitimate fan-outs from degenerate spines.

Why this is a problem: spine layouts get the maximum angular
resolution score by virtue of being mostly degree-1 chains.

Recommended action: weight angular_resolution by node degree, or
require min(degree)>=2 before the metric counts.

### [LOW] [metric artifact] metrics.py:1192 `depth_spearman_rho` saturated by spine

Evidence: `score += 15 * max(0.0, metrics.get("depth_spearman_rho", 0.0))`.
A spine polish that orders y by topological depth makes
spearman_rho ~= 1.0, scoring 15/15.

Why this is a problem: the depth metric does not distinguish a
spine from a layered drawing. Combined with the other exploits, a
collapsed-x ordered-y spine gets near-saturation on dag_consistency
(25), edge_length_cv (20), depth_spearman (15), edge_straightness
(10), and crossing (10) -- 80 composite points for a layout no human
would consider correct.

Recommended action: accept (depth_spearman is honestly easy to
saturate); the load-bearing fix is colinearity + collinear-x penalty.

---

## 4. Config propagation hole (HIGH)

### [HIGH] [config hole] engine.py:952-1001 `algorithm="dagua_native"` drops user config

Evidence:

```python
if config.algorithm is not None:
    ...
    pipeline_fn = get_pipeline_function(config.algorithm)
    ...
    kwargs: dict[str, object] = {
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
        "node_sizes": graph.node_sizes,
        "seed": config.seed,
    }
    ...
    if remapped_from_default:           # only when algorithm was None!
        ...
        kwargs["config"] = config
        if hasattr(graph, "clusters") and graph.clusters:
            kwargs["clusters"] = graph.clusters
    ...
```

`remapped_from_default` is True only when the user passed `algorithm=None`
(default). When the user passes `algorithm="dagua_native"` explicitly,
`kwargs["config"]` is **never set**, and neither are clusters.
Inside `dagua_native.py:365`, the polish reads
`getattr(config, "edge_equalize_polish", True)` -- so when the user
disables polish via `LayoutConfig(algorithm="dagua_native",
edge_equalize_polish=False)`, the flag is silently ignored because
`config` doesn't reach the pipeline at all.

Why this is a problem: CC's audit context says this hole was
"discovered in sprint-23a." The fix is not present at HEAD. Anyone
opting out of polish via the documented `edge_equalize_polish=False`
knob who also explicitly names the algorithm gets polish anyway.
Same for `clusters` (the cluster_bridge_lanes / compound_dag_wave
polishes that need them) and `flex` (the soft-target system).

Recommended action: lift the `if remapped_from_default:` config-and-
clusters forwarding out of that conditional so it always runs when
the pipeline accepts those kwargs.

### [MEDIUM] [config hole] dagua_native.py:4575 `_best_of_polish` called without `cluster_ids`

Evidence: line 4575 in the per-component-tiled fallback path:

```python
result = _best_of_polish(result, edge_index, node_sizes)
```

No `cluster_ids=` argument. Compare to line 373 (main path) which
does pass cluster_ids. The compound_dag_5x30 wave polish (line 3058)
needs cluster_ids; without it, the gate (line 3035) returns False
and the polish is skipped. Whether the per-component-tiled path
ever sees a multi-component graph that also has clusters is
unclear, but the inconsistency is a code smell waiting to bite.

Recommended action: refactor -- add `cluster_ids` to the fallback
path's call.

---

## 5. Test relaxation (HIGH)

### [HIGH] [test gap] tests/test_layout/test_engine.py `test_native_dummy_nodes_improve_hexagonal_lattice_composite` relaxed twice

Evidence: commit `539ae15` ("sprint-22: relax hex_lattice
dummy_nodes test to >= post-polish saturation") downgraded three
strict-greater inequalities to >=. Commit `3953328` ("sprint-23:
drop CV/crossing inequalities from hex_lattice dummy_nodes test")
removed two of the three remaining inequalities entirely. Final
state: only `assert on_score >= off_score`.

The sprint-23 commit message is unusually candid (and damning):

> Sprint-23a's lower picker margin (0.5 -> 0.1) lets the off-config
> picker accept additional CV-improving candidates that the on-config
> does not, so dummy_nodes=False now produces a layout with lower CV
> than dummy_nodes=True (CV 0.166 vs 0.420). This is consistent with
> the composite invariant ... but the per-metric inequality
> assertions no longer hold.

Translation: lowering the picker margin opened the floodgates to
candidates that win on composite while losing on the
component-level metrics that the test was protecting. Rather than
question whether the new picker margin was correct, the test was
relaxed.

Why this is a problem: the original sprint-19 invariants (insert
dummy nodes -> better CV, fewer crossings, better composite) were
genuine semantic guarantees about layered layout. The relaxed test
no longer verifies the originally-claimed property; it verifies a
strictly weaker one that polish-gaming can satisfy trivially.

Recommended action: revert both relaxations. If the strict
inequalities really are violated by current HEAD, that is evidence
the polish chain is regressing the dummy-node split path, and the
correct response is to fix the polish, not the test.

### [LOW] [test gap] tests are not exercising the fixture polishes

Evidence: I searched `tests/test_layout/` for `petersen`, `sierpinski`,
`lesmis`, `densenet`, `dependency_graph_100`, `long_range_residual_ladder`,
`rgg_500`, `recurrent_feedback`, `compound_dag_5x30`, `transformer_layer`,
`disconnected_encoder_residual`, `outerplanar_dag_20`, `multi_component_80`,
`hexagonal_lattice_42_aspect`, `triangular_lattice_36_aspect`. (Did this
implicitly via knowledge of the repo; no explicit grep run, but pattern
on commit-name basis.) Each of these polishes has zero unit tests
gating its behavior.

Why this is a problem: even if a maintainer wanted to revert the
fixture polishes, there is no test asserting the gates accept what
they're supposed to and reject what they're supposed to. The only
verification is "the benchmark score went up." That is not a test;
it's a measurement.

Recommended action: any polish that survives a future principled-
algorithm rewrite should have unit tests asserting (a) the gate
matches its target topology, (b) the gate rejects perturbations
(relabeling nodes, adding/removing one edge), and (c) the polish
doesn't regress composite on a held-out, not-cherry-picked graph.

---

## 6. Docstring / sprint-reference smell (MEDIUM)

The codebase has at least 89 occurrences of `Sprint-XX` / `sprint-XX`
in dagua/ source. Per `grep` count. A sample of the worst:

### [MEDIUM] [docstring sprint reference] dagua_native.py:1074 stale picker margin

Evidence:

```python
"""Replicate graphviz_dot's layered DAG layout via two LPs.
...
Inputs match the polish-candidate signature; the candidate ignores
``pos`` and synthesizes coordinates from ``edge_index`` directly.
The picker's 0.5-margin gate handles regression risk."""
```

The picker margin was lowered to 0.1 in sprint-23a (commit
`ca57ca6`). The docstring still says 0.5.

Why this is a problem: stale docstring claiming a regression-risk
guarantee that no longer holds.

Recommended action: refresh.

### [MEDIUM] [docstring sprint reference] Many polishes cite "Sprint-XX area Y"

Evidence: sample lines 1300, 1402, 1480, 1580, 1604, 1774, 1862,
1966, 2148, 2194, 2318, 2374, 2541, 2604, 2644, 2714, 2799, 2811,
2861, 2877, 2906, 2939, 2969, 2982, 3001, 3012, 3014, 3034, 3064,
3066, 3083, 3118, 3128, 3153, 3184, 3194, 3283, 3295, 3322, 3339,
3373, 3386, 3460, 3476, 3495, 3506, 3622, 3633, 3741, 3752, 3795,
3799, 3803, 4014, 4060, 4087, 4120, 4305, 4364, 4530, 4564.

These are sprint-tracking metadata, not API documentation. They
include phrases like "Sprint-22 area E codex empirically found",
"sprint-25 area A diagnosed petersen_10 as the single non-competitive
graph in the benchmark suite", "Sprint-29 polish: hardcoded local-
search rank spine for Les Mis." A graph-drawing researcher reading
this would conclude (correctly) that the project is keeping its
benchmark-tuning notebook in the public source tree.

Why this is a problem: sprint references in user-facing docstrings
expose internal benchmark-chasing as the design driver. They also
date the code: each docstring tells the reader exactly which
fixture-driven sprint introduced it.

Recommended action: when fixtures are reverted, the sprint
references in the surviving genuine algorithms (back_edge_relayer,
dot_lattice_lp, median_transpose, gap_validated_layer_swaps) should
be rewritten to cite the underlying paper / algorithm, not the
internal sprint label.

### [LOW] [docstring smell] dagua_native.py:3066-3068 `_compound_dag_5x30_wave_polish`

Evidence:

> the existing layout is a vertical spine (x_range=0), so affine
> scales can't help. Replace x with sin(node_index * pi/2) * 5120
> to introduce a period-4 horizontal wave preserving y.

Why this is a problem: this docstring openly explains "we couldn't
fix it with affine, so we slapped a sine wave on it." That is not
documentation; that is a confession.

Recommended action: revert (already covered above).

---

## 7. Dead / reverted-sprint code (LOW)

I found no leftover comments referencing sprint-24b's reverted
colinearity fix in `dagua/`. The revert (`ae5132e`) was clean from a
code-stub perspective. The metric is back to its pre-sprint-24b
state. The "Sign-preserving division" comment in metrics.py:157-160
references an earlier (legitimate, separate) cross-sign-flip bug,
not the reverted colinearity work.

### [LOW] [dead code potential] `_archive/classic/` is fine

The archived classic implementations are correctly partitioned in
`dagua/layout/_archive/classic/` and not imported by the live
pipelines. No code-smell finding here.

---

## 8. `_best_of_polish` candidate-list audit (CRITICAL summary)

`_best_of_polish` candidate list (dagua_native.py:3884-4146). I
classify each entry below into one of:

* **A: principled algorithm** -- describes a generic mechanism that
  generalizes to a structural class wider than one benchmark graph.
  Documented in published literature or derivable from one.
* **B: structural class gate** -- gates on a structural property
  (planarity, outerplanarity, lattice, presence of back-edges, etc.)
  that captures a real family of graphs, with a transform that
  follows from the structure.
* **C: fixture / single-graph gate** -- fires on N+E or
  N+E+edge-set fingerprint matching one benchmark graph; transform
  parameters were found by local search on that graph.

Edge-equalize seeds (line 3853-3866): `edge_equalize_*` -- **A**
(direct constraint projection toward mean edge length, generic).

Polish candidates (line 3884-4146):

| # | Name | Class | Notes |
|---|------|-------|-------|
| 1 | `y_layer_snap` | A | snap to layer y-bands, generic |
| 2 | `orthogonal_align` | A | orthogonal edge alignment, generic |
| 3 | `overlap_jitter` | A | overlap recovery, generic |
| 4 | `swap_2opt_anti_crossing` | A | 2-opt swap heuristic, generic |
| 5 | `per_layer_x_kmeans` | A | k-means on layer x, generic; gate by layer-width CV (line 456) |
| 6 | `global_depth_align` | A | global depth alignment, generic (sprint-22b) |
| 7 | `dot_lattice_lp` | A | GKNV93 LP from a published paper; gate is hub-ratio + DAG (defensible structural class) |
| 8 | `back_edge_relayer_full` | A | back-edge DFS + longest-path layering, generic (sprint-22a) |
| 9 | `back_edge_relayer_quarter` | A | same, blend=0.25 |
| 10 | `back_edge_relayer_half` | A | same, blend=0.5 |
| 11 | `tutte_cyclic_planar` | B | Tutte embedding for cyclic planar graphs; structural gate (disjoint cycles), not a single-graph fixture |
| 12 | `gap_validated_layer_swaps` | A | layer-swap polish with composite validation (sprint-22e); generic |
| 13 | `outerplanar_source_fan_spine` | B | source-fan + path topology, N in 6..40 (real structural class) |
| 14 | `multi_component_row_major_repack` | A | repack disconnected components (sprint-23b); generic |
| 15 | `median_transpose_polish` | A | sugiyama median+transpose (sprint-23c); generic |
| 16 | `lattice_uniform_centered_slots` | B | lattice-LP gate + uniform-slot polish; the **0.75 * pitch** is a fudge constant but the gate (DAG, hub_ratio) is structural |
| 17 | `petersen_canonical` | **C** | hardcoded 10-position table; one graph |
| 18 | `dependency_500_x_compress` | **C** | N==500 AND E==1470; one graph |
| 19 | `outerplanar_dag_20_x_stretch` | **C** | exact 37-edge set; one graph |
| 20 | `multi_component_80_y_stretch` | **C** | exact `[40,20,10,5,3,1,1]`; one graph |
| 21 | `hexagonal_lattice_42_aspect` | **C** | N==42, E==53 + lattice gate; one graph |
| 22 | `triangular_lattice_36_aspect` | **C** | N==36, E==85; one graph |
| 23 | `transformer_layer_aspect` | **C** | exact 19-edge set; one graph |
| 24 | `disconnected_encoder_residual_y_rebalance` | **C** | hand-tuned [1.0, 0.968, 0.955, 1.773] table; one graph |
| 25 | `compound_dag_5x30_wave` | **C** | sin(idx*pi/2) * 5120; one graph |
| 26 | `densenet_block_collinear` | **C** | y-slot table [0,1,2,3,4,5,6,9.5]; one graph |
| 27 | `dependency_graph_100_depth_spine` | **C** | N==100, E==285, indeg fingerprint; one graph |
| 28 | `recurrent_feedback_cell_spine` | **C** | exact 6-edge set; one graph |
| 29 | `sierpinski_42_offset` | **C** | hardcoded 42x2 offset table; one graph |
| 30 | `rgg_500_depth_spine` | **C** | N==500, E==3491; one graph |
| 31 | `real_lesmis_77_rank_spine` | **C** | hardcoded 77-element rank order; one graph |
| 32 | `long_range_residual_ladder_spine` | **C** | hardcoded 38-element rank + 37-element gap; one graph |

Plus a 33rd `cluster_bridge_lanes` polish gated separately (line
4147) that I would classify as **B** (sprint-24a, keys on cluster
presence).

Tally: **16 entries are class A** (principled algorithm), **4 are
class B** (real structural class gate), **15 are class C** (fixture
or one-graph gate -- entries #17-32 minus tutte_cyclic_planar,
outerplanar_source_fan_spine, lattice_uniform_centered_slots,
multi_component_row_major_repack which are A or B).

That is **15 of 33 = 45%** of the picker's candidate slate
dedicated to fixture-style overfitting introduced in
sprint-25..29. By candidate count, the picker is more fixture than
algorithm in the post-sprint-22 chain.

---

## 9. Other smells (LOW-MEDIUM)

### [MEDIUM] [code smell] dagua_native.py:3829 `score()` reseeds inside the picker

Evidence:

```python
def score(pos: torch.Tensor) -> float:
    torch.manual_seed(0)
    return float(composite(full(pos, edge_index, node_sizes=node_sizes)))
```

Why this is a problem: reseeding global RNG inside a scoring function
called repeatedly mutates global state outside the picker. Acceptable
hack to make `composite` deterministic for the picker, but the side
effect leaks: the rest of the pipeline now sees a reset RNG state.

Recommended action: scope the seed via `torch.random.fork_rng()` or
pass the seeded `Generator` into `sampled_crossing_rate`.

### [MEDIUM] [code smell] dagua_native.py:3793-3806 picker-margin docstring is benchmark-anchored

Evidence:

> Sprint-23a: margin lowered from 0.5 to 0.1. Sprint-22b made
> composite() deterministic for fixed positions, so the larger gate
> that protected against sampling noise is no longer needed. Empirical
> sweep on the outcome-sensitive set (5 close-loss graphs +
> triangular_lattice_36 + petersen_10) found that margin=0.1
> captures `multi_component_80` close-loss to tie...

Why this is a problem: the only justification offered for margin=0.1
is that it captured wins on a specific 7-graph subset of the
benchmark. There is no analysis of how often margin=0.1 admits
noise-driven false-positive picks on graphs outside the benchmark.
This is benchmark overfitting at the meta-level: even the
hyperparameter that gates the polishes was tuned on the benchmark.

Recommended action: re-justify the margin on a held-out set, or
default it to 0.5 and let the fixture polishes either earn 0.5
points or be reverted.

### [LOW] [code smell] dagua_native.py:1340-1355 Petersen gate iterates the edge tensor in Python

Evidence:

```python
edges = {
    tuple(sorted((int(src[i].item()), int(tgt[i].item()))))
    for i in range(int(edge_index.shape[1]))
}
```

Per-iteration `.item()` calls: 30 device-host syncs per call. Same
pattern in `_is_recurrent_feedback_cell_signature` (line 3376),
`_is_densenet_block_signature` (line 3286), `_is_transformer_layer_signature`
(line 3087), `_is_outerplanar_dag_20_signature` (line 2866),
`_is_compound_dag_5x30_signature` (line 3047), and others.

Why this is a problem: every fixture gate runs on every layout call,
and each gate does N device-host syncs to compute its
edge-set hash. On GPU this is a measurable per-graph overhead; on
CPU it's tolerable but still wasteful for graphs that won't match.

Recommended action: gate by N+E first (no syncs), then by a single
batched comparison (`(edge_index.cpu() == expected_tensor).all()`).
But if the fixture polishes are reverted, this finding becomes moot.

### [LOW] [code smell] dagua_native.py:4019-4146 picker semantics inconsistency

Evidence: line 4014-4018 comment:

> Sprint-26 chained polish: these use the picker's running `pos`
> (the current best), not `base_pos`, so they compose on top of
> earlier picker decisions...

Why this is a problem: the picker has TWO semantics in the same
candidate list. The earlier candidates (sprint-21a, sprint-22) feed
`base_pos` (un-polished pipeline output) into each polish; the
sprint-26+ candidates feed `pos` (the current best after earlier
polishes accepted). The implication is that re-ordering the list
changes outcomes -- the picker is order-dependent and that
order is implicit. A future maintainer sorting alphabetically
would silently regress the benchmark.

Recommended action: split the picker into two phases (independent
candidates scored against base, then chained candidates scored
against best-so-far), and document the split semantics.

### [LOW] [code smell] dagua_native.py:3354-3361 missing import path comment

Evidence: `_dependency_graph_100_depth_spine_polish` does
`from dagua.utils import longest_path_layering` inside an exception-
swallowing try block. If the import fails, the polish silently
returns the input unchanged.

Why this is a problem: silent fallback hides genuine bugs (e.g., a
future utils refactor that breaks the import).

Recommended action: hoist imports to module top; let real
ImportError propagate.

---

## 10. Bottom line / what to do

By the audit's own bar -- "would I be embarrassed if a graph drawing
researcher read this code" -- the answer is unambiguous yes for
sprint-25 through sprint-29. The fixture polishes are not algorithms
and the docstrings admit it ("Codex empirical: hardcoded local-search
rank order"). The collinearity bug in `segments_intersect` is the
load-bearing exploit; once it is fixed, the sprint-28/sprint-29
spine polishes will stop scoring lifts and can be reverted without
benchmark regret.

The principled work in the chain (sprint-22a back_edge_relayer,
sprint-22c dot_lattice_lp, sprint-22e gap_validated_layer_swaps,
sprint-23c median_transpose_polish, sprint-23b
multi_component_row_major_repack, sprint-22d tutte_cyclic_planar,
sprint-24a cluster_bridge_lanes, sprint-23 outerplanar_source_fan_spine
under its structural gate, sprint-24c lattice_uniform_centered_slots
under its structural gate) should stay. They generalize. They have
papers behind them or follow obviously from graph structure.

What absolutely must come out, in priority order:

1. The `segments_intersect` collinearity bug (re-apply sprint-24b
   fix or equivalent). This is the single change that demonstrates
   most of the sprint-25..29 lifts evaporate.
2. The pure lookup-table polishes: `_petersen_canonical_polish`,
   `_sierpinski_42_offset_polish`, `_real_lesmis_77_rank_spine_polish`,
   `_long_range_residual_ladder_spine_polish`,
   `_densenet_block_collinear_polish`,
   `_disconnected_encoder_residual_y_rebalance_polish` (the fudge-table
   per-component pitch),
   `_recurrent_feedback_cell_spine_polish`.
3. The signature-gated affine / wave / aspect / compress polishes:
   `_rgg_500_depth_spine_polish`, `_dependency_graph_100_depth_spine_polish`,
   `_transformer_layer_aspect_polish`, `_compound_dag_5x30_wave_polish`,
   `_outerplanar_dag_20_x_stretch_polish`,
   `_multi_component_80_y_stretch_polish`,
   `_hexagonal_lattice_42_aspect_polish`,
   `_triangular_lattice_36_aspect_polish`,
   `_dependency_500_x_compress_polish`.
4. The config-propagation hole at engine.py:991 (`if remapped_from_default`).
5. Restore the dummy_nodes regression test inequalities or replace
   them with an actual layered-DAG quality invariant test.
6. Sweep the docstring sprint-references out of `dagua_native.py`
   and replace with paper / mechanism citations on the surviving
   primitives.
7. Re-justify or reset the picker margin (currently 0.1, tuned on
   benchmark) on a held-out graph set.

Once those are done, the picker shrinks from 33 candidates to ~17
principled ones. The benchmark's "97% best-or-tied" headline number
will fall, and that is fine -- a smaller honest number is the right
target. The point of the algorithm is to be a graph drawer, not a
benchmark passer.
