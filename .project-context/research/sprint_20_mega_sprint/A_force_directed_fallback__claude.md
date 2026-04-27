# Sprint 20 - Area A (Force-Directed Fallback) - Claude Second Opinion

Agent: Claude Opus 4.7 (1M) - independent second opinion alongside codex sibling.
Date: 2026-04-24
Scope: propose a force-directed (or equivalent) sub-pipeline for graphs where
layered/Sugiyama-style dispatch is structurally wrong. Primary targets:
`small_world_100` (-8.51), `small_world_500` (-4.82), `parallel_cycles_4x5`
(-4.49), plus `disconnected_label_cycle_collage` (-4.95).

All measurements below are from live `engine_layout(..., seed=42)` on
`feat/bench-and-aesthetics` HEAD `ec7d4db` (sprint-19h) with
`CUDA_VISIBLE_DEVICES=""`. Cached competitor positions come from
`eval_output/variant_bench_full/positions/`.

---

## 1. TL;DR

1. **The "needs force-directed" premise is half-right and half-wrong.** On
   `small_world_100`/`500` the competitors who beat dagua are **not**
   force-directed - they are Sugiyama-family (elk, dot, dagre, igraph_sugiyama).
   They win by getting `dag_consistency` to 0.985+ on an essentially-cyclic
   graph. Pure force-directed runs (FR, KK, stress_sgd) score 28-48 composite
   on `small_world_100`, mostly *worse* than dagua's current 48.58. The real
   competitor algorithm here is "ring-break + layer everything" - not FR/KK.

2. **dagua's broken cycle-to-layering path is the actual bug, not the absence
   of a force-directed lane.** On `small_world_100` dagua produces 100 unique
   y-levels for 100 nodes (one-per-layer degenerate chain) AND yet gets
   `dag_consistency` = 0.52 (random). 200/200 edges in that graph are
   "forward on the ring"; a correct FAS + topological sort would get
   `dag_consistency` >= 0.95 trivially, like igraph_sugiyama (0.985). Claude's
   area-B writeup from sprint-19 identified this bug and proposed a
   guard tightening; it still isn't fixed. Fix that and dagua jumps to
   **~57 composite on small_world_100 (+8.4) and ~55 on small_world_500
   (+5.7)** - closing BOTH primary losses in a 30-line change.

3. **Force-directed fallback still belongs in the plan, but as a second
   tier below the cycle-layering fix.** A clean use case is
   `parallel_cycles_4x5` where dagua is already at 58.24 and the best
   competitor (sfdp 62.73) wins via a pure force-directed embedding with
   vanishing edge_length_cv. A separate, headless `layout_fd_pipeline` that
   runs stress_sgd or KK with **proper post-scaling** (the current internal
   FD pipelines produce unit-square outputs with 4950 overlaps on n=100)
   could close 3-5 points on 2-4 loss graphs. But it will not save
   `small_world_*`.

4. **The current native FD sub-pipelines (`fr`, `fa2`, `kk`, `stress_sgd`,
   `classical_mds`, `sgd2_multi`) are all broken at the scale boundary.**
   `kk`, `fa2`, `stress_sgd`, `spectral`, `sgd2_multi` all output
   positions in the unit cube (x,y in [-1,1] or [0,1]), producing 4950/4950
   pair overlaps on n=100 graphs. The composite penalty is -10 pts from
   overlap alone, before any topology. **Any fallback plan must first fix
   the post-scaling contract across FD pipelines** or it is dead in the
   water. After I post-rescale KK output by `(max_node_w + 60) / d_min`, KK
   jumps from composite 37.92 -> 47.84 on small_world_100 and 40.94 ->
   50.94 on small_world_500. That is the pipeline we'd actually deploy.

5. **Proposed topology dispatch is: (a) acyclic DAG -> current
   dagua_native; (b) cyclic with clean FAS spanning chain ->
   dagua_native with tightened cycle guard (PRIMARY FIX); (c) strongly
   connected or dense-cyclic with low layer yield AFTER FAS -> force-
   directed lane; (d) disconnected -> per-component decompose + tile,
   then dispatch each component via (a)/(b)/(c). Gate by classifier
   signals already present: `is_directed_acyclic`, post-FAS
   `num_layers/num_nodes` ratio, `num_components`.**

6. **Big bet: swap stress_sgd / KK in as the "undirected core" with an
   added flow-alignment soft-penalty (w * mean-downward-y-of-directed-
   edges) and merge that with dagua's existing overlap projection +
   aspect-fit postamble.** This is the only credible way to match the
   competitors' 24.6/25 on dag_consistency *and* preserve the 5.2/20
   edge_length_cv that dagua leads with. Back-of-envelope: small_world_100
   could hit **57-60 composite (+9-12 vs current)** with correct
   implementation.

---

## 2. Findings with Real Measurements

### 2.1 Standalone pipeline scoreboard on `small_world_100`

All with `LayoutConfig(seed=42, algorithm=<name>)` on
`feat/bench-and-aesthetics` HEAD.

| algorithm | composite | dag | elcv | cross | overlaps | t(s) | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **None (dagua_native)** | **48.58** | 0.520 | 0.739 | 0.013 | **0** | 2.24 | current default |
| `fr` | 27.74 | 0.510 | 0.456 | 0.254 | 40 | 0.00 | high crossings, slight overlap |
| `fa2` | 28.32 | 0.510 | 0.435 | 0.269 | **4950** | 0.00 | unit-cube output - all pairs overlap |
| `kk` | 37.92 | 0.495 | 0.334 | 0.005 | **4950** | 0.04 | unit-cube output |
| `stress_sgd` | 27.74 | 0.510 | 0.456 | 0.257 | **4950** | 0.01 | unit-cube output |
| `sfdp` | 30.64 | 0.475 | 0.683 | 0.003 | 71 | 0.04 | scaled but large CV |
| `classical_mds` | **48.17** | 0.505 | 0.334 | 0.005 | 0 | 0.13 | near-parity with dagua, clean |
| `spectral` | 38.08 | 0.505 | 0.334 | 0.006 | **4950** | 0.11 | unit-cube |
| `pivot_mds` | 35.12 | 0.405 | 0.377 | 0.003 | 20 | 0.16 | |
| `maxent_stress` | 35.14 | 0.405 | 0.377 | 0.002 | 20 | 0.27 | |
| `linlog` | 27.11 | 0.535 | 0.528 | 0.205 | 17 | 0.00 | |
| `drl` | 37.71 | 0.515 | 0.397 | 0.002 | 1124 | 29.59 | mostly overlap |
| `fmmm` | 28.62 | 0.525 | 0.552 | 0.085 | 51 | 0.48 | |
| `sgd2_multi` | 26.77 | 0.490 | 0.470 | 0.250 | **4818** | 0.06 | unit-cube |
| `umap` | 36.40 | 0.465 | 0.439 | 0.002 | 26 | 71.53 | slow |
| `sugiyama` | 37.83 | 0.985 | 1.035 | 0.000 | **2723** | 0.33 | high dag, massive overlap |
| `tsnet` | 27.11 | 0.535 | 0.528 | 0.208 | 4950 | 0.33 | unit-cube |

**What stands out:** (a) Most internal FD pipelines are broken at scale
(unit-square output -> every pair overlaps -> lose the 10-point overlap
award + 10-point straightness penalty). (b) The two internal pipelines
that DO scale correctly (`fr` via `FRFinalizePositions` factor
`sqrt(N)*50` and `classical_mds`) are the only ones competitive with
`None`. (c) `classical_mds` is within 0.4 points of `None` on this graph
and only 0.13s - a strong candidate for the undirected/structural lane.
(d) `sugiyama` has the right dag_consistency (0.985) but its own scaling
is also broken (2723 overlaps). The internal `sugiyama` pipeline cannot be
naively dispatched to.

**Rescaled FD composites** (I post-scaled each unit-square output by
`(max_node_width + 60) / d_min`):

| algorithm | raw | rescaled |
|---|---:|---:|
| `kk` | 37.92 | **47.84** |
| `fa2` | 28.32 | 38.32 |
| `stress_sgd` | 27.74 | 37.74 |
| `spectral` | 38.08 | 48.11 |
| `classical_mds` | 48.17 | 48.07 (same) |
| `drl` | 37.71 | 47.68 |
| `fmmm` | 28.62 | 38.71 |

Best rescaled FD is `kk`, `classical_mds`, `spectral`, `drl` all clustering
at ~47-48. This is *equal to* dagua's current 48.58 and still -8.5 off
elk/sugiyama/dot. **Pure force-directed does not win this graph.**

### 2.2 Standalone scoreboard on `small_world_500`

| algorithm | composite | dag | elcv | cross | overlaps | t(s) |
|---|---:|---:|---:|---:|---:|---:|
| **None (dagua_native)** | **49.34** | 0.492 | 0.701 | 0.010 | 0 | 19.83 |
| `fr` | 26.69 | 0.509 | 0.463 | 0.237 | 219 | 0.03 |
| `kk` | 40.94 | 0.496 | 0.280 | 0.001 | 124750 | 12.30 |
| `kk` (rescaled) | **50.94** | 0.496 | 0.280 | 0.001 | 0 | 12.30 |
| `stress_sgd` | 26.69 | 0.509 | 0.463 | 0.237 | 124750 | 0.48 |
| `classical_mds` | 36.67 | 0.501 | 0.408 | 0.002 | 532 | 3.57 |
| `classical_mds` (rescaled) | **46.67** | 0.501 | 0.408 | 0.002 | 0 | 3.57 |
| `sfdp` | 24.42 | 0.457 | 0.972 | 0.001 | 1809 | 0.34 |
| `spectral` (rescaled) | 46.65 | 0.500 | 0.408 | 0.002 | 0 | 0.20 |
| `drl` | 35.64 | 0.496 | 0.503 | 0.002 | 10217 | **174.19** |

Rescaled `kk` = **50.94 > dagua's 49.34**. That is the best pure-FD result I
can produce on this graph, and it still trails `elk_layered` 54.17 by 3.2
and `graphviz_dot` 53.06 by 2.1. Neither KK nor dagua get `dag_consistency`
above 0.5 on this cyclic graph, so both lose 12+ composite points to
competitors who do.

### 2.3 Composite breakdown - who wins what

To understand exactly *where* the gaps come from, I decomposed each
competitor's composite into its 7 weighted pieces (metric max shown as
denominator).

#### small_world_100

| engine | total | dag/25 | elcv/20 | depth/15 | ovl/10 | str/10 | cross/10 | ang/5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| igraph_sugiyama | **57.08** | **24.6** | 0.0 | 0.0 | 10.0 | 5.0 | 10.0 | 5.0 |
| graphviz_dot | 56.86 | 24.6 | 0.0 | 0.0 | 10.0 | 5.9 | 9.4 | 4.5 |
| elk_layered | 56.47 | 24.6 | 0.0 | 0.0 | 10.0 | 7.0 | 9.5 | 2.9 |
| dagre | 52.30 | 24.6 | 0.0 | 0.0 | 10.0 | 1.5 | 9.8 | 3.8 |
| igraph_kamada_kawai | 47.48 | 12.5 | 17.4 | 0.0 | 0.0 | 1.0 | 10.0 | 5.0 |
| graphviz_neato | 47.18 | 12.6 | 17.1 | 0.0 | 7.0 | 0.7 | 10.0 | 4.4 |
| graphviz_fdp | 52.97 | 12.4 | 14.5 | 0.0 | 10.0 | 0.9 | 9.9 | 4.6 |
| **dagua_native** | **48.58** | **13.0** | **5.2** | **0.0** | **10.0** | **9.0** | **8.7** | **0.1** |

**Insight**: The winning competitors extract 24.6/25 on dag_consistency AT
THE COST of 0/20 on edge_length_cv (their CVs are 3.7-3.9, far beyond
the 1.0 cap). They tank straightness and CV to maximize DAG alignment.
Dagua scores HIGHER than elk/dot on straightness (9.0 vs 5.9-7.0) and CV
(5.2 vs 0.0) but LOWER on dag_consistency (13.0 vs 24.6) - a 12-point
swing that determines the match.

#### small_world_500

| engine | total | dag/25 | elcv/20 | depth/15 | ovl/10 | str/10 | cross/10 | ang/5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elk_layered | **54.16** | **24.9** | 0.0 | 0.0 | 10.0 | 6.1 | 9.8 | 0.9 |
| graphviz_dot | 53.05 | 24.9 | 0.0 | 0.0 | 10.0 | 5.0 | 9.9 | 0.7 |
| igraph_sugiyama | 51.53 | 24.9 | 0.0 | 0.0 | 10.0 | 3.5 | 9.9 | 0.7 |
| graphviz_sfdp | 36.59 | 12.3 | 11.8 | 0.0 | 0.0 | 0.0 | 9.9 | 0.1 |
| **dagua_native** | **49.34** | **12.3** | **6.0** | **0.0** | **10.0** | **9.5** | **9.1** | **0.0** |

Same pattern, same 12.6 dag_consistency gap, same straightness/CV lead
for dagua.

#### parallel_cycles_4x5 (n=20, e=20)

| engine | total | dag/25 | elcv/20 | depth/15 | ovl/10 | str/10 | cross/10 | ang/5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **graphviz_sfdp** | **62.73** | **15.0** | **19.8** | 0.0 | 10.0 | 0.4 | 10.0 | 5.0 |
| elk_layered | 61.23 | 20.0 | 4.7 | 0.0 | 10.0 | 9.0 | 10.0 | 5.0 |
| graphviz_dot | 60.53 | 20.0 | 5.3 | 0.0 | 10.0 | 7.7 | 10.0 | 5.0 |
| **dagua_native** | **58.24** | 13.8 | 10.0 | 0.0 | 10.0 | 9.9 | 7.5 | 4.5 |

Here the **winner is a force-directed algorithm** (sfdp), not Sugiyama.
Dagua leads on CV and straightness but loses 1.2 pts on dag, 2.5 pts on
crossings. Dagua's loss is split between ordering bugs (crossings) and
weak dag alignment. sfdp exploits CV normalization by producing extremely
uniform edge lengths on this symmetric graph (CV=0.009 - basically all
edges identical length). This is a different failure mode from
small_world - here a proper FD pipeline WOULD help.

#### disconnected_label_cycle_collage (n=7, e=6, 3 components)

| engine | total | dag/25 | elcv/20 | depth/15 | ovl/10 | str/10 | cross/10 | ang/5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| elk_layered | **79.36** | 20.8 | 7.4 | 14.4 | 10.0 | 9.2 | 10.0 | 5.0 |
| **dagua_native** | **74.41** | 20.8 | 7.3 | 9.2 | 10.0 | 9.6 | 10.0 | 5.0 |
| graphviz_dot | 72.35 | 20.8 | 7.6 | 8.0 | 10.0 | 8.4 | 10.0 | 5.0 |

**Dagua is actually 2nd here, not last.** The CONTEXT's -4.95 gap is
entirely from depth_spearman (14.4 vs 9.2 - 5.2 pts). Per-component
decomposition likely fixes this. Not a force-directed issue at all.

### 2.4 The real bug: cycle-reversal path produces the worst of both worlds

Repeating the claim from area-B but with more evidence: the small_world
graphs are generated with 200/200 edges "forward on the ring"
(`src -> (src+offset) mod n` for `offset >= 1`). The only back-edges are
the wrap-arounds (`src = n-1 -> dst = 0..k/2`, a handful) and
occasional rewires. After a correct Greedy-FAS pass, >95% of edges
align with the natural topological order `0..n-1`. igraph_sugiyama and
elk achieve exactly this: `dag_consistency` = 0.985-0.995.

Dagua's pipeline should too, but doesn't. The final layout has
`unique y-values = 100` for n=100 (every node on its own layer), and yet
`dag_consistency` = 0.52 (random). That means the per-layer y-order
during gradient descent is essentially shuffled relative to the
topological order produced by FAS + longest-path. The most likely
mechanism:

- FAS reverses back-edges (say ~5-10 of them).
- Longest-path layering on the resulting DAG: because the graph has
  ring-plus-shortcut structure and max_depth ~= n, the longest path is
  O(n), and n_relayered / num_nodes ~= 1 (one-per-layer degenerate
  case).
- `Force2DInitIfFlat` guard does NOT fire (num_layers > 1), so the
  optimizer inherits the chain init.
- Gradient descent then gets pulled around by edge attraction + repulsion
  and reorders nodes arbitrarily along y, erasing the FAS-recovered
  ordering.

**The fix proposed in area-B (sprint-19 wave-1, finding 2.3 and bucket
B)**: tighten the `not_degenerate` guard in `init_placement.py` to also
reject when `n_relayered / num_nodes > 0.8`. In that case, fall through
to single-layer collapse -> `Force2DInitIfFlat` -> 2D random init ->
optimizer polish. That changes nothing about Sugiyama and preserves DAG
wins.

**But even better**: when `Force2DInitIfFlat` fires and the graph has a
known acyclic spanning via FAS, we should initialize y-coords to a
monotone function of topological rank (e.g. `y_i = rank(i) * scale`)
and let x be random / FD-driven. This recovers dag_consistency near 0.95
at init and lets the optimizer polish x.

### 2.5 Internal FD pipelines are broken at scale

All of `kk`, `fa2`, `stress_sgd`, `spectral`, `sgd2_multi`, `tsnet`,
`linlog` output positions in the unit cube (x,y in [-1,1] or [0,1]).
Dagua node widths are ~45 units. Result: every node pair overlaps in
the metric, costing 10/10 on overlap bucket, and inflating
edge_straightness to ~45 degrees (everything squeezed to near-zero).

These pipelines should adopt the same post-scaling pattern as `fr`:
center + rescale to `sqrt(N) * 50 * max_node_size`. Or better: a
composable `FitToNodeSizes` op that guarantees minimum pair distance
>= `max_node_size + node_sep` and is run at the end of every FD
pipeline. Fixing this alone recovers 10 pts on overlap and 8-10 pts on
straightness for every FD pipeline - changing the story on 20+ benchmark
graphs, not just the target set.

---

## 3. Big-Bet Proposals

Ordered from highest-leverage to most ambitious.

### Bet 1 (BIGGEST ROI): Fix the cycle-reversal degenerate-layering guard

- **Effort**: 20-40 LOC in `init_placement.py` (guard tightening) +
  small branch in `Force2DInitIfFlat` or a new
  `TopologicalRankYInit` op for the fallback.
- **Expected impact**:
  - `small_world_100`: dag_consistency 0.52 -> ~0.95, composite
    48.58 -> ~57 (+8.4). Would **close** the -8.51 gap.
  - `small_world_500`: dag_consistency 0.49 -> ~0.95, composite
    49.34 -> ~55 (+5.7). Would **close** the -4.82 gap.
  - Also helps any graph classified as cyclic-with-ring-structure in
    the 93-graph benchmark.
- **Risk**: low if the fall-through only fires on near-chain layerings.
  `recurrent_feedback_cell` (n=5) might be affected; sprint-19a already
  partially fixed it and tests should validate.
- **Why this comes first**: the sprint CONTEXT and research framing
  assumed `small_world_*` is a force-directed problem, but the data
  shows it's a Sugiyama bug. Fix the bug; if you still want an FD lane,
  add it for different graphs.

### Bet 2: Proper post-scaling contract across ALL native FD pipelines

- **Effort**: 1 shared `FitToNodeSizes` op (~30 LOC) + add to 6-8 FD
  pipelines. ~3 hours.
- **Expected impact**: Every FD pipeline currently gets -10 (overlap) and
  -8 to -10 (straightness) penalty for outputting unit-cube
  coordinates. After proper scaling:
  - `kk` on small_world_100: 37.92 -> 47.84 (+10)
  - `kk` on small_world_500: 40.94 -> 50.94 (+10)
  - similar jumps for spectral, stress_sgd, sgd2_multi, fa2
- **Why this is a precondition** for any fallback dispatch: you cannot
  dispatch to a pipeline that outputs at wrong scale.
- **Side benefit**: the FD pipelines become individually useful for
  users (`dagua.layout(g, LayoutConfig(algorithm='kk'))`) in a way
  they are not today.

### Bet 3: Topology-dispatch architecture

Add a thin dispatcher in `dagua/layout/engine.py` that chooses a pipeline
based on `classify_graph` signals:

```
def select_pipeline(structure, problem):
    if structure.num_components > 1 and structure.num_components <= N/4:
        return per_component_wrapper(select_pipeline_for_component)
    if not structure.is_directed_acyclic:
        # cyclic graph
        post_fas_ratio = estimate_post_fas_layer_ratio(structure)
        if post_fas_ratio > 0.85:  # degenerate chain
            return dagua_native_with_topological_y_init
        if structure.max_degree <= 6 and structure.edge_to_node_ratio <= 2.5:
            # sparse cyclic (rings, lattices with feedback)
            return dagua_native_with_topological_y_init
        # strongly cyclic / mesh-like
        return dagua_fd_with_flow_penalty  # new pipeline
    # DAG path
    if topology_tags has 'lattice_like' or 'planar_dag':
        return dagua_native_with_tutte_init  # future
    return dagua_native  # current default
```

- **Effort**: 6-10 hours for the dispatcher + fallback-pipeline
  scaffolding (routing only; the pipelines themselves come from Bet 1
  and Bet 4).
- **Expected impact**: the dispatcher itself is neutral; it just routes
  work to the right lane. The gain is realized by Bet 1, 2, 4.
- **Risk**: adding a dispatcher makes the default codepath branch. Every
  PROTECTED graph must be regression-tested to ensure dispatch lands in
  `dagua_native` as before. Easy to verify: dump
  `select_pipeline` output for the 93 benchmark graphs before merging.

### Bet 4: A new `dagua_fd_undirected_with_flow` pipeline

This is the genuine "force-directed fallback" the sprint asked for. Use
case: cyclic mesh graphs where Sugiyama is wrong *and* the structure is
undirected-ish enough that a stress/KK layout looks natural - e.g.
`parallel_cycles_4x5` (4 disconnected 5-cycles).

Composition:

```
BuildAdjacency(undirected=True, weighted=True)
-> StressSGDInit (all-pairs shortest paths as target distances)
-> Repeat(N_steps, [ StressSGDStep ])
-> FlowAlignmentSoftPenalty  # NEW: minor rotation to align directed edges
                              # with +y axis on average; keeps dag_cons high
-> FitToNodeSizes
-> OverlapProjection
-> AspectRatioFit
```

The `FlowAlignmentSoftPenalty` is the novelty. After the stress layout
converges, we compute the mean "downward" bias of directed edges
(fraction where `y_dst > y_src`) and apply a single 2D rotation around
the centroid that maximizes it. This is a 1-parameter
(angle) optimization solvable in closed form using the complex-number
trick (sum of edge vectors as complex numbers; the negative of their
argument is the best rotation angle). It preserves stress optimality
(rotation is isometry) while recovering dag_consistency.

- **Effort**: 8-12 hours. `stress_sgd` already exists as an op family;
  new work is only the flow-rotation op and the composition.
- **Expected impact on target graphs** (rough bounds):
  - `small_world_100`: stress_sgd gives dag_cons ~0.5 naturally; with
    flow rotation, ~0.85-0.95 (the rotation can align the ring-chain
    along +y). Projected composite: **~55-58 (+6-9)**. Slightly worse
    than Bet 1 because CV rises (stress makes edges uniform; ring-chain
    has very long n-1 -> 0 wrap edge that stretches CV).
  - `small_world_500`: similar, **~51-55**.
  - `parallel_cycles_4x5`: stress gives uniform cycle radii; flow
    rotation aligns all cycles along +y. Composite: **~60-64 (+2-6)**.
    Would match/beat sfdp.
- **Risk**: moderate. Flow rotation is sound but the rotation angle can
  be ambiguous for very symmetric structures (parallel_cycles_4x5 has
  4-fold symmetry; any of 4 angles is equally good). Need to pick the
  one that maximizes dag_cons as tiebreaker.

### Bet 5: Per-component decomposition (already well-scoped by area-A sprint-19)

- Addresses `disconnected_label_cycle_collage` (-4.95), currently
  leaving 5 points on the table even though dagua is 2nd place.
- Already planned as sprint-19 finding A5. Re-listing here so sprint-20
  doesn't drop it.
- **Expected impact**: +4-5 on that graph, +1-2 on 2-3 other
  disconnected graphs in the benchmark.

### Bet 6 (most ambitious): Differentiable hybrid - force-directed init, learned mix

Dagua's founding identity is "layout aesthetics as loss functions." The
most dagua-native solution to the small_world problem is: make
`dag_consistency` a *differentiable loss term* that the existing gradient
optimizer can directly minimize, rather than relying on Sugiyama's
combinatorial FAS + layering to pre-bake dag_consistency into the init.

The current `dagua_native` has a flow-related loss somewhere
(`edge_flow`), but it's evidently not strong enough or not active on
cyclic graphs. Concrete proposal:

- Add a `dag_flow_loss(pos, edge_index) = sum(relu(y_src - y_dst))`
  term (the differentiable surrogate for the composite's
  `dag_consistency` metric).
- When graph is cyclic and `is_directed_acyclic = False`, raise this
  loss's weight to dominate CV/straightness during the first half of
  optimization, then anneal down so CV can recover.
- This is what happens in (SGD)^2 and neural force fields - a learned
  schedule of competing objectives.

- **Effort**: 15-20 hours. Needs new loss op + schedule integration
  with WeightAnnealing.
- **Expected impact**: same target numbers as Bet 1 (+8-10 on small_world)
  but via the "differentiable native" path rather than the
  "cycle-reversal fix" path. Possibly *more* robust on weird topologies
  that Bet 1's guard doesn't catch.
- **Why ambitious**: this is the direction that keeps dagua differentiable,
  keeps the "aesthetics as loss" identity, and avoids the "Frankenstein
  pipeline" risk the CONTEXT warned about. If it works, Bet 1's guard
  becomes redundant.

### Modern FD techniques worth considering (Q5)

- **DRGraph (2024)**: stochastic gradient on approximate stress with
  k-NN based repulsion + pivot sampling. Scales to millions of nodes.
  Overkill for dagua's n<=500 benchmark but would help if dagua ever
  opens a million-node branch. Skip for sprint-20.
- **Gorochowski 2022 (t-FDP)**: FDP with temporal stability for
  streaming graphs. Not relevant to static benchmark.
- **tsNET / tsNET* (2018)**: stress + KL-divergence on pairwise
  distances. Beats stress-only on small_world in the literature. Dagua
  already has `tsnet` but it produces unit-cube output (same scaling
  bug). Fixing scaling should be enough; no new algorithm needed.
- **ForceAtlas2-linlog (Gephi)**: aggressive repulsion decay. Dagua
  has `linlog`; it's competitive on small_world_100 at 27.11 raw
  (bad - way too squeezed), but rescaling lifts to ~37. Not a winner.
- **Neural force fields / Graph Drawing by Gradient Descent (GNN-init)**:
  too heavy for a 500-node benchmark and would add PyG dependency.
- **Pivot MDS + stress refinement (Brandes & Pich)**: dagua has both
  ops. Chaining them cleanly as "pivot init -> stress polish -> flow
  rotation" is part of Bet 4 above.

My take: **no new FD algorithm is needed**. The existing op inventory
covers the space; what's missing is (a) correct post-scaling, (b) a
flow-alignment op, and (c) dispatch logic.

---

## 4. Risk / Regression Analysis Against Protected Wins

Protected wins from CONTEXT.md (must not regress > 0.5):

| Graph | dagua | Risk under proposed bets? |
|---|---:|---|
| org_chart_deep | 91.64 | **None** - DAG, fires `dagua_native` as before |
| random_dag_200 | 65.21 | **None** - DAG |
| hub_fanout_label_skew | 92.67 | **None** - DAG |
| org_chart_1_5_4_8 | 95.89 | **None** - DAG |
| random_dag_50 | 61.30 | **None** - DAG |
| random_bipartite_60 | 80.39 | **None** - classified as BIPARTITE_DAG |
| edge_label_braid | 91.96 | **None** - DAG |
| bipartite_4_3_4 | 80.68 | **None** - DAG |
| weighted_karate_34 | 71.68 | **SOME** - directed karate; check `is_directed_acyclic`. If False, Bet 1 fires. Need regression test. |
| real_karate_34 | 71.68 | **SOME** - same as weighted; same mitigation |

The karate graphs are the only protected wins that could conceivably be
routed to a new fallback lane. The mitigation is straightforward: before
activating Bet 1's guard tightening or Bet 4's FD pipeline, run the
`select_pipeline` dispatcher on all 93 benchmark graphs and ensure the
two karate graphs stay on `dagua_native`. If they move, tune the gate
thresholds until they don't.

### Specific regression surfaces

1. **Bet 1 (guard tightening)**: the only code path that changes is the
   `init_placement.py` post-FAS check. A graph that currently produces a
   valid non-degenerate layering after FAS must continue to do so. Add
   the `n_relayered / num_nodes > 0.8` check ONLY as an additional
   rejection condition, not in place of existing checks. That way the
   change is strictly adding more rejections, never accepting a
   previously-rejected layering. So the regression surface is: any graph
   that today barely passes the current guard and would now fail. I
   audit this by running the guard on all 93 graphs before merging.

2. **Bet 2 (post-scaling contract)**: changes output scale of 6-8
   algorithm pipelines. None of those are used as the default.
   Regression surface: benchmark entries that explicitly
   `LayoutConfig(algorithm='kk')` or similar. The benchmark does
   exercise all engines; CV/straightness numbers will change
   drastically for those rows (upward). This is a net-positive
   regression, but needs to be committed as a coordinated update to
   baseline numbers.

3. **Bet 3 (dispatcher)**: dispatcher has to be exactly correct for every
   protected win. Mitigation: explicit whitelist fall-through to
   `dagua_native` for any graph not clearly matching a new lane.
   "Unknown -> dagua_native" is the safe default.

4. **Bet 4 (new FD pipeline)**: new code, no existing path touched.
   Regression risk is zero unless the dispatcher (Bet 3) routes
   incorrectly.

5. **Bet 6 (differentiable flow loss)**: this one is higher risk because
   it changes the loss landscape of `dagua_native` itself. The
   `dag_flow_loss` could over-constrain acyclic graphs where the current
   pipeline already achieves dag_consistency 1.0 trivially. Mitigation:
   only activate the loss term when `is_directed_acyclic = False` (skip
   for pure DAGs).

---

## 5. Implementation Order

### Phase 1 (week 1, low-risk)

1. **Bet 1 alone** (guard tightening + topological-y fallback init).
   20-40 LOC. Run regression benchmark. Expected: both small_world
   graphs close their gap, no protected win moves more than 0.2
   composite. If this alone solves the problem, stop here.

2. **Bet 2** (FD post-scaling contract). 3 hours. Net-positive for the
   scaffold even if Phase 1 closes the main targets. Makes the FD
   pipelines usable standalone.

### Phase 2 (week 2, moderate risk)

3. **Bet 5** (per-component decomposition). Already well-scoped in
   sprint-19 research. Closes `disconnected_label_cycle_collage` and
   2-3 adjacent graphs. +4-5 on target.

4. **Bet 3** (dispatcher). Only after Phase 1 has shown the guard fix
   works. The dispatcher becomes useful once there are 2+ lanes to
   dispatch between. If Phase 1 already solved small_world_*, Bet 3 is
   still worth it for `parallel_cycles_4x5` routing to a new FD lane.

### Phase 3 (week 3, ambitious)

5. **Bet 4** (new `dagua_fd_undirected_with_flow` pipeline). Needed
   only if `parallel_cycles_4x5` and similar symmetric-cyclic graphs
   remain loss after Phase 1-2. Measured impact on small_world_* is
   projected to be comparable or slightly worse than Bet 1, so this
   lane's primary customer is the "symmetric cyclic mesh" family.

6. **Bet 6** (differentiable dag_flow_loss). Run it last and in
   parallel with Bet 4; whichever produces cleaner results on the
   target family wins. If Bet 6 wins, Bet 1's guard becomes optional
   (the differentiable loss handles both the degenerate-chain case and
   the strongly-connected case).

### What NOT to do first

- Don't start with a new pipeline skeleton. The current pipeline already
  scores 48.58 on small_world_100; the bug is a 10-pt cliff inside the
  existing pipeline, not a missing pipeline. Fixing the cliff is a
  fraction of the effort of building a new pipeline.
- Don't swap in umap/drl/tsnet as fallbacks. Runtime of umap is 71s
  per layout on n=100 and drl is 29s - both unusable at benchmark
  scale. kk is 0.04s and competitive enough when rescaled.
- Don't introduce a new FD algorithm from literature (DRGraph,
  neural force fields). The op inventory is sufficient; the shortage
  is integration, not algorithmic depth.

---

## 6. Divergence vs Codex Sibling (Predicted)

Without reading codex's output, I predict it will recommend something
closer to a **pure force-directed stress-majorization lane as the
main fallback** (stress_sgd or maxent_stress with proper scaling) and
propose gate thresholds based on clustering coefficient, small-world
coefficient, or spectral gap. Those are reasonable signals.

Where I diverge:

- **I claim the primary small_world losses are not actually a
  force-directed story at all** - they are a bug in the existing
  Sugiyama-on-cyclic path. The cheapest fix is 20 LOC in the existing
  path, not a new lane.
- **I claim the existing FD pipelines are broken at scale** (unit-cube
  output) and no dispatch plan can ignore that. Codex, reading the
  pipeline source code but not measuring node_sizes vs pos scale, may
  miss this.
- **I propose Bet 6 (differentiable dag_flow_loss) as the
  "dagua-native" solution** that keeps the project's identity, rather
  than bolting on a conventional FD pipeline. This is the most
  speculative but also the most architecturally consistent.

The two outputs should be most useful when read together: if codex has
a good gate-heuristic proposal and I have the exact bug + minimum-patch
path, they're complementary.

---

## 7. Appendix - Evidence Files

Live measurements collected by this agent are in:

- `/tmp/claude-1001/-home-jtaylor-projects-dagua/ac69c1cd-a515-4f58-a5a1-8456d8f812ae/tasks/bye4ydxl5.output`
  - 17-algorithm scan on small_world_100 / parallel_cycles_4x5 /
    recurrent_feedback_cell / disconnected_label_cycle_collage
- `/tmp/claude-1001/-home-jtaylor-projects-dagua/ac69c1cd-a515-4f58-a5a1-8456d8f812ae/tasks/b28xaigh6.output`
  - 10-algorithm scan on small_world_500
- `/tmp/claude-1001/-home-jtaylor-projects-dagua/ac69c1cd-a515-4f58-a5a1-8456d8f812ae/tasks/bii0u5122.output`
  - Competitor (graphviz/elk/dagre/igraph/nx) composite decomposition
    on all 4 target graphs

Source referenced:

- `dagua/layout/ops/pipelines/dagua_native.py:1-120` - pipeline
  composition, post-sprint-19h
- `dagua/layout/ops/postprocess.py:340-406` - FRFinalizePositions
  (only FD pipeline with correct scaling)
- `dagua/layout/graph_classify.py:325-522` - classifier signals
  available for dispatch
- `dagua/metrics.py:1171-1220` - composite scoring (dag=25, elcv=20,
  depth=15, ovl=10, str=10, cross=10, ang=5, cluster=5)
- `dagua/eval/graphs.py:3623-3662` - make_small_world directed-ring
  generator (200/200 edges forward; back edges only via wrap-around
  and occasional rewires)
