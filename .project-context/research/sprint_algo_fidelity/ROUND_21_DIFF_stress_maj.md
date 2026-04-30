# Round 21 Diff: stress_maj (`classic_stress_maj` vs `ogdf_stress`)

Diagnosis-only pass. No source files were edited. This report compares dagua's
active stress-majorization pipeline against OGDF's `StressMinimization` reference
as reached through the current benchmark adapters.

## 1. Files read

### Dagua side

- `dagua/layout/ops/stress.py:1-589` -- active composable SMACOF ops:
  state preparation, MDS warm start, SMACOF step, trace collection, finalization.
- `dagua/layout/ops/pipelines/stress_majorization.py:1-172` -- active pipeline
  wiring for `layout_stress_majorization_pipeline`.
- `dagua/layout/ops/graph_utils.py:1-352` -- shortest-path helpers and
  disconnected finite fill used by the active ops pipeline.
- `dagua/layout/ops/pipelines/__init__.py:12-88` -- pipeline registry entry for
  `"stress_majorization"`.
- `dagua/layout/classic/stress_majorization.py:1-295` -- legacy/classic
  implementation with the same dense pseudoinverse SMACOF structure; still
  imported by tests and useful as historical baseline.
- `dagua/layout/classic/classical_mds.py:1-235` -- legacy MDS warm start and
  finite shortest-path fill used by legacy stress majorization.
- `dagua/layout/classic/_graph_distances.py:1-227` -- legacy unweighted BFS,
  weighted Dijkstra, and duplicate-edge handling.
- `dagua/eval/variants.py:1-1888`, especially `dagua/eval/variants.py:826-856`
  for stress-majorization variants and `dagua/eval/variants.py:1820-1879` for
  stochastic/heavy classification.
- `dagua/eval/competitors/classic_competitor.py:1-1635`, especially
  `dagua/eval/competitors/classic_competitor.py:189-193`,
  `dagua/eval/competitors/classic_competitor.py:1062-1091`, and
  `dagua/eval/competitors/classic_competitor.py:1570-1620`.
- `dagua/eval/competitors/ogdf_competitor.py:1-303`, especially
  `dagua/eval/competitors/ogdf_competitor.py:105-171` and
  `dagua/eval/competitors/ogdf_competitor.py:264-270`.
- `dagua/eval/competitors/base.py:1-152` -- seed interface and base
  `layout_with_variant` behavior.
- `dagua/graph.py:80-105`, `dagua/graph.py:328-362`,
  `dagua/graph.py:815-850`, and `dagua/graph.py:1619-1684` -- edge-weight
  storage and graph construction.
- `scripts/ogdf_runner.cpp:1-242` -- standalone OGDF subprocess bridge,
  algorithm selection, graph construction, seed behavior, and JSON output.
- `eval_output/fidelity_report/report.md:1-105` -- current mega-run verdict.
- `eval_output/fidelity_report/data/algorithm_summary.csv` -- skimmed stress
  rows via `rg`; the stress-majorization variants are strong-equivalent with
  median RMSD around 0.041.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md` --
  sprint context and note that OGDF families are already strong-equivalent.

### Reference side

- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:1-238`
  -- defaults, exposed setters, convergence criterion enum, private methods.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:1-344`
  -- actual stress minimization implementation.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/graphalg/ShortestPathAlgorithms.h:1-154`
  -- OGDF BFS and Dijkstra all-pairs shortest paths.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/PivotMDS.h:1-168`
  -- default pivot-MDS settings used by OGDF's initial layout.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:1-393`
  -- pivot-MDS implementation, path special case, pivot selection, SVD/power
  iteration, RNG.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/packing/ComponentSplitterLayout.h:50-67`
  -- component splitter API and default border setter.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/packing/ComponentSplitterLayout.cpp:1-180`
  -- disconnected component layout and packing flow used by OGDF's initial
  layout path.

## 2. Overall pipeline structure

### Dagua active pipeline

The active benchmark path is `classic_stress_maj` in
`dagua/eval/competitors/classic_competitor.py`. Its layout spec imports
`dagua.layout.ops.pipelines.stress_majorization` and calls
`layout_stress_majorization_pipeline` with default `{"iterations": 200}`
(`dagua/eval/competitors/classic_competitor.py:189-193`). Variant wrappers
override only the reimplementation side: `iter=200`, `iter=50`, and `iter=500`
are configured at `dagua/eval/variants.py:826-856`.

The pipeline builder is:

1. set `FixedSteps(n=iterations)`,
2. `PrepareStressMajorizationState`,
3. `InitializeStressMajorizationPositions`,
4. repeat `SmacofStep` plus optional trace collection exactly `iterations`
   times,
5. `FinalizeStressMajorizationPositions`
   (`dagua/layout/ops/pipelines/stress_majorization.py:60-75`).

The public entrypoint validates shapes, short-circuits empty and singleton
graphs, builds a `LayoutProblem`, sets `trace_every`, and applies the pipeline
on a CPU execution plan (`dagua/layout/ops/pipelines/stress_majorization.py:78-166`).

The core numerical structure is dense all-pairs:

- shortest paths via dagua's graph utility
  (`dagua/layout/ops/stress.py:123-127`);
- inverse-square weights and weighted Laplacian
  (`dagua/layout/ops/stress.py:128-140`);
- dense pseudoinverse `np.linalg.pinv` (`dagua/layout/ops/stress.py:140`);
- classical-MDS warm start plus seeded Gaussian jitter
  (`dagua/layout/ops/stress.py:314-330`);
- global SMACOF matrix update using `laplacian_pinv @ (B(X) @ X)`
  (`dagua/layout/ops/stress.py:415-448`).

### OGDF reference pipeline

The benchmark reaches OGDF through `ogdf_competitor.py`, which serializes only
node count, edge pairs, and algorithm name (`dagua/eval/competitors/ogdf_competitor.py:138-144`).
For `ogdf_stress`, the adapter class sets `algorithm = "stress"` and
`max_nodes = 10_000` (`dagua/eval/competitors/ogdf_competitor.py:264-270`).
The C++ runner selects `ogdf::StressMinimization layout; layout.call(...)` for
`algorithm == "stress"` (`scripts/ogdf_runner.cpp:159-162`).

OGDF's `StressMinimization::call(GraphAttributes&)`:

1. handles `n <= 1` by setting all coordinates to zero
   (`StressMinimization.cpp:54-68`);
2. initializes dense shortest-path and weight matrices
   (`StressMinimization.cpp:71-73`, `StressMinimization.cpp:334-341`);
3. runs either Dijkstra if edge-cost attributes are enabled or BFS with uniform
   edge cost otherwise (`StressMinimization.cpp:74-83`);
4. computes an initial layout unless `hasInitialLayout(true)` was set
   (`StressMinimization.cpp:87-92`);
5. for disconnected graphs, replaces infinite distances by
   `avgEdgeCosts * sqrt(n)` unless component layout mode is active
   (`StressMinimization.cpp:94-100`, `StressMinimization.cpp:126-137`);
6. calculates inverse-square weights (`StressMinimization.cpp:139-149`);
7. minimizes stress with repeated serial coordinate updates
   (`StressMinimization.cpp:191-231`, `StressMinimization.cpp:233-303`).

The default OGDF constructor uses `m_numberOfIterations = 200`,
`m_edgeCosts = 100`, `m_componentLayout = false`, termination criterion
`None`, no fixed coordinates, 2D unless attributes request 3D
(`StressMinimization.h:54-66`). The runner never calls `setIterations`,
`setEdgeCosts`, `useEdgeCostsAttribute`, `hasInitialLayout`, or convergence
setters; therefore every `ogdf_stress` benchmark run uses those defaults
(`scripts/ogdf_runner.cpp:159-162`).

### Structural verdict

Both sides optimize classical stress with inverse-square graph-distance weights,
but they are not implementation-equivalent:

- Dagua uses a global dense pseudoinverse SMACOF update
  (`dagua/layout/ops/stress.py:426-448`).
- OGDF uses an in-place serial per-node coordinate vote sweep
  (`StressMinimization.cpp:237-303`).
- Dagua initializes with full classical MDS plus jitter
  (`dagua/layout/ops/stress.py:220-251`, `dagua/layout/ops/stress.py:327-330`).
- OGDF initializes with PivotMDS through `ComponentSplitterLayout` for the
  disconnected case (`StressMinimization.cpp:107-124`).
- Dagua edge length scale is roughly `sqrt(n) * 5` unless node sizes dictate
  otherwise (`dagua/layout/ops/stress.py:187-199`); OGDF edge cost is `100`
  (`StressMinimization.h:57-59`), later absorbed by Procrustes/scale-invariant
  comparisons but still affects finite-distance fill and intermediate stress.

## 3. Energy / loss / objective

### Shared objective family

Both sides intend weighted stress over unordered node pairs:

`sum_{i<j} w_ij * (d_ij - ||x_i - x_j||)^2`, with `w_ij = d_ij^-2`.

Dagua computes a symmetric full-matrix version with a `0.5` multiplier:

- weights: `1.0 / np.square(target_distances)` for positive target distances,
  diagonal zeroed (`dagua/layout/ops/stress.py:128-136`);
- stress: `0.5 * np.sum(weights * errors * errors)`
  (`dagua/layout/ops/stress.py:341-342`,
  `dagua/layout/ops/stress.py:446-448`,
  `dagua/layout/ops/stress.py:463-465`).

Because the weight and distance matrices are symmetric, dagua's `0.5 * sum over
all i,j` matches an unordered-pair sum if all off-diagonal pairs are symmetric.

OGDF computes only `v < w` in node order and has no `0.5` factor:

- weights: `1 / (shortestPathMatrix[v][w] * shortestPathMatrix[v][w])`
  (`StressMinimization.cpp:139-149`);
- stress: loop from first node, inner loop from `v->succ()`; add
  `weightMatrix[v][w] * (shortestPathMatrix[v][w] - dist)^2`
  (`StressMinimization.cpp:151-170`).

These are objective-equivalent for symmetric shortest paths, but not literally
identical in summation order. Dagua sums an `N x N` NumPy array in row-major
order (`dagua/layout/ops/stress.py:446-448`), while OGDF uses nested node
successor loops over half the matrix (`StressMinimization.cpp:154-167`).
Residual sub-percent differences can arise from floating-point summation order
even when coordinates are otherwise close.

### Distance terms

Dagua target distances:

- builds an undirected adjacency keeping the minimum duplicate-edge weight
  (`dagua/layout/ops/graph_utils.py:26-49`);
- unweighted path lengths come from BFS with integer edge increments of `1`
  (`dagua/layout/ops/graph_utils.py:52-72`);
- weighted path lengths use Dijkstra over float weights
  (`dagua/layout/ops/graph_utils.py:75-98`);
- unreachable pairs are filled with `max_distance + 1.0`
  (`dagua/layout/ops/graph_utils.py:340-352`).

OGDF target distances:

- unweighted default uses BFS with constant edge cost `m_edgeCosts`, default
  `100` (`StressMinimization.cpp:80-83`,
  `ShortestPathAlgorithms.h:49-54`, `ShortestPathAlgorithms.h:62-82`);
- weighted mode uses `GraphAttributes::edgeDoubleWeight`, but the current runner
  never enables that attribute or exports weights (`scripts/ogdf_runner.cpp:203-217`);
- unreachable pairs are filled with `m_avgEdgeCosts * sqrt(n)`, default
  `100 * sqrt(n)` (`StressMinimization.cpp:94-100`,
  `StressMinimization.cpp:126-137`).

This means connected unweighted graphs are mostly scale-equivalent:
dagua distances are `hop_count`, OGDF distances are `100 * hop_count`.
Disconnected graphs are not exactly scale-equivalent because dagua uses
`diameter + 1`, while OGDF uses `edgeCost * sqrt(n)`. After global scaling,
the relative disconnected-component target can differ materially, especially
for small sparse graphs where `diameter + 1` and `sqrt(n)` are not proportional.

### Objective exclusions

Dagua zeroes the diagonal explicitly (`dagua/layout/ops/stress.py:136`) and
uses `active_mask = weights > 0.0` for the `B(X)` ratio
(`dagua/layout/ops/stress.py:426-428`). Self-pairs therefore never participate.
OGDF skips `v == w` in weight calculation (`StressMinimization.cpp:141-146`),
calculates stress only for `w = v->succ()` (`StressMinimization.cpp:154-155`),
and skips the current node during serial updates
(`StressMinimization.cpp:244-247`).

## 4. Force / gradient computation

Neither implementation exposes an explicit force integrator. Both perform
majorization-style coordinate updates, but with different numerical schemes.

### Dagua global SMACOF update

Dagua constructs a `B(X)` matrix:

- pairwise Euclidean distances with floor `min_distance = 1e-9`
  (`dagua/layout/ops/stress.py:415-424`);
- ratio `target_distances / current_distances` on active pairs
  (`dagua/layout/ops/stress.py:426-428`);
- off-diagonal `B_ij = -w_ij * ratio_ij` and diagonal
  `B_ii = -sum_j B_ij` (`dagua/layout/ops/stress.py:430-435`);
- candidate `X' = L^+ B(X) X` (`dagua/layout/ops/stress.py:436`);
- candidate centered by subtracting mean (`dagua/layout/ops/stress.py:437`).

The weighted Laplacian is prepared as `L_ij = -w_ij`, `L_ii = sum_j w_ij`, then
pseudoinverted (`dagua/layout/ops/stress.py:138-140`). This is a parallel
matrix update over all nodes.

Dagua adds a monotonicity safeguard that is not present in OGDF: if candidate
stress exceeds current stress by more than `1e-8`, it blends halfway toward the
current state up to eight times, else rejects the step
(`dagua/layout/ops/stress.py:450-473`).

### OGDF serial coordinate voting

OGDF updates nodes one at a time and writes each new coordinate immediately:

- outer loop iterates `for (node v : G.nodes)` (`StressMinimization.cpp:237`);
- `currXCoord` and `currYCoord` reference the live graph attributes
  (`StressMinimization.cpp:241-242`);
- for each other node `w`, compute current Euclidean distance from live
  coordinates (`StressMinimization.cpp:244-252`);
- vote in x: start at `GA.x(w)`, then if distance is nonzero add
  `desDistance * (currXCoord - voteX) / euclideanDist`
  (`StressMinimization.cpp:259-266`);
- vote in y similarly (`StressMinimization.cpp:270-276`);
- divide accumulated weighted votes by total weight and assign immediately
  (`StressMinimization.cpp:290-300`).

This is a Gauss-Seidel-like serial majorization sweep. It uses updated positions
for earlier nodes when later nodes are processed. Dagua's global update is closer
to a Jacobi/dense linear solve step. That is the highest-confidence algorithmic
divergence and explains why the family can be strong-equivalent at the shape
level while not pixel-identical.

## 5. Initialization

### Dagua initialization

Dagua active pipeline:

- computes a full classical MDS embedding from all-pairs graph distances by
  double-centering squared distances (`dagua/layout/ops/stress.py:220-228`);
- eigen-decomposes the dense Gram matrix via `np.linalg.eigh`
  (`dagua/layout/ops/stress.py:228-240`);
- falls back to a deterministic line for degenerate spectra
  (`dagua/layout/ops/stress.py:241-249`);
- converts raw coordinates to `torch.float32` (`dagua/layout/ops/stress.py:251`);
- normalizes into a box using an extent based on `sqrt(n) * 5.0` or node sizes
  (`dagua/layout/ops/stress.py:187-199`,
  `dagua/layout/ops/stress.py:253-283`);
- adds seeded NumPy Gaussian jitter with `np.random.default_rng(problem.seed)`,
  `loc=0`, `scale=0.05`, shape `[N, 2]`
  (`dagua/layout/ops/stress.py:327-329`);
- recenters after jitter (`dagua/layout/ops/stress.py:329-330`).

The default seed comes from the classic adapter `_layout_seed`, defaulting to
`42` (`dagua/eval/competitors/classic_competitor.py:29-42`), and is passed into
the layout function by `_quick_classic`
(`dagua/eval/competitors/classic_competitor.py:1613-1618`).

### OGDF initialization

The runner initializes graph attributes with `std::rand() % 1000 / 10.0` after
`std::srand(42)` (`scripts/ogdf_runner.cpp:219-228`), but `StressMinimization`
defaults `m_hasInitialLayout = false` (`StressMinimization.h:54-57`) and then
calls `computeInitialLayout(GA)` (`StressMinimization.cpp:87-92`). Therefore
the runner's random initial coordinates are overwritten for the default stress
path.

OGDF's actual default warm start is PivotMDS:

- `computeInitialLayout` constructs `PivotMDS`, sets number of pivots to
  `DEFAULT_NUMBER_OF_PIVOTS = 50`, forwards edge-cost mode and edge cost, and
  calls it through `ComponentSplitterLayout` when `m_componentLayout` is false
  (`StressMinimization.cpp:107-124`);
- PivotMDS's own default constructor uses 250 pivots and edge cost 100, but
  stress minimization overrides pivots to 50 (`PivotMDS.h:59-64`,
  `StressMinimization.cpp:107-111`);
- for paths, PivotMDS skips SVD and lays nodes on a line with increments of
  `m_edgeCosts` or edge weights (`PivotMDS.cpp:114-118`,
  `PivotMDS.cpp:152-179`);
- otherwise it chooses pivots by max-min strategy starting from `G.firstNode()`
  (`PivotMDS.cpp:238-285`);
- it centers the pivot matrix and computes coordinates via SVD/power iteration
  (`PivotMDS.cpp:60-90`, `PivotMDS.cpp:360-390`);
- power iteration starts from `srand(SEED)` where `SEED = 0`, then `rand()`
  (`PivotMDS.h:108-109`, `PivotMDS.cpp:337-344`).

Disconnected OGDF initialization additionally goes through
`ComponentSplitterLayout`, which lays out each connected component separately
and reassembles/pack them (`ComponentSplitterLayout.cpp:63-135`). Dagua never
does component splitting or packing in this pipeline.

### Initialization divergence

Dagua's full classical MDS plus Gaussian jitter is not equivalent to OGDF's
50-pivot MDS, path special case, component splitter, and C `rand()` power
iteration initialization. Because stress majorization is convex only up to
translation/rotation for connected ideal stress but practical finite iterations,
disconnected fills, and serial updates leave initialization effects visible in
residual RMSD.

## 6. Iteration / convergence

### Dagua

- Public default iterations: `200`
  (`dagua/layout/ops/pipelines/stress_majorization.py:29-32`,
  `dagua/layout/ops/pipelines/stress_majorization.py:78-86`).
- Negative iterations rejected (`dagua/layout/ops/pipelines/stress_majorization.py:120-125`).
- Pipeline repeats exactly `iterations` times
  (`dagua/layout/ops/pipelines/stress_majorization.py:65-71`).
- No convergence threshold stops the loop early.
- The only adaptive behavior is the monotonicity safeguard:
  `stress_tolerance = 1e-8`, `max_halving_steps = 8`,
  `min_distance = 1e-9`
  (`dagua/layout/ops/stress.py:56-72`, `dagua/layout/ops/stress.py:450-473`).

### OGDF

- Constructor default `m_numberOfIterations = 200`
  (`StressMinimization.h:54-58`).
- Setter documentation says non-positive values use default 200
  (`StressMinimization.h:95-97`), but implementation uses `100`
  (`StressMinimization.h:222-224`). The runner never calls the setter, so this
  is not active in current comparisons but is a latent reference-adapter trap.
- Termination criterion defaults to `None` (`StressMinimization.h:60-62`), so
  `finished` returns true only when `numberOfPerformedIterations ==
  m_numberOfIterations` (`StressMinimization.cpp:305-331`).
- If callers set convergence by position difference or stress, OGDF uses
  `EPSILON = 10e-4` (`StressMinimization.cpp:50`) and tests either relative
  coordinate movement or relative stress decrease
  (`StressMinimization.cpp:312-328`). The runner does not enable either.

### Variant alignment issue

Dagua variants set `iterations` to 50/200/500 on the reimplementation side
(`dagua/eval/variants.py:826-856`). Their `original_params` are `{}` for all
three variants (`dagua/eval/variants.py:830-853`). The OGDF adapter has no
`layout_with_variant` override, so the base implementation ignores
`variant_params` and delegates to `layout` (`dagua/eval/competitors/base.py:64-91`).
The runner also exposes no iteration field (`dagua/eval/competitors/ogdf_competitor.py:138-144`,
`scripts/ogdf_runner.cpp:145-183`). Therefore:

- `classic_stress_maj_default` compares dagua 200 iterations vs OGDF 200.
- `classic_stress_maj_iter50` compares dagua 50 iterations vs OGDF 200.
- `classic_stress_maj_iter500` compares dagua 500 iterations vs OGDF 200.

The current report still calls all three strong-equivalent
(`eval_output/fidelity_report/report.md:85-87`), which implies the iteration
count is not a large fidelity lever in the mega-run, but it is an actual
parameter-alignment divergence.

## 7. Hyperparameter alignment table

| Parameter / behavior | Dagua default / current | OGDF default / current | Match? | Evidence |
| --- | --- | --- | --- | --- |
| Algorithm family | Dense stress majorization / SMACOF | Stress minimization by majorization | Partial | Dagua pipeline name and ops at `stress_majorization.py:29-75`; OGDF class at `StressMinimization.h:49-66` |
| Iterations, default variant | 200 | 200 | Y | Dagua default `iterations=200` at `stress_majorization.py:29-32`; OGDF constructor `m_numberOfIterations(200)` at `StressMinimization.h:54-58` |
| Iterations, iter50 variant | 50 | still 200 | N | Dagua variant at `variants.py:837-845`; OGDF original params `{}` at `variants.py:841-842`; runner no param at `ogdf_competitor.py:138-144` |
| Iterations, iter500 variant | 500 | still 200 | N | Dagua variant at `variants.py:848-856`; OGDF original params `{}` at `variants.py:852-853` |
| Convergence criterion | fixed steps only | fixed steps only by default | Y | Dagua fixed repeat at `stress_majorization.py:65-71`; OGDF criterion `None` at `StressMinimization.h:60-62` and `finished` default false until max iterations at `StressMinimization.cpp:305-331` |
| Early stress convergence | none | available but disabled | Y current / N capability | OGDF stress criterion at `StressMinimization.cpp:326-328`; no runner setter at `scripts/ogdf_runner.cpp:159-162` |
| Edge length / edge cost | unweighted graph distance unit 1; output extent `sqrt(n)*5` | edge cost 100 | Scale-equivalent only for connected unweighted graphs | Dagua BFS unit at `graph_utils.py:52-72`; OGDF BFS edgeCosts at `ShortestPathAlgorithms.h:62-82`; OGDF default at `StressMinimization.h:57-59` |
| Weighted edges | supported in dagua shortest paths | supported by OGDF class, not runner | N in benchmark | Dagua forwards `edge_weights` at `classic_competitor.py:1607-1608`; runner exports only `"edges"` at `ogdf_competitor.py:138-144` |
| Duplicate edges | dagua keeps minimum edge weight in shortest-path adjacency | OGDF graph stores all edges; BFS sees adjacency entries, Dijkstra has edge costs if enabled | Partial | Dagua min duplicate at `graph_utils.py:42-49`; OGDF iterates adjacency entries at `ShortestPathAlgorithms.h:73-80` |
| Self-loops | skipped in dagua adjacency | created in runner, but BFS sees twin self; stress skips self pairs | Mostly Y | Dagua skip at `graph_utils.py:42-44`; runner creates every edge at `scripts/ogdf_runner.cpp:214-217`; OGDF update skips `v == w` at `StressMinimization.cpp:244-247` |
| Disconnected target distance | `max finite distance + 1.0` | `avgEdgeCosts * sqrt(n)` | N | Dagua fill at `graph_utils.py:347-350`; OGDF fill at `StressMinimization.cpp:94-100` |
| Component-aware initialization | none | ComponentSplitterLayout around PivotMDS when disconnected | N | Dagua MDS on filled matrix at `stress.py:314-325`; OGDF component splitter at `StressMinimization.cpp:113-119` |
| Initial layout algorithm | full classical MDS + Gaussian jitter | PivotMDS, 50 pivots, component splitter; path special case | N | Dagua at `stress.py:220-251` and `stress.py:327-330`; OGDF at `StressMinimization.cpp:107-124`, `PivotMDS.cpp:114-149` |
| Initial RNG | NumPy `default_rng(seed)` Gaussian jitter | PivotMDS C `rand()` seeded to 0 for power iteration; runner C `rand(42)` overwritten | N | Dagua at `stress.py:327-329`; OGDF at `PivotMDS.cpp:337-344`; runner at `ogdf_runner.cpp:219-228` |
| Benchmark seed control | benchmark seed forwarded to dagua | OGDF adapter ignores seed | N | Dagua `_layout_seed` at `classic_competitor.py:29-42`; OGDF `del seed` at `ogdf_competitor.py:179-203` |
| Update scheme | global pseudoinverse update | serial coordinate votes, in-place | N | Dagua at `stress.py:426-437`; OGDF at `StressMinimization.cpp:237-303` |
| Monotonic safeguard | stress check + halving/reject | none | N | Dagua at `stress.py:450-473`; OGDF no equivalent in `nextIteration` at `StressMinimization.cpp:233-303` |
| Coordinate centering after each step | yes | no explicit centering after sweep | N | Dagua at `stress.py:437`; OGDF update writes raw weighted votes at `StressMinimization.cpp:290-300` |
| Fixed coordinates | unsupported in dagua pipeline | available but disabled | Y current / N capability | OGDF flags at `StressMinimization.h:62-64`, checks at `StressMinimization.cpp:259-299` |
| 3D | dagua stress pipeline returns `[N,2]` only | OGDF supports 3D if attributes include it; runner only node/edge graphics | Y current | Dagua return shape at `stress_majorization.py:108-110`; runner attrs at `ogdf_runner.cpp:203-206`; OGDF `m_use3D` at `StressMinimization.cpp:54-55` |
| Output dtype | `torch.float32` | C++ double printed to JSON, parsed as `torch.float32` | Final Y / internal N | Dagua final cast at `stress.py:577-578`; OGDF adapter cast at `ogdf_competitor.py:162-166` |
| Max nodes in benchmark | 500 for `classic_stress_maj` | 10,000 for `ogdf_stress` | N | Dagua at `classic_competitor.py:1062-1068`; OGDF at `ogdf_competitor.py:264-270` |
| Variant stochastic classification | dagua stress_maj marked stochastic | ogdf_stress marked deterministic | N | `variants.py:1829` and `variants.py:1858` |

## 8. Edge cases

### Empty graph

Dagua active entrypoint returns an empty `[0, 2]` float32 tensor and empty trace
when requested (`dagua/layout/ops/pipelines/stress_majorization.py:134-139`).
OGDF adapter also returns `torch.zeros((0, 2), dtype=torch.float32)` before
subprocess execution (`dagua/eval/competitors/ogdf_competitor.py:131-133`).
This is aligned for shape and dtype; values are vacuous.

### Singleton graph

Dagua returns a single zero coordinate (`dagua/layout/ops/pipelines/stress_majorization.py:140-143`).
OGDF `StressMinimization::call` sets x/y/z to zero for `numberOfNodes() <= 1`
(`StressMinimization.cpp:57-68`). Aligned.

### Self-loops

Dagua shortest-path adjacency skips self-loops (`dagua/layout/ops/graph_utils.py:42-44`).
OGDF runner creates self-loop edges if present (`scripts/ogdf_runner.cpp:214-217`).
In OGDF BFS, a self-loop appears as an adjacency entry whose twin node is the
current node; since the current node is already marked, it does not change
distance (`ShortestPathAlgorithms.h:63-80`). Stress and serial update skip
self-pairs (`StressMinimization.cpp:141-146`, `StressMinimization.cpp:244-247`).
For unweighted stress, self-loops should be effectively ignored on both sides.
For weighted mode, dagua also skips self-loop weights; OGDF runner does not
expose weighted mode, so benchmark behavior remains effectively aligned.

### Multi-edges

Dagua's shortest-path adjacency keeps the minimum duplicate weight and ignores
duplicate unweighted edges after the first neighbor map entry
(`dagua/layout/ops/graph_utils.py:42-49`). OGDF graph stores all edges
(`scripts/ogdf_runner.cpp:214-217`). For unweighted BFS, duplicate adjacency
entries do not change shortest-path distances because nodes are marked after
first discovery (`ShortestPathAlgorithms.h:63-80`). For weighted Dijkstra,
multiple edges would matter through the edge-cost array, but benchmark OGDF does
not use edge costs. Current unweighted benchmark behavior is mostly aligned.

### Disconnected components

This is a major divergence.

Dagua fills every unreachable pair with `max finite distance + 1.0`
(`dagua/layout/ops/graph_utils.py:347-350`) and then optimizes one global dense
stress problem (`dagua/layout/ops/stress.py:123-145`). OGDF fills unreachable
distances with `m_avgEdgeCosts * sqrt(n)` after computing a PivotMDS initial
layout through `ComponentSplitterLayout` (`StressMinimization.cpp:94-100`,
`StressMinimization.cpp:107-124`). The component splitter lays out components
individually and packs them (`ComponentSplitterLayout.cpp:63-135`).

For disconnected graphs, relative inter-component separation is therefore not
the same even after Procrustes scaling. Dagua's fill depends on graph diameter;
OGDF's depends on total node count and edge cost.

### Weighted edges

Dagua graph supports `edge_weights` as a float32 tensor
(`dagua/graph.py:90-95`, `dagua/graph.py:1619-1684`), fills missing weights
with 1.0 when any weight exists (`dagua/graph.py:345-359`), and forwards weights
to classic pipelines (`dagua/eval/competitors/classic_competitor.py:1607-1608`).
Dagua stress then computes weighted shortest paths if `edge_weights is not None`
(`dagua/layout/ops/graph_utils.py:345-346`).

OGDF `StressMinimization` supports `GraphAttributes::edgeDoubleWeight` via
`useEdgeCostsAttribute(true)` (`StressMinimization.h:102-103`,
`StressMinimization.cpp:74-83`), but the current runner creates graph attributes
with only `nodeGraphics | edgeGraphics` (`scripts/ogdf_runner.cpp:203-206`) and
serializes no weights (`dagua/eval/competitors/ogdf_competitor.py:138-144`).
Weighted fidelity graphs therefore compare weighted dagua stress against
unweighted OGDF stress. That is likely one of the largest remaining systematic
divergences hidden by aggregate strong-equivalence.

## 9. Numerical precision

Dagua mixes NumPy float64, torch float32, and final torch float32:

- shortest-path matrices are returned as `np.float64`
  (`dagua/layout/ops/graph_utils.py:350-352`);
- state distance matrix is stored as `torch.from_numpy(target_distances)`, so
  it remains float64 on CPU (`dagua/layout/ops/stress.py:142-144`);
- classical MDS builds centering and coordinates in `np.float64`
  (`dagua/layout/ops/stress.py:220-240`), then casts raw coordinates to
  `torch.float32` (`dagua/layout/ops/stress.py:251`);
- normalized baseline is torch float32 (`dagua/layout/ops/stress.py:325`);
- jitter is NumPy float64 and addition promotes initialized positions to
  float64 (`dagua/layout/ops/stress.py:327-330`);
- SMACOF iteration runs in NumPy float64 arrays
  (`dagua/layout/ops/stress.py:407-448`);
- final positions are cast to `torch.float32`
  (`dagua/layout/ops/stress.py:573-578`).

OGDF uses C++ `double` throughout stress minimization:

- matrices are `NodeArray<NodeArray<double>>`
  (`StressMinimization.cpp:71-72`);
- coordinate differences and stress are `double`
  (`StressMinimization.cpp:151-169`);
- serial update accumulators are `double`
  (`StressMinimization.cpp:237-288`).

The benchmark adapter converts OGDF JSON positions to `torch.float32`
(`dagua/eval/competitors/ogdf_competitor.py:162-166`). Thus both sides meet at
float32 for evaluation, but dagua has internal float32 boundaries in its MDS
warm start that OGDF does not. The most relevant numerical residuals are:

- MDS eigenvector sign and ordering (`np.linalg.eigh`) vs OGDF power iteration;
- row-major full-matrix NumPy summation vs OGDF half-matrix nested loops;
- dense pseudoinverse solve (`np.linalg.pinv`) vs serial weighted averaging;
- final text JSON precision from OGDF runner using default `std::cout` numeric
  formatting (`scripts/ogdf_runner.cpp:232-240`) before float32 parsing.

## 10. RNG semantics

Dagua's active stress-majorization pipeline does **not** use torch RNG for its
stochastic component. It accepts a seed, then uses NumPy's modern PCG-backed
`np.random.default_rng(problem.seed)` to draw Gaussian jitter
(`dagua/layout/ops/stress.py:327-329`). The classic competitor forwards the
benchmark seed, defaulting to 42 (`dagua/eval/competitors/classic_competitor.py:29-42`,
`dagua/eval/competitors/classic_competitor.py:1613-1618`).

OGDF benchmark runs ignore the Python-side seed. `_OGDFBase.layout` explicitly
deletes `seed` (`dagua/eval/competitors/ogdf_competitor.py:179-203`), and the
payload contains no seed (`dagua/eval/competitors/ogdf_competitor.py:138-144`).
The runner always calls `ogdf::setSeed(42)` and `std::srand(42)`, then fills
initial x/y coordinates with C `rand()` (`scripts/ogdf_runner.cpp:219-228`).
For stress specifically, those random x/y coordinates are overwritten because
`m_hasInitialLayout` defaults false and `computeInitialLayout` runs
(`StressMinimization.h:54-57`, `StressMinimization.cpp:87-92`).

PivotMDS then seeds C `rand()` with `SEED = 0` for power iteration
(`PivotMDS.h:108-109`, `PivotMDS.cpp:337-344`). That sequence is unrelated to
NumPy `default_rng`, and also unrelated to PyTorch RNG.

Answer to the required RNG question: dagua's torch seed does not produce the
same sequence as the reference RNG, because dagua stress majorization does not
use torch RNG here. Its stochastic jitter uses NumPy `default_rng`; OGDF stress
uses deterministic PivotMDS with C `rand()` seeded to 0 inside PivotMDS and the
adapter does not forward benchmark seeds.

## 11. Edge-case bugs / suspicious divergences

1. **Variant iteration parameters are not applied to OGDF.** `iter50` and
   `iter500` variants configure dagua only (`dagua/eval/variants.py:837-856`);
   OGDF original params are `{}` (`dagua/eval/variants.py:841-853`), base
   `layout_with_variant` ignores params (`dagua/eval/competitors/base.py:64-91`),
   and the runner cannot accept an iteration field
   (`dagua/eval/competitors/ogdf_competitor.py:138-144`). This is not an
   off-by-one; it is a benchmark wiring mismatch.
2. **Weighted graphs compare different problems.** Dagua forwards graph weights
   (`classic_competitor.py:1607-1608`), while OGDF runner serializes no weights
   and does not enable `edgeDoubleWeight` (`ogdf_runner.cpp:203-217`). Weighted
   benchmark rows can be strong-equivalent only after metric normalization, not
   objective-equivalent.
3. **Disconnected finite fill is not OGDF-compatible.** Dagua uses
   `max_distance + 1.0` (`graph_utils.py:347-350`); OGDF uses
   `m_avgEdgeCosts * sqrt(n)` (`StressMinimization.cpp:94-100`). This changes
   relative inter-component forces.
4. **Update method mismatch.** Dagua's global pseudoinverse update is not OGDF's
   serial in-place vote update (`stress.py:426-437` vs
   `StressMinimization.cpp:237-303`). This is the largest algorithmic
   mismatch for connected unweighted graphs.
5. **Initialization mismatch.** Dagua full classical MDS + jitter is not OGDF
   50-pivot MDS + component splitter/path special case (`stress.py:220-251`,
   `stress.py:327-330`, `StressMinimization.cpp:107-124`,
   `PivotMDS.cpp:114-149`). This can dominate small graphs and disconnected
   cases.
6. **Potential doc/code mismatch in OGDF setter.** OGDF header comments say
   `setIterations <= 0` uses default 200 (`StressMinimization.h:95-97`), but
   implementation uses 100 (`StressMinimization.h:222-224`). Not active today,
   but any future runner exposing `iterations` should avoid relying on
   non-positive fallback.
7. **Runner default coordinates are misleading for stress.** The runner seeds
   and writes random initial positions (`scripts/ogdf_runner.cpp:219-228`), but
   default `StressMinimization` overwrites them (`StressMinimization.cpp:87-92`).
   If future code sets `hasInitialLayout(true)`, this latent path suddenly
   becomes active and uses C `rand(42)` coordinates unlike dagua.
8. **OGDF output precision is uncontrolled.** Runner uses raw `std::cout`
   formatting for doubles (`scripts/ogdf_runner.cpp:232-240`), then Python
   parses to float32 (`ogdf_competitor.py:162-166`). This is probably tiny RMSD,
   but it is an avoidable residual source.

## 12. Ranked fix list

Ranked by expected RMSD/fidelity impact, not by implementation desirability.
All are future-scope only; no edits were made in this round.

1. **Implement an OGDF-compatible serial stress sweep mode in dagua.**
   Replace or optionally bypass the dense pseudoinverse SMACOF step
   (`dagua/layout/ops/stress.py:415-448`) with OGDF's in-place per-node vote
   formula (`StressMinimization.cpp:237-303`). Expected impact: highest on
   connected unweighted graphs where all other differences are mostly scale or
   initialization. Fix size: medium, ~80-140 lines including tests and a config
   flag or alternate op.
2. **Align OGDF initialization: PivotMDS 50 pivots plus component handling.**
   Dagua currently uses full classical MDS plus Gaussian jitter
   (`dagua/layout/ops/stress.py:220-251`, `dagua/layout/ops/stress.py:327-330`);
   OGDF uses `PivotMDS` with 50 pivots and `ComponentSplitterLayout`
   (`StressMinimization.cpp:107-124`, `PivotMDS.cpp:238-390`,
   `ComponentSplitterLayout.cpp:63-135`). Expected impact: high on small,
   path-like, and disconnected graphs. Fix size: large, ~200-400 lines if
   implemented faithfully, smaller if reusing existing dagua pivot-MDS pipeline
   as a warm start.
3. **Expose and forward OGDF iteration counts in the runner and adapter.**
   Current variants configure dagua only (`dagua/eval/variants.py:826-856`);
   OGDF always defaults to 200 (`scripts/ogdf_runner.cpp:159-162`). Add
   optional JSON `iterations`, call `layout.setIterations`, and set
   `original_params` for iter50/iter500. Expected impact: medium/low in current
   aggregate because all three variants already sit at median RMSD ~0.041, but
   correctness impact is high. Fix size: small, ~30-60 lines across
   `scripts/ogdf_runner.cpp`, `ogdf_competitor.py`, and `variants.py`.
4. **Export weighted edges to OGDF stress.** Dagua forwards `edge_weights`
   (`classic_competitor.py:1607-1608`); OGDF runner drops them
   (`ogdf_competitor.py:138-144`, `scripts/ogdf_runner.cpp:203-217`). Add
   weights to JSON, enable `GraphAttributes::edgeDoubleWeight`, set
   `doubleWeight`, and call `useEdgeCostsAttribute(true)`. Expected impact:
   high on `weighted_chain_20`, `weighted_clusters_3x10`,
   `weighted_karate_34`, and `heavy_tail_weights_50`; low on unweighted rows.
   Fix size: medium, ~80-160 lines because the runner's hand-rolled JSON parser
   must parse numeric arrays.
5. **Change dagua disconnected-distance fill for OGDF compatibility.**
   Dagua fill: `max_distance + 1.0` (`dagua/layout/ops/graph_utils.py:347-350`);
   OGDF fill: `avgEdgeCosts * sqrt(n)` (`StressMinimization.cpp:94-100`). For
   unweighted connected scale, use `sqrt(n)` in dagua's unit-distance scale.
   Expected impact: medium on disconnected graphs, near zero on connected.
   Fix size: small/medium, ~30-80 lines plus regression tests.
6. **Remove Gaussian jitter or make it OGDF-like for this family.**
   Dagua adds `np.random.default_rng(seed).normal(..., scale=0.05)`
   (`dagua/layout/ops/stress.py:327-329`); OGDF default stress does not add a
   comparable post-PivotMDS jitter. Expected impact: medium on symmetric small
   graphs and seed-to-seed variability. Fix size: small if adding
   `jitter_scale=0` option, medium if matching PivotMDS C `rand()` semantics.
7. **Preserve double precision through dagua MDS warm start.**
   Dagua casts raw classical-MDS coordinates to float32 before normalization
   (`dagua/layout/ops/stress.py:251`), then returns to float64 after adding
   jitter (`dagua/layout/ops/stress.py:329`). OGDF remains double until adapter
   output (`StressMinimization.cpp:151-303`, `ogdf_competitor.py:162-166`).
   Expected impact: low/sub-percent. Fix size: small, ~10-30 lines.
8. **Set explicit OGDF output precision.**
   Runner prints doubles without `std::setprecision`
   (`scripts/ogdf_runner.cpp:232-240`). Expected impact: tiny, but cheap and
   makes downstream comparisons less lossy before float32 conversion. Fix size:
   tiny, ~5 lines.

## 13. Recommended Round 22+ fix scope

Recommended bundle for one follow-up round:

1. Add an OGDF-compatible mode for dagua stress majorization that uses the
   serial in-place vote update from `StressMinimization.cpp:237-303`, while
   keeping the current dense SMACOF path as the existing public behavior unless
   benchmark variants explicitly request OGDF fidelity.
2. Add an OGDF distance-fill option: connected unweighted graphs stay
   scale-equivalent, disconnected unreachable pairs use `sqrt(n)` in dagua's
   unit-cost scale, matching OGDF's `100 * sqrt(n)` up to global scale
   (`StressMinimization.cpp:94-100`).
3. Set dagua's stress-majorization jitter to zero or make it configurable for
   OGDF-fidelity variants, because OGDF default warm start is deterministic
   PivotMDS, not MDS plus Gaussian jitter.
4. In a separate small adapter round, expose OGDF `iterations` in
   `scripts/ogdf_runner.cpp` and `ogdf_competitor.py`, then populate
   `original_params` for `classic_stress_maj_iter50` and
   `classic_stress_maj_iter500`. This should be separate because it changes
   reference artifacts and benchmark cache keys.
5. Defer full weighted-edge export unless weighted stress rows become a priority
   target. It is correctness-important, but the runner parser work is larger
   than the iteration fix and its impact is limited to weighted graphs.

Expected outcome: the serial update + disconnected fill + no-jitter bundle is
the top-K dagua-side lever for reducing residual RMSD without touching the OGDF
runner. Iteration forwarding is the top benchmark-wiring correctness fix. Full
PivotMDS initialization compatibility is likely the next-largest algorithmic
lever, but it is large enough to deserve its own round.
