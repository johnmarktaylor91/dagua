# Round 21 adversarial diff: `classic_maxent_stress` vs `ogdf_stress`

Diagnosis-only report for the maxent-stress family. The current mega-run marks all
maxent-stress variants `strong_equivalent`, but this pass catalogs every visible
implementation and harness divergence between dagua's `classic_maxent_stress` and
the OGDF `StressMinimization` reference used by `ogdf_stress`.

## 1. Files read

Dagua implementation and wiring:

- `dagua/layout/ops/maxent_stress.py:1-782` -- main maxent-stress ops:
  initialization, state preparation, Adam gradient branch, dense majorization branch,
  and final normalization.
- `dagua/layout/ops/pipelines/maxent_stress.py:1-241` -- branch selection and public
  `layout_maxent_stress_pipeline`.
- `dagua/layout/ops/pipelines/pivot_mds.py:1-137` -- PivotMDS warm start used by
  `MaxentInitializePositions`.
- `dagua/layout/ops/distance.py:839-1044` -- PivotMDS pivot selection and pivot
  distance queries.
- `dagua/layout/ops/embed.py:277-309` and `dagua/layout/ops/embed.py:1437-1475` --
  PivotMDS coordinate recovery.
- `dagua/layout/ops/postprocess.py:870-890` and `dagua/layout/ops/postprocess.py:946-999`
  -- classical/PivotMDS final normalization.
- `dagua/layout/ops/graph_utils.py:26-129` and `dagua/layout/ops/graph_utils.py:166-213`
  -- undirected adjacency, BFS/Dijkstra/APSP, output extent, and normalization helpers.
- `dagua/layout/ops/base.py:202-281` and `dagua/layout/ops/base.py:364-438` -- pipeline
  sequencing and `Repeat` step semantics.
- `dagua/layout/ops/converge.py:64-134` -- fixed-step budget op.
- `dagua/layout/ops/state.py:113-160` and `dagua/layout/ops/state.py:320-410` --
  problem fields and state bookkeeping.
- `dagua/layout/ops/loss_classic.py:291-398`, `dagua/layout/ops/loss_classic.py:1898-1931`,
  `dagua/layout/ops/loss_classic.py:2310-2373`, and
  `dagua/layout/ops/loss_classic.py:2750-2771` -- related maxent loss primitives still
  present in the ops loss library.
- `dagua/eval/variants.py:800-870`, `dagua/eval/variants.py:1023-1088`, and
  `dagua/eval/variants.py:1820-1865` -- variant configs, original pairings, and
  stochasticity metadata.
- `dagua/eval/competitors/classic_competitor.py:219-223` and
  `dagua/eval/competitors/classic_competitor.py:1354-1410` -- classic competitor
  registration and wrapper behavior.
- `dagua/eval/competitors/ogdf_competitor.py:1-303` -- OGDF adapter and `ogdf_stress`
  registration.
- `scripts/ogdf_runner.cpp:1-246` -- C++ subprocess wrapper used by `ogdf_stress`.

Reference implementation:

- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:1-344`
  -- OGDF stress majorization implementation.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:1-238`
  -- OGDF stress defaults and configuration API.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:1-393`
  -- OGDF PivotMDS warm-start implementation used by `StressMinimization`.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/PivotMDS.h:1-168`
  -- OGDF PivotMDS defaults and RNG seed.

Existing analysis and run data:

- `eval_output/fidelity_report/report.md:55-59` -- current strong-equivalence verdicts
  for maxent-stress variants.
- `eval_output/fidelity_report/data/algorithm_summary.csv:46-50` -- numeric summary for
  maxent-stress variants.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:196-197` --
  sprint context noting OGDF families already strong-equivalent but still correctness
  targets.

## 2. Overall pipeline structure

OGDF reference:

1. `StressMinimization::call(GraphAttributes&)` detects 2D/3D mode, returns zeros for
   graphs with at most one node, initializes shortest-path and weight matrices, then
   computes APSP via either Dijkstra over edge weights or BFS with uniform edge costs
   (`StressMinimization.cpp:54-84`).
2. `StressMinimization::call(GA, shortestPathMatrix, weightMatrix)` computes PivotMDS
   initialization when no initial layout is declared, replaces disconnected infinite
   distances with `avgEdgeCosts * sqrt(n)` when components are not laid out separately,
   computes weights `1 / d_ij^2`, then minimizes stress (`StressMinimization.cpp:87-105`).
3. `computeInitialLayout()` constructs `PivotMDS`, sets 50 pivots, propagates edge-cost
   settings, and runs through `ComponentSplitterLayout` when not already in component
   layout mode (`StressMinimization.cpp:107-124`).
4. `minimizeStress()` repeatedly calls `nextIteration()` until either the fixed iteration
   count is reached or an optional convergence criterion fires (`StressMinimization.cpp:191-231`).
   Default termination criterion is `None`, so fixed 200 iterations win by default
   (`StressMinimization.h:54-66`, `StressMinimization.cpp:305-331`).
5. `nextIteration()` performs serial in-place Gauss-Seidel majorization over all nodes
   (`StressMinimization.cpp:233-303`).

Dagua `classic_maxent_stress`:

1. `layout_maxent_stress_pipeline()` validates inputs, special-cases 0 and 1 nodes, builds
   a `LayoutProblem`, and applies a selected pipeline (`pipelines/maxent_stress.py:145-233`).
2. Branch selection chooses the dense majorization branch only when
   `use_majorization=True`, `use_entropy=False`, `num_nodes <= 5000`, and `steps == 200`
   (`pipelines/maxent_stress.py:101-142`). This is OGDF-like only for the default
   non-entropy 200-step case.
3. The majorization branch is `FixedSteps -> MaxentInitializePositions(for_majorization=True)
   -> MaxentPrepareState(for_majorization=True) -> Repeat(MaxentMajorizationStep) ->
   MaxentFinalizePositions(for_majorization=True)` (`pipelines/maxent_stress.py:30-58`).
4. The gradient branch is `FixedSteps -> MaxentInitializePositions(for_majorization=False)
   -> MaxentPrepareState(for_majorization=False, use_entropy=...) ->
   MaxentInitializeOptimizer -> Repeat(MaxentGradientStep) -> MaxentFinalizePositions`
   (`pipelines/maxent_stress.py:61-98`).
5. The evaluation wrapper for `classic_maxent_stress` hard-codes `steps=200`, `alpha=1.0`,
   and a seed passed through `_layout_seed(seed)` (`classic_competitor.py:1392-1399`).
   Variant params in `variants.py` define entropy/alpha/step variants, but the specialized
   class wrapper itself does not expose them (`classic_competitor.py:1354-1410`). The generic
   spec table does list `classic_maxent_stress` with default params
   `{"steps": 200, "alpha": 1.0}` (`classic_competitor.py:219-223`).

Important structure mismatch:

- OGDF `StressMinimization` has no entropy objective and no Adam/gradient branch
  (`StressMinimization.cpp:191-303`). Dagua's entropy, alpha=2, steps=50, and steps=400
  variants are therefore intentionally paired against the same `ogdf_stress` baseline
  but are not same-algorithm configurations (`variants.py:1035-1088`).
- Dagua's default maxent-stress branch is structurally close to OGDF stress majorization:
  both use PivotMDS initialization, full APSP, weights `d^-2`, and serial in-place updates
  (`maxent_stress.py:115-128`, `maxent_stress.py:194-223`,
  `maxent_stress.py:697-731`; `StressMinimization.cpp:87-105`,
  `StressMinimization.cpp:233-303`).

## 3. Energy / loss / objective

OGDF stress objective:

- OGDF evaluates stress over unordered pairs `v < w` in `calcStress()`:
  Euclidean distance is `sqrt(xDiff^2 + yDiff^2 + zDiff^2)`, and the energy term is
  `weightMatrix[v][w] * (shortestPathMatrix[v][w] - dist)^2`
  (`StressMinimization.cpp:151-169`).
- The pair weight is exactly `w_ij = d_ij^-2` in `calcWeights()`
  (`StressMinimization.cpp:139-149`).
- Shortest-path target distances use BFS with uniform edge cost `m_edgeCosts` when no
  edge-cost attribute is present (`StressMinimization.cpp:76-83`). The default edge cost
  is `100` (`StressMinimization.h:54-66`).
- Disconnected infinite distances are replaced by `m_avgEdgeCosts * sqrt(n)` when
  `m_componentLayout` is false (`StressMinimization.cpp:94-100`). The header documents
  the same policy (`StressMinimization.h:1-7`).

Dagua majorization objective:

- Dagua builds raw APSP, fills disconnected entries with `average_edge_cost * sqrt(n)`,
  converts to `torch.float64`, and computes `weight_matrix = graph_distances^-2` off the
  diagonal (`maxent_stress.py:194-223`).
- Dagua's majorization step uses the same vote formula as OGDF:
  for each other node, compute Euclidean distance, vote at the target graph distance along
  the ray from the other node, accumulate `weight * vote`, then divide by total weight
  (`maxent_stress.py:697-731`). OGDF does the same at
  `StressMinimization.cpp:237-303`.
- Dagua does not explicitly calculate stress for convergence in the default branch.
  `FixedSteps` sets `state.total_steps`, and `Repeat` runs exactly `n` iterations unless
  an inner op sets `state.converged`, which this branch does not do
  (`converge.py:129-134`, `base.py:429-438`, `pipelines/maxent_stress.py:46-56`).
  This matches OGDF's default `TerminationCriterion::None`, which returns false until
  `numberOfPerformedIterations == m_numberOfIterations` (`StressMinimization.h:54-66`,
  `StressMinimization.cpp:305-331`).

Dagua gradient objective:

- For small graphs, dagua enumerates all upper-triangle stress pairs and computes
  `sum((1 / target^2) * (||x_i - x_j|| - target)^2)` (`maxent_stress.py:225-245`,
  `maxent_stress.py:523-536`).
- If `use_entropy=True`, dagua adds `alpha * -sum(log(non_edge_distance))` for exact
  non-edges on graphs at or below the full-stress cutoff (`maxent_stress.py:248-269`,
  `maxent_stress.py:538-548`).
- For larger graphs, dagua uses pivot distances as an approximation to all-pairs stress:
  `torch.cdist(positions, pivot_positions)` against `pivot_distances`, with weights
  `target^-2` where targets are positive (`maxent_stress.py:559-572`). There is no OGDF
  equivalent inside `StressMinimization`; OGDF always materializes full APSP matrices
  for the stress run (`StressMinimization.cpp:71-84`, `StressMinimization.cpp:334-341`).
- For larger entropy graphs, dagua samples non-edges, computes `-sum(log(d))`, and scales
  by `total_non_edges / sampled_count` (`maxent_stress.py:576-641`). OGDF stress has no
  entropy term (`StressMinimization.cpp:151-169`, `StressMinimization.cpp:233-303`).

Residual objective mismatches:

- Edge length scale differs by a global constant before downstream normalization: OGDF
  default `m_edgeCosts=100` (`StressMinimization.h:54-66`), while dagua unweighted BFS
  distance increments are `1` (`graph_utils.py:52-72`) and `average_edge_cost=1.0`
  when no edge weights are provided (`maxent_stress.py:178-184`). Pure stress layouts are
  scale-invariant under the final normalization used by fidelity, but finite precision
  and runner quantization can still leave sub-percent differences.
- Dagua clamps gradient Euclidean distances at `1e-3` in the Adam branch
  (`maxent_stress.py:37`, `maxent_stress.py:530-534`, `maxent_stress.py:561-572`).
  OGDF's majorization branch checks `dist != 0` and otherwise leaves the vote at the
  neighbor coordinate (`StressMinimization.cpp:248-264`, `StressMinimization.cpp:270-285`).
  The dense dagua majorization branch matches OGDF's zero-distance behavior
  (`maxent_stress.py:714-723`).

## 4. Force / gradient computation

OGDF:

- The reference is not a force or gradient descent solver. It performs stress majorization
  by serial in-place coordinate updates (`StressMinimization.cpp:233-303`).
- Each node update uses the current coordinate references `double& currXCoord = GA.x(v)`
  and `double& currYCoord = GA.y(v)`, and writes back immediately after computing the
  node's weighted average (`StressMinimization.cpp:237-244`,
  `StressMinimization.cpp:290-300`). This means nodes later in iteration order see earlier
  nodes' already-updated coordinates.
- 3D is possible if `GraphAttributes::threeD` is present and 2D is not forced
  (`StressMinimization.cpp:54-56`, `StressMinimization.cpp:248-252`,
  `StressMinimization.cpp:278-300`), but the dagua runner creates only node/edge graphics,
  not 3D attributes (`scripts/ogdf_runner.cpp:203-207`), so this pairing is 2D.

Dagua majorization:

- Dagua's `MaxentMajorizationStep` is also serial and in-place: it iterates `node_index`
  from `0` to `N-1`, reads `positions` directly, and writes `positions[node_index]` after
  accumulating votes (`maxent_stress.py:697-731`).
- The vote formula is line-for-line equivalent to OGDF's 2D formula:
  `vote_x = other_x + desired_distance * (current_x - other_x) / euclidean_distance`
  when the distance is nonzero (`maxent_stress.py:716-723`;
  `StressMinimization.cpp:257-276`).
- Dagua has no fixed-coordinate flags in this branch; OGDF supports fixed X/Y/Z flags,
  but the runner never sets them (`StressMinimization.h:78-85`,
  `scripts/ogdf_runner.cpp:159-162`).

Dagua gradient branch:

- Dagua uses PyTorch autograd and Adam. `MaxentInitializeOptimizer` sets
  `requires_grad_(True)`, creates `torch.optim.Adam`, and stores initial/final learning
  rates (`maxent_stress.py:450-465`).
- `MaxentGradientStep` builds stress and entropy losses, calls `loss.backward()`,
  `optimizer.step()`, then linearly anneals LR toward a computed floor
  (`maxent_stress.py:522-652`).
- This branch is algorithmically different from OGDF stress majorization and should not be
  expected to reproduce OGDF beyond layout-family similarity.

## 5. Initialization

OGDF initialization:

- `StressMinimization` computes initial layout unless `m_hasInitialLayout` is true
  (`StressMinimization.cpp:87-92`). The constructor default is false
  (`StressMinimization.h:54-66`).
- Initial layout is `PivotMDS` with exactly 50 pivots for stress minimization
  (`StressMinimization.cpp:107-112`). `DEFAULT_NUMBER_OF_PIVOTS` is 50
  (`StressMinimization.cpp:50-52`, `StressMinimization.h:115-120`).
- For disconnected graphs, OGDF routes PivotMDS through `ComponentSplitterLayout`
  (`StressMinimization.cpp:113-119`); otherwise it calls `PivotMDS` directly
  (`StressMinimization.cpp:120-123`).
- `PivotMDS` itself has a path fast path: if `getRootedPath()` identifies a simple path
  after `makeSimpleUndirected`, it lays nodes on a line with increments of average edge
  cost / edge cost (`PivotMDS.cpp:114-118`, `PivotMDS.cpp:152-179`,
  `PivotMDS.cpp:296-317`).
- For non-path connected graphs, OGDF PivotMDS starts with `G.firstNode()` as the first
  pivot and uses deterministic max-min pivot selection (`PivotMDS.cpp:238-284`).
- OGDF's PivotMDS SVD uses a custom power iteration. The eigenvectors are randomized with
  C `rand()` after `srand(SEED)`, where `SEED=0` (`PivotMDS.h:101-110`,
  `PivotMDS.cpp:181-236`, `PivotMDS.cpp:337-391`).

Dagua initialization:

- `MaxentInitializePositions` calls dagua's `layout_pivot_mds_pipeline` with
  `n_pivots=min(50, num_nodes)`, `seed=problem.seed`, and optional edge weights
  (`maxent_stress.py:115-122`).
- Majorization then casts the warm start to CPU `float64`; the gradient branch casts to
  output device `float32` (`maxent_stress.py:123-128`).
- Dagua PivotMDS builds undirected adjacency with `dedup="min"` via `BuildAdjacency`,
  selects pivots, queries pivot distances, computes coordinates, and normalizes
  (`pipelines/pivot_mds.py:51-64`).
- Dagua pivot selection chooses its first pivot with `torch.randint(0, num_nodes, (1,))`
  from a `torch.Generator` seeded by `problem.seed`, then uses max-min selection
  (`distance.py:855-874`, `distance.py:956-980`). This differs from OGDF's
  `G.firstNode()` first pivot (`PivotMDS.cpp:264-284`).
- Dagua PivotMDS coordinate recovery uses `torch.linalg.svd` on the centered pivot-distance
  matrix (`embed.py:277-309`). OGDF uses a custom power-iteration SVD over `C^T C`
  (`PivotMDS.cpp:360-391`).
- Dagua finalizes PivotMDS by centering/scaling into dagua's layout extent and casting to
  `float32` (`postprocess.py:946-999`). OGDF PivotMDS writes raw coordinates into
  `GraphAttributes` with no analogous dagua extent normalization before stress
  (`PivotMDS.cpp:139-148`).

Initialization impact:

- For the default 200-step majorization branch, the stress majorization objective can reduce
  many initialization differences, but it is not guaranteed to eliminate all because only
  200 Gauss-Seidel sweeps are run (`StressMinimization.h:54-66`;
  `pipelines/maxent_stress.py:30-58`).
- The first-pivot mismatch is probably a major cause of residual differences on non-path
  graphs. OGDF first pivot is deterministic graph-order node 0 (`PivotMDS.cpp:264`), while
  dagua first pivot is seeded torch random (`distance.py:956-964`).
- On path graphs, OGDF initializes exactly on a line (`PivotMDS.cpp:114-118`,
  `PivotMDS.cpp:152-179`), while dagua runs generic PivotMDS and may rely on later
  normalization/majorization to converge (`pipelines/pivot_mds.py:51-64`,
  `embed.py:277-309`).

## 6. Iteration / convergence

OGDF defaults:

- Constructor default iterations: `m_numberOfIterations(200)` (`StressMinimization.h:54-66`).
- `setIterations()` uses the supplied positive value, but falls back to `100` for
  non-positive input, not the constructor's `200` (`StressMinimization.h:222-224`).
  The runner never calls `setIterations`, so the default remains 200
  (`scripts/ogdf_runner.cpp:159-162`).
- Default termination criterion is `None` (`StressMinimization.h:54-66`), so
  `finished()` returns false except for hitting the iteration count
  (`StressMinimization.cpp:305-331`).
- Optional convergence criteria exist: position difference uses
  `sqrt(sum(delta^2)) / sqrt(sum(prev^2)) < EPSILON`; stress convergence uses
  `curStress == 0 || prevStress - curStress < prevStress * EPSILON`
  (`StressMinimization.cpp:312-328`). `EPSILON` is `10e-4` (0.001)
  (`StressMinimization.cpp:50`).

Dagua defaults:

- Public default `steps=200` (`pipelines/maxent_stress.py:145-155`).
- `FixedSteps` stores the requested total step count (`converge.py:129-134`).
- `Repeat` loops exactly `n` times unless `state.converged` becomes true, and increments
  `state.step` after each iteration (`base.py:425-438`). The maxent majorization branch
  has no convergence op, so it runs exactly `steps` sweeps (`pipelines/maxent_stress.py:46-56`).
- Dagua gradient LR is not OGDF-like: initial LR is `min(0.04, 0.8 / max(N,1))`,
  final LR is `max(initial_lr * 0.1, initial_lr / sqrt(total_steps))`, and LR is linearly
  annealed after each Adam step (`maxent_stress.py:454-465`, `maxent_stress.py:646-652`).

Iteration mismatches:

- Default `classic_maxent_stress` vs `ogdf_stress`: iteration count matches at 200
  (`classic_competitor.py:1392-1399`, `StressMinimization.h:54-66`).
- Variants `steps50` and `steps400` do not align with the current OGDF runner because the
  runner exposes no JSON iteration parameter and always calls default `layout.call`
  (`variants.py:1068-1088`, `scripts/ogdf_runner.cpp:138-143`,
  `scripts/ogdf_runner.cpp:159-162`).
- The gradient branch's Adam updates are not comparable to OGDF majorization. This affects
  entropy, alpha2, steps50, steps400, and any default run outside the majorization dispatch
  conditions (`pipelines/maxent_stress.py:131-142`).

## 7. Hyperparameter alignment table

| Parameter / behavior | Dagua default / variant | OGDF default through runner | Match? | Evidence |
|---|---:|---:|---|---|
| Base algorithm for default small graph | Serial stress majorization | Serial stress majorization | Y | `pipelines/maxent_stress.py:131-137`; `StressMinimization.cpp:233-303` |
| Default steps | 200 | 200 | Y | `pipelines/maxent_stress.py:145-155`; `StressMinimization.h:54-66` |
| Step variants | 50 / 400 force gradient branch | Runner still 200 | N | `variants.py:1068-1088`; `scripts/ogdf_runner.cpp:159-162` |
| Entropy term | Optional `use_entropy=True` | None | N | `maxent_stress.py:538-548`; `StressMinimization.cpp:151-169` |
| Entropy alpha | `1.0` or `2.0` variants | None | N | `variants.py:1046-1065`; `StressMinimization.h:49-204` |
| Edge cost scale | Unweighted BFS increment 1 | `m_edgeCosts=100` | Scale-only N | `graph_utils.py:52-72`; `StressMinimization.h:54-66` |
| Weight formula | `d^-2` | `d^-2` | Y | `maxent_stress.py:215-222`; `StressMinimization.cpp:139-149` |
| Disconnected fill | `average_edge_cost * sqrt(n)` | `m_avgEdgeCosts * sqrt(n)` | Y for unweighted scale | `maxent_stress.py:202-212`; `StressMinimization.cpp:94-100` |
| Initial layout | Dagua PivotMDS | OGDF PivotMDS | Partial | `maxent_stress.py:115-128`; `StressMinimization.cpp:107-124` |
| Pivot count for stress init | 50 | 50 | Y | `maxent_stress.py:37-40`, `maxent_stress.py:115-122`; `StressMinimization.cpp:50-52`, `StressMinimization.cpp:107-110` |
| Pivot first node | Torch random | `G.firstNode()` | N | `distance.py:956-964`; `PivotMDS.cpp:264-284` |
| Pivot SVD | `torch.linalg.svd` | custom power iteration with `rand()` seed 0 | N | `embed.py:299-305`; `PivotMDS.cpp:181-236`, `PivotMDS.cpp:337-391` |
| Path init fast path | No observed path special case in dagua PivotMDS | Yes | N | `pipelines/pivot_mds.py:51-64`; `PivotMDS.cpp:114-179` |
| Component init | Generic dagua PivotMDS on graph | OGDF `ComponentSplitterLayout` wrapping PivotMDS | N | `maxent_stress.py:115-122`; `StressMinimization.cpp:113-119` |
| Dense stress pair set | All ordered matrix entries for majorization, effectively all pairs | All ordered matrix entries for updates | Y | `maxent_stress.py:705-725`; `StressMinimization.cpp:244-289` |
| Update order | Node index order | OGDF graph node order | Likely Y through runner | `maxent_stress.py:697-705`; `scripts/ogdf_runner.cpp:208-217`, `StressMinimization.cpp:237-244` |
| Coordinate fixed flags | Not implemented | Available but unset | Y for runner | `maxent_stress.py:697-731`; `StressMinimization.h:78-85`, `scripts/ogdf_runner.cpp:159-162` |
| 3D mode | No | No in runner | Y | `pipelines/maxent_stress.py:145-155`; `scripts/ogdf_runner.cpp:203-207`, `StressMinimization.cpp:54-56` |
| Output dtype | `float32` | C++ double serialized then Python `float32` | Partial | `maxent_stress.py:775-781`; `ogdf_competitor.py:162-171` |
| Output normalization | Dagua normalizes internally | OGDF runner emits raw coords; benchmark aligns later | N at adapter output | `maxent_stress.py:772-781`; `scripts/ogdf_runner.cpp:232-240` |
| Runner coordinate precision | Full `float32` tensor output after dagua | Default C++ stream precision before Python parse | N | `ogdf_competitor.py:165`; `scripts/ogdf_runner.cpp:232-240` |
| RNG seed from benchmark | Dagua consumes supplied seed | OGDF adapter ignores seed | N | `classic_competitor.py:1392-1399`; `ogdf_competitor.py:179-204` |
| Runner fixed seed | Dagua default seed 42 if wrapper passes none | OGDF global seed and `srand(42)` before layout | Partial | `pipelines/maxent_stress.py:145-155`; `scripts/ogdf_runner.cpp:219-228` |
| Empty graph | Empty `[0,2]` tensor | Empty `[0,2]` tensor | Y | `pipelines/maxent_stress.py:206-210`; `ogdf_competitor.py:131-132` |
| Single node | Zero coordinate | Zero coordinate in OGDF algorithm | Y | `pipelines/maxent_stress.py:206-210`; `StressMinimization.cpp:57-68` |
| Self-loops | Ignored in adjacency builder | Graph accepts self-loop; shortest paths unaffected in practical BFS | Partial | `graph_utils.py:42-49`; `scripts/ogdf_runner.cpp:214-217` |
| Multi-edges | Dagua keeps minimum weight / simple adjacency | OGDF graph has parallel edges; BFS effectively duplicate-insensitive | Partial | `graph_utils.py:42-49`; `scripts/ogdf_runner.cpp:214-217`, `PivotMDS.cpp:164-167` |
| Weighted edges from benchmark graph | Dagua can accept `edge_weights`; wrapper does not pass weights | OGDF runner payload has no weights | N for weighted graph semantics | `classic_competitor.py:1392-1399`; `ogdf_competitor.py:138-143` |

## 8. Edge cases

Self-loops:

- Dagua's shared min-weight undirected adjacency skips self-loops (`graph_utils.py:42-44`).
  The maxent majorization branch uses this helper through `_shared_build_undirected_adjacency`
  (`graph_utils.py:125-129`, `maxent_stress.py:186-190`).
- OGDF runner creates every edge directly, including self-loops if present
  (`scripts/ogdf_runner.cpp:214-217`). OGDF PivotMDS path layout explicitly ignores self-loops
  while traversing a path (`PivotMDS.cpp:164-167`). For stress APSP, self-loops should not
  change shortest paths because diagonal distances are initialized to zero
  (`StressMinimization.cpp:334-341`), but they can affect path detection or component
  splitter internals before stress.

Multi-edges:

- Dagua collapses duplicate undirected edges by minimum weight in
  `_build_min_weight_undirected_adjacency` (`graph_utils.py:26-49`).
- OGDF runner creates parallel graph edges (`scripts/ogdf_runner.cpp:214-217`). For
  unweighted BFS shortest paths, parallel edges do not change distances, but OGDF's
  `PivotMDS::doPathLayout` explicitly ignores multi-edges in path traversal
  (`PivotMDS.cpp:164-167`), and `getRootedPath()` calls `makeSimpleUndirected`
  before degree tests (`PivotMDS.cpp:296-317`). Dagua has no path-layout equivalent,
  so multi-edge paths can initialize differently.

Disconnected components:

- Dagua majorization fills disconnected distances with `average_edge_cost * sqrt(n)` and
  solves one global layout (`maxent_stress.py:202-212`).
- OGDF stress also fills infinite distances with `m_avgEdgeCosts * sqrt(n)` when not in
  component layout mode (`StressMinimization.cpp:94-100`).
- However, OGDF initialization uses `ComponentSplitterLayout` around PivotMDS when
  `m_componentLayout` is false (`StressMinimization.cpp:113-119`). Dagua initialization
  calls its PivotMDS pipeline directly (`maxent_stress.py:115-122`). That makes disconnected
  initial placement a known divergence even when later stress distances are aligned.
- For large gradient-branch dagua runs, pivot selection intentionally seeds one node per
  component before random fill (`maxent_stress.py:278-352`), which is a dagua-specific
  approximation not present in OGDF stress.

Weighted edges:

- Dagua's core pipeline supports `edge_weights`, uses Dijkstra for weighted APSP, and uses
  the average edge cost for disconnected fills (`maxent_stress.py:178-223`,
  `graph_utils.py:75-115`).
- OGDF `StressMinimization` supports edge-cost attributes and Dijkstra
  (`StressMinimization.cpp:74-83`, `StressMinimization.h:102-103`), but the runner's JSON
  payload contains only `"nodes"`, `"edges"`, and `"algorithm"` (`ogdf_competitor.py:138-143`),
  and `scripts/ogdf_runner.cpp` never parses or sets edge weights (`scripts/ogdf_runner.cpp:80-130`,
  `scripts/ogdf_runner.cpp:203-217`).
- The `classic_maxent_stress` wrapper also does not pass `graph.edge_weights` into
  `layout_maxent_stress` (`classic_competitor.py:1392-1399`), so benchmark weighted graphs
  in this pairing may be effectively unweighted on both sides despite the core dagua API
  supporting weights.

Empty graph / singleton:

- Dagua returns empty or zero tensors before building a pipeline
  (`pipelines/maxent_stress.py:206-210`).
- The OGDF adapter returns an empty tensor before invoking the runner for `N=0`
  (`ogdf_competitor.py:131-132`), and OGDF `StressMinimization` zeros singleton coordinates
  (`StressMinimization.cpp:57-68`).

## 9. Numerical precision

Dagua:

- Majorization warm-start positions are cast to CPU `float64`
  (`maxent_stress.py:123-125`).
- Majorization graph distances are initially created as `torch.float32`, then converted to
  `float64` before weight-matrix construction (`maxent_stress.py:212-222`). This means
  integer BFS distances survive exactly, but weighted Dijkstra distances would already have
  been rounded through float32 before double majorization.
- The majorization loop extracts Python floats from tensor values (`.item()`), computes with
  Python `float` / `math.hypot` (C double), and writes back into a `float64` tensor
  (`maxent_stress.py:697-731`).
- Final positions are normalized and cast to `float32` (`maxent_stress.py:772-781`).
- Gradient branch operates in `float32` positions (`maxent_stress.py:126-128`) and uses
  PyTorch reductions whose summation order differs from the serial OGDF loops
  (`maxent_stress.py:523-641`).

OGDF:

- Coordinates and matrices are `double` throughout `StressMinimization`
  (`StressMinimization.cpp:71-83`, `StressMinimization.cpp:151-169`,
  `StressMinimization.cpp:233-303`).
- PivotMDS also uses `double` arrays and custom double power iteration
  (`PivotMDS.cpp:60-90`, `PivotMDS.cpp:181-236`, `PivotMDS.cpp:360-391`).
- The runner serializes coordinates using default `std::cout` formatting, with no
  `std::setprecision`, at `scripts/ogdf_runner.cpp:232-240`. Default iostream precision is
  6 significant digits, so reference coordinates are quantized before Python reads them.
- The Python adapter converts parsed positions to `torch.float32` (`ogdf_competitor.py:162-171`).

Precision impact ranking:

1. Runner serialization precision can introduce visible residual RMSD after alignment,
   especially on large-coordinate OGDF outputs (`scripts/ogdf_runner.cpp:232-240`).
2. Dagua's final normalization before adapter return differs from raw OGDF output
   (`maxent_stress.py:772-781`; `scripts/ogdf_runner.cpp:232-240`). The fidelity pipeline
   likely performs its own Procrustes/scale alignment, but adapter output differences still
   affect any direct comparisons.
3. Dagua's majorization state rounds graph distances through float32 before float64
   (`maxent_stress.py:212-214`). Unweighted paths are safe; weighted paths are not.
4. PivotMDS SVD differs: torch LAPACK SVD versus OGDF randomized power iteration
   (`embed.py:299-305`; `PivotMDS.cpp:181-236`, `PivotMDS.cpp:360-391`).

## 10. RNG semantics

The short answer is no: dagua's torch seed does not produce the same sequence as the
reference RNG.

- Dagua PivotMDS first pivot uses `torch.randint` from a CPU `torch.Generator` seeded by
  `problem.seed` (`distance.py:855-874`, `distance.py:956-964`).
- OGDF PivotMDS max-min pivot selection starts from `G.firstNode()` and is not random
  (`PivotMDS.cpp:264-284`).
- OGDF PivotMDS's custom power iteration randomizes eigenvectors with C `rand()` after
  `srand(SEED)`, where `SEED=0` (`PivotMDS.h:101-110`, `PivotMDS.cpp:337-343`).
- The runner sets OGDF global seed and C `srand(42)` before running layout and also fills
  initial graph attributes with `std::rand() % 1000 / 10.0`
  (`scripts/ogdf_runner.cpp:219-228`). For `StressMinimization`, those runner-initialized
  coordinates are overwritten by PivotMDS because `m_hasInitialLayout` defaults false
  (`StressMinimization.h:54-66`, `StressMinimization.cpp:87-92`), but the RNG reset may
  still affect algorithms that use OGDF's global random state.
- The OGDF Python adapter explicitly ignores the competitor `seed` argument
  (`ogdf_competitor.py:179-204`).
- The variant registry marks `classic_maxent_stress` stochastic and `ogdf_stress`
  non-stochastic (`variants.py:1820-1865`). This is accurate for the harness but means
  seed-to-seed dagua variation has no one-to-one OGDF counterpart.

Residual RNG conclusion:

- For strict reference matching, dagua's PivotMDS warm start should not use a torch-random
  first pivot when emulating OGDF stress. It should use node 0 / first node, and if power
  iteration is emulated, it should use C `rand()` seeded with 0 or avoid random SVD by
  matching OGDF output another way (`distance.py:956-964`; `PivotMDS.cpp:264-284`,
  `PivotMDS.cpp:337-343`).

## 11. Edge-case bugs and suspicious divergences

1. OGDF runner quantizes reference coordinates.
   `scripts/ogdf_runner.cpp:232-240` prints raw doubles without `std::setprecision(17)`;
   `ogdf_competitor.py:157-171` parses those truncated coordinates into `float32`.
   This is not an algorithm bug, but it is a fidelity harness bug for sub-percent work.

2. Dagua default branch is OGDF-like only for `steps == 200`.
   `build_maxent_stress_pipeline()` sends `steps=50` and `steps=400` to the Adam gradient
   branch because of the `steps == 200` condition (`pipelines/maxent_stress.py:131-142`).
   OGDF exposes `setIterations()` and would remain a majorization algorithm for 50 or 400
   iterations (`StressMinimization.h:222-224`). This is a likely wrong-family comparison
   for the step variants.

3. Dagua entropy and alpha variants are compared to OGDF stress, which has no entropy.
   Variants declare `use_entropy=True` and `alpha=1.0/2.0` while `original_engine` is
   still `ogdf_stress` with empty params (`variants.py:1046-1065`). OGDF objective has
   only stress (`StressMinimization.cpp:151-169`). This is expected if the benchmark
   treats OGDF stress as a proxy, but not an exact reference pairing.

4. Dagua PivotMDS first pivot does not match OGDF.
   Dagua uses seeded `torch.randint` (`distance.py:956-964`), while OGDF starts from
   first graph node (`PivotMDS.cpp:264`). This creates deterministic but different warm
   starts and can persist after finite majorization sweeps.

5. Dagua lacks OGDF PivotMDS path fast path.
   OGDF detects simple paths and lays them out exactly on a line (`PivotMDS.cpp:114-179`,
   `PivotMDS.cpp:296-317`). Dagua generic PivotMDS goes through max-min pivots and SVD
   (`pipelines/pivot_mds.py:51-64`, `embed.py:277-309`). Path-like graphs can therefore
   differ despite equivalent stress updates.

6. Component initialization differs.
   OGDF uses `ComponentSplitterLayout` around PivotMDS for disconnected graphs
   (`StressMinimization.cpp:113-119`). Dagua's init directly calls PivotMDS on the full
   graph (`maxent_stress.py:115-122`). Later disconnected distance fill matches, but
   finite-step convergence may retain component placement differences.

7. Weighted support is dropped by both benchmark wrappers.
   Dagua core accepts `edge_weights` (`pipelines/maxent_stress.py:145-155`), but
   `ClassicMaxentStress.layout()` does not pass weights (`classic_competitor.py:1392-1399`).
   OGDF core supports edge-cost attributes (`StressMinimization.cpp:74-83`), but the runner
   payload has no weights (`ogdf_competitor.py:138-143`; `scripts/ogdf_runner.cpp:80-130`).
   Weighted fidelity cases are therefore not testing weighted stress semantics.

8. Dagua majorization distance tensor has a float32 round-trip.
   `torch.tensor(cleaned, dtype=torch.float32)` is immediately converted to float64
   (`maxent_stress.py:212-214`). This is harmless for unweighted integer distances but
   avoidable divergence for weighted Dijkstra.

9. Dagua final normalization is internal to the algorithm; OGDF returns raw coordinates.
   Dagua calls `normalize_positions` after stress (`maxent_stress.py:772-781`), while the
   runner prints raw `GraphAttributes` (`scripts/ogdf_runner.cpp:232-240`). If downstream
   metrics do not fully normalize away translation, scale, and reflection, this is a direct
   adapter mismatch.

10. The specialized classic wrapper ignores variant parameters.
    The class `ClassicMaxentStress.layout()` hard-codes `steps=200`, `alpha=1.0`, and does
    not expose `use_entropy` (`classic_competitor.py:1392-1399`). The registry defines
    maxent variants with step and entropy params (`variants.py:1035-1088`). If the benchmark
    uses the generic `_ClassicLayoutSpec` route for variants, this may be fine; if it uses
    the registered class directly, all variants collapse to the default.

## 12. Ranked fix list

1. Add full-precision coordinate serialization in the OGDF runner.
   - Evidence: default `std::cout` coordinate output at `scripts/ogdf_runner.cpp:232-240`;
     Python parse to `float32` at `ogdf_competitor.py:162-171`.
   - Proposed fix: include `<iomanip>` and set `std::setprecision(17)` before printing
     positions.
   - Expected RMSD impact: high for residual sub-percent discrepancies; removes harness
     quantization.
   - Size estimate: XS, 3-5 lines.

2. Expose OGDF stress iteration count through JSON and variant `original_params`.
   - Evidence: runner calls `StressMinimization layout; layout.call(...)` with no
     `setIterations()` (`scripts/ogdf_runner.cpp:159-162`); variants steps50/steps400 map
     to `ogdf_stress` with `{}` params (`variants.py:1068-1088`); OGDF supports
     `setIterations()` (`StressMinimization.h:222-224`).
   - Proposed fix: parse optional `"iterations"` in `scripts/ogdf_runner.cpp`, call
     `layout.setIterations(iterations)` for stress, and set original params for step
     variants.
   - Expected RMSD impact: high for steps50/steps400; converts wrong-parameter reference
     to aligned reference.
   - Size estimate: S/M, 30-60 lines including adapter payload.

3. Keep step variants on the dagua majorization branch when `use_entropy=False`.
   - Evidence: branch condition requires `steps == 200` (`pipelines/maxent_stress.py:131-137`);
     OGDF majorization supports any positive iteration count (`StressMinimization.h:222-224`).
   - Proposed fix: remove `and steps == 200` from majorization dispatch for
     non-entropy graphs under the node limit.
   - Expected RMSD impact: high for `classic_maxent_stress_steps50` and
     `classic_maxent_stress_steps400`; low/no effect for default and entropy variants.
   - Size estimate: XS, 1-3 lines plus regression tests.

4. Add an OGDF-compatible PivotMDS initialization mode for maxent stress.
   - Evidence: dagua uses torch-random first pivot (`distance.py:956-964`); OGDF starts at
     first node (`PivotMDS.cpp:264-284`); dagua uses `torch.linalg.svd`
     (`embed.py:299-305`), OGDF uses custom power iteration (`PivotMDS.cpp:181-236`,
     `PivotMDS.cpp:360-391`).
   - Proposed fix: for `MaxentInitializePositions(for_majorization=True)`, provide a
     PivotMDS mode with first pivot 0, OGDF centering, and either a deterministic sign/order
     convention validated against OGDF or direct power-iteration parity.
   - Expected RMSD impact: medium/high on graphs where 200 sweeps do not erase warm-start
     differences.
   - Size estimate: M/L depending on how exactly the OGDF SVD is matched.

5. Add OGDF path fast path to the PivotMDS warm start.
   - Evidence: OGDF path detection and line layout at `PivotMDS.cpp:114-179` and
     `PivotMDS.cpp:296-317`; dagua generic PivotMDS pipeline at
     `pipelines/pivot_mds.py:51-64`.
   - Proposed fix: detect simple undirected paths before generic PivotMDS and return line
     coordinates with uniform edge cost scale for the OGDF-compatible mode.
   - Expected RMSD impact: medium on chains, trees with path components, and sparse path-like
     cases; probably low on dense graphs.
   - Size estimate: S/M, 40-80 lines plus tests.

6. Align disconnected-component initialization.
   - Evidence: OGDF wraps PivotMDS with `ComponentSplitterLayout` for disconnected graphs
     (`StressMinimization.cpp:113-119`); dagua initializes via one full-graph PivotMDS call
     (`maxent_stress.py:115-122`).
   - Proposed fix: add component-wise PivotMDS initialization and deterministic packing in
     the OGDF-compatible maxent branch, or explicitly choose the full-distance fill model
     and document that component init remains a proxy.
   - Expected RMSD impact: medium on multi-component graphs; low on connected graphs.
   - Size estimate: M/L due to packing parity risk.

7. Remove the float32 round-trip for majorization distances.
   - Evidence: `cleaned` is converted to `torch.float32` then `float64`
     (`maxent_stress.py:212-214`).
   - Proposed fix: create the tensor directly as `torch.float64` in the majorization branch.
   - Expected RMSD impact: low for current unweighted runner; medium if weighted edges are
     ever passed through.
   - Size estimate: XS, 1 line plus test for weighted precision.

8. Pass weights through benchmark wrappers or mark weighted cases explicitly unweighted.
   - Evidence: dagua core supports `edge_weights` (`pipelines/maxent_stress.py:145-155`),
     but the classic wrapper omits it (`classic_competitor.py:1392-1399`); OGDF runner
     payload omits weights (`ogdf_competitor.py:138-143`, `scripts/ogdf_runner.cpp:80-130`).
   - Proposed fix: add optional weights to both payloads and set OGDF edge double weights,
     or split weighted fidelity out as unsupported for this OGDF runner.
   - Expected RMSD impact: high on weighted graphs if weighted semantics are intended;
     none on unweighted graphs.
   - Size estimate: M, 50-100 lines across adapters and runner.

9. Decide whether entropy variants should keep using `ogdf_stress` as an "original".
   - Evidence: entropy variants map to `ogdf_stress` with empty original params
     (`variants.py:1046-1065`); OGDF has no entropy term (`StressMinimization.cpp:151-169`).
   - Proposed fix: mark `is_true_original=False` is already set (`variants.py:1039-1064`);
     consider excluding entropy variants from exact diff rounds or adding a separate
     reference implementation of Gansner-Hu-North maxent-stress if available.
   - Expected RMSD impact: improves interpretation rather than output.
   - Size estimate: XS for metadata/reporting, L for a true external reference.

10. Make the specialized `ClassicMaxentStress` wrapper honor params or ensure variants never
    use it directly.
    - Evidence: wrapper hard-codes `steps=200`, `alpha=1.0` and omits `use_entropy`
      (`classic_competitor.py:1392-1399`); registry variants contain params
      (`variants.py:1035-1088`).
    - Proposed fix: either route variants exclusively through `_ClassicLayoutSpec` with
      merged params, or extend the class wrapper to accept params from the benchmark harness.
    - Expected RMSD impact: high if variants currently collapse; none if generic route is
      already used.
    - Size estimate: S/M after checking benchmark dispatch.

## 13. Recommended Round 22+ fix scope

Recommended bundle for one follow-up implementation round:

1. Fix the OGDF runner precision first.
   This is the cleanest harness correction and should reduce residual noise across all OGDF
   families, not just maxent-stress (`scripts/ogdf_runner.cpp:232-240`).

2. Keep non-entropy dagua step variants on the majorization branch.
   Remove the `steps == 200` dispatch guard so `steps50` and `steps400` are still
   stress-majorization variants (`pipelines/maxent_stress.py:131-142`). Pair this with
   targeted tests proving the majorization branch is selected for `use_entropy=False`.

3. Expose stress iterations in the OGDF runner and wire `original_params`.
   Add JSON parsing for `"iterations"`, call `StressMinimization::setIterations`, and set
   original params for `classic_maxent_stress_steps50` and
   `classic_maxent_stress_steps400` (`StressMinimization.h:222-224`,
   `variants.py:1068-1088`, `scripts/ogdf_runner.cpp:159-162`).

4. Make majorization distances stay float64.
   Change the majorization branch tensor creation from float32-to-float64 to direct float64
   (`maxent_stress.py:212-214`). This is low risk and removes one avoidable numeric mismatch.

5. Defer full PivotMDS parity to a later, larger round.
   Matching OGDF's first pivot, path fast path, component splitter, and power-iteration SVD
   is likely the next meaningful algorithmic improvement, but it is broader than a small
   harness cleanup (`distance.py:956-964`; `PivotMDS.cpp:114-179`,
   `PivotMDS.cpp:264-284`, `PivotMDS.cpp:337-391`;
   `StressMinimization.cpp:113-119`).

Expected outcome of that bundle:

- Default `classic_maxent_stress_default` should remain strong-equivalent with lower
  residual RMSD from precision cleanup.
- `steps50` and `steps400` should become same-family majorization comparisons instead of
  dagua Adam-gradient vs OGDF fixed-200 majorization.
- Entropy and alpha variants should remain proxy comparisons unless a true maxent-stress
  reference with entropy is added.
