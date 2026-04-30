# Round 21 Diff: `classic_pivot_mds` vs `ogdf_pivot_mds`

Diagnosis-only adversarial diff for the Pivot-MDS family: dagua `classic_pivot_mds`
against reference `ogdf_pivot_mds`.

Current mega-run verdict is `strong_equivalent` for all tested Pivot-MDS variants, but
the implementations are not algorithmically identical. The current equivalence is best
read as "same family and similar normalized geometry under Procrustes," not bitwise or
parameter-equivalent behavior.

## 1. Files read

Dagua implementation and wiring:

- `dagua/layout/ops/pipelines/pivot_mds.py:1-137` -- active pipeline entry point.
- `dagua/layout/ops/distance.py:296-329` -- pivot shortest-path row computation and disconnected fill.
- `dagua/layout/ops/distance.py:838-1044` -- `PivotSelectionConfig`, RNG resolution, max-min pivot selection, and pivot distance queries.
- `dagua/layout/ops/embed.py:277-309` -- active rectangular Pivot-MDS centering and SVD coordinate recovery.
- `dagua/layout/ops/embed.py:1390-1475` -- generic SVD op and `PivotMDSComputeCoordinates` op.
- `dagua/layout/ops/preprocess.py:189-227` -- edge weight resolution.
- `dagua/layout/ops/preprocess.py:342-403` -- list adjacency construction, self-loop handling, duplicate edge policy.
- `dagua/layout/ops/preprocess.py:759-896` -- `BuildAdjacencyConfig` and `BuildAdjacency` application.
- `dagua/layout/ops/postprocess.py:870-890` -- classical/pivot position normalization helper.
- `dagua/layout/ops/postprocess.py:945-999` -- `PivotMDSFinalizePositions`.
- `dagua/layout/classic/pivot_mds.py:1-311` -- legacy/classic implementation retained under `dagua.layout.classic`.
- `dagua/layout/_archive/classic/pivot_mds.py:1-311` -- archived source with same classic algorithm and comments.
- `dagua/eval/competitors/classic_competitor.py:194-198` -- classic competitor spec defaulting `n_pivots=50`.
- `dagua/eval/competitors/classic_competitor.py:1094-1148` -- `ClassicPivotMDS` adapter.
- `dagua/eval/competitors/ogdf_competitor.py:1-280` -- OGDF subprocess adapter and connectivity guard.
- `dagua/eval/variants.py:1177-1220` -- Pivot-MDS variant definitions for 10/50/100/200 pivots.
- `dagua/eval/variants.py:1820-1859` -- stochasticity registry marks classic pivot stochastic and OGDF pivot non-stochastic.
- `scripts/ogdf_runner.cpp:1-246` -- standalone runner used by `ogdf_pivot_mds`.
- `eval_output/fidelity_report/report.md:66-69` -- current Pivot-MDS verdict rows.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:192-198` -- sprint context noting OGDF seed adapter issue.

Reference implementation:

- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/PivotMDS.h:1-168` -- OGDF declarations and defaults.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:1-393` -- OGDF implementation.

Related search results that were inspected for routing/context but are not the active
single-family implementation:

- `dagua/layout/ops/init.py` -- contains a `PivotMDSInit` warm-start op for other pipelines, but
  `classic_pivot_mds` uses `dagua/layout/ops/pipelines/pivot_mds.py`.
- `dagua/layout/ops/maxent_stress.py`, `dagua/layout/ops/loss_classic.py`,
  `dagua/layout/ops/stress_sgd.py` -- consume pivot distances for other algorithms but are not
  the classic Pivot-MDS competitor path.

## 2. Overall pipeline structure

Dagua active path:

1. `ClassicPivotMDS.layout` ignores timeout, imports `layout_pivot_mds_pipeline`, and calls it
   with `graph.edge_index`, `graph.num_nodes`, `graph.node_sizes`, hard-coded `n_pivots=50`, and
   `seed=self._layout_seed(seed)` (`dagua/eval/competitors/classic_competitor.py:1124-1138`).
2. `layout_pivot_mds_pipeline` validates `num_nodes`, `n_pivots`, and optional `edge_weights`,
   builds a `LayoutProblem` with `seed`, and runs on a CPU `ExecutionPlan`
   (`dagua/layout/ops/pipelines/pivot_mds.py:69-131`).
3. `build_pivot_mds_pipeline` wires:
   `BuildAdjacency(weighted=..., dedup="min", format="list")`,
   `PivotSelection(n_pivots=...)`,
   `PivotDistanceQueries`,
   `PivotMDSComputeCoordinates`,
   `PivotMDSFinalizePositions`
   (`dagua/layout/ops/pipelines/pivot_mds.py:51-65`).
4. `BuildAdjacency` defaults to undirected, optional weighted, min-deduplicated list adjacency
   (`dagua/layout/ops/preprocess.py:759-786`, `dagua/layout/ops/pipelines/pivot_mds.py:53-58`).
5. `PivotSelection` chooses a random first pivot with torch, then greedy max-min pivots
   (`dagua/layout/ops/distance.py:955-979`).
6. `PivotDistanceQueries` recomputes shortest-path rows for selected pivots
   (`dagua/layout/ops/distance.py:1008-1044`).
7. `PivotMDSComputeCoordinates` calls `_pivot_mds_coordinates` (`dagua/layout/ops/embed.py:1437-1475`).
8. `PivotMDSFinalizePositions` recenters, rescales, and casts output to `float32`
   (`dagua/layout/ops/postprocess.py:945-999`).

OGDF reference path as actually benchmarked:

1. `OGDFPivotMDS` sets `name="ogdf_pivot_mds"` and `algorithm="pivot_mds"`
   (`dagua/eval/competitors/ogdf_competitor.py:273-279`).
2. `_OGDFBase.layout` ignores the incoming Python seed and rejects disconnected Pivot-MDS graphs
   before subprocess execution (`dagua/eval/competitors/ogdf_competitor.py:179-215`).
3. `_run_ogdf` serializes only node count, edge pairs, and algorithm name to JSON. It sends no
   pivot count, no seed, no weights, and no node sizes (`dagua/eval/competitors/ogdf_competitor.py:138-144`).
4. `scripts/ogdf_runner.cpp` builds an OGDF graph, seeds OGDF and C RNG with `42`, initializes
   graph attribute coordinates with `rand()%1000/10`, then calls `ogdf::PivotMDS`
   (`scripts/ogdf_runner.cpp:203-230`).
5. For `"pivot_mds"`, the runner constructs a default `ogdf::PivotMDS layout;` and calls
   `layout.call(graphAttributes)` (`scripts/ogdf_runner.cpp:164-167`).
6. OGDF `PivotMDS::call` asserts connectedness and calls `pivotMDSLayout`
   (`/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:54-58`).
7. `pivotMDSLayout` handles `n=0`, `n=1`, path graphs, and general graphs separately
   (`/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:92-150`).
8. General graphs run `getPivotDistanceMatrix`, `centerPivotmatrix`,
   `singularValueDecomposition`, multiply coordinates by `sqrt(eVals)`, then write graph
   attributes (`/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:119-148`).

Pipeline-level deltas:

- Dagua always uses the general rectangular MDS path for `N >= 2`; OGDF special-cases paths into a
  straight-line layout (`PivotMDS.cpp:114-118`, `PivotMDS.cpp:152-179`).
- Dagua chooses a random first pivot; OGDF starts with `G.firstNode()` and then greedy max-min
  (`PivotMDS.cpp:260-284`).
- Dagua default benchmark pivot count is 50 (`classic_competitor.py:194-198`,
  `pivot_mds.py:73-75`); OGDF default object pivot count is 250
  (`PivotMDS.h:59-64`).
- Dagua normalizes final output to an extent derived from `sqrt(N)` and node sizes
  (`postprocess.py:870-890`, `postprocess.py:993-998`); OGDF emits raw coordinates from its
  algorithm (`PivotMDS.cpp:140-148`) and the runner prints them without normalization
  (`scripts/ogdf_runner.cpp:232-240`).

## 3. Energy / loss / objective

Pivot-MDS is not run as an iterative energy minimizer in either implementation. There is no
gradient descent loss loop. The objective is implicit: approximate classical MDS from a
pivot-to-node shortest-path distance matrix.

Dagua rectangular centered matrix:

- Input is `distance_matrix` with shape `[P, N]` (`dagua/layout/ops/embed.py:277-288`).
- Formula in code:
  - `squared = D^2` (`embed.py:293`).
  - `row_means = mean_j squared[p, j]` (`embed.py:294`).
  - `col_means = mean_p squared[p, j]` (`embed.py:295`).
  - `grand_mean = mean_{p,j} squared[p,j]` (`embed.py:296`).
  - `centered[p,j] = -0.5 * (squared[p,j] - row_means[p] - col_means[j] + grand_mean)`
    (`embed.py:297`).
- Dagua computes `torch.linalg.svd(centered, full_matrices=False)` and uses right singular vectors:
  `coordinates = V[:, 0:k] * singular_values[0:k]`, implemented as
  `vh[:coord_dims].T * scales` (`embed.py:299-305`).

OGDF rectangular centered matrix:

- Input is `pivDistMatrix[p][node]` from shortest paths (`PivotMDS.cpp:238-285`).
- `centerPivotmatrix` computes:
  - For each pivot row `p`, `colNormalization[p] = sum_j D[p,j]^2 / N`
    (`PivotMDS.cpp:69-76`).
  - `normalizationFactor = sum_{p,j} D[p,j]^2 / (N * P)` (`PivotMDS.cpp:74-77`).
  - For each node `j`, `rowColNormalizer = sum_p D[p,j]^2 / P`
    (`PivotMDS.cpp:78-85`).
  - `pivotMatrix[p][j] = -0.5 * (D[p,j]^2 + grand - rowMean[p] - colMean[j])`
    (`PivotMDS.cpp:81-88`).
- Algebraically this matches dagua's rectangular double-centering formula in
  `embed.py:293-297`.

OGDF SVD/objective realization:

- OGDF computes `K = C C^T` by `selfProduct(pivDistMatrix, K)`
  (`PivotMDS.cpp:360-370`, `PivotMDS.cpp:346-357`).
- It runs power-iteration eigen decomposition on `K` (`PivotMDS.cpp:376`,
  `PivotMDS.cpp:181-235`).
- It computes `C^T x` into node-coordinate vectors (`PivotMDS.cpp:378-386`), normalizes each
  coordinate vector (`PivotMDS.cpp:388-390`), returns eigen-like values through `eVals`,
  and then `pivotMDSLayout` multiplies each coordinate row by `sqrt(eVals[i])`
  (`PivotMDS.cpp:131-137`).
- Because `eigenValueDecomposition` returns norms from multiplying by `K` (`PivotMDS.cpp:223-225`),
  `singularValueDecomposition` first `sqrt`s those values (`PivotMDS.cpp:379-381`) and the caller
  `sqrt`s again (`PivotMDS.cpp:133-136`). Net scale is intended to become singular-value scale,
  but it is mediated by iterative approximate eigenvalues and two normalization steps.

Per-term comparison:

| Term / operation | Dagua | OGDF | Match |
| --- | --- | --- | --- |
| Shortest path distance matrix | Pivot rows from BFS/Dijkstra, then fill disconnected nodes with `max+1` (`distance.py:318-328`) | Pivot rows from OGDF `bfs_SPSS`/`dijkstra_SPSS`; no disconnected fill because connectedness is asserted (`PivotMDS.cpp:54-57`, `PivotMDS.cpp:268-274`) | Partial |
| Squared-distance matrix | `distance_matrix.square()` (`embed.py:293`) | `pivotMatrix[i][j] * pivotMatrix[i][j]` (`PivotMDS.cpp:71-83`) | Yes |
| Row mean over nodes | `squared.mean(dim=1)` (`embed.py:294`) | `colNormalization[i] = rowSum / nodeCount` (`PivotMDS.cpp:69-76`) | Yes |
| Column mean over pivots | `squared.mean(dim=0)` (`embed.py:295`) | `rowColNormalizer /= numberOfPivots` (`PivotMDS.cpp:78-87`) | Yes |
| Grand mean | `squared.mean()` (`embed.py:296`) | `normalizationFactor / (nodeCount * numberOfPivots)` (`PivotMDS.cpp:65-77`) | Yes |
| Centering sign/factor | `-0.5 * (...)` (`embed.py:297`) | `FACTOR=-0.5`; applied at `PivotMDS.cpp:87` | Yes |
| Eigen/SVD coordinates | LAPACK-backed `torch.linalg.svd` (`embed.py:299`) | Power iteration on `C C^T`, then `C^T x`, normalize, scale (`PivotMDS.cpp:360-390`, `PivotMDS.cpp:131-137`) | Family-equivalent, not numerically identical |
| Final objective/energy iterations | None | None | Yes |

## 4. Force / gradient computation

There is no force or gradient computation in the active Pivot-MDS family.

- Dagua pipeline consists only of preprocessing, distance queries, SVD embedding, and postprocess
  (`dagua/layout/ops/pipelines/pivot_mds.py:51-65`).
- `PivotMDSComputeCoordinates` writes `state.pos` directly from SVD output
  (`dagua/layout/ops/embed.py:1469-1475`).
- OGDF `pivotMDSLayout` writes coordinates after SVD-like decomposition and scaling
  (`/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/PivotMDS.cpp:131-148`).
- No learning rate, force accumulator, edge attraction, repulsion, or gradient term exists in
  either Pivot-MDS path.

## 5. Initialization

Dagua initialization:

- For `num_nodes == 0`, `layout_pivot_mds_pipeline` still runs the pipeline; `PivotSelection`
  writes empty pivots (`distance.py:952-954`), `PivotDistanceQueries` writes empty distances
  (`distance.py:1033-1036`), `_pivot_mds_coordinates` returns zero/empty coordinates
  (`embed.py:290-291`), and finalization returns an empty `[0, 2]` tensor
  (`postprocess.py:989-991`).
- For `num_nodes >= 1`, first pivot is random:
  `torch.randint(0, num_nodes, (1,), generator=generator)` (`distance.py:960-963`).
- Generator source is `ctx.generator` if present; otherwise a private CPU `torch.Generator` seeded
  from `problem.seed` (`distance.py:855-874`).
- Pipeline default seed is `42` (`dagua/layout/ops/pipelines/pivot_mds.py:73-75`).
- Classic competitor uses `self._layout_seed(seed)` and the docstring says `None` preserves
  historical default `42` (`classic_competitor.py:1115-1117`, `classic_competitor.py:1132-1138`).
- After first pivot, Dagua chooses the node with maximum current minimum distance, masking selected
  pivots with `-1.0` (`distance.py:966-979`).

OGDF initialization:

- `PivotMDS` default constructor sets `m_numberOfPivots=250`, `m_dimensionCount=2`,
  `m_edgeCosts=100`, no edge-cost attribute, no forced 2D flag
  (`PivotMDS.h:59-64`).
- `pivotMDSLayout` sets dimension count from graph attributes: 3D if `GraphAttributes::threeD`
  exists and not forcing 2D; otherwise 2D (`PivotMDS.cpp:92-96`). The runner only constructs
  node/edge graphics attributes, not `threeD`, so the benchmark path is 2D
  (`scripts/ogdf_runner.cpp:203-206`).
- `getPivotDistanceMatrix` initializes current pivot to `G.firstNode()`
  (`PivotMDS.cpp:260-265`), not random.
- The greedy update sets the current pivot's `minDistances` to zero, updates min distances for
  all nodes, and picks the node whose `minDistances[v]` exceeds current pivot's score
  (`PivotMDS.cpp:275-283`).
- The runner seeds OGDF global RNG and C `srand(42)` before calling the layout
  (`scripts/ogdf_runner.cpp:219-222`), but OGDF Pivot-MDS pivot selection itself does not consume
  that RNG. The only RNG inside `PivotMDS` is in eigenvector initialization:
  `randomize(eVecs)` calls `srand(SEED)` with `SEED=0`, then C `rand()` (`PivotMDS.h:108-109`,
  `PivotMDS.cpp:181-184`, `PivotMDS.cpp:337-343`).

Critical initialization divergence:

- Dagua first pivot is stochastic and seed-dependent (`distance.py:960-963`).
- OGDF first pivot is the first graph node (`PivotMDS.cpp:264`) and is independent of the adapter
  seed. This can change all pivot rows and usually dominates residual RMSD on non-path graphs.

## 6. Iteration / convergence

Dagua:

- No layout iterations and no convergence test.
- Pivot selection loops until `len(pivot_indices) == min(n_pivots, N)` or until all candidates are
  already selected (`distance.py:956-979`).
- SVD is delegated to `torch.linalg.svd` with no exposed iteration tolerance
  (`embed.py:299`).
- No learning-rate schedule exists.

OGDF:

- No layout-force iterations.
- Pivot selection loops exactly `numberOfPivots = min(n, m_numberOfPivots)` times
  (`PivotMDS.cpp:242-265`).
- Eigen decomposition uses power iteration:
  - Initial evecs are randomized and normalized (`PivotMDS.cpp:181-188`).
  - Loop continues while `r < EPSILON` where `EPSILON = 1 - 1e-10`
    (`PivotMDS.cpp:51`, `PivotMDS.cpp:189`).
  - It multiplies by `K`, Gram-Schmidt orthogonalizes against earlier vectors, normalizes, and
    computes `r = min_i abs(prod(new_i, old_i))` (`PivotMDS.cpp:205-234`).
  - NaN/inf raises `AlgorithmFailureException` (`PivotMDS.cpp:189-195`).
- There is no explicit max-iteration guard in the OGDF power loop (`PivotMDS.cpp:189-235`).

Iteration divergence:

- Dagua's SVD is deterministic up to backend numeric choices and typically high precision for the
  given tensor dtype; OGDF's power iteration has a hard angular convergence threshold
  `1 - 1e-10`, C `rand()` initialization, Gram-Schmidt ordering, and no max iteration.
- Even with identical centered matrices, signs, tiny rotations in nearly degenerate singular
  subspaces, and scale drift can differ.

## 7. Hyperparameter alignment table

| Parameter / behavior | Dagua value | OGDF/reference value | Match? | Evidence |
| --- | --- | --- | --- | --- |
| Public engine name | `classic_pivot_mds` | `ogdf_pivot_mds` | N/A | `classic_competitor.py:1094-1099`; `ogdf_competitor.py:273-279` |
| Default benchmark variant pivot counts | 10, 50, 100, 200 in variants | Same reference target but no reference params sent | No | `variants.py:1177-1220`; `_run_ogdf` payload lacks params at `ogdf_competitor.py:138-144` |
| Classic competitor default `n_pivots` | 50 | N/A | No | `classic_competitor.py:194-198`; `classic_competitor.py:1132-1137` |
| OGDF internal default pivots | N/A | 250 | No | `PivotMDS.h:59-64` |
| Pivot setter lower bound | Dagua rejects `<=0` | OGDF clamps to at least dimension count; comment says default 250 but code uses `max(numberOfPivots, m_dimensionCount)` | No | `pivot_mds.py:107-110`; `PivotMDS.h:68-72` |
| First pivot | Random `torch.randint` | `G.firstNode()` | No | `distance.py:960-963`; `PivotMDS.cpp:260-265` |
| Pivot selection after first | Greedy max-min via `torch.argmax` | Greedy max-min by graph node iteration and `>` | Mostly | `distance.py:966-979`; `PivotMDS.cpp:275-283` |
| Tie handling | Lowest tensor index from `torch.argmax` after sorted adjacency; selected masked | Keeps earlier current pivot on equality because condition is `>` | Mostly | `distance.py:970-971`; `PivotMDS.cpp:278-282` |
| Shortest path unweighted edge cost | 1.0 | 100.0 | No for raw scale, mostly irrelevant after separate normalization/Procrustes | `_resolve_weights` returns 1.0 at `preprocess.py:220-224`; `PivotMDS.h:59-64`; `PivotMDS.cpp:272` |
| Weighted edge support | Optional `edge_weights` in pipeline | OGDF supports edge weight attribute if enabled, but runner never sends weights or enables it | No in benchmark adapter | `pivot_mds.py:75`; `PivotMDS.h:95-99`; `scripts/ogdf_runner.cpp:138-144`, `scripts/ogdf_runner.cpp:203-206` |
| Weight transform | None by default, optional inverse in generic adjacency | None in Pivot-MDS runner | Partial | `preprocess.py:223-227`; `PivotMDS.cpp:249-259` |
| Duplicate edges | Dagua dedups with min | OGDF path detection simplifies for path check; BFS over graph with multi-edges should not change unweighted distances | Partial | `pivot_mds.py:53-58`; `preprocess.py:388-403`; `PivotMDS.cpp:296-317` |
| Self-loops | Ignored in undirected adjacency | Ignored by path traversal; likely harmless in shortest paths, but graph still contains them | Partial | `preprocess.py:389-394`; `PivotMDS.cpp:164-168` |
| Directedness | Undirected adjacency | OGDF graph is effectively traversed undirected via node adjacency/twin nodes and shortest path helpers on undirected graph object | Yes for benchmark graphs | `BuildAdjacencyConfig.directed=False` at `preprocess.py:765-786`; runner creates OGDF edges at `scripts/ogdf_runner.cpp:214-217` |
| Empty graph | Empty tensor | OGDF algorithm returns early; adapter also returns empty before runner | Yes | `postprocess.py:989-991`; `PivotMDS.cpp:99-102`; `ogdf_competitor.py:131-132` |
| Single node | Zero after normalization | Sets x/y zero | Yes | `_normalize_classical_positions` at `postprocess.py:873-874`; `PivotMDS.cpp:104-111` |
| Path graph | General pivot MDS | Straight-line path layout | No | `pipeline/pivot_mds.py:51-65`; `PivotMDS.cpp:114-118`, `PivotMDS.cpp:152-179` |
| Disconnected graph | Supported with `max_distance+1` fill | Adapter rejects before running; OGDF asserts connected | No | `distance.py:325-328`; `ogdf_competitor.py:207-214`; `PivotMDS.cpp:54-57` |
| Centering formula | Rectangular double-centering | Same algebra | Yes | `embed.py:293-297`; `PivotMDS.cpp:60-90` |
| SVD/eigensolver | `torch.linalg.svd` | Power iteration on `C C^T` | No | `embed.py:299`; `PivotMDS.cpp:181-235`, `PivotMDS.cpp:360-390` |
| Numeric dtype for distances | `float32` pivot rows | `double` arrays | No | `distance.py:328`, `distance.py:1039-1043`; `PivotMDS.cpp:238-285` |
| Numeric dtype for centered matrix | `float32` if input pivot distances are `float32` | `double` | No | `embed.py:293-309`; `PivotMDS.cpp:60-90` |
| Final output dtype | `torch.float32` | JSON decimal parsed into `torch.float32` by adapter | Mostly | `postprocess.py:998`; `ogdf_competitor.py:165` |
| Final scale | Dagua extent normalization | OGDF raw coordinates; benchmark likely Procrustes-normalizes outside this adapter | No | `postprocess.py:993-998`; `scripts/ogdf_runner.cpp:232-240` |
| Seed semantics | Python adapter seed affects first pivot | Python adapter seed ignored; OGDF internal Pivot-MDS eigen RNG always `srand(0)` | No | `classic_competitor.py:1132-1138`; `ogdf_competitor.py:193-203`; `PivotMDS.cpp:337-343` |
| 3D support | Always 2D output | Can do 3D if GA has `threeD`; runner does not | Benchmark yes, library no | `embed.py:277-309`; `PivotMDS.cpp:92-96`; `scripts/ogdf_runner.cpp:203-206` |

## 8. Edge cases

Self-loops:

- Dagua ignores self-loops while building undirected adjacency: if `not directed and source == target`,
  the loop is skipped (`dagua/layout/ops/preprocess.py:389-394`).
- OGDF path traversal explicitly ignores self-loops (`PivotMDS.cpp:164-168`). General shortest-path
  behavior likely treats self-loops as non-improving with positive cost, but the source file does not
  explicitly filter them in `getPivotDistanceMatrix` (`PivotMDS.cpp:268-274`).
- Expected impact: low for general connected graphs, potentially high for path detection because OGDF
  `getRootedPath` first calls `makeSimpleUndirected(GC)` (`PivotMDS.cpp:296-299`), while dagua never
  switches to path layout.

Multi-edges:

- Dagua collapses duplicate undirected edges by `dedup="min"` in the pipeline
  (`pivot_mds.py:53-58`) and `_aggregate_neighbor_weights(..., "min")`
  (`preprocess.py:230-259`, `preprocess.py:388-403`).
- OGDF path detection simplifies a graph copy via `makeSimpleUndirected(GC)`
  (`PivotMDS.cpp:296-299`). Shortest-path traversal on multi-edges should produce the same
  unweighted distances when all edges have the same `m_edgeCosts=100`, but weighted multi-edge
  semantics could diverge if a future runner passes edge weights.
- Expected impact: low for unweighted general graphs, but could affect whether OGDF triggers its
  path special case.

Disconnected components:

- Dagua supports disconnected graphs by filling unreachable pivot distances with `max_distance + 1`
  for each pivot row (`distance.py:325-328`).
- OGDF `PivotMDS::call` asserts connectedness (`PivotMDS.cpp:54-57`), and the Python OGDF adapter
  returns an error `"requires connected graph"` for disconnected `pivot_mds` before invoking the
  runner (`ogdf_competitor.py:207-214`).
- Existing fidelity report shows only 92 OK cases for each Pivot-MDS variant, unlike many families
  with 104/105 OK cases (`eval_output/fidelity_report/report.md:66-69`), consistent with skipped or
  rejected graph cases.
- Expected impact: binary mismatch on disconnected graphs, but these may be excluded from RMSD
  aggregation due to OGDF error.

Weighted edges:

- Dagua accepts `edge_weights`, validates shape, treats adjacency as weighted when weights exist, and
  switches pivot distances to Dijkstra (`pivot_mds.py:111-130`; `distance.py:318-328`).
- OGDF library supports an edge-cost attribute controlled by `useEdgeCostsAttribute`
  (`PivotMDS.h:95-99`) and reads `GA.doubleWeight(e)` when enabled (`PivotMDS.cpp:249-259`).
- The benchmark runner sends no weights (`ogdf_competitor.py:138-144`) and constructs graph
  attributes without the edge double weight attribute (`scripts/ogdf_runner.cpp:203-206`), so OGDF
  benchmark distances always use uniform `m_edgeCosts=100` (`PivotMDS.cpp:269-273`).
- Expected impact: any weighted test comparing dagua and OGDF runner is not aligned.

Empty graph:

- Dagua pipeline returns an empty `[0, 2]` tensor after empty pivot selection and finalization
  (`distance.py:952-954`, `distance.py:1033-1036`, `postprocess.py:989-991`).
- OGDF adapter returns an empty `[0, 2]` tensor before invoking the runner
  (`ogdf_competitor.py:131-132`); OGDF library itself also returns immediately for `n==0`
  (`PivotMDS.cpp:99-102`).
- Expected impact: aligned.

Single node:

- Dagua normalization returns zero for one coordinate row (`postprocess.py:873-874`).
- OGDF sets x/y/z to zero for one node (`PivotMDS.cpp:104-111`).
- Expected impact: aligned.

Two-node and simple path graphs:

- OGDF detects paths after simplifying the graph and lays them out at x positions separated by edge
  cost (`PivotMDS.cpp:114-118`, `PivotMDS.cpp:152-179`).
- Dagua runs general Pivot-MDS for all `N >= 1` (`pivot_mds.py:51-65`) and then span-normalizes
  (`postprocess.py:993-998`).
- Expected impact: Procrustes RMSD may remain low because both are collinear, but raw scale and
  possibly centered endpoint coordinates diverge.

## 9. Numerical precision

Dagua precision boundaries:

- `_graph_distances_for_pivot` converts cleaned distances to `torch.float32`
  (`dagua/layout/ops/distance.py:325-328`).
- `PivotDistanceQueries` stacks those `float32` rows (`distance.py:1038-1043`).
- `_pivot_mds_coordinates` squares, averages, centers, and runs SVD in the tensor dtype inherited
  from pivot distances, therefore normally `float32` (`embed.py:293-309`).
- Final output is explicitly `float32` (`postprocess.py:998`).
- Dagua adjacency weights are resolved through Python floats from `float64` tensors when weighted,
  but the final pivot row is still `float32` (`preprocess.py:218-227`, `distance.py:328`).

OGDF precision boundaries:

- Pivot distance matrix is `Array<Array<double>>` throughout
  (`PivotMDS.cpp:119-131`, `PivotMDS.cpp:238-285`).
- Centering accumulates `double` sums in loop order (`PivotMDS.cpp:65-88`).
- `selfProduct`, dot products, normalization, and eigenvalue iteration are all `double`
  (`PivotMDS.cpp:319-357`).
- Runner prints doubles through default `std::cout` formatting, then Python adapter parses JSON and
  casts to `torch.float32` (`scripts/ogdf_runner.cpp:232-240`; `ogdf_competitor.py:157-166`).

Summation order:

- Dagua uses vectorized torch reductions for means (`embed.py:293-297`), with backend-dependent
  summation order.
- OGDF uses explicit nested loops:
  row sums over nodes (`PivotMDS.cpp:69-76`), column sums over pivots (`PivotMDS.cpp:78-87`),
  self product with `k` loop over nodes (`PivotMDS.cpp:346-357`), and dot products with linear
  order (`PivotMDS.cpp:329-335`).
- These are sub-percent divergence sources when pivot sets match and singular values are separated.
  They can be larger on near-degenerate spectra because the 2D basis can rotate/reflection-flip
  within the subspace before Procrustes alignment.

Scale precision:

- Uniform edge cost differs by a factor of 100: dagua unweighted edges cost `1.0`
  (`preprocess.py:220-224`), OGDF unweighted BFS uses `m_edgeCosts=100`
  (`PivotMDS.h:59-64`, `PivotMDS.cpp:272`). Since squared distances and SVD scaling are homogeneous,
  this is mostly removed by Procrustes and dagua final normalization, but it affects raw coordinate
  magnitudes and numerical conditioning.

## 10. RNG semantics

No, dagua's torch seed does not produce the same sequence as reference RNG, and it does not even
drive the same algorithmic event.

Dagua:

- Seed enters `LayoutProblem(seed=seed)` (`dagua/layout/ops/pipelines/pivot_mds.py:119-125`).
- `_resolve_generator` builds a CPU `torch.Generator` and calls `manual_seed(problem.seed)`
  (`distance.py:870-874`).
- Exactly one random sample is consumed for the first pivot:
  `torch.randint(0, num_nodes, (1,), generator=generator)` (`distance.py:960-963`).
- The variants registry marks `classic_pivot_mds` stochastic (`variants.py:1820-1835`).

OGDF adapter and runner:

- `_OGDFBase.layout` deletes the Python `seed`; it cannot affect the subprocess
  (`ogdf_competitor.py:179-204`).
- Sprint summary already flags this as an OGDF adapter seed issue
  (`algo_fidelity_SUMMARY.md:192-198`).
- Runner calls `ogdf::setSeed(42)` and `std::srand(42)` before initialization
  (`scripts/ogdf_runner.cpp:219-222`), but the Pivot-MDS first pivot is not random.
- OGDF `PivotMDS::randomize` resets C RNG to static `SEED=0` and fills eigenvectors with
  `rand()/RAND_MAX` (`PivotMDS.h:108-109`, `PivotMDS.cpp:337-343`).

Net:

- Dagua seed changes pivot rows.
- OGDF Python seed is ignored.
- OGDF runner seed `42` does not control pivot selection.
- OGDF internal eigen RNG always restarts from C seed `0`, independent of runner seed.
- Therefore seed-to-seed sequence equivalence is impossible without changing both adapter and
  dagua pivot/eigensolver semantics.

## 11. Edge-case bugs

Potential or confirmed divergence bugs:

1. **Reference pivot count is not configurable in benchmark runner.**
   Variants advertise `n_pivots=10/50/100/200` on the dagua side
   (`variants.py:1177-1220`), but `_run_ogdf` sends only `"nodes"`, `"edges"`, and
   `"algorithm"` (`ogdf_competitor.py:138-144`). `scripts/ogdf_runner.cpp` constructs default
   `ogdf::PivotMDS layout;` with no `setNumberOfPivots` call (`scripts/ogdf_runner.cpp:164-167`).
   OGDF default is 250 (`PivotMDS.h:59-64`). This means every Pivot-MDS variant is comparing
   dagua P against OGDF min(N,250), not OGDF P.

2. **Dagua first pivot does not match OGDF.**
   Dagua uses a random first pivot (`distance.py:960-963`); OGDF uses `G.firstNode()`
   (`PivotMDS.cpp:260-265`). This is the highest-impact algorithmic mismatch on connected,
   non-path graphs.

3. **OGDF path special case is missing in dagua.**
   OGDF detects simple paths and calls `doPathLayout` (`PivotMDS.cpp:114-118`), which writes a
   straight line with edge-cost increments (`PivotMDS.cpp:152-179`). Dagua never performs this
   branch (`pivot_mds.py:51-65`). Strong equivalence survives because path layouts are collinear,
   but raw and normalized details diverge.

4. **Disconnected behavior is intentionally incompatible.**
   Dagua fills unreachable distances (`distance.py:325-328`), while OGDF adapter rejects
   disconnected graphs (`ogdf_competitor.py:207-214`) because OGDF asserts connectedness
   (`PivotMDS.cpp:54-57`). This is not a silent numerical bug but is a fidelity-surface mismatch.

5. **Uniform edge cost scale differs by 100x.**
   Dagua unweighted adjacency uses `1.0` (`preprocess.py:220-224`); OGDF default edge cost is `100`
   (`PivotMDS.h:59-64`, `PivotMDS.cpp:272`). Procrustes usually removes it, but raw coordinates and
   power-iteration conditioning are not the same.

6. **Dagua uses `float32` too early.**
   Dagua pivot rows are `float32` before centering/SVD (`distance.py:328`, `distance.py:1043`);
   OGDF uses `double` for all internal arrays (`PivotMDS.cpp:60-90`, `PivotMDS.cpp:319-390`).
   This is a residual sub-percent divergence source.

7. **SVD semantics differ.**
   Dagua uses exact backend SVD on centered rectangular matrix (`embed.py:299-305`); OGDF uses power
   iteration on `C C^T` and reconstructs coordinates through `C^T x`
   (`PivotMDS.cpp:360-390`). This is expected to affect small graphs with close singular values.

8. **Final normalization is dagua-only.**
   Dagua centers and scales to an extent (`postprocess.py:870-890`, `postprocess.py:993-998`);
   OGDF writes and prints raw coordinates (`PivotMDS.cpp:140-148`,
   `scripts/ogdf_runner.cpp:232-240`). The evaluation's Procrustes handling likely hides this, but
   adapter-level outputs are not equivalent.

9. **`ClassicPivotMDS.layout` ignores variant params when called directly.**
   The generic spec has default params (`classic_competitor.py:194-198`), and variants carry
   params (`variants.py:1177-1220`), but `ClassicPivotMDS.layout` directly hard-codes
   `n_pivots=50` (`classic_competitor.py:1132-1137`). If variant execution uses
   `layout_with_variant`, it may bypass this method; if not, pivot variants collapse to P=50.
   This needs confirmation in the benchmark dispatcher, but the direct method is suspicious.

10. **OGDF runner seeds initial graph coordinates even though Pivot-MDS overwrites them.**
    Runner initializes x/y with `rand()%1000/10` (`scripts/ogdf_runner.cpp:219-228`). For
    Pivot-MDS this appears dead because `pivotMDSLayout` writes every coordinate in all branches
    (`PivotMDS.cpp:104-111`, `PivotMDS.cpp:140-148`, `PivotMDS.cpp:152-179`). It is harmless, but
    can mislead seed debugging.

No obvious wrong-sign bug was found in the centering formula: dagua's
`-0.5 * (D^2 - row_mean - col_mean + grand)` (`embed.py:293-297`) matches OGDF's
`FACTOR * (D^2 + grand - row_mean - col_mean)` (`PivotMDS.cpp:81-88`) algebraically.

## 12. Ranked fix list

Ranked by expected RMSD/fidelity impact for `classic_pivot_mds` vs `ogdf_pivot_mds`.

1. **Expose and apply OGDF pivot count in the runner and adapter.**
   - Evidence: dagua variants set P=10/50/100/200 (`variants.py:1177-1220`), but `_run_ogdf`
     payload has no params (`ogdf_competitor.py:138-144`) and runner never calls
     `setNumberOfPivots` (`scripts/ogdf_runner.cpp:164-167`). OGDF default is 250
     (`PivotMDS.h:59-64`).
   - Proposed fix: add optional `"numberOfPivots"` JSON key, parse it in `scripts/ogdf_runner.cpp`,
     call `layout.setNumberOfPivots(value)` for `pivot_mds`, and pass reference variant params from
     `variants.py`.
   - Size estimate: M, touching runner parser, OGDF adapter variant plumbing, and tests.
   - Expected RMSD impact: high for graphs with `N > P` where dagua currently samples fewer pivots
     than OGDF.

2. **Add an OGDF-compatible pivot-selection mode to dagua.**
   - Evidence: dagua random first pivot (`distance.py:960-963`) vs OGDF `G.firstNode()`
     (`PivotMDS.cpp:260-265`).
   - Proposed fix: extend `PivotSelectionConfig` with `first_pivot="random"|"first_node"` or an
     explicit initial pivot; use first node for `ogdf_pivot_mds` fidelity variants.
   - Size estimate: S/M, plus regression tests around first pivot and tie handling.
   - Expected RMSD impact: high; pivot rows define the entire embedding.

3. **Implement OGDF path special case in dagua fidelity mode.**
   - Evidence: OGDF branches to `doPathLayout` (`PivotMDS.cpp:114-118`) and increments x by edge
     costs (`PivotMDS.cpp:152-179`); dagua always runs general pipeline (`pivot_mds.py:51-65`).
   - Proposed fix: add an optional pre-embedding path detector matching OGDF's simplified
     undirected path test (`PivotMDS.cpp:296-317`) and emit straight-line positions before final
     normalization, or bypass normalization in strict fidelity mode.
   - Size estimate: M.
   - Expected RMSD impact: high on path graphs, low on non-path graphs.

4. **Move Pivot-MDS internal math to float64 until final cast.**
   - Evidence: dagua converts pivot rows to `float32` (`distance.py:328`) and SVD uses that dtype
     (`embed.py:293-309`), while OGDF uses `double` arrays throughout (`PivotMDS.cpp:60-90`,
     `PivotMDS.cpp:319-390`).
   - Proposed fix: return `float64` pivot distances in Pivot-MDS fidelity mode or cast to
     `float64` before centering/SVD, then cast final output to `float32`.
   - Size estimate: S.
   - Expected RMSD impact: medium to low; important for residual sub-percent differences.

5. **Match OGDF uniform edge cost scale or explicitly normalize before SVD.**
   - Evidence: dagua default edge costs 1.0 (`preprocess.py:220-224`); OGDF default edge cost 100
     (`PivotMDS.h:59-64`, `PivotMDS.cpp:272`).
   - Proposed fix: for OGDF-fidelity mode, use `edge_cost=100.0` in unweighted pivot distances.
     If evaluation always Procrustes-normalizes, this is less urgent, but it improves raw parity and
     numeric conditioning.
   - Size estimate: S.
   - Expected RMSD impact: low after Procrustes, medium for raw-output comparisons.

6. **Offer an OGDF-style eigensolver path for strict fidelity.**
   - Evidence: dagua uses `torch.linalg.svd` (`embed.py:299`); OGDF uses C `rand()` initialized
     power iteration on `C C^T` with `EPSILON=1-1e-10` (`PivotMDS.cpp:51`,
     `PivotMDS.cpp:181-235`, `PivotMDS.cpp:337-390`).
   - Proposed fix: only if needed after higher-impact fixes, implement a deterministic power
     iteration matching OGDF loop order and scaling, probably in numpy float64 for CPU fidelity.
   - Size estimate: L.
   - Expected RMSD impact: low/medium; mostly degenerate spectra and residual precision.

7. **Align final normalization modes.**
   - Evidence: dagua finalizes with `_normalize_classical_positions` (`postprocess.py:993-998`);
     OGDF prints raw coordinates (`scripts/ogdf_runner.cpp:232-240`).
   - Proposed fix: add a raw-output fidelity option or normalize both sides identically before
     metric computation. Prefer metric-side handling if RMSD already uses Procrustes.
   - Size estimate: S/M depending on where normalization is controlled.
   - Expected RMSD impact: low for Procrustes RMSD, high for raw coordinate tests.

8. **Clarify disconnected graph policy.**
   - Evidence: dagua supports disconnected fill (`distance.py:325-328`); OGDF adapter rejects
     disconnected graphs (`ogdf_competitor.py:207-214`).
   - Proposed fix: either skip disconnected graphs for this family explicitly in variant metadata,
     or add component-wise OGDF-compatible fallback to dagua only outside OGDF fidelity comparisons.
   - Size estimate: S for metadata/reporting; M/L for component layout.
   - Expected RMSD impact: high on excluded cases, none on current OK set.

9. **Audit `ClassicPivotMDS.layout` vs variant-param execution.**
   - Evidence: direct method hard-codes `n_pivots=50` (`classic_competitor.py:1132-1137`) while
     variants define 10/50/100/200 (`variants.py:1177-1220`).
   - Proposed fix: verify `layout_with_variant` path for classic competitors; if direct layout is
     used for variants, route variant params into the method.
   - Size estimate: S if only adapter plumbing, M with tests.
   - Expected RMSD impact: high for variants if currently collapsed; no impact if dispatcher already
     bypasses the direct method.

## 13. Recommended Round 22+ fix scope

Recommended next bundle: "OGDF parameter and pivot selection fidelity."

Top-K scope:

1. Add reference-side pivot-count plumbing for `ogdf_pivot_mds`:
   pass P from variants to the OGDF runner and call `PivotMDS::setNumberOfPivots`.
   This addresses the largest confirmed benchmark-contract mismatch
   (`variants.py:1177-1220`, `ogdf_competitor.py:138-144`, `scripts/ogdf_runner.cpp:164-167`,
   `PivotMDS.h:59-72`).

2. Add dagua `first_node` pivot initialization mode and use it for OGDF-fidelity comparisons.
   This aligns the pivot rows with OGDF's `G.firstNode()` start (`distance.py:960-963`,
   `PivotMDS.cpp:260-265`).

3. Switch dagua Pivot-MDS internal centering/SVD to `float64` in fidelity mode.
   This removes avoidable precision noise after pivot sets are aligned (`distance.py:328`,
   `embed.py:293-309`, `PivotMDS.cpp:60-90`).

4. Add explicit tests comparing selected pivot indices and pivot distance rows on small connected
   graphs before comparing final positions. This catches the root cause earlier than Procrustes RMSD.

Defer to later rounds:

- OGDF path special-case emulation. It matters, but after pivot-count and first-pivot alignment the
  remaining path failures will be easy to isolate (`PivotMDS.cpp:114-179`).
- OGDF power-iteration eigensolver emulation. It is high effort and should only be attempted if
  aligned pivots plus float64 still leave meaningful residual divergence (`PivotMDS.cpp:181-235`,
  `PivotMDS.cpp:360-390`).
- Disconnected component policy. OGDF does not define general disconnected Pivot-MDS behavior in this
  path, and the current adapter already rejects it (`ogdf_competitor.py:207-214`).

Assumption for this report: "classic_pivot_mds vs ogdf_pivot_mds" means the active ops pipeline
`dagua/layout/ops/pipelines/pivot_mds.py`, not the older archived implementation. The archived and
`dagua/layout/classic/pivot_mds.py` sources were still read because tests and historical imports
reference them, and they confirm the same dagua-side algorithmic choices.
