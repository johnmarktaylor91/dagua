# Round 21 Adversarial Diff: `classic_classical_mds` vs `igraph_mds`

Date: 2026-04-30
Branch: `develop`
Scope: diagnosis only; no source changes.

## 1. Files Read

Dagua implementation and wiring:

- `dagua/layout/ops/pipelines/classical_mds.py:1-104` -- composable
  classical-MDS pipeline and public adapter.
- `dagua/layout/ops/distance.py:765-835` -- pipeline distance-matrix op.
- `dagua/layout/ops/embed.py:312-353` and `dagua/layout/ops/embed.py:1932-1991`
  -- pipeline classical-MDS embedding math and op wrapper.
- `dagua/layout/ops/postprocess.py:870-942` -- classical-MDS final
  centering, fallback, scale normalization, and dtype/device cast.
- `dagua/layout/classic/classical_mds.py:1-235` -- legacy/classic
  implementation that the pipeline is tested against exactly.
- `dagua/layout/_archive/classic/_graph_distances.py:1-227` -- shared
  undirected adjacency, self-loop handling, multi-edge collapse, BFS, Dijkstra,
  and connectivity helpers used by the legacy implementation.
- `dagua/layout/ops/graph_utils.py:311-352` -- copied shortest-path helper
  notes and equivalent global unreachable-fill behavior.
- `dagua/eval/competitors/classic_competitor.py:26-98`,
  `dagua/eval/competitors/classic_competitor.py:153-188`,
  `dagua/eval/competitors/classic_competitor.py:1029-1059`, and
  `dagua/eval/competitors/classic_competitor.py:1570-1627` -- classic
  competitor dispatch, defaults, classical-MDS wrapper, and edge-weight
  forwarding.
- `dagua/eval/competitors/igraph_competitor.py:18-99`,
  `dagua/eval/competitors/igraph_competitor.py:102-184`, and
  `dagua/eval/competitors/igraph_competitor.py:240-245` -- igraph adapter,
  graph conversion, scaling, RNG context, and `igraph_mds` registration.
- `dagua/eval/variants.py:814-824` and `dagua/eval/variants.py:1820-1874`
  -- canonical pairing, stochastic flag, and heavy flag.
- `tests/test_pipeline_classical_mds.py:1-230` -- existing bit-exact tests
  against dagua's legacy implementation, not against igraph.

Reference implementation:

- `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:1-295` --
  igraph classical MDS, connected-component decomposition, and component merge.
- `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:1-190`
  and `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:264-298`
  -- DLA component merge used by disconnected igraph MDS.
- `/home/jtaylor/projects/_references/igraph/include/igraph_layout.h:254`
  -- exported `igraph_layout_merge_dla` declaration found while locating the
  merge implementation.

Existing sprint / verdict context:

- `eval_output/fidelity_report/report.md:1-12` -- current mega-run verdict:
  `classical_mds_default` is `strong_equivalent`, deterministic, `N OK=104`,
  median Procrustes RMSD `0.000`.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:27-35`
  -- graphviz/neato classical-MDS context.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:188-198`
  -- prior known future-cleanup pattern for already-strong families.

## 2. Overall Pipeline Structure

Dagua high-level flow:

1. `layout_classical_mds_pipeline()` validates `num_nodes` and optional
   `edge_weights`, constructs a `LayoutProblem`, creates an empty `SolveState`,
   and runs a CPU `ExecutionPlan` through `build_classical_mds_pipeline()`
   (`dagua/layout/ops/pipelines/classical_mds.py:42-101`).
2. The pipeline is exactly three ops: `ClassicalMDSDistanceMatrix`,
   `ClassicalMDSComputeEmbedding`, and `ClassicalMDSFinalizePositions`
   (`dagua/layout/ops/pipelines/classical_mds.py:21-39`).
3. `ClassicalMDSDistanceMatrix` writes a CPU float64 dense shortest-path matrix
   into `state.distance_matrix` (`dagua/layout/ops/distance.py:804-835`).
4. `ClassicalMDSComputeEmbedding` turns that distance matrix into raw 2D
   coordinates using double-centered eigendecomposition (`dagua/layout/ops/embed.py:1959-1991`).
5. `ClassicalMDSFinalizePositions` recenters, rescales to dagua's layout extent,
   casts to float32, and returns on the layout device
   (`dagua/layout/ops/postprocess.py:904-942`).

igraph high-level flow:

1. The Python adapter creates a directed `igraph.Graph`, adds all dagua vertices,
   adds all dagua edges, and stores edge weights as an edge attribute when
   present (`dagua/eval/competitors/igraph_competitor.py:53-76`).
2. `IgraphMDS` calls `ig.layout("mds", **{})`; no explicit distance matrix,
   weights, or dimension parameter is passed by the adapter
   (`dagua/eval/competitors/igraph_competitor.py:167-180`,
   `dagua/eval/competitors/igraph_competitor.py:240-245`).
3. C-side `igraph_layout_mds()` checks distance-matrix shape and dimension,
   computes undirected shortest-path distances if `dist == 0`, or copies the
   caller's matrix and zeros its diagonal otherwise
   (`/home/jtaylor/projects/_references/igraph/src/layout/mds.c:188-221`).
4. igraph checks weak connectivity (`IGRAPH_WEAK`). Connected graphs go directly
   to `igraph_i_layout_mds_single()` (`mds.c:223-228`).
5. Disconnected graphs are split into weak components, each component receives a
   distance submatrix and its own single-component MDS layout, then all component
   layouts are merged with `igraph_layout_merge_dla()` and rows are reordered
   to original vertex order (`mds.c:229-288`).
6. The adapter converts the igraph layout to a torch tensor and multiplies both
   coordinates by `50.0` (`dagua/eval/competitors/igraph_competitor.py:79-99`).

Main structural diagnosis: for connected, unweighted, non-trivial graphs, both
sides are recognizable Torgerson classical MDS on shortest-path distances.
For disconnected graphs, weighted graphs, empty graphs, and two-node graphs,
the structural flow is not equivalent.

## 3. Energy / Loss / Objective

Classical MDS is closed-form, not iterative stress descent in either
implementation. There is no explicit loss accumulator or force loop. The
objective implied by both sides is the rank-2 Euclidean embedding recovered from
the double-centered squared distance matrix.

Dagua formula:

- Distances are squared with `squared = distances_np * distances_np`
  (`dagua/layout/ops/embed.py:334-335`).
- The centering matrix is `J = I - 11^T / N`
  (`dagua/layout/ops/embed.py:336-338`).
- The Gram matrix is `B = -0.5 * J @ D^2 @ J`
  (`dagua/layout/ops/embed.py:339`).
- Eigenpairs are computed with `np.linalg.eigh(gram)` and sorted descending
  (`dagua/layout/ops/embed.py:341-342`).
- Dagua selects only strictly positive eigenvalues, up to two dimensions:
  `positive_indices = [index for index in sorted_indices if eigenvalues[index] > 0.0][:2]`
  (`dagua/layout/ops/embed.py:342-344`).
- Coordinates are `v_i * sqrt(lambda_i)` for selected positive eigenvalues
  (`dagua/layout/ops/embed.py:345-350`).
- If no positive eigenvalue exists, dagua creates a deterministic line fallback
  in x from `-1.0` to `1.0` (`dagua/layout/ops/embed.py:350-352`).

igraph formula:

- igraph squares the distance matrix in place (`mds.c:91-96`).
- It computes row means via BLAS matrix-vector multiply with an all-`1/N`
  vector, then `grand_mean = sum(row_means) / N` (`mds.c:98-103`).
- Each matrix entry becomes
  `-0.5 * (d_ij^2 + grand_mean - row_mean_i - row_mean_j)`
  through `add_constant(grand_mean)`, subtracting both row means, and multiplying
  by `-0.5` (`mds.c:103-108`). This is algebraically the same double centering
  as `-0.5 * J D^2 J`.
- It asks for the largest algebraic eigenpairs with LAPACK:
  `which.pos = IGRAPH_EIGEN_LA`, `algorithm = IGRAPH_EIGEN_LAPACK`
  (`mds.c:113-121`).
- It uses `sqrt(fabs(lambda))` for each requested dimension (`mds.c:123-126`).
- It writes coordinates in reversed eigen-dimension order:
  `for (j = 0, k = nev - 1; j < nev; j++, k--) res[i,k] = value[j] * vector[i,j]`
  (`mds.c:127-132`).

Per-term comparison:

| Term / operation | Dagua | igraph | Match? | RMSD impact |
|---|---|---|---|---|
| Distance target on connected unweighted graph | Undirected shortest paths via shared adjacency/BFS (`_graph_distances.py:21-64`, `_graph_distances.py:108-140`) | `igraph_distances(..., IGRAPH_ALL)` when `dist == 0` (`mds.c:210-214`) | Mostly yes | Low for simple connected unweighted graphs |
| Squared distances | `D * D` (`embed.py:334-335`) | in-place multiply (`mds.c:91-96`) | Yes | None |
| Double centering | Explicit `J @ D^2 @ J` (`embed.py:336-340`) | row/grand mean centering (`mds.c:98-108`) | Algebraically yes | Tiny numerical only |
| Eigen target | Full symmetric eigensolve, all eigenpairs (`embed.py:341`) | LAPACK largest algebraic eigenpairs (`mds.c:113-121`) | Similar but not identical API | Low-to-medium on near-degenerate spectra |
| Negative eigenvalues | Dropped; only `> 0` selected (`embed.py:342-350`) | `sqrt(fabs(lambda))` even if selected eigenvalue is negative (`mds.c:123-130`) | No | High on non-Euclidean graph distances with negative second eigenvalue |
| Axis order | Highest eigenvalue goes into column 0 (`embed.py:345-350`) | Highest eigenvalue goes into last requested column (`mds.c:127-132`) | No, but Procrustes absorbs | Low for Procrustes RMSD; high for raw orientation |
| Final scale | Dagua extent normalization (`postprocess.py:935-941`) | igraph adapter multiplies by 50 (`igraph_competitor.py:94-99`) | No | Low for Procrustes RMSD; high for raw coordinates |

## 4. Force / Gradient Computation

No force or gradient computation applies to this family.

- Dagua has no optimizer in the classical-MDS pipeline; the pipeline list is
  distance, embedding, finalize only (`dagua/layout/ops/pipelines/classical_mds.py:32-37`).
- Dagua's embedding op simply validates `state.distance_matrix` and calls
  `_classical_mds_embedding()` (`dagua/layout/ops/embed.py:1983-1991`).
- igraph's single-component routine modifies the distance matrix into a Gram
  matrix and calls `igraph_eigen_matrix_symmetric()` (`mds.c:91-121`).
- The only function named "step" in igraph MDS is a matrix-vector multiply
  callback for the eigen solver, not a layout iteration:
  `igraph_i_layout_mds_step()` calls BLAS `dgemv` (`mds.c:36-48`).

## 5. Initialization

Dagua:

- No random initialization is used. The seed argument is explicitly ignored in
  the public pipeline adapter via `_ = seed, node_sizes`
  (`dagua/layout/ops/pipelines/classical_mds.py:60-64`,
  `dagua/layout/ops/pipelines/classical_mds.py:78`).
- The legacy implementation also ignores `seed` with `_ = seed`
  (`dagua/layout/classic/classical_mds.py:194-196`,
  `dagua/layout/classic/classical_mds.py:211`).
- The first computed state is the all-pairs distance matrix
  (`dagua/layout/ops/distance.py:829-835`).

igraph:

- Connected MDS has no random initialization. If `dist == 0`, it computes
  distances, then directly applies eigendecomposition (`mds.c:210-228`).
- Disconnected MDS has stochastic component merge. `igraph_layout_mds()`
  decomposes and lays out components (`mds.c:229-276`), then calls
  `igraph_layout_merge_dla()` (`mds.c:277-280`).
- `igraph_layout_merge_dla()` starts the largest component at the origin and
  then uses random DLA placement for subsequent components
  (`merge_dla.c:123-155`).
- The random DLA walk uses `RNG_UNIF()` for angle and radius at particle start
  and each walk step (`merge_dla.c:276-296`).

Initialization diagnosis: the family is deterministic only for connected
graphs. For disconnected graphs, igraph MDS uses RNG in the component merge, but
the current adapter does not mark `IgraphMDS` as RNG-using
(`dagua/eval/competitors/igraph_competitor.py:102-109`,
`dagua/eval/competitors/igraph_competitor.py:240-245`).

## 6. Iteration / Convergence

Dagua:

- No step count, learning rate, cooling schedule, or convergence test exists in
  the classical-MDS pipeline (`dagua/layout/ops/pipelines/classical_mds.py:32-37`).
- The only postprocessing loop is a simple normalization/fallback check in
  `_normalize_classical_positions()` (`dagua/layout/ops/postprocess.py:870-890`).

igraph:

- Connected MDS has no layout iteration or convergence test. The heavy work is
  BLAS centering plus LAPACK eigensolve (`mds.c:98-121`).
- Disconnected merge does iterate a random walk until a component sphere
  touches an occupied sphere: `while (sp < 0)` and nested `while (sp < 0 &&
  DIST(*x, *y) < killr)` (`merge_dla.c:276-296`). This is a stochastic packing
  loop, not stress optimization.

Convergence diagnosis: there is no tunable MDS convergence to align. Remaining
differences come from distance construction, eigensolver semantics,
postprocessing, and disconnected component packing.

## 7. Hyperparameter Alignment Table

| Parameter / behavior | Dagua default | igraph/reference default through current adapter | Match? | Source refs |
|---|---:|---:|---|---|
| Dimensions | Fixed 2D output | Python adapter passes no `dim`; igraph layout default is exercised as 2D by `layout("mds")`; C routine requires `dim > 1` and `dim <= N` | Yes for normal calls | `classical_mds.py:42-48`; `igraph_competitor.py:240-245`; `mds.c:201-208` |
| Distance input | Always computed internally from dagua graph | `dist == 0`, so igraph computes distances internally | Yes for unweighted | `distance.py:829-835`; `mds.c:210-214` |
| Direction for distances | Undirected adjacency from directed edge tensor | `IGRAPH_ALL` distances and weak connectivity | Mostly yes | `_graph_distances.py:21-64`; `mds.c:213-224` |
| Self-loops | Ignored in dagua adjacency | igraph graph receives self-loops; shortest path diagonal still zero and loops should not reduce paths | Mostly yes | `_graph_distances.py:56-58`; `igraph_competitor.py:70-75`; `mds.c:217-220` |
| Multi-edges | Collapsed to minimum weight / one neighbor | igraph graph retains parallel edges; unweighted distance unaffected, weighted distance not passed to MDS | Partial | `_graph_distances.py:59-63`; `igraph_competitor.py:70-75` |
| Edge weights | Forwarded into dagua layout by `_quick_classic()` and used in Dijkstra | Stored as edge attribute but not passed as MDS distances | No | `classic_competitor.py:1607-1618`; `_graph_distances.py:51-63`; `igraph_competitor.py:74-75`; `igraph_competitor.py:167-180` |
| Seed | Accepted and ignored | Adapter seeds only if `accepts_seed_matrix` or `uses_igraph_rng` is true; `IgraphMDS` sets neither | No for disconnected; irrelevant for connected | `classical_mds.py:60-64`; `igraph_competitor.py:170-178`; `igraph_competitor.py:240-245`; `merge_dla.c:276-296` |
| Empty graph | Returns empty `[0,2]` tensor | C has no explicit `N==0` trivial branch; the `dim > N` guard is skipped when `N==0`, so behavior depends on later distance/connectivity/eigensolver paths | Unclear; needs direct adapter regression | `classical_mds.py:80-81`; `classic/classical_mds.py:223-225`; `mds.c:205-207`; `mds.c:71-84` |
| One-node graph | Returns zeros | Single-component special case returns zeros | Yes | `embed.py:328-329`; `mds.c:71-76` |
| Two-node graph | General MDS then dagua extent normalization | Special case returns row 0 all zeros, row 1 all ones | No raw; Procrustes mostly absorbs | `embed.py:334-353`; `postprocess.py:935-941`; `mds.c:77-84` |
| Positive eigenvalue selection | Strictly positive only | Largest algebraic eigenvalues, then `sqrt(fabs(lambda))` | No | `embed.py:341-350`; `mds.c:113-130` |
| Axis order | Descending eigenvalues in columns 0, 1 | Descending eigenvalues reversed into columns 1, 0 | No raw; Procrustes absorbs | `embed.py:345-350`; `mds.c:127-132` |
| Final scale | `_layout_extent`: `sqrt(N)*5` or node-size-driven | Adapter `* 50.0` | No raw; Procrustes absorbs | `classic/classical_mds.py:45-64`; `postprocess.py:935-941`; `igraph_competitor.py:94-99` |
| Output dtype | torch float32 | torch float32 allocated by default and assigned Python floats | Yes output, no internal | `postprocess.py:941`; `igraph_competitor.py:94-99` |
| Max nodes in benchmark | 5,000 | 5,000 | Yes | `classic_competitor.py:1033-1034`; `igraph_competitor.py:241-243` |
| Variant stochastic flag | `False` | `False` via base map / no `uses_igraph_rng` | Wrong for disconnected igraph MDS | `variants.py:814-824`; `variants.py:1820-1828`; `merge_dla.c:276-296` |
| Variant heavy flag | `True` | Original side inherits pairing; classical MDS is heavy due dense eigensolve | Reasonable | `variants.py:814-824`; `variants.py:1870-1874` |

## 8. Edge Cases

Self-loops:

- Dagua drops self-loops before adjacency insertion (`_graph_distances.py:56-58`).
- igraph adapter adds the edge list directly, including any self-loops
  (`igraph_competitor.py:70-75`).
- igraph's internally computed shortest paths with `IGRAPH_ALL` and C-side
  diagonal handling make self-loops unlikely to affect unweighted MDS distances;
  if a user supplied a `dist` matrix, igraph zeroes the diagonal explicitly
  (`mds.c:217-220`). Dagua also fills the diagonal with zero after finite-fill
  cleanup (`classic/classical_mds.py:131-132`).

Multi-edges:

- Dagua adjacency is a dictionary per node and keeps only the minimum weight for
  duplicates (`_graph_distances.py:43-64`).
- igraph adapter adds all edges (`igraph_competitor.py:70-75`). For unweighted
  shortest-path distances, parallel edges do not change path length. For
  weighted graphs, the current igraph MDS adapter does not pass edge weights to
  the MDS distance routine, so weighted multi-edge behavior diverges from dagua.

Disconnected components:

- Dagua replaces every unreachable pair with a global finite value
  `max_distance + 1.0` when `N > 1` (`classic/classical_mds.py:127-132`;
  identical pipeline helper path via `distance.py:829-835`).
- igraph explicitly decomposes the graph, lays out each component using its own
  submatrix, then DLA-merges component layouts (`mds.c:223-288`).
- The DLA merge chooses largest component first, gives each component a sphere
  radius `pow(size, .75)`, creates a fixed `200 x 200` merge grid, and uses
  random walks for placement (`merge_dla.c:100-155`, `merge_dla.c:276-296`).
- This is the highest-impact structural mismatch for disconnected graphs. It is
  also currently hidden by the benchmark stochastic metadata because `igraph_mds`
  is treated as deterministic (`variants.py:1820-1828`).

Weighted edges:

- Dagua forwards graph edge weights in the competitor adapter:
  `_quick_classic()` adds `edge_weights=graph.edge_weights` when present
  (`classic_competitor.py:1607-1618`).
- Dagua weighted distances use Dijkstra over the undirected adjacency
  (`_graph_distances.py:143-178`, `_graph_distances.py:181-209`).
- igraph adapter stores `g.es["weight"]`, but `IgraphMDS.layout_kwargs = {}`
  and `ig.layout("mds")` is called without `dist` or weights
  (`igraph_competitor.py:74-75`, `igraph_competitor.py:167-180`,
  `igraph_competitor.py:240-245`).
- Reference C `igraph_layout_mds()` accepts a distance matrix but not a weights
  vector (`mds.c:188-221`). So weighted parity requires adapter-side weighted
  distance construction, not just setting edge attributes.

Empty graph:

- Dagua public pipeline validates non-negative `num_nodes` and then can produce
  an empty result through the embedding/finalization path
  (`classical_mds.py:80-98`; legacy direct function returns empty at
  `classic/classical_mds.py:223-225`).
- igraph C checks `dim > no_of_nodes` for `N > 0`? The actual condition is
  `if (no_of_nodes > 0 && dim > no_of_nodes)` (`mds.c:205-207`), so the C guard
  does not reject `N=0` there. However, there is no explicit `N==0` trivial case
  in `igraph_i_layout_mds_single()` (`mds.c:71-84` only covers 1 and 2), and
  connectedness / distance behavior for an empty graph is adapter-dependent.
  This should be tested directly before any fix round changes empty behavior.

One-node graph:

- Dagua returns zero coordinates (`embed.py:326-329`).
- igraph returns a `1 x dim` zero matrix (`mds.c:71-76`).
- This is aligned.

Two-node graph:

- Dagua runs general double-centered MDS for `N=2`, then final normalization
  (`embed.py:334-353`, `postprocess.py:935-941`).
- igraph special-cases `N=2` and returns node 0 at all zeros, node 1 at all ones
  for every dimension (`mds.c:77-84`).
- In 2D this means igraph's raw two-node layout lies on the diagonal; dagua's
  raw layout lies on the principal eigen-axis. Procrustes can rotate and scale
  this away for simple two-node graphs, but raw-coordinate fidelity is not exact.

## 9. Numerical Precision

Dagua:

- Distances are stored as NumPy float64 after shortest-path cleanup
  (`classic/classical_mds.py:127-133`; pipeline op stores `torch.from_numpy`
  without downcasting at `distance.py:829-835`).
- Embedding converts the torch matrix to float64 NumPy before squaring and
  centering (`embed.py:334-340`).
- `np.linalg.eigh()` computes all eigenpairs in double precision
  (`embed.py:341`).
- Coordinates are converted to torch float32 at the end of embedding
  (`embed.py:353`), then finalization also returns float32
  (`postprocess.py:941`).
- Dagua's explicit centering matrix performs two dense matrix multiplications,
  which can have a different summation order from igraph's BLAS row-mean formula
  (`embed.py:336-340`).

igraph:

- `igraph_real_t` is double in normal igraph builds; this C file uses
  `igraph_real_t` matrices and vectors throughout (`mds.c:36-60`).
- Squaring and centering mutate the same matrix in place (`mds.c:91-108`).
- Row means use BLAS `dgemv` with a uniform vector (`mds.c:99-103`).
- Eigenvectors are computed through igraph's symmetric eigen wrapper with LAPACK
  (`mds.c:113-121`).
- The Python adapter converts resulting Python floats into a default torch
  tensor created by `torch.zeros(num_nodes, 2)`, therefore float32 in default
  PyTorch settings (`igraph_competitor.py:79-99`).

Precision diagnosis:

- Connected, well-conditioned graphs should differ only in harmless
  eigensolver sign/axis choices and tiny float64 summation differences before
  both sides are cast to float32.
- Near-degenerate eigenvalues can produce basis rotations within the degenerate
  subspace. Procrustes should absorb this; raw layout comparisons should not.
- The negative-eigenvalue semantic mismatch is not a precision issue:
  dagua discards negative selected modes while igraph takes `sqrt(abs(lambda))`
  (`embed.py:342-350` vs `mds.c:123-130`).

## 10. RNG Semantics

Connected graphs:

- Dagua's seed has no effect (`classical_mds.py:60-64`,
  `classical_mds.py:78`).
- igraph connected MDS has no random initialization (`mds.c:223-228`).
- Therefore dagua's torch seed cannot and need not match igraph's RNG sequence
  for connected graphs.

Disconnected graphs:

- Dagua still has no RNG; it embeds a single finite-filled distance matrix
  (`classic/classical_mds.py:127-133`, `embed.py:334-353`).
- igraph uses random DLA component placement through `RNG_UNIF()`
  (`merge_dla.c:276-296`).
- The current adapter only installs a Python `random.Random(seed)` RNG when
  `uses_igraph_rng` is true (`igraph_competitor.py:18-50`,
  `igraph_competitor.py:177-178`).
- `IgraphMDS` does not set `uses_igraph_rng = True`
  (`igraph_competitor.py:240-245`), and `variants.py` marks
  `classic_classical_mds` as non-stochastic (`variants.py:814-824`,
  `variants.py:1820-1828`).

Answer to the requested specific question: dagua's torch seed does not produce
the same sequence as the reference RNG. For connected MDS there is no reference
RNG sequence. For disconnected igraph MDS, the reference RNG comes from igraph's
global RNG through `RNG_UNIF()` in `merge_dla.c:279-289`; dagua does not use
torch RNG at all, and the adapter currently does not seed igraph for this
algorithm.

## 11. Edge-Case Bugs / Suspicious Semantics

1. **`igraph_mds` is incorrectly treated as deterministic for disconnected
   graphs.** DLA merge uses `RNG_UNIF()` (`merge_dla.c:276-296`), but the
   adapter's `IgraphMDS` class leaves `uses_igraph_rng = False`
   (`igraph_competitor.py:102-109`, `igraph_competitor.py:240-245`) and the
   variant registry marks the paired family non-stochastic
   (`variants.py:814-824`, `variants.py:1820-1828`).

2. **Weighted graph comparison is not apples-to-apples.** Dagua forwards
   `edge_weights` into classical MDS (`classic_competitor.py:1607-1618`) and
   Dijkstra consumes them (`_graph_distances.py:143-178`). igraph only stores
   the edge attribute (`igraph_competitor.py:74-75`) and then calls
   `layout("mds")` without constructing/passing a weighted distance matrix
   (`igraph_competitor.py:167-180`, `igraph_competitor.py:240-245`).

3. **Negative eigenvalue behavior differs.** Dagua filters eigenvalues to
   `> 0.0` (`embed.py:342-344`), while igraph takes `sqrt(fabs(lambda))`
   (`mds.c:123-126`). Classical MDS on graph distances can produce indefinite
   Gram matrices, so this is a real residual divergence, not just formatting.

4. **Disconnected component model differs completely.** Dagua uses finite-fill
   all-pairs distances (`classic/classical_mds.py:127-132`); igraph lays out
   components independently and DLA-packs them (`mds.c:229-288`,
   `merge_dla.c:50-56`, `merge_dla.c:123-155`). This will dominate RMSD on
   multi-component graphs if the comparator does not normalize it away.

5. **Two-node special case differs.** igraph hard-codes the second node to
   coordinate `1` in every dimension (`mds.c:77-84`), while dagua computes a
   one-dimensional MDS embedding plus normalization (`embed.py:334-353`,
   `postprocess.py:935-941`). Procrustes masks this for a single edge, but raw
   coordinates and axis semantics are not equivalent.

6. **Axis order differs.** igraph reverses largest eigenpair order into output
   columns (`mds.c:127-132`); dagua writes selected eigenpairs in descending
   order into columns from left to right (`embed.py:345-350`). This is mostly
   harmless for Procrustes but matters for deterministic raw snapshots.

7. **Final normalization differs.** Dagua recenters and rescales to a graph-size
   or node-size extent (`postprocess.py:870-890`, `postprocess.py:935-941`);
   igraph adapter applies a fixed `* 50.0` scale (`igraph_competitor.py:94-99`).
   Procrustes removes global scale, but any non-Procrustes fidelity metric will
   see this.

8. **Existing tests prove only pipeline-vs-legacy dagua equivalence.** The
   exact tests compare `layout_classical_mds_pipeline()` with
   `dagua.layout.classic.classical_mds.layout_classical_mds()`
   (`tests/test_pipeline_classical_mds.py:140-230`). They do not assert igraph
   parity for connected, disconnected, weighted, or negative-eigenvalue cases.

## 12. Ranked Fix List

Ranked by expected effect on dagua-vs-igraph RMSD and residual divergence
catalogue, not by implementation desirability.

1. **Implement igraph-style disconnected component handling for fidelity mode.**
   - Impact: highest on disconnected graphs. Dagua currently global-fills
     unreachable distances (`classic/classical_mds.py:127-132`), while igraph
     components are independently embedded and DLA-merged (`mds.c:229-288`,
     `merge_dla.c:123-155`, `merge_dla.c:276-296`).
   - Proposed fix: in a followup fidelity mode, split weak components, run the
     same single-component MDS routine per component, and DLA-pack or call a
     deterministic equivalent seeded through the benchmark seed.
   - Size estimate: large, 1-2 days if reimplementing DLA carefully; medium if
     delegating component merge to python-igraph for reference snapshots only.

2. **Mark `igraph_mds` as RNG-using when disconnected graphs are in the suite.**
   - Impact: high for measurement correctness, because igraph's DLA merge is
     stochastic (`merge_dla.c:276-296`) but the adapter does not seed it
     (`igraph_competitor.py:177-178`, `igraph_competitor.py:240-245`).
   - Proposed fix: set `IgraphMDS.uses_igraph_rng = True` or conditionally seed
     around MDS calls. Update stochastic metadata for the paired original or
     document connected-only determinism.
   - Size estimate: small, under 20 lines plus tests/benchmark metadata.

3. **Align negative-eigenvalue handling with igraph.**
   - Impact: medium-to-high on graphs whose centered graph-distance matrix has
     a negative second-largest selected mode. Dagua drops non-positive modes
     (`embed.py:342-350`); igraph uses `sqrt(fabs(lambda))` (`mds.c:123-126`).
   - Proposed fix: for igraph-fidelity mode, select the two largest algebraic
     eigenvalues and scale eigenvectors by `sqrt(abs(lambda))`, not by
     `sqrt(max(lambda, 0))` after filtering.
   - Size estimate: small-to-medium, about 20-40 lines plus regression cases.

4. **Reverse output eigen-dimension order in igraph-fidelity mode.**
   - Impact: low under Procrustes, medium for raw-coordinate comparisons and
     deterministic snapshot matching. igraph writes largest eigenpair into the
     last output dimension (`mds.c:127-132`), while dagua writes it into column
     0 (`embed.py:345-350`).
   - Proposed fix: after selecting/scaling eigenvectors, reverse the selected
     coordinate columns for igraph-compatible output.
   - Size estimate: tiny, under 10 lines if bundled with eigenvalue handling.

5. **Add igraph's two-node special case in fidelity mode.**
   - Impact: low in Procrustes metrics, but exact raw parity for tiny graphs.
     igraph returns `[0,0]` and `[1,1]` in 2D (`mds.c:77-84`); dagua uses
     general MDS and final normalization (`embed.py:334-353`,
     `postprocess.py:935-941`).
   - Proposed fix: before eigensolve, return igraph's two-node raw layout and
     then apply the chosen dagua/reference scale policy.
   - Size estimate: tiny, under 15 lines plus one test.

6. **Choose a weighted-edge parity policy.**
   - Impact: high on weighted benchmark graphs, irrelevant on unweighted graphs.
     Dagua uses weights (`classic_competitor.py:1607-1618`,
     `_graph_distances.py:143-178`); igraph MDS adapter does not pass weighted
     distances (`igraph_competitor.py:167-180`, `igraph_competitor.py:240-245`).
   - Proposed fix options: either ignore weights in `classic_classical_mds`
     when comparing to `igraph_mds`, or compute an igraph-side weighted distance
     matrix and pass it to MDS if the Python API supports `dist`.
   - Size estimate: medium, because benchmark semantics and variant metadata
     need an explicit decision.

7. **Align final scale only if raw-coordinate fidelity matters.**
   - Impact: low for current Procrustes RMSD; high for raw overlays. Dagua
     normalizes by `_layout_extent` (`postprocess.py:935-941`) while igraph
     adapter multiplies by `50.0` (`igraph_competitor.py:94-99`).
   - Proposed fix: add a `skip_finalization` or `reference_scale="igraph50"`
     option for the classical-MDS pipeline only in fidelity comparisons.
   - Size estimate: small-to-medium, depending on public API exposure.

8. **Add direct igraph parity tests for adversarial cases.**
   - Impact: medium prevention value. Current tests only assert
     pipeline-vs-legacy identity (`tests/test_pipeline_classical_mds.py:140-230`).
   - Proposed fix: add tests for connected path, two-node graph, disconnected
     two-component graph with seeded igraph DLA, weighted graph policy, and a
     synthetic indefinite Gram case.
   - Size estimate: medium, mostly test harness work and optional igraph skip.

## 13. Recommended Round 22+ Fix Scope

Recommended bundle for one followup round:

1. Fix measurement correctness first: set or conditionally apply igraph RNG
   seeding for `igraph_mds` disconnected cases and update stochastic metadata or
   documentation (`igraph_competitor.py:177-178`, `igraph_competitor.py:240-245`,
   `variants.py:814-824`, `variants.py:1820-1828`,
   `merge_dla.c:276-296`).
2. Add an opt-in `igraph_fidelity` path for connected classical MDS that:
   selects largest algebraic eigenvalues, uses `sqrt(abs(lambda))`, reverses
   output columns, and preserves the current default behavior for normal dagua
   callers (`embed.py:341-353`, `mds.c:113-132`).
3. Add tiny-graph special cases for igraph parity: `N=1` already matches, `N=2`
   should follow igraph in the opt-in path (`mds.c:71-84`).
4. Make an explicit weighted-graph policy in variant comparison. The most
   conservative diagnosis-preserving choice is to mark current weighted
   comparisons as semantically mismatched until the igraph adapter can pass a
   weighted distance matrix (`classic_competitor.py:1607-1618`,
   `igraph_competitor.py:74-75`, `igraph_competitor.py:167-180`).

Do not try to fully reimplement igraph DLA component merge in the same round as
the eigenvalue/RNG fixes unless the next task is explicitly large. Component
merge is the biggest RMSD lever on disconnected graphs, but it is also the
largest surface area because it brings a stochastic packing algorithm, grid
collision behavior, component ordering, radius normalization, and seed semantics
into a layout that is otherwise closed-form (`mds.c:229-288`,
`merge_dla.c:100-155`, `merge_dla.c:266-298`).

## Assumptions

- I treated `dagua/layout/ops/pipelines/classical_mds.py` plus
  `distance.py`, `embed.py`, and `postprocess.py` as the active dagua ops
  implementation because no standalone `dagua/layout/ops/classical_mds.py`
  exists in this checkout.
- I treated `/home/jtaylor/projects/_references/igraph/src/layout/mds.c` as the
  C reference and `merge_dla.c` as relevant reference code because `mds.c`
  directly calls `igraph_layout_merge_dla()` for disconnected graphs
  (`mds.c:277-280`).
- I did not modify source, run lint, or run tests because this round is
  diagnosis-only and asks for one markdown report.

## Verification

- This report is the only file intentionally created:
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_classical_mds.md`.
- It is designed to exceed the requested 10 KB threshold and includes
  file:line references throughout.
