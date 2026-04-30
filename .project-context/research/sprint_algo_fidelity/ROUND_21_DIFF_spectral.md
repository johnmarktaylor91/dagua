# Round 21 adversarial diff: spectral family

Pairing: dagua `classic_spectral` vs reference `nx_spectral`.

Current mega-run verdict: `spectral_default` is `strong_equivalent`, non-stochastic, 105 cases, RMSD `0.000` in `eval_output/fidelity_report/report.md:84`. This report is intentionally stricter than that verdict and catalogs residual and edge-case divergences, including cases likely outside the current graph subset.

## 1. Files read

Dagua implementation and wiring:

- `dagua/layout/ops/pipelines/spectral.py:1-127` -- composable spectral pipeline and public `layout_spectral_pipeline`.
- `dagua/layout/ops/preprocess.py:899-1094` -- `_build_spectral_adjacency`, `_symmetrize_spectral_adjacency`, `_spectral_laplacian`, and `SpectralPrepareState`.
- `dagua/layout/ops/embed.py:1-47` -- spectral constants, imports, and scipy/numpy boundaries.
- `dagua/layout/ops/embed.py:1124-1281` -- eigenvector selection, dense/sparse eigensolvers, and `SpectralEmbed`.
- `dagua/layout/ops/postprocess.py:798-867` -- spectral rescale/finalization.
- `dagua/layout/_archive/classic/spectral.py:1-370` -- archived classic implementation reached through symlink `dagua/layout/classic/spectral.py`; this mirrors the pipeline behavior and is used by classic tests.
- `dagua/eval/competitors/classic_competitor.py:26-97` -- classic competitor base, seed resolution, and variant parameter forwarding.
- `dagua/eval/competitors/classic_competitor.py:153-183` -- `classic_spectral` registry entry.
- `dagua/eval/competitors/classic_competitor.py:973-1025` -- `ClassicSpectral` adapter.
- `dagua/eval/competitors/networkx_competitor.py:1-173` -- NetworkX graph conversion, position scaling, and `NetworkXSpectral`.
- `dagua/eval/competitors/base.py:16-23` and `dagua/eval/competitors/base.py:100-119` -- competitor result shape and runtime seed convention.
- `dagua/eval/variants.py:781-812` -- spectral variants and reference pairing.
- `dagua/eval/variants.py:1820-1848` and `dagua/eval/variants.py:1870-1873` -- stochastic/heavy metadata for spectral engines.
- `dagua/eval/benchmark.py:60-74` and `dagua/eval/benchmark.py:601-620` -- competitor ordering and NetworkX version key.
- `dagua/layout/ops/state.py:113-165` -- `LayoutProblem` fields, especially `edge_index`, `edge_weights`, and `seed`.
- `dagua/graph.py:69-95`, `dagua/graph.py:337-359`, and `dagua/graph.py:844-848` -- graph topology/weights representation and parallel-edge accumulation.
- `tests/test_classic_new_layouts.py:349-389` -- tests encode Dagua default as symmetric normalized spectral.
- `tests/test_classic_reference_r2.py:783-815` -- reference-style tests also target symmetric normalized Laplacian, not NetworkX spectral.
- `tests/test_pipeline_spectral.py:1-280` -- exact pipeline-vs-archive fidelity tests.
- `tests/test_fa2_ogdf_competitors.py:201-210` -- smoke coverage for `nx_spectral`.

Dagua docs/results:

- `eval_output/fidelity_report/report.md:70-90` -- current verdict table with spectral row.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:29-45` -- sprint context and family status framing.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:77-108` -- measurement infrastructure notes.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:132-198` -- per-round residual/known issue context.

Reference implementation:

- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:42-60` -- `_process_params`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1025-1119` -- `spectral_layout`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1122-1140` -- dense `_spectral`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1143-1166` -- sparse `_sparse_spectral`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1882-1924` -- `rescale_layout`.

Absent file:

- `dagua/layout/ops/spectral.py` does not exist. The spectral op implementation is distributed across `preprocess.py`, `embed.py`, `postprocess.py`, and `pipelines/spectral.py`.

## 2. Overall pipeline structure

Dagua `classic_spectral`:

1. The classic registry maps `"classic_spectral"` to `dagua.layout.ops.pipelines.spectral.layout_spectral_pipeline` with no default params in `dagua/eval/competitors/classic_competitor.py:179-183`.
2. `ClassicSpectral.layout` imports `layout_spectral_pipeline`, passes `graph.edge_index`, `graph.num_nodes`, `graph.node_sizes`, and `seed=self._layout_seed(seed)` in `dagua/eval/competitors/classic_competitor.py:1005-1016`. It does **not** pass `graph.edge_weights`, despite the pipeline supporting them.
3. `layout_spectral_pipeline` validates `num_nodes` and optional `edge_weights`, creates a `LayoutProblem`, forces `ExecutionPlan(device="cpu")`, runs `build_spectral_pipeline(normalization=normalization)`, and returns `final_state.pos` in `dagua/layout/ops/pipelines/spectral.py:62-124`.
4. `build_spectral_pipeline` composes exactly three ops: `SpectralPrepareState`, `SpectralEmbed`, and `SpectralFinalizePositions` in `dagua/layout/ops/pipelines/spectral.py:52-57`.
5. `SpectralPrepareState` handles `N=0` and `N=1`, otherwise builds a scipy CSR adjacency, symmetrizes it, creates the selected Laplacian, and records whether the matrix is symmetric in `dagua/layout/ops/preprocess.py:1076-1094`.
6. `SpectralEmbed` skips if `state.pos` already exists, chooses dense eigensolve when `num_nodes < 500`, sparse eigensolve otherwise, and stores `torch.from_numpy(coordinates)` in `dagua/layout/ops/embed.py:1255-1281`.
7. `SpectralFinalizePositions` moves through CPU float64 NumPy, rescales like NetworkX, then returns `float32` on the inferred output device in `dagua/layout/ops/postprocess.py:853-867`.

NetworkX `nx_spectral` as run by Dagua benchmark:

1. `NetworkXSpectral` is registered with `layout_func = "spectral_layout"` and `layout_kwargs = {"dim": 2}` in `dagua/eval/competitors/networkx_competitor.py:165-173`.
2. `_NetworkXBase.layout_with_variant` converts `DaguaGraph` to `nx.DiGraph`, resolves the NetworkX function, calls `nx.spectral_layout(G, dim=2)`, converts the position dict to a tensor, and returns it in `dagua/eval/competitors/networkx_competitor.py:118-133`.
3. `_graph_to_nx` always uses `nx.DiGraph`, adds integer nodes, and calls `G.add_edge(source, target)` or `G.add_edge(source, target, weight=...)` for each Dagua edge in `dagua/eval/competitors/networkx_competitor.py:33-47`.
4. NetworkX `spectral_layout` processes `center` and `dim`, special-cases `len(G) <= 2`, uses dense eigensolve for `<500` nodes, sparse ARPACK for `>=500`, rescales with `rescale_layout`, adds `center`, and zips positions back to node iteration order in `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1083-1119`.

High-level structure match:

- Both paths build an adjacency, symmetrize directed graphs, compute Laplacian eigenvectors, and rescale to unit extent.
- Both paths use dense eigensolve under 500 nodes and sparse ARPACK at 500+ nodes: Dagua threshold is `SPARSE_EIGEN_THRESHOLD = 500` in `dagua/layout/ops/pipelines/spectral.py:20`; NetworkX uses `if len(G) < 500` to force dense and sparse otherwise in `layout.py:1096-1105`.
- Dagua adds normalization modes and defaults to `"symmetric"` in `dagua/layout/ops/pipelines/spectral.py:23-26` and `dagua/layout/ops/pipelines/spectral.py:62-69`. NetworkX spectral layout uses only the unnormalized Laplacian, documented in `layout.py:1028-1031` and implemented in `layout.py:1133-1135` and `layout.py:1156-1158`.

## 3. Energy / loss / objective

There is no iterative energy minimization, force simulation, or explicit loss term on either side. The objective is spectral embedding through low-frequency Laplacian eigenvectors.

Reference objective:

- NetworkX documentation states that positions are entries of `dim` eigenvectors corresponding to ascending eigenvalues starting from the second one in `layout.py:1028-1031`.
- Dense reference constructs the unnormalized graph Laplacian `L = D - A`, where `D = identity * sum(A, axis=1)`, in `layout.py:1133-1135`.
- Dense reference solves `np.linalg.eig(L)` in `layout.py:1137`, then selects `np.argsort(eigenvalues)[1 : dim + 1]` in `layout.py:1138-1140`.
- Sparse reference constructs `D = dia_array((A.sum(axis=1), 0))` and `L = D - A` in `layout.py:1156-1158`.
- Sparse reference solves the smallest-magnitude eigenpairs with `sp.sparse.linalg.eigsh(L, k=dim+1, which="SM", ncv=ncv)` in `layout.py:1160-1165`.

Dagua objective:

- Dagua builds weighted CSR adjacency with data `edge_weights` or ones in `dagua/layout/ops/preprocess.py:942-947`.
- Dagua symmetrizes with `A + A.T` only if `adjacency - adjacency.T` has nonzeros in `dagua/layout/ops/preprocess.py:950-955`.
- For `normalization == "unnormalized"`, Dagua computes the same `D - A` objective in `dagua/layout/ops/preprocess.py:976-980`.
- For default `normalization == "symmetric"`, Dagua computes `I - D^{-1/2} A D^{-1/2}` in `dagua/layout/ops/preprocess.py:981-987`.
- For `normalization == "random_walk"`, Dagua computes `I - D^{-1} A` in `dagua/layout/ops/preprocess.py:988-994`.
- Dagua then chooses the first nontrivial eigenvectors by absolute eigenvalue tolerance `1.0e-9` in `dagua/layout/ops/embed.py:42` and `dagua/layout/ops/embed.py:1145-1155`.

Objective mismatch:

- Default pairing mismatch is real at source level: Dagua default is symmetric normalized (`dagua/layout/ops/pipelines/spectral.py:23-26`, `dagua/layout/ops/preprocess.py:981-987`), while NetworkX uses unnormalized (`layout.py:1028-1031`, `layout.py:1133-1135`). The current strong-equivalence result means the tested graphs are insensitive after the comparator normalization/alignment, not that the formulas are identical.
- Dagua exposes an unnormalized variant (`dagua/eval/variants.py:803-812`), but that variant has no reference competitor (`None` at `dagua/eval/variants.py:808`) and therefore is not the default `classic_spectral_default` vs `nx_spectral` pairing.
- Existing tests intentionally validate Dagua against symmetric normalized spectral, not NetworkX spectral: `tests/test_classic_new_layouts.py:349-364` and `tests/test_classic_reference_r2.py:783-806`.

## 4. Force / gradient computation

Not applicable.

- Dagua spectral has no optimization loop, optimizer, force accumulation, or gradient descent. Pipeline stages are only preprocessing, eigensolve, and postprocess in `dagua/layout/ops/pipelines/spectral.py:52-57`.
- `layout_spectral_pipeline` accepts `seed` only for interface compatibility and discards it via `_ = seed` in `dagua/layout/ops/pipelines/spectral.py:101`.
- NetworkX spectral similarly has no iteration: it constructs `A`, computes `_spectral` or `_sparse_spectral`, then rescales in `layout.py:1096-1114`.
- There are no force-law signs or gradient directions to compare. The closest equivalent is eigenpair selection and eigenvector sign/rotation ambiguity.

## 5. Initialization

There is no random initialization.

Dagua:

- `layout_spectral_pipeline` documents `seed` as deterministic/interface-only in `dagua/layout/ops/pipelines/spectral.py:81-83` and discards it at `dagua/layout/ops/pipelines/spectral.py:101`.
- `ClassicSpectral.layout` still resolves a seed using `_layout_seed`, but that seed is ignored by the pipeline in `dagua/eval/competitors/classic_competitor.py:29-42` and `dagua/eval/competitors/classic_competitor.py:1011-1016`.
- Trivial initialization/output is explicit: `N=0` gets an empty float32 tensor and `N=1` gets zeros in `dagua/layout/ops/preprocess.py:1076-1081`; finalization repeats this in `dagua/layout/ops/postprocess.py:853-859`.

NetworkX:

- `spectral_layout` has no seed parameter in `layout.py:1025`.
- `_process_params` sets `center = np.zeros(dim)` when absent in `layout.py:51-52`.
- `len(G) <= 2` is initialized by hard-coded positions: empty array, one node at `center`, or two nodes `[zeros(dim), center * 2.0]` in `layout.py:1088-1095`.

Important edge-case initialization divergence:

- With default center zero, NetworkX returns both two-node positions at `[0, 0]` because `center * 2.0` is zero in `layout.py:1091-1095`.
- Dagua does **not** special-case `N=2`; it enters eigensolve and rescales, producing separated points such as `[[-1, 0], [1, 0]]` for one edge through `dagua/layout/ops/preprocess.py:1076-1094`, `dagua/layout/ops/embed.py:1267-1281`, and `dagua/layout/ops/postprocess.py:861-867`.

## 6. Iteration / convergence

No iteration or convergence loop exists in either implementation.

Dense branch:

- Dagua dense branch condition is `problem.num_nodes < self.sparse_threshold` with default `500` in `dagua/layout/ops/embed.py:1267-1273` and `dagua/layout/ops/pipelines/spectral.py:20-26`.
- Dagua dense symmetric branch uses `np.linalg.eigh`; non-symmetric branch uses `np.linalg.eig` in `dagua/layout/ops/embed.py:1167-1172`.
- NetworkX dense branch is forced when `len(G) < 500` via raising `ValueError` in `layout.py:1096-1099`; dense `_spectral` uses `np.linalg.eig(L)` in `layout.py:1137`.

Sparse branch:

- Dagua sparse branch computes `eigen_count = min(num_nodes - 1, max(dim + 4, dim + 1))`; for `dim=2`, this requests up to six eigenpairs in `dagua/layout/ops/embed.py:1181-1184`.
- Dagua ARPACK `ncv` is `min(max((2 * eigen_count) + 1, sqrt(N), eigen_count + 2), N)` through constants in `dagua/layout/ops/embed.py:44-46` and the calculation in `dagua/layout/ops/embed.py:1186-1203`.
- NetworkX sparse branch uses `k = dim + 1`; for `dim=2`, this requests three eigenpairs in `layout.py:1160`.
- NetworkX sparse `ncv = max(2 * k + 1, int(sqrt(nnodes)))` in `layout.py:1161-1164`.

Convergence:

- No tolerance-based convergence criterion exists. Numerical convergence is delegated to NumPy LAPACK or SciPy ARPACK defaults. Neither side passes `tol`, `maxiter`, or `v0` to ARPACK (`dagua/layout/ops/embed.py:1190-1203`; `layout.py:1163-1164`).

## 7. Hyperparameter alignment table

| Parameter / behavior | Dagua default | NetworkX default/reference | Match? | Notes |
|---|---:|---:|---|---|
| Output dimension | fixed `_EMBEDDING_OUTPUT_DIM = 2` in `dagua/layout/ops/embed.py:43` | `dim=2` in `layout.py:1025`; adapter passes `{"dim": 2}` in `networkx_competitor.py:171-172` | Y | Default pairing aligns at 2D. Dagua pipeline does not expose `dim`. |
| Scale inside algorithm | `_SPECTRAL_RESCALE_UNIT` passed as 1.0 through `postprocess.py:798-807` and `postprocess.py:863-866` | `scale=1` in `layout.py:1025`, applied at `layout.py:1113` | Y | Algorithmic scale matches. |
| Adapter output scaling | Dagua returns direct layout tensor in `classic_competitor.py:1017-1018` | `_nx_pos_to_tensor` multiplies every coordinate by `500.0` in `networkx_competitor.py:50-58` | N | Benchmark comparators appear scale-normalized; raw tensor users see a 500x difference. |
| Center | Dagua always centers at mean zero in `postprocess.py:803-807` | `center=None` becomes zeros in `layout.py:51-52`; final `+ center` at `layout.py:1113` | Y for default | Dagua has no nonzero-center option. |
| Weight attribute | Pipeline accepts `edge_weights` in `pipelines/spectral.py:67` and `preprocess.py:942-947` | `weight="weight"` in `layout.py:1025` and adapter writes weights in `networkx_competitor.py:43-44` | Partial | `ClassicSpectral.layout` does not forward `graph.edge_weights` in `classic_competitor.py:1011-1016`. |
| Laplacian normalization | `"symmetric"` default in `pipelines/spectral.py:23-26` and `pipelines/spectral.py:68` | unnormalized only, `D - A` in `layout.py:1133-1135` and `layout.py:1156-1158` | N | Largest source-level mismatch. Dagua unnormalized mode exists but is not default reference-paired. |
| Directed graph treatment | Builds directed CSR then symmetrizes by `A + A.T` if needed in `preprocess.py:950-955` | Converts to DiGraph, then `A + A.T` when directed in `layout.py:1101-1103` and `layout.py:1108-1110` | Mostly Y | Both double reciprocal directed edges. Multiplicity differs because NetworkX adapter uses `DiGraph`. |
| Dense threshold | sparse if `N >= 500` via `SPARSE_EIGEN_THRESHOLD = 500` and `num_nodes < threshold` in `pipelines/spectral.py:20-26`, `embed.py:1267-1274` | sparse if `len(G) >= 500` in `layout.py:1096-1105` | Y | Boundary aligns. |
| Dense eigensolver | `np.linalg.eigh` for symmetric; `np.linalg.eig` for random-walk in `embed.py:1167-1172` | Always `np.linalg.eig` in `layout.py:1137` | Partial | For unnormalized/symmetric real Laplacians, `eigh` is numerically more stable but not bit-identical to NetworkX. |
| Sparse eigensolver | `eigsh` for symmetric, `eigs` for random-walk in `embed.py:1190-1203` | `eigsh` only in `layout.py:1163-1164` | Y for unnormalized/symmetric | Dagua random-walk variant has no NetworkX counterpart. |
| Sparse eigenpair count | `max(dim + 4, dim + 1)` capped by `N-1` in `embed.py:1181-1184` | `k = dim + 1` in `layout.py:1160` | N | Dagua asks for extra eigenpairs, then filters. |
| Sparse `ncv` | based on Dagua `eigen_count`, padding, and cap at `N` in `embed.py:1186-1203` | `max(2 * k + 1, sqrt(N))` in `layout.py:1161-1164` | N | Usually close, not identical. |
| Eigenvector selection | skip eigenvalues with `abs(lambda) <= 1e-9` in `embed.py:42`, `embed.py:1145-1155` | blindly skip sorted index 0 and use next `dim` in `layout.py:1138-1140`, `layout.py:1165-1166` | N | Major difference on disconnected graphs with multiple zero eigenvalues. |
| Empty graph | Dagua returns empty `[0,2]` float32 in `preprocess.py:1076-1078` and `postprocess.py:853-856` | NetworkX returns empty dict from empty array in `layout.py:1088-1095` | Y at adapter tensor level | Adapter creates zeros tensor of shape `[0,2]` in `networkx_competitor.py:50-58`. |
| One node | Dagua zeros in `preprocess.py:1079-1081` and `postprocess.py:857-859` | NetworkX returns `center`, default zero, in `layout.py:1091-1092` | Y for default center | Dagua has no nonzero-center equivalent. |
| Two nodes | Dagua eigensolves and separates | NetworkX returns zeros/default-center pair in `layout.py:1093-1095` | N | Clear edge-case bug relative to NetworkX. |
| RNG / seed | ignored in `pipelines/spectral.py:101` | no seed parameter in `layout.py:1025` | Y | Both deterministic subject to LAPACK/ARPACK sign/ordering. |
| Output dtype | Dagua final `float32` in `postprocess.py:864-866` | NetworkX NumPy float64, adapter writes into default `torch.zeros` float32 in `networkx_competitor.py:50-58` | Y at adapter level | Raw NetworkX returns float64 arrays. |

## 8. Edge cases

Self-loops:

- NetworkX dense unnormalized Laplacian places self-loop weight in both degree diagonal and adjacency diagonal, so it cancels out of `D - A` (`layout.py:1133-1135`). Sparse does the same (`layout.py:1156-1158`).
- Dagua unnormalized mode also cancels self-loops through `degree_matrix - adjacency` in `preprocess.py:976-980`.
- Dagua symmetric normalized mode does **not** fully cancel a self-loop in the same way because the self-loop contributes to degree used by `D^{-1/2}` and to the normalized adjacency diagonal in `preprocess.py:981-987`. This can change coordinates relative to NetworkX on graphs with self-loops.
- For `N=2`, the NetworkX `len(G) <= 2` special case masks self-loop behavior and returns default zeros (`layout.py:1088-1095`); Dagua still separates two nodes.

Multi-edges / parallel edges:

- Dagua graph construction preserves pending parallel edges in `_pending_edges` and concatenates them into `edge_index` in `dagua/graph.py:337-343`; weights are likewise accumulated or backfilled in `dagua/graph.py:345-359`.
- Dagua spectral adjacency is scipy CSR from `(data, (rows, cols))` in `preprocess.py:947`; scipy sums duplicate `(row, col)` entries, so parallel edges increase effective weight.
- The NetworkX adapter uses `nx.DiGraph` in `networkx_competitor.py:35`, and repeated `G.add_edge(source, target)` calls in `networkx_competitor.py:40-46` collapse parallel edges. With weights, later duplicate writes overwrite the edge attribute rather than summing.
- Therefore Dagua treats unweighted duplicates as stronger connections, while `nx_spectral` adapter treats them as one edge. This is not a NetworkX `spectral_layout` limitation: the adapter chooses `DiGraph` rather than `MultiDiGraph`.

Disconnected components:

- NetworkX selects sorted eigenvectors `[1 : dim + 1]` after only skipping the first eigenvalue in `layout.py:1138-1140` and `layout.py:1165-1166`. For `C` connected components, the Laplacian has `C` zero eigenvalues, so NetworkX may use additional component-indicator zero modes as drawing dimensions.
- Dagua filters all eigenvalues with absolute value `<= 1e-9` and only takes nontrivial eigenvectors in `embed.py:1145-1155`. This intentionally avoids component-indicator zero modes but diverges from NetworkX. On disconnected graphs this is a high-impact semantic mismatch.
- Existing Dagua pipeline tests check disconnected graphs against the archived Dagua implementation, not NetworkX, in `tests/test_pipeline_spectral.py:200-215`.

Weighted edges:

- Dagua pipeline supports `edge_weights` all the way into adjacency data in `pipelines/spectral.py:67`, `preprocess.py:942-947`, and `state.py:158`.
- The archived classic implementation also supports `edge_weights` in `dagua/layout/_archive/classic/spectral.py:288-295` and uses them in `_build_adjacency` at `dagua/layout/_archive/classic/spectral.py:107-111`.
- But `ClassicSpectral.layout` does not pass `graph.edge_weights` into `layout_spectral` in `classic_competitor.py:1011-1016`. That means the competitor pairing ignores Dagua graph weights on the Dagua side while `_graph_to_nx` forwards weights to NetworkX in `networkx_competitor.py:39-44`.
- If weighted spectral graphs enter the benchmark, `classic_spectral` vs `nx_spectral` will be biased even before the normalized-vs-unnormalized issue.

Empty graph:

- Dagua returns an empty tensor at preprocess/finalize (`preprocess.py:1076-1078`, `postprocess.py:853-856`).
- NetworkX returns an empty mapping after `pos = np.array([])` and `dict(zip(G, pos))` in `layout.py:1088-1095`.
- Adapter conversion of an empty dict returns `torch.zeros(0, 2)` in `networkx_competitor.py:50-58`. This should align.

One-node graph:

- Dagua and NetworkX both return zero for default center (`preprocess.py:1079-1081`; `layout.py:1091-1092`).
- Dagua final dtype is float32 (`postprocess.py:857-859`), while raw NetworkX is NumPy float64; adapter returns float32 because `torch.zeros` defaults to float32 in `networkx_competitor.py:52`.

Two-node graph:

- NetworkX special-cases all two-node graphs to `[zeros(dim), center * 2.0]` in `layout.py:1093-1095`. With default center zero, both nodes are coincident at zero.
- Dagua spectral eigensolves `N=2`, then rescales to unit extent. This is the clearest direct edge-case mismatch.

## 9. Numerical precision

Dagua precision boundaries:

- Dagua builds scipy adjacency as `np.float64` even if `edge_weights` are torch float32 in `preprocess.py:942-947`.
- Dagua dense eigensolve happens in NumPy float64 via `laplacian.toarray()` and `np.linalg.eigh/eig` in `embed.py:1167-1172`.
- Dagua stores raw coordinates with `torch.from_numpy(coordinates)` in `embed.py:1280`; this is float64 until finalization.
- Finalization explicitly converts to CPU float64 NumPy before rescaling in `postprocess.py:861-865`, then converts to float32 on output in `postprocess.py:864-866`.

NetworkX precision boundaries:

- NetworkX sparse adjacency is built with `dtype="d"` (float64) in `layout.py:1100`.
- NetworkX dense adjacency uses `nx.to_numpy_array(G, weight=weight)` with NumPy default float dtype in `layout.py:1107`.
- NetworkX dense eigensolve uses `np.linalg.eig`, returning float64/complex128 as applicable, then `np.real` in `layout.py:1137-1140`.
- NetworkX rescale mutates/returns NumPy float arrays in `layout.py:1918-1924`.
- Dagua's NetworkX adapter converts to torch float32 by assigning float64 NumPy values into `torch.zeros(num_nodes, 2)` in `networkx_competitor.py:50-58`.

Residual numerical differences:

- `np.linalg.eigh` vs `np.linalg.eig` can choose different bases in repeated or nearly repeated eigenspaces. Dagua uses `eigh` for symmetric Laplacians in `embed.py:1167-1172`; NetworkX dense `_spectral` uses generic `eig` in `layout.py:1137`.
- Dagua's eigenvalue tolerance (`1e-9`) can remove very small eigenvalues that NetworkX would retain after the first sorted eigenvalue, especially in disconnected or near-disconnected graphs (`embed.py:1145-1155`; `layout.py:1138-1140`).
- Sparse ARPACK is not seeded (`embed.py:1190-1203`; `layout.py:1163-1164`), so sign and basis orientation remain library-dependent. Procrustes/sign-invariant comparison hides much of this, but raw coordinates may differ.
- Dagua requests extra sparse eigenpairs (`dim + 4`) and then filters; NetworkX requests exactly `dim + 1` (`embed.py:1181-1184`; `layout.py:1160`). Extra eigenpairs can improve robustness around zero modes but changes ARPACK subspace construction and summation/order effects.

## 10. RNG semantics

- Dagua `classic_spectral` is marked non-stochastic in `dagua/eval/variants.py:1820-1828`; `nx_spectral` is also marked non-stochastic in `dagua/eval/variants.py:1846-1848`.
- `layout_spectral_pipeline` accepts `seed` but discards it (`dagua/layout/ops/pipelines/spectral.py:81-83`, `dagua/layout/ops/pipelines/spectral.py:101`).
- `NetworkXSpectral` has no `seed` in `layout_kwargs` (`networkx_competitor.py:171-172`), and `_NetworkXBase.layout_with_variant` only forwards a seed if `"seed"` is already in `layout_kwargs` (`networkx_competitor.py:125-130`).
- Reference `networkx.spectral_layout` has no seed parameter in `layout.py:1025`.
- Therefore Dagua's torch seed does **not** produce the same sequence as the reference RNG because neither side uses an RNG sequence for spectral layout. The only nondeterminism risk is numerical library eigensolver behavior, not Python/NumPy/torch random draws.

## 11. Edge-case bugs

1. **Default Laplacian mismatch.** `classic_spectral_default` pairs Dagua's default `"symmetric"` normalization (`pipelines/spectral.py:23-26`, `pipelines/spectral.py:68`) against NetworkX's unnormalized `D - A` (`layout.py:1028-1031`, `layout.py:1133-1135`). This is a semantic mismatch even if current RMSD is zero after alignment on the suite.
2. **Two-node behavior mismatch.** NetworkX returns coincident default-center points for every two-node graph (`layout.py:1093-1095`), while Dagua eigensolves and separates nodes because only `N=0` and `N=1` are special-cased (`preprocess.py:1076-1081`).
3. **Disconnected zero-mode mismatch.** NetworkX skips only the first eigenvalue (`layout.py:1138-1140`, `layout.py:1165-1166`); Dagua skips all eigenvalues under `1e-9` (`embed.py:1145-1155`). This changes component placement for disconnected graphs.
4. **Parallel-edge collapse in reference adapter.** Dagua preserves and sums duplicate edges (`graph.py:337-359`, `preprocess.py:947`), but `_graph_to_nx` uses `nx.DiGraph` and `G.add_edge`, which collapses duplicates (`networkx_competitor.py:35-46`). The adapter is not faithful to Dagua multigraph inputs.
5. **ClassicSpectral drops weights.** The pipeline accepts `edge_weights`, but `ClassicSpectral.layout` never forwards `graph.edge_weights` (`classic_competitor.py:1011-1016`). NetworkX adapter does forward weights (`networkx_competitor.py:39-44`). Weighted graph comparisons are therefore not aligned.
6. **Dense eigensolver mismatch.** Dagua's symmetric path uses `np.linalg.eigh` (`embed.py:1167-1169`), while NetworkX uses `np.linalg.eig` even for symmetric dense Laplacians (`layout.py:1137`). This is numerically reasonable but not exact-fidelity.
7. **Sparse `k/ncv` mismatch.** Dagua requests extra eigenpairs and different `ncv` (`embed.py:1181-1203`); NetworkX requests `dim + 1` and its own `ncv` formula (`layout.py:1160-1164`). This can alter sparse layouts at 500+ nodes.
8. **Adapter scale mismatch.** NetworkX adapter multiplies positions by 500 (`networkx_competitor.py:50-58`) while Dagua returns unit-scale positions (`postprocess.py:861-866`). This is likely intentionally normalized downstream, but raw result consumers see different units.

## 12. Ranked fix list

1. **Add a NetworkX-fidelity mode/default for the reference-paired variant: unnormalized Laplacian.**
   - Evidence: Dagua default `"symmetric"` in `pipelines/spectral.py:23-26`; NetworkX unnormalized in `layout.py:1133-1135` and `layout.py:1156-1158`.
   - Proposed fix: either change `classic_spectral_default` variant params to `{"normalization": "unnormalized"}` for the `nx_spectral` pairing, or introduce `classic_spectral_nx_default` while preserving current public default.
   - Expected RMSD impact: highest on irregular weighted/degree-skewed graphs; small on regular/current suite after Procrustes.
   - Size estimate: S if variant-only, M if public default compatibility and tests are updated.

2. **Mirror NetworkX `len(G) <= 2` special case in a fidelity mode.**
   - Evidence: NetworkX special-cases `N=0/1/2` in `layout.py:1088-1095`; Dagua only special-cases `N=0/1` in `preprocess.py:1076-1081` and finalization in `postprocess.py:853-859`.
   - Proposed fix: before eigensolve, if `num_nodes == 2` and NetworkX fidelity is requested, return `[[0, 0], [0, 0]]` for default center/2D. If nonzero centers are ever exposed, match `[zeros(dim), center * 2.0]`.
   - Expected RMSD impact: total for two-node graphs; none for larger graphs.
   - Size estimate: S.

3. **Match NetworkX eigenvector selection for the `nx_spectral` fidelity path.**
   - Evidence: Dagua filters all near-zero eigenvalues in `embed.py:1145-1155`; NetworkX uses sorted slice `[1 : dim + 1]` or `[1:k]` in `layout.py:1138-1140` and `layout.py:1165-1166`.
   - Proposed fix: add selection mode `skip_first` for NetworkX fidelity, leaving `skip_near_zero` as current robust Dagua behavior.
   - Expected RMSD impact: high on disconnected graphs and near-disconnected graphs; low on connected graphs with one zero eigenvalue.
   - Size estimate: M because it touches embed API and tests.

4. **Preserve/align weighted edges in `ClassicSpectral.layout`.**
   - Evidence: `LayoutProblem` carries `edge_weights` in `state.py:158`; pipeline accepts `edge_weights` in `pipelines/spectral.py:67` and validates it in `pipelines/spectral.py:105-111`; `ClassicSpectral.layout` omits it in `classic_competitor.py:1011-1016`.
   - Proposed fix: pass `edge_weights=graph.edge_weights` from `ClassicSpectral.layout`.
   - Expected RMSD impact: high on weighted graph cases; zero on unweighted cases.
   - Size estimate: S plus regression test.

5. **Handle parallel edges consistently in the NetworkX adapter or Dagua fidelity path.**
   - Evidence: Dagua preserves duplicate edges in `graph.py:337-343` and CSR sums duplicates at `preprocess.py:947`; NetworkX adapter uses `nx.DiGraph` in `networkx_competitor.py:35-46`.
   - Proposed fix options: use `nx.MultiDiGraph` and verify `to_numpy_array` sums weights as desired, or pre-aggregate Dagua edges before both sides so both see the same matrix.
   - Expected RMSD impact: high on multiedge graphs such as bundled/parallel cases; none on simple graphs.
   - Size estimate: M because it affects all NetworkX competitors if changed globally.

6. **Match NetworkX sparse eigenpair count and `ncv` under fidelity mode.**
   - Evidence: Dagua `eigen_count = max(dim + 4, dim + 1)` in `embed.py:1181-1184`; NetworkX `k = dim + 1` in `layout.py:1160`; Dagua `ncv` formula differs in `embed.py:1186-1203` from NetworkX `layout.py:1161-1164`.
   - Proposed fix: add `extra_eigenpairs=0`/`networkx_sparse=True` path for spectral fidelity.
   - Expected RMSD impact: moderate at `N>=500`, especially disconnected or clustered graphs.
   - Size estimate: M.

7. **Use `np.linalg.eig` instead of `eigh` for dense NetworkX fidelity.**
   - Evidence: Dagua symmetric dense branch uses `eigh` in `embed.py:1167-1169`; NetworkX dense branch uses `eig` in `layout.py:1137`.
   - Proposed fix: add eigensolver mode for fidelity. Keep `eigh` for the normal Dagua path because it is mathematically appropriate for symmetric Laplacians.
   - Expected RMSD impact: low for simple connected graphs, moderate in repeated eigenspaces where bases can rotate.
   - Size estimate: S-M.

8. **Remove or gate NetworkX adapter's hard-coded 500x coordinate scale for spectral.**
   - Evidence: `_nx_pos_to_tensor` multiplies by `500.0` in `networkx_competitor.py:50-58`; Dagua final unit-scale output comes from `postprocess.py:861-866`; NetworkX algorithm default scale is already `1` in `layout.py:1025`.
   - Proposed fix: either make scaling comparator-level instead of adapter-level, or add per-competitor scale policy so spectral/KK/FR can be compared raw when needed.
   - Expected RMSD impact: high for raw RMSD, likely zero for normalized Procrustes metrics.
   - Size estimate: M because it impacts benchmark history and possibly multiple adapters.

## 13. Recommended Round 22+ fix scope

Recommended bundle for one follow-up round: implement a **NetworkX fidelity mode** for spectral rather than changing the public Dagua default.

Scope:

1. Add a `fidelity_mode: Optional[str] = None` or narrower `networkx_fidelity: bool = False` parameter to `layout_spectral_pipeline` and the underlying ops.
2. Under NetworkX fidelity:
   - use unnormalized Laplacian (`preprocess.py:979-980`);
   - special-case `N <= 2` like NetworkX (`layout.py:1088-1095`);
   - select eigenvectors by sorted slice `[1 : dim + 1]` instead of tolerance filtering (`layout.py:1138-1140`, `layout.py:1165-1166`);
   - use sparse `k = dim + 1` and NetworkX `ncv` formula (`layout.py:1160-1164`);
   - optionally use dense `np.linalg.eig` for exact dense fidelity (`layout.py:1137`).
3. Add a variant such as `classic_spectral_nx_fidelity` paired to `nx_spectral`, rather than rewriting `classic_spectral_default`, because current tests and public behavior explicitly expect symmetric normalized spectral (`tests/test_classic_new_layouts.py:349-389`, `tests/test_classic_reference_r2.py:783-815`).
4. Separately fix `ClassicSpectral.layout` to pass `graph.edge_weights` because this is a correctness bug independent of fidelity mode (`classic_competitor.py:1011-1016`).
5. Defer global NetworkX adapter multigraph handling unless Round 22 is scoped to all NetworkX competitors; changing `_graph_to_nx` from `DiGraph` at `networkx_competitor.py:35-46` can affect `nx_spring` and `nx_kamada_kawai` too.

This bundle targets the highest-impact source-level mismatches while preserving the current Dagua spectral default that existing tests assert. The likely smallest landing plan is: variant/fidelity mode first, then a separate adapter/weights cleanup round if weighted or multiedge cases appear in the spectral residual set.
