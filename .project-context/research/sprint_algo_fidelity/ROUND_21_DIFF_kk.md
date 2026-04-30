# Round 21 Adversarial Diff: `classic_kk` vs `nx_kamada_kawai`

Diagnosis-only report for the dagua `classic_kk` family against the reference
NetworkX `kamada_kawai_layout` implementation.

Current measured verdict is already `strong_equivalent`: the fidelity report
lists `kk_steps100`, `kk_steps300`, and `kk_steps1000` with median Procrustes
RMSD `0.000` across 103 OK graphs (`eval_output/fidelity_report/report.md:47-49`).
This report focuses on exact implementation parity and latent divergences that
can be hidden by Procrustes normalization, by unweighted benchmark graphs, or by
the adapter layer.

## 1. Files read

Dagua side:

- `AGENTS.md:1-220` for project-level quality and task constraints.
- `dagua/layout/AGENTS.md:1-122` for ops/pipeline architecture and RNG policy.
- `dagua/layout/ops/pipelines/kk.py:1-287` for pipeline composition, public
  wrapper arguments, direction-orientation logic, and early returns.
- `dagua/layout/ops/distance.py:1-140`, `dagua/layout/ops/distance.py:560-762`
  for KK all-pairs shortest-path construction and unreachable fill.
- `dagua/layout/ops/init.py:1-70`, `dagua/layout/ops/init.py:520-630` for KK
  initialization.
- `dagua/layout/ops/optimize.py:1-1120` for the NetworkX-style objective,
  analytic gradient, SciPy L-BFGS-B call, dtype conversion, maxiter plumbing,
  callback traces, and constants.
- `dagua/layout/ops/postprocess.py:1-80`, `dagua/layout/ops/postprocess.py:720-795`
  for final rescaling, dtype/device conversion, and trace movement.
- `dagua/layout/ops/graph_utils.py:1-175`,
  `dagua/layout/ops/graph_utils.py:300-430` for BFS, Dijkstra, directed
  adjacency, duplicate-edge handling, output device choice, and rescale helper.
- `dagua/layout/ops/state.py:130-190` for `LayoutProblem` fields, including
  direction, `edge_weights`, and seed.
- `dagua/layout/ops/base.py:1-120` for the op/pipeline interface context.
- `dagua/eval/variants.py:360-456`, `dagua/eval/variants.py:1800-1860` for KK
  variant definitions and stochasticity flags.
- `dagua/eval/competitors/classic_competitor.py:140-180`,
  `dagua/eval/competitors/classic_competitor.py:620-720`, and
  `dagua/eval/competitors/classic_competitor.py:1580-1630` for the
  `classic_kk` adapter, default steps, variant plumbing, edge-weight forwarding,
  and direction-orientation defaults.
- `dagua/eval/competitors/networkx_competitor.py:1-173` for conversion to a
  NetworkX `DiGraph`, layout invocation, missing `variant_param_names`, and
  adapter scaling by 500.
- `dagua/graph.py:320-370`, `dagua/graph.py:830-855`,
  `dagua/graph.py:1580-1690` for Dagua edge storage, weight dtype, NetworkX
  import semantics, and edge-index import semantics.
- `scripts/algo_fidelity_panel.py:23-78` and
  `scripts/algo_fidelity_cross.py:245-290` for the Procrustes/fidelity
  normalization that explains why scale and reflection divergences can disappear
  in the reported RMSD.
- `dagua/eval/report.py:811-835` for report rendering normalization.
- `eval_output/fidelity_report/report.md:1-80` for the current KK verdict.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:1-220`
  for sprint context and accepted-residual methodology.

Reference side:

- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:120-235`
  for `random_layout` and `circular_layout` initialization behavior.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:876-1022`
  for `kamada_kawai_layout`, `_kamada_kawai_solve`, and `_kamada_kawai_costfn`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1882-1930`
  for `rescale_layout`.
- Installed versions read via Python: NetworkX `3.6.1`, SciPy `1.17.1`.

File-location note: `dagua/layout/ops/kk.py` does not exist in this checkout
(`test -f dagua/layout/ops/kk.py` returned exit status 1). KK logic is in the
pipeline plus shared ops listed above.

## 2. Overall pipeline structure

NetworkX reference flow:

1. `kamada_kawai_layout` normalizes graph input and center with
   `_process_params` (`layout.py:941`).
2. It computes `nNodes = len(G)` and returns `{}` immediately for the empty
   graph (`layout.py:941-944`).
3. If no distance dict is supplied, it computes `dict(nx.shortest_path_length(G,
   weight=weight))` (`layout.py:946-947`).
4. It initializes an `nNodes x nNodes` distance matrix with `1e6`, then fills
   known distances by iterating graph node order for rows and columns
   (`layout.py:948-956`).
5. If no initial positions are supplied, it uses random layout for `dim >= 3`,
   circular layout for `dim == 2`, or linear layout for `dim == 1`
   (`layout.py:958-965`).
6. It calls `_kamada_kawai_solve(dist_mtx, pos_arr, dim)`, which uses SciPy
   `optimize.minimize(..., method="L-BFGS-B", jac=True)` with no explicit
   options (`layout.py:967`, `layout.py:978-997`).
7. It rescales optimized positions with `rescale_layout(pos, scale=scale) +
   center`, zips them back into graph node order, and optionally stores node
   attributes (`layout.py:969-975`).

Dagua pipeline flow:

1. `layout_kk_pipeline` validates `num_nodes`, `steps`, `trace_every`, `solver`,
   `edge_weights`, and optional `pos` (`dagua/layout/ops/pipelines/kk.py:232-250`).
2. It chooses an output device from input tensors (`kk.py:252`;
   `graph_utils.py:136-158`) and returns a float32 empty or single-node tensor
   before building a pipeline (`kk.py:253-258`).
3. It constructs `LayoutProblem` with `edge_index`, `num_nodes`, `node_sizes`,
   optional `edge_weights`, and `direction` (`kk.py:260-266`).
4. It stashes caller-supplied initial positions under `state.extras["kk_initial_pos"]`
   (`kk.py:267-269`) and forces execution plan device to CPU (`kk.py:270`).
5. `build_kk_pipeline` composes `FixedSteps`, `KamadaKawaiAllPairsShortestPaths`,
   `KamadaKawaiInitializePositions`, `LBFGSStep`, and
   `KamadaKawaiFinalizePositions` (`kk.py:153-165`).
6. The distance op constructs directed adjacency from `edge_index`, computes BFS
   or Dijkstra rows, replaces unreachable entries with `1e6`, and writes a
   float64 distance matrix (`distance.py:731-762`).
7. The init op uses caller positions when present, otherwise it creates the same
   2D circular seed formula, including float32 theta, and rescales it
   (`init.py:612-630`).
8. `LBFGSStep` converts positions and distances to CPU float64 NumPy arrays,
   builds `inverse_distances = 1 / (distance_matrix + eye * 1e-3)`, calls SciPy
   `optimize.minimize` with `method="L-BFGS-B"` and `jac=True`, and optionally
   adds `options={"maxiter": steps}` (`optimize.py:999-1038`).
9. `KamadaKawaiFinalizePositions` rescales with the torch helper, casts to
   float32 on output device, and moves traces (`postprocess.py:786-795`).
10. The classic competitor path additionally enables orientation flipping by
    default (`classic_competitor.py:664-672` and `classic_competitor.py:1609-1613`),
    while direct pipeline calls default `orient_to_direction=False`
    (`kk.py:180-181`, `kk.py:213-216`).

Structural verdict: the core 2D solve is intentionally a line-for-line port of
NetworkX. The main structural differences are (a) the dagua public family
introduces a finite `steps` budget where NetworkX uses SciPy defaults, (b) the
dagua evaluation adapter can flip the result after solve, (c) dagua returns
torch float32 tensors while NetworkX returns NumPy arrays that the adapter stores
as torch float32 after multiplying by 500, and (d) dagua supports direct
`edge_index`/`edge_weights` input instead of a NetworkX graph object.

## 3. Energy / loss / objective

Both implementations minimize the same KK stress-like objective in the 2D path:

`E = 0.5 * sum_ij (||x_i - x_j|| * invdist_ij - 1)^2 + 0.5 * meanweight * ||sum_i x_i||^2`

NetworkX:

- Inverse distances are built as `1 / (dist_mtx + eye * 1e-3)` with
  `meanwt = 1e-3` (`layout.py:986-988`).
- Pairwise deltas are `pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]`
  (`layout.py:1005`).
- Separations use `np.linalg.norm(delta, axis=-1)` (`layout.py:1006`).
- Unit directions divide by `nodesep + eye * 1e-3` (`layout.py:1007`).
- Offset is `nodesep * invdist - 1.0`, and diagonal offsets are set to 0
  (`layout.py:1009-1010`).
- Cost is `0.5 * np.sum(offset**2)` (`layout.py:1012`).
- Centering cost is `0.5 * meanweight * np.sum(sumpos**2)` where
  `sumpos = np.sum(pos_arr, axis=0)` (`layout.py:1017-1020`).

Dagua:

- Constants are `DISTANCE_EPSILON = 1.0e-3` and `CENTERING_WEIGHT = 1.0e-3`
  (`optimize.py:23-24`).
- `inverse_distances = 1.0 / (distance_matrix + np.eye(...) * DISTANCE_EPSILON)`
  (`optimize.py:1021-1023`).
- Pairwise deltas, norm, direction denominator, offset, diagonal zeroing, cost,
  and centering term are implemented in `_kamada_kawai_costfn`
  (`optimize.py:57-84`).
- The dagua formula uses the same `0.5 * sum(offset**2)` at `optimize.py:68`
  and the same parabolic mean-position penalty at `optimize.py:81-83`.

Objective parity verdict: exact for the objective when `dim == 2`, the distance
matrix matches, and input positions match. The only objective-level latent
divergence is distance-matrix construction from graph semantics, not the cost
function itself.

Distance/objective caveats:

- NetworkX obtains distances through `nx.shortest_path_length(G, weight=weight)`
  (`layout.py:946-947`). In the eval adapter, `G` is a `nx.DiGraph`
  (`networkx_competitor.py:35-47`), so directed distances are used.
- Dagua builds a directed adjacency list (`distance.py:741-745`;
  `graph_utils.py:360-400`), so directedness matches the eval adapter.
- Duplicate edges are collapsed to the minimum weight in dagua
  (`graph_utils.py:396-398`). NetworkX `DiGraph.add_edge` overwrites an existing
  edge attribute when the same `(source, target)` pair is added repeatedly
  (`networkx_competitor.py:40-46` creates a plain `DiGraph`, not a `MultiDiGraph`).
  Therefore weighted duplicate-edge graphs can diverge if later duplicate weights
  are not the minimum.

## 4. Force / gradient computation

This family does not use an iterative physical force update in the FR sense. It
uses a full analytic gradient passed to L-BFGS-B.

NetworkX gradient:

- Direction tensor: `direction = np.einsum("ijk,ij->ijk", delta, 1 /
  (nodesep + eye * 1e-3))` (`layout.py:1005-1007`).
- Gradient: `np.einsum("ij,ij,ijk->ik", invdist, offset, direction) -
  np.einsum("ij,ij,ijk->jk", invdist, offset, direction)` (`layout.py:1013-1015`).
- Centering gradient: `grad += meanweight * sumpos` (`layout.py:1017-1020`).
- Flattened gradient returned with cost (`layout.py:1022`).

Dagua gradient:

- Direction tensor is identical, using `np_module.einsum` and
  `DISTANCE_EPSILON` (`optimize.py:57-63`).
- Gradient einsum terms are identical (`optimize.py:69-79`).
- Centering gradient is identical (`optimize.py:81-83`).
- Flattened gradient returned with cost (`optimize.py:84`).

Gradient parity verdict: exact in formula and einsum index order. Summation
order is also effectively identical because both call NumPy `einsum` with the
same operands after conversion to float64 NumPy arrays.

## 5. Initialization

NetworkX 2D initialization:

- If `pos` is `None` and `dim == 2`, NetworkX calls `circular_layout(G, dim=dim)`
  (`layout.py:958-963`).
- `circular_layout` returns center for a one-node graph (`layout.py:194-198`).
- For two or more nodes it creates `theta = np.linspace(0, 1, len(G) + 1)[:-1]
  * 2 * np.pi`, casts theta to `np.float32`, computes cos/sin, rescales, and
  zips to graph order (`layout.py:199-206`).
- `rescale_layout` subtracts axis means and scales by the largest absolute
  coordinate (`layout.py:1918-1924`).

Dagua 2D initialization:

- `layout_kk_pipeline` documents that the seed is accepted but not consumed for
  2D classic KK because circular initialization is deterministic (`kk.py:197-199`);
  it explicitly discards the seed with `_ = seed` (`kk.py:232`).
- Empty and single-node graphs return before the pipeline: empty is
  `torch.empty((0, 2), dtype=torch.float32)` and single is `torch.zeros((1, 2),
  dtype=torch.float32)` (`kk.py:252-258`).
- The init op handles empty and single-node states similarly in float64
  (`init.py:605-610`) if it is reached.
- Caller-supplied `kk_initial_pos` is converted to float64 without rescaling
  (`init.py:612-621`).
- Fallback circular initialization uses `np.linspace`, casts theta to
  `np.float32`, computes cos/sin, casts coordinates to float64, then torch
  rescales (`init.py:623-630`).
- Dagua rescale helper subtracts the torch mean and scales by max abs coordinate
  (`graph_utils.py:408-427`), matching NetworkX `rescale_layout`
  (`layout.py:1918-1924`) modulo NumPy-vs-torch reduction details.

Initialization parity verdict: exact for default 2D multi-node unweighted
benchmarks except for tiny NumPy-vs-torch reduction/rounding differences after
the same float32-theta seed. One-node and empty return values are semantically
equivalent for the eval adapters (`{}`/center vs empty tensor; center vs zero
tensor), and Procrustes-style comparison usually ignores absolute center.

Initialization divergences:

- NetworkX supports `dim >= 3` by using `random_layout` (`layout.py:958-961`);
  dagua `layout_kk_pipeline` is hard-coded to validate optional `pos` as
  `(num_nodes, 2)` and returns `[N, 2]` tensors (`kk.py:249-250`, `kk.py:170-182`).
- NetworkX supports `dim == 1` linear initialization (`layout.py:963-965`);
  dagua does not expose 1D.
- NetworkX accepts `pos` as a node-keyed dict (`layout.py:898-901`,
  `layout.py:965`); dagua accepts a positional tensor (`kk.py:205-207`,
  `kk.py:249-250`).

## 6. Iteration / convergence

NetworkX:

- `_kamada_kawai_solve` calls SciPy `sp.optimize.minimize` with method
  `"L-BFGS-B"`, `jac=True`, and no `options` argument (`layout.py:989-995`).
- Therefore max iterations, function tolerance, projected-gradient tolerance,
  line-search budget, and stopping behavior are SciPy defaults for the installed
  version.

Dagua:

- `LBFGSStepConfig.maxiter` defaults to `None`, and the docstring says `None`
  and `0` leave SciPy defaults unchanged (`optimize.py:905-923`).
- The pipeline wrapper default is `steps: Optional[int] = None`, documented as
  leaving `maxiter` unset to match classic KK (`kk.py:121-131`, `kk.py:170-196`).
- However the `classic_kk` competitor default is `steps=300`
  (`classic_competitor.py:159-163`), and `ClassicKK.layout` passes `steps=300`
  (`classic_competitor.py:664-669`).
- The variant registry defines `classic_kk_steps100`, `classic_kk_steps300`, and
  `classic_kk_steps1000` with dagua `steps` values, but the reference side
  `nx_kamada_kawai` receives `{}` for all three variants (`variants.py:381-408`).
- `LBFGSStep` only forwards `options={"maxiter": maxiter}` if maxiter is not
  `None` and not `0` (`optimize.py:1031-1033`).
- Dagua also has a callback trace path, but it only records positions and does
  not alter the solver (`optimize.py:1007-1019`, `optimize.py:1025-1038`).

Convergence verdict: direct dagua `steps=None` or `steps=0` aligns with
NetworkX. The benchmarked family variants with `steps=100/300/1000` are not
literal parameter matches because NetworkX receives no `maxiter` override. The
measured `0.000` RMSD indicates SciPy converges before these caps on the tested
graphs, but a hard or poorly conditioned graph can diverge at `steps=100` if
NetworkX's default budget is higher.

## 7. Hyperparameter alignment table

| Parameter / behavior | Dagua default / source | NetworkX default / source | Match? | Notes |
|---|---:|---:|---|---|
| Graph type in eval | `nx.DiGraph` equivalent via directed `edge_index`; adapter creates Dagua graph elsewhere | `NetworkXKamadaKawai` uses `_graph_to_nx` returning `nx.DiGraph` (`networkx_competitor.py:20-47`) | Y in eval | Directed shortest paths align for simple edges. |
| Distance algorithm | BFS when no `edge_weights`, Dijkstra when weights present (`distance.py:736-759`) | `nx.shortest_path_length(G, weight="weight")` (`layout.py:946-947`) | Y | Weighted semantics align for positive weights. |
| Unreachable distance | `1.0e6` default (`distance.py:665-670`, `distance.py:747-759`) | `1e6 * np.ones` matrix (`layout.py:948`) | Y | Exact value matches. |
| Diagonal distance | BFS/Dijkstra source distance 0 (`graph_utils.py:61-63`, `graph_utils.py:83-85`) | Filled from shortest path `rdist[nc]` including self (`layout.py:952-956`) | Y | Both become 0 before inverse-distance eye epsilon. |
| Duplicate directed edge weights | Keep minimum (`graph_utils.py:396-398`) | Plain `DiGraph.add_edge` overwrites previous attributes (`networkx_competitor.py:40-46`) | N | Latent weighted multiedge divergence. |
| Self-loop topology | Kept in directed adjacency unless overwritten (`graph_utils.py:396-398`) | Added to `DiGraph`; shortest path self remains 0 (`networkx_competitor.py:40-46`, `layout.py:946-956`) | Mostly Y | Self-loop weight should not change self distance; duplicate self-loop side effects negligible. |
| Initial positions, default 2D | Deterministic circular, float32 theta (`init.py:623-630`) | `circular_layout`, float32 theta (`layout.py:958-963`, `layout.py:199-206`) | Y | Same scheme. |
| Initial positions, user supplied | Tensor shape `(N,2)` (`kk.py:205-207`, `kk.py:249-250`) | Node-keyed dict (`layout.py:898-901`, `layout.py:965`) | Interface N | Equivalent values can be supplied with manual conversion. |
| RNG seed | Accepted but unused (`kk.py:197-199`, `kk.py:232`) | No `seed` parameter on `kamada_kawai_layout` (`layout.py:876-885`) | Y for 2D | No RNG consumed. |
| 3D init | Not exposed; fixed `[N,2]` output (`kk.py:170-182`, `kk.py:249-250`) | Random layout for `dim >= 3` (`layout.py:958-961`) | N | Outside current eval pair. |
| 1D init | Not exposed | Linear layout for `dim == 1` (`layout.py:963-965`) | N | Outside current eval pair. |
| Objective epsilon | `1.0e-3` (`optimize.py:23`, `optimize.py:62`, `optimize.py:1021-1023`) | `1e-3` (`layout.py:986-988`, `layout.py:1007`) | Y | Exact. |
| Centering weight | `1.0e-3` (`optimize.py:24`, `optimize.py:1027`) | `meanwt = 1e-3` (`layout.py:986-988`) | Y | Exact. |
| Optimizer | SciPy L-BFGS-B (`optimize.py:1025-1038`) | SciPy L-BFGS-B (`layout.py:989-995`) | Y | Same method and jac flag. |
| Max iterations | Direct default unset; classic adapter default 300 (`kk.py:121-131`, `classic_competitor.py:159-163`) | Unset SciPy default (`layout.py:989-995`) | Direct Y, adapter N | Biggest benchmark-hidden divergence. |
| Final scale | Dagua core rescales to unit span (`postprocess.py:790-795`) | NetworkX rescales to `scale=1` then adapter multiplies by 500 (`layout.py:969`, `networkx_competitor.py:50-58`) | Shape Y, absolute N | Procrustes hides scale. |
| Center | Dagua final centered at zero (`graph_utils.py:423-427`) | NetworkX default center from `_process_params`, then adapter keeps returned center (`layout.py:941`, `layout.py:969`) | Y for default | Non-default center not exposed in dagua. |
| Output dtype | float32 tensor (`postprocess.py:790-795`) | NumPy float64 from SciPy, converted to torch default float32 in adapter (`networkx_competitor.py:50-58`) | Eval Y | Direct NetworkX return has higher precision. |
| Output device | Dagua follows input edge/node tensor (`kk.py:252`, `graph_utils.py:136-158`) | NetworkX adapter builds CPU tensor (`networkx_competitor.py:50-58`) | N | Does not affect geometry. |
| Orientation flip | Enabled in classic competitor (`classic_competitor.py:664-672`, `classic_competitor.py:1609-1613`) | No such postprocess | N | Procrustes with reflection can hide this. |
| Variant params | Dagua variants alter `steps` (`variants.py:381-408`) | Reference variants `{}` (`variants.py:385-408`) | N | Measured equivalent because convergence reaches same optimum. |
| Stochasticity flag | `classic_kk: False` (`variants.py:1822-1824`) | `nx_kamada_kawai: False` (`variants.py:1846-1848`) | Y | Correct for default 2D path. |

## 8. Edge cases

Self-loops:

- Dagua's directed adjacency builder does not drop self-loops (`graph_utils.py:396-398`).
- NetworkX adapter adds self-loop edges to the `DiGraph` (`networkx_competitor.py:40-46`).
- In both implementations, shortest path from a node to itself is 0, then the
  inverse-distance diagonal is stabilized by adding `1e-3` (`layout.py:986-988`;
  `optimize.py:1021-1023`) and offset diagonal is zeroed (`layout.py:1009-1010`;
  `optimize.py:65-67`). Expected impact: none for isolated self-loop semantics.

Multi-edges:

- Dagua stores every edge in `edge_index` (`graph.py:337-343`, `graph.py:847-848`)
  but KK's distance builder collapses duplicate directed edges by minimum weight
  (`graph_utils.py:396-398`).
- NetworkX adapter uses plain `nx.DiGraph`, so repeated edges are collapsed by
  NetworkX's edge storage with later `add_edge` calls overwriting previous
  attributes (`networkx_competitor.py:35-47`).
- Unweighted duplicate edges converge to the same topology. Weighted duplicate
  edges can diverge if the last duplicate weight differs from the minimum
  duplicate weight. This is a real edge-case bug for weighted multigraphs.

Disconnected components:

- Both implementations fill unreachable distances with `1e6` (`distance.py:747-759`;
  `layout.py:948-956`).
- This creates inverse distances near `1e-6`, so disconnected pairs are weakly
  coupled rather than completely independent. Both share this behavior.

Weighted edges:

- NetworkX default `weight="weight"` means weights are path lengths, not spring
  strengths (`layout.py:903-905`, `layout.py:946-947`).
- Dagua uses weights as Dijkstra path lengths when `edge_weights is not None`
  (`distance.py:736-759`).
- Dagua stores/imports weights as float32 (`graph.py:347-350`,
  `graph.py:1677-1683`), and the KK distance builder converts weights through
  `.float().tolist()` (`graph_utils.py:391-392`) before creating float64
  distances. NetworkX adapter converts edge weights with `float(weights[e].item())`
  from the same tensor (`networkx_competitor.py:39-45`). Both start from float32
  in dagua graphs, but a direct NetworkX graph could retain higher precision than
  the dagua path.

Empty graph:

- NetworkX returns `{}` for `nNodes == 0` (`layout.py:941-944`).
- Dagua returns an empty float32 tensor on output device (`kk.py:252-255`).
- The adapter/fidelity pipeline can compare empty results only if it has its own
  empty-position handling. Semantically equivalent but interface-divergent.

Single node:

- NetworkX `circular_layout` returns `{node: center}` for one node
  (`layout.py:194-198`), then KK solve and rescale keep a centered coordinate.
- Dagua returns `[[0, 0]]` early (`kk.py:256-258`).
- Equivalent for default center `(0,0)` and Procrustes-style metrics.

Negative or zero weights:

- Neither dagua KK nor NetworkX adapter validates positivity in the KK path
  (`kk.py:242-248`; `networkx_competitor.py:40-46`).
- NetworkX shortest path algorithms can reject or behave badly with negative
  weights depending on backend/path routine. Dagua's Dijkstra helper assumes
  nonnegative weights (`graph_utils.py:75-98`) and would be incorrect for
  negative weighted edges. This is latent because test graphs are not negative
  weighted.

## 9. Numerical precision

Core solve precision:

- Dagua converts positions to CPU float64 NumPy before SciPy (`optimize.py:999-1006`).
- Dagua converts distances to CPU float64 NumPy before SciPy (`optimize.py:1004-1006`).
- NetworkX builds `dist_mtx` as NumPy float64 because `1e6 * np.ones` defaults
  to float64 (`layout.py:948`), uses NumPy positions, and SciPy optimizes
  float64 arrays (`layout.py:965-997`).
- Objective and gradient use the same NumPy operations in the same order
  (`layout.py:1005-1022`; `optimize.py:57-84`).

Boundary precision:

- Dagua circular theta is cast to `np.float32`, then coordinates are cast to
  float64 (`init.py:625-628`). NetworkX circular theta is cast to `np.float32`
  and coordinates are passed through `rescale_layout` (`layout.py:199-206`).
- Dagua final positions are cast to float32 (`postprocess.py:790-795`).
- NetworkX adapter writes into `torch.zeros(num_nodes, 2)`, which defaults to
  float32, after multiplying each coordinate by 500 (`networkx_competitor.py:50-58`).
- Direct NetworkX users get float64 NumPy arrays; dagua direct users get float32
  tensors. The fidelity report compares adapter outputs, so this direct API
  precision divergence is hidden.

Summation order:

- Cost/gradient summation order is NumPy `sum`/`einsum` on both sides
  (`layout.py:1012-1020`; `optimize.py:68-83`).
- Dagua final rescale uses torch mean/max (`graph_utils.py:423-427`), whereas
  NetworkX final rescale uses NumPy mean/max (`layout.py:1918-1924`). Because
  Dagua rescale happens after converting SciPy result back to torch
  (`optimize.py:1039-1045`, `postprocess.py:794`), final coordinate rounding can
  differ at float32 epsilon level. Procrustes RMSD rounds this to zero in the
  report.

## 10. RNG semantics

Default 2D KK consumes no RNG on either side:

- Dagua documents that `seed` is accepted only for interface compatibility and
  the 2D classic path uses deterministic circular initialization (`kk.py:197-199`).
- Dagua explicitly discards `seed` (`kk.py:232`).
- NetworkX `kamada_kawai_layout` has no `seed` parameter (`layout.py:876-885`).
- NetworkX only uses random initialization for `dim >= 3` (`layout.py:958-961`),
  outside dagua's exposed 2D KK path.

Answer to the explicit RNG question: no, a dagua torch seed cannot produce the
same sequence as NetworkX here because neither implementation consumes a seed in
the compared 2D path. If comparing a hypothetical 3D KK path, NetworkX would use
its `random_layout` machinery (`layout.py:120-122`, `layout.py:958-961`) while
dagua currently has no 3D KK implementation in `layout_kk_pipeline`.

## 11. Edge-case bugs and suspicious divergences

1. Weighted duplicate-edge collapse mismatch. Dagua keeps the minimum duplicate
   edge weight (`graph_utils.py:396-398`), while the reference adapter builds a
   plain `nx.DiGraph` and each `add_edge` can overwrite an earlier weight
   (`networkx_competitor.py:40-46`). Unweighted multi-edge benchmarks are safe;
   weighted multiedges are not exact.

2. `steps` variants are not parameter-aligned with the reference. Dagua variants
   pass `steps=100/300/1000` (`variants.py:381-408`), while reference
   `nx_kamada_kawai` receives `{}` (`variants.py:385-408`) and NetworkX passes
   no `options` to SciPy (`layout.py:989-995`). Measured RMSD is still zero, but
   this can fail on harder graphs.

3. Adapter-level orientation flip is not a NetworkX behavior. `ClassicKK.layout`
   passes `orient_to_direction=True` (`classic_competitor.py:664-672`), and the
   generic classic runner also defaults it for `layout_kk_pipeline`
   (`classic_competitor.py:1609-1613`). The flip applies if the aligned-edge
   fraction improves by at least 0.05 (`kk.py:23`, `kk.py:101-118`). NetworkX
   does not orient by edge direction. Procrustes with reflection can hide this.

4. Absolute scale mismatch in eval adapters. Dagua final output is unit-scale
   float32 (`postprocess.py:790-795`), while NetworkX adapter multiplies returned
   coordinates by 500 (`networkx_competitor.py:50-58`). Fidelity Procrustes
   normalizes by vector norm before SVD (`scripts/algo_fidelity_cross.py:263-282`),
   so geometry verdicts hide this. Rendering/metrics that use absolute
   coordinates may diverge if not normalized elsewhere.

5. Direct API dimension mismatch. NetworkX supports `dim=1`, `dim=2`, and
   `dim>=3` (`layout.py:958-965`); dagua `layout_kk_pipeline` is 2D-only and
   validates supplied positions as `(N,2)` (`kk.py:249-250`). This is outside
   the current eval pairing but matters for claiming a complete NetworkX drop-in.

6. Negative weights are unchecked. Dagua Dijkstra assumes nonnegative weights
   (`graph_utils.py:75-98`), and `layout_kk_pipeline` only checks edge-weight
   shape (`kk.py:242-248`). NetworkX shortest-path behavior with negative
   weights is not equivalent to this helper. Add validation or Bellman-Ford only
   if negative weights are intended to be supported.

7. Dagua finalization uses torch rescale after SciPy, not NumPy rescale. The
   formulas match (`graph_utils.py:408-427`; `layout.py:1918-1924`), but dtype
   and reduction backends differ. Expected impact is sub-float32 epsilon in
   adapter comparisons.

## 12. Ranked fix list

Ranked by expected impact on adversarial RMSD or exact parity, not by current
mega-run impact.

1. Align `classic_kk` variant `steps` semantics with NetworkX defaults.
   Source refs: dagua variant steps at `variants.py:381-408`, classic default
   `steps=300` at `classic_competitor.py:159-163`, `ClassicKK.layout` hard-code
   at `classic_competitor.py:664-669`, SciPy option injection at
   `optimize.py:1031-1033`, NetworkX no-options solve at `layout.py:989-995`.
   Proposed fix: for fidelity variants, compare `steps=None` or pass matching
   `maxiter` to a custom reference wrapper if the point is capped comparison.
   Size estimate: S, variant/adapter-only. Expected RMSD impact: high on
   pathological slow-convergence graphs; zero on current suite.

2. Fix weighted duplicate-edge semantics for KK distance building or the
   NetworkX adapter.
   Source refs: dagua min duplicate policy at `graph_utils.py:396-398`,
   NetworkX adapter plain `DiGraph` at `networkx_competitor.py:35-47`, dagua edge
   storage preserves duplicates at `graph.py:337-343` and `graph.py:847-848`.
   Proposed fix: choose a documented policy. For NetworkX parity with current
   adapter, dagua should mimic last-edge-wins when building KK directed adjacency
   from a Dagua graph destined for NetworkX comparison. For graph-theoretic
   weighted shortest paths, alternatively update the NetworkX adapter to use a
   min-weight collapse before adding edges. Size estimate: M because it affects
   shared `build_directed_adjacency` users. Expected RMSD impact: high only on
   weighted multiedge graphs.

3. Disable `orient_to_direction` in fidelity comparison, or apply the same
   deterministic orientation to both sides before comparison.
   Source refs: `ClassicKK.layout` enables orientation at
   `classic_competitor.py:664-672`, generic classic runner at
   `classic_competitor.py:1609-1613`, flip threshold and logic at
   `kk.py:23-118`, direct default disabled at `kk.py:180-216`.
   Proposed fix: set `orient_to_direction=False` for `classic_kk` when the
   target is `nx_kamada_kawai`, or explicitly classify orientation as a
   user-facing postprocess outside algorithm fidelity. Size estimate: S.
   Expected RMSD impact: medium if comparison disallows reflection; low under
   current Procrustes.

4. Normalize adapter scale policy instead of relying on Procrustes to erase it.
   Source refs: dagua final unit scale at `postprocess.py:790-795`, NetworkX
   adapter `* 500.0` at `networkx_competitor.py:50-58`, Procrustes scale
   normalization at `scripts/algo_fidelity_cross.py:263-282`.
   Proposed fix: either return NetworkX KK at unit scale for algorithm fidelity
   or scale dagua classic outputs by the same display factor in competitor
   adapters. Size estimate: S. Expected RMSD impact: zero under current
   Procrustes, high for absolute-coordinate metrics.

5. Preserve float64 through final adapter comparison for sub-percent audits.
   Source refs: dagua final float32 at `postprocess.py:790-795`, NetworkX adapter
   float32 tensor allocation at `networkx_competitor.py:50-58`, direct SciPy
   float64 at `layout.py:989-997` and dagua pre-final SciPy float64 at
   `optimize.py:999-1046`.
   Proposed fix: add an opt-in fidelity mode that returns final KK positions in
   float64 and makes `_nx_pos_to_tensor` allocate `dtype=torch.float64`. Size
   estimate: S/M depending on adapter surface. Expected RMSD impact: tiny but
   relevant for adversarial residual cataloging.

6. Add explicit negative-weight validation for KK.
   Source refs: shape-only dagua validation at `kk.py:242-248`, Dijkstra helper
   at `graph_utils.py:75-98`, NetworkX weight default at `layout.py:903-905`.
   Proposed fix: reject negative `edge_weights` in `layout_kk_pipeline` for now,
   with a docstring note that KK weighted shortest paths require nonnegative
   path lengths. Size estimate: S. Expected RMSD impact: prevents undefined
   divergence rather than improving current scores.

7. Add a parity regression for weighted duplicate edges and capped-iteration
   slow convergence.
   Source refs: test scope is allowed by project instructions (`AGENTS.md:77-81`),
   changed module test mapping for pipelines is `tests/test_pipeline_<name>.py`
   (`AGENTS.md:52-53`). Proposed fix: create targeted regression tests in a
   followup implementation round. Size estimate: M. Expected RMSD impact:
   validates fixes above.

## 13. Recommended Round 22+ fix scope

Recommended one-round bundle:

1. Make the KK fidelity pairing use literal NetworkX iteration semantics:
   compare dagua `steps=None` against `nx_kamada_kawai`, or create explicit
   reference wrappers for capped `maxiter` if the `steps100/300/1000` variants
   are meant to be capped-solver variants. This is the highest-leverage cleanup
   because it fixes a real parameter mismatch (`variants.py:381-408`,
   `classic_competitor.py:159-163`, `layout.py:989-995`).

2. Remove or gate `orient_to_direction=True` from the `classic_kk` fidelity
   adapter path. Keep the user-facing pipeline option intact, but classify it as
   dagua postprocessing, not NetworkX KK (`classic_competitor.py:664-672`,
   `kk.py:101-118`).

3. Add a weighted multiedge adversarial test and decide duplicate semantics.
   If the fidelity target remains current `nx_kamada_kawai`, the test should
   exercise repeated `(u, v)` edges with weights `[10, 1]` and `[1, 10]` to
   expose min-weight-vs-last-write behavior (`graph_utils.py:396-398`,
   `networkx_competitor.py:40-46`).

4. For sub-percent audits, add an opt-in float64/no-display-scale comparison
   mode rather than modifying production adapters. This keeps existing render
   comparability while letting future reports catch tiny finalization residuals
   (`postprocess.py:790-795`, `networkx_competitor.py:50-58`,
   `scripts/algo_fidelity_cross.py:263-282`).

Not recommended for Round 22 unless requested:

- Implementing 1D/3D KK in dagua. It is a real API gap against NetworkX
  (`layout.py:958-965`), but the current dagua-vs-reference pairing is 2D only
  (`kk.py:170-182`) and all measured variants are already `strong_equivalent`.
- Rewriting the objective. The objective and gradient are already formula-matched
  line-for-line (`layout.py:1000-1022`; `optimize.py:27-84`).

Bottom line: `classic_kk` is genuinely strong-equivalent for the current 2D,
simple/unweighted benchmark suite. The remaining adversarial risk is not the KK
energy. It is adapter semantics: capped iterations versus SciPy defaults,
weighted duplicate-edge collapse, optional orientation flipping, and scale/dtype
boundaries hidden by Procrustes normalization.
