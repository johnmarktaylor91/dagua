# Round 21 Adversarial Diff: `fr` Family

Pairing: dagua `classic_fr` vs reference `nx_spring`.

Date: 2026-04-30. Repo branch observed: `develop`.

Scope: diagnosis only. No source edits or commits. This report catalogs every divergence found between Dagua's Fruchterman-Reingold reimplementation and NetworkX `spring_layout`, including residual numerical differences that are below normal visual significance.

## 1. Files Read

Dagua side:

- `dagua/layout/ops/pipelines/fr.py:1-339` -- top-level FR pipeline, default selector, public `layout_fr_pipeline()` / `layout_fr_default_pipeline()` entrypoints.
- `dagua/layout/ops/force.py:26-295` -- FR constants and `k = sqrt(area / N)` helper.
- `dagua/layout/ops/force.py:849-912` -- `FRCombinedForce`, the dense FR displacement law.
- `dagua/layout/ops/force.py:2605-2679` -- `ApplyDisplacement`, the per-node temperature-clamped move.
- `dagua/layout/ops/preprocess.py:262-311` -- `_build_fr_adjacency_matrix`, dense directed adjacency construction.
- `dagua/layout/ops/preprocess.py:1111-1181` -- `FRPrepareAdjacency`, adjacency and force-area setup.
- `dagua/layout/ops/init.py:736-868` -- `RandomUniformInit` and NumPy-backed initialization.
- `dagua/layout/ops/anneal.py:357-422` -- `LinearCool`.
- `dagua/layout/ops/anneal.py:425-492` -- `InitTemperatureFromExtent`.
- `dagua/layout/ops/converge.py:227-307` -- `FRConvergenceCheck`.
- `dagua/layout/ops/base.py:250-281` -- `Pipeline.apply()` sequencing.
- `dagua/layout/ops/base.py:364-438` -- `Repeat` loop and early-stop semantics.
- `dagua/layout/ops/postprocess.py:68-83` -- centering helper.
- `dagua/layout/ops/postprocess.py:238-319` -- `ScalePositions`.
- `dagua/layout/ops/postprocess.py:323-406` -- `FRFinalizePositions`.
- `dagua/eval/competitors/classic_competitor.py:26-97` -- classic adapter seed and variant dispatch.
- `dagua/eval/competitors/classic_competitor.py:153-158` -- `classic_fr` spec.
- `dagua/eval/competitors/classic_competitor.py:569-624` -- direct `ClassicFR.layout()` path.
- `dagua/eval/competitors/networkx_competitor.py:20-58` -- DaguaGraph to NetworkX conversion and NetworkX position scaling.
- `dagua/eval/competitors/networkx_competitor.py:61-136` -- NetworkX adapter call semantics.
- `dagua/eval/competitors/networkx_competitor.py:147-154` -- `nx_spring` adapter defaults.
- `dagua/eval/variants.py:335-379` -- FR variant definitions and dagua/reference parameter pairing.
- `dagua/eval/variants.py:1820-1847` -- stochasticity flags for `classic_fr` and `nx_spring`.

Reference side:

- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:42-60` -- `_process_params` graph/center handling.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:452-656` -- `spring_layout` public entrypoint and dense/sparse dispatch.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:659-720` -- dense `_fruchterman_reingold`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:723-811` -- sparse force-mode `_sparse_fruchterman_reingold`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:814-873` -- sparse energy-mode `_energy_fruchterman_reingold`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py:1882-1924` -- `rescale_layout`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/utils/decorators.py:264-308` -- `np_random_state` decorator used by `spring_layout`.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/utils/misc.py:271-298` -- `create_random_state`; int seeds become `np.random.RandomState(seed)`.

Existing analysis:

- `eval_output/fidelity_report/report.md:34-37` -- current mega-run verdicts for `fr_steps50`, `fr_steps100`, `fr_steps200`, `fr_steps500`.
- `eval_output/fidelity_report/report.md:184-189` -- methodology for Procrustes, within-vs-between, TOST, BH correction, and quality metrics.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:273-274` -- prior residual note for `fr_steps200` / `fr_steps500`.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:111-130` -- stochastic-fidelity methodology context.

Environment note: the installed NetworkX version observed by `python -c "import networkx as nx; print(nx.__version__)"` is `3.6.1`.

## 2. Overall Pipeline Structure

Dagua `classic_fr` has two relevant paths, and the default path is not a plain single call:

- Generic variant path: `_ClassicBase.layout_with_variant()` resolves the classic spec, merges variant params, and calls `_quick_classic()` with `self._layout_seed(seed)` (`dagua/eval/competitors/classic_competitor.py:80-97`). The `classic_fr` spec points to `dagua.layout.ops.pipelines.fr.layout_fr_default_pipeline` with default `steps=200` (`dagua/eval/competitors/classic_competitor.py:153-158`).
- Direct adapter path: `ClassicFR.layout()` calls `layout_fr_default_pipeline(graph.edge_index, graph.num_nodes, node_sizes=graph.node_sizes, steps=200, seed=self._layout_seed(seed))` (`dagua/eval/competitors/classic_competitor.py:608-614`).
- Pipeline path: `build_fr_pipeline(steps)` sequences `FixedSteps`, NumPy uniform init, `FRPrepareAdjacency`, `InitTemperatureFromExtent`, a repeated inner loop of `FRCombinedForce -> ApplyDisplacement -> FRConvergenceCheck -> LinearCool`, then `FRFinalizePositions` (`dagua/layout/ops/pipelines/fr.py:153-177`).
- Default selector: `layout_fr_default_pipeline()` runs exactly requested `steps` only when `steps != 200` or warm-start `pos` is supplied (`dagua/layout/ops/pipelines/fr.py:300-309`). For default `steps=200` with no warm start, it computes both a 200-step `legacy_pos` and a 50-step `canonical_pos`, then chooses between them using `_choose_fr_default_layout()` (`dagua/layout/ops/pipelines/fr.py:311-332`). The selector rejects the 50-step candidate if TB directed consistency drops by more than `0.1` or composite score drops by more than `1e-6` (`dagua/layout/ops/pipelines/fr.py:27-30`, `dagua/layout/ops/pipelines/fr.py:115-124`).

Reference `nx_spring` path:

- The adapter builds a `networkx.DiGraph`, copies all nodes, and adds directed edges with optional `weight` attributes (`dagua/eval/competitors/networkx_competitor.py:20-47`).
- `NetworkXSpring` invokes `nx.spring_layout` with adapter defaults `{"seed": 42, "iterations": 50}` and permits `gravity`, `iterations`, `k`, and `scale` overrides (`dagua/eval/competitors/networkx_competitor.py:147-154`).
- `spring_layout()` validates `method`, picks `"force"` for `len(G) < 500` and `"energy"` otherwise when `method="auto"` (`networkx/drawing/layout.py:589-593`), processes center/dim (`networkx/drawing/layout.py:594`), builds an adjacency matrix (`networkx/drawing/layout.py:627-644`), calls either dense `_fruchterman_reingold()` or sparse `_sparse_fruchterman_reingold()` / `_energy_fruchterman_reingold()` (`networkx/drawing/layout.py:627-645`), rescales if `fixed is None and scale is not None` (`networkx/drawing/layout.py:646-647`), and returns a node-position dict (`networkx/drawing/layout.py:648-653`).

High-level conclusion:

- For graphs with `N < 500`, no fixed positions, default `k`, default `scale=1`, `iterations=50`, and no multi-edge overwrite issue, Dagua's inner 50-step force path is a close algorithmic port of NetworkX dense `_fruchterman_reingold`.
- The benchmark pairing still compares Dagua `classic_fr` output scaled by `50 * sqrt(N)` (`dagua/layout/ops/postprocess.py:399-405`) against adapter-scaled NetworkX output scaled by `500` (`dagua/eval/competitors/networkx_competitor.py:50-58`). Procrustes RMSD usually absorbs uniform scale, but any metric that sees raw units does not.
- For `N >= 500`, NetworkX default `method="auto"` switches to energy mode (`networkx/drawing/layout.py:591-593`, `networkx/drawing/layout.py:628-636`), while Dagua always runs the force pipeline (`dagua/layout/ops/pipelines/fr.py:153-177`). This is a major algorithmic divergence outside the small dense regime.

## 3. Energy / Loss / Objective

Dagua force-mode objective is implicit, not optimized as an explicit scalar loss:

- `FRCombinedForce` resolves `k = sqrt(area / N)` through `_resolve_area_k()` (`dagua/layout/ops/force.py:279-295`), where default area is `1.0` (`dagua/layout/ops/force.py:253-276`) and `FRPrepareAdjacency` sets `state.force_area = 1.0` (`dagua/layout/ops/preprocess.py:1175-1181`).
- The displacement tensor uses NetworkX's dense formula: `delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]`, `distance = norm(delta)`, `distance = clamp(distance, min=0.01)`, and `displacement = einsum(delta, (k*k / distance.square()) - (adjacency * distance / k))` (`dagua/layout/ops/force.py:900-910`).
- The "objective" being followed is the classic FR force field, equivalent to pairwise repulsion proportional to `k^2 / distance` in vector form and edge attraction proportional to `distance^2 / k`, but Dagua does not compute or minimize an explicit energy scalar in this pipeline.

NetworkX dense force mode:

- Default optimal distance `k = sqrt(1.0 / nnodes)` (`networkx/drawing/layout.py:680-682`).
- Dense displacement is exactly `np.einsum("ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k))` after clamping distance to `0.01` (`networkx/drawing/layout.py:696-705`).
- Like Dagua, dense force mode has no explicit scalar objective in code; it iterates the FR displacement law (`networkx/drawing/layout.py:695-720`).

NetworkX sparse energy mode:

- For `method="energy"` or `method="auto"` with `len(G) >= 500`, NetworkX calls `_energy_fruchterman_reingold()` (`networkx/drawing/layout.py:628-636`, `networkx/drawing/layout.py:763-766`).
- It takes absolute edge weights and symmetrizes adjacency: `A = np.abs(A); A = (A + A.T) / 2` (`networkx/drawing/layout.py:831-833`).
- Cost terms: attraction integrates as `cost += np.sum(Ad * distance2) / (3 * k)` (`networkx/drawing/layout.py:855-856`); repulsion integrates as `cost -= k**2 * np.sum(np.log(distance))` (`networkx/drawing/layout.py:857-858`); component gravity adds `gravity * 0.5 * sum(bincount * ||delta0||^2)` (`networkx/drawing/layout.py:859-864`).
- Gradient term is `grad[l:r] = 2 * einsum(Ad / k - k**2 / distance2, delta)` (`networkx/drawing/layout.py:851-854`), plus gravity (`networkx/drawing/layout.py:859-864`), then L-BFGS-B optimizes it (`networkx/drawing/layout.py:869-873`).

Objective match matrix:

| Term | Dagua `classic_fr` | NetworkX dense force | NetworkX energy | Match? |
|---|---|---|---|---|
| Repulsion | `k*k / distance.square()` inside displacement (`dagua/layout/ops/force.py:905-910`) | `k*k / distance**2` (`networkx/drawing/layout.py:703-705`) | `-k**2 * log(distance)` cost and `-k**2/distance2` gradient (`networkx/drawing/layout.py:854-858`) | Yes vs dense, no vs energy |
| Attraction | `adjacency * distance / k` inside displacement (`dagua/layout/ops/force.py:905-910`) | `A * distance / k` (`networkx/drawing/layout.py:703-705`) | integrated `Ad * distance2 / (3*k)` with symmetrized absolute `A` (`networkx/drawing/layout.py:831-856`) | Yes vs dense, no vs energy |
| Gravity | None | None | Component gravity to center `0.5` (`networkx/drawing/layout.py:859-864`) | Yes vs dense, no vs energy |
| Fixed-node term | None | Zero `delta_pos[fixed]` if fixed (`networkx/drawing/layout.py:712-715`) | `grad[fixed] = 0.0` (`networkx/drawing/layout.py:865-866`) | Dagua lacks fixed API |
| Final scaling | Center then max-abs scale to `50*sqrt(N)` (`dagua/layout/ops/postprocess.py:398-405`) | `rescale_layout(..., scale=1)` then adapter multiplies by `500` (`networkx/drawing/layout.py:646-647`, `dagua/eval/competitors/networkx_competitor.py:50-58`) | Same public scaling after energy mode (`networkx/drawing/layout.py:646-647`) | Not raw-scale matched |

## 4. Force / Gradient Computation

Dense force path:

- Dagua and NetworkX use the same pairwise difference orientation: source row position minus target column position (`dagua/layout/ops/force.py:901`, `networkx/drawing/layout.py:697`).
- Dagua uses Torch `torch.linalg.norm(delta, dim=-1)` (`dagua/layout/ops/force.py:902`); NetworkX uses NumPy `np.linalg.norm(delta, axis=-1)` (`networkx/drawing/layout.py:699`).
- Both clamp pair distances to `0.01`: Dagua `_FR_MIN_DISTANCE = 0.01` (`dagua/layout/ops/force.py:26`, `dagua/layout/ops/force.py:903`); NetworkX `np.clip(distance, 0.01, None, out=distance)` (`networkx/drawing/layout.py:700-701`).
- Both use `einsum` over `delta` and the scalar coefficient matrix: Dagua `torch.einsum("ijk,ij->ik", ...)` (`dagua/layout/ops/force.py:905-910`); NetworkX `np.einsum("ijk,ij->ik", ...)` (`networkx/drawing/layout.py:703-705`).
- Dagua casts adjacency to position dtype/device before force computation (`dagua/layout/ops/force.py:904`); NetworkX constructs `A` as NumPy array default float64 for dense path (`networkx/drawing/layout.py:638`) and initializes random `pos` as `dtype=A.dtype` (`networkx/drawing/layout.py:673-675`).

Gradient/energy path:

- Dagua has no equivalent to NetworkX `_energy_fruchterman_reingold()` in the `classic_fr` pipeline. It never symmetrizes absolute weights, never adds component gravity, and never calls L-BFGS-B (`dagua/layout/ops/pipelines/fr.py:153-177` vs `networkx/drawing/layout.py:831-873`).
- This matters for any graph with `N >= 500` under `nx_spring` default `method="auto"` (`networkx/drawing/layout.py:591-593`, `networkx/drawing/layout.py:628-636`).

Application of force to positions:

- Dagua computes force norm with `torch.linalg.vector_norm(forces, dim=1).clamp(min=0.01)`, then `delta_pos = forces * (temperature / length).unsqueeze(1)` and updates `state.pos = pos + delta_pos` (`dagua/layout/ops/force.py:2673-2678`).
- NetworkX computes `length = np.linalg.norm(displacement, axis=-1)`, clips to `0.01`, computes `delta_pos = np.einsum("ij,i->ij", displacement, t / length)`, and updates `pos += delta_pos` (`networkx/drawing/layout.py:706-715`).
- Formula matches in dense mode. Summation order and backend differ: Torch `einsum` vs NumPy `einsum` can diverge at the last bit even with float64.

Empirical spot check:

- On a 3-node directed path with `steps=50`, `seed=42`, no weights, and Dagua output divided by its `50*sqrt(N)` final scale, the maximum absolute coordinate difference from `nx.spring_layout(..., seed=42, iterations=50)` was about `2.9e-08`. This confirms the core dense path is effectively equivalent for the small dense default case.

## 5. Initialization

Dagua initialization:

- `build_fr_pipeline()` uses `RandomUniformInit(RandomUniformInitConfig(scale="none", rng_backend="numpy"))` only when `state.pos is None` (`dagua/layout/ops/pipelines/fr.py:155-163`).
- NumPy backend uses `np.random.RandomState(problem.seed).rand(problem.num_nodes, position_dim)` and converts the NumPy array with `torch.from_numpy(...)` (`dagua/layout/ops/init.py:854-859`).
- Because NumPy `RandomState.rand()` returns float64, Dagua initializes the FR path in float64 when using `rng_backend="numpy"` (`dagua/layout/ops/init.py:854-859`) and leaves scale unchanged because `scale="none"` (`dagua/layout/ops/init.py:865-867`).
- For `num_nodes == 0`, the FR-specific empty path returns a `(0, dim)` float64 tensor (`dagua/layout/ops/init.py:818-828`). For single-node graphs, `_maybe_set_empty_or_single_positions()` short-circuits before random sampling (`dagua/layout/ops/init.py:830-836`), producing Dagua-side special behavior before the FR loop.
- Warm starts are accepted via `layout_fr_pipeline(..., pos=...)`, cloned and cast to float64 (`dagua/layout/ops/pipelines/fr.py:247-249`).

NetworkX initialization:

- `spring_layout` is decorated with `@np_random_state("seed")`, so an integer seed is converted to a NumPy `RandomState` (`networkx/drawing/layout.py:451`, `networkx/utils/decorators.py:264-308`, `networkx/utils/misc.py:286-292`).
- If `pos is None`, dense `_fruchterman_reingold` uses `pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)` (`networkx/drawing/layout.py:673-675`).
- Dense `A = nx.to_numpy_array(G, weight=weight)` defaults to float64 (`networkx/drawing/layout.py:638`), so dense random positions are also float64 (`networkx/drawing/layout.py:673-675`).
- If user supplies `pos`, NetworkX constructs random positions over the existing domain and overwrites specified nodes (`networkx/drawing/layout.py:605-615`); Dagua's warm-start path expects a complete `[N,2]` tensor and does not support partial node positions (`dagua/layout/ops/pipelines/fr.py:236-249`).
- For empty graph, NetworkX returns `{}` before internal solver (`networkx/drawing/layout.py:619-620`). For one node, it returns the center coordinate (`networkx/drawing/layout.py:621-625`).

Initialization match:

- Default no-warm-start small dense case: yes. Both use NumPy `RandomState(seed).rand(N,2)` in float64 when seed is an integer (`dagua/layout/ops/init.py:854-859`; `networkx/drawing/layout.py:673-675`; `networkx/utils/misc.py:286-292`).
- Torch seed semantics: Dagua's FR path deliberately avoids Torch RNG by using `rng_backend="numpy"` (`dagua/layout/ops/pipelines/fr.py:159-162`). Therefore Dagua's Torch seed sequence is irrelevant for `classic_fr`; the `seed` parameter is a Python int passed into NumPy `RandomState`.

## 6. Iteration / Convergence

Dagua:

- `FixedSteps(FixedStepsConfig(n=steps))` is placed before initialization (`dagua/layout/ops/pipelines/fr.py:153-156`). This presumably writes `state.total_steps`; the `LinearCool` op later reads `state.total_steps` (`dagua/layout/ops/anneal.py:379-386`).
- Repeat loop runs up to `steps` iterations and stops early if `state.converged` is set (`dagua/layout/ops/base.py:425-438`).
- Per iteration, Dagua runs force, applies displacement, checks convergence, then cools temperature (`dagua/layout/ops/pipelines/fr.py:167-174`).
- Initial temperature is `max(x_extent, y_extent) * 0.1` (`dagua/layout/ops/anneal.py:489-491`).
- Cooling subtracts `initial_temperature / (state.total_steps + 1)` every step when no explicit rate is configured (`dagua/layout/ops/anneal.py:412-421`).
- Convergence computes the same normalized Frobenius displacement rule as NetworkX: recomputed `delta_pos = state.forces * (temperature / length)` and `mean_displacement = norm(delta_pos) / N`, then `state.converged = state.converged or mean_displacement < 1e-4` (`dagua/layout/ops/converge.py:301-306`).
- Important order issue: Dagua checks convergence after already applying the displacement (`dagua/layout/ops/pipelines/fr.py:170-173`; `dagua/layout/ops/force.py:2677-2678`; `dagua/layout/ops/converge.py:301-306`). NetworkX also applies `pos += delta_pos` before checking threshold (`networkx/drawing/layout.py:711-719`). This order matches.

NetworkX dense:

- Iteration count default is `50` (`networkx/drawing/layout.py:452-458`; adapter default at `dagua/eval/competitors/networkx_competitor.py:151-152`).
- Initial temperature is `max(x_extent, y_extent) * 0.1` (`networkx/drawing/layout.py:683-687`).
- Cooling decrement is `dt = t / (iterations + 1)` (`networkx/drawing/layout.py:688-690`).
- Loop runs `for iteration in range(iterations)` (`networkx/drawing/layout.py:695`), computes force, applies move, cools, then breaks if `np.linalg.norm(delta_pos) / nnodes < threshold` (`networkx/drawing/layout.py:715-719`).

Iteration divergence:

- Variant alignment exists for steps 50/100/200/500: `classic_fr_steps50` pairs `steps=50` with `nx_spring iterations=50`, etc. (`dagua/eval/variants.py:335-379`).
- But base `ClassicFR.layout()` hardcodes `steps=200` into `layout_fr_default_pipeline` (`dagua/eval/competitors/classic_competitor.py:608-614`), while `NetworkXSpring` hardcodes `iterations=50` (`dagua/eval/competitors/networkx_competitor.py:147-152`).
- `layout_fr_default_pipeline(steps=200)` computes both 200 and 50 steps, then may choose the 50-step canonical candidate or keep 200-step legacy depending on Dagua-specific DAG/composite criteria (`dagua/layout/ops/pipelines/fr.py:311-332`). NetworkX has no such dual-candidate selector.

## 7. Hyperparameter Alignment Table

| Parameter | Dagua default / behavior | NetworkX default / behavior | Match? | Impact |
|---|---|---|---|---|
| Core algorithm for `N < 500` | Dense force pipeline (`dagua/layout/ops/pipelines/fr.py:153-177`) | Dense `_fruchterman_reingold` under `method="auto"` (`networkx/drawing/layout.py:591-645`) | Yes | Low residual only |
| Core algorithm for `N >= 500` | Still dense force pipeline (`dagua/layout/ops/pipelines/fr.py:153-177`) | Energy mode under `method="auto"` (`networkx/drawing/layout.py:591-593`, `networkx/drawing/layout.py:628-636`) | No | High on large graphs |
| Default iterations | Adapter direct path passes `steps=200` (`dagua/eval/competitors/classic_competitor.py:608-614`) but default selector may choose 50 (`dagua/layout/ops/pipelines/fr.py:311-332`) | Adapter passes `iterations=50` (`dagua/eval/competitors/networkx_competitor.py:147-152`) | Partial | Medium |
| Variant iterations | `steps` overrides merge through `_ClassicBase.layout_with_variant()` (`dagua/eval/competitors/classic_competitor.py:80-97`) | `iterations` overrides merge into layout kwargs (`dagua/eval/competitors/networkx_competitor.py:124-130`) | Yes for declared variants | Low |
| `k` | Always `sqrt(1/N)` from unit `force_area` (`dagua/layout/ops/force.py:253-295`; `dagua/layout/ops/preprocess.py:1175-1181`) | Default `sqrt(1/N)`, user-overridable `k` (`networkx/drawing/layout.py:491-494`, `networkx/drawing/layout.py:680-682`) | Default yes, override no | Medium if `k` variant used |
| Distance floor | `0.01` (`dagua/layout/ops/force.py:26`, `dagua/layout/ops/force.py:903`) | `0.01` dense/sparse force (`networkx/drawing/layout.py:700-701`, `networkx/drawing/layout.py:792-793`) | Yes | Low |
| Move norm floor | `0.01` (`dagua/layout/ops/force.py:2610-2615`, `dagua/layout/ops/force.py:2673-2677`) | `0.01` (`networkx/drawing/layout.py:706-711`) | Yes | Low |
| Initial temperature | `max(extent)*0.1` (`dagua/layout/ops/anneal.py:489-491`) | `max(extent)*0.1` (`networkx/drawing/layout.py:683-687`) | Yes | Low |
| Cooling schedule | `initial / (total_steps + 1)` (`dagua/layout/ops/anneal.py:412-421`) | `t / (iterations + 1)` (`networkx/drawing/layout.py:688-690`) | Yes if `state.total_steps == steps` | Low |
| Threshold | `1e-4` (`dagua/layout/ops/converge.py:227-239`) | `1e-4` (`networkx/drawing/layout.py:457-458`, `networkx/drawing/layout.py:718-719`) | Yes | Low |
| RNG | NumPy `RandomState(seed).rand` (`dagua/layout/ops/init.py:854-859`) | NumPy `RandomState(seed).rand` for int seeds (`networkx/utils/misc.py:286-292`, `networkx/drawing/layout.py:673-675`) | Yes | Very low |
| Initial dtype | float64 from NumPy backend (`dagua/layout/ops/init.py:854-859`) | float64 dense path via `to_numpy_array` + `dtype=A.dtype` (`networkx/drawing/layout.py:638`, `networkx/drawing/layout.py:673-675`) | Yes small dense | Very low |
| Output dtype | Cast to `torch.float32` (`dagua/layout/ops/postprocess.py:405`) | Adapter converts to default `torch.zeros` float32 and assigns Python floats (`dagua/eval/competitors/networkx_competitor.py:50-58`) | Effectively yes after adapter | Low |
| Public scale | `50*sqrt(N)` max-abs (`dagua/layout/ops/postprocess.py:399-405`) | `scale=1` in NetworkX then adapter `*500` (`networkx/drawing/layout.py:646-647`; `dagua/eval/competitors/networkx_competitor.py:50-58`) | No | Low for Procrustes, high for raw metrics |
| Center | Mean-center to zero (`dagua/layout/ops/postprocess.py:68-83`, `dagua/layout/ops/postprocess.py:398`) | Mean-center in `rescale_layout`, then add `center` default zeros (`networkx/drawing/layout.py:646-647`, `networkx/drawing/layout.py:1918-1924`) | Yes default | Low |
| Directionality | Directed dense adjacency, no symmetrization (`dagua/layout/ops/preprocess.py:262-311`) | `nx.DiGraph` adjacency from directed graph; dense path not symmetrized (`dagua/eval/competitors/networkx_competitor.py:35-47`, `networkx/drawing/layout.py:638`) | Yes small dense | Low |
| Multi-edge aggregation | Tensor assignment overwrites duplicate edge pairs (`dagua/layout/ops/preprocess.py:306-310`) | Adapter converts to `nx.DiGraph`; repeated same directed edge overwrites same edge attr (`dagua/eval/competitors/networkx_competitor.py:35-47`) | Mostly yes after adapter, but last-edge weight order matters | Low/medium |
| Self-loops | Diagonal adjacency can be 1/weight (`dagua/layout/ops/preprocess.py:306-310`) but `delta` diagonal is zero (`dagua/layout/ops/force.py:901-910`) | `to_numpy_array` includes diagonal, but `delta` diagonal is zero (`networkx/drawing/layout.py:638`, `networkx/drawing/layout.py:696-705`) | Yes | Low |
| Fixed nodes | No public fixed-node support in `layout_fr_pipeline` (`dagua/layout/ops/pipelines/fr.py:182-262`) | Supports `fixed` and disables rescale (`networkx/drawing/layout.py:596-604`, `networkx/drawing/layout.py:646-647`) | No | Medium if used |
| Partial warm start | Requires full tensor shape `(N,2)` (`dagua/layout/ops/pipelines/fr.py:236-249`) | Allows partial dict over nodes (`networkx/drawing/layout.py:605-615`) | No | Medium if used |
| `weight=None` | No explicit weight-name control; weights from `graph.edge_weights` or ones (`dagua/layout/ops/preprocess.py:306-310`) | `weight` keyword defaults `"weight"`; `None` forces all ones (`networkx/drawing/layout.py:513-516`) | Partial | Low in benchmark default |
| `gravity` | No parameter | Used only in energy mode (`networkx/drawing/layout.py:550-552`, `networkx/drawing/layout.py:859-864`) | No | High for `N>=500`/energy variants |

## 8. Edge Cases

Self-loops:

- Dagua adjacency records self-loop weights on the diagonal (`dagua/layout/ops/preprocess.py:306-310`). In `FRCombinedForce`, diagonal `delta` is zero because `pos[i]-pos[i]=0`, so the diagonal attraction coefficient multiplies zero vector (`dagua/layout/ops/force.py:901-910`).
- NetworkX dense path similarly has diagonal `A` possible from `to_numpy_array`, but diagonal `delta` is zero (`networkx/drawing/layout.py:638`, `networkx/drawing/layout.py:696-705`).
- Expected divergence: negligible in dense force mode; both self-loops are effectively no-op for force. They can still affect energy mode if diagonal survives symmetrization, but delta zero likely nulls force contribution.

Multi-edges:

- DaguaGraph stores an edge tensor; `_build_fr_adjacency_matrix` writes `adjacency[sources, targets] = weights` or `1.0` (`dagua/layout/ops/preprocess.py:306-310`). Duplicate directed pairs are overwritten by advanced indexing assignment, not summed.
- NetworkX adapter uses `nx.DiGraph`, not `MultiDiGraph`, and calls `G.add_edge(source, target, weight=...)` for each edge (`dagua/eval/competitors/networkx_competitor.py:35-47`). Duplicate directed pairs overwrite the same DiGraph edge attribute.
- Likely benchmark match: both collapse duplicate directed edges. Subtle risk: PyTorch advanced indexing with repeated indices has backend-dependent "last write" behavior, while Python `DiGraph.add_edge` deterministically leaves the last inserted edge attribute. With unweighted duplicates both end at `1.0`; with weighted duplicates, duplicate-pair final weight may diverge on GPU or non-deterministic scatter semantics.

Disconnected components:

- Dagua dense force mode applies repulsion among all nodes and no component gravity (`dagua/layout/ops/force.py:900-910`).
- NetworkX dense force mode does the same all-pairs force (`networkx/drawing/layout.py:696-705`).
- NetworkX energy mode adds component gravity by connected component (`networkx/drawing/layout.py:835-864`), which Dagua lacks. This becomes important for `N>=500` default auto energy mode and possibly disconnected large graphs.

Weighted edges:

- Dagua accepts `edge_weights` and writes them directly into dense adjacency (`dagua/layout/ops/preprocess.py:306-310`); `layout_fr_pipeline` validates weight length (`dagua/layout/ops/pipelines/fr.py:229-235`).
- NetworkX adapter copies Dagua edge weights to edge attribute `"weight"` (`dagua/eval/competitors/networkx_competitor.py:39-45`), and `spring_layout` defaults `weight="weight"` (`networkx/drawing/layout.py:513-516`).
- Dense force mode match is good for positive weights. Negative weights diverge in energy mode because NetworkX energy takes absolute values (`networkx/drawing/layout.py:831-833`), while Dagua force mode would use negative adjacency directly in the attraction term (`dagua/layout/ops/force.py:905-910`).

Empty graph:

- Dagua `RandomUniformInit` returns an empty float64 position for NumPy `scale="none"` (`dagua/layout/ops/init.py:818-828`), `FRPrepareAdjacency` creates `(0,0)` float64 adjacency (`dagua/layout/ops/preprocess.py:1171-1174`), convergence no-ops for `num_nodes <= 0` (`dagua/layout/ops/converge.py:298-299`), and finalization preserves empty positions through centering/scaling/cast (`dagua/layout/ops/postprocess.py:81-83`, `dagua/layout/ops/postprocess.py:405`).
- NetworkX returns `{}` before solver (`networkx/drawing/layout.py:619-620`), and the adapter converts it to a zero-sized tensor by allocating `torch.zeros(num_nodes, 2)` (`dagua/eval/competitors/networkx_competitor.py:50-58`).
- Expected match at shape level; dtype both adapter outputs float32.

Single-node graph:

- NetworkX returns the center for one node (`networkx/drawing/layout.py:621-625`).
- Dagua uses `_maybe_set_empty_or_single_positions()` before random init (`dagua/layout/ops/init.py:830-836`), then finalization center/scales and returns float32. Exact behavior of the helper was not expanded in this pass, but expected output is a single coordinate at or near zero after centering.
- If Dagua returns `[0,0]`, it matches NetworkX default center after adapter scaling.

Fixed nodes / partial positions:

- NetworkX supports `fixed` and partial/full `pos`; fixed positions disable rescale (`networkx/drawing/layout.py:596-604`, `networkx/drawing/layout.py:646-647`).
- Dagua only supports a full warm-start tensor and no fixed nodes in the FR public API (`dagua/layout/ops/pipelines/fr.py:182-262`).
- Not relevant to current adapter defaults, but it is a clear API divergence if `nx_spring` variants grow to cover fixed/pos.

## 9. Numerical Precision

Dense small-graph precision:

- Dagua FR initialization uses NumPy float64 (`dagua/layout/ops/init.py:854-859`), force adjacency is float64 (`dagua/layout/ops/preprocess.py:291`, `dagua/layout/ops/preprocess.py:306-310`), warm-start positions are cast to float64 (`dagua/layout/ops/pipelines/fr.py:247-249`), and force computation keeps position dtype (`dagua/layout/ops/force.py:904-910`).
- NetworkX dense path uses `nx.to_numpy_array(G, weight=weight)` without explicit dtype, giving float64 in the default dense path (`networkx/drawing/layout.py:638`), and initializes positions as `dtype=A.dtype` (`networkx/drawing/layout.py:673-675`).
- Both cast/effectively return float32 through the benchmark adapters: Dagua `FRFinalizePositions` casts `state.pos` to `torch.float32` (`dagua/layout/ops/postprocess.py:405`); NetworkX adapter allocates `torch.zeros(num_nodes, 2)` with default float32 and assigns scaled floats (`dagua/eval/competitors/networkx_competitor.py:50-58`).

Sparse/large precision:

- NetworkX sparse path explicitly uses `dtype="f"` when constructing SciPy sparse adjacency (`networkx/drawing/layout.py:627-630`), so `A.dtype` and random positions are float32 (`networkx/drawing/layout.py:748-753`) for sparse force/energy entry.
- Dagua remains float64 internally for the FR pipeline. Thus for `N>=500`, both algorithm and precision diverge: Dagua float64 dense force vs NetworkX float32 sparse energy under default auto dispatch.

Summation order:

- Dagua uses Torch `einsum`; NetworkX uses NumPy `einsum` in dense mode (`dagua/layout/ops/force.py:905-910`, `networkx/drawing/layout.py:703-705`). Even with identical formulas and float64, backend summation order may produce last-bit differences. The observed 3-node spot check was within `~3e-08` after final float32 scaling.
- Dagua computes convergence via Torch norm and `.item()` (`dagua/layout/ops/converge.py:301-306`); NetworkX uses NumPy norm (`networkx/drawing/layout.py:706-719`). If a run is exactly near the `1e-4` threshold, this can cause one side to take one extra iteration.

Output scaling:

- Dagua final extent is `50 * sqrt(N)` (`dagua/layout/ops/postprocess.py:399-405`).
- NetworkX public `scale=1` first rescales max coordinate magnitude to 1 (`networkx/drawing/layout.py:646-647`, `networkx/drawing/layout.py:1918-1924`), then the competitor adapter multiplies by 500 (`dagua/eval/competitors/networkx_competitor.py:50-58`).
- This raw scale mismatch is not a numerical precision issue, but it can contaminate non-Procrustes metrics. For `N=100`, Dagua extent is `500`, matching NetworkX adapter. For `N=3`, Dagua extent is about `86.6`, while NetworkX adapter extent is `500`.

## 10. RNG Semantics

Question: does Dagua's Torch seed produce the same sequence as the reference RNG?

- For `classic_fr`, Dagua does not use Torch RNG for default initialization. The pipeline explicitly sets `rng_backend="numpy"` (`dagua/layout/ops/pipelines/fr.py:159-162`), and `RandomUniformInit` calls `np.random.RandomState(problem.seed).rand(...)` (`dagua/layout/ops/init.py:854-859`).
- NetworkX integer seeds are converted to `np.random.RandomState(seed)` by `create_random_state()` (`networkx/utils/misc.py:286-292`) through `@np_random_state("seed")` (`networkx/drawing/layout.py:451`, `networkx/utils/decorators.py:264-308`).
- Therefore, for integer seeds, the Dagua FR initialization sequence should match NetworkX's NumPy sequence exactly in the dense no-warm-start path.
- If some caller bypasses `classic_fr` and constructs `RandomUniformInit` with the default `rng_backend="torch"` (`dagua/layout/ops/init.py:736-755`, `dagua/layout/ops/init.py:842-847`), it will not match NetworkX. That is not the `fr` pipeline under review.
- The adapter seed defaults also align: `_ClassicBase._layout_seed(None)` returns `42` (`dagua/eval/competitors/classic_competitor.py:29-42`), and `NetworkXSpring.layout_kwargs` carries `seed=42` (`dagua/eval/competitors/networkx_competitor.py:147-152`). When an explicit benchmark seed is supplied, both adapters forward it (`dagua/eval/competitors/classic_competitor.py:90-97`; `dagua/eval/competitors/networkx_competitor.py:124-130`).

## 11. Edge-Case Bugs / Suspicious Divergences

1. Default `classic_fr` is a selector, not a pure reference match.
   - Dagua default `steps=200` path computes both 200-step and 50-step layouts and chooses by Dagua-specific directed metrics (`dagua/layout/ops/pipelines/fr.py:300-332`).
   - NetworkX default reference adapter is a direct 50-iteration `spring_layout` call (`dagua/eval/competitors/networkx_competitor.py:147-152`).
   - This is likely intentional for benchmark quality, but adversarially it means base `classic_fr` is not a strict `nx_spring` clone.

2. Large-graph `method="auto"` mismatch.
   - NetworkX changes behavior at `len(G) >= 500`: `"auto"` selects `"energy"` (`networkx/drawing/layout.py:591-593`) and dispatches through sparse/energy code (`networkx/drawing/layout.py:628-636`, `networkx/drawing/layout.py:763-873`).
   - Dagua has no energy path or gravity in `classic_fr` (`dagua/layout/ops/pipelines/fr.py:153-177`).
   - Any fidelity run with graphs at or above this threshold is comparing different algorithms.

3. Output scale mismatch depends on graph size.
   - Dagua final scale is `50*sqrt(N)` (`dagua/layout/ops/postprocess.py:399-405`); NetworkX adapter scale is `500` after NetworkX `scale=1` (`dagua/eval/competitors/networkx_competitor.py:50-58`).
   - They coincide only around `N=100`. For small graphs, raw coordinate metrics can differ by large uniform factors even when shape is identical.

4. Directed adjacency is preserved in dense mode, but NetworkX energy symmetrizes.
   - Dagua `_build_fr_adjacency_matrix` is directed (`dagua/layout/ops/preprocess.py:267-281`, `dagua/layout/ops/preprocess.py:306-310`).
   - NetworkX dense mode also uses directed `DiGraph` adjacency (`dagua/eval/competitors/networkx_competitor.py:35-47`, `networkx/drawing/layout.py:638`).
   - NetworkX energy mode symmetrizes absolute weights (`networkx/drawing/layout.py:831-833`), changing direction and negative-weight semantics for larger graphs.

5. Duplicate weighted edges can be nondeterministic on Dagua tensor assignment.
   - Dagua advanced indexing assignment with repeated `(source,target)` pairs (`dagua/layout/ops/preprocess.py:306-310`) is not an explicit deterministic aggregation policy.
   - NetworkX `DiGraph.add_edge` overwrites deterministically in Python insertion order (`dagua/eval/competitors/networkx_competitor.py:40-46`).
   - For unweighted duplicate edges, both collapse to `1.0`; for weighted duplicates, this is a potential last-edge mismatch.

6. Variant table advertises `k`, `scale`, and `gravity` on NetworkX side without Dagua equivalents.
   - `NetworkXSpring.variant_param_names = {"gravity", "iterations", "k", "scale"}` (`dagua/eval/competitors/networkx_competitor.py:151-154`).
   - FR variants currently only use `iterations` (`dagua/eval/variants.py:335-379`), so this does not affect the listed FR variants.
   - If future exhaustive sweeps add `k`, `scale`, or `gravity` variants, Dagua `classic_fr` has no aligned parameters except indirectly through code changes.

7. Convergence check recomputes `delta_pos` after applying displacement.
   - Dagua applies displacement, then recomputes the same formula from unchanged force/temperature (`dagua/layout/ops/pipelines/fr.py:170-173`, `dagua/layout/ops/force.py:2673-2678`, `dagua/layout/ops/converge.py:301-306`).
   - NetworkX checks the actual `delta_pos` object after applying it (`networkx/drawing/layout.py:711-719`).
   - Because Dagua does not mutate force or temperature between `ApplyDisplacement` and `FRConvergenceCheck`, this is formula-equivalent. But it is fragile if any op is inserted between them.

8. No support for NetworkX fixed-node semantics.
   - NetworkX `fixed` prevents movement and disables final rescale (`networkx/drawing/layout.py:596-604`, `networkx/drawing/layout.py:712-715`, `networkx/drawing/layout.py:646-647`).
   - Dagua has no `fixed` argument in `layout_fr_pipeline` (`dagua/layout/ops/pipelines/fr.py:182-262`).

9. Empty/single graph special cases are handled by different layers.
   - NetworkX returns before solver (`networkx/drawing/layout.py:619-625`).
   - Dagua enters the pipeline but initializer/preprocess/convergence/finalize handle degenerate shapes (`dagua/layout/ops/init.py:818-836`, `dagua/layout/ops/preprocess.py:1171-1174`, `dagua/layout/ops/converge.py:298-299`, `dagua/layout/ops/postprocess.py:81-83`).
   - Expected outputs match, but the control path is not identical.

## 12. Ranked Fix List

Ranked by expected RMSD / metric impact for the current `classic_fr` vs `nx_spring` family.

1. Add a strict NetworkX-compatible finalization mode for FR variants.
   - Evidence: Dagua scales to `50*sqrt(N)` (`dagua/layout/ops/postprocess.py:399-405`); NetworkX rescales to `scale=1` (`networkx/drawing/layout.py:646-647`, `networkx/drawing/layout.py:1918-1924`) and adapter multiplies by 500 (`dagua/eval/competitors/networkx_competitor.py:50-58`).
   - Proposed fix: add an opt-in `output_scale_factor` / `networkx_scale` mode for `layout_fr_pipeline`, or update comparison adapter to normalize raw units consistently for FR variants.
   - Estimate: small (20-50 LOC plus tests).
   - Expected impact: high for raw metric divergences; low for Procrustes-only RMSD.

2. Make `classic_fr_steps50` bypass `layout_fr_default_pipeline` selector and call exact `layout_fr_pipeline`.
   - Evidence: `classic_fr` spec always points to `layout_fr_default_pipeline` (`dagua/eval/competitors/classic_competitor.py:153-158`); selector only bypasses for `steps != 200` or warm starts (`dagua/layout/ops/pipelines/fr.py:300-309`), so steps50 already bypasses, but base `ClassicFR.layout()` hardcodes selector path with `steps=200` (`dagua/eval/competitors/classic_competitor.py:608-614`).
   - Proposed fix: for any explicitly reference-targeted comparison, register a separate `classic_fr_nx` or variant that directly calls `layout_fr_pipeline(steps=50)` and uses NetworkX finalization.
   - Estimate: small (adapter/variant only).
   - Expected impact: medium for base-family confusion; low for current `fr_steps50` variant.

3. Implement NetworkX `method` selection or force `method="force"` on the reference side for fair classic FR comparison.
   - Evidence: NetworkX auto switches to energy for `N >= 500` (`networkx/drawing/layout.py:591-593`, `networkx/drawing/layout.py:628-636`); Dagua always force-mode (`dagua/layout/ops/pipelines/fr.py:153-177`).
   - Proposed fix option A: add Dagua energy-mode FR with symmetrized absolute adjacency, gravity, and L-BFGS-B-compatible optimizer for `N>=500`. Option B: set reference `method="force"` in `NetworkXSpring.layout_kwargs` for a pure FR-force pairing.
   - Estimate: A large (150-300 LOC plus scipy/torch optimizer tests); B tiny (1-5 LOC plus report semantics).
   - Expected impact: high for large graphs, none for small graphs.

4. Align Dagua output scale to NetworkX adapter scale for all `N`.
   - Evidence: Dagua `50*sqrt(N)` vs NetworkX adapter `500` (`dagua/layout/ops/postprocess.py:399-405`; `dagua/eval/competitors/networkx_competitor.py:50-58`).
   - Proposed fix: set FR finalize factor to constant 500 under fidelity mode, or remove adapter-specific 500 scaling and let both use unit coordinates for algorithm fidelity tests.
   - Estimate: small.
   - Expected impact: high for absolute-coordinate metrics; possible broader benchmark side effects if changed globally.

5. Add explicit duplicate-edge aggregation policy matching NetworkX `DiGraph`.
   - Evidence: Dagua repeated advanced-index assignment (`dagua/layout/ops/preprocess.py:306-310`) vs NetworkX repeated `add_edge` on `DiGraph` (`dagua/eval/competitors/networkx_competitor.py:40-46`).
   - Proposed fix: for FR adjacency, coalesce duplicate directed pairs on CPU in input order and keep the last value, with a regression test for weighted duplicate pairs.
   - Estimate: small/medium (30-80 LOC).
   - Expected impact: medium only on weighted multigraph cases; low on unweighted graphs.

6. Expose `k` in Dagua FR pipeline and wire it through variants.
   - Evidence: NetworkX accepts `k` (`networkx/drawing/layout.py:491-494`, `networkx/drawing/layout.py:680-682`); Dagua hardcodes `k` through unit area (`dagua/layout/ops/force.py:253-295`) with no public override in `layout_fr_pipeline` (`dagua/layout/ops/pipelines/fr.py:182-262`).
   - Proposed fix: add optional `k: Optional[float]` or `force_area` parameter; if `k` is provided, set `force_area = k*k*N`.
   - Estimate: medium (API, op config, tests).
   - Expected impact: low for current defaults, high for future `k` variants.

7. Add fixed-node and partial-position parity only if the sweep starts testing those APIs.
   - Evidence: NetworkX fixed/partial pos support (`networkx/drawing/layout.py:596-615`, `networkx/drawing/layout.py:712-715`); Dagua full `pos` only (`dagua/layout/ops/pipelines/fr.py:236-249`).
   - Proposed fix: accept `fixed: Optional[Sequence[int]]` and partial `pos` mapping/tensor mask; zero selected `delta_pos`; skip final rescale when fixed is provided.
   - Estimate: medium/large (100-200 LOC plus behavior tests).
   - Expected impact: none for current benchmark, high if API parity is required.

8. Guard convergence equivalence with a test near threshold.
   - Evidence: Dagua recomputes `delta_pos` for convergence (`dagua/layout/ops/converge.py:301-306`) rather than storing the exact displacement generated by `ApplyDisplacement` (`dagua/layout/ops/force.py:2673-2678`); NetworkX checks the exact delta (`networkx/drawing/layout.py:711-719`).
   - Proposed fix: store last `delta_pos` in `state.extras` in `ApplyDisplacement` and have `FRConvergenceCheck` use it.
   - Estimate: small/medium.
   - Expected impact: very low; improves robustness against future op insertions.

9. Split benchmark goals: "legacy quality FR" vs "NetworkX-compatible FR".
   - Evidence: Dagua default selector intentionally preserves legacy 200-step quality when canonical 50-step loses directed consistency or composite score (`dagua/layout/ops/pipelines/fr.py:83-124`, `dagua/layout/ops/pipelines/fr.py:265-332`).
   - Proposed fix: keep current default for product quality, but add a named fidelity variant that removes Dagua-specific selection.
   - Estimate: small.
   - Expected impact: improves interpretability; avoids future accidental regressions in the fidelity report.

## 13. Recommended Round 22+ Fix Scope

Recommended bundle for one follow-up round:

1. Add a `networkx_compat: bool = False` or equivalent config to `layout_fr_pipeline` / `FRFinalizePositions`.
   - In compat mode: final rescale should match `networkx.rescale_layout(..., scale=1)` exactly before the benchmark adapter scale decision, or emit unit-scale coordinates and update both competitors to compare unit-scale positions.
   - Primary refs: Dagua finalization (`dagua/layout/ops/postprocess.py:398-405`), NetworkX rescale (`networkx/drawing/layout.py:1918-1924`), NetworkX adapter scaling (`dagua/eval/competitors/networkx_competitor.py:50-58`).

2. Register a strict `classic_fr_nx` or `classic_fr_steps50_nxscale` variant for fidelity measurement.
   - It should call `layout_fr_pipeline(steps=50, seed=seed)` directly, not the default selector.
   - Primary refs: default selector (`dagua/layout/ops/pipelines/fr.py:300-332`), classic spec (`dagua/eval/competitors/classic_competitor.py:153-158`), NetworkX defaults (`dagua/eval/competitors/networkx_competitor.py:147-152`).

3. Decide large-graph semantics explicitly.
   - If the goal is "Fruchterman-Reingold force law", force NetworkX `method="force"` in the reference variants so both sides use the dense/sparse force algorithm.
   - If the goal is "NetworkX spring_layout default", implement or call an energy-mode equivalent for `N>=500`, including absolute symmetrized weights and component gravity.
   - Primary refs: NetworkX auto switch (`networkx/drawing/layout.py:591-593`), energy cost/gradient (`networkx/drawing/layout.py:831-873`), Dagua force-only pipeline (`dagua/layout/ops/pipelines/fr.py:153-177`).

4. Add two regression tests:
   - Tiny directed path: Dagua compat 50-step output matches NetworkX dense after identical finalization within float32 tolerance.
   - Weighted duplicate directed edge: Dagua adjacency coalescing matches `nx.DiGraph` last-edge semantics.
   - Primary refs: Dagua adjacency (`dagua/layout/ops/preprocess.py:306-310`), NetworkX adapter graph construction (`dagua/eval/competitors/networkx_competitor.py:35-47`).

Do not spend Round 22 on fixed-node support unless the exhaustive sweep includes fixed/partial-pos NetworkX variants. It is real API divergence, but it is not currently visible in the `fr_steps*` benchmark definitions (`dagua/eval/variants.py:335-379`).

## Assumptions

- "Reference `nx_spring`" means the installed NetworkX 3.6.1 `spring_layout` used by `dagua/eval/competitors/networkx_competitor.py`, not an older NetworkX release.
- Current FR fidelity variants are the four entries in `dagua/eval/variants.py:335-379`; no hidden fixed-node, custom-`k`, custom-`scale`, or energy-mode variants were assumed.
- Because this round is diagnosis-only, I did not edit source code or run the project quality gates. Verification was limited to source inspection, one small read-only parity probe, and file existence/size checks.

## Knowledge

- Dagua's current FR dense core is already an extremely close port of NetworkX dense `_fruchterman_reingold` when configured as 50 iterations with NumPy initialization.
- The main remaining differences are not the force formula. They are benchmark and wrapper semantics: default 200/50 selector, raw output scale, NetworkX `method="auto"` energy switch at 500 nodes, and edge aggregation/API coverage.
- `classic_fr_steps200` / `classic_fr_steps500` being `partial_match` in the report (`eval_output/fidelity_report/report.md:34-37`) is consistent with comparing longer Dagua FR runs to longer NetworkX runs and/or operating near stochastic/layout basin floors, not with an obvious sign error in the force law.
