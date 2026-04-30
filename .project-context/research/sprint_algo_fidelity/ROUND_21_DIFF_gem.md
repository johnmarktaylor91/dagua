# Round 21 Adversarial Diff: GEM

Pairing: dagua `classic_gem` vs reference `ogdf_gem`.

Scope: diagnosis only. No source changes were made for this round.

## 1. Files Read

Dagua implementation and wiring:

- `dagua/layout/ops/gem.py`: registered GEM ops, constants, sequential path, batched fallback, final normalization. Key refs: constants/config at `dagua/layout/ops/gem.py:26-72`, initialization at `dagua/layout/ops/gem.py:310-372`, state prep at `dagua/layout/ops/gem.py:375-429`, sequential solve at `dagua/layout/ops/gem.py:432-640`, batched solve at `dagua/layout/ops/gem.py:679-1082`, finalization at `dagua/layout/ops/gem.py:1085-1139`.
- `dagua/layout/ops/pipelines/gem.py`: pipeline composition and public runner. Key refs: pipeline sequence at `dagua/layout/ops/pipelines/gem.py:26-60`, public signature at `dagua/layout/ops/pipelines/gem.py:63-125`.
- `dagua/layout/ops/graph_utils.py`: shared device, extent, normalization, undirected adjacency helper. Key refs: device at `dagua/layout/ops/graph_utils.py:136-158`, normalization at `dagua/layout/ops/graph_utils.py:166-186`, extent at `dagua/layout/ops/graph_utils.py:194-213`, adjacency at `dagua/layout/ops/graph_utils.py:226-268`.
- `dagua/layout/ops/state.py`: problem and state fields. Key refs: `LayoutProblem` at `dagua/layout/ops/state.py:113-165`, `SolveState` fields at `dagua/layout/ops/state.py:272-330`.
- `dagua/layout/_archive/classic/gem.py`: archive monolith used as historical dagua reference. Key refs: constants at `dagua/layout/_archive/classic/gem.py:16-33`, sequential helper at `dagua/layout/_archive/classic/gem.py:620-690`, sequential loop at `dagua/layout/_archive/classic/gem.py:779-855`, public layout at `dagua/layout/_archive/classic/gem.py:858-984`.
- `dagua/eval/competitors/classic_competitor.py`: adapter for `classic_gem`. Key refs: layout spec at `dagua/eval/competitors/classic_competitor.py:209-213`, generic variant dispatch at `dagua/eval/competitors/classic_competitor.py:54-97`, direct `ClassicGEM.layout()` at `dagua/eval/competitors/classic_competitor.py:1240-1292`.
- `dagua/eval/competitors/ogdf_competitor.py`: adapter for `ogdf_gem`. Key refs: runner resolution at `dagua/eval/competitors/ogdf_competitor.py:19-45`, JSON edge export at `dagua/eval/competitors/ogdf_competitor.py:48-64`, subprocess runner at `dagua/eval/competitors/ogdf_competitor.py:105-171`, seed-ignore in `_OGDFBase.layout()` at `dagua/eval/competitors/ogdf_competitor.py:179-217`, class registration at `dagua/eval/competitors/ogdf_competitor.py:246-252`.
- `dagua/eval/competitors/base.py`: competitor API and runtime seed helper. Key refs: `layout()` contract at `dagua/eval/competitors/base.py:34-58`, `layout_with_variant()` default at `dagua/eval/competitors/base.py:64-91`, runtime seed at `dagua/eval/competitors/base.py:96-119`.
- `dagua/eval/variants.py`: GEM variants and stochastic flags. Key refs: `classic_gem_iters100/500/2000` variants at `dagua/eval/variants.py:968-1000`, stochastic map at `dagua/eval/variants.py:1820-1865`.
- `dagua/graph.py`: graph node-size storage and computation. Key refs: graph fields at `dagua/graph.py:68-130`, node-size invalidation at `dagua/graph.py:397-408`, node-size computation at `dagua/graph.py:933-1086`.
- `dagua/styles.py`: default node style and graphviz-strict node style. Key refs: default `NodeStyle` geometry at `dagua/styles.py:284-360`, graphviz strict default style at `dagua/styles.py:915-945`.

Dagua harness and existing analysis:

- `scripts/ogdf_runner.cpp`: standalone OGDF runner used by `ogdf_gem`. Key refs: graph construction at `scripts/ogdf_runner.cpp:203-217`, GEM dispatch at `scripts/ogdf_runner.cpp:145-152`, seeding and initial coordinates at `scripts/ogdf_runner.cpp:219-228`, JSON output at `scripts/ogdf_runner.cpp:232-240`.
- `eval_output/fidelity_report/report.md`: current verdict summary. Key refs: GEM rows at `eval_output/fidelity_report/report.md:38-40`.
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`: sprint context. Key refs: OGDF seed-ignore known issue at `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:194-198`.

Reference implementation:

- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h`: public parameters, defaults, private helpers. Key refs: option table at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:60-105`, member fields at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:111-143`, setters at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:160-275`, degree weight helper at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-289`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp`: actual OGDF implementation. Key refs: constructor defaults at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-71`, connected-component loop at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:109-238`, main loop at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186`, impulse at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:240-289`, temperature update at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:291-342`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/basic.cpp`: OGDF global RNG. Key refs: global `std::mt19937` and `randomSeed()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/basic.cpp:120-133`.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h`: OGDF permutation implementation. Key refs: `Array::permute` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h:953-968`.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/SList.h`: SList permutation wrapper. Key refs: public permute at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/SList.h:796-806`, array-backed shuffle at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/SList.h:1106-1130`.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/List.h`: List permutation analog. Key refs: public permute at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/List.h:1402-1412`, array-backed shuffle at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/List.h:1767-1790`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp`: default sizes and separations. Key refs: node width/height and separation at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:38-51`.
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/LayoutStandards.h`: accessors for default sizes/separations. Key refs: node separation accessor at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/LayoutStandards.h:180-191`, component separation accessor at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/LayoutStandards.h:193-203`.
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp`: default node graphics initialization. Key refs: default x/y/width/height initialization at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp:94-99`.

## 2. Overall Pipeline Structure

OGDF pipeline:

1. `GEMLayout::call()` exits immediately for an empty graph at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:109-113`.
2. It clears all edge bends at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:115-116`.
3. It computes connected components on the original graph and stores nodes per component at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:121-130`.
4. It lays out each connected component independently by inserting that component into a `GraphCopy` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:136-148`.
5. For each component, it initializes impulse, skew, local temperature, global temperature, weighted barycenter, and trigonometric thresholds at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:150-168`.
6. It runs a one-node-at-a-time randomized Gauss-Seidel loop for up to `m_numberOfRounds` node updates, drawing a fresh random permutation when the list is exhausted at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186`.
7. It copies component coordinates back, shifts the component so its padded lower-left corner is positive, records bounding boxes, then packs connected components with `TileToRowsCCPacker` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229`.

Dagua pipeline:

1. `build_gem_pipeline()` creates a pure ops pipeline: fixed steps, initialize positions, prepare state, sequential solve, batched solve, finalize positions at `dagua/layout/ops/pipelines/gem.py:50-58`.
2. `layout_gem_pipeline()` creates a `LayoutProblem`, a default `SolveState`, and a CPU `RuntimeContext`, then applies the pipeline at `dagua/layout/ops/pipelines/gem.py:113-125`.
3. `InitializeGEMPositions` special-cases empty and singleton graphs and otherwise initializes all nodes with CPU Torch Gaussian noise at `dagua/layout/ops/gem.py:351-371`.
4. `GEMPrepareState` computes a dagua layout extent, caps requested iterations to 30,000, chooses an output device, and selects sequential vs batched branch by `N <= 5000` at `dagua/layout/ops/gem.py:420-428`.
5. `GEMSequentialSolve` runs the exact-ish one-node path for `N <= 5000` at `dagua/layout/ops/gem.py:481-640`; `GEMBatchedSolve` runs a vectorized fallback above that cutoff at `dagua/layout/ops/gem.py:1054-1082`.
6. `GEMFinalizePositions` recenters and rescales all positions to a dagua extent at `dagua/layout/ops/gem.py:1130-1139`, using `normalize_positions()` from `dagua/layout/ops/graph_utils.py:166-186`.

Structural divergences:

- OGDF always uses the same sequential GEM algorithm for every component size, subject only to `numberOfRounds` and temperature stop at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186`. Dagua switches to a batched approximation for `N > 5000` at `dagua/layout/ops/gem.py:425-428` and `dagua/layout/ops/gem.py:1054-1082`.
- OGDF decomposes disconnected graphs into connected components, lays out each component independently, and packs them with component padding at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:121-229`. Dagua treats the whole graph as one system with global repulsion/gravity and final whole-graph normalization; no connected-component split appears in `dagua/layout/ops/pipelines/gem.py:50-58` or `dagua/layout/ops/gem.py:420-428`.
- OGDF output coordinates are the raw packed coordinates after component shifts at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:202-229`. Dagua normalizes coordinates after solving at `dagua/layout/ops/gem.py:1137-1139`.
- OGDF runner initializes `GraphAttributes` coordinates before `layout.call()` at `scripts/ogdf_runner.cpp:219-228`. Dagua initializes internally in `InitializeGEMPositions` at `dagua/layout/ops/gem.py:364-371`.

## 3. Energy / Loss / Objective

Neither side implements GEM as a scalar loss minimized by an optimizer. Both are force/impulse simulations with adaptive local temperatures.

Per-term comparison:

| Term | OGDF formula | Dagua formula | Match |
| --- | --- | --- | --- |
| Degree weight | `weight(v) = degree(v) / 2.5 + 1.0` in header helper at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-283`. | `_compute_degree_weights()` counts source and target endpoints, then divides by `degree_divisor` and adds `degree_offset` at `dagua/layout/ops/gem.py:165-172`. | Mostly yes for non-self-loop simple/multigraph degree semantics. Self-loop semantics may diverge depending on OGDF `v->degree()` handling; dagua skips self-loops in adjacency but counts both endpoints in degree at `dagua/layout/ops/gem.py:165-172`. |
| Desired length | `desiredLength = m_desiredLength + length(AG.height(v), AG.width(v))` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:246-248`. `m_desiredLength` defaults to `LayoutStandards::defaultNodeSeparation()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:62-63`; that is 20.0 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:50-51`. | `_compute_node_desired_lengths()` computes a node diagonal from `problem.node_sizes` and adds `base_desired_length=20.0` at `dagua/layout/ops/gem.py:202-221`; config default is at `dagua/layout/ops/gem.py:38-64`. | Yes if dagua `node_sizes` match OGDF `GraphAttributes` width/height. In the OGDF runner, default width/height are 20.0 each via `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp:94-99` and `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:38-39`; dagua may have computed label/style-dependent sizes through `dagua/graph.py:933-1086`. |
| Gravity | `m_newImpulseX = (m_barycenterX / n - AG.x(v)) * m_gravitationalConstant` and same for Y at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:250-252`. | Sequential path computes `(barycenter / max(num_nodes,1) - position) * gravitational_constant` at `dagua/layout/ops/gem.py:551-556`; batched path does the same at `dagua/layout/ops/gem.py:808-811`. | Yes for connected single-component sequential path. For disconnected graphs, OGDF uses per-component `n`; dagua uses whole-graph `num_nodes`. |
| Random disturbance | OGDF computes `maxIntDisturbance = int(m_maximalDisturbance * 10000)`, samples two uniform ints, and adds `/10000.0` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:254-258`. Default is 0 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:62-63`. | Dagua raises `NotImplementedError` if non-zero disturbance is configured at `dagua/layout/ops/gem.py:535-536`; config default is 0.0 at `dagua/layout/ops/gem.py:66-68`. | Yes for default; unsupported for non-default. |
| Repulsion | For all other nodes in the component, `deltaX * desiredSqu / deltaSqu` and `deltaY * desiredSqu / deltaSqu` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:260-272`. | Sequential path matches the same loop and formula at `dagua/layout/ops/gem.py:558-567`. Batched exact path uses vectorized `delta * desired_square / distance_square` at `dagua/layout/ops/gem.py:762-772`. Sampled large-graph path uses sampled neighbors and `ideal_distance^2 / distance`, not per-node desired length squared, at `dagua/layout/ops/gem.py:740-759`. | Sequential yes, batched fallback no for large graphs. |
| Attraction formula 1 | For each adjacency entry, subtract `deltaX * delta / (desiredLength * weight(v))` and Y analog at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:274-283`. Default formula is 1 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:68-69`. | Sequential path subtracts `edge_weight * delta * distance / (desired_length * node_weight)` at `dagua/layout/ops/gem.py:569-581`. Batched path computes source and target forces at `dagua/layout/ops/gem.py:793-807`. | Yes for unweighted simple/multiedges after adjacency aggregation. Dagua adds optional `edge_weights`, which OGDF runner does not pass. |
| Attraction formula 2 | OGDF alternative subtracts `deltaX * deltaSqu / (desiredSqu * weight(v))` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:283-287`; setter accepts 1 or 2 at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:255-263`. | Dagua has the formula-2 branch in sequential path at `dagua/layout/ops/gem.py:582-590`; config default is 1 at `dagua/layout/ops/gem.py:67-69`. | Potentially yes internally, but no public pipeline parameter exposes this. |
| Temperature-scaled move | OGDF scales impulse by `m_localTemperature[v] / impulseLength` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:295-303`. | Dagua sequential path uses `raw * local_temperature / impulse_length` at `dagua/layout/ops/gem.py:592-602`; batched uses vectorized norm scaling at `dagua/layout/ops/gem.py:868-874`. | Sequential yes. Batched simultaneous update changes semantics. |
| Final objective/packing | OGDF component shift and pack, no whole-graph recenter/scale at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229`. | Dagua centers by mean and scales by max absolute span to `extent` at `dagua/layout/ops/graph_utils.py:181-186`, called by `dagua/layout/ops/gem.py:1137-1139`. | No. RMSD evaluation may align/normalize externally, but raw outputs differ. |

The highest-impact objective-level divergences are disconnected-component handling and final normalization, because they affect absolute component relationships even if the per-component force equations match.

## 4. Force / Gradient Computation

OGDF force computation is imperative and node-local:

- `computeImpulse()` receives one node `v` and current component positions at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:240-241`.
- It computes a weighted-barycenter gravity impulse at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:250-252`.
- It adds random disturbance, normally zero by default, at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:254-258`.
- It scans every other node for repulsion at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:260-272`.
- It scans every adjacency entry of `v` for attraction at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:274-288`.
- `updateNode()` immediately mutates `AG.x(v)`/`AG.y(v)` and updates barycenter at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:291-342`.

Dagua sequential force computation:

- `GEMSequentialStep.apply()` runs a single monolithic loop rather than smaller ops because immediate positions and barycenter are required; this is documented at `dagua/layout/ops/gem.py:432-441`.
- It computes gravity at `dagua/layout/ops/gem.py:551-556`, repulsion at `dagua/layout/ops/gem.py:558-567`, attraction at `dagua/layout/ops/gem.py:569-590`, movement at `dagua/layout/ops/gem.py:592-602`, and barycenter/temperature update at `dagua/layout/ops/gem.py:604-634`.
- Sequential math is intentionally close to OGDF, but it uses Torch-generated initial positions and Torch-generated node permutations, so even exact equations do not trace the same trajectory.

Dagua batched force computation:

- `GEMComputeImpulse` computes all-node impulse fields at once at `dagua/layout/ops/gem.py:698-815`.
- For `num_nodes > 2000`, it samples `sampled_repulsion_neighbors=96` random neighbors and uses a sampled ideal distance at `dagua/layout/ops/gem.py:740-759`.
- For `num_nodes <= 2000` in the batched branch, it uses full vectorized all-pairs repulsion at `dagua/layout/ops/gem.py:761-772`.
- Attraction is vectorized with `index_add_` at `dagua/layout/ops/gem.py:774-807`.
- Temperature update and displacement are applied to all nodes simultaneously at `dagua/layout/ops/gem.py:818-915` and `dagua/layout/ops/gem.py:918-962`.

The batched branch is not a faithful OGDF implementation because OGDF's updated position of node `v` affects all later nodes in the same permutation round, and the barycenter is updated immediately at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:181-185` and `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:301-307`.

## 5. Initialization

OGDF reference via runner:

- The runner creates nodes and edges first at `scripts/ogdf_runner.cpp:203-217`.
- It calls `ogdf::setSeed(42)` and `std::srand(42)` at `scripts/ogdf_runner.cpp:219-222`.
- It sets each initial coordinate to `std::rand() % 1000 / 10.0`, so coordinates are discrete values in `{0.0, 0.1, ..., 99.9}` at `scripts/ogdf_runner.cpp:223-228`.
- OGDF `GEMLayout` itself constructs `m_rng(randomSeed())` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-71`. `randomSeed()` draws from a global `std::mt19937`, returning `7 * s_random() + 3` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/basic.cpp:120-133`.
- Because the runner sets the OGDF global seed after constructing `GEMLayout` (`scripts/ogdf_runner.cpp:149-151` constructs it inside `runLayout`, called after seeding at `scripts/ogdf_runner.cpp:219-230`), the GEM internal `std::minstd_rand` is deterministic but seeded with a transformed mt19937 draw, not with literal 42.

Dagua:

- Empty graph returns an empty float32 tensor, and singleton returns zero at `dagua/layout/ops/gem.py:354-362`.
- Nontrivial graphs create a CPU `torch.Generator`, seed it with `problem.seed`, and draw `torch.randn((N, 2), dtype=torch.float32)` at `dagua/layout/ops/gem.py:364-371`.
- The public pipeline default seed is 42 at `dagua/layout/ops/pipelines/gem.py:63-70`.
- The classic adapter passes `seed=self._layout_seed(seed)` and `_layout_seed(None)` returns 42 at `dagua/eval/competitors/classic_competitor.py:29-42` and `dagua/eval/competitors/classic_competitor.py:1276-1282`.

Initialization verdict:

- RNG type: no match. OGDF runner uses C `rand()` for initial positions and OGDF `std::minstd_rand` for permutations; dagua uses PyTorch CPU generators for both initialization and permutations.
- Distribution: no match. OGDF runner uses uniform discrete `[0, 99.9]`; dagua uses standard normal centered at 0.
- Scale: no match. OGDF initial coordinate span is roughly 100 units; dagua initial span is usually a few standard deviations before solving.
- Deterministic seed plumbing: partial. Dagua honors benchmark seeds; the OGDF adapter explicitly discards `seed` at `dagua/eval/competitors/ogdf_competitor.py:179-204`, and the runner hardcodes 42 at `scripts/ogdf_runner.cpp:219-228`.

## 6. Iteration / Convergence

OGDF:

- `m_numberOfRounds` defaults to 30,000 in the constructor at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-58`. The header comment says 20,000 at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:67-69`, so the actual compiled default is 30,000.
- `m_minimalTemperature` defaults to 0.005 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:57-59`.
- Main loop stops when `m_globalTemperature <= m_minimalTemperature` or the counter reaches zero at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186`.
- The counter is per connected component because it is reset inside the component loop at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:136-171`.
- Node order is a list permutation consumed via `popFrontRet()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:172-179`.

Dagua:

- Pipeline `max_iters` defaults to 500 at `dagua/layout/ops/pipelines/gem.py:26-32` and public runner default is 500 at `dagua/layout/ops/pipelines/gem.py:63-70`.
- The classic adapter's direct `ClassicGEM.layout()` hardcodes `max_iters=500` at `dagua/eval/competitors/classic_competitor.py:1276-1281`.
- The generic classic spec also defaults `max_iters=500` at `dagua/eval/competitors/classic_competitor.py:209-213`.
- Variant sweep has `max_iters` 100, 500, and 2000 at `dagua/eval/variants.py:968-1000`.
- `GEMPrepareState` caps iterations to 30,000 at `dagua/layout/ops/gem.py:420-423`.
- Sequential path stops when `global_temperature <= minimal_temperature` or `rounds_remaining == 0` at `dagua/layout/ops/gem.py:538-640`.
- Batched path runs `Repeat(n=capped_iters, ...)` at `dagua/layout/ops/gem.py:1069-1082`; it sets an early-stop flag when mean temperature drops below the threshold at `dagua/layout/ops/gem.py:908-909`, but `GEMBatchedSolve` does not include `GEMConvergenceCheck` in the repeated ops at `dagua/layout/ops/gem.py:1070-1079`. `GEMComputeImpulse` sees the early-stop flag and writes zero impulse at `dagua/layout/ops/gem.py:731-733`, so it still pays loop overhead but stops moving.

Iteration verdict:

- Base dagua `classic_gem` vs base `ogdf_gem` uses 500 node updates vs OGDF's 30,000 unless comparing a specific variant. This is a high-impact divergence for raw adapters.
- Even `classic_gem_iters2000` remains far below OGDF default 30,000.
- OGDF does 30,000 updates per connected component; dagua does one global update budget across the whole graph.
- Dagua sequential consumes permutations by `torch.randperm(...).tolist()` then `pop()` at `dagua/layout/ops/gem.py:540-545`; OGDF consumes `SList` front after `SList::permute()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:172-179`. This is not only different RNG but likely a different permutation algorithm/order.

## 7. Hyperparameter Alignment Table

| Parameter / behavior | Dagua default | OGDF reference default / runner behavior | Match? | Evidence |
| --- | --- | --- | --- | --- |
| Node updates / rounds | Pipeline 500; cap 30,000 | 30,000 actual constructor default | No | `dagua/layout/ops/pipelines/gem.py:26-32`, `dagua/layout/ops/gem.py:420-423`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-58` |
| Variant rounds | 100, 500, 2000 | Original params `{}` so OGDF default 30,000 | No | `dagua/eval/variants.py:968-1000`; `dagua/eval/competitors/ogdf_competitor.py:179-217` |
| Minimal temperature | 0.005 | 0.005 | Yes | `dagua/layout/ops/gem.py:64-65`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:57-59` |
| Initial temperature | 12.0 | 12.0 actual constructor default | Yes | `dagua/layout/ops/gem.py:64-65`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:57-60` |
| Header-documented initial temperature | 12.0 | Header says 10.0 | No vs docs, yes vs implementation | `dagua/layout/ops/gem.py:64-65`; `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:73-75` |
| Gravity constant | 1/16 | 1/16 | Yes | `dagua/layout/ops/gem.py:66-66`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:59-61` |
| Base desired length | 20.0 | `LayoutStandards::defaultNodeSeparation()` = 20.0 | Yes | `dagua/layout/ops/gem.py:63-64`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:62-63`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:50-51` |
| Node diagonal addition | From `problem.node_sizes`; zero if missing | From `GraphAttributes` width/height; runner defaults width=20, height=20 | Conditional / often no | `dagua/layout/ops/gem.py:202-221`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp:94-99`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:38-39` |
| Max disturbance | 0.0; non-zero unsupported | 0.0; non-zero implemented | Default yes, non-default no | `dagua/layout/ops/gem.py:67-68`, `dagua/layout/ops/gem.py:535-536`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:62-63`, `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:254-258` |
| Rotation angle threshold | `sin(pi/2 + pi/6)` | `sin(pi/2 + rotationAngle/2)`, default rotationAngle=pi/3 | Yes | `dagua/layout/ops/gem.py:71-72`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:64-67`, `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:166-167` |
| Oscillation threshold | `cos(pi/4)` | `cos(oscillationAngle/2)`, default oscillationAngle=pi/2 | Yes | `dagua/layout/ops/gem.py:71-72`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:64-67`, `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:166-167` |
| Rotation sensitivity | 0.01 | 0.01 | Yes | `dagua/layout/ops/gem.py:69-70`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:64-67` |
| Oscillation sensitivity | 0.3 | 0.3 | Yes | `dagua/layout/ops/gem.py:69-70`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:64-67` |
| Attraction formula | 1 | 1 | Yes | `dagua/layout/ops/gem.py:67-69`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:67-69` |
| Degree divisor/offset | 2.5, 1.0 | `degree()/2.5 + 1.0` | Yes except self-loop interpretation risk | `dagua/layout/ops/gem.py:60-63`, `dagua/layout/ops/gem.py:165-172`; `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-283` |
| Connected component separation | None in GEM ops | `LayoutStandards::defaultCCSeparation()` = 30.0 | No | `dagua/layout/ops/pipelines/gem.py:50-58`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:69-70`, `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:50-51` |
| Page ratio | None | 1.0 | No | `dagua/layout/ops/pipelines/gem.py:50-58`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:69-70` |
| Initial position RNG | Torch CPU `randn`, seed configurable | C `rand()` uniform-ish discrete, hardcoded seed 42 | No | `dagua/layout/ops/gem.py:364-371`; `scripts/ogdf_runner.cpp:219-228` |
| Permutation RNG | Torch CPU `randperm`, seed configurable | `std::minstd_rand` seeded by `randomSeed()` from global mt19937 | No | `dagua/layout/ops/gem.py:531-544`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-71`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/basic.cpp:120-133` |
| Final normalization | Center and scale to extent | Shift/pack components; no normalize-to-extent | No | `dagua/layout/ops/gem.py:1137-1139`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229` |
| Large graph exactness | Batched fallback above 5000, sampled repulsion above 2000 within batched path | Sequential exact for all sizes | No | `dagua/layout/ops/gem.py:425-428`, `dagua/layout/ops/gem.py:740-759`, `dagua/layout/ops/gem.py:1054-1082`; `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186` |
| Edge weights | Supported in dagua attraction | Runner does not pass weights; OGDF layout uses unweighted graph adjacency | No extra feature | `dagua/layout/ops/gem.py:293-295`, `dagua/layout/ops/gem.py:569-590`; `scripts/ogdf_runner.cpp:138-143` |

## 8. Edge Cases

Empty graph:

- OGDF `GEMLayout::call()` returns without mutation when `G.empty()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:109-113`. The dagua OGDF adapter separately returns `torch.zeros((0, 2), dtype=torch.float32)` before subprocess at `dagua/eval/competitors/ogdf_competitor.py:131-132`.
- Dagua returns an empty float32 tensor and marks converged at `dagua/layout/ops/gem.py:354-357`.
- Verdict: equivalent shape; raw value empty.

Singleton:

- OGDF runner initializes one node to `rand()%1000/10.0` at `scripts/ogdf_runner.cpp:223-228`, then GEM lays out the single-node component. In `computeImpulse()`, gravity is `(weight*x/n - x) * gravity`. With degree 0, weight=1, n=1, gravity is zero; repulsion and attraction are absent at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:250-288`. Then component shift subtracts `minX = x - width/2 - minDistCC` and `minY = y - height/2 - minDistCC`, so the final singleton coordinate becomes approximately `(width/2 + minDistCC, height/2 + minDistCC)`, before packing offset, via `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229`.
- Dagua returns `[0, 0]` for singleton at `dagua/layout/ops/gem.py:359-362`, and finalization returns early when converged at `dagua/layout/ops/gem.py:1130-1131`.
- Verdict: raw output no; Procrustes/RMSD harness may erase translation/scale in pair metrics, but adapter output differs.

Self-loops:

- OGDF iterates `for (adjEntry adj : v->adjEntries)` and `u = adj->twinNode()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:274-277`. A self-loop's twin node is the same node, so `delta=0`; formula 1 subtracts zero at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:280-283`. Degree weight uses `v->degree()` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-283`, so self-loop impact depends on OGDF degree semantics.
- Dagua explicitly skips self-loops in `build_undirected_adjacency()` at `dagua/layout/ops/graph_utils.py:260-264`, but `_compute_degree_weights()` counts both source and target endpoints for every edge, including self-loops, at `dagua/layout/ops/gem.py:165-172`.
- Verdict: attraction likely equivalent zero, degree-weight semantics need targeted confirmation. If OGDF self-loop degree is 2, dagua matches; if 1, dagua overweights self-loops.

Multi-edges:

- OGDF attraction iterates adjacency entries at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:274-288`, so parallel edges contribute repeatedly. Degree weight uses graph degree at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-283`.
- Dagua adjacency accumulates duplicate-edge weights additively in a map at `dagua/layout/ops/graph_utils.py:250-268`. Sequential attraction then multiplies by the accumulated `edge_weight` at `dagua/layout/ops/gem.py:569-590`.
- Verdict: equivalent for unweighted parallel edges in sequential path, modulo sorted neighbor order and floating summation order. Batched path uses `index_add_` over original edges at `dagua/layout/ops/gem.py:774-807`, so multiedge summation order differs from sequential sorted adjacency.

Disconnected components:

- OGDF explicitly splits components, solves each independently, shifts each component, and packs components at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:121-229`.
- Dagua uses all nodes in one force system and all-pairs repulsion regardless of graph connectivity at `dagua/layout/ops/gem.py:558-567` and `dagua/layout/ops/gem.py:762-772`.
- Verdict: no. This is one of the largest residual mismatches for disconnected graphs.

Weighted edges:

- OGDF runner JSON payload includes only `"nodes"`, `"edges"`, and `"algorithm"` at `dagua/eval/competitors/ogdf_competitor.py:138-143`; `scripts/ogdf_runner.cpp:80-130` parses only edge endpoints. OGDF GEM itself uses unweighted graph adjacency at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:274-288`.
- Dagua public pipeline accepts `edge_weights` at `dagua/layout/ops/pipelines/gem.py:63-70`, stores them in `LayoutProblem` at `dagua/layout/ops/pipelines/gem.py:113-119`, and applies them in sequential attraction at `dagua/layout/ops/gem.py:569-590` and batched attraction at `dagua/layout/ops/gem.py:800-807`.
- Verdict: no exact OGDF counterpart in current adapter.

Node sizes:

- OGDF runner creates `GraphAttributes` with `nodeGraphics`, which initializes x/y to 0 and width/height to default node width/height at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp:94-99`; those defaults are 20/20 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:38-39`.
- Dagua uses `graph.node_sizes` from `DaguaGraph`, which can be computed from labels/styles at `dagua/graph.py:933-1086`; defaults depend on active theme/style, not OGDF's fixed 20/20.
- Verdict: likely no for many benchmark graphs unless their node sizes were deliberately graphviz/OGDF aligned.

## 9. Numerical Precision

OGDF:

- Graph coordinates and algorithm fields are `double`: `m_barycenterX`, `m_newImpulseX`, `m_globalTemperature`, etc. are declared as `double` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:127-140`.
- `computeImpulse()` uses local `double` variables at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:240-248`.
- `length()` returns `double` using `sqrt` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-280`.
- The runner prints JSON positions using default stream formatting at `scripts/ogdf_runner.cpp:232-240`, then Python converts them to `torch.float32` at `dagua/eval/competitors/ogdf_competitor.py:162-171`. This means OGDF internal precision is double, but benchmark adapter output is float32 after a decimal-string boundary.

Dagua:

- Initialization is float32 at `dagua/layout/ops/gem.py:366-370`.
- Sequential path converts positions to CPU float64 at `dagua/layout/ops/gem.py:487-488`, node desired lengths to float64 at `dagua/layout/ops/gem.py:496-502`, temperatures to float64 at `dagua/layout/ops/gem.py:510-527`, and impulse state to float64 at `dagua/layout/ops/gem.py:527-528`.
- Degree weights in sequential path are computed as float32 at `dagua/layout/ops/gem.py:489-495`, then multiplied with float64 positions for barycenter at `dagua/layout/ops/gem.py:529-530`. The archive version explicitly converted degree weights to float64 at `dagua/layout/_archive/classic/gem.py:809-815`, but the current ops path leaves them float32 until promotion.
- Sequential output casts to float32 at `dagua/layout/ops/gem.py:638-639`; final normalization casts to float32 at `dagua/layout/ops/gem.py:1137-1139`.
- Batched path is predominantly float32: cache degree weights, desired lengths, temperatures, previous impulse, skew gauge are float32 at `dagua/layout/ops/gem.py:258-279`; impulse math uses tensor operations over float32 positions at `dagua/layout/ops/gem.py:735-815`.

Numerical verdict:

- Sequential dagua and OGDF both mostly compute force updates in double, but dagua starts from float32 Gaussian positions and ends with dagua-specific normalization.
- Dagua's degree weights are float32 before promotion, so degree-derived barycenter values can differ at sub-ULP double scale for larger degrees.
- Batched dagua is float32 and simultaneous, making it numerically and algorithmically different from OGDF double sequential.
- Summation order differs: OGDF iterates graph node and adjacency container order at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:260-288`; dagua sequential loops `range(num_nodes)` for repulsion and sorted adjacency for attraction at `dagua/layout/ops/gem.py:558-590`, while batched uses vectorized reductions and `index_add_` at `dagua/layout/ops/gem.py:762-807`.

## 10. RNG Semantics

Dagua's Torch seed does not produce the same sequence as the reference RNG.

Specific differences:

- OGDF internal GEM RNG is `std::minstd_rand m_rng` declared at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:142-143`.
- The constructor seeds `m_rng(randomSeed())` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-71`.
- `randomSeed()` returns `7 * s_random() + 3`, where `s_random` is a global `std::mt19937` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/basic.cpp:120-133`.
- The runner calls `ogdf::setSeed(42)` before `runLayout()` and hardcodes `std::srand(42)` at `scripts/ogdf_runner.cpp:219-222`.
- OGDF permutations use `SList::permute(m_rng)` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:172-179`. `SList::permute()` stores list elements in an array and calls `Array::permute(0, n-1, rng)` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/SList.h:1106-1123`. `Array::permute()` uses a single uniform distribution over `[0, r-l]` and for each position swaps with `pStart + dist(rng)`, not with a shrinking suffix range, at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h:953-968`.
- Dagua creates a CPU `torch.Generator`, seeds it with `problem.seed`, and uses `torch.randn()` for initialization at `dagua/layout/ops/gem.py:364-371`.
- Dagua sequential creates another CPU `torch.Generator`, seeds it with the same `problem.seed`, and uses `torch.randperm(num_nodes, generator=generator).tolist()` at `dagua/layout/ops/gem.py:531-544`.
- Dagua consumes the permutation using `.pop()` from the end at `dagua/layout/ops/gem.py:540-545`; OGDF consumes from the front via `popFrontRet()` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:172-179`.
- Dagua batched sampled repulsion reseeds a CPU generator every step as `problem.seed + step_index + 1` at `dagua/layout/ops/gem.py:740-751`, which has no OGDF equivalent.

Adapter-level issue:

- `_OGDFBase.layout()` accepts `seed` but deletes it at `dagua/eval/competitors/ogdf_competitor.py:179-204`. The sprint summary already calls this out at `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:194-198`.
- `engine_is_stochastic()` marks `classic_gem` as stochastic and `ogdf_gem` as non-stochastic at `dagua/eval/variants.py:1820-1857`, which matches the current hardcoded runner but prevents within-seed stochastic floor accounting from reflecting real OGDF seed control.

## 11. Edge-Case Bugs

1. **Base adapter iteration mismatch.** `ClassicGEM.layout()` hardcodes `max_iters=500` at `dagua/eval/competitors/classic_competitor.py:1276-1281`, while OGDF default is 30,000 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-58`. The variant rows are intentionally lower-iteration variants, but the base `classic_gem` pairing is not default-aligned.

2. **OGDF adapter ignores seed.** `_OGDFBase.layout()` deletes `seed` at `dagua/eval/competitors/ogdf_competitor.py:179-204`, and the C++ runner hardcodes seed 42 at `scripts/ogdf_runner.cpp:219-228`. This makes `ogdf_gem` deterministic across all benchmark seeds even though OGDF GEM itself is stochastic via `m_rng` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:142-143`.

3. **Dagua initialization distribution is not OGDF runner initialization.** Dagua uses `torch.randn()` at `dagua/layout/ops/gem.py:364-371`; runner uses C `rand()%1000/10.0` at `scripts/ogdf_runner.cpp:223-228`. This affects every graph and likely dominates residual stochastic divergence.

4. **Connected-component behavior missing.** OGDF solves and packs components at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:121-229`; dagua has no component split in pipeline ops at `dagua/layout/ops/pipelines/gem.py:50-58`. This can produce wrong relative placement and different forces for disconnected graphs.

5. **Final normalization not reference-like.** Dagua normalizes to centered bounded extent at `dagua/layout/ops/gem.py:1137-1139` and `dagua/layout/ops/graph_utils.py:181-186`; OGDF shifts/packs components but does not normalize to a symmetric extent at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229`.

6. **Singleton raw output mismatch.** Dagua returns `[0, 0]` for singleton at `dagua/layout/ops/gem.py:359-362`; OGDF runner initializes a nonzero coordinate and then shifts component by node size/component separation through `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229`.

7. **Large-graph fallback is approximate.** Dagua switches to batched mode above 5000 nodes at `dagua/layout/ops/gem.py:425-428` and sampled repulsion for large batched graphs at `dagua/layout/ops/gem.py:740-759`; OGDF stays sequential at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186`. This is a deliberate scalability tradeoff, but it is a fidelity divergence.

8. **Permutation algorithm/order mismatch.** Dagua `torch.randperm(...).pop()` at `dagua/layout/ops/gem.py:540-545` does not match OGDF `SList::permute(m_rng)` and front-pop at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:172-179`; OGDF's array shuffle uses fixed-range swaps at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h:953-968`.

9. **Node-size default mismatch risk.** OGDF runner defaults width/height to 20/20 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp:94-99` and `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/LayoutStandards.cpp:38-39`; dagua may use label/style-computed sizes at `dagua/graph.py:933-1086` and default style padding/font at `dagua/styles.py:284-360`. Desired length can therefore be substantially different.

10. **Batched convergence check not wired.** `GEMConvergenceCheck` exists at `dagua/layout/ops/gem.py:965-1010`, but `GEMBatchedSolve` repeats only compute/update/apply at `dagua/layout/ops/gem.py:1070-1079`. Movement stops through an early-stop flag at `dagua/layout/ops/gem.py:731-733` and `dagua/layout/ops/gem.py:908-909`, but `state.converged` is not updated inside the batched loop.

## 12. Ranked Fix List

1. **Align initialization RNG and distribution to the OGDF runner.**
   - Impact: very high. Every stochastic trajectory begins differently today.
   - Evidence: dagua Torch normal at `dagua/layout/ops/gem.py:364-371`; runner C `rand()%1000/10.0` at `scripts/ogdf_runner.cpp:219-228`.
   - Proposed fix: add an OGDF-runner-compatible initialization mode for GEM, likely a small deterministic C-rand emulation in Python/Torch or a helper seeded exactly like `std::srand(42)` when matching the current runner.
   - Size estimate: M. Requires tests pinning first few coordinates for seed 42 and arbitrary seed if runner is updated.

2. **Expose and honor seed in `ogdf_runner` and `ogdf_competitor`.**
   - Impact: very high for fair stochastic comparisons and residual analysis.
   - Evidence: `_OGDFBase.layout()` deletes seed at `dagua/eval/competitors/ogdf_competitor.py:179-204`; runner hardcodes seed at `scripts/ogdf_runner.cpp:219-228`; known issue noted at `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md:194-198`.
   - Proposed fix: include `"seed"` in payload, parse it in `scripts/ogdf_runner.cpp`, call `ogdf::setSeed(seed)` and `std::srand(seed)`, and stop deleting seed in the adapter.
   - Size estimate: S-M. Requires runner rebuild and adapter regression test.

3. **Align default iteration count for base `classic_gem` or compare via explicit OGDF rounds.**
   - Impact: high. 500 vs 30,000 updates changes convergence depth.
   - Evidence: dagua defaults at `dagua/layout/ops/pipelines/gem.py:26-32` and `dagua/eval/competitors/classic_competitor.py:1276-1281`; OGDF actual default at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:56-58`.
   - Proposed fix: either set dagua base GEM default to 30,000 for fidelity mode, or extend OGDF runner to accept `numberOfRounds` and make variants compare 100/500/2000 to the same OGDF setting.
   - Size estimate: M. Risk is runtime cost; may need benchmark lane gating.

4. **Implement OGDF connected-component solve and packing semantics.**
   - Impact: high for disconnected graphs; low for connected graphs.
   - Evidence: OGDF component split/packing at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:121-229`; dagua global solve at `dagua/layout/ops/gem.py:558-567` and final global normalization at `dagua/layout/ops/gem.py:1137-1139`.
   - Proposed fix: before GEM sequential solve, split by connected components, run GEM per component with its own temperature/counter/permutation state, shift by node bbox and `minDistCC=30`, then pack with an OGDF-compatible or approximate `TileToRowsCCPacker`.
   - Size estimate: L. Packing parity may require reading OGDF packer code.

5. **Replace final normalization with OGDF-style component shift/packing in fidelity mode.**
   - Impact: high for raw coordinate comparisons, moderate if external alignment removes translation/scale.
   - Evidence: dagua final normalization at `dagua/layout/ops/gem.py:1137-1139`; OGDF shift/pack at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:188-229`.
   - Proposed fix: introduce an option for GEM finalizer to skip `normalize_positions()` and instead apply OGDF lower-left shift and connected-component offsets.
   - Size estimate: M-L depending on whether full packer is implemented.

6. **Match OGDF permutation RNG and shuffle algorithm.**
   - Impact: medium-high after initialization/iterations are aligned.
   - Evidence: dagua `torch.randperm` at `dagua/layout/ops/gem.py:540-545`; OGDF `SList::permute(m_rng)` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:172-179`; fixed-range swap at `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h:953-968`.
   - Proposed fix: implement a tiny minstd_rand-compatible generator and OGDF `Array::permute` equivalent, seeded via OGDF `randomSeed()` semantics.
   - Size estimate: M. Needs deterministic fixtures comparing first permutations for small `N`.

7. **Force node-size defaults to OGDF runner dimensions for this pairing.**
   - Impact: medium. Desired length controls both repulsion and attraction scale.
   - Evidence: OGDF width/height 20/20 at `/home/jtaylor/projects/_references/ogdf/src/ogdf/basic/GraphAttributes.cpp:94-99`; dagua uses supplied/computed `node_sizes` at `dagua/layout/ops/gem.py:202-221` and `dagua/graph.py:933-1086`.
   - Proposed fix: in `classic_gem` fidelity mode, default missing node sizes to 20x20 or ensure benchmark graphs compute node sizes matching the OGDF runner's 20x20 when the original runner lacks labels/styles.
   - Size estimate: S-M. Risk: affects normal dagua visual layout if not scoped to classic/fidelity adapter.

8. **Remove or gate the batched fallback for reference-fidelity comparisons.**
   - Impact: medium for large graphs; zero for small graphs.
   - Evidence: branch at `dagua/layout/ops/gem.py:425-428`; sampled repulsion at `dagua/layout/ops/gem.py:740-759`; OGDF exact sequential loop at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-186`.
   - Proposed fix: add `exact=True` or `fidelity_mode=True` to force sequential semantics, perhaps with max-node guard to avoid pathological runtime.
   - Size estimate: S for option, L if optimizing exact sequential for large graphs.

9. **Wire `GEMConvergenceCheck` into the batched pipeline or remove dead op.**
   - Impact: low for coordinate fidelity because early-stop movement already zeros out; medium for state correctness.
   - Evidence: convergence op at `dagua/layout/ops/gem.py:965-1010`; repeated batched ops exclude it at `dagua/layout/ops/gem.py:1070-1079`.
   - Proposed fix: add `GEMConvergenceCheck()` to repeated ops after temperature update, or document/remove the unused op if not needed.
   - Size estimate: S.

10. **Confirm and pin self-loop degree semantics.**
    - Impact: low overall, high for self-loop-specific graphs.
    - Evidence: dagua counts both endpoints at `dagua/layout/ops/gem.py:165-172` and skips self-loop adjacency at `dagua/layout/ops/graph_utils.py:260-264`; OGDF degree helper uses `v->degree()` at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/GEMLayout.h:279-283`.
    - Proposed fix: add a tiny runner fixture or inspect OGDF `Graph` degree semantics for self-loops, then adjust dagua degree counting if needed.
    - Size estimate: S.

## 13. Recommended Round 22+ Fix Scope

Recommended next bundle: **RNG and iteration alignment only**, before touching component packing.

Proposed Round 22 scope:

1. Extend `scripts/ogdf_runner.cpp` and `dagua/eval/competitors/ogdf_competitor.py` to accept and honor `seed`.
2. Add a GEM fidelity initializer in dagua that matches the runner's current initialization distribution exactly, or update both runner and dagua to a documented common initializer if the goal is algorithm rather than current-runner parity.
3. Add OGDF-compatible permutation generation for GEM sequential mode: `std::minstd_rand` plus OGDF fixed-range `Array::permute` behavior.
4. Expose `max_iters=30000` for base `classic_gem` fidelity comparisons, or expose OGDF `numberOfRounds` so `classic_gem_iters100/500/2000` compare against OGDF with the same number of rounds.
5. Add targeted fixtures for `N=2`, `N=3 path`, and `N=4 cycle` that compare first initial coordinates, first permutation, and first one-node update against a small OGDF trace or a compiled runner debug mode.

Rationale:

- These four levers affect every connected graph and are smaller than reproducing `TileToRowsCCPacker`.
- They will make residual differences more interpretable. Today any force-law comparison is confounded by different starting positions, different node order, and different iteration count.
- Connected-component packing should be Round 23+ because it requires broader behavior design: exact OGDF raw-coordinate parity conflicts with dagua's current normalized-layout convention at `dagua/layout/ops/gem.py:1137-1139`.

Assumption used in this diagnosis: RMSD verdicts in `eval_output/fidelity_report/report.md:38-40` are computed after some form of alignment/normalization, because raw outputs have clear initialization, final normalization, and singleton/disconnected differences. I did not inspect the metric implementation in this round because the requested source set centered on GEM implementation and adapters.
