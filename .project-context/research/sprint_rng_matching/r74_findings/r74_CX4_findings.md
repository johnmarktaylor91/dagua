# r74 CX4 findings: classical_mds, gem, maxent_stress

Scope: read-only source and verdict analysis. No repository files were edited.

Data source: `eval_output/fidelity_definitive_r73/per_combo.json` plus reference source under `/home/jtaylor/projects/_references/{igraph,ogdf}`.

## Divergent subgroup counts

### classical_mds: 34 rung-4 combos

- Variants: 17 `classic_classical_mds_default`, 17 `classic_classical_mds_igraph_fidelity`.
- Modes: 30 mode A, 4 mode B.
- Disconnected: 20 combos.
- Connected degenerate: 14 combos, including 2 `degenerate_heavy` combos.
- Disconnected graph pairs: `disconnected_encoder_residual`, `disconnected_label_cycle_collage`, `er_100`, `er_500`, `kitchen_sink_platform_graph`, `multi_component_80`, `parallel_cycles_4x5`, `random_bipartite_60`, `random_dag_200`, `random_dag_50` across both variants.
- Connected degenerate graph pairs: `bipartite_4_3_4`, `center_port_backedge_hub`, `densenet_block`, `org_chart_1_5_4_8`, `petersen_10`, `wide_3_50_3`, `wide_single_layer_1_50_1` across both variants.

### gem: 22 rung-4 combos

- Variants: 20 `classic_gem_iters100`, 2 `classic_gem_iters500`.
- Modes: all 22 mode A.
- Disconnected: 3 combos (`random_dag_50` at iters100/500, `random_dag_200` at iters500).
- Connected: 19 combos, overwhelmingly short-run `iters100`.

### maxent_stress: 3 rung-4 combos

- Variants: `classic_maxent_stress_default`, `classic_maxent_stress_steps400`, `classic_maxent_stress_steps50`.
- Modes: all 3 mode A.
- Disconnected: all 3, graph `random_dag_50`.

## 1. classical_mds root cause

### Verdict

Verified: the residual is a mix of two causes.

1. Connected degenerate graphs: genuine degenerate-eigenspace basis ambiguity. Best-achievable tier is not bit-exact without vendoring/porting the exact igraph LAPACK path; likely rung 3/3Q only, with a PROVEN FLOOR experiment still needed for final acceptance.
2. Disconnected graphs: fixable semantic mismatch. igraph decomposes components, runs per-component MDS, then packs with stochastic DLA. Dagua computes one global filled distance matrix, producing one coupled embedding. Best-achievable tier should be rung 1 if igraph's per-component + DLA path is ported closely, otherwise rung 2/3 with deterministic approximation.

### Dagua evidence

- Dagua igraph-fidelity path computes one global shortest-path matrix for all nodes at `dagua/layout/ops/pipelines/classical_mds.py:241-245`.
- The helper fills all unreachable pairs with one global `max_distance + 1` at `dagua/layout/ops/graph_utils.py:347-350` and returns a single dense matrix at line 352. The comment explicitly says this is a global, not per-row, fill at `graph_utils.py:311-316`.
- Dagua then double-centers the single filled matrix at `classical_mds.py:251-257`, solves one global eigensystem with SciPy `eigh(... subset_by_index=(N-2,N-1), driver="evr")` at `classical_mds.py:259-265`, and returns a single layout at `classical_mds.py:267-278`.
- Dagua already documents the eigenbasis floor: repeated top eigenvalues larger than the requested two dimensions are implementation-dependent and SciPy `evr/evx/dsyevr` did not match igraph on Petersen/complete fixtures (`classical_mds.py:50-66`).

### igraph evidence

- `igraph_i_layout_mds_single` handles connected MDS by squaring distances at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:91-96`, double-centering at `mds.c:98-108`, requesting top eigenvectors at `mds.c:113-121`, and writing dimensions in reverse order at `mds.c:123-131`.
- `igraph_layout_mds` documentation states disconnected graphs are decomposed, subgraphs laid out, then merged with `igraph_layout_merge_dla()` at `mds.c:154-160`.
- The implementation checks connectivity at `mds.c:223-228`; for disconnected graphs it walks components at `mds.c:250-256`, creates each induced subgraph at `mds.c:257-260`, selects the component distance submatrix at `mds.c:261-262`, runs per-component MDS at `mds.c:263-264`, stores layouts at `mds.c:265-266`, merges with DLA at `mds.c:277-278`, then reorders rows at `mds.c:279-280`.
- `igraph_layout_merge_dla` sorts by component size at `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:123`, places the largest at origin at `merge_dla.c:134-137`, performs random-walk placement for the rest at `merge_dla.c:139-150`, then rescales/translates each component at `merge_dla.c:158-177`.
- igraph's symmetric eigensolver path calls `igraph_lapack_dsyevr` for selected eigenpairs with `abstol=1e-14` at `/home/jtaylor/projects/_references/igraph/src/linalg/eigen.c:85-96`; LAPACK wrapper uses `range='I'`, `uplo='U'` at `/home/jtaylor/projects/_references/igraph/src/linalg/lapack.c:440-448` and invokes `dsyevr` at `lapack.c:513-530`.

### Combo split

- Disconnected fixable group, 20 combos: all disconnected classical MDS rung-4 pairs listed above. This includes all 4 mode-B combos (`er_500` and `random_dag_200`, both variants) and 16 mode-A disconnected combos.
- Connected degenerate floor group, 14 combos: connected graph pairs listed above. `wide_3_50_3` is the heaviest case and likely the clearest eigenbasis floor fixture.

### Fix sketch

- Add an igraph-fidelity disconnected branch before global `_layout_igraph_classical_mds` embedding.
- Compute weak components in original vertex order.
- For each component: extract/relabel induced edges, compute component-only APSP, run the same `_layout_igraph_classical_mds` single-component kernel without global unreachable fill.
- Port `igraph_layout_merge_dla` enough for benchmark parity: component circle radius `size^.75`, size-descending order, grid bounds `sqrt(5 * area)`, same random stream as igraph, placement and final rescale/translate.
- Preserve current connected path; do not try to stabilize degenerate eigenspaces by arbitrary canonicalization because that would move away from igraph's chosen LAPACK basis.

### Floor experiment for degenerate group

- Build Gram matrices for connected degenerate fixtures and compute top eigenvalue multiplicity.
- Compare projectors `Q Q^T` for Dagua/SciPy vs igraph: projectors should match while 2D bases differ.
- Apply all 2D Procrustes transforms; if residual remains but subspace projector error is near machine epsilon, this proves the 2D selected-basis floor.
- Run 1-ULP perturbation or tiny diagonal jitter on the Gram matrix; if selected 2D basis rotates within the repeated eigenspace while stress/projector remain stable, classify as proven floor.

Confidence: high for disconnected semantic mismatch, high for degenerate basis ambiguity, medium until floor experiment is run on all 14 connected-degenerate combos.

Effort: disconnected port medium (1-2 days if DLA is ported carefully); floor proof low-medium (half day).

## 2. GEM root cause

### Verdict

Refute a blanket "FP summation-order chaos" diagnosis. The source strongly suggests at least one fixable semantic mismatch: Dagua's OGDF-fidelity update budget multiplies requested rounds by `num_nodes`, while the reference runner passes the variant value straight to `GEMLayout::numberOfRounds`, and OGDF decrements that counter once per node update. The divergent connected subgroup is almost entirely `iters100`, and Dagua's stress is usually lower than reference, consistent with over-running the short budget.

I do not claim all 22 are fixed by this alone. The remaining residual after budget correction could be genuine chaotic amplification, but that floor is not proven yet.

### Dagua evidence

- Dagua's fidelity seed bridge is present: `_mt19937_first_uint32` at `dagua/layout/ops/gem.py:239-269`, `_ogdf_gem_rng_seed(seed) = 7 * first_mt19937 + 3` at `gem.py:272-286`, and `std::minstd_rand` reproduction at `gem.py:145-183`.
- Dagua's OGDF permutation uses a Fisher-style swap loop with `uniform_int_distribution` reproduction at `gem.py:335-363`.
- Dagua consumes disturbance draws even when disturbance is zero at `gem.py:806-827`, matching OGDF's two distribution draws per impulse.
- Dagua component splitting and TileToRows packing exist in the OGDF path: components at `gem.py:1049`, shared RNG at `gem.py:1050`, per-component solve at `gem.py:1056-1064`, lower-left shift and box at `gem.py:1065-1067`, tile offsets at `gem.py:1069-1075`.
- The per-node update loop mirrors OGDF: global temperature/termination at `gem.py:896-900`, permutation/refill at `gem.py:898-903`, gravity at `gem.py:909-910`, repulsion in node order at `gem.py:915-925`, attraction over adjacency at `gem.py:926-943`, movement and barycenter update at `gem.py:944-953`, rotation/oscillation/skew/temperature at `gem.py:955-972`, previous impulse save at `gem.py:974-975`.
- But Dagua translates requested rounds into `requested_rounds * num_nodes` at `gem.py:213-236`, used in fidelity preparation at `gem.py:1288-1297`, then passed as scalar update count to `_layout_ogdf_fidelity` at `gem.py:1367-1373` and `_run_ogdf_component_gem` at `gem.py:899-900`.

### OGDF/reference evidence

- The OGDF runner passes benchmark `max_iters`/`rounds` directly as `gemRounds` at `dagua/eval/competitors/ogdf_competitor.py:288-292`, and the C++ runner calls `layout.numberOfRounds(gemRounds)` directly at `scripts/ogdf_runner.cpp:311-316`.
- OGDF `GEMLayout::call` sets `int counter = m_numberOfRounds` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:169-171`; the loop updates exactly one popped node per iteration at `GEMLayout.cpp:172-185`. There is no multiplication by node count in this source.
- OGDF defaults and RNG seed are in the constructor at `GEMLayout.cpp:56-71`.
- OGDF splits components at `GEMLayout.cpp:121-141`, copies initial positions at `GEMLayout.cpp:143-148`, initializes local state at `GEMLayout.cpp:150-167`, solves per component at `GEMLayout.cpp:169-186`, shifts each component by min bounds and `m_minDistCC` at `GEMLayout.cpp:188-211`, packs with `TileToRowsCCPacker` at `GEMLayout.cpp:214-229`.
- OGDF impulse/update formulas are at `GEMLayout.cpp:240-289` and `GEMLayout.cpp:291-341`.
- OGDF TileToRows sorts by decreasing height at `/home/jtaylor/projects/_references/ogdf/src/ogdf/packing/TileToRowsCCPacker.cpp:132-146`, chooses best row at `TileToRowsCCPacker.cpp:74-118`, and assigns row offsets at `TileToRowsCCPacker.cpp:168-182`.

### Combo split and best tiers

- 19 connected `iters100` rung-4 combos: likely rung-1 candidate after round-budget correction, because seed/order/update code already has source-faithful ports and 269/309 GEM combos are already rung 1.
- 3 disconnected combos (`random_dag_50` iters100/500, `random_dag_200` iters500): also candidate rung 1 if budget correction preserves current component packing; if not, next suspect is component order/packing tie behavior.
- Remaining possible floor: after budget correction, compare matched-seed impulse traces. If first delta appears only after many updates from <1 ULP perturbations and grows Lyapunov-style, then classify residual as FP chaos. Current evidence is insufficient for a floor claim.

### Fix sketch

- Change `_resolve_ogdf_gem_update_budget` in fidelity mode to `min(requested_rounds, max_rounds)` rather than `requested_rounds * num_nodes` for exact OGDF runner parity.
- Verify on benchmark path, not direct pipeline calls: run the definitive fidelity producer for `classic_gem_iters100`, `classic_gem_iters500`, `classic_gem_iters2000` against `ogdf_gem` at matched seeds.
- If many currently rung-1 combos regress, inspect whether the benchmark wrapper is bypassing variant parameters on the Dagua side. `ClassicGEM.layout` currently hardcodes `max_iters=30_000` at `dagua/eval/competitors/classic_competitor.py:1425-1432`, while the generic spec default is also `30_000` at `classic_competitor.py:238-242`; this needs benchmark-path verification because variant handling may be external.
- Add a trace harness that logs first 100 node IDs, RNG draws, impulse, movement, local temperature, skew, global temperature for Dagua and OGDF. This will prove or falsify residual FP chaos after budget parity.

Confidence: high that the budget mismatch is real in source; medium that it explains most/all 22 rung-4 GEM combos until the benchmark-path variant plumbing is verified.

Effort: low for budget experiment and benchmark rerun; medium for trace harness if needed.

## 3. maxent_stress root cause

### Verdict

Verified: the 3 maxent_stress rung-4 combos are disconnected `random_dag_50` cases, and Dagua's OGDF-fidelity maxent/stress path uses one global stress majorization with cross-component finite fill, while OGDF has explicit support for component-separate layout and uses ComponentSplitter/PivotMDS for disconnected initial layout when no initial layout is supplied. This is likely rung-1 fixable for these three if the reference runner does not call `hasInitialLayout(true)`; if the runner supplies an initial layout, then the specific mismatch is not ComponentSplitter initial layout but the global disconnected distance fill and serial stress across components.

### Dagua evidence

- `layout_maxent_stress_pipeline` dispatches all small `use_majorization=True` cases to `_layout_ogdf_stress_majorization` at `dagua/layout/ops/pipelines/maxent_stress.py:291-303`.
- `_layout_ogdf_stress_majorization` calls `layout_stress_majorization_pipeline(... fidelity_mode="ogdf")` at `maxent_stress.py:93-101`.
- OGDF stress pipeline in Dagua prepares one global distance matrix: `OgdfPrepareStressMajorizationState` builds all-pairs distances for the full graph at `dagua/layout/ops/pipelines/stress_majorization.py:589-595`, fills all unreachable pairs with `100 * sqrt(num_nodes)` at `stress_majorization.py:596-603`, and stores global weights at `stress_majorization.py:606-612`.
- Dagua initializes one global random layout from the runner seed at `stress_majorization.py:656-667`.
- Dagua repeats one global OGDF serial sweep over every node at `stress_majorization.py:1003-1030`; the serial sweep loops all nodes and all other nodes in one global coordinate set at `dagua/layout/ops/stress.py:642-664`.

### OGDF evidence

- OGDF `StressMinimization` header says disconnected graphs either replace infinite distances or process components separately at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:1-6`; the controlling flag is `layoutComponentsSeparately` at `StressMinimization.h:87-90` and implemented at `StressMinimization.h:216`.
- OGDF asserts non-connected graphs cannot enter with `m_componentLayout` true at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:69-70`, initializes matrices at `StressMinimization.cpp:71-83`, then calls the main minimizer at `StressMinimization.cpp:84`.
- If no initial layout exists, OGDF computes PivotMDS initial layout; for non-componentLayout it wraps PivotMDS in `ComponentSplitterLayout` at `StressMinimization.cpp:107-120`.
- If `m_componentLayout` is false and graph is disconnected, OGDF replaces infinities with `m_avgEdgeCosts * sqrt(n)` at `StressMinimization.cpp:93-100`, matching Dagua's global fill in scale.
- OGDF serial update is in-place over nodes at `StressMinimization.cpp:233-303`.
- `ComponentSplitterLayout` splits connected components and calls the secondary layout on each component at `/home/jtaylor/projects/_references/ogdf/src/ogdf/packing/ComponentSplitterLayout.cpp:63-134`, then rotates/reassembles drawings at `ComponentSplitterLayout.cpp:184-260+` and uses TileToRows packing.

### Combo split and best tiers

- All three rung-4 maxent_stress combos are disconnected `random_dag_50` mode A. Best-achievable tier: rung 1 if the runner's exact initialization/component route is matched; otherwise rung 2/3 if only global-fill semantics remain.
- The two combos annotated `TRACKING_BUT_SHIFTED` (`default`, `steps50`) strongly suggest component placement/translation rather than force-law mismatch.

### Fix sketch

- First verify `scripts/ogdf_runner.cpp` stress branch settings. If it does not call `hasInitialLayout(true)`, port the `computeInitialLayout` disconnected route: per-component PivotMDS through ComponentSplitter/TileToRows, then run stress as OGDF does.
- If the runner does call `hasInitialLayout(true)`, focus on component packing/translation and whether final comparison should account for global vs per-component origin; Dagua currently performs no component reassembly in `maxent_stress`.
- Add a disconnected stress fidelity branch for small graphs: split components, run OGDF serial stress per component or reproduce ComponentSplitter/PivotMDS initialization, then pack with TileToRows. Reuse the GEM TileToRows helper only if its tie ordering is proven against OGDF.

Confidence: high that the remaining three are disconnected-component semantics; medium on exact OGDF route until runner stress initialization flags are confirmed.

Effort: medium. If only final packing/translation is missing, low-medium; if full ComponentSplitter + PivotMDS init must be ported, medium-high.

## ROI order

1. GEM round-budget parity experiment/fix: low effort, up to 22 combos, likely largest impact per effort. Not a floor yet.
2. classical_mds disconnected igraph component+DLA branch: medium effort, 20 combos, source-confirmed fixable.
3. maxent_stress disconnected component route: medium effort, 3 combos, likely deterministic fixable.
4. classical_mds connected degenerate floor proof: low-medium effort, 14 combos, likely not fixable without exact LAPACK vendoring; needed to stop chasing false parity.

## Concerns

- All fixes must be verified through `scripts/definitive_fidelity_analysis.py`/benchmark path, because direct pipeline calls can bypass variant plumbing.
- GEM has a suspicious benchmark-path issue: `ClassicGEM.layout` hardcodes `max_iters=30_000`, while variants advertise `max_iters`. Confirm variant application before changing code.
- Do not classify GEM residual as FP floor until a trace or ULP perturbation experiment proves first divergence arises only from last-bit differences after all semantic counters match.
