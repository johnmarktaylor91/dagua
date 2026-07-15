# r75 FMMM Fidelity Sweep - Codex

## 1. Executive summary

- The 32-row target JSON uses `n=100` as sample count, not graph node count.
- Actual graph/component sizes show only one OGDF-FMMM target has a component above OGDF's 50-node multilevel threshold: `random_dag_200::classic_fmmm_steps10`.
- The highest-impact confirmed divergence is therefore not galaxy coarsening. It is dagua's OGDF-FMMM fidelity force loop using exact all-pairs repulsion, while OGDF defaults to NMM.
- This affects all 29 OGDF-FMMM targets, including the 26 small-component targets that never coarsen.
- A second confirmed divergence in multilevel prolongation affects the one true multilevel target: dagua omits OGDF Advanced same-solar-system placement and placement-sector logic.
- RNG matching is incomplete for multilevel coarsening/prolongation: dagua uses Python `random.Random`, while OGDF uses `randomNumber` seeded through OGDF's global RNG.
- The three `classic_fmmm_graphviz_fdp_fidelity` rows are out-of-family for this bucket; they dispatch to Graphviz FDP compatibility, not OGDF FMMM.
- A competitor-path spot check on `deep_chain_20`, seed 0, steps 10 produced normalized RMSD 0.2088, rejecting an FP-rounding-floor explanation.

## 2. Findings ranked by expected combo-count impact

### 1. CONFIRMED: dagua FMMM fidelity uses exact repulsion; OGDF default is NMM

Expected impact: all 29 non-Graphviz target combos.

Dagua evidence:
- `dagua/layout/ops/pipelines/fmmm.py:784-851` defines `_ogdf_fmmm_force_iteration`.
- The force iteration always calls `_ogdf_fmmm_tensor_repulsive_forces(position_tensor)` at `dagua/layout/ops/pipelines/fmmm.py:833-835`.
- `_layout_ogdf_fmmm_small_fidelity` runs this loop for small graphs at `dagua/layout/ops/pipelines/fmmm.py:1899-1918`.
- `_layout_ogdf_fmmm_multilevel_fidelity` also runs the same force iteration at every level at `dagua/layout/ops/pipelines/fmmm.py:1652-1663`.

OGDF evidence:
- OGDF defaults to `repulsiveForcesCalculation(FMMMOptions::RepulsiveForcesMethod::NMM)` in `FMMMLayout.cpp:283-288`.
- OGDF initializes the NMM solver in `FMMMLayout.cpp:1069-1080`.
- The OGDF force loop calls `calculate_repulsive_forces` inside `calculate_forces` at `FMMMLayout.cpp:972-983`.

Command evidence:

```text
$ MPLCONFIGDIR=/tmp/mplconfig python3 - <<'PY'
> import torch
> from dagua.eval.competitors import get_competitor
> from scripts.regen_ogdf_multiseed_cache import graph_registry
> graph=graph_registry()['deep_chain_20'].graph
> c=get_competitor('classic_fmmm').layout_with_variant(
>     graph, timeout=120, seed=0, variant_params={'steps':10,'fidelity_mode':True})
> o=get_competitor('ogdf_fmmm').layout_with_variant(
>     graph, timeout=120, seed=0, variant_params={'fixed_iterations':10})
> X=c.pos.double(); Y=o.pos.double()
> X=(X-X.mean(0))/(torch.linalg.norm(X-X.mean(0))+1e-12)
> Y=(Y-Y.mean(0))/(torch.linalg.norm(Y-Y.mean(0))+1e-12)
> print('classic_error', c.error, 'ogdf_error', o.error)
> print('normalized_rmsd', float(torch.sqrt(((X-Y)**2).sum(1).mean()).item()))
> print('classic_bbox', [float(v) for v in [c.pos[:,0].min(), c.pos[:,0].max(), c.pos[:,1].min(), c.pos[:,1].max()]])
> print('ogdf_bbox', [float(v) for v in [o.pos[:,0].min(), o.pos[:,0].max(), o.pos[:,1].min(), o.pos[:,1].max()]])
> PY
classic_error None ogdf_error None
normalized_rmsd 0.2088388139291783
classic_bbox [25.0, 408.0, 25.0, 390.0]
ogdf_bbox [25.0, 541.0, 25.0, 540.0]
```

Fix sketch:
- Port OGDF NMM path for the fidelity loop, or add a fidelity-only subprocess-instrumented parity harness for NMM forces and then port `NewMultipoleMethod.cpp`.
- Keep the existing exact force path behind a non-reference option; do not replace force behavior for other algorithms.

Risk to existing bit-exact/3Q combos:
- High. NMM is the central force path and can shift every OGDF-FMMM fidelity layout. Gate with target-only reruns plus the 33 r74 FMMM quality-identical combos before widening.

### 2. CONFIRMED: most target combos do not enter multilevel coarsening

Expected impact: avoids chasing coarsening for 31/32 rows; only `random_dag_200::classic_fmmm_steps10` has a component above 50.

Dagua evidence:
- The fidelity dispatch splits connected components at `dagua/layout/ops/pipelines/fmmm.py:1787-1803`.
- It only calls `_layout_ogdf_fmmm_multilevel_fidelity` when a component has `local_nodes > 50` at `dagua/layout/ops/pipelines/fmmm.py:1810-1821`.
- OGDF uses `minGraphSize(50)` by default in `FMMMLayout.cpp:274-277`, and the multilevel builder continues while node count is greater than that value in `Multilevel.cpp:69-70`.

Command evidence:

```text
actual_node_count_summary {14: 1, 10: 4, 22: 2, 25: 1, 50: 2, 48: 1, 42: 2, 18: 1, 19: 1, 38: 2, 80: 1, 15: 1, 8: 1, 20: 3, 97: 4, 383: 1, 30: 1, 7: 1, 26: 1, 36: 1}
multi_component_80 nodes 80 components [40, 20, 10, 5, 3, 1, 1] any_comp_gt50 False
random_dag_50 nodes 97 components [45, 2, 1, ...] any_comp_gt50 False
random_dag_200 nodes 383 components [181, 2, 1, ...] any_comp_gt50 True
```

Fix sketch:
- Prioritize force-model parity first. Treat coarsening/prolongation as a targeted fix for `random_dag_200` and future large FMMM rows.

Risk to existing bit-exact/3Q combos:
- Low for analysis; high only if a blanket multilevel fix is applied to small components.

### 3. CONFIRMED: dagua multilevel prolongation omits OGDF Advanced placement terms

Expected impact: `random_dag_200::classic_fmmm_steps10`; future connected or component-local graphs above 50 nodes.

Dagua evidence:
- `_prolong_positions` places suns, then planets/moons from lambda-neighbor-sun interpolation or random fallback at `dagua/layout/ops/fmmm.py:1516-1612`.
- It has no equivalent of OGDF's same-solar-system adjacent placed-node candidate generation or placement-sector calculation.

OGDF evidence:
- OGDF calls `set_initial_positions_of_sun_nodes`, `set_initial_positions_of_planet_and_moon_nodes`, and `set_initial_positions_of_pm_nodes` in `Multilevel.cpp:405-413`.
- The Advanced path adds same-solar-system adjacent placed-node candidates via `calculate_position` at `Multilevel.cpp:444-460` and `Multilevel.cpp:583-604`.
- OGDF creates placement sectors before random placement at `Multilevel.cpp:492-565`.
- Random sector placement and waggle use `randomNumber(1, BILLION)` semantics at `Multilevel.cpp:634-656`.

Fix sketch:
- Extend `_HierarchyStep` to retain moon-edge markers and same-solar-system adjacency needed by OGDF's Advanced placement.
- Port `create_all_placement_sectors`, `calculate_position`, and the PM-node second pass directly, then add golden tests on a graph whose first coarsening creates PM nodes.

Risk to existing bit-exact/3Q combos:
- Medium if guarded by `local_nodes > 50`; high if shared with the single-level path.

### 4. CONFIRMED: multilevel RNG stream is not OGDF-matched

Expected impact: only true multilevel OGDF-FMMM targets today; necessary for future bit-exactness.

Dagua evidence:
- `_build_hierarchy` seeds `rng = random.Random(seed)` at `dagua/layout/ops/fmmm.py:704-709`.
- Sun selection consumes that Python RNG at `dagua/layout/ops/fmmm.py:510-520`.
- Prolongation uses `random.Random(problem.seed)` at `dagua/layout/ops/fmmm.py:1921-1926`.
- Dagua has an `_OgdfMt19937` helper for random placement at `dagua/layout/ops/pipelines/fmmm.py:66-149`, but it is not used by coarsening or prolongation.

OGDF evidence:
- OGDF seeds the multilevel builder with `setSeed(rand_seed)` at `Multilevel.cpp:55-60`.
- `Set` reseeds with `setSeed` and selects candidates through `randomNumber` at `Set.cpp:51-79` and `Set.cpp:116-139`.
- Prolongation random placement uses `randomNumber(1, BILLION)` at `Multilevel.cpp:634-656`.

Cheapest decisive experiment:
- Add a temporary C++ trace in `/tmp` or a local scratch runner that prints first-level selected sun node indices and first five prolongation random doubles for a 60-node path, then compare to dagua `_build_hierarchy(... galaxy_choice=lower)` and `_prolong_positions`. Estimated runtime: under 10 minutes after compile.

Fix sketch:
- Reuse or generalize `_OgdfMt19937` for `randomNumber` calls in coarsening/prolongation, including the `(randomNumber(1, BILLION)+1)/(BILLION+2)` normalization.
- Preserve Python RNG for non-reference mode.

Risk to existing bit-exact/3Q combos:
- Medium. This should improve multilevel parity but will move any large-layout distributions; gate by engine variant and component size.

### 5. CONFIRMED: three rows are Graphviz FDP fidelity, not OGDF FMMM

Expected impact: 3 target rows should not be fixed in OGDF FMMM code.

Dagua evidence:
- Variants pair `classic_fmmm_graphviz_fdp_fidelity` with `graphviz_fdp`, not `ogdf_fmmm`, at `dagua/eval/variants.py:1113-1118`.
- `layout_fmmm_pipeline` dispatches `fidelity_mode == "graphviz_fdp"` to `graphviz_fdp_fidelity` or `_layout_fmmm_fidelity_components` at `dagua/layout/ops/pipelines/fmmm.py:6914-6938`.

Fix sketch:
- Move these rows to the FDP/Graphviz bucket. Do not use them to validate OGDF FMMM fixes.

Risk to existing bit-exact/3Q combos:
- None for OGDF FMMM if left untouched.

## 3. Root-cause fix sketches and risks

1. Port OGDF NMM repulsion in the fidelity force loop.
   Expected impact: all 29 OGDF-FMMM rows, especially the 26 small-component rows. Risk: high; NMM changes every iteration and postprocessing force vector.

2. Complete OGDF Advanced multilevel prolongation.
   Expected impact: `random_dag_200::classic_fmmm_steps10` in this bucket. Risk: medium if component-size-gated; high if blended into non-reference FMMM.

3. Replace Python RNG with OGDF-compatible `randomNumber` stream for multilevel coarsening/prolongation.
   Expected impact: only true multilevel rows today; prerequisite for bit-exact multilevel parity. Risk: medium.

4. Rebucket Graphviz FDP fidelity rows.
   Expected impact: removes 3 rows from OGDF FMMM analysis. Risk: none to FMMM.

## 4. Target combos not fully explained

- `random_dag_50::{classic_fmmm_steps10,classic_fmmm_steps100,classic_fmmm_steps200}`: no component exceeds 50 nodes, so multilevel is not involved. Exact-vs-NMM force parity is the leading explanation, but disconnected-component packing and singleton ordering should be rechecked after NMM parity.
- `multi_component_80::classic_fmmm_steps10`: graph has 80 nodes but max component is 40, so it is single-level per component. Same note as `random_dag_50`.
- All connected small targets with severe crossing gaps, such as `grid_5x5`, `grid_rect_6x8`, `weighted_chain_20`, and `sparse_pair_50`: explained by force-model mismatch as a confirmed first divergence, but not yet proven as the sole cause.
- The 3 `classic_fmmm_graphviz_fdp_fidelity` rows are intentionally not explained here because they do not use OGDF FMMM.
