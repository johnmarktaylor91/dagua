# r75 mds_tails codex report

## 1. Executive summary

- CONFIRMED: 16/30 classical_mds rows are disconnected-graph failures explained by missing igraph component split plus `igraph_layout_merge_dla`.
- CONFIRMED: dagua currently feeds one global finite-filled distance matrix to MDS; igraph splits weak components, runs MDS on each submatrix, then DLA-merges them.
- CONFIRMED: default and igraph_fidelity classical_mds fail for the same reasons; their target metrics are identical except tiny reference jitter on seeded DLA.
- HYPOTHESIS: most connected classical_mds rows are eigensolver basis/sign/order residuals around LAPACK `dsyevr`; wide layered graphs may also expose metric/fill or orientation differences.
- CONFIRMED: UMAP variant params are passed through, but the remaining 7 rows are not due to missing `n_neighbors`, `min_dist`, or `spread` forwarding.
- HYPOTHESIS: UMAP residue is in fuzzy graph / spectral init / negative-sampling RNG parity; `parallel_multiedge_bundle` is a comparison red flag because dagua is massively better on stress.
- HYPOTHESIS: GEM, DrL, maxent, and neato tails are small residuals in already-ported stochastic/iterative algorithms, not proven FP floors.
- No runtime code changes were made.

## 2. Findings ranked by expected combo-count impact

### 1. CONFIRMED: classical_mds disconnected graphs need igraph DLA merge (16 rows)

Evidence command:

```bash
python3 - <<'PY'
import json, collections
rows=json.load(open(".project-context/research/sprint_rng_matching/r75_findings/r75_targets_classical_mds.json"))
print("rows", len(rows))
print("by engine", dict(collections.Counter(r["engine"] for r in rows)))
print("disconnected", dict(collections.Counter(bool(r.get("disconnected")) for r in rows)))
for r in rows:
    failing=[k for k in ["battery_stress","cross","np"] if not r[k]["equiv"]]
    print(r["combo_id"], "disc="+str(r.get("disconnected")), "fail="+",".join(failing),
          "stress_delta=%+.6g"%(r["battery_stress"]["D"]-r["battery_stress"]["R"]))
PY
```

Output excerpt:

```text
rows 30
by engine {'classic_classical_mds_default': 15, 'classic_classical_mds_igraph_fidelity': 15}
disconnected {False: 14, True: 16}
multi_component_80::classic_classical_mds_default disc=True fail=battery_stress,cross,np stress_delta=+0.477318
random_bipartite_60::classic_classical_mds_default disc=True fail=battery_stress,cross,np stress_delta=+0.10247
parallel_cycles_4x5::classic_classical_mds_default disc=True fail=battery_stress stress_delta=+0.988977
```

Dagua side: `layout_classical_mds_pipeline` routes both default/no weights and igraph_fidelity through `_layout_igraph_classical_mds` at [dagua/layout/ops/pipelines/classical_mds.py:171](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:171). That function computes one all-pairs matrix and one eigensolve for the whole graph at [dagua/layout/ops/pipelines/classical_mds.py:241](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:241) and [dagua/layout/ops/pipelines/classical_mds.py:259](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:259). The shared distance helper replaces unreachable pairs with `max_distance + 1.0` globally at [dagua/layout/ops/graph_utils.py:347](/home/jtaylor/projects/dagua/dagua/layout/ops/graph_utils.py:347).

Reference side: igraph explicitly documents disconnected splitting at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:155`, checks weak connectivity at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:223`, lays out each subcomponent at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:256` and `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:264`, then calls `igraph_layout_merge_dla` at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:278`.

Impact: the 16 disconnected rows:

```text
disconnected_encoder_residual x 2
disconnected_label_cycle_collage x 2
kitchen_sink_platform_graph x 2
multi_component_80 x 2
parallel_cycles_4x5 x 2
random_bipartite_60 x 2
random_dag_50 x 2
random_dag_200 x 2
```

Fix sketch: add an igraph-fidelity disconnected branch only inside classical_mds. Compute weak components in vertex order, run the existing single-component MDS on each component subgraph/submatrix, then merge with a faithful DLA port and reorder rows by igraph's `vertex_order` semantics.

Risk: medium. Scope the branch to disconnected graphs only. Do not alter connected graphs; prior blanket component logic broke exact combos.

### 2. CONFIRMED: igraph merge_dla port spec

Reference algorithm:

- Build one layout per component before merge, preserving component discovery order from the first unseen vertex. Source: `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:250` to `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:280`.
- For each component layout, compute `size = nrow`, target radius `r = size^0.75`, and original bounding sphere center/radius `(nx, ny, nr)` from bbox diagonal / 2. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:100` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:122` and sphere helper at `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:192`.
- Sort component indices by descending size with `igraph_vector_sort_ind(..., IGRAPH_DESCENDING)`. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:32` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:47` and call at `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:123`.
- Allocate a 200 x 200 merge grid spanning `[-sqrt(5*area), +sqrt(5*area)]` in x and y, where `area=sum(r_i^2)`. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:125` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:129`.
- Place largest sphere at origin. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:134` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:136`.
- For each remaining sphere, call random walk with center `(0,0)`, `startr=maxx`, `killr=maxx+5`. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:144` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:150`.
- Particle start: repeat drawing `angle=RNG_UNIF(0,2*pi)` and `len=RNG_UNIF(0.5*startr, startr)` until the candidate sphere does not collide. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:276` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:284`.
- Walk step: while not colliding and distance from center is below kill radius, draw `angle=RNG_UNIF(0,2*pi)` and `len=RNG_UNIF(0, startr/100)`, test next candidate, accept only non-colliding moves. Stop on collision; current coordinate is the last non-colliding point adjacent to the occupied set. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:286` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:296`.
- Occupancy is approximate grid rasterization of placed circles; grid cell lookup and sphere collision are in `/home/jtaylor/projects/_references/igraph/src/layout/merge_grid.c:70` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_grid.c:198`. Preserve the quadrant loop quirks for bit parity.
- Final coordinates rescale each component from original sphere radius `nr` to target `r`, recenter at `(nx, ny)`, then translate by DLA `(x,y)`. If `nr==0`, scale is 1. Source: `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:161` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:177`.

RNG stream requirements:

- C source uses `RNG_UNIF` in the DLA walk at `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:279`, `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:280`, `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:288`, and `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:289`.
- Benchmark reference does not use igraph default PCG directly: the adapter wraps stochastic igraph layouts in `igraph.set_random_number_generator(random.Random(seed))` at [dagua/eval/competitors/igraph_competitor.py:46](/home/jtaylor/projects/dagua/dagua/eval/competitors/igraph_competitor.py:46), and `IgraphMDS` sets `uses_igraph_rng=True` at [dagua/eval/competitors/igraph_competitor.py:267](/home/jtaylor/projects/dagua/dagua/eval/competitors/igraph_competitor.py:267). Therefore benchmark-bit parity needs Python `random.Random(seed).random()` uniform draws in C-call order, not just igraph's default C RNG.

Why a naive port can hang:

- The reference walk is intentionally unbounded. There is no max step or restart count in `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:276` to `/home/jtaylor/projects/_references/igraph/src/layout/merge_dla.c:297`.
- A port that misses grid collision semantics, uses too-small steps, treats boundary contact differently, or requires landing exactly on an occupied cell can loop for a very long time because particles restart only after crossing `killr`.

Bounded implementation strategy:

- Implement exact scalar NumPy/Python first, not torch vectorization, because the branch only affects small benchmark disconnected components and DLA is control-flow heavy.
- Use `random.Random(seed)` draws through a small `rng_unif(lo, hi)` wrapper.
- Port `mergegrid_which`, `place_sphere`, and `get_sphere` literally, including boundary comparisons and quadrant loop conditions.
- Add instrumentation-only caps during development (`max_steps_per_particle`, `max_restarts`) that raise, not fallback, so hangs expose port bugs. Once validated against python-igraph on 5 seeds, either remove caps or set very high guardrails with explicit error.
- Estimated effort: 180-260 LOC production plus 120-180 LOC tests/probes. Most complexity is preserving grid quirks and row reordering.

Expected end-state: likely 3Q or distributional-equivalence for all 16 disconnected classical_mds rows; bit-exact may require exact Python RNG integration and matching python-igraph custom RNG float conversion.

### 3. HYPOTHESIS: connected classical_mds residuals are eigensolver basis / degenerate eigenspace issues (14 rows)

Connected rows:

```text
bipartite_4_3_4 x 2
center_port_backedge_hub x 2
densenet_block x 2
org_chart_1_5_4_8 x 2
petersen_10 x 2
wide_single_layer_1_50_1 x 2
wide_3_50_3 x 2
```

Dagua source already calls SciPy `eigh(..., subset_by_index=(n-2,n-1), driver="evr")` at [dagua/layout/ops/pipelines/classical_mds.py:259](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:259). The docstring records prior evidence that igraph asks LAPACK for largest algebraic eigenpairs and repeated top eigenvalues select implementation-dependent bases at [dagua/layout/ops/pipelines/classical_mds.py:50](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:50).

Reference source confirms top-`dim` LAPACK eigenvectors via `IGRAPH_EIGEN_LAPACK` at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:113` to `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:121`, then reversed column write at `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:123` to `/home/jtaylor/projects/_references/igraph/src/layout/mds.c:131`.

Cheapest decisive experiment: benchmark-path run for the 7 connected graphs using a temporary branch that exposes alternative LAPACK drivers (`evr`, `evx`, `evd`) and an eigenspace-invariant metric. Runtime estimate: <10 min for 7 graphs x deterministic row. If eigenspace RMSD passes but coordinate RMSD fails, classify as basis-only. If quality battery still differs on wide graphs, inspect graph-distance construction and crossing metric sensitivity separately.

Fix sketch: only if needed, vendor/use the same LAPACK path or add a deterministic post-basis rotation selected by reference fixtures. Do not add broad rotation heuristics for all MDS outputs.

Risk: high for connected exact combos, because changing eigenvector selection can perturb every connected MDS graph.

### 4. CONFIRMED: classical_mds default and igraph_fidelity variants fail for same reasons

Variant registry: default has empty reimpl/original params at [dagua/eval/variants.py:843](/home/jtaylor/projects/dagua/dagua/eval/variants.py:843); igraph_fidelity only adds `{"igraph_fidelity": True}` at [dagua/eval/variants.py:855](/home/jtaylor/projects/dagua/dagua/eval/variants.py:855). Both compare to `igraph_mds` with `{}` original params.

Wrapper behavior: both no-edge-weight default and igraph_fidelity enter `_layout_igraph_classical_mds`; the only visible difference is the two-node special case and output dtype path at [dagua/layout/ops/pipelines/classical_mds.py:171](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:171) to [dagua/layout/ops/pipelines/classical_mds.py:177](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/classical_mds.py:177). None of the 30 target rows are two-node graphs.

Evidence: target rows occur in identical pairs for each graph, and metrics are identical except tiny reference-side stochastic differences on disconnected DLA rows.

### 5. CONFIRMED/HYPOTHESIS: UMAP params match; residue is fuzzy/spectral/SGD parity (7 rows)

CONFIRMED parameter matching:

- Registry forwards default, nn5, nn30, mindist001, and spread2 params to both reimplementation and reference at [dagua/eval/variants.py:1637](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1637) to [dagua/eval/variants.py:1698](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1698).
- Reference adapter accepts `n_epochs`, `negative_sample_rate`, `learning_rate`, `repulsion_strength`, `min_dist`, `spread`, and `n_neighbors` at [dagua/eval/competitors/umap_competitor.py:100](/home/jtaylor/projects/dagua/dagua/eval/competitors/umap_competitor.py:100), sets `metric="precomputed"`, `random_state=seed`, default `init=random` for N<10 else `spectral`, then updates variant params at [dagua/eval/competitors/umap_competitor.py:185](/home/jtaylor/projects/dagua/dagua/eval/competitors/umap_competitor.py:185) to [dagua/eval/competitors/umap_competitor.py:196](/home/jtaylor/projects/dagua/dagua/eval/competitors/umap_competitor.py:196).
- Dagua has matching default epochs, initial RNG-state draw, positive-edge pruning, and optimizer call surfaces at [dagua/layout/ops/umap.py:1595](/home/jtaylor/projects/dagua/dagua/layout/ops/umap.py:1595), [dagua/layout/ops/umap.py:1825](/home/jtaylor/projects/dagua/dagua/layout/ops/umap.py:1825), [dagua/layout/ops/umap.py:1919](/home/jtaylor/projects/dagua/dagua/layout/ops/umap.py:1919), and [dagua/layout/ops/umap.py:1971](/home/jtaylor/projects/dagua/dagua/layout/ops/umap.py:1971).
- umap-learn reference defaults are `n_epochs=None`, `negative_sample_rate=5`, `init="spectral"`, `min_dist=0.1`, `spread=1.0`, `random_state=None` constructor defaults at `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py:1665` to `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py:1688`; default epochs are 500 for N<=10000 and 200 otherwise at `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py:1072` to `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py:1083`; optimizer RNG state is drawn at `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py:1152`.

Target evidence: 5/7 UMAP rows are `parallel_multiedge_bundle` with stress deltas around -0.08, meaning dagua is much lower-stress than reference. Two disconnected rows (`random_dag_50`, `random_dag_200`) are mixed.

Hypothesis: the remaining mismatch is not high-level params. It is likely one of:

- duplicate-edge CSR coalescing / shortest-path distance parity for `parallel_multiedge_bundle`;
- spectral initialization component handling for disconnected fuzzy graphs;
- exact `tau_rand_int` / numba float32 operation ordering in negative sampling.

Cheapest decisive experiment: benchmark-path run for `parallel_multiedge_bundle::classic_umap_nn5` with probes dumping reference `distances`, fuzzy graph COO, initial embedding before SGD, and final embedding after 1 epoch. Runtime estimate: 5-8 min after numba warmup. First divergence point determines fix.

Expected end-state: 3Q for all 7; bit-exact unlikely without numba operation-order parity.

### 6. HYPOTHESIS: GEM tail is small stochastic residual, not proven floor (5 rows)

Reference source: igraph GEM initializes random positions with `RNG_UNIF(-width_half,width_half)` at `/home/jtaylor/projects/_references/igraph/src/layout/gem.c:135` to `/home/jtaylor/projects/_references/igraph/src/layout/gem.c:143`, shuffles a permutation when exhausted at `/home/jtaylor/projects/_references/igraph/src/layout/gem.c:161` to `/home/jtaylor/projects/_references/igraph/src/layout/gem.c:166`, and adds random impulse jitter at `/home/jtaylor/projects/_references/igraph/src/layout/gem.c:171` to `/home/jtaylor/projects/_references/igraph/src/layout/gem.c:172`.

Dagua target is actually OGDF GEM, not igraph GEM: variants compare to `ogdf_gem` at [dagua/eval/variants.py:1047](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1047) to [dagua/eval/variants.py:1074](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1074), and the adapter forwards `gemRounds` at [dagua/eval/competitors/ogdf_competitor.py:288](/home/jtaylor/projects/dagua/dagua/eval/competitors/ogdf_competitor.py:288). Dagua's GEM pipeline documents remaining topology-sensitive residuals at [dagua/layout/ops/pipelines/gem.py:46](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/gem.py:46), translates rounds to node updates at [dagua/layout/ops/gem.py:213](/home/jtaylor/projects/dagua/dagua/layout/ops/gem.py:213), and runs an OGDF-fidelity per-component branch at [dagua/layout/ops/gem.py:1024](/home/jtaylor/projects/dagua/dagua/layout/ops/gem.py:1024).

Target rows are all near variance margins except `grid_5x5` stress delta -0.0436. I do not claim FP floor: no 1-ULP perturbation experiment was run.

Cheapest decisive experiment: add scratch-only perturbation harness that takes the saved benchmark-path initial state for one GEM target, applies +/-1 ULP to one coordinate or impulse, and measures divergence over updates against the OGDF runner trace. Runtime estimate: 10-15 min for one target if existing trace hooks are available; otherwise 30 min.

Expected end-state: likely aggregate-equivalence or documented FP-chaos floor if perturbation reproduces seed-level spread; otherwise fix update-order/rounding at first divergence.

### 7. HYPOTHESIS: maxent_stress tail is disconnected OGDF StressMinimization parity (3 rows)

Reference side: OGDF StressMinimization applies to connected and unconnected graphs; header says disconnected graphs either replace infinity distances or process components separately at `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:1` to `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/StressMinimization.h:6`. The implementation asserts separate component layout only on connected graphs at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:69`, computes PivotMDS initial layout via `ComponentSplitterLayout` when not component-layout mode at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:107` to `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:123`, replaces infinite distances by `avgEdgeCosts*sqrt(n)` at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:94` to `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:100`, and uses serial node update at `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:233` to `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/StressMinimization.cpp:302`.

Dagua registry maps maxent_stress variants to OGDF stress with matching iteration counts at [dagua/eval/variants.py:1124](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1124), [dagua/eval/variants.py:1157](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1157), and [dagua/eval/variants.py:1168](/home/jtaylor/projects/dagua/dagua/eval/variants.py:1168). The three remaining rows are all `random_dag_50` disconnected.

Hypothesis: residue is disconnected initialization/packing or serial update parity, not the reverted blanket component split. Do not reintroduce blanket splitting.

Cheapest decisive experiment: run only `random_dag_50` with OGDF runner and dagua maxent trace at steps 0,1,2,50,200,400 and compare initial PivotMDS/component placement first. Runtime estimate: <10 min.

Expected end-state: 3Q after matching initial layout/disconnected distance semantics, or documented near-margin residual.

### 8. HYPOTHESIS: DrL tail is density-grid/float32 boundary residue (3 rows)

Reference side: igraph DrL options presets are initialized in `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:240` onward; the documentation states igraph includes only a subset and not the full recursive/multilevel layout at `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:207` to `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp:210`.

Dagua side: DrL pipeline documents known dependence on C++ float rounding and density-grid boundary behavior at [dagua/layout/ops/pipelines/drl.py:39](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/drl.py:39). Fidelity mode initializes with the benchmark adapter seed matrix at [dagua/layout/ops/drl.py:425](/home/jtaylor/projects/dagua/dagua/layout/ops/drl.py:425) and [dagua/layout/ops/drl.py:1925](/home/jtaylor/projects/dagua/dagua/layout/ops/drl.py:1925). The igraph adapter also passes a seeded initial matrix for DrL because `accepts_seed_matrix=True` at [dagua/eval/competitors/igraph_competitor.py:290](/home/jtaylor/projects/dagua/dagua/eval/competitors/igraph_competitor.py:290) to [dagua/eval/competitors/igraph_competitor.py:299](/home/jtaylor/projects/dagua/dagua/eval/competitors/igraph_competitor.py:299).

Rows: `real_karate_34::classic_drl_refine`, `real_lesmis_77::classic_drl_refine`, `real_lesmis_77::classic_drl_coarsen`.

Cheapest decisive experiment: single-seed matched benchmark-path trace comparing phase boundaries and density-grid cell assignment for `real_lesmis_77::classic_drl_coarsen`. Runtime estimate: 10-15 min.

Expected end-state: aggregate-equivalence or floor only after perturbation evidence.

### 9. HYPOTHESIS: neato tail is disconnected packing / Graphviz solver residual (2 rows)

Reference adapter passes both seed and start to Graphviz engines at [dagua/eval/competitors/graphviz_competitor.py:413](/home/jtaylor/projects/dagua/dagua/eval/competitors/graphviz_competitor.py:413) to [dagua/eval/competitors/graphviz_competitor.py:420](/home/jtaylor/projects/dagua/dagua/eval/competitors/graphviz_competitor.py:420). Variant params match `maxiter=200`, `epsilon=0.0001`, `pack=True` at [dagua/eval/variants.py:877](/home/jtaylor/projects/dagua/dagua/eval/variants.py:877) to [dagua/eval/variants.py:883](/home/jtaylor/projects/dagua/dagua/eval/variants.py:883).

Graphviz source initializes random positions with `drand48()` when no position is supplied at `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:127` to `/home/jtaylor/projects/_references/graphviz/lib/neatogen/stress.c:166`, and neato has a global `Pack` component-layout flag at `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:54` to `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c:57`.

Dagua side documents unported exact CG, raw `drand48`, edge len semantics, and post-processing at [dagua/layout/ops/pipelines/neato.py:1272](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/neato.py:1272). Its disconnected branch slices components and seeds each as `seed + component_index` at [dagua/layout/ops/pipelines/neato.py:1404](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/neato.py:1404) to [dagua/layout/ops/pipelines/neato.py:1419](/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/neato.py:1419), which is a likely mismatch with Graphviz's single process RNG stream.

Rows: `parallel_cycles_4x5::classic_neato`, `random_dag_50::classic_neato`.

Cheapest decisive experiment: benchmark-path run of these two graphs with `pack=false` and with a scratch one-stream component seed policy. Runtime estimate: <10 min.

Expected end-state: 3Q; bit-exact unlikely until Graphviz's pack and CG details are fully ported.

## 3. Root-cause fixes, impact, and risk

1. Classical MDS DLA port.
   Fix: component-split + exact DLA merge for disconnected graphs only.
   Impact: 16 rows.
   Risk: low-to-medium if gated to disconnected classical_mds; high if applied as a general packing replacement.

2. Classical MDS connected eigensolver basis.
   Fix: no code change until eigenspace experiment proves basis-only residual. Potentially use igraph LAPACK-compatible basis selection.
   Impact: up to 14 rows.
   Risk: high to existing connected exact combos.

3. UMAP fuzzy/spectral/SGD parity.
   Fix: trace first divergence between reference adapter and dagua port; likely duplicate-edge/fuzzy graph or RNG stream adjustment.
   Impact: 7 rows.
   Risk: medium; UMAP tails include one multiedge graph and two disconnected graphs, so isolate fixes to those semantics.

4. GEM residual.
   Fix: first-divergence or perturbation test before changing scalar update math.
   Impact: 5 rows.
   Risk: high; existing GEM bit-exact rows can break from tiny update-order changes.

5. Maxent StressMinimization disconnected parity.
   Fix: inspect initial PivotMDS/component placement and infinity-distance replacement; do not blanket component-split.
   Impact: 3 rows.
   Risk: high for prior reverted failure mode.

6. DrL residue.
   Fix: trace density-grid boundary and float rounding in target phases.
   Impact: 3 rows.
   Risk: medium-high; stochastic phase changes affect all DrL variants.

7. Neato disconnected packing/RNG.
   Fix: match Graphviz single RNG stream across packed components or Graphviz pack semantics after confirming.
   Impact: 2 rows.
   Risk: medium; only disconnected `pack=True` branch should change.

## 4. Target combos not fully explained

I could not fully explain these without the decisive experiments above:

- Connected classical_mds: `bipartite_4_3_4`, `center_port_backedge_hub`, `densenet_block`, `org_chart_1_5_4_8`, `petersen_10`, `wide_single_layer_1_50_1`, `wide_3_50_3` for both variants.
- UMAP: all 7 rows need first-divergence tracing; high-level params are ruled out.
- GEM: all 5 rows need perturbation/trace evidence before calling floor.
- Maxent: all 3 `random_dag_50` rows need initial-layout trace.
- DrL: all 3 rows need phase/density-grid trace.
- Neato: both rows need Graphviz pack/RNG trace.
