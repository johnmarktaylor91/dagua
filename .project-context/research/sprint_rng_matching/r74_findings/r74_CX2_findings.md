# R74 CX2 SFDP findings

Scope: read-only investigation of 184 divergent `classic_sfdp_*` combos in `eval_output/fidelity_definitive_r73/per_combo.json`. Sources read: dagua `dagua/layout/ops/pipelines/sfdp.py`, `dagua/layout/ops/sfdp.py`; Graphviz `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/{sfdpinit.c,spring_electrical.c,Multilevel.c}`, `/home/jtaylor/projects/_references/graphviz/lib/common/utils.c`, `/home/jtaylor/projects/_references/graphviz/lib/util/random.c`, `/home/jtaylor/projects/_references/graphviz/lib/sparse/general.c`.

## Executive split

184 divergent SFDP combos split best as:

- **73 p_neg2 semantic bug**: dagua honors `repulsive_exponent=-2`, Graphviz clamps `repulsiveforce=-2` to internal `p=-1`. High confidence, one-line/variant-mapping fix.
- **48 non-p_neg2 disconnected component bug**: Graphviz lays out each connected component independently then packs; dagua SFDP has no component decomposition/packing path. High confidence, medium implementation.
- **63 non-p_neg2 connected floor-or-ordering combos**: no single proven behavioral bug from C. Most likely accumulation/order chaos after small remaining differences: Graphviz quadtree/supernode order, coarsening permutation RNG, and output normalization/orientation. Needs the ULP experiment below before declaring floor.

Overlap note: 9 of the 73 p_neg2 divergences are disconnected. Counting by primary fix priority gives 73 + 48 + 63 = 184. Counting raw disconnected rows gives 57; raw connected rows gives 127.

## 1. p_neg2 root cause: VERIFIED

Graphviz parsing:

- `lib/sfdpgen/sfdpinit.c:211-212`: `ctrl->K = late_double(..., 0.0)` and `ctrl->p = -1.0*late_double(g, agfindgraphattr(g, "repulsiveforce"), -AUTOP, 0.0);`
- `lib/common/utils.c:55-68`: `late_double` returns `minimum` if parsed value is below it (`if (rv < minimum) return minimum;`). Therefore `repulsiveforce=-2.0` and `repulsiveforce=-1.0` both become `0.0`; after negation, `ctrl->p` is `-0.0`.
- `lib/sfdpgen/spring_electrical.c:574-576` and fast/slow equivalents at `287-289`, `437-439`, `742-744`: `if (p >= 0) ctrl->p = p = -1;`, then `KP = pow(K, 1 - p)`, `CRK = pow(C, (2.-p)/3.)/K`. So both Graphviz default and p_neg2 run `p=-1`, denominator exponent `1-p = 2`.

Dagua behavior:

- `dagua/eval/variants.py:1616-1621` maps p_neg2 to dagua `repulsive_exponent=-2.0` while Graphviz receives `repulsiveforce=-2.0`.
- `dagua/layout/ops/pipelines/sfdp.py:406-426` computes scales directly from `repulsive_exponent` with no Graphviz clamp.
- `dagua/layout/ops/pipelines/sfdp.py:529-541` uses `denominator = pow(distance, 1.0 - self.repulsive_exponent)`. With `-2`, dagua uses `distance^3`; Graphviz p_neg2 uses `distance^2`.
- Same direct pass-through into fidelity ops at `dagua/layout/ops/pipelines/sfdp.py:887-896` and `1020-1025`.

Reference equality check:

- Direct sampled raw reference cache check in `eval_output/benchmark_100seed_seeded_refs/positions`: for seeds `{42,43,44,100,141}` across 101 graph pairs, 495/505 `graphviz_sfdp__for__classic_sfdp_default` vs `graphviz_sfdp__for__classic_sfdp_p_neg2` `.pt` files were bit-identical.
- The only differing checked graphs were `random_dag_50` and `random_dag_200` for all sampled seeds; their `results.json` entries came from different git SHAs (`default` `a0f9399...`, `p_neg2` `43f23d...`), so treat those as cache-generation contamination/anomaly, not source-level semantics. Re-run the benchmark path after fixing before relying on old metric deltas.
- Per-combo sampled R metrics are not a clean identity test because some fields use sampled/evaluation randomness. They showed 23/101 identical selected metric rows, but raw coordinates are the correct evidence.

Impact estimate:

- Current p_neg2 divergences: 73. If dagua clamps Graphviz-fidelity `repulsive_exponent >= 0` equivalent after translating `repulsiveforce`, p_neg2 should collapse to default behavior. The default rung distribution for p_neg2 divergent graphs is: 24 rung 1, 23 rung 2, 6 rung 3, 20 still rung 4. So expect about 53/73 p_neg2 divergent rows to improve immediately; the remaining 20 inherit default SFDP residuals.

Concrete fix sketch:

- In Graphviz fidelity mode only, normalize any requested `repulsive_exponent >= 0` or any benchmark value that corresponds to Graphviz-clamped `repulsiveforce < 0` to `-1.0`. Since dagua receives already-negated `repulsive_exponent`, the practical benchmark fix is: for `fidelity_mode="graphviz"`, if `repulsive_exponent < -1.0` and it came from a negative Graphviz `repulsiveforce`, use `-1.0`. The cleaner design is to map variant params so `classic_sfdp_p_neg2` passes `repulsive_exponent=-1.0` under Graphviz fidelity.

Confidence: very high.

## 2. Disconnected graphs: VERIFIED missing component model

Graphviz behavior:

- `lib/sfdpgen/sfdpinit.c:268-287`: `ccomps(g, &ncc, 0)` splits connected components. If `ncc == 1`, Graphviz calls `sfdpLayout(g, &ctrl, pad)` once at `270-273`. Else it loops `for (size_t i = 0; i < ncc; i++)`, induces each subgraph, calls `sfdpLayout(sg, &ctrl, pad)`, splines each component, then `packSubgraphs(ncc, ccs, g, &pinfo)` at `287`.
- Each component call resets random start internally: `lib/sfdpgen/spring_electrical.c:567-570` calls `srand(ctrl->random_seed)` and fills `x` using `drand()`. The same reset exists in fast/slow functions at `280-283`, `430-433`, `735-738`. `drand()` is `rand()/(double)RAND_MAX` in `lib/sparse/general.c:24-26`.

Dagua behavior:

- `dagua/layout/ops/pipelines/sfdp.py:923-1026` validates inputs, builds one `LayoutProblem`, and applies one SFDP pipeline over the full graph. There is no connected-component split.
- `dagua/layout/ops/pipelines/sfdp.py:912-917` builds/refines/finalizes a single hierarchy.
- Graph construction at `dagua/layout/ops/sfdp.py:492-500` is global, and the refinement loops in `dagua/layout/ops/pipelines/sfdp.py:511-592` include all nodes in one force field. For disconnected components, this means inter-component repulsion affects layout; Graphviz never lets components repel during component-local layout.

Data:

- Total disconnected divergent rows: 57/184.
- Non-p_neg2 disconnected rows: 48, across 12 unique graphs. Most are 5/5 variants: `disconnected_encoder_residual`, `disconnected_label_cycle_collage`, `er_100`, `kitchen_sink_platform_graph`, `multi_component_80`, `parallel_cycles_4x5`, `random_bipartite_60`, `random_dag_200`, `random_dag_50`; plus `dependency_500`, `dependency_graph_100`, `er_500` for `theta08` only.
- Median `e_rel` for disconnected divergences is much larger than connected within the same variants (examples: default disconnected median 1.905 vs connected 0.531; theta08 disconnected 3.185 vs connected 0.531), consistent with a component-layout/packing mismatch.

Concrete fix sketch:

- Add a Graphviz-fidelity connected-component wrapper to `layout_sfdp_pipeline`, analogous to existing neato/fdp/fmmm component code. For each weak component: slice local edges/weights/sizes, run the exact same SFDP fidelity pipeline with the **same seed** (Graphviz resets `srand(ctrl->random_seed)` per component), compute component boxes, then pack with the existing Graphviz packer helpers already ported in neato/fmmm if acceptable within scope. Preserve component order from Graphviz `ccomps` as closely as possible.
- Verify on benchmark path, not direct pipeline calls.

Confidence: high.

## 3. Adaptive cooling suspect: REFUTED

Graphviz:

- Initial control has adaptive cooling enabled: `lib/sfdpgen/spring_electrical.c:59-62` sets `step=0.1`, `adaptive_cooling=true`.
- After each prolongation to a finer level, Graphviz explicitly sets `ctrl->adaptive_cooling = false` and `ctrl->step = .1` at `lib/sfdpgen/spring_electrical.c:1176-1179`.
- `update_step` applies adaptive or fixed cooling at `lib/sfdpgen/spring_electrical.c:171-185`.

Dagua Graphviz path:

- Coarsest refinement passes `adaptive_cooling=True` at `dagua/layout/ops/pipelines/sfdp.py:717-727`.
- Finer-level refinement passes `adaptive_cooling=False` at `dagua/layout/ops/pipelines/sfdp.py:809-819`.

Conclusion: current dagua matches Graphviz on this point. This is not the connected floor cause.

## 4. Other connected systematic suspects

### Coarsening permutation RNG: likely small systematic mismatch

- Graphviz `lib/sfdpgen/Multilevel.c:102-104` calls `gv_permutation(m)`.
- Current Graphviz `lib/util/random.c:15-32` implements Fisher-Yates with `gv_random(i + 1)` at line `28`; `gv_random` uses rejection sampling through `random_small` (`36-59`, `85-95`).
- Dagua `GraphvizRandom.permutation` says `lib/sparse/general.c::random_permutation calls irand(len)` and uses raw modulo at `dagua/layout/ops/sfdp.py:247-253`.

This looks like a real source-version mismatch. It may only bite when rejection actually discards, but even a rare extra `rand()` during coarsening can change the entire hierarchy and trigger attractor-scale divergence. It should be tested before floor classification.

### Barnes-Hut / supernode accumulation order: plausible floor driver

- Graphviz sequential path builds a quadtree at `lib/sfdpgen/spring_electrical.c:587-591`, obtains supernodes per vertex at `615-616`, and accumulates them in C array order at `624-628`, then immediately updates the vertex at `646-648`.
- Dagua sequential path builds `GraphvizQuadTree` at `dagua/layout/ops/pipelines/sfdp.py:502-509`, then calls `graphviz_supernode_repulsive_force` at `569-576`, accumulating through the Python/PyTorch quadtree implementation. Even if formula-equivalent, ordering and cell-boundary choices can differ at last-bit level.

### K-decay schedule: appears matched

- Graphviz after prolongation: `ctrl->K = ctrl->K * 0.75` at `lib/sfdpgen/spring_electrical.c:1176-1178`; prolongation jitter uses old `ctrl->K * 0.001` at `1173`.
- Dagua Graphviz path preserves old K for prolongation (`prolongation_ideal_length = ideal_length`) then multiplies by `0.75` for refinement at `dagua/layout/ops/pipelines/sfdp.py:794-818`. This matches intent.

### Output normalization/orientation: residual non-reference behavior

- Graphviz does principal-component rotation before overlap removal at `lib/sfdpgen/spring_electrical.c:1192-1195` and then outputs Graphviz coordinates.
- Dagua finalizes with flow-oriented candidate selection and normalization at `dagua/layout/ops/sfdp.py:1855-1862`; `_orient_positions_to_flow` considers original, reflected, rotated, reflected-rotated candidates at `dagua/layout/ops/sfdp.py:450-489`.

This can affect bit/layout matching, especially on directed graphs. It is less likely than p_neg2/components for the 184 count, but should be included in floor experiments.

## 5. Floor-proving experiment design

Goal: distinguish chaotic FP/summation-order floor from a bounded deterministic bug for the 63 non-p_neg2 connected divergences.

Benchmark-path-only experiment:

1. Select representative connected divergent graphs: one small (`real_karate_34` or `weighted_chain_20`), one medium (`small_world_100`), one lattice/regular (`hexagonal_lattice_42` or `planar_60`), one weighted (`weighted_karate_34`). Include one p_neg2-after-clamp graph as a negative control and one disconnected graph as a bug control.
2. Add an instrumentation-only branch/flag, not a production fix, that runs the exact benchmark path and logs per-iteration aligned RMSD or raw max-norm deltas. Do not alter scoring thresholds.
3. Create three dagua variants from the same initial positions/seed:
   - baseline Graphviz-fidelity path;
   - 1-ULP perturbation to one coordinate after random initialization or after coarsest solve start;
   - deterministic summation-order perturbation only (e.g., reverse exact-repulsion loop order or force Kahan/pairwise sum) with all high-level choices unchanged.
4. Measure distance between the two dagua variants over iterations after Procrustes alignment and scale normalization. Also compare final quality metrics.
5. Classification rule:
   - **Floor**: 1-ULP perturbation grows roughly exponentially from <=1e-12 scale to attractor-scale layout difference by ~iter 100, and final quality remains in the same distribution as Graphviz/reference. Different summation orders produce similarly large coordinate divergence without stable bias.
   - **Bug**: perturbation stays bounded/constant or grows to a stable offset, while dagua-vs-reference has a coherent directional/metric bias; or changing one suspect (component handling, p clamp, permutation RNG) collapses many rows to default/reference.
6. Run the same experiment after applying p_neg2 clamp and component packing so connected-floor candidates are not polluted by known bugs.

Expected outcome: many of the 63 connected non-p_neg2 rows will prove floor/ordering-sensitive, but coarsening RNG (`gv_random` vs modulo) should be tested first because it is a concrete source mismatch.

## Suggested ROI order

1. Fix p_neg2 mapping/clamp: highest ROI, low effort, ~53 immediate row improvements expected, 73 rows explained.
2. Add SFDP Graphviz component decomposition + packing: high ROI, medium effort, 48 non-p_neg2 rows plus 9 overlapping p_neg2 rows explained.
3. Correct `GraphvizRandom.permutation` to current Graphviz `gv_random` rejection sampling and rerun connected SFDP: low-medium effort, uncertain impact, but source-backed.
4. Run the ULP/summation-order floor experiment on remaining connected rows before attempting Barnes-Hut or normalization rewrites.

## Assumptions and caveats

- `per_combo.json` r73 has 607 SFDP rows and 184 divergent SFDP rows. Counts above use `final_rung == "4"` and engines `classic_sfdp_default`, `classic_sfdp_graphviz_fidelity`, `classic_sfdp_theta04`, `classic_sfdp_theta08`, `classic_sfdp_p_neg2`, `classic_sfdp_steps200`.
- The raw seeded reference cache check used `eval_output/benchmark_100seed_seeded_refs`, not `fidelity_definitive_r73`, because r73 positions were not present under that directory. Two anomalous graphs were generated at different git SHAs; rerun benchmark path after any fix.
- No repository files were modified.
