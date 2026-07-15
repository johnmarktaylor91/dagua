# r76 PROBE: SFDP/FDP-Family Divergence Triage

Date: 2026-07-03
Scope: research/probe only. Repo code was read-only. Scratch artifacts are under `/tmp`. This file is the only project output.

## Inputs and Commands

Rows came from `eval_output/fidelity_definitive/r75_final.jsonl` with:

- `engine` containing `sfdp` or `fdp`
- `quality_identical_raw == false`
- `no_canonical_reference != true`

Count: 47 rows = 44 SFDP rows plus 3 `classic_fmmm_graphviz_fdp_fidelity` rows transferred from FMMM triage. No-canonical SFDP `theta04`, `theta08`, and `steps200` rows were excluded.

Position overlay used this order, later directory wins when a file exists:

`escalation_final`, `seeded_refs`, `r72_fixes`, `fdp_fix`, `r73_fixes`, `r75_fixes`, `r75_mds_topup`, `r75_topup2`, `r76_refs`, `r76_gem_fix`.

Commands run:

```bash
sed -n '1,220p' .project-context/research/sprint_rng_matching/r75_findings/r75_sfdp_codex.md
sed -n '1,220p' .project-context/research/sprint_rng_matching/r75_findings/r75_sfdp_sonnet.md
sed -n '1,260p' .project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md
sed -n '1,220p' .project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_fmmm_triage.md
sed -n '1,220p' .project-context/research/sprint_rng_matching/r75_RESULTS.md
python - <<'PY' > /tmp/r76_sfdp_triage_tables.json
# filtered rows, computed failing-leg patterns, 5-seed saved-position Procrustes RMSD,
# and per-row/cluster summaries from r75_final.jsonl plus benchmark_100seed_* tensors
PY
python - <<'PY' > /tmp/r76_sfdp_stage_probe.txt
# imported graph builders and SFDP pipeline modules; ran dot -v -Tjson -Ksfdp/-Kfdp;
# captured Dagua hierarchy levels, Graphviz verbose control output, and component-pack bboxes
PY
gcc /tmp/rand_probe.c -o /tmp/rand_probe && /tmp/rand_probe
git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:lib/sfdpgen/sfdpinit.c
git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:lib/sfdpgen/spring_electrical.c
git -C /home/jtaylor/projects/_references/graphviz show 7.0.5:lib/sfdpgen/Multilevel.c
```

Notes:

- Position files are sparse across overlay dirs. Typical SFDP Dagua tensors came from `benchmark_100seed_escalation_final`; typical Graphviz references came from `benchmark_100seed_seeded_refs`. The lookup was per side and seed, taking the freshest existing file.
- Procrustes RMSD was computed after centering and optimal rotation on unit-norm coordinates. Threshold for near-match: median RMSD `< 0.01` over the first 5 common seeds, usually seeds 100-104.
- Do not revive rejected r75 theories here: Graphviz 7.0.5 raw-modulo coarsening RNG is correct, `overlap=false` is rejected, and `p_neg2` clamp is already correct.

## Executive Verdict

- All 47 rows are structurally apart by the requested saved-position RMSD criterion. Near-match count `<0.01`: 0/47. Structural count `>=0.01`: 47/47.
- The 44 SFDP rows split evenly enough to justify two primary clusters:
  - disconnected/component-packer rows: 21/44 SFDP rows; strongest first-divergence evidence is the shared disconnected packer/component loop, before any whole-graph SFDP hierarchy can be compared.
  - connected spring-electrical residual rows: 23/44 SFDP rows; initial random coordinates match libc/Graphviz exactly, and Graphviz runtime parameters match Dagua's intended fidelity path. First unresolved divergence is inside multilevel matching/prolongation or spring-electrical iterations, not the rejected RNG/overlap/theta hypotheses.
- The 3 FDP-family transfer rows are Graphviz FDP rows, not SFDP. They should stay out of SFDP fixes. Their first divergent stage is Graphviz FDP layout/packing (`fdp`, not `sfdp`); route to Graphviz FDP parity or aggregate-tier review.
- No "FP-chaos floor" label is assigned. Per JMT ruling, bisection has not stopped finding op/stage differences, and no 1-ULP perturbation experiment was run.

## 1. Leg Breakdown

### High-Level Counts

| Bucket | Rows | Disconnected | Engines |
|---|---:|---:|---|
| SFDP only | 44 | 21 | `default` 13, `graphviz_fidelity` 14, `p_neg2` 17 |
| FDP-family transfer | 3 | 2 | `classic_fmmm_graphviz_fdp_fidelity` 3 |
| All rows | 47 | 23 | 4 engine labels |

### Failing-Leg Pattern Clusters

`D>R` means Dagua metric mean is greater than reference by more than that leg's margin. `D<R` means lower by more than margin. `margin` means direct-equivalence failed, but the mean gap itself is inside the scalar margin and the failure came from the battery/direct-equivalence machinery.

| Failing-leg pattern | Rows | Disc | Graphs | Stress rel median/min/max | Cross gap median/min/max | RMSD median/min/max |
|---|---:|---:|---|---:|---:|---:|
| battery_D>R; stress_D>R; cross_margin | 10 | 6 | asymmetric_hourglass_hub, disconnected_encoder_residual, disconnected_label_cycle_collage, random_dag_50, real_lesmis_77 | 0.2766 / 0.0979 / 3.2256 | 0.0667 / -28.5 / 58.9167 | 0.2230 / 0.0767 / 0.3841 |
| battery_D<R; stress_D<R; cross_D<R | 8 | 0 | hexagonal_lattice_42, real_karate_34, weighted_karate_34 | -0.1122 / -0.2788 / -0.1082 | -9.9833 / -11.45 / -2.05 | 0.0720 / 0.0355 / 0.0929 |
| battery_D>R; stress_D>R | 4 | 1 | disconnected_label_cycle_collage, weighted_chain_20 | 0.1958 / 0.1958 / 3.2256 | 0 / 0 / 0 | 0.0724 / 0.0535 / 0.3841 |
| battery_D<R; stress_D<R; cross_margin | 3 | 0 | extreme_mixed_width_transformer, real_karate_34, sparse_pair_50 | -0.1567 / -0.2773 / -0.1122 | 0.2333 / -9.8167 / 0.3833 | 0.0668 / 0.0372 / 0.1555 |
| battery_D<R; stress_D<R; cross_margin; np_D<R | 3 | 3 | multi_component_80 | -0.1208 / -0.1208 / -0.1208 | 0.1 / 0.05 / 0.1333 | 0.0940 / 0.0916 / 0.0940 |
| battery_D>R; stress_margin; cross_margin | 3 | 3 | kitchen_sink_platform_graph | 0.0279 / 0.0279 / 0.0279 | 0.0833 / 0.05 / 0.1167 | 0.2874 / 0.2874 / 0.2877 |
| battery_margin; stress_margin; cross_D<R | 3 | 3 | random_dag_200 | 0.0115 / -0.0045 / 0.0157 | -484.1667 / -652.0333 / -347.0167 | 0.0692 / 0.0684 / 0.0693 |
| battery_margin; stress_margin; np_D<R | 3 | 3 | parallel_cycles_4x5 | 0.0000145 / same / same | 0 / 0 / 0 | 0.2207 / 0.2207 / 0.2246 |
| battery_D>R; stress_D>R; cross_D>R | 2 | 0 | planar_60, real_lesmis_77 | 0.2493 / 0.0979 / 0.2493 | 66.7667 / 5.5833 / 66.7667 | 0.0767 / 0.0201 / 0.0767 |
| battery_margin; stress_D<R; cross_margin | 2 | 1 | planar_60, random_bipartite_60 | -0.0576 / -0.0639 / -0.0576 | 8.0667 / 2.1667 / 8.0667 | 0.1717 / 0.0116 / 0.1717 |
| singletons | 5 | 3 | clustered_longlabel_handoffs, long_range_residual_ladder, parallel_cycles_4x5, random_dag_50, real_lesmis_77 | mixed | mixed | 0.0231 to 0.1239 |

Important size flags:

- All rows in `r75_final.jsonl` report `n` near 99-100 for this filtered set, but graph builder node counts can differ because benchmark scoring includes layout/sample metadata and some reference graph builders include isolated/label-derived nodes. I used the row's `n` for the table and graph-builder counts only for stage probes.
- `random_dag_200` is a disconnected outlier with huge crossing gaps: -347.02 to -652.03 crossings while stress is within or near margin.
- `parallel_cycles_4x5` SFDP rows fail battery/NP even though stress gap is only `1.45e-05` relative and crossings match exactly. They are still structurally apart by RMSD around 0.22.

## 2. Hairline vs Structural

Saved-position Procrustes RMSD summary:

| Group | Rows | Median RMSD min/median/max | Near `<0.01` | Structural `>=0.01` |
|---|---:|---:|---:|---:|
| SFDP `p_neg2` | 17 | 0.0201 / 0.0871 / 0.3724 | 0 | 17 |
| SFDP `default` | 13 | 0.0535 / 0.0940 / 0.3841 | 0 | 13 |
| SFDP `graphviz_fidelity` | 14 | 0.0116 / 0.0940 / 0.3841 | 0 | 14 |
| FDP-transfer | 3 | 0.0761 / 0.1336 / 0.1555 | 0 | 3 |
| All | 47 | 0.0116 / 0.0930-ish / 0.3841 | 0 | 47 |

Smallest structural rows:

| Combo | Disc | Legs | Median RMSD | Main gap |
|---|---:|---|---:|---|
| `planar_60::classic_sfdp_graphviz_fidelity` | N | battery+stress+cross | 0.0116 | stress -5.76%, cross +2.17 |
| `planar_60::classic_sfdp_p_neg2` | N | battery+stress+cross | 0.0201 | stress +24.93%, cross +5.58 |
| `clustered_longlabel_handoffs::classic_sfdp_p_neg2` | N | battery+stress | 0.0231 | stress -1.47%, cross 0 |
| `hexagonal_lattice_42::classic_sfdp_p_neg2` | N | battery+stress+cross | 0.0355 | stress -27.88%, cross -2.05 |

Largest structural rows:

| Combo | Disc | Median RMSD | Main gap |
|---|---:|---:|---|
| `disconnected_label_cycle_collage::classic_sfdp_default` | Y | 0.3841 | stress +322.56% |
| `disconnected_label_cycle_collage::classic_sfdp_graphviz_fidelity` | Y | 0.3841 | stress +322.56%, cross +0.0667 |
| `disconnected_label_cycle_collage::classic_sfdp_p_neg2` | Y | 0.3724 | stress +322.56%, cross +0.0667 |
| `kitchen_sink_platform_graph::*sfdp*` | Y | 0.2874-0.2877 | stress inside margin-ish, cross +0.05 to +0.1167 |

Verdict: this 47-row residue is not a crossings-discreteness/hairline-only bucket under the requested RMSD test. Some failing legs are margin-discrete, but the underlying coordinates are not near-matches.

## 3. First-Divergence Hypotheses by Cluster

### Cluster A: Disconnected SFDP Component/Packer Rows

Rows: 21 SFDP rows. Graphs include `disconnected_encoder_residual`, `disconnected_label_cycle_collage`, `kitchen_sink_platform_graph`, `multi_component_80`, `parallel_cycles_4x5`, `random_bipartite_60`, `random_dag_50`, `random_dag_200`.

Source/stage evidence:

- Graphviz 7.0.5 `sfdp_layout()` creates one `spring_electrical_control ctrl`, calls `tuneControl()`, then splits components with `ccomps()`. For `ncc > 1`, it calls `sfdpLayout(sg, ctrl, pad)` for every component and then `packSubgraphs()`. Pinned source: `7.0.5:lib/sfdpgen/sfdpinit.c:268-315`.
- Dagua branches before whole-graph pipeline construction when graphviz fidelity sees multiple weak components. It calls `_layout_graphviz_sfdp_components()`, recursively runs `layout_sfdp_pipeline(..., fidelity_mode="graphviz")` per component, and then calls `_pack_component_positions()` from the neato packer.
- Graphviz `-v` on `parallel_cycles_4x5` prints `pack info:` once, then four component layouts with `avg edge len=0.649666`, then `step size = 72`.
- Dagua scratch pack probe on `parallel_cycles_4x5` found components `[5,5,5,5]`. Each component was laid out independently with identical local bbox `[644.134399, 672.969238]`; final packed bbox was `[1594.134277, 1702.135986]`.
- Graphviz `-v` on `random_dag_200` printed `pack info:` followed by many singleton component passes with `avg edge len=1.000000`, then the large component tail. Dagua component probe found 200-ish components dominated by singletons plus one 181-node component and one 2-node component; singleton local bboxes were `[0,0]`.

First divergent quantity named:

- Component orchestration and packing state. Specifically: Graphviz reuses one mutable `spring_electrical_control` through the `ccomps()` loop and uses Graphviz `packSubgraphs`; Dagua recursively constructs independent component pipeline states and uses `_pack_component_positions`.
- This happens before a whole-graph SFDP hierarchy comparison is meaningful for disconnected graphs.

Ruling:

- NEEDS-EXPERIMENT, not approved fix yet. The source/stage difference is real and matches the structural disconnected cluster, but no scratch branch showed before/after flips.

Minimal close path:

- Fix path: implement a Graphviz-faithful disconnected SFDP component loop behind graphviz-fidelity only, preserving mutable `ctrl` fields across components and validating pack offsets against Graphviz `packSubgraphs`.
- Expected rung: structural rows first, especially `parallel_cycles_4x5`, `random_dag_200`, `kitchen_sink_platform_graph`, `multi_component_80`.
- Effort: M-L. Risk is shared packer blast radius; gate to SFDP graphviz-fidelity + disconnected only.

### Cluster B: Connected SFDP Spring-Electrical / Multilevel Residual Rows

Rows: 23 SFDP rows. Graphs include `asymmetric_hourglass_hub`, `hexagonal_lattice_42`, `real_karate_34`, `weighted_karate_34`, `weighted_chain_20`, `planar_60`, `real_lesmis_77`, `sparse_pair_50`, `long_range_residual_ladder`, `clustered_longlabel_handoffs`.

Stage evidence:

- Initial RNG check: libc `srand(100); rand()/RAND_MAX` first values are `0.315598, 0.284943, 0.240601, 0.484127, 0.375793, 0.053703`. Dagua `GraphvizRandom(seed=100)` coarsest random start produced the same first pairs: `[[0.315598,0.284943],[0.240601,0.484127],[0.375793,0.053703]]`.
- Graphviz `-v` on `asymmetric_hourglass_hub` printed: random start 1 seed 100; `K : -1.000 C : 0.200`; Barnes-Hutt 0.600; tolerance 0.001; maxiter 500; cooling 0.900; step 0.100; smoothing NONE; overlap 0; `avg edge len=0.925345`.
- Dagua hierarchy probe for `asymmetric_hourglass_hub`: levels `[14,8,4]`, coarsest `K0=0.4382954539`, coarsest adjacency entries 6.
- Graphviz `-v` on `hexagonal_lattice_42`: same control settings; `avg edge len=2.115616`.
- Dagua hierarchy probe for `hexagonal_lattice_42`: levels `[42,24,12,6]`, coarsest `K0=0.3065298621`, coarsest adjacency entries 18.
- Pinned source confirms Graphviz dispatches normal `spring_electrical_embedding()` for normal quadtree mode unless fast/hybrid conditions apply: `7.0.5:lib/sfdpgen/spring_electrical.c:1853-1863`; iteration loop computes per-node forces, updates coordinates immediately, then updates step: `spring_electrical.c:1202-1265`, `1278` area and `290-303`.

First divergent quantity named:

- Not initial random coordinates. The first unresolved divergence is between matrix hierarchy/matching/prolongation state and the first spring-electrical iterations. The probes did not capture a Graphviz internal hierarchy dump, so this remains at the boundary: "after matched random start, before final spring-electrical convergence."

Ruling:

- NEEDS-EXPERIMENT. No FP-chaos/floor label is allowed yet. The next bisection must compare Graphviz `Multilevel.c` hierarchy levels/matching against Dagua `BuildGraphvizSFDPMatrixHierarchy` for one connected graph, then compare first iteration force norm/step on the coarsest level.

Minimal close path:

- Experiment path: compile or locally instrument pinned Graphviz 7.0.5 with DEBUG output for hierarchy sizes, cluster maps, initial coarsest coordinates, first 3 iteration force norms, and final step. Compare to Dagua scratch instrumentation for `asymmetric_hourglass_hub` and `hexagonal_lattice_42`.
- If op difference found: fix the named op (`sfdp_graphviz_matrix_coarsen_hierarchy`, `_SFDPGraphvizSequentialStep`, or prolongation jitter) and target rung-1/2 flips on connected SFDP rows.
- If no op difference remains: run the required 1-ULP perturbation experiment from the same bisection endpoint before proposing floor evidence.
- Effort: M for hierarchy/first-iteration trace; L if Graphviz debug build is required.

### Cluster C: FDP-Family Transfer Rows

Rows:

| Combo | Disc | Legs | Stress D/R/rel | Cross D/R/gap | RMSD |
|---|---:|---|---:|---:|---:|
| `extreme_mixed_width_transformer::classic_fmmm_graphviz_fdp_fidelity` | N | battery+stress+cross | 0.08938 / 0.1237 / -0.2773 | 1.883 / 1.5 / +0.3833 | 0.1555 |
| `parallel_cycles_4x5::classic_fmmm_graphviz_fdp_fidelity` | Y | battery+stress+cross | 0.07859 / 0.1311 / -0.4008 | 2.4 / 0.2 / +2.2 | 0.0761 |
| `random_dag_50::classic_fmmm_graphviz_fdp_fidelity` | Y | battery+stress+cross | 0.2937 / 0.2652 / +0.1073 | 365.7 / 394.2 / -28.5 | 0.1336 |

Stage evidence:

- Graphviz `-v -Kfdp` on `extreme_mixed_width_transformer` printed `pack info:`, `layout G`, `xLayout tries = 9, mode = prism`, node separation, and edge separation. It does not enter SFDP `spring_electrical_control`.
- Therefore these 3 rows are not explained by SFDP multilevel or SFDP component control.

First divergent quantity named:

- FDP layout stage, not SFDP. For disconnected FDP-transfer rows, packing may also be involved, but the reference engine is Graphviz FDP.

Ruling:

- Route separately. Do not validate SFDP fixes against these three rows except as non-regression data.

Minimal close path:

- Either triage Graphviz FDP fidelity directly (`fdp` xLayout/prism/packer path) or mark as aggregate-tier candidate if population metrics allow.
- Effort: S to route/exclude from SFDP closure; M for real FDP parity probe.

## 4. Row-Level Numbers

| Combo | Disc | Legs | Stress rel | Cross gap | NP rel | RMSD |
|---|---:|---|---:|---:|---:|---:|
| asymmetric_hourglass_hub::classic_sfdp_p_neg2 | N | battery+stress+cross | +0.1271 | +0.0333 | -0.0009 | 0.1001 |
| asymmetric_hourglass_hub::classic_sfdp_default | N | battery+stress+cross | +0.1271 | +0.0333 | -0.0009 | 0.1406 |
| asymmetric_hourglass_hub::classic_sfdp_graphviz_fidelity | N | battery+stress+cross | +0.1271 | +0.0667 | -0.0009 | 0.1406 |
| clustered_longlabel_handoffs::classic_sfdp_p_neg2 | N | battery+stress | -0.0147 | 0 | 0 | 0.0231 |
| disconnected_encoder_residual::classic_sfdp_graphviz_fidelity | Y | battery+stress+cross | +0.2766 | +0.1333 | -0.0052 | 0.2230 |
| disconnected_encoder_residual::classic_sfdp_default | Y | battery+stress+cross | +0.2766 | +0.1000 | -0.0052 | 0.2230 |
| disconnected_encoder_residual::classic_sfdp_p_neg2 | Y | battery+stress+cross | +0.2766 | +0.1167 | -0.0031 | 0.2252 |
| disconnected_label_cycle_collage::classic_sfdp_default | Y | battery+stress | +3.2256 | 0 | +0.3118 | 0.3841 |
| disconnected_label_cycle_collage::classic_sfdp_graphviz_fidelity | Y | battery+stress+cross | +3.2256 | +0.0667 | +0.3118 | 0.3841 |
| disconnected_label_cycle_collage::classic_sfdp_p_neg2 | Y | battery+stress+cross | +3.2256 | +0.0667 | +0.3032 | 0.3724 |
| extreme_mixed_width_transformer::classic_fmmm_graphviz_fdp_fidelity | N | battery+stress+cross | -0.2773 | +0.3833 | 0 | 0.1555 |
| hexagonal_lattice_42::classic_sfdp_graphviz_fidelity | N | battery+stress+cross | -0.2788 | -2.3667 | +0.0377 | 0.0592 |
| hexagonal_lattice_42::classic_sfdp_default | N | battery+stress+cross | -0.2788 | -2.1500 | +0.0377 | 0.0592 |
| hexagonal_lattice_42::classic_sfdp_p_neg2 | N | battery+stress+cross | -0.2788 | -2.0500 | +0.0316 | 0.0355 |
| kitchen_sink_platform_graph::classic_sfdp_default | Y | battery+stress+cross | +0.0279 | +0.1167 | +0.0595 | 0.2874 |
| kitchen_sink_platform_graph::classic_sfdp_graphviz_fidelity | Y | battery+stress+cross | +0.0279 | +0.0500 | +0.0621 | 0.2874 |
| kitchen_sink_platform_graph::classic_sfdp_p_neg2 | Y | battery+stress+cross | +0.0279 | +0.0833 | +0.0687 | 0.2877 |
| long_range_residual_ladder::classic_sfdp_p_neg2 | N | battery+stress+cross+np | -0.2054 | +0.5667 | -0.0247 | 0.0555 |
| multi_component_80::classic_sfdp_default | Y | battery+stress+cross+np | -0.1208 | +0.1000 | -0.1080 | 0.0940 |
| multi_component_80::classic_sfdp_p_neg2 | Y | battery+stress+cross+np | -0.1208 | +0.0500 | -0.1156 | 0.0916 |
| multi_component_80::classic_sfdp_graphviz_fidelity | Y | battery+stress+cross+np | -0.1208 | +0.1333 | -0.1053 | 0.0940 |
| parallel_cycles_4x5::classic_fmmm_graphviz_fdp_fidelity | Y | battery+stress+cross | -0.4008 | +2.2000 | +0.1836 | 0.0761 |
| parallel_cycles_4x5::classic_sfdp_default | Y | battery+stress+np | +0.0000145 | 0 | -0.1118 | 0.2207 |
| parallel_cycles_4x5::classic_sfdp_graphviz_fidelity | Y | battery+stress+np | +0.0000145 | 0 | -0.1095 | 0.2207 |
| parallel_cycles_4x5::classic_sfdp_p_neg2 | Y | battery+stress+np | +0.0000145 | 0 | -0.1218 | 0.2246 |
| planar_60::classic_sfdp_graphviz_fidelity | N | battery+stress+cross | -0.0576 | +2.1667 | -0.0066 | 0.0116 |
| planar_60::classic_sfdp_p_neg2 | N | battery+stress+cross | +0.2493 | +5.5833 | -0.0054 | 0.0201 |
| random_bipartite_60::classic_sfdp_p_neg2 | Y | battery+stress+cross | -0.0639 | +8.0667 | -0.0098 | 0.1717 |
| random_dag_200::classic_sfdp_p_neg2 | Y | battery+stress+cross | -0.0045 | -484.1667 | -0.3196 | 0.0692 |
| random_dag_200::classic_sfdp_default | Y | battery+stress+cross | +0.0115 | -347.0167 | -0.2746 | 0.0693 |
| random_dag_200::classic_sfdp_graphviz_fidelity | Y | battery+stress+cross | +0.0157 | -652.0333 | -0.1946 | 0.0684 |
| random_dag_50::classic_fmmm_graphviz_fdp_fidelity | Y | battery+stress+cross | +0.1073 | -28.5000 | +0.6214 | 0.1336 |
| random_dag_50::classic_sfdp_default | Y | battery+stress+cross | +0.0422 | +27.1833 | +0.0018 | 0.1239 |
| random_dag_50::classic_sfdp_graphviz_fidelity | Y | battery+stress+cross+np | -0.0873 | +34.6000 | -0.1760 | 0.1222 |
| real_karate_34::classic_sfdp_default | N | battery+stress+cross | -0.1122 | -9.8167 | +0.0326 | 0.0668 |
| real_karate_34::classic_sfdp_graphviz_fidelity | N | battery+stress+cross | -0.1122 | -9.9833 | +0.0326 | 0.0668 |
| real_karate_34::classic_sfdp_p_neg2 | N | battery+stress+cross | -0.1122 | -10.4333 | +0.0332 | 0.0929 |
| real_lesmis_77::classic_sfdp_graphviz_fidelity | N | battery+stress+cross | +0.0979 | +66.7667 | +0.0115 | 0.0767 |
| real_lesmis_77::classic_sfdp_default | N | battery+stress+cross | +0.0979 | +58.9167 | +0.0115 | 0.0767 |
| real_lesmis_77::classic_sfdp_p_neg2 | N | battery+stress+cross | -0.0366 | +132.2833 | -0.0287 | 0.0871 |
| sparse_pair_50::classic_sfdp_p_neg2 | N | battery+stress+cross | -0.1567 | +0.2333 | +0.0202 | 0.0372 |
| weighted_chain_20::classic_sfdp_graphviz_fidelity | N | battery+stress | +0.1958 | 0 | -0.0095 | 0.0535 |
| weighted_chain_20::classic_sfdp_default | N | battery+stress | +0.1958 | 0 | -0.0095 | 0.0535 |
| weighted_chain_20::classic_sfdp_p_neg2 | N | battery+stress | +0.1958 | 0 | -0.0106 | 0.0724 |
| weighted_karate_34::classic_sfdp_default | N | battery+stress+cross | -0.1082 | -11.1667 | +0.0761 | 0.0720 |
| weighted_karate_34::classic_sfdp_p_neg2 | N | battery+stress+cross | -0.1082 | -10.9333 | +0.0761 | 0.0785 |
| weighted_karate_34::classic_sfdp_graphviz_fidelity | N | battery+stress+cross | -0.1082 | -11.4500 | +0.0761 | 0.0720 |

## 5. Recommendations and ROI

| ROI | Cluster | Rows | Minimal close path | Expected rung | Effort |
|---:|---|---:|---|---|---|
| 1 | Disconnected SFDP component/packer | 21 | Scratch patch shared Graphviz-like component control + packSubgraphs parity probe; gate to `fidelity_mode="graphviz"` and `len(components)>1`. | Structural flips first; likely rung 1/2 for disconnected rows if pack/control is correct. | M-L |
| 2 | FDP-transfer rows | 3 | Rebucket out of SFDP triage. Either FDP-specific probe or aggregate-tier candidate. | Accounting/routing, not SFDP layout. | S |
| 3 | Connected SFDP hierarchy/iteration residual | 23 | Instrument pinned Graphviz 7.0.5 hierarchy and first iteration; compare to Dagua hierarchy/prolongation/force norm. | If op mismatch: targeted layout fix. If no op mismatch: perturbation evidence before floor. | M-L |
| 4 | Floor-evidence path | 0 approved now | Only after bisection endpoint shows no op differences; run 1-ULP initial-position or first-force perturbation and require final metric deltas comparable to observed gaps. | Evidenced floor or aggregate-tier. | S after bisection, not before |

Do not spend time on:

- `theta`/`maxiter` no-canonical variants: already excluded.
- `GraphvizRandom.permutation()` rejection sampling: rejected for Graphviz 7.0.5.
- `overlap=false`: rejected for installed Graphviz 7.0.5 build.
- Reverting `p_neg2` clamp: already approved as correct.

## Concerns

- The connected SFDP first-divergence claim is intentionally conservative. I did not have a Graphviz debug build dumping internal `Multilevel` cluster maps or first force norms, so I stop at the first unproven boundary rather than calling floor.
- The position overlay is sparse. I used freshest existing tensors per side/seed; this follows the requested overlay direction but means Dagua and reference positions often came from different benchmark directories.
- The stage probe used generated benchmark graphs directly. That is appropriate for first-divergence probing, but exact benchmark scoring rows may include metadata-normalized node sizing that should be confirmed before implementing a fix.

## Knowledge

- Remaining SFDP rows are structurally apart by saved-position RMSD; this residue is not a near-match-only crossings-margin bucket.
- Disconnected SFDP remains the clearest fix candidate: Graphviz's `ccomps` + shared control + `packSubgraphs` path diverges from Dagua's recursive component pipeline + neato packer.
- Connected SFDP initial random coordinates match libc/Graphviz for seed 100, so the next useful work is hierarchy/first-iteration tracing, not RNG or overlap.
- FDP-transfer rows are real Graphviz FDP rows and should not be used to judge SFDP fixes.
