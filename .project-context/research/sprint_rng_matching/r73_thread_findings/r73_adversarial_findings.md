# R73 Adversarial Findings

Scope: read-only review of `/tmp/r73_synthesis.md`, `/tmp/r73_thread{1..6}_findings.md`, `eval_output/fidelity_definitive_r72/per_combo.json`, control JSONL files, and local upstream Graphviz/OGDF source trees.

## 1. Pile 2 / 3Q-Mode-B Extension

**Verdict: claim is wrong; proposed rule is laundering-prone and internally inconsistent. Reject as written.**

### Evidence

1. The stated rule is `stress_p_tost < 0.05 AND cross_p_tost < 0.05 AND np_p_tost < 0.05`.
   Applying exactly that rule:
   - `gate3_negative.jsonl`: 0 / 20 pass.
   - `gate4_chance.jsonl`: 5 / 20 pass.
   - Combined controls: 5 / 40 pass = 12.5%, above the existing 5% laundering guard.

2. The proposed rule would promote **0 / 617** current rung-4 rows in `per_combo.json`.
   The claimed 161-218 promotions are not produced by the stated `< 0.05` rule.

3. The thread’s claimed 161-ish count comes from the opposite convention, effectively requiring all three p-values to be `> 0.05`:
   - Current rung-4 rows with all three p-values `> 0.05`: 208 total.
   - Breakdown: 94 Mode A, 114 Mode B.
   - Families: sugiyama 101, sfdp 47, fmmm 29, pivot 13, gem 10, drl 3, umap 2, maxent 2, neato 1.
   - That opposite rule is also unsafe on controls: 8 / 20 negative controls and 3 / 20 chance controls pass, 11 / 40 = 27.5%.

4. The margin story is misstated:
   - `stress_p_tost` uses `max(0.05 * stress_R, 1e-6)` in `compute_mode_b_stress`; this is a **5% stress margin**, not the strict 2% quality battery.
   - Strict stress battery is in `battery_stress_p_tost` and uses `QUALITY_STRESS_REL_MARGIN = 0.02`.
   - `cross_p_tost` and `np_p_tost` are strict battery fields: crossings use `max(2% * cross_R, 0.5)`, kNN uses absolute 0.02.
   - Therefore the proposed gate mixes loose 5% stress with strict crossing/kNN fields. It is not the registered 3Q battery.

5. Correct statistical interpretation:
   - Berger-Hsu/IUT justifies no multiplicity correction **within one combo across the three metrics**. It does not justify no error control across 3,955 equivalence claims.
   - Without family correction across combos, the rule controls only per-comparison Type I error. Under a global-null family, expected false equivalence claims can be about `0.05 * 3955 = 197.75`, and FWER is effectively 1.
   - A published “quality-identical” inventory needs cross-combo FDR/FWER control: BH over a pre-registered family for FDR, BY/Holm if dependence or any false claim is unacceptable. Family splitting can be valid only if pre-specified, not chosen after seeing the counts.

### Corrected Recommendation

Do not promote any rows via the proposed per-combo p-field gate. Keep the strict battery as the final 3Q authority, with cross-combo FDR control on `battery_p_iut`, or create a separate exploratory label such as `quality_candidate_unadjusted` that is explicitly not final-rung 3Q. If the team wants more power, re-run controls and calibrate a pre-registered family/margin; do not reclassify from these fields now.

## 2. Pile 2 / “Degenerate-Optimal == Quality-Identical”

**Verdict: claim is wrong as a blanket reclassification. Some cases may be alternate optima, but the quality battery does not pass.**

### Sugiyama Samples

All sampled rows below have `quality_identical_raw=False` and `battery_p_iut=1.0`.

| Combo | Stress D/R gap vs margin | Cross D/R gap vs margin | kNN D/R gap vs margin | Binding metric |
|---|---:|---:|---:|---|
| `moe_router_sparse::classic_sugiyama_default` | 0.00735 vs 0.00250 | 0 vs 0.5 | 0 vs 0.02 | stress |
| `kitchen_sink_platform_graph::classic_sugiyama_default` | 0.00640 vs 0.00474 | 0 vs 0.5 | 0.111 vs 0.02 | kNN |
| `binary_tree::classic_sugiyama_graphviz_fidelity` | 0.0284 vs 0.00813 | 0 vs 0.5 | 0 vs 0.02 | stress |
| `ba_500::classic_sugiyama_default` | 0.135 vs 0.0237 | 22,344 vs 2,805.52 | 0.0404 vs 0.02 | all |
| `ba_2000::classic_sugiyama_default` | 0.190 vs 0.0263 | 197,914 vs 42,115.94 | 0.0300 vs 0.02 | all |
| `random_dag_50::classic_sugiyama_graphviz_fidelity` | 0.0693 vs 0.0190 | 17 vs 9.86 | ~0 vs 0.02 | stress/cross |
| `triangular_lattice_36::classic_sugiyama_graphviz_fidelity` | 0.0534 vs 0.00108 | 0 vs 0.5 | 0.0556 vs 0.02 | stress/kNN |
| `width_skew_late_merge::classic_sugiyama_default` | 0.0356 vs 0.00894 | 0 vs 0.5 | 0.0294 vs 0.02 | stress/kNN |
| `org_chart_1_5_4_8::classic_sugiyama_graphviz_fidelity` | 0.0715 vs 0.0104 | 0 vs 0.5 | 0.0444 vs 0.02 | stress/kNN |
| `densenet_block::classic_sugiyama_default` | 0.0651 vs 0.0106 | 4 vs 0.5 | 0 vs 0.02 | stress/cross |

These are not “battery-quality-identical.” The layer/LP degeneracy explanation may describe why positions differ, but it does not imply equal drawing quality under the registered metrics.

### MDS Samples

The r72 summary lacks cross/kNN battery fields for classical MDS rows (`None`), so the “pass all three metrics” claim is unsupported by the definitive data. The stored stress diagnostics already fail 5% stress margins for named degenerate cases:

| Combo | Stored stress D/R | 5% margin | Result |
|---|---:|---:|---|
| `bipartite_4_3_4::classic_classical_mds_default` | 0.2344 / 0.2677 | 0.0134 | fail |
| `center_port_backedge_hub::classic_classical_mds_default` | 0.1869 / 0.3455 | 0.0173 | fail |
| `densenet_block::classic_classical_mds_default` | 0.1291 / 0.1487 | 0.00744 | fail |
| `petersen_10::classic_classical_mds_default` | 0.1736 / 0.1605 | 0.00802 | fail |
| `wide_single_layer_1_50_1::classic_classical_mds_default` | 0.5999 / 0.6749 | 0.0337 | fail |
| `wide_3_50_3::classic_classical_mds_default` | 1.0000 / 0.4388 | 0.0219 | fail |

Additional cached-layout spot check from `eval_output/benchmark_5seed_fidelity/positions` using project metric functions showed identity-sensitive kNN/crossing can diverge:
`wide_single_layer_1_50_1` kNN delta 0.0231 (>0.02), crossing delta 5; `wide_3_50_3` kNN delta 0.0304 (>0.02), crossing delta 886.

### Corrected Recommendation

Do not reclassify sugiyama-LP-A or MDS-degenerate rows as 3Q from degeneracy arguments. If alternate optima should be accepted, define a new explicit category such as `ALTERNATE_OPTIMUM_NOT_QUALITY_IDENTICAL`. It must not be merged into “quality-identical” unless the strict stress/cross/kNN battery passes.

## 3. Pile 3 / FLOOR Calls

**Verdict: risky-needs-guard. Some floor calls are plausible, but the evidence is too weak and sometimes mislabels fixable/canonical divergences as absurd.**

### Evidence

1. `sfdp-137`: The thread asserts Lyapunov/libm chaos, but no actual Lyapunov estimate, 1-ULP perturbation experiment, or instrumented Graphviz trace is provided in the reviewed findings. “Stress passes but kNN/crossings fail” is a symptom, not proof of irreducibility.

2. `fmmm-72`: Many “floor” calls are low-iteration or not-yet-converged variants. That is a benchmark-definition limitation, not automatically an absurd implementation floor. If step count is part of the reference contract, it may still be divergent; but the floor label should require matched RNG, matched packing, matched component order, matched edge handling, and a perturbation experiment.

3. `sugiyama-A`: Calling GLPK zero-objective degeneracy “floor” is over-strong. Matching GLPK’s pivot trajectory may be absurd, but choosing a canonical optimal vertex is not. A deterministic secondary objective or lexicographic canonicalization could make dagua stable and arguably better, though not necessarily bit-identical to igraph. That is a product/contract decision, not proven irreducible FP chaos.

4. For sugiyama-A specifically, the quality metrics often fail. Even if both layerings are LP-optimal, they are not quality-identical under the registered drawing battery.

### Corrected Recommendation

Use a stricter floor standard:
- Accept `floor` only after a matched-reference trace has ruled out parameter, packing, component-order, edge-weight, and seed-semantics mismatches.
- Require either a 1-ULP perturbation/basin experiment or a source-level proof that remaining divergence is from an implementation-defined solver pivot/libm path.
- For sugiyama-A, dispatch a short design spike: test secondary objectives/canonical tie-breaks and compare quality, not GLPK pivot reproduction. Keep rows divergent unless the benchmark contract is changed to “canonical optimal,” and do not call them 3Q.

## 4. Pile 1 / Fixes, Overlap, and Source Claims

**Verdict: mixed. Some fixes hold, some are risky, and at least one upstream-source claim is wrong.**

### 4a. Sugiyama B+C+D overlap

The sub-bucket counts are non-exclusive. Igraph sugiyama has 166 divergent combos. If A is ~135 and B is ~70, inclusion-exclusion forces at least `135 + 70 - 166 = 39` B combos to also have A divergence. Fixing B alone cannot flip those rows.

Graphviz bucket has 65 combos; C+D+E counts 23+26+16 = 65, so that bucket can be disjoint. The actual maximum post-fix flip estimate is therefore about:
- igraph non-A capacity: 166 - 135 = 31
- graphviz C+D: 23 + 26 = 49
- total max flip: about 80

So `231 -> 151` can be a defensible upper bound only if it already accounts for A/B overlap. The table’s individual B/C/D counts should not be summed as independent.

### 4b. GEM seeded-reference

The harness bug is real: `classic_gem_iters100/500/2000` variants are stochastic (`classic_gem=True`), but `ogdf_gem` is registered deterministic (`ogdf_gem=False`), and `OGDFGem` does not declare stochasticity.

However, converting the reference to stochastic is legitimate only if seed semantics are guarded. The local runner does call both `ogdf::setSeed(seed)` and `std::srand(seed)` for GEM, but the reviewed synthesis does not include enough evidence that dagua consumes random draws in the same order for all relevant graphs/round caps. The matched-seed RMSD claim is encouraging, but should become a regression test across representative low-round and capped cases before promoting 22 combos.

Corrected recommendation: implement the harness fix, but require a matched-seed parity test matrix before accepting the reclassification. Do not use stochastic reference clouds to hide seed-mapping mismatch.

### 4c. Component Packing Source Claims

Graphviz/neato:
- Local Graphviz source confirms neato uses `getPackModeInfo(g, l_undef, &pinfo)`, defaults `pinfo.mode = l_node` when packing is turned on, then calls `packGraphs`.
- `pack.c` with graph/node modes routes to polyomino packing (`polyGraphs`/`polyRects`) sorted by perimeter, not the simple “area-sorted shelf” described in the thread.
- `arrayRects` exists for `packmode=array`; it is row/column array packing sorted by height+width unless input/user sort is requested. That is not the default neato path shown in `neatoinit.c`.

OGDF/FMMM:
- The claim “FMMM uses `TileToRowsCCPacker`” is wrong for `FMMMLayout`.
- Local OGDF source shows `FMMMLayout::pack_subGraph_drawings` calls `energybased::fmmm::MAARPacking::pack_rectangles_using_Best_Fit_strategy`.
- `TileToRowsCCPacker` is used by other OGDF layout paths (`GEMLayout`, `SpringEmbedderFRExact`, `SimpleCCPacker`, etc.), not the FMMM path under review.
- `FMMMLayout` defaults include `presortCCs(FMMMOptions::PreSort::DecreasingHeight)` and `tipOverCCs(FMMMOptions::TipOver::NoGrowingRow)`.

Corrected recommendation: keep component-packing fixes as plausible, but rewrite the specs against actual upstream algorithms:
- Neato: emulate Graphviz pack mode actually used by neato defaults (`l_node`/polyomino unless the benchmark sets `packmode=array`).
- FMMM: emulate `MAARPacking::pack_rectangles_using_Best_Fit_strategy`, not `TileToRowsCCPacker`.

## Prioritized De-Risked Dispatch Plan

1. **Reject the 3Q p-field promotion.** No reclassification from the proposed rule. It fails chance controls and the stated `<0.05` rule does not promote the claimed rows.

2. **Fix clear algorithm mismatches first:**
   - UMAP parallel-edge CSR accumulation. High confidence, bounded, testable.
   - Pivot MDS scale normalization. High confidence, but also audit weighted/unweighted distance handling after scale fix.
   - Classical MDS weighted default comparison: decide whether default should match unweighted igraph or be marked invalid-comparison; do not call degenerate MDS 3Q.

3. **Harness fixes with guardrails:**
   - GEM stochastic reference: implement only with matched-seed parity regression tests across low-round and capped cases.

4. **Packing fixes after source-correct specs:**
   - Graphviz/neato: verify actual `packmode` in benchmark inputs, then emulate the corresponding Graphviz `pack.c` branch.
   - OGDF/FMMM: replace the TileToRows spec with MAARPacking Best-Fit, decreasing-height presort, and FMMM tip-over defaults.

5. **Sugiyama:**
   - Do not reclassify LP-degenerate rows as quality-identical.
   - Fix graphviz mincross and x-coordinate mismatches only where layers/order are already matched.
   - Run a canonical-optimum spike for igraph LP degeneracy; classify outcomes as alternate-optimum only if the benchmark contract explicitly accepts that.

6. **Floor acceptance:**
   - Tentatively accept only narrow floors with source/trace evidence and no known structural mismatch.
   - Do not accept broad sfdp/fmmm floor buckets until perturbation or instrumented-trace evidence exists.

## Commands / Checks Run

- Parsed `eval_output/fidelity_definitive_r72/per_combo.json` (3,955 rows; 617 rung-4).
- Applied proposed IUT gates to `eval_output/fidelity_definitive/controls/gate3_negative.jsonl` and `gate4_chance.jsonl`.
- Queried strict battery and p-field margins in `scripts/definitive_fidelity_analysis.py`, `scripts/definitive_fidelity_report.py`, and `dagua/eval/distributional_fidelity.py`.
- Sampled sugiyama and MDS quality metrics from `per_combo.json`; computed extra MDS cached-layout kNN/crossing samples from `eval_output/benchmark_5seed_fidelity/positions`.
- Verified upstream packing paths in local Graphviz source `/home/jtaylor/projects/_references/graphviz/lib/pack/pack.c`, `/home/jtaylor/projects/_references/graphviz/lib/neatogen/neatoinit.c`, and OGDF source `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/FMMMLayout.cpp`, `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/fmmm/MAARPacking.cpp`, `/home/jtaylor/projects/_references/ogdf/src/ogdf/packing/TileToRowsCCPacker.cpp`.
