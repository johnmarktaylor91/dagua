# r76 Probe: classical_mds + GEM Triage

Date: 2026-07-02

Scope: research/probe only. No repo code changes. Scratch script used:
`/tmp/r76_probe.py`.

## Commands

```bash
sed -n '1,240p' .project-context/research/sprint_rng_matching/r75_RESULTS.md
python /tmp/r76_probe.py
python - <<'PY'  # connected-MDS eigensolver rerun with full evd solve
...
PY
```

Data:
- Divergent rows: `eval_output/fidelity_definitive/r75_final.jsonl`.
- Dagua positions: newest row from `benchmark_100seed_r75_fixes`,
  `benchmark_100seed_r75_mds_topup`, `benchmark_100seed_r75_topup2`.
- Paired historical references found in `benchmark_100seed_seeded_refs`.
- Fresh GEM first-divergence reference: current `scripts/ogdf_runner` with
  `--gemRounds`/JSON `gemRounds`.

## MDS Split

Confirmed count: 22 classical MDS divergent rows = 14 connected + 8 disconnected.

The connected 14 are exactly the r75 E2 set, both variants for:

| graph | variants |
| --- | --- |
| `bipartite_4_3_4` | default, igraph_fidelity |
| `center_port_backedge_hub` | default, igraph_fidelity |
| `densenet_block` | default, igraph_fidelity |
| `org_chart_1_5_4_8` | default, igraph_fidelity |
| `petersen_10` | default, igraph_fidelity |
| `wide_single_layer_1_50_1` | default, igraph_fidelity |
| `wide_3_50_3` | default, igraph_fidelity |

Disconnected residual rows:

| combo | failing legs | stress delta D-R | crossing delta D-R | verdict |
| --- | --- | ---: | ---: | --- |
| `disconnected_label_cycle_collage::default` | stress, battery_stress | +1.77e-06 | 0 | tiny stress-only, no crossing issue |
| `disconnected_label_cycle_collage::igraph_fidelity` | stress, battery_stress | +1.77e-06 | 0 | tiny stress-only, no crossing issue |
| `multi_component_80::default` | stress, battery_stress | -9.04e-04 | 0 | stress-only, no crossing issue |
| `multi_component_80::igraph_fidelity` | stress, battery_stress | -9.04e-04 | 0 | stress-only, no crossing issue |
| `random_dag_50::default` | stress, cross, battery_stress | +1.08e-02 | +60.98 | large-component crossing + stress |
| `random_dag_50::igraph_fidelity` | stress, cross, battery_stress | +9.45e-02 | -37.26 | large-component crossing + stress |
| `random_dag_200::default` | stress, cross, battery_stress | +7.70e-03 | -663.74 | sampled large-component crossing + stress |
| `random_dag_200::igraph_fidelity` | stress, cross, battery_stress | -5.22e-03 | +453.24 | sampled large-component crossing + stress |

Mechanism: stress sampling does not include cross-component pairs. Both
`dagua.eval.distributional_fidelity.sample_pairs()` and
`dagua.metrics.sampled_stress()` filter to finite/reachable graph distances.
The remaining disconnected stress failures are therefore within-component
geometry, not inter-component DLA spacing.

Component crossing verification, seed 100:

| graph/variant | graph size | components | total D crossings | total R crossings | finite stress pairs | cross-component pairs excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_dag_50::default` | 97 nodes / 70 edges | 52 | 346 | 369 | 991 | 3665 |
| `random_dag_200::default` | 383 nodes / 300 edges | 202 | 9782 | 9705 | 16291 | 56862 |

For both probes, all nonzero crossings are in the single nontrivial component
(`45 nodes / 69 edges` for `random_dag_50`; `181 nodes / 299 edges` for
`random_dag_200`). Singleton components contribute no crossings. DLA packing
itself is not creating crossings across components because there are no
inter-component edges.

## Connected MDS Floor Dossier

Source basis:
- Dagua uses SciPy `eigh(..., driver="evr")` in
  `dagua/layout/ops/pipelines/classical_mds.py:333-336`.
- r75 notes igraph asks LAPACK for top algebraic eigenvectors through its MDS
  LAPACK path, and the r75 E2 probe classified the connected rows as a
  degenerate eigenspace floor.

Current eigengap + driver demonstration:

| graph | top eigenvalues | gap12 | gap23 | evr-vs-evx RMSD | evr-vs-evd RMSD |
| --- | --- | ---: | ---: | ---: | ---: |
| `bipartite_4_3_4` | 2, 2, 2, 2, 2 | 0 | 4.44e-16 | 5.09e-18 | 0.213 |
| `center_port_backedge_hub` | 2, 2, 2, 2, ~0 | 4.44e-16 | 6.66e-16 | 5.51e-17 | 0.0151 |
| `densenet_block` | 3.3435, 0.5, 0.5, 0.5, 0.5 | 2.84 | 5.55e-17 | 1.14e-17 | 0.128 |
| `org_chart_1_5_4_8` | 44.4499, 44.4499, 44.4499, 2.7940, 2 | 7.11e-15 | 0 | 0 | 0.0142 |
| `petersen_10` | 3.5, 3.5, 3.5, 3.5, 3.5 | 0 | 8.88e-16 | 7.76e-19 | 0.195 |
| `wide_3_50_3` | 2, 2, 2, 2, 2 | 1.78e-15 | 4.44e-16 | 0.0722 | 0.0645 |
| `wide_single_layer_1_50_1` | 2, 2, 2, 2, 2 | 2.22e-15 | 1.33e-15 | 0.0374 | 0.0315 |

Disposition: floor-with-evidence. Six of seven have top-dimensional ties at or
near machine precision; `densenet_block` has the second coordinate drawn from a
four-way tied 0.5 eigenspace. A full `evd` solve changes coordinates materially
for every connected graph after Procrustes registration. This is enough for
r76 to cite as an eigensolver-basis floor rather than a port-fix target.

## GEM Rows

Confirmed count: 7 divergent GEM rows.

| combo | disc | failing legs | stress D/R | crossings D/R | mean diag B | plain W_D/W_R | tier |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `grid_5x5::iters100` | no | stress, cross, battery_stress | 0.0196 / 0.0658 | 0.310 / 2.143 | 0.253 | 0.087 / 0.400 | final_3q |
| `regular_4_40::iters100` | no | stress, cross, battery_stress | 0.122 / 0.143 | 106.5 / 111.6 | 0.402 | 0.456 / 0.637 | final_3q |
| `tl_resnet_2block::iters100` | no | stress, cross, battery_stress | 0.0151 / 0.0495 | 0 / 0.0952 | 0.276 | 0.235 / 0.458 | final_3q |
| `triangular_lattice_36::iters100` | no | stress, cross, battery_stress | 0.0118 / 0.0457 | 2.0 / 6.43 | 0.248 | 0.177 / 0.419 | final_3q |
| `random_dag_50::iters100` | yes | np, stress, cross, battery_stress | 0.344 / 0.347 | 410.7 / 384.1 | 1.091 | 1.051 / 1.061 | exploratory_noncanonical_reference |
| `random_dag_50::iters500` | yes | np, stress, cross, battery_stress | 0.341 / 0.316 | 407.2 / 360.2 | 1.114 | 1.044 / 1.029 | exploratory_noncanonical_reference |
| `random_dag_200::iters500` | yes | stress, cross, battery_stress | 0.285 / 0.275 | 8580 / 8349 | 1.023 | 1.000 / 1.012 | exploratory_noncanonical_reference |

Procrustes sample against the saved paired references:

| combo | seed100 | seed101 | seed102 | seed103 | seed104 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `grid_5x5::iters100` | 0.0375 | 0.0666 | 0.0178 | 0.0198 | 0.00876 |
| `random_dag_50::iters100` | 0.0757 | 0.0801 | 0.0802 | 0.0740 | 0.0906 |
| `random_dag_50::iters500` | 0.0805 | 0.0664 | 0.0765 | 0.0826 | 0.0824 |

Important artifact check: the only saved reference row found for
`grid_5x5::ogdf_gem__for__classic_gem_iters100::seed100` was in
`benchmark_100seed_seeded_refs` (`git_sha=0766227`). It does not match the
current rebuilt runner with `gemRounds=100`:

| comparison | Procrustes RMSD |
| --- | ---: |
| saved Dagua vs current direct Dagua pipeline | 3.3e-17 |
| saved reference vs fresh current runner | 0.165 |
| saved Dagua vs saved reference | 0.0375 |
| direct Dagua vs fresh current runner | 0.163 |

This supports r75's stale-reference warning for GEM: first-divergence probes
must use the current runner, not the old `seeded_refs` tensor.

## GEM First-Divergence Probe

Graph: `grid_5x5`, seed 100. Dagua was run through
`layout_gem_pipeline(..., max_iters=rounds, fidelity_mode="ogdf")`; reference
through current `scripts/ogdf_runner` with JSON `gemRounds=rounds`.

| rounds | Procrustes RMSD | max raw abs delta | Dagua bbox | runner bbox |
| ---: | ---: | ---: | --- | --- |
| 20 | 0.144 | 374.45 | [439.42, 451.85] | [119.98, 116.79] |
| 100 | 0.163 | 470.48 | [508.86, 517.98] | [187.61, 184.74] |

Verdict: divergence is already large by round 20 and grows only modestly by
round 100. This is not a late chaotic divergence signature. It points to an
early mismatch: initialization/reference artifact, exact RNG distribution, or
force arithmetic/geometry scale. The saved Dagua tensor equals the current
direct Dagua pipeline, so this is not a benchmark overlay issue on the Dagua
side.

Relevant source anchors:
- Runner seeds both OGDF and `std::rand`, initializes positions from
  `rand() % 1000 / 10.0`, then calls GEM:
  `scripts/ogdf_runner.cpp:309-317`, `scripts/ogdf_runner.cpp:414-423`.
- OGDF GEM component loop, random permutation, force impulse, and update:
  `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/GEMLayout.cpp:150-186`,
  `:240-288`, `:291-340`.
- OGDF `SList::permute` converts to an array and uses `Array::permute`:
  `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/SList.h:1108-1128`,
  `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/Array.h:956-967`.
- Dagua seed bridge, permutation, runner-style initialization, scalar loop, and
  component packing:
  `dagua/layout/ops/gem.py:272-286`, `:335-363`, `:415-453`, `:830-978`,
  `:1049-1075`.
- Dagua translates OGDF `numberOfRounds` into per-node update budget:
  `dagua/layout/ops/gem.py:1291-1296`.

Concrete behavioral deltas still plausible:
1. RNG/distribution exactness: Dagua has a custom `minstd_rand` and
   `uniform_int_distribution` clone. OGDF/libstdc++ distribution semantics
   should be traced at the first permutation and disturbance draws.
2. Force/geometry scale: Dagua's early bboxes are ~3x the fresh runner's by
   round 20 despite identical high-level constants. Trace first 1-5 node
   impulses (`desiredLength`, barycenter, repulsion, attraction, move) against
   instrumented OGDF.
3. Reference data artifact: old saved `seeded_refs` GEM positions are not the
   current rebuilt runner output for this probe. Do not use them for
   first-divergence conclusions.

## Recommendations

| cluster | disposition | effort |
| --- | --- | --- |
| Connected classical MDS 14 | floor-with-evidence | no port; cite eigengap + driver dossier |
| Small disconnected MDS 4 (`disconnected_label_cycle_collage`, `multi_component_80`) | aggregate/floor candidate | low; stress-only within-component residuals near margins |
| Large disconnected MDS 4 (`random_dag_50/200`) | aggregate-tier or metric-tier, not DLA port-fix | medium; crossings are inside one large component, stress pairs exclude cross-component pairs |
| Connected GEM 4 | port-fix candidate | medium; early divergence, trace RNG/impulse scale |
| Disconnected GEM 3 | aggregate-tier first, then port-fix only if population tier rejected | medium-high; rows are broad stochastic/noncanonical tails |

No dead code identified.
