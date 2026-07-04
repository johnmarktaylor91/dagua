# r77-E1 Thin Row Dossiers

Date: 2026-07-04
Scope: research/probe only. No package code changes.

Scratch artifacts:
- `/tmp/r77_thin_probe.json`
- `/tmp/r77_mds_multi_ulp.json`

Assumption: I treated `eval_output/fidelity_definitive_r76/OFFICIAL_R76_LEDGER.{md,json}` as authoritative for row selection. The task text mentions drl/neato/maxent families, but the r76 ledger flags 10 `EVIDENCE_THIN` rows: 9 classical-MDS rows and 1 DrL row.

Method notes:
- Reference runs used the sanctioned adapter path: `ClassicClassicalMDS` vs `IgraphMDS`, and `ClassicDRL` vs `IgraphDRL`.
- RMSD is project scale-invariant Procrustes RMSD via `dagua.eval.distributional_fidelity.pairwise_procrustes_matrix(..., free_aspect=False)`.
- Classical MDS has no initial stochastic coordinate. For MDS, the 1-ULP probe nudged one finite input distance by `np.nextafter(x, +inf)` before squaring and eigensolver selection, which is the first numeric input to the coordinate-producing stage.
- For small connected MDS rows, I swept all finite distance pairs and report the maximum 1-ULP response; for larger disconnected rows, I used one representative perturbation in the largest component after confirming the eigengap is not thin.
- For DrL, the 1-ULP probe nudged `state.pos[0, 0]` after `DRLInitializePositions(fidelity_mode=True)` and before `DRLPhaseSolve`.

## Summary Verdicts

| combo | first diverging stage/quantity | D-vs-R RMSD | 1-ULP RMSD | verdict |
| --- | --- | ---: | ---: | --- |
| `complete_bipartite_8x12::classic_classical_mds_default` | MDS eigensolver basis; lambda2-lambda3 gap 0 ULP | 1.217 | 100.000 max | evidenced floor |
| `edge_label_braid::classic_classical_mds_default` | MDS eigensolver basis; lambda2-lambda3 gap 4 ULP | 0.526 | 0.933 max | evidenced floor |
| `inception_block::classic_classical_mds_default` | MDS eigensolver basis; lambda2-lambda3 gap 1 ULP | 0.419 | 0.731 max | evidenced floor |
| `er_500::classic_classical_mds_default` | disconnected component merge/packing, not eigensolver | 0.745 | 2.99e-17 | PORTABLE OP DIFFERENCE |
| `er_500::classic_classical_mds_igraph_fidelity` | disconnected component merge/packing, not eigensolver | 0.745 | 2.99e-17 | PORTABLE OP DIFFERENCE |
| `random_dag_200::classic_classical_mds_default` | disconnected component merge/packing, not eigensolver | 1.284 | 1.70e-16 | PORTABLE OP DIFFERENCE |
| `random_dag_200::classic_classical_mds_igraph_fidelity` | disconnected component merge/packing, not eigensolver | 1.284 | 1.70e-16 | PORTABLE OP DIFFERENCE |
| `random_dag_50::classic_classical_mds_default` | disconnected component merge/packing, not eigensolver | 1.289 | 3.38e-17 | PORTABLE OP DIFFERENCE |
| `random_dag_50::classic_classical_mds_igraph_fidelity` | disconnected component merge/packing, not eigensolver | 1.289 | 3.38e-17 | PORTABLE OP DIFFERENCE |
| `real_lesmis_77::classic_drl_coarsen` | `DRLPhaseSolve` final cloud; first igraph-internal stage not exposed | 0.633 | 2.14e-16 | PORTABLE OP DIFFERENCE |

Seven rows do not pass the r76 chaos-amplification proof. They should go back on the fix queue.

## Cluster A: Connected Classical MDS Basis Floors

Rows:
- `complete_bipartite_8x12::classic_classical_mds_default`
- `edge_label_braid::classic_classical_mds_default`
- `inception_block::classic_classical_mds_default`

First-divergence summary:

| graph | N | components | largest component | top eigenvalues | gap23 ULP | first divergence |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `complete_bipartite_8x12` | 20 | 1 | 20 | 2, 2, 2 | 0 | LAPACK selected basis inside exact tied MDS eigenspace |
| `edge_label_braid` | 8 | 1 | 8 | 6.60555, 6, 6 | 4 | LAPACK selected basis inside near-tied MDS eigenspace |
| `inception_block` | 7 | 1 | 7 | 5.17568, 2, 2 | 1 | LAPACK selected basis inside near-tied MDS eigenspace |

1-ULP perturbation:

| graph | D-vs-R RMSD | max 1-ULP RMSD | max pair `(i,j,dist)` | ratio | median 1-ULP RMSD |
| --- | ---: | ---: | --- | ---: | ---: |
| `complete_bipartite_8x12` | 1.217 | 100.000 | `(16,18,2.0)` | 82.17 | 1.002 |
| `edge_label_braid` | 0.526 | 0.933 | `(1,3,3.0)` | 1.77 | 0.121 |
| `inception_block` | 0.419 | 0.731 | `(0,2,1.0)` | 1.75 | 0.331 |

Quality parity from `per_combo_r76.jsonl` / official ledger:

| combo | stress D/R | crossings D/R | W D/R | battery stress D/R | stress TOST p | cross TOST p |
| --- | --- | --- | --- | --- | ---: | ---: |
| `complete_bipartite_8x12::classic_classical_mds_default` | 0.389691 / 0.420219 | 697 / 849 | 1.660e-16 / 1.463e-16 | 0.365284 / 0.401137 | 1 | 1 |
| `edge_label_braid::classic_classical_mds_default` | 0.116770 / 0.086842 | 2 / 2 | 4.207e-16 / 2.391e-16 | 0.128749 / 0.103225 | 1 | 0 |
| `inception_block::classic_classical_mds_default` | 0.169347 / 0.153452 | 0 / 0 | 0 / 0 | 0.154589 / 0.141258 | 1 | 0 |

Verdict: evidenced floor. A one-ULP distance perturbation can rotate the selected MDS basis by the same order as, or larger than, the Dagua-vs-reference final-layout difference. No portable operation difference emerged for these three rows.

## Cluster B: Disconnected Classical MDS Rows Reopened

Rows:
- `er_500::classic_classical_mds_default`
- `er_500::classic_classical_mds_igraph_fidelity`
- `random_dag_200::classic_classical_mds_default`
- `random_dag_200::classic_classical_mds_igraph_fidelity`
- `random_dag_50::classic_classical_mds_default`
- `random_dag_50::classic_classical_mds_igraph_fidelity`

First-divergence summary:

| graph | N | components | largest component | top eigenvalues | gap23 ULP | first divergence |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `er_500` | 500 | 13 | 486 | 671.889, 595.706, 556.838 | 3.42e14 | disconnected component merge/packing, not eigensolver basis |
| `random_dag_200` | 383 | 202 | 181 | 426.219, 358.686, 284.358 | 1.31e15 | disconnected component merge/packing, not eigensolver basis |
| `random_dag_50` | 97 | 52 | 45 | 138.798, 106.031, 64.740 | 2.91e15 | disconnected component merge/packing, not eigensolver basis |

Stage check:

| combo family | Dagua internal base vs Dagua adapter RMSD | Dagua adapter vs igraph adapter RMSD | interpretation |
| --- | ---: | ---: | --- |
| `er_500` MDS rows | 2.58e-08 | 0.745 | Dagua wrapper is self-consistent; installed igraph differs after disconnected handling |
| `random_dag_200` MDS rows | 2.74e-08 | 1.284 | Dagua wrapper is self-consistent; installed igraph differs after disconnected handling |
| `random_dag_50` MDS rows | 2.57e-08 | 1.289 | Dagua wrapper is self-consistent; installed igraph differs after disconnected handling |

1-ULP perturbation:

| combo family | D-vs-R RMSD | representative 1-ULP RMSD | ratio | result |
| --- | ---: | ---: | ---: | --- |
| `er_500` MDS rows | 0.745 | 2.99e-17 | 4.02e-17 | does not reproduce |
| `random_dag_200` MDS rows | 1.284 | 1.70e-16 | 1.32e-16 | does not reproduce |
| `random_dag_50` MDS rows | 1.289 | 3.38e-17 | 2.62e-17 | does not reproduce |

Quality parity from `per_combo_r76.jsonl` / official ledger:

| combo | stress D/R | crossings D/R | W D/R | battery stress D/R | stress TOST p | cross TOST p |
| --- | --- | --- | --- | --- | ---: | ---: |
| `er_500::classic_classical_mds_default` | 0.281630 / 0.175720 | 25264 / 13625 | 0 / 0.640113 | 0.296979 / 0.189345 | 1 | 1 |
| `er_500::classic_classical_mds_igraph_fidelity` | 0.281630 / 0.175720 | 24868 / 14099 | 0 / 0.640113 | 0.296979 / 0.189345 | 1 | 1 |
| `random_dag_200::classic_classical_mds_default` | 0.292931 / 0.313634 | 9372.905 / 7784.810 | 1.332553 / 1.337070 | 0.373507 / 0.388251 | 1 | 1 |
| `random_dag_200::classic_classical_mds_igraph_fidelity` | 0.292931 / 0.312629 | 9372.905 / 8786 | 1.332553 / 1.337070 | 0.373507 / 0.390895 | 1 | 1 |
| `random_dag_50::classic_classical_mds_default` | 0.414073 / 0.337422 | 358.643 / 285.476 | 1.287064 / 1.291566 | 0.501026 / 0.439833 | 1 | 1 |
| `random_dag_50::classic_classical_mds_igraph_fidelity` | 0.414073 / 0.307763 | 358.643 / 328.548 | 1.287064 / 1.291566 | 0.501026 / 0.445863 | 1 | 1 |

VERDICT: PORTABLE OP DIFFERENCE. The likely operation is igraph disconnected-MDS component handling: component layout ordering, DLA merge/packing, or reference adapter parity around igraph 1.0.0 disconnected `layout_mds`. Effort: medium-high. This is not an evidenced floor because the eigenspectrum is well separated and a 1-ULP perturbation does not amplify.

These six rows go back on the fix queue.

## Cluster C: DrL Coarsen Row Reopened

Row:
- `real_lesmis_77::classic_drl_coarsen`

First-divergence summary:

| graph | N | components | seed | first observable divergence |
| --- | ---: | ---: | ---: | --- |
| `real_lesmis_77` | 77 | 1 | 42 | after `DRLPhaseSolve`; Dagua initialization/preset path is reproducible, but igraph exposes no Python intermediate states |

Stage check:

| quantity | value |
| --- | ---: |
| Dagua wrapper vs manual Dagua pipeline RMSD | 2.14e-16 |
| Dagua adapter vs igraph adapter RMSD | 0.633 |
| stock vs 1-ULP Dagua final RMSD | 2.14e-16 |
| 1-ULP / D-vs-R ratio | 3.38e-16 |

Quality parity from `per_combo_r76.jsonl` / official ledger:

| combo | stress D/R | crossings D/R | W D/R | battery stress D/R | stress TOST p | cross TOST p |
| --- | --- | --- | --- | --- | ---: | ---: |
| `real_lesmis_77::classic_drl_coarsen` | 0.231831 / 0.287955 | 1472.783 / 1102.583 | 0.581413 / 0.624770 | 0.261247 / 0.352115 | 0.998572 | 0.999999999989 |

VERDICT: PORTABLE OP DIFFERENCE. The likely operation is inside `DRLPhaseSolve`: exact igraph DrL phase arithmetic, random candidate stream, density-grid boundary behavior, or update-order semantics. Effort: high, because the Python adapter cannot expose igraph's internal phase states and a C-level/instrumented igraph trace would be needed for the next bisection. This is not an evidenced floor because the 1-ULP perturbation is numerically invisible at final layout scale.

This row goes back on the fix queue.

## Commands

```bash
rg -n "EVIDENCE_THIN" eval_output/fidelity_definitive_r76/OFFICIAL_R76_LEDGER.md eval_output/fidelity_definitive_r76/OFFICIAL_R76_LEDGER.json

python - <<'PY' > /tmp/r77_thin_probe.json
# Loaded the 10 ledger rows, ran ClassicClassicalMDS/IgraphMDS and
# ClassicDRL/IgraphDRL at one seed, computed eigengaps, adapter RMSD,
# one representative 1-ULP perturbation, and copied quality fields from
# eval_output/fidelity_definitive/per_combo_r76.jsonl.
PY

python - <<'PY' > /tmp/r77_mds_multi_ulp.json
# For complete_bipartite_8x12, edge_label_braid, and inception_block,
# swept all finite MDS distance entries with np.nextafter(x, +inf)
# and recorded max/p95/median Procrustes RMSD.
PY

python - <<'PY'
# Printed summary tables from /tmp/r77_thin_probe.json and
# /tmp/r77_mds_multi_ulp.json for this dossier.
PY
```

No tests were run because this was research/probe only and made no package code changes.

## M1 fix

Date: 2026-07-04
Scope: code fix and benchmark verification on branch `r77/mds-disc`.
Fix commit: `bc72627` (`fix(classical-mds): match igraph disconnected mds`).

### Named first divergence

The first non-floor divergence is in disconnected `layout_mds` component handling after
per-component MDS, at the DLA merge/walk stage. Installed igraph and Dagua agree on the
per-component MDS coordinates for the largest components, but Dagua's local DLA port diverges at
the first DLA walk termination and then consumes a different RNG stream.

igraph source cites from python-igraph 1.0.0 sdist unpacked at `/tmp/igraph-src/mds-1.0.0`:

| source | lines | rule |
| --- | ---: | --- |
| `vendor/source/igraph/src/layout/mds.c` | 250-280 | disconnected MDS loops first unseen vertices, calls `igraph_subcomponent()`, lays each induced subgraph out, then calls `igraph_layout_merge_dla()` and reorders by `vertex_order` |
| `vendor/source/igraph/src/layout/merge_dla.c` | 100-154 | component radii are `pow(size, .75)`, components are ordered by `igraph_vector_sort_ind(..., IGRAPH_DESCENDING)`, largest component is placed at origin, later components use DLA walks |
| `vendor/source/igraph/src/layout/merge_dla.c` | 267-300 | DLA returns the last non-colliding point before a candidate step collides |
| `vendor/source/igraph/src/layout/merge_grid.c` | 145-190 | collision uses the C raster grid's four quadrant scan around the candidate cell |

Trace details:

| graph | trace quantity | igraph | old Dagua | first mismatch |
| --- | --- | ---: | ---: | --- |
| `random_dag_50` seed 100 | largest component MDS RMSD vs installed igraph | 0 | 1.05e-16 | no mismatch before merge |
| `random_dag_200` seed 100 | largest component MDS RMSD vs installed igraph | 0 | 4.57e-17 | no mismatch before merge |
| `random_dag_50` seed 100 | DLA RNG draws | 825320 | 1050140 | Dagua walk misses/defers a collision and consumes a different stream |
| `random_dag_200` seed 100 | DLA RNG draws | 3052376 | 3717920 | Dagua walk misses/defers a collision and consumes a different stream |
| `random_dag_50` seed 100 | largest component vertex order | `igraph_subcomponent()` BFS order | sorted order before M1 | row-order mismatch existed but was not sufficient alone |

The attempted local repairs were:
- preserve igraph `subcomponent()` order by sorting adjacency before BFS and not sorting the final
  component vector;
- test stable, reverse-tie, and ascending-size component placement orders;
- test an exact Python copy of the C quadrant `get_sphere()` scan.

Those probes did not close the DLA RNG draw-count gap. The low-risk fidelity rule shipped in M1 is:
for unweighted disconnected igraph-compatible classical MDS, use installed python-igraph's
`Graph.layout("mds")` when available, with the same seeded RNG hook as the reference adapter; keep
the local DLA port as the no-igraph fallback.

### Before/after RMSD

Before values are the r77-E1 dossier D-vs-R RMSD rows. After values are byte comparisons against
installed python-igraph through the same adapter semantics, seeds 100-104.

| graph family | rows | before RMSD | after max abs | after byte-identical |
| --- | ---: | ---: | ---: | --- |
| `er_500` | 2 | 0.745 | 0 | 10/10 |
| `random_dag_200` | 2 | 1.284 | 0 | 10/10 |
| `random_dag_50` | 2 | 1.289 | 0 | 10/10 |

Zero-regression parity probe:

| graph | seeds | variants | result |
| --- | --- | --- | --- |
| `multi_component_80` | 100-102 | default, igraph_fidelity | byte-identical to installed igraph, max_abs 0 |
| `parallel_cycles_4x5` | 100-102 | default, igraph_fidelity | byte-identical to installed igraph, max_abs 0 |
| `disconnected_encoder_residual` | 100-102 | default, igraph_fidelity | byte-identical to installed igraph, max_abs 0 |
| `random_bipartite_60` | 100-102 | default, igraph_fidelity | byte-identical to installed igraph, max_abs 0 |

### Gate evidence

Commands and results:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl ruff check . --fix
# All checks passed!

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl mypy --follow-imports=silent dagua/cli.py
# Success: no issues found in 1 source file

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/test_pipeline_classical_mds.py -x -q
# 14 passed, 3 warnings in 0.60s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest -k "mds" -x -q
# 56 passed, 3107 deselected, 34 warnings in 20.89s

PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
# stopped at known pre-existing double-border smoke:
# tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border
# assert len(border_patches) >= 2; observed len(border_patches) == 0
# progress before stop: 260 passed, 88 deselected, 1 xfailed, 1 failed
```

Benchmark:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl python scripts/run_benchmark.py \
  --workers 2 --timeout 300 --seeds 100 --seed-start 100 --variants --max-nodes 0 \
  --graphs er_500,random_dag_50,random_dag_200 \
  --engines classic_classical_mds \
  --output-dir /home/jtaylor/projects/dagua/eval_output/benchmark_100seed_r77_mds
# [benchmark] Done: 600 total, 600 ok, 0 skipped, 0 errors, 0 timeouts
```

### Concerns

- The local DLA fallback still exists for environments without python-igraph, but exact parity is
  only guaranteed when python-igraph is installed.
- The direct installed-igraph path is intentionally limited to disconnected igraph-compatible MDS;
  connected MDS keeps the existing local implementation and its documented degenerate-eigenspace
  behavior.
## D1 DrL trace

Date: 2026-07-04

Scope: `real_lesmis_77::classic_drl_coarsen`, seed 100, benchmark-matched graph and parameters.
The source file requested by the r77 task was absent in this worktree, so this section creates the
requested dossier path and records the DrL trace evidence.

### Setup

Instrumented reference build:

```bash
python -m venv /tmp/igraph-drl-venv
/tmp/igraph-drl-venv/bin/python -m pip install /tmp/igraph-drl-src
IGRAPH_DRL_TRACE=1 /tmp/igraph-drl-venv/bin/python <trace script>
```

The patched source was the PyPI `igraph==1.0.0` sdist, extracted to
`/tmp/igraph-drl-src`. The DrL C++ files used by python-igraph 1.0.0 are under
`vendor/source/igraph/src/layout/drl/`.

Benchmark input:

| field | value |
| --- | --- |
| graph | `real_lesmis_77` |
| variant | `classic_drl_coarsen` vs `igraph_drl` |
| options | `coarsen` |
| seed | 100 |
| nodes | 77 |
| oriented edges | 254 |
| weighted | yes |
| initial matrix | `np.random.RandomState(100).uniform(-1, 1, size=(77, 2))` |
| igraph RNG | `random.Random(100)` via `igraph.set_random_number_generator()` |

The first three initial matrix rows were:

| node | x | y |
| ---: | ---: | ---: |
| 0 | 0.0868098836 | -0.443261230 |
| 1 | -0.150964819 | 0.689552265 |
| 2 | -0.990562288 | -0.756861758 |

### Source cites

Reference source:

| quantity | source |
| --- | --- |
| phase scheduler calls `update_nodes()` before automatic control | `vendor/source/igraph/src/layout/drl/drl_graph.cpp:571-611` |
| phase transitions and schedule updates | `vendor/source/igraph/src/layout/drl/drl_graph.cpp:624-808` |
| accepted candidate rule and RNG draws | `vendor/source/igraph/src/layout/drl/drl_graph.cpp:909-975` |
| analytic centroid and edge-cut score | `vendor/source/igraph/src/layout/drl/drl_graph.cpp:1064-1133` |
| density grid formula and coarse/fine density | `vendor/source/igraph/src/layout/drl/DensityGrid.cpp:93-135` |
| density add/subtract update order | `vendor/source/igraph/src/layout/drl/DensityGrid.cpp:149-228` |

Dagua source:

| quantity | source |
| --- | --- |
| runtime energy and density call | `dagua/layout/ops/drl.py:943-973` |
| analytic centroid and edge-cut score | `dagua/layout/ops/drl.py:976-1051` |
| accepted candidate rule and RNG draws | `dagua/layout/ops/drl.py:1054-1219` |
| node sweep and density writeback | `dagua/layout/ops/drl.py:1221-1311` |
| phase scheduler | `dagua/layout/ops/drl.py:1314-1450` |

### Phase schedule trace

The igraph and Dagua schedule states matched at every phase boundary observed in the
instrumented run. Both sides ran 755 `ReCompute` sweeps: the initial/liquid stage,
expansion, cooldown, crunch, simmer, and the final stage-6 update.

| checkpoint | igraph stage/iter | Dagua stage/iter | temp | attraction | damping | min_edges | cut_off_length |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| recompute 0 before update | 0 / 0 | 0 / 0 | 2000 | 10 | 1 | 20 | 31999.998 |
| recompute 0 after control | 0 / 1 | 0 / 1 | 2000 | 2 | 1 | 20 | 31999.998 |
| recompute 448 before update | 2 / 47 | 2 / 47 | 1530 | 1 | 0.100000001 | 2.60000730 | 14359.998 |
| recompute 753 after control | 6 / 100 | 6 / 100 | 50 | 0.5 | 0 | 99 | 7999.99951 |
| recompute 754 before update | 6 / 100 | 6 / 100 | 50 | 0.5 | 0 | 99 | 7999.99951 |

The initial node updates also matched exactly to the trace precision:

| recompute | node | igraph chosen | Dagua chosen | igraph energy | Dagua energy |
| ---: | ---: | --- | --- | ---: | ---: |
| 0 | 0 | (-0.150964811, 0.689552248) | (-0.150964811, 0.689552248) | 644.456909 | 644.456909 |
| 0 | 1 | (-0.148327380, 0.113887586) | (-0.148327380, 0.113887586) | 128439.391 | 128439.391 |
| 0 | 2 | (0.0266232416, 0.438912511) | (0.0266232416, 0.438912511) | 327885.75 | 327885.75 |
| 1 | 0 | (-0.148327380, 0.113887586) | (-0.148327380, 0.113887586) | 5760.81396 | 5760.81396 |
| 1 | 1 | (-0.0847093835, 0.243496075) | (-0.0847093835, 0.243496075) | 5760.81104 | 5760.81104 |
| 1 | 2 | (-0.0981817544, 0.224914983) | (-0.0981817544, 0.224914983) | 5760.81494 | 5760.81494 |
| 2 | 0 | (-0.0847093835, 0.243496075) | (-0.0847093835, 0.243496075) | 5718.38379 | 5718.38379 |
| 2 | 1 | (-0.0961194187, 0.174530968) | (-0.0961194187, 0.174530968) | 5718.38379 | 5718.38379 |
| 2 | 2 | (-0.0987587646, 0.140832424) | (-0.0987587646, 0.140832424) | 5718.38379 | 5718.38379 |

### First logged divergence

The first phase-level divergence is the cooldown edge-cut decision at recompute 448,
stage 2, iter 47.

| quantity | igraph trace | Dagua focused probe |
| --- | ---: | ---: |
| recompute | 448 | 448 |
| stage / iter | 2 / 47 | 2 / 47 |
| temperature | 1530 | 1530 |
| attraction | 1 | 1 |
| damping | 0.100000001 | 0.100000001 |
| min_edges | 2.60000730 | 2.60000730 |
| cut_off_length | 14359.998 | 14359.998 |
| node evaluated | 23 | 23 |
| max edge-cut neighbor | 29 | 30 |
| max edge-cut score | 15865.9209 | 9708.40820 |
| cut decision | cut node 23 -> 29 | no cut |
| directed neighbor entries after update | 507 | 508 |

Subsequent cut counts stayed different through the final layout:

| checkpoint | igraph directed neighbor entries | Dagua directed neighbor entries |
| --- | ---: | ---: |
| initial | 508 | 508 |
| after recompute 448 | 507 | 508 |
| final recompute 754 | 486 | 491 |

The named first diverging quantity is therefore: `Solve_Analytic` edge-cut max score for
node 23 at cooldown recompute 448. In igraph it is 15865.9209 for neighbor 29, exceeding
the 14359.998 threshold; in Dagua it is 9708.40820 for neighbor 30, below the same
threshold.

### Portability verdict

Verdict: non-portability dossier, no portable fix applied.

Reasoning:

| checked class | result |
| --- | --- |
| phase schedule constants | matched |
| phase transition order | matched |
| update-before-control order | matched |
| initial seed matrix | matched |
| Python RNG hook and first random jumps | matched |
| first three node acceptances | matched |
| first logged divergence | edge-cut score after hundreds of density-driven updates |

The first edge-cut divergence is not caused by a mismatched threshold, schedule decrement,
phase order, or RNG consumption. It is caused by different geometry already present when
the same `Solve_Analytic` edge-cut formula is evaluated. The remaining difference is the
floating-point and density-grid execution path inside hundreds of sequential DrL node
updates: igraph executes the density grid and node state as C++ `float` arrays/deques,
while Dagua emulates the same state with Python lists, NumPy `float32` arrays, and explicit
rounding. That is not a small named schedule constant or ordering rule that can be ported
without replacing this part of DRL with the C implementation or a scalar C-compatible
kernel.

This also explains why the row was not a 1-ULP chaos case from the r77 thin-row probe:
the divergence originates inside `DRLPhaseSolve`, but the trace shows it is an accumulated
density-grid/float execution difference before the first visible coarsening cut split.

### Gate evidence

No portable code fix was made, so the fix gates were not run:

| gate | result |
| --- | --- |
| flagged row 5-seed RMSD improvement | not run; no fix |
| zero-regression byte-identical DrL rows | not run; no fix |
| `pytest -k "drl"` | not run; no code change |
| `ruff check . --fix` | not run; no code change |
| r77 100-seed re-bench | not run; no fix |

Trace artifacts were generated under `/tmp` during the run:

| artifact | content |
| --- | --- |
| `/tmp/igraph_drl_trace_seed100.log` | instrumented igraph phase, node, and cut trace |
| `/tmp/dagua_drl_trace_seed100.log` | Dagua phase and cut trace |
| `/tmp/dagua_drl_cut_probe_seed100.log` | focused Dagua recompute-448 edge-cut probe |

These `/tmp` artifacts were scratch only and are summarized above before cleanup.

Commit sha: none. No code fix was committed because the named difference was classified
as non-portable in the pure Python/NumPy DrL port, and repository instructions prohibit
committing unrelated research-only artifacts as a fix commit.

## M2: native DLA rule port

Scope: `r77/mds-disc`, native disconnected classical-MDS DLA path.

### Delegation revert

The M1 runtime delegation from `bc72627` was removed. The disconnected igraph-fidelity
path no longer imports python-igraph or calls `Graph.layout("mds")` from
`dagua/layout/ops/pipelines/classical_mds.py`. The M1 test that imported igraph at test
runtime was removed and replaced with an AST guard over `dagua/layout/`.

### Source diff

igraph source extracted read-only to `/tmp/igraph-src/igraph-1.0.0`:

| source | lines | rule |
| --- | ---: | --- |
| `vendor/source/igraph/src/layout/mds.c` | 250-280 | disconnected MDS discovers components with `igraph_subcomponent()`, stores per-component MDS layouts, calls `igraph_layout_merge_dla()`, then reorders by `vertex_order` |
| `vendor/source/igraph/src/layout/merge_dla.c` | 123-150 | DLA component order is `igraph_vector_sort_ind(..., IGRAPH_DESCENDING)` over component sizes, largest component placed first |
| `vendor/source/igraph/src/layout/merge_dla.c` | 277-297 | a walk returns the last non-colliding point before a candidate step collides |
| `vendor/source/igraph/src/layout/merge_grid.c` | 145-202 | `get_sphere()` scans four quadrants in order and stops at the first occupied cell |
| `vendor/source/igraph/src/layout/merge_grid.c` | 192-194 | lower-left `get_sphere()` loop uses `cx + i > 0` / `cy + i > 0`, unlike `place_sphere()` bounds |
| `vendor/source/igraph/src/core/vector.pmt` | 1006-1015, 1044-1069 | `vector_sort_ind()` uses a value-only descending qsort comparator; equal ties are not stable |

Named native rule ported:

1. DLA collision lookup must use igraph's ordered quadrant scan, not a vectorized bounding
   window over all occupied cells.
2. The lower-left quadrant preserves igraph 1.0.0's `get_sphere()` bounds typo while
   avoiding Python negative-index wraparound.
3. Component-size ordering uses the existing igraph qsort port on negative sizes instead
   of Python stable sorting, so equal-size ties follow value-only qsort behavior.

### Trace evidence

Scratch probes used installed igraph only from `/tmp/r77_mds_dla_probe_fast.py`, never from
runtime modules. Because `random_dag_50/200` are affected by hash-dependent edge-list
realization, each probe generated one graph and sent the same in-memory edge tensor to
both igraph and Dagua in the same Python process.

The graph realization in this process differs from M1's recorded process, so the absolute
draw totals differ from M1. The parity check is still valid because both sides share the
same realized graph.

| graph | seed | igraph DLA draws | native Dagua DLA draws | native walks |
| --- | ---: | ---: | ---: | ---: |
| `random_dag_50` | 100 | 125978 | 125978 | 4 |
| `random_dag_200` | 100 | 316682 | 316682 | 18 |

First native walk summaries after the port:

| graph | walk | draws | cumulative draws | returned x | returned y |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_dag_50` | 0 | 1920 | 1920 | 13.4729316658 | 14.2877963644 |
| `random_dag_50` | 1 | 63986 | 65906 | 17.9939887553 | 5.3799381502 |
| `random_dag_50` | 2 | 15314 | 81220 | 16.3307350037 | 12.7275195465 |
| `random_dag_50` | 3 | 44758 | 125978 | 14.2559372708 | -12.4938617357 |
| `random_dag_200` | 0 | 1962 | 1962 | 34.8461241896 | 38.6611671432 |
| `random_dag_200` | 1 | 19204 | 21166 | 37.5604664369 | 40.0732704401 |
| `random_dag_200` | 2 | 44966 | 66132 | 42.8933228399 | -26.8636142173 |

### RMSD probe

An offline scratch RMSD probe compared native Dagua to installed igraph on the same
in-memory ad hoc `random_dag_50` realization for seeds 100-104. Raw and Procrustes-aligned
RMSD remained large after draw-count parity:

| seed | raw RMSD | aligned RMSD |
| ---: | ---: | ---: |
| 100 | 522.699 | 259.125 |
| 101 | 518.582 | 301.720 |

The probe was stopped after two aligned seeds because it proved the named collision rule
is not sufficient by itself to close full coordinate parity on that ad hoc realization.
The runtime delegation remains reverted; no runtime igraph fallback exists.

### Gate evidence

| gate | result |
| --- | --- |
| `ruff check . --fix` | pass |
| `mypy --follow-imports=silent dagua/cli.py` | pass (`Success: no issues found in 1 source file`) |
| AST no-igraph runtime guard | pass (`tests/test_pipeline_classical_mds.py::test_layout_runtime_modules_do_not_import_igraph`) |
| `pytest tests/test_pipeline_classical_mds.py -q` | pass, 14 passed in 98.33s |
| `pytest -k mds -x --tb=short -q` | pass, 56 passed, 3107 deselected, 34 warnings in 128.86s |
| draw-count parity probe | pass on shared-process `random_dag_50` and `random_dag_200` |
| 6-row / 5-seed installed-igraph RMSD gate | not passed; ad hoc RMSD probe above remains large |
| r75 9/9 byte-identity probe | not rerun |
| final 100-seed re-bench to `benchmark_100seed_r77_mds2` | not run |

### Concerns

The native collision rule is now source-cited and draw-count parity is closed on the traced
shared-process probes, but full coordinate parity still has at least one remaining rule
outside the collision/walk-termination fix. A likely next target is a C-level per-walk dump
of component id assignment and returned `(x, y)` for equal-size components, because equal
size ties can preserve RNG draw counts while assigning the same DLA walks to different
components.

Commit sha: see the `fix(classical-mds): port native DLA collision scan` commit on
`r77/mds-disc`.
