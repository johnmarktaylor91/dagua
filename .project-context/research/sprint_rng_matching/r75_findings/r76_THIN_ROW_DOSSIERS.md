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
