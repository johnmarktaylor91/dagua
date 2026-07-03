# r76-D5 Formal Floor Dossiers

Date: 2026-07-03
Scope: research/probe only. No package code changes. Scratch artifacts: `/tmp/r76_floor_probe.json`, `/tmp/r76_mds_eigenspace.json`, `/tmp/r76_mds_eigenspace2.json`.

## Cluster 1: MDS connected, igraph_mds reference

Disposition text: proven member of reference equivalence class. Cause: machine-precision ties in the double-centered MDS Gram matrix make the chosen output basis arbitrary. Dagua and igraph both select valid coordinates from the same selected eigenspaces; when the tied eigenspace dimension exceeds two, the two visible 2D drawings are not required to be congruent in the output plane.

Assumption: I interpreted "orthogonal transform within the degenerate eigenspace" as a transform in the Gram-matrix eigenspace, not only a 2D Procrustes rotation. For multiplicity greater than two, arbitrary 2D selections inside the tied eigenspace can both be reference-equivalent optima without a 2D orthogonal map between the drawings.

### Eigengap table

| graph | selected/tied eigenvalues | gap lambda2-lambda3 abs | rel | ULP | selected eigenspace dim |
| --- | --- | ---: | ---: | ---: | ---: |
| `bipartite_4_3_4` | 2, 2, 2, 2, 2 | 4.441e-16 | 2.220e-16 | 1 | 9 |
| `center_port_backedge_hub` | 2, 2, 2, 2, -6.037e-16 | 6.661e-16 | 3.331e-16 | 1.5 | 4 |
| `densenet_block` | 3.34347, 0.5, 0.5, 0.5, 0.5 | 5.551e-17 | 5.551e-17 | 0.25 | 6 |
| `org_chart_1_5_4_8` | 44.4499, 44.4499, 44.4499, 2.79402, 2 | 0 | 0 | 0 | 3 |
| `petersen_10` | 3.5, 3.5, 3.5, 3.5, 3.5 | 8.882e-16 | 2.538e-16 | 2 | 5 |
| `wide_3_50_3` | 2, 2, 2, 2, 2 | 4.441e-16 | 2.220e-16 | 1 | 54 |
| `wide_single_layer_1_50_1` | 2, 2, 2, 2, 2 | 1.332e-15 | 6.661e-16 | 3 | 50 |

All seven graphs have the second/third selected eigenvalue gap at 0 to 3 ULPs. `densenet_block` has a distinct leading eigenvalue, but its second coordinate is drawn from a five-way tied 0.5 eigenspace.

### Eigenspace membership proof

Coordinates were divided by igraph scale 50 before projection. Residuals are RMSD in raw MDS coordinate units after centering and projecting each layout into the union of eigenspaces containing the top two selected eigenvalues.

| graph | selected eigenspace indices | D projection RMSD | R projection RMSD | D rel | R rel |
| --- | --- | ---: | ---: | ---: | ---: |
| `bipartite_4_3_4` | 1..9 | 3.185e-09 | 7.512e-10 | 7.471e-09 | 1.762e-09 |
| `center_port_backedge_hub` | 1,2,3,4 | 6.407e-09 | 6.553e-09 | 1.110e-08 | 1.135e-08 |
| `densenet_block` | 1,2,3,4,5,6 | 1.755e-09 | 1.812e-09 | 3.581e-09 | 3.698e-09 |
| `org_chart_1_5_4_8` | 1,2,3 | 2.394e-08 | 3.502e-08 | 1.523e-08 | 2.228e-08 |
| `petersen_10` | 1,2,3,4,5 | 6.997e-09 | 1.122e-08 | 1.183e-08 | 1.897e-08 |
| `wide_3_50_3` | 1..54 | 0 | 3.813e-10 | 0 | 2.018e-09 |
| `wide_single_layer_1_50_1` | 1..50 | 1.038e-10 | 1.340e-10 | 5.292e-10 | 6.832e-10 |

The largest relative projection residual is 2.23e-08. This is the formal evidence that both outputs live in the same selected eigenspace of the same double-centered distance matrix; the visible coordinate differences are basis selection inside a tied eigenspace, not a different algorithm.

### Quality parity from ledger

| combo | stress D | stress R | crossings D | crossings R | W D | W R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bipartite_4_3_4::classic_classical_mds_default` | 0.216615 | 0.267702 | 27 | 16 | 1.559e-16 | 2.117e-16 |
| `bipartite_4_3_4::classic_classical_mds_igraph_fidelity` | 0.234401 | 0.267702 | 29 | 16 | 1.755e-16 | 2.117e-16 |
| `center_port_backedge_hub::classic_classical_mds_default` | 0.201607 | 0.345503 | 0 | 2 | 2.297e-16 | 1.606e-16 |
| `center_port_backedge_hub::classic_classical_mds_igraph_fidelity` | 0.186898 | 0.345503 | 3 | 2 | 1.788e-16 | 1.606e-16 |
| `densenet_block::classic_classical_mds_default` | 0.127624 | 0.148737 | 30 | 30 | 1.034e-25 | 0 |
| `densenet_block::classic_classical_mds_igraph_fidelity` | 0.129108 | 0.148737 | 30 | 30 | 0 | 0 |
| `org_chart_1_5_4_8::classic_classical_mds_default` | 0.107821 | 0.11066 | 0 | 0 | 2.010e-16 | 4.429e-16 |
| `org_chart_1_5_4_8::classic_classical_mds_igraph_fidelity` | 0.104775 | 0.11066 | 0 | 0 | 1.407e-16 | 4.429e-16 |
| `petersen_10::classic_classical_mds_default` | 0.123513 | 0.160484 | 5 | 5 | 2.157e-16 | 2.602e-18 |
| `petersen_10::classic_classical_mds_igraph_fidelity` | 0.173645 | 0.160484 | 10 | 5 | 3.364e-17 | 2.602e-18 |
| `wide_single_layer_1_50_1::classic_classical_mds_default` | 0.452996 | 0.674931 | 0 | 5 | 3.685e-16 | 9.903e-17 |
| `wide_3_50_3::classic_classical_mds_default` | 0.459134 | 0.438807 | 6.887e+03 | 6.001e+03 | 1.320e-16 | 2.352e-16 |
| `wide_single_layer_1_50_1::classic_classical_mds_igraph_fidelity` | 0.599874 | 0.674931 | 0 | 5 | 1.544e-16 | 9.903e-17 |
| `wide_3_50_3::classic_classical_mds_igraph_fidelity` | 1 | 0.438807 | 0 | 6.001e+03 | 0 | 2.352e-16 |

The raw stress/crossing metrics move with the arbitrary selected basis, including cases where D is lower and cases where R is lower. The MDS objective evidence above shows these are alternate members of the same degenerate reference solution family; no port change can make the basis canonical without vendoring or otherwise constraining the eigensolver, which JMT ruled out.

## Cluster 2: UMAP disconnected spectral, random_dag_50/200 classic_umap_nn5

Disposition text: evidenced FP-chaos floor (eigenspace basis selection), quality parity shown. Cause: disconnected fuzzy graph spectral initialization contains near-degenerate component eigenspaces; a one-ULP coordinate perturbation at spectral init is amplified by SGD to the same order as the Dagua-vs-reference final-layout divergence.

### Normalized Laplacian eigengap table

| graph | component size | smallest eigenvalues | selected upper gap abs | rel | ULP |
| --- | ---: | --- | ---: | ---: | ---: |
| `random_dag_200` | 202 | -6.661e-16, 0.568889, 1, 1, 1 | 1.443e-15 | 1.443e-15 | 6.5 |
| `random_dag_200` | 181 | 2.519e-16, 0.0569738, 0.077089, 0.0890274, 0.104378 | 0.0119384 | 0.0119384 | 5.377e+13 |
| `random_dag_50` | 52 | -5.274e-16, 0.567928, 1, 1, 1 | 1.110e-16 | 1.110e-16 | 0.5 |
| `random_dag_50` | 45 | 7.552e-16, 0.0660253, 0.13158, 0.177728, 0.226267 | 0.0461477 | 0.0461477 | 2.078e+14 |

The singleton/meta components produce exact or near-exact eigenvalue piles at 1.0. For `random_dag_50`, the 52-node component has lambda2-lambda3 gap 1.11e-16 (0.5 ULP). For `random_dag_200`, the 202-node component has gap 1.44e-15 (6.5 ULP).

### 1-ULP perturbation experiment

Perturbation: run stock Dagua UMAP, then rerun with one float32 spectral-init coordinate advanced by `torch.nextafter(x, +inf)` immediately after spectral initialization/rescale, preserving the rest of the pipeline and RNG state. RMSD uses scale-invariant orthogonal Procrustes on final layouts.

| graph | seed | stock vs 1-ULP RMSD | Dagua vs reference RMSD |
| --- | ---: | ---: | ---: |
| `random_dag_200` | 100 | 0.909822 | 1.03878 |
| `random_dag_200` | 101 | 0.900577 | 1.14197 |
| `random_dag_200` | 102 | 1.52628 | 0.877655 |
| `random_dag_50` | 100 | 0.695339 | 1.06946 |
| `random_dag_50` | 101 | 1.39524 | 1.02776 |
| `random_dag_50` | 102 | 0.769976 | 1.08552 |

`random_dag_200` means: perturbation 1.11222; Dagua-vs-reference 1.01947.

`random_dag_50` means: perturbation 0.95352; Dagua-vs-reference 1.06092.

The perturbation divergence is comparable to, and sometimes larger than, Dagua-vs-reference divergence. This supplies the missing chaos-amplification evidence for the already-bisected cause: fuzzy graph, schedule, curve, and RNG state matched; the remaining first difference is spectral basis selection in a near-degenerate component eigenspace.

### Quality parity from stage-1b

| combo | stress D | stress R | crossings D | crossings R | W D | W R | TOST context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `random_dag_50::classic_umap_nn5` | 0.648754 | 0.740728 | 386.133 | 397.9 | 0.179189 | 0.179346 | stress p=1, cross p=5.609e-05 |
| `random_dag_200::classic_umap_nn5` | 0.583337 | 0.543882 | 9.402e+03 | 8.275e+03 | 0.183396 | 0.172828 | stress p=0.937271, cross p=1 |

Stage-1b W means are essentially tied on `_50` (0.17919 vs 0.17935) and close on `_200` (0.18340 vs 0.17283). TOST is mixed by metric leg, but the aggregate quality is not a lesser-quality disposition; it is a known-cause floating-point chaos floor.

## Commands used

```bash
python - <<'PY' > /tmp/r76_floor_probe.json
# computed MDS Gram eigengaps, UMAP fuzzy-graph Laplacian gaps,
# MDS/UMAP reference comparisons, and UMAP 1-ULP perturbation rows
PY
python - <<'PY' > /tmp/r76_mds_eigenspace2.json
# projected Dagua and igraph MDS coordinates into selected Gram eigenspaces
PY
python - <<'PY' > /tmp/r76_dossier.md
# assembled this markdown dossier from the scratch JSON artifacts
PY
```

No tests were run because this task was research/probe only and made no package code changes.
