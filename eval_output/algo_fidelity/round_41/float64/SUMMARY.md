# Round 41 Float64 Fidelity Summary

This pass added the `fidelity_dtype` plumbing needed to run fidelity-mode
pipelines with double-precision internal tensors while keeping public returns
as `torch.float32`. The smoke harnesses below were run from the current
worktree after the wiring pass. Most existing harnesses measure the current
fidelity path rather than exposing an explicit float32-vs-float64 switch, so
rows marked `n/a` did not have a comparable before value in the harness.

| Engine | RMSD before (float32) | RMSD after (float64) | Improvement |
|---|---:|---:|---:|
| classical_mds | 0.091111468 | 0.000000001 | 0.091111467 |
| dagua_native | n/a | 0.000000001 | n/a |
| davidson_harel | 0.374099000 | 0.000000000 | 0.374099000 |
| drl | n/a | 0.000000036 | n/a |
| fa2 | n/a | 0.000771710 | n/a |
| fmmm/fdp kernels | n/a | 0.226572489 | n/a |
| gem | n/a | 0.003333469 | n/a |
| graphopt | n/a | 0.000000007 | n/a |
| lgl | n/a | 0.171220750 | n/a |
| maxent_stress | n/a | 0.000000022 | n/a |
| neato | n/a | 0.000689496 | n/a |
| pivot_mds | 0.097790357 | 0.000000022 | 0.097790335 |
| reingold_tilford | 0.085457115 | 0.000000000 | 0.085457115 |
| sfdp | n/a | 0.049772587 | n/a |
| stress_sgd | 0.201599339 | 0.000000000 | 0.201599339 |
| tsnet | 0.300040925 | 0.000000000 | 0.300040925 |

## Harness Notes

- `tsnet` is the clearest float64 win in this pass: after adding dtype support
  to `TsnetInitializePositionsConfig`, the round 41 smoke check reports
  `before_mean=0.300040925` and `after_mean=0.000000000`.
- `classical_mds`, `davidson_harel`, `pivot_mds`, `reingold_tilford`, and
  `stress_sgd` already had round 41 harnesses with before/after rows and are
  now at or near zero residual.
- `dagua_native`, `graphopt`, and `maxent_stress` are effectively numerical
  floor-limited in the smoke harnesses, with residuals around `1e-9` to `1e-8`.
- `fa2` still has a Barnes-Hut residual (`overall max=0.008407409`), which is
  algorithmic rather than plain dtype drift; the exact non-Barnes-Hut rows are
  zero within numerical precision.
- `gem` and `lgl` still show non-zero residuals in their smoke harnesses. These
  are likely algorithmic-order or reference-behavior gaps, not solved by simply
  forwarding `fidelity_dtype`.
- The old round 37 integration smoke failed before exercising layouts because
  `classic_fmmm_graphviz_fdp_fidelity` was missing from its variant registry.
- The round 40 fdp-cluster smoke failed to import
  `_fdp_recursion_tlayout_component` from the current `fmmm` pipeline. That is
  a harness/API drift issue separate from the dtype plumbing.

## Fidelity Branch Inventory

Branches with explicit `fidelity_mode` conditionals or fidelity-only selectors
were found at:

- `dagua_native.py`: 195, 216, 240, 258, 4549, 4736, 4746.
- `davidson_harel.py`: 272.
- `drl.py`: 215.
- `fa2.py`: 330, 345, 445.
- `fmmm.py`: 2943, 3176.
- `fr.py`: 187, 189, 191, 548.
- `kk.py`: 500, 511.
- `linlog.py`: 165.
- `neato.py`: 817, 821.
- `neulay.py`: 100, 107, 113, 119, 244, 278, 281.
- `reingold_tilford.py`: 188.
- `sfdp.py`: 379, 381.
- `sgd2_multi.py`: 405.
- `spectral.py`: 49, 51.
- `stress_majorization.py`: 924, 927, 945, 979, 1075, 1082.
- `stress_sgd.py`: 292, 400.
- `sugiyama.py`: 98, 111, 212.
- `tsnet.py`: 169, 257.
- `umap_layout.py`: 295.
