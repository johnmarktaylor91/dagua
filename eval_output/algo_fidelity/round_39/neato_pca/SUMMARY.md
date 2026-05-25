# Round 39 Neato PCA/CG Solver Port Summary

## Per-Component Diagnosis

The R36 `fidelity_mode="graphviz"` path diverged because it used PCA/smart
initialization for Graphviz's default neato reference. In Graphviz
`neatoinit.c`, normal major-mode neato defaults to `INIT_RANDOM`; the
PCA/smart-init branch in `stress.c` is reached only when `start=self`.

Round 39 smoke diagnostics show:

- PCA init vs Graphviz final is a large mismatch on the path graph
  (`~0.23` RMSD), confirming initialization was the main R36 regression.
- Source-derived `srand48`/`drand48` random init plus the packed-CG solver
  matches Graphviz before overlap at near-zero RMSD across all topologies.
- The VPSC overlap postprocess was a second divergence: Graphviz default neato
  does not run VPSC unless overlap adjustment requests it, while the R36 Dagua
  path applied overlap removal for every truthy fidelity mode.

## Port Fix Applied

- Added a Graphviz-compatible `drand48` generator and default random
  initializer in `dagua/layout/ops/pipelines/stress_majorization.py`.
- Changed the `fidelity_mode="graphviz"` solver path to use Graphviz's default
  random initialization while keeping the packed-CG majorization loop.
- Left the PCA helpers in place for diagnostics and future `start=self` work,
  but they are no longer used for default neato fidelity.
- Changed `dagua/layout/ops/pipelines/neato.py` so the packed-CG Graphviz path
  does not unconditionally run VPSC overlap removal.
- Switched `classic_neato_graphviz_fidelity` back to
  `fidelity_mode="graphviz"` in `dagua/eval/variants.py`.

## Before/After Smoke RMSD

| Topology | Before `graphviz` | After `graphviz` mean | Compatibility mean |
| --- | ---: | ---: | ---: |
| path | 0.442291663 | 0.001663854 | 0.031397689 |
| star | not recorded in R38 for `graphviz` | 0.000010555 | 0.384045885 |
| clustered | not recorded in R38 for `graphviz` | 0.001076041 | 0.024789197 |
| grid | not recorded | 0.000007533 | 0.215707243 |
| overall | not recorded | 0.000689496 | 0.163985003 |

Round 39 harness:

```text
path: graphviz_mean=0.001663854 compat_mean=0.031397689
star: graphviz_mean=0.000010555 compat_mean=0.384045885
clustered: graphviz_mean=0.001076041 compat_mean=0.024789197
grid: graphviz_mean=0.000007533 compat_mean=0.215707243
overall: graphviz_mean=0.000689496 compat_mean=0.163985003
```

## Final Variant Alias

`classic_neato_graphviz_fidelity` now uses `fidelity_mode="graphviz"`.

Justification: the fixed packed-CG path beats the `graphviz_neato`
compatibility path on the original path smoke and on the full Round 39
4-topology x 3-seed matrix.

## Residual Notes

The PCA/smart-init path itself remains a separate `start=self` fidelity surface.
The default Graphviz benchmark adapter does not exercise it, so wiring PCA into
default `fidelity_mode="graphviz"` was the wrong behavior for this variant.
