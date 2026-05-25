# Round 41 dagua_native self-reference fidelity

## Reference source lines identified

- `dagua/layout/engine.py:1055-1069`: `algorithm=None` is remapped to
  `algorithm="dagua_native"` for the default engine path.
- `dagua/layout/engine.py:1093-1127`: the adapter/reference path forwards
  `edge_index`, `num_nodes`, `node_sizes`, `seed`, config, clusters, and
  cluster parents into the selected pipeline.
- `dagua/layout/engine.py:1129-1138`: the default adapter path wraps the
  pipeline call with graph cycle preparation/restoration and applies final
  direction handling.
- `dagua/layout/ops/pipelines/dagua_native.py:4559-4882`: direct
  `layout_dagua_native_pipeline` entrypoint, seed resolution, tensor
  preparation, fidelity-mode preprocessing, config preparation, and
  `LayoutProblem` construction.

## Sub-component diagnosis

Smoke harness:

```bash
python eval_output/algo_fidelity/round_41/dagua_native/smoke_harness.py
```

The harness compares the direct dagua_native pipeline in `fidelity_mode="none"`
against the Dagua adapter reference (`dagua.layout(..., algorithm=None)`) on
path, star, clustered, and grid topologies for seeds 42, 43, and 44. It also
runs a fixed-seed direct repeat and targeted toggles for edge iteration order,
postprocess/force-polish, convergence config, initialization, and output
normalization.

Dominant candidate divergences when intentionally perturbed:

| Sub-component | Max RMSD |
|---|---:|
| force kernel / postprocess (`edge_equalize_polish=False`) | 0.072896897793 |
| edge iteration order (reversed edge file order) | 0.058068949729 |
| convergence config (`steps=0`, default auto-steps) | 0.000000002245 |
| initialization | 0.000000002245 |
| RNG fixed-seed repeat | 0.000000002245 |
| normalization/direct adapter parity | 0.000000002245 |

Interpretation: the current unperturbed adapter and direct pipeline already
share the same initialization, seed semantics, convergence settings, force /
postprocess path, and output normalization for these smoke cases. The only
large residuals are deliberately introduced by changing protected inputs or
config knobs.

## Port implementation summary

No production pipeline port was applied in this round. The self-reference
smoke is already below the bit-exact target by more than six orders of
magnitude. Changing `dagua_native.py` would risk disturbing concurrent R41
work and would not reduce the measured residual, which is already at the
Procrustes/SVD numerical floor.

Added:

- `eval_output/algo_fidelity/round_41/dagua_native/smoke_harness.py`
  - 426 lines
  - typed/dataclass smoke cases and result rows
  - 4 topology builders
  - direct-pipeline and adapter-reference runners
  - Procrustes RMSD and fixed-seed reproducibility reporting
  - sub-component perturbation diagnosis

## Before/after smoke RMSD table

Because the measured baseline was already at floor, before and after are the
same. Values are Procrustes RMSD after scale-normalized alignment.

| Topology | Seed | Before | After | Repeat RMSD |
|---|---:|---:|---:|---:|
| path | 42 | 0 | 0 | 0 |
| path | 43 | 0 | 0 | 0 |
| path | 44 | 0 | 0 | 0 |
| star | 42 | 0.000000000245 | 0.000000000245 | 0.000000000245 |
| star | 43 | 0.000000000245 | 0.000000000245 | 0.000000000245 |
| star | 44 | 0.000000000245 | 0.000000000245 | 0.000000000245 |
| clustered | 42 | 0 | 0 | 0 |
| clustered | 43 | 0 | 0 | 0 |
| clustered | 44 | 0 | 0 | 0 |
| grid | 42 | 0.000000002245 | 0.000000002245 | 0.000000002245 |
| grid | 43 | 0.000000002245 | 0.000000002245 | 0.000000002245 |
| grid | 44 | 0.000000002245 | 0.000000002245 | 0.000000002245 |

Overall mean reference RMSD: `0.000000000623`.

Raw max absolute coordinate delta for every unperturbed adapter/direct pair:
`0`.

## Final verdict

Bit-exact for practical coordinate output on the smoke: adapter and direct
pipeline coordinates are exactly equal before Procrustes (`max_abs_delta=0`).
The non-zero RMSD entries are the numerical floor of the scale-normalized
Procrustes calculation, not a layout residual.

Quantified floor: `6.23e-10` overall mean RMSD, with max smoke RMSD
`2.25e-9`, from floating-point SVD/alignment on already-identical coordinate
tensors.

## Test output

```text
topology,seed,pipeline,reference_rmsd,repeat_rmsd,max_abs_delta
path,42,tree,0,0,0
path,43,tree,0,0,0
path,44,tree,0,0,0
star,42,tree,2.45428039003e-10,2.45428039003e-10,0
star,43,tree,2.45428039003e-10,2.45428039003e-10,0
star,44,tree,2.45428039003e-10,2.45428039003e-10,0
clustered,42,clustered-layered_dag,0,0,0
clustered,43,clustered-layered_dag,0,0,0
clustered,44,clustered-layered_dag,0,0,0
grid,42,layered_dag,2.2454798021e-09,2.2454798021e-09,0
grid,43,layered_dag,2.2454798021e-09,2.2454798021e-09,0
grid,44,layered_dag,2.2454798021e-09,2.2454798021e-09,0
overall_mean_reference_rmsd,6.22726960275e-10
component,max_rmsd
convergence_steps_default,2.2454798021e-09
force_kernel_no_polish,0.0728968977928
initialization,2.2454798021e-09
iteration_order_reversed_edges,0.0580689497292
normalization_direct,2.2454798021e-09
rng_repeat,2.2454798021e-09
```

The clustered reference emits the existing warning:
`dagua.layout: cluster_aware=True is not yet supported for algorithm='dagua_native'; falling back to legacy flat placement.`
The warning does not affect parity because both paths still converge to
identical coordinates for the smoke graph.
