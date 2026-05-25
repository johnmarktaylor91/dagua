# Round 38 Residual Debug Summary

## Scope

Investigated the three non-bit-exact R37 Graphviz-fidelity variants:

- `classic_sfdp_graphviz_fidelity`
- `classic_fmmm_graphviz_fdp_fidelity`
- `classic_neato_graphviz_fidelity`

Read the requested R36 summaries for sfdp coarsening/sequential/quadtree,
fdp recursion/tilepack/ports, and neato solver/overlap. Also checked
`git show acb03e0` and `git show bf026eb` for neato fidelity aliases.

## Baseline Reproduction

Command:

```bash
python eval_output/algo_fidelity/round_37/integration/smoke_check.py
```

Output before fixes:

```text
classic_sugiyama_graphviz_fidelity: 0.000000000
classic_sfdp_graphviz_fidelity: 0.023935233
classic_fmmm_graphviz_fdp_fidelity: 0.210095198
classic_neato_graphviz_fidelity: 0.442291663
```

## Fixes Applied

- `dagua/eval/variants.py`
  - Changed `classic_neato_graphviz_fidelity` from
    `fidelity_mode="graphviz"` to `fidelity_mode="graphviz_neato"`.
  - Reason: `layout_neato_pipeline` accepts both aliases. The newer
    `"graphviz"` alias routes into the R36 PCA + packed-CG solver, but smoke
    RMSD is much worse than the established Graphviz-neato compatibility path.
- `dagua/layout/ops/pipelines/fmmm.py`
  - Routed `_fdp_recursion_component_offsets()` through the R36
    `_graphviz_tile_pack_offsets()` component packer.
  - Reason: the clustered fdp recursion path still used its old row packer,
    bypassing the fdp tilepack port.
- `tests/test_layout/test_fmmm_fdp_recursion.py`
  - Added regression coverage proving recursive fdp sibling components use the
    Graphviz tile packer.

## Updated R37 Smoke

Command:

```bash
python eval_output/algo_fidelity/round_37/integration/smoke_check.py
```

Output after fixes:

```text
classic_sugiyama_graphviz_fidelity: 0.000000000
classic_sfdp_graphviz_fidelity: 0.023935233
classic_fmmm_graphviz_fdp_fidelity: 0.244951112
classic_neato_graphviz_fidelity: 0.028738625
```

## Wider Smoke Matrix

Harness: 3 graphs (`path`, `star`, `clustered`) x 3 seeds (`1`, `2`, `3`).
Reference Graphviz engines were first run like the R37 smoke harness
(`seed=None`).

| Variant | Path RMSD | Star RMSD | Clustered RMSD | Overall mean | Max |
| --- | --- | --- | --- | ---: | ---: |
| `classic_sfdp_graphviz_fidelity` | 0.023935233, 0.022571208, 0.017429141 | 0.353847238, 0.296679793, 0.356800226 | 0.044169177, 0.000266634, 0.039216829 | 0.128323942 | 0.356800226 |
| `classic_fmmm_graphviz_fdp_fidelity` | 0.372943517, 0.366775396, 0.376313311 | 0.402858710, 0.311181778, 0.397770123 | 0.304277774, 0.176696923, 0.288676109 | 0.333054849 | 0.402858710 |
| `classic_neato_graphviz_fidelity` | 0.028738625, 0.029365738, 0.028982350 | 0.431445630, 0.307570138, 0.370611705 | 0.184954822, 0.179970560, 0.254280900 | 0.201768941 | 0.431445630 |

Seeded-reference check (`seed` also passed to Graphviz) did not remove the
floor:

| Variant | Overall mean | Max |
| --- | ---: | ---: |
| `classic_sfdp_graphviz_fidelity` | 0.145618453 | 0.427435441 |
| `classic_fmmm_graphviz_fdp_fidelity` | 0.319884444 | 0.402858710 |
| `classic_neato_graphviz_fidelity` | 0.222992917 | 0.433364991 |

## Per-Engine Triage

### SFDP

Hypothesis tested:

- Verified `fidelity_mode="graphviz"` is accepted by `sfdp.py`.
- Verified the graphviz alias selects all R36 SFDP ports in the pipeline:
  `sfdp_graphviz_matrix_coarsen_hierarchy`,
  `sfdp_graphviz_refine_coarsest`, and
  `sfdp_graphviz_prolongate_and_refine`.
- Compared `fidelity_mode="graphviz"`, `True`, and `False` on the path smoke.
  `True` and `"graphviz"` matched exactly; default was nearly identical.

Fix applied:

- None.

Residual diagnosis:

- Original R37 path smoke remains low at `0.023935233`.
- Wider topology coverage shows a star-graph residual around `0.30-0.36`.
- R36 already documents remaining SFDP floors from random initialization,
  output normalization, unmatched-node permutation using torch rather than
  Graphviz `gv_random`, and other non-ported integration details.

Verdict:

- `numerical floor` for the original R37 smoke.
- `needs more work` for wider topology parity.
- Ship recommendation: ship only if the acceptance criterion is the original
  R37 path smoke. Otherwise hold for another SFDP parity round focused on star
  / hub topology and Graphviz RNG semantics.

### FMMM / FDP

Hypothesis tested:

- Verified `fidelity_mode=True` is the only accepted fdp fidelity selector.
- Verified `layout_fmmm_pipeline(..., fidelity_mode=True, clusters=...)`
  enters `graphviz_fdp_fidelity()`.
- Found a wiring miss: recursive sibling component offsets still used a local
  row packer and bypassed the R36 Graphviz tile packer.

Fix applied:

- `_fdp_recursion_component_offsets()` now calls `_graphviz_tile_pack_offsets()`.

Residual diagnosis:

- The single R37 smoke moved from `0.210095198` to `0.244951112`; the wiring is
  now more faithful to the ported tilepack component, but the end-to-end smoke
  is not better.
- R36 `fdp_recursion` explicitly notes that Graphviz `tLayout`, `xLayout`, and
  `packGraphs` numerical kernels are still represented by Dagua FM^3 plus
  partial packing semantics. That is the dominant architectural mismatch.
- Ports currently record attachment metadata, but Dagua returns only node
  coordinates, not Graphviz fdp splines or route-aware endpoint geometry.

Verdict:

- `real architectural mismatch`.
- Ship recommendation: drop this R37 variant for now. It is not a safe
  Graphviz-fdp fidelity variant while smoke remains `>0.05` and is worse after
  routing through the tilepack port.

### Neato

Hypothesis tested:

- Verified `layout_neato_pipeline` accepts `"graphviz"` and `"graphviz_neato"`.
- `git show bf026eb` confirms overlap fidelity is gated by any truthy
  fidelity mode, but only `"graphviz"` enables the newer PCA + packed-CG solver.
- Alias toggle on path smoke:
  - `"graphviz"`: mean `0.442291663`
  - `"graphviz_neato"`: mean `0.029028904`
  - `True`: mean `0.029028904`
  - `False`: mean `0.015874333`

Fix applied:

- Updated the R37 `classic_neato_graphviz_fidelity` variant to use
  `fidelity_mode="graphviz_neato"`.

Residual diagnosis:

- R37 path smoke improves from `0.442291663` to `0.028738625`.
- Wider smoke still has high star and clustered residuals, so the established
  compatibility path is smoke-good on the path graph but not bit-exact across
  topology classes.
- The newer R36 PCA + packed-CG solver is not ready to ship as the benchmark
  variant; it is a separate solver-port follow-up, not an alias miss.

Verdict:

- `needs more work`.
- Ship recommendation: ship the revised variant only for the narrow R37 path
  smoke criterion. For broader smoke coverage, keep it out of the final
  Graphviz-fidelity set until the PCA/CG solver parity is debugged or the
  compatibility path is characterized against more topologies.

## Verification

```text
ruff check . --fix
Found 1 error (1 fixed, 0 remaining).

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_layout/test_fmmm_fdp_recursion.py tests/test_layout/test_fmmm_fidelity.py -x --tb=short -q
15 passed, 2 warnings in 0.15s

pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
422 passed, 8 warnings in 1683.83s (0:28:03)

pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
ERROR tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
2 warnings, 1 error in 0.55s
```

The final non-slow collection failure matches the pre-existing blocker recorded
in the R36 summaries and was not modified for R38 because it is outside this
task's scope.
