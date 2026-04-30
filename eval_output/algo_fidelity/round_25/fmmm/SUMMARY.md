# Round 25 FMMM Straggler Fix

## Measurement

Baseline command:

```text
python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_25/fmmm/baseline
```

Baseline result:

```text
graphs: 5
median: 0.028078
p25: 0.025648
p75: 0.074976
p95: 0.212938
worst: parallel_multiedge_bundle 0.247428
```

Post-fix command:

```text
python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_25/fmmm/post_fix
```

Post-fix result:

```text
graphs: 5
median: 0.011510
p25: 0.011510
p75: 0.017002
p95: 0.033429
worst: tl_mlp_3layer 0.037536
```

## Per-graph Medians

| graph | baseline | post_fix | delta |
| --- | ---: | ---: | ---: |
| linear_3layer_mlp | 0.025648 | 0.011510 | -0.014138 |
| mixed_width_labels | 0.074976 | 0.017002 | -0.057974 |
| nested_shallow_enc_dec | 0.025597 | 0.011510 | -0.014087 |
| parallel_multiedge_bundle | 0.247428 | 0.004498 | -0.242930 |
| tl_mlp_3layer | 0.028078 | 0.037536 | +0.009458 |

Median improvement: `0.016568`.
Worst-graph improvement for `parallel_multiedge_bundle`: `0.242930`.

## Fixes Applied

- Enabled the OGDF-aligned FMMM path for `classic_fmmm` competitor defaults via `fidelity_mode=True`.
- Removed the live small-graph quality selector from `ClassicFMMM`; it was bypassing the Round 22 reference-mode knobs and choosing non-OGDF candidate layouts.
- Added a `fidelity_mode` alias to the FMMM pipeline so evaluation defaults can request the reference path directly.
- In FMMM fidelity mode, changed reduced parallel-edge weights from summed multiplicity to averaged unit strength. OGDF `make_simple_loopfree()` averages reduced edge lengths and does not make duplicate edges a stronger spring; the old Dagua behavior made `parallel_multiedge_bundle` an asymmetric weighted triangle.
- Corrected the OGDF oscillation damping factors to match `FMMMLayout.cpp`.

## Worst-graph Diagnosis

`parallel_multiedge_bundle` has three nodes and six raw edges: three `src-mid`, two `mid-dst`, and one `src-dst`. OGDF reduces this to a simple triangle with averaged ideal edge lengths. Dagua already collapsed parallel edges, but it stored duplicate counts as attraction weights (`3, 2, 1`), so the solver intentionally made the triangle non-equilateral. With fidelity-mode averaged weights, the seed-42 panel RMSD dropped from `0.247608` to `0.004458`.

## Verification

```text
git log --oneline --grep "round 2[23] fmmm" -- dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py dagua/eval/competitors/classic_competitor.py
06e530f feat(fidelity): round 23 fmmm -- revert regressed postprocess
a308fd7 feat(fidelity): round 23 fmmm -- reference postprocess
536cff4 feat(fidelity): round 22 fmmm -- add reference mode
```

```text
ruff check dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py dagua/eval/competitors/classic_competitor.py tests/test_layout/test_fmmm_fidelity.py --fix
All checks passed!
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

```text
pytest tests/test_layout/ -x --tb=short -q -k "fmmm" --ignore=tests/test_layout/test_gem_fidelity.py
......                                                                   [100%]
6 passed, 329 deselected in 0.29s
```

Final Tier 2 command:

```text
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

Result:

```text
ERROR collecting tests/test_classic_drl.py
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic' (unknown location)
1 error in 0.11s
```

Full `ruff check . --fix` is currently blocked by pre-existing untracked file `scripts/round_24_aggregate.py`, which reports `F841 Local variable any_eq is assigned to but never used`.

## Residual Rationale

No `principled_residual` classification is needed for this round because the required improvement threshold was met. Remaining FMMM residuals are now below `0.04` median RMSD on the bounded set, with `tl_mlp_3layer` the new worst graph at `0.037536`.
