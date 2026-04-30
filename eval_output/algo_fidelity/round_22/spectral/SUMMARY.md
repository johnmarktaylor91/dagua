# Round 22 Spectral Summary

## Changes

- Added opt-in `networkx_fidelity` support for spectral layout, following `ROUND_21_DIFF_spectral.md:267-283` and `ROUND_21_DIFF_spectral.md:321-328`.
- Under `networkx_fidelity=True`, spectral uses the unnormalized Laplacian, mirrors NetworkX's two-node zero-center special case, and selects eigenvectors by sorted `[1 : dim + 1]` instead of filtering all near-zero eigenvalues.
- Registered `classic_spectral_nx_fidelity` as an opt-in variant paired with `nx_spectral`.

## Measurement

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_spectral nx_spectral --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_22/spectral/baseline
```

Baseline median: `0.100482`; p25: `0.100482`; p75: `0.111416`; p95: `0.299828`; worst: `mixed_width_labels 0.346932`.

Post-change default command:

```bash
python scripts/algo_fidelity_live_compare.py classic_spectral nx_spectral --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_22/spectral/after
```

Post-change default median: `0.100482`; p25: `0.100482`; p75: `0.111416`; p95: `0.299828`; worst: `mixed_width_labels 0.346932`.

Opt-in variant command:

```bash
python scripts/algo_fidelity_live_compare.py classic_spectral_nx_fidelity nx_spectral --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_22/spectral/nx_fidelity
```

Variant live-compare rows: `0`; the script requires cached successful records for the Dagua-side engine name, and `classic_spectral_nx_fidelity` is new in this round.

## Verification

- `pytest tests/test_layout/ -x --tb=short -q -k "spectral"`: failed during unrelated collection on `tests/test_layout/test_fmmm_fidelity.py` because `dagua.layout.ops.fmmm` does not export `_GALAXY_CHOICE_LOWER`.
- `pytest tests/test_layout/test_spectral_fidelity.py tests/test_pipeline_spectral.py -x --tb=short -q`: passed, `14 passed`.
- `ruff check .`: passed.
- `ruff check dagua/layout/ops/pipelines/spectral.py dagua/layout/ops/preprocess.py dagua/layout/ops/embed.py dagua/eval/variants.py tests/test_layout/test_spectral_fidelity.py --fix`: passed.
- `ruff format dagua/layout/ops/pipelines/spectral.py dagua/layout/ops/preprocess.py dagua/layout/ops/embed.py dagua/eval/variants.py tests/test_layout/test_spectral_fidelity.py --check`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: failed during unrelated collection on `tests/test_classic_drl.py` because `dagua.layout.classic` does not export `layout_drl`.

## Decision

Commit criterion met via clean opt-in `networkx_fidelity` infrastructure with regression tests. The public `classic_spectral` default intentionally remains unchanged.
