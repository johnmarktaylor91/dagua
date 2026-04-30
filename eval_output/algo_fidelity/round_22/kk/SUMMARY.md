# Round 22 KK Summary

## Changes

- Applied the top three KK fidelity fixes from
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_kk.md`.
- Aligned `classic_kk` with NetworkX's SciPy default iteration semantics by
  using `steps=None` in the adapter and KK fidelity variants
  (`ROUND_21_DIFF_kk.md:490-498`, `ROUND_21_DIFF_kk.md:564-569`).
- Disabled the adapter-level `orient_to_direction` default for `classic_kk`
  fidelity comparison while keeping the pipeline option available
  (`ROUND_21_DIFF_kk.md:513-523`, `ROUND_21_DIFF_kk.md:571-574`).
- Added KK-specific duplicate-edge collapse policy support and made KK directed
  distances use NetworkX `DiGraph` last-write semantics. Added regression tests
  for `[10, 1]` and `[1, 10]` duplicate weights
  (`ROUND_21_DIFF_kk.md:500-511`, `ROUND_21_DIFF_kk.md:576-580`).

## Measurement

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_kk nx_kamada_kawai \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/kk/baseline
```

Baseline:

- Rows: 30
- Median: 0.000000
- P25: 0.000000
- P75: 0.000000
- P95: 0.000000
- Worst: `tl_mlp_3layer` 0.000000

After:

- Rows: 30
- Median: 0.000000
- P25: 0.000000
- P75: 0.000000
- P95: 0.000000
- Worst: `tl_mlp_3layer` 0.000000

Result: median unchanged on this already-zero subset. This round is retained as
clean fidelity infrastructure with regression coverage for hidden adversarial
cases.

## Verification

- `pytest tests/test_layout/test_kk_fidelity.py tests/test_pipeline_kk.py -x --tb=short -q`:
  passed, `14 passed in 0.14s`.
- `python -m py_compile tests/test_layout/test_kk_fidelity.py dagua/eval/competitors/classic_competitor.py dagua/eval/variants.py dagua/layout/ops/distance.py dagua/layout/ops/graph_utils.py`:
  passed.
- `ruff check dagua/eval/competitors/classic_competitor.py dagua/eval/variants.py dagua/layout/ops/distance.py dagua/layout/ops/graph_utils.py tests/test_layout/test_kk_fidelity.py --fix`:
  passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "kk"`: blocked during
  collection by out-of-scope `ImportError: cannot import name
  '_GALAXY_CHOICE_LOWER'` in `tests/test_layout/test_fmmm_fidelity.py:9`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: blocked
  during collection by out-of-scope
  `ImportError: cannot import name '_FA2_REFERENCE_PACKAGE_ORDER'` in
  `tests/test_layout/test_fa2_fidelity.py:7`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  blocked during import by out-of-scope
  `NameError: name '_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY' is not defined` in
  `dagua/layout/ops/sugiyama.py:1413`.
- `ruff check . --fix`: blocked by out-of-scope line-length errors in
  `dagua/eval/competitors/sgd2_multi_competitor.py:36` and
  `dagua/layout/ops/fmmm.py:1389`; the command briefly applied broad fixes
  outside kk scope, which were reverted.
