# Round 22 FA2 Summary

## Changes

- Added an opt-in FA2 `fidelity_mode` flag that runs FA2 initialization, degree/mass,
  edge-weight caches, and old-force history in `torch.float64`.
- Changed FA2 strong-gravity validity to match live `fa2`: apply when any coordinate is
  nonzero, not only when both x/y are nonzero.
- Made the `fa2_ref` package target explicit as `("fa2", "fa2_modified")`, preserving
  the Round 21 decision to target live `fa2`.

## Spec References

- Float64 fidelity path: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fa2.md`
  ranked fix 1 and recommended scope item 3 cite dagua float32 internals at
  `dagua/layout/ops/init.py:727-731`, `dagua/layout/ops/preprocess.py:1303-1325`,
  and `dagua/layout/ops/force.py:1748-1999` versus live `fa2` C double fields.
- Strong gravity: ranked fix 2 and recommended scope item 2 cite dagua's `and`
  condition at `dagua/layout/ops/force.py:1897-1900` versus live `fa2`'s `or`
  condition at `fa2/fa2util.pyx:219-224`.
- Reference target: ranked fix 3 and recommended scope item 1 cite adapter load order
  at `dagua/eval/competitors/fa2_competitor.py:31-38`.

## Measurement

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_fa2 fa2_ref \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/fa2/{baseline,after}
```

Baseline median: `0.088598`
After median: `0.088598`

Per-graph medians were unchanged because the new dtype path is opt-in and the requested
default live comparison does not exercise strong-gravity mode.

## Commit Criterion

Met by the clean opt-in `fidelity_mode` flag with regression tests.

## Verification

- `ruff check . --fix`: failed on out-of-scope FMMM line length at
  `dagua/layout/ops/fmmm.py:1389`.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "fa2"`: failed during collection in
  out-of-scope `tests/test_layout/test_classical_mds_fidelity.py` with
  `ImportError: cannot import name 'ops' from 'layout'`.
- `pytest tests/test_layout/test_fa2_fidelity.py -x --tb=short -q`: passed,
  `3 passed in 0.05s`.
- `pytest tests/test_pipeline_fa2.py -x --tb=short -q`: passed,
  `19 passed in 1.24s`.
- `pytest tests/test_graph.py -x --tb=short -q`: passed,
  `37 passed in 0.51s`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection in out-of-scope `tests/test_build_gallery_audit.py` with
  `ImportError: cannot import name 'DEFAULT_COMPARISON_STROKE'`.
