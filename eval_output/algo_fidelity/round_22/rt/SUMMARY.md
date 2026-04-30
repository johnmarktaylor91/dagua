# Round 22 RT Summary

## Changes

- Added opt-in `fidelity_mode="igraph"` and `traversal_mode` support to the
  Reingold-Tilford pipeline.
- In igraph fidelity mode, RT uses unit sibling/layer spacing, directed
  traversal semantics, reachability-based root ordering, and synthetic-root
  packing for multi-root forests.
- Added regression coverage for unit spacing, OUT traversal, duplicate-edge root
  ranking, and invalid fidelity mode handling.

## Spec References

- Root mismatch: `ROUND_21_DIFF_rt.md:426-431`.
- Mode/direction mismatch: `ROUND_21_DIFF_rt.md:433-439`.
- Component super-root mismatch: `ROUND_21_DIFF_rt.md:451-454`.
- Recommended scope: `ROUND_21_DIFF_rt.md:505-521`.

## Baseline

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_rt igraph_rt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/rt/baseline
```

Output:

```text
graphs: 5
median: 0.000000
p25: 0.000000
p75: 0.074164
p95: 0.194303
worst: mixed_width_labels 0.224338
```

## Remeasure

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_rt igraph_rt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/rt/remeasure
```

Output:

```text
graphs: 5
median: 0.000000
p25: 0.000000
p75: 0.074164
p95: 0.194303
worst: mixed_width_labels 0.224338
```

## Verification

- `ruff check dagua/layout/ops/coordinate.py dagua/layout/ops/pipelines/reingold_tilford.py tests/test_layout/test_rt_fidelity.py --fix`: passed.
- `pytest tests/test_layout/test_rt_fidelity.py -q`: 4 passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "rt"`: blocked during collection by unrelated `tests/test_layout/test_fmmm_fidelity.py` import error for `_GALAXY_CHOICE_LOWER`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked during collection by unrelated `tests/test_classic_drl.py` import error for `layout_drl`.

## Decision

Committed under the opt-in infrastructure criterion. Default `classic_rt`
remeasure is unchanged because the fidelity path is not enabled by default.
