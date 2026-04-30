# Round 22 Stress-SGD Summary

## Changes

- Added `fidelity_mode` to the Stress-SGD pipeline.
- In `fidelity_mode`, weighted duplicate/reverse edges use summed adjacency preprocessing,
  matching the `s_gd2` adapter behavior called out in
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_stress_sgd.md:671-675`.
- In `fidelity_mode`, exact stress distances and weights are stored as `float64`, following
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_stress_sgd.md:676-678`.
- In `fidelity_mode`, edgeless `N > 1` graphs return zeros and disconnected non-empty graphs
  raise, matching the policy described in
  `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_stress_sgd.md:594-603`.
- Wired `classic_stress_sgd` benchmark dispatch to pass `fidelity_mode=True` and forward
  graph edge weights.
- Added focused regression tests in `tests/test_layout/test_stress_sgd_fidelity.py`.

## Measurements

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_stress_sgd sgd2 \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/stress_sgd/baseline
```

Baseline result: median `0.026369`, p25 `0.002486`, p75 `0.042833`, worst
`linear_3layer_mlp 0.042833`.

After command:

```bash
python scripts/algo_fidelity_live_compare.py classic_stress_sgd sgd2 \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/stress_sgd/after
```

After result: median `0.026369`, p25 `0.002486`, p75 `0.042833`, worst
`linear_3layer_mlp 0.042833`.

The measured subset did not move because these five rows are small connected cases; the
implemented mode targets weighted multiedges, float64 weighted shortest-path drift, and
edgeless/disconnected policy rows.

## Verification

- `ruff check dagua/layout/ops/stress_sgd.py dagua/layout/ops/pipelines/stress_sgd.py dagua/eval/competitors/classic_competitor.py tests/test_layout/test_stress_sgd_fidelity.py --fix` passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "stress_sgd"` passed: `5 passed, 286 deselected`.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- Final Tier 2 command failed before running this area: `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` hit `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py`.
- Earlier full `ruff check . --fix` is blocked by unrelated line-length failures in
  `dagua/eval/competitors/sgd2_multi_competitor.py` and `dagua/layout/ops/fmmm.py`.

## Commit Criterion

Median did not improve by `0.03`, but this round meets the relaxed criterion via a clean
opt-in `fidelity_mode` flag with regression tests.
