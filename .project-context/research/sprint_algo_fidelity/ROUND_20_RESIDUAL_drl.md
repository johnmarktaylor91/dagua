# Round 20 Residual -- DRL

Status: REVERTED
Family: drl
Date: 2026-04-30

## Attempted Stage 1 Bundle

- Matched igraph's effective `ReCompute()` sweep counts by adding:
  - one init-parameter sweep,
  - cooldown/crunch/simmer boundary sweeps,
  - one final stage-6 sweep.
- Changed candidate acceptance to igraph's literal old-energy-vs-random-energy
  comparison, accepting the analytic coordinate when old energy wins.
- Aligned unambiguous preset values:
  - `REFINE.init.damping_mult = 0.0`
  - `REFINE.cooldown.temperature = 200.0`
  - `FINAL.expansion = (50, 50.0, 0.1, 0.25)`

## Measurement

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/drl/baseline
```

After command:

```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/drl/after
```

Results:

| Run | Median | P25 | P75 | P95 | Worst |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.206197 | 0.192176 | 0.263942 | 0.263942 | linear_3layer_mlp 0.263942 |
| Stage 1 bundle | 0.205830 | 0.173965 | 0.236631 | 0.236631 | linear_3layer_mlp 0.236631 |

Median improvement: `0.000367`, below the required `0.030000` commit threshold.

## Test Results Before Revert

```text
pytest tests/test_layout/ -x --tb=short -q -k "drl"
........                                                                 [100%]
8 passed, 239 deselected in 0.36s
```

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

```text
ruff check . --fix
E501 Line too long (113 > 100)
  --> dagua/layout/ops/pipelines/neulay.py:88:101
```

The ruff failure is in an out-of-scope, pre-existing `neulay` file. A scoped
ruff check on the touched DRL implementation and DRL test passed before revert.

## Decision

Per the Round 20 spec, the code and test patch was reverted and no commit was
created.

## Residual

The Stage 1 bundle moved several graph medians in the right direction but barely
changed the aggregate median. This suggests the next high-impact divergences are
still the deferred coupled trajectory controls from Round 19:

- fine-density lifecycle,
- coarse density kernel/cell boundary behavior,
- edge-cut semantics,
- initialization/RNG contract.

Density grid and edge cutting remain intentionally untouched in this round.
