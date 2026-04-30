# Round 20 GraphOpt Fix Results

## Scope

Applied the top three Round 19 adversarial fixes:

- GraphOpt initialization now uses NumPy `RandomState`, samples `[-1, 1]`, and
  draws in igraph-compatible column-major order.
- RNG semantics are no longer Python `random.Random`; fixed-seed expectations
  are documented by regression tests against NumPy `RandomState`.
- GraphOpt spring forces ignore `edge_weights`, matching igraph's unweighted
  `graphopt.c` implementation.

## Measurement

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_20/graphopt/<baseline|after>
```

Results:

| Run | Median | p25 | p75 | p95 | Worst |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.067702 | 0.018174 | 0.067702 | 0.260675 | tl_mlp_3layer 0.308918 |
| After | 0.051250 | 0.051250 | 0.151859 | 0.302256 | tl_mlp_3layer 0.339855 |

Median improvement: `0.016452`.

## Commit Decision

The required commit threshold was median improvement `>= 0.02`. The measured
improvement was below threshold, so no commit was made.

## Verification

Passed:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_layout/ -x --tb=short -q -k "graphopt or init"
```

The broader targeted command
`pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` was started but
did not complete under heavy concurrent workspace load.
