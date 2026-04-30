# Round 16 Residual -- GraphOpt vs igraph

Status: RESIDUAL
Family: graphopt
Date: 2026-04-30

## Attempted Lever

Applied the scoped init-range alignment in `GraphOptInitializePositions`:

- Before: Python `random.Random(seed).random()` for each coordinate, uniform
  `[0, 1]`.
- Attempt: Python `random.Random(seed).uniform(-1.0, 1.0)` for each coordinate,
  uniform `[-1, 1]`, matching igraph's documented `igraph_layout_random()`
  range.

The rest of the graphopt pipeline defaults were inspected and already matched
the requested igraph defaults:

- `niter=500`
- `node_charge=0.001`
- `node_mass=30.0`
- `spring_length=0.0`
- `spring_constant=1.0`
- `max_sa_movement=5.0`
- `_GRAPHOPT_COULOMBS_CONSTANT = 8_987_500_000.0`

## Measurement

Command:

```text
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_16/post_fix
```

Output:

```text
Wrote 75 rows to eval_output/algo_fidelity/round_16/post_fix/multi_seed_rmsd.csv
Wrote summary to eval_output/algo_fidelity/round_16/post_fix/multi_seed_summary.json
graphs: 5
median: 0.069149
p25: 0.018140
p75: 0.069149
p95: 0.261128
worst: tl_mlp_3layer 0.309122
```

Round 15 baseline:

```text
median: 0.067702
p25: 0.018174
p75: 0.067702
p95: 0.260675
worst: tl_mlp_3layer 0.308918
```

Result: median regressed by `+0.001447`; worst graph regressed by `+0.000204`.

## TOST

No aggregate movement toward equivalence occurred.

Graph-level verdicts remained unchanged:

- `parallel_multiedge_bundle`: `equivalent_at_0.5x`
- `tl_mlp_3layer`: `equivalent_at_1x`
- `linear_3layer_mlp`: `not_equivalent`
- `nested_shallow_enc_dec`: `not_equivalent`
- `mixed_width_labels`: `not_equivalent`

## Classification

`algorithmic_residual: init_range_not_causal`

The simple init-domain mismatch was real, but changing the domain without
matching igraph's exact random stream/order did not improve fidelity. Given
that defaults and the main force constants already match, the remaining gap is
likely in a lower-level algorithmic detail rather than this scalar init range.

## Disposition

The attempted code change was reverted because it failed the commit criterion.
No commit was made.

## Tests

Command:

```text
pytest tests/test_layout/ -x --tb=short -q -k "graphopt or init" 2>&1 | tail -30
```

Output:

```text
.....                                                                    [100%]
5 passed, 228 deselected in 1.31s
```
