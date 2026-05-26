# Round 65 DRL Final Predicate Summary

## Result

Documented the current DrL pruning floor. I did not change
`dagua/layout/ops/drl.py` or `dagua/layout/ops/pipelines/drl.py` because the
current native path does not execute any edge cut for the three Round 62 named
failures, so changing only `maxLength > cut_off_length` cannot close those
cases below `1e-6`.

Round 62 reference-smoke failures:

| Graph | Options | Seed | RMSD | Max delta |
| --- | --- | ---: | ---: | ---: |
| 8-node star | default | 42 | 51.25846862792969 | 106.71883392333984 |
| 8-node star | coarsest | 42 | 37.208343505859375 | 78.26630401611328 |
| 6-node weighted | final | 17 | 62.6909065246582 | 120.63502502441406 |

## Predicate Trace

Local instrumentation of `_runtime_solve_analytic` found no removed neighbor in
any of those cases. The closest local predicate margins were:

| Graph | Options | Seed | Sweep | Stage | Node | maxLength | cut_off_length | Margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8-node star | default | 42 | 505 | 2 | 0 | 1005.7819213867188 | 8000.0 | -6994.218078613281 |
| 8-node star | coarsest | 42 | 502 | 2 | 0 | 698.0407104492188 | 8000.0 | -7301.959289550781 |
| 6-node weighted | final | 17 | 151 | 3 | 0 | 3.3361589908599854 | 8000.0 | -7996.66384100914 |

The first local pruning-eligible windows were:

| Graph | Options | Seed | Sweep | Stage | Node | Degree | min_edges | max neighbor | maxLength | cut_off_length |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8-node star | default | 42 | 427 | 2 | 0 | 7 | 6.800004959106445 | 2 | 1310.988037109375 | 16880.0 |
| 8-node star | coarsest | 42 | 427 | 2 | 0 | 7 | 6.800004959106445 | 7 | 509.87396240234375 | 16880.0 |
| 6-node weighted | final | 17 | 102 | 3 | 0 | 2 | 1.0 | 1 | 0.005768483504652977 | 8000.0 |

As a control, a 100-node deterministic sparse graph did perform a local cut at
sweep 465, stage 2, node 4, degree 13, `min_edges=0.8000069260597229`,
`cut_off_length=12320.0`, removing neighbor 21. That confirms the trace catches
real cuts; it just does not see cuts in the Round 62 named failures.

## Why This Is a Floor

The active solver already uses the igraph-style state machine in
`_run_reference_drl`: node updates occur before stage control, `min_edges` and
`cut_off_length` are rounded through float32, neighbor iteration is sorted by
node id, and cuts are one-sided from the current node's neighbor map.

For the measured failing cases, the local predicate is not near the decision
boundary. The closest margin is roughly `-6994`, far beyond any float32, Python
double, decimal, or mpmath rounding effect. Arbitrary precision in the predicate
would preserve the same no-cut decision locally.

The exact reference-side first disagreement iteration could not be collected in
this pass without executing the forbidden reference adapter or adding forbidden
delegation. Under the allowed native-only trace, the irreducible fact is that
the named failures are not locally pruning-active, so the requested predicate
change has no causal path to the observed RMSD floor.

## Verification

Commands run:

```text
python -m pytest tests/test_pipeline_drl.py -x --tb=short -q
```

Result:

```text
30 passed, 2 warnings in 20.48s
```

Forbidden-pattern scan on the scoped Python modules returned no matches.
