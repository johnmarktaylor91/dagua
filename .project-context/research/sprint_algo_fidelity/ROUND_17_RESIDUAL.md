# Round 17 Residual -- NeuLay

Status: RESIDUAL
Family: neulay
Date: 2026-04-30

## Classification

```text
principled_residual: source_unavailable_cached_reference_floor
```

## Rationale

The installed environment has PyTorch Geometric but does not have the upstream
`neulay` or `NeuLay` Python package. The independent reference therefore cannot
be run live.

The requested live comparator did run against cached upstream positions, but
its graph selector narrowed the requested five-graph set to two graphs because
it requires cached records for both `neulay` and `classic_neulay`:

```text
linear_3layer_mlp
parallel_multiedge_bundle
```

The two selected graphs give mixed evidence:

```text
linear_3layer_mlp: dagua-vs-neulay median 0.139584, within-neulay mean 0.174630
parallel_multiedge_bundle: dagua-vs-neulay median 0.104444, within-neulay mean 0.000916
```

That means one graph is already within a large upstream stochastic floor, while
the parallel-edge graph is a real mismatch against a near-deterministic cached
target. Without upstream source, the parallel-edge mismatch could be caused by
edge multiplicity, loss normalization, package version, or cached-reference
drift. There is no source-backed small lever that can be defended as the
correct NeuLay fix.

## Attempted Lever

No code lever was applied. The only confirmed local inconsistency is that the
public pipeline default is `lr=0.01` while the benchmarked `classic_neulay`
competitor and variant registry pass `lr=0.1`. Changing that public default
would not affect the measured comparator path, so it would not satisfy the
Round 17 commit criterion.

## Follow-Up

The next useful NeuLay round should install or vendor the exact upstream
`neulay` package used to generate the cached tensors, then compare source-level
details for:

- edge multiplicity handling in the elastic term;
- exact GCN architecture and activation;
- RMSprop defaults and learning-rate split between GCN and direct phases;
- random initialization order and PyTorch/PyG version effects;
- loss scaling or normalization.
