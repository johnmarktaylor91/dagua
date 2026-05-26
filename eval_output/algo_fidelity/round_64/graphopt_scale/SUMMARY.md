# Round 64 GraphOpt Scale Residuals

## Scope

Audited `dagua/layout/ops/pipelines/graphopt.py` and the GraphOpt op implementation
classes in `dagua/layout/ops/force.py`. The requested
`dagua/layout/ops/graphopt.py` path does not exist in this checkout.

No runtime delegation was added.

## Variant Parameters

From `dagua/eval/variants.py`:

| Variant | Reimplementation params | Reference params |
|---|---|---|
| `classic_graphopt_mass_low` | `niter=500`, `node_charge=0.001`, `node_mass=10.0`, `spring_constant=1.0` | same |
| `classic_graphopt_spring2` | `niter=500`, `node_charge=0.001`, `node_mass=30.0`, `spring_constant=2.0` | same |

Both variants raise the effective explicit-step gain relative to the default:
`mass_low` triples force-to-position movement, and `spring2` doubles the spring
force term.

## Diagnosis

The force law is not failing on the first step. Serial direct probes against
python-igraph show machine-epsilon agreement through early iterations, then
growth after repeated high-gain updates:

| Case | niter=1 | niter=20 | niter=50 | niter=100 | niter=500 |
|---|---:|---:|---:|---:|---:|
| `mass_low` on `real_lesmis_77` | `1.77e-17` | `6.00e-15` | `1.35e-08` | `4.81e-04` | `3.99e-02` |
| `mass_low` on `dense_pair_50` | `3.64e-17` | `2.52e-13` | `1.02e-07` | `1.39e-03` | `4.97e-03` |
| `spring2` on `real_lesmis_77` | `1.77e-17` | `1.11e-13` | `4.73e-08` | `5.29e-03` | `8.60e-03` |

Those values are sqrt-mean Procrustes RMSD. The R56 report uses the normalized
Frobenius norm form, which scales by `sqrt(N)` and corresponds to the reported
scale failures.

Fresh direct comparison at 500 iterations using the R56 normalized Frobenius
metric:

| Graph | default | charge_low | charge_high | mass_low | mass_high | spring2 |
|---|---:|---:|---:|---:|---:|---:|
| `real_lesmis_77` | `7.19e-12` | `1.94e-06` | `1.59e-13` | `3.50e-01` | `1.85e-13` | `7.54e-02` |
| `dense_pair_50` | `1.12e-14` | `5.75e-11` | `1.01e-15` | `3.52e-02` | `5.24e-16` | `6.14e-09` |

This matches the R56 pattern: the stable GraphOpt variants stay at machine
epsilon or near-machine epsilon on the same graphs, while only the higher-gain
variants drift on the dense real cases.

## Conclusion

This is expected chaotic residual for the `node_mass=10.0` and
`spring_constant=2.0` GraphOpt regimes, not a parameter-specific code bug. The
port matches python-igraph through the early force steps; the residual appears
only after many explicit force updates where tiny floating-point differences
are amplified by the larger movement gain.

No algorithmic fix was applied. The pipeline fidelity notes now document this
known scale floor.

## Commands

Focused benchmark attempt:

```bash
python3 scripts/run_benchmark.py --seeds 1 --variants \
  --output-dir eval_output/algo_fidelity/round_64/graphopt_scale/scratch \
  --graphs braided_feedback_tails,densenet_block,regular_3_30,planar_60,dense_pair_50,real_lesmis_77,wide_1_100_1,powerlaw_500,rgg_2000 \
  --workers 8 \
  --engines classic_graphopt_default,igraph_graphopt__for__classic_graphopt_default,classic_graphopt_charge_low,igraph_graphopt__for__classic_graphopt_charge_low,classic_graphopt_charge_high,igraph_graphopt__for__classic_graphopt_charge_high,classic_graphopt_mass_low,igraph_graphopt__for__classic_graphopt_mass_low,classic_graphopt_mass_high,igraph_graphopt__for__classic_graphopt_mass_high,classic_graphopt_spring2,igraph_graphopt__for__classic_graphopt_spring2 \
  --timeout 600 --watchdog-timeout 720
```

That run was stopped after partial results because concurrent exact GraphOpt
jobs caused worker layout timeouts. The diagnostic results above come from
serial direct python-igraph comparisons, which avoided the concurrency artifact.
