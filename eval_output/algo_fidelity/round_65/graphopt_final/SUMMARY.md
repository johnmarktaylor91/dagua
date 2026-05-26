# Round 65 GraphOpt Final

## Scope

Implemented a fidelity-only scalar GraphOpt iteration in
`dagua/layout/ops/pipelines/graphopt.py`. Non-fidelity GraphOpt still uses the
existing tensorized `GraphOptIteration`.

The scalar path matches igraph 1.0.0 C GraphOpt arithmetic order:

- pending x/y force vectors are separate Python float lists
- repulsion loops `this_node` then `other_node = this_node + 1..N`
- repulsion applies only when `distance != 0.0` and `distance < 500.0`
- springs are applied in prepared edge order
- movement is clamped independently per axis after all forces are accumulated

Source checked: `igraph/igraph` tag `1.0.0`, `src/layout/graphopt.c`.

## Result

The high-gain variants now close below `1e-6` on the prior failing smoke cases.

| Graph | Variant | RMSD | Normalized Frobenius |
|---|---:|---:|---:|
| `real_lesmis_77` | `mass_low` | `4.35691959479e-09` | `3.82318142830e-08` |
| `real_lesmis_77` | `spring2` | `4.33631728319e-09` | `3.80510297324e-08` |
| `dense_pair_50` | `mass_low` | `5.11583138324e-09` | `3.61743906249e-08` |
| `dense_pair_50` | `spring2` | `4.81578121622e-09` | `3.40527155470e-08` |

This confirms the Round 64 floor was not irreducible chaos in the algorithm
itself. The residual came from tensor reduction/order differences that were
large enough to enter chaotic-amplification regimes under `node_mass=10.0` and
`spring_constant=2.0`.

## Verification

Smoke run:

```bash
python scripts/run_benchmark.py --seeds 1 --variants \
  --output-dir eval_output/algo_fidelity/round_65/graphopt_final/smoke \
  --graphs real_lesmis_77,dense_pair_50 \
  --workers 1 \
  --engines classic_graphopt_mass_low,igraph_graphopt__for__classic_graphopt_mass_low,classic_graphopt_spring2,igraph_graphopt__for__classic_graphopt_spring2 \
  --timeout 600 --watchdog-timeout 720
```

Benchmark summary: `8 total, 8 ok, 0 skipped, 0 errors, 0 timeouts`.

Residuals were computed from the saved smoke tensors and written to
`eval_output/algo_fidelity/round_65/graphopt_final/smoke/residuals.json`.
