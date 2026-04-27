# Sprint-40 No-Fix Report

Date: 2026-04-26
Branch: codex/sprint-31a-gate-refinement
Baseline: git HEAD 3eaa01c

## Outcome

No fix shipped. `torch.compile` is available in this environment
(`torch==2.8.0+cu128`), but compiling the `dagua_native` `LossGroup`
boundary failed the Sprint-40 validation gate on both correctness and
runtime. The exploratory code changes were reverted and no commit was
made.

## Attempted implementation

The trial implementation added `LayoutConfig.use_torch_compile = True`
and routed the `dagua_native` `build_gradient_core()` loss step through
a compiled combined `LossGroup` forward pass using:

```python
torch.compile(fn, mode="reduce-overhead")
```

The fallback path caught compile failures and returned to eager
`per_loss` backward. Unit coverage used a fake `torch.compile` to verify
the plumbing and fallback, but the real runtime validation failed.

## Validation command

```bash
python /tmp/sprint40_validate.py
```

The script compared `LayoutConfig(use_torch_compile=False)` against the
compiled path on the eight requested representative graphs and evaluated
position tolerance, composite metric delta, and runtime speedup.

## Validation results

| graph | eager | compiled | speedup | max_abs_diff | composite delta |
|---|---:|---:|---:|---:|---:|
| er_500 | 16.019s | 99.678s | 0.16x | 20832.4 | +0.270 |
| rgg_100 | 9.388s | 56.574s | 0.17x | 280 | -0.001 |
| dependency_graph_100 | 2.090s | 8.168s | 0.26x | 0.78125 | -0.004 |
| real_lesmis_77 | 1.865s | 9.155s | 0.20x | 0 | +0.000 |
| small_world_100 | 1.484s | 1.522s | 0.98x | 0 | +0.000 |
| random_dag_200 | 1.842s | 3.197s | 0.58x | NaN | +27.342 |
| dense_pair_50 | 1.128s | 3.855s | 0.29x | 0 | +0.000 |
| ba_500 | 3.876s | 28.806s | 0.13x | NaN | +15.639 |

Aggregate:

- geometric speedup: `0.273x`
- graphs at `>=1.5x`: `0/8`
- correctness: failed (`max_abs_diff > 1e-3`, NaNs)
- metric stability: failed (`random_dag_200`, `ba_500`)
- speed gate: failed

## Root cause

The `LossGroup` boundary is too object-heavy and dynamic for this direct
`torch.compile` application:

- Dynamo graph-broke on scalar extraction such as
  `node_sizes.detach().max().item()` in the spatial-hash cutoff path.
- Dynamo hit the recompile limit on dynamic candidate-pair shapes in
  `spatial_hash._cross_cell_pairs`.
- Cluster cache construction triggered backend failures around
  `.tolist()` / `.item()` in `_ClusterCache`.
- Switching from per-loss backward to compiled combined backward changed
  numerical behavior enough to produce large position differences and
  NaNs on two target graphs.

## Recommendation

Do not enable `torch.compile` at the whole `LossGroup` or full inner-loop
boundary. A future runtime sprint should first isolate pure tensor kernels
with stable tensor signatures, likely individual exact losses or
precomputed-shape kernels, and avoid compiling the dynamic spatial-hash,
cluster-cache, and Python object orchestration layers.
