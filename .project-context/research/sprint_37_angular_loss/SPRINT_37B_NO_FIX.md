# Sprint-37b No-Fix Report

Date: 2026-04-26
Branch: codex/sprint-31a-gate-refinement
Baseline: git HEAD 3eaa01c

## Outcome

No fix shipped. I implemented the requested `AngularResolutionLoss` as a
registered loss op with undirected incident-edge soft margins, wired it through
the resolved native loss stack, added the `w_angular_resolution` config and late
annealing ramp, and validated it against a `git archive` checkout of HEAD via
`/tmp/sprint37b_validate.py`.

The empirical gate failed: the new loss produced no measurable angular
resolution movement on the six target graphs.

## Validation Summary

Target graph deltas:

| graph | angular_res_mean_deg delta | composite delta |
|---|---:|---:|
| rgg_100 | +0.000 | +0.000 |
| dense_pair_50 | +0.000 | +0.000 |
| real_lesmis_77 | +0.000 | +0.000 |
| real_football_115 | +0.000 | +0.000 |
| sbm_4x30 | +0.000 | +0.000 |
| ba_500 | -0.000 | -0.000 |

Protected graph deltas:

| graph | avg_degree | composite delta |
|---|---:|---:|
| deep_chain_20 | 1.909 | +0.000 |
| random_dag_200 | 1.567 | +0.225 |
| ba_500 | 5.976 | -0.000 |
| org_chart_deep | 1.975 | +0.000 |
| hub_fanout_label_skew | 2.600 | +0.000 |

Jitter deltas:

| graph | angular mean/std/min/max | composite mean/std/min/max |
|---|---:|---:|
| rgg_100 | +0.000 / 0.000 / +0.000 / +0.000 | +0.000 / 0.000 / +0.000 / +0.000 |
| real_football_115 | +0.000 / 0.000 / +0.000 / +0.000 | +0.000 / 0.000 / +0.000 / +0.000 |
| sbm_4x30 | +0.000 / 0.000 / +0.000 / +0.000 | +0.000 / 0.000 / +0.000 / +0.000 |

Failed pass conditions:

- Target angular improvements were `0/6` at the required `>= 1.0` degree lift
  threshold, below the required `4/6`.
- All three jitter graphs had angular mean delta equal to zero, not positive.

## Additional Checks

- `ruff check` on the attempted implementation/test files passed.
- Focused unit tests passed:
  `pytest tests/test_ops_loss_engine.py tests/test_ops_anneal.py tests/test_resolve_cyclic_skip.py tests/test_config_defaults.py -q`
  -> `161 passed in 2.06s`.
- Targeted layout gate passed:
  `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  -> `256 passed, 1 warning in 1369.34s`.
- Repo-wide `ruff check . --fix` was blocked by pre-existing untracked
  `scripts/` files with E501 line-length failures.
- `mypy --follow-imports=silent dagua/cli.py` was blocked by existing imported
  op metadata typing errors, primarily `ClassVar` override diagnostics in
  `dagua/layout/ops/anneal.py` and `dagua/layout/ops/loss_engine.py`.

## Notes

- The attempted op was confirmed to be present in the resolved loss stack for
  `rgg_100`.
- A sensitivity check with `w_angular_resolution=0.0`, `0.5`, and `100.0` on a
  fresh `rgg_100` graph produced identical final metrics, so the requested
  soft-margin formulation does not move the default pipeline output in its
  current placement.
- Per the Sprint-37b contract, no commit was made and the attempted code/test
  changes were reverted.
