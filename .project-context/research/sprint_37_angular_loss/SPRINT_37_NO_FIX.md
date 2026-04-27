# Sprint-37 No-Fix Report

Date: 2026-04-26
Branch: codex/sprint-31a-gate-refinement
Baseline: git HEAD 3eaa01c

## Outcome

No fix shipped. The requested retune was implemented and validated against a
`git archive` checkout of HEAD via `/tmp/sprint37_validate.py`, but the
empirical gate failed because the target angular-resolution lift was effectively
zero.

## Validation Summary

Target graph deltas:

| graph | avg_degree | angular_res_mean_deg delta | composite delta |
|---|---:|---:|---:|
| rgg_100 | 15.100 | +0.000000 | +0.000001 |
| dense_pair_50 | 8.320 | +0.000000 | +0.000000 |
| real_lesmis_77 | 6.597 | +0.000080 | +0.000012 |
| real_football_115 | 11.357 | +0.000014 | +0.000003 |
| sbm_4x30 | 9.417 | -0.000000 | +0.000001 |

Protected graph deltas:

| graph | avg_degree | composite delta |
|---|---:|---:|
| deep_chain_20 | 1.909 | +0.000000 |
| random_dag_200 | 1.567 | -0.018119 |
| ba_500 | 5.976 | +0.000000 |
| org_chart_deep | 1.975 | +0.000000 |
| hub_fanout_label_skew | 2.600 | -0.000004 |

Jitter deltas:

| graph | angular mean/std/min/max | composite mean/std/min/max |
|---|---:|---:|
| rgg_100 | +0.000000 / 0.000000 / +0.000000 / +0.000000 | +0.000000 / 0.000000 / -0.000000 / +0.000000 |
| real_football_115 | +0.000014 / 0.000000 / +0.000014 / +0.000014 | +0.000003 / 0.000002 / -0.000001 / +0.000006 |
| sbm_4x30 | -0.000000 / 0.000000 / -0.000001 / +0.000000 | -0.000000 / 0.000001 / -0.000003 / +0.000001 |

Failed pass conditions:

- Target graphs improved angular resolution on only 3 of 5 graphs, and those
  improvements were numerical noise rather than meaningful degree lift.
- `sbm_4x30` jitter angular mean was negative at numerical-noise scale.

## Notes

- The fanout loss was confirmed to be wired for `rgg_100`, `dense_pair_50`,
  and `ba_500` under the requested `avg_degree >= 3.0` gate.
- `ba_500` is not protected by the average-degree gate; its average degree is
  5.976. It still showed no meaningful composite regression in this validation.
- The requested retune appears too weak, too late, or too aligned with the
  current equilibrium to move the measured layouts. Per the Sprint-37 contract,
  no commit was made.
