# Sprint-X3 No-Fix Report

Date: 2026-04-26
Branch: `codex/sprint-31a-gate-refinement`
Baseline: git HEAD `26acab0`

## Outcome

No fix shipped. The strict top-3 cheap-proxy finalist filter failed the protected
win gate before the full 30-graph validation completed.

Failing protected row observed during `/tmp/sprint_x3_validate.py`:

| graph | baseline | cheap-proxy top-3 | delta | gate |
|---|---:|---:|---:|---|
| `hub_fanout_label_skew` | 93.737 | 93.053 | -0.684 | fail (`abs(delta) > 0.5`) |

The run was stopped after this failure was clear. Earlier protected rows were:

| graph | baseline | cheap-proxy top-3 | delta |
|---|---:|---:|---:|
| `ba_500` | 63.138 | 62.784 | -0.354 |
| `deep_chain_20` | 97.500 | 97.500 | +0.000 |
| `org_chart_deep` | 92.830 | 92.665 | -0.165 |
| `random_dag_200` | 73.920 | 73.935 | +0.015 |

Runtime was promising on the largest protected/heavy rows (`ba_500` and
`dependency_500` were roughly 2x faster), but the protected composite regression
means the sprint cannot ship under the stated contract.

## What Was Tried

Implemented locally, then reverted:

- `_cheap_proxy_score(pos, edge_index, node_sizes)` in
  `dagua/layout/ops/pipelines/dagua_native.py`
- proxy scoring for all `_best_of_polish` candidates
- top-3 full-composite finalist scoring
- focused tests for the proxy formula and full-scoring cap
- `/tmp/sprint_x3_validate.py` to compare detached HEAD vs current working tree

Validation and probes:

- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  passed: `260 passed, 1 warning in 839.20s`
- `mypy --follow-imports=silent dagua/cli.py` passed
- changed files were ruff clean and formatted
- `ruff check . --fix` was blocked by pre-existing untracked scripts with line
  length errors under `scripts/`, unrelated to this sprint

## Root Cause

The cheap proxy ranks the protected winner too low on
`hub_fanout_label_skew`. A one-off audit of candidate rankings showed:

- full-composite best: `y_layer_snap_after_edge_equalize_5_0.05`
- proxy rank for that candidate: 16
- top-3 proxy candidates all scored `93.053` full composite, below the
  protected baseline by `0.684`

Raising finalist count recovered quality only when the filter became much less
aggressive:

| finalist count | score |
|---:|---:|
| 3 | 93.053 |
| 4 | 93.053 |
| 5 | 93.053 |
| 8 | 93.340 |
| 20 | 93.737 |

That indicates this is not just a top-3 boundary issue. The cheap terms are
nearly tied across several candidates, while the full winner gains on terms the
proxy intentionally omits.

## Decision

Do not ship Sprint-X3 as specified. The strict cheap-proxy top-3 filter is not
composite-neutral on a protected graph, and widening the finalist count enough
to recover quality undermines the intended "13 of 16 full evaluations removed"
mechanism.

No commit was made.
