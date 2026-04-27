# Sprint-FIDELITY-FR Result

## Summary

Closed `96.1%` of the measured top-5 positive `classic_fr` fidelity gap to
external `nx_spring` on the 20-graph side-by-side comparison.

The root cause was a default-parameter mismatch: NetworkX `spring_layout` runs
the dense Fruchterman-Reingold solver for `iterations=50`, while dagua's
`classic_fr` benchmark adapter forced `steps=200`. The low-level force equation,
NumPy random initialization, temperature schedule, and distance clamps already
match the canonical NetworkX implementation closely enough that a 50-step
candidate lands on the `nx_spring` composite score for most graphs.

The fix keeps explicit `layout_fr_pipeline(..., steps=N)` calls exact. Only the
`classic_fr` default path now evaluates both candidates:

- legacy dagua default: 200 FR steps
- canonical NetworkX-style default: 50 FR steps

It selects the canonical candidate only when it does not reduce TB DAG
consistency by more than `0.1` and does not reduce the cheap Tier-1 directed
composite score. This closes fidelity gaps while preserving the old 200-step
layout on graphs where it was already better than NetworkX.

## Empirical Result

Comparison command:

```bash
python -u /tmp/sprint_fidelity_fr_compare.py
```

The script compared `nx_spring`, old raw `layout_fr_pipeline(..., steps=200)`,
and new `ClassicFR().layout(...)` on 20 representative graphs with deterministic
Tier-1 metrics plus sampled stress, sampled crossing rate, and angular
resolution.

Top-5 positive gaps:

| graph | before delta | after delta |
|---|---:|---:|
| `linear_3layer_mlp` | 36.137 | 0.000 |
| `clustered_medium_5x20` | 7.131 | 0.000 |
| `real_karate_34` | 3.729 | 2.083 |
| `residual_block` | 3.551 | 0.000 |
| `sparse_pair_50` | 2.484 | -0.000 |

Mean top-5 positive gap: `10.606 -> 0.417`, closing `96.1%` of the measured
target gap.

Regression check over the same 20 graphs compared the new default against the
prior raw 200-step default:

```text
worst_new_minus_old=0.000
```

No measured graph regressed below the prior 200-step composite score.

## Validation

Passed:

```bash
ruff format dagua/layout/ops/pipelines/fr.py dagua/eval/competitors/classic_competitor.py tests/test_pipeline_fr.py
ruff check dagua/layout/ops/pipelines/fr.py dagua/eval/competitors/classic_competitor.py tests/test_pipeline_fr.py --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_pipeline_fr.py -q --tb=short
pytest tests/test_classic_competitor.py::test_classic_fr_seed_override_changes_layout -q --tb=short
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

Targeted layout result:

```text
258 passed, 1 warning in 1293.65s (0:21:33)
```

Blocked by unrelated pre-existing repo state:

```bash
ruff check . --fix
```

failed on untracked cleanup scripts under `scripts/` with `E501` line-length
errors.

```bash
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

failed during collection:

```text
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

The failed full-suite collection path is unrelated to the FR files changed here.

Formatter note: the first commit attempt ran the repository pre-commit hooks and
`ruff-format` rewrote the touched Python files. After that hook, default Black
still wanted to reformat those files differently, so the committed tree follows
the repo's enforced `ruff-format` output.

## Changed Files

- `dagua/layout/ops/pipelines/fr.py`
  - Added default-candidate constants and selector helpers.
  - Added `layout_fr_default_pipeline()` for the benchmark default path.
  - Preserved exact requested-step behavior in `layout_fr_pipeline()`.
- `dagua/eval/competitors/classic_competitor.py`
  - Pointed `classic_fr` default dispatch and direct adapter path at
    `layout_fr_default_pipeline()`.
- `tests/test_pipeline_fr.py`
  - Added selector regression coverage.
  - Added coverage that explicit non-default step counts bypass the selector.

## Follow-Up

The selector intentionally uses cheap deterministic Tier-1 scoring. A deeper
future pass could make `ClassicFR.layout()` pass edge weights in the direct
adapter path, matching the generic `_quick_classic()` adapter behavior for
weighted graphs, but that is a separate divergence from the iteration-count
root cause fixed here.
