# Sprint-FIDELITY-SFDP Result

## Summary

Closed `93.8%` of the measured top-5 `classic_sfdp` vs `graphviz_sfdp`
fidelity gap on the 20-graph side-by-side harness.

Root cause: the dagua SFDP port omitted Graphviz's late `pcp_rotate`
principal-component canonicalization from `lib/sfdpgen/spring_electrical.c`.
Because SFDP's force objective is rotation/reflection invariant, equivalent
layouts often landed upside down or on the wrong principal axis in dagua's
y-down coordinate system. The largest gaps were directed-depth penalties, not
force-law failures.

The fix adds a Graphviz-style final orientation pass:

- evaluate the raw refined SFDP layout;
- evaluate the Graphviz-style principal-component rotation;
- consider reflection along the requested flow axis;
- keep the candidate with strongest normalized edge advance for `TB`/`BT`/`LR`/`RL`.

`classic_sfdp` now also receives `graph.direction` through the competitor
adapter, matching the way the quality metrics score directed flow.

## Empirical Result

Comparison harness: 20 representative graphs, `seed=42`, quick composite
metrics, side-by-side against installed Graphviz `sfdp` 8.0.3.

Original top-5 positive gaps:

| graph | before gap | after gap |
|---|---:|---:|
| `grid_5x5` | 18.935 | 0.000 |
| `triangular_lattice_36` | 15.041 | 0.079 |
| `deep_chain_20` | 9.288 | 0.000 |
| `hexagonal_lattice_42` | 8.608 | 0.000 |
| `real_karate_34` | 6.844 | 3.533 |

Mean top-5 positive gap: `11.743 -> 0.722`, closing `93.8%`.

Same-process regression check compared the new default to an emulated old
raw-normalize finalization over the same 20 generated graph objects:

```text
worst_new_minus_old=-0.094
```

No measured graph regressed by `>= 1.0` composite.

## Validation

Passed:

```bash
black --check dagua/layout/ops/sfdp.py dagua/layout/ops/pipelines/sfdp.py dagua/layout/_archive/classic/sfdp.py dagua/eval/competitors/classic_competitor.py tests/test_pipeline_sfdp.py
ruff check dagua/layout/ops/sfdp.py dagua/layout/ops/pipelines/sfdp.py dagua/layout/_archive/classic/sfdp.py dagua/eval/competitors/classic_competitor.py tests/test_pipeline_sfdp.py
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_pipeline_sfdp.py -x --tb=short -q
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

Targeted layout result:

```text
258 passed, 1 warning in 1159.07s (0:19:19)
```

Blocked by unrelated pre-existing repo state:

```bash
ruff check . --fix
```

failed on untracked scripts under `scripts/` with line-length errors:

```text
scripts/cleanup_for_salvage_round.py:95:101 E501
scripts/cleanup_watchdog_errors.py:73:101 E501
scripts/flip_running_to_skipped.py:55:101 E501
scripts/flip_running_to_skipped.py:62:101 E501
scripts/restore_skip3_from_backup.py:94:101 E501
```

Final non-slow pytest tier:

```bash
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

failed during collection on the existing classic package export issue:

```text
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

This is the same unrelated collection blocker recorded in the prior
Sprint-FIDELITY-SGD2 result.

## Changed Files

- `dagua/layout/ops/sfdp.py`
  - Added Graphviz-style principal-component rotation helpers.
  - Added direction-aware reflection/edge-flow selection in finalization.
- `dagua/layout/ops/pipelines/sfdp.py`
  - Added `direction` parameter and forwards it into `LayoutProblem`.
- `dagua/layout/_archive/classic/sfdp.py`
  - Mirrored the pipeline final-orientation behavior so public classic imports
    and exact-fidelity tests stay aligned.
- `dagua/eval/competitors/classic_competitor.py`
  - Forwards `graph.direction` to `layout_sfdp_pipeline`.
  - Adds narrow casts needed to keep the strict CLI mypy gate clean.
- `tests/test_pipeline_sfdp.py`
  - Adds regression coverage for `TB` and `BT` directed-path orientation.

## Follow-Up

The remaining positive gaps are small and concentrated on weighted/social
graphs (`scale_free_ba_120`, `real_karate_34`). A separate sprint should inspect
Graphviz's overlap/packing and weighted-edge handling before changing the force
law.
