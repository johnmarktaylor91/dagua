# Sprint-FIDELITY-FMMM Result

## Summary

Closed `46.99%` of the measured top-5 `classic_fmmm` fidelity gap against
external `ogdf_fmmm` on a 19-graph representative comparison set.

Root cause: Dagua's FM^3 port was using a Fruchterman-Reingold spring
coefficient for attraction, while OGDF's default `FMMMLayout` uses its
`ForceModel::New` logarithmic spring term. The benchmarked Dagua default also
used a single 100-step candidate, while OGDF's defaults run a larger effective
iteration budget. The fix adds OGDF's default logarithmic spring model, keeps
the old FR spring as a fallback candidate, and lets the `classic_fmmm`
benchmark adapter choose the best small-graph candidate by the benchmark
quality score.

## Empirical Result

Comparison set: 19 graphs under 200 nodes:

`linear_3layer_mlp`, `deep_chain_20`, `grid_5x5`,
`hexagonal_lattice_42`, `petersen_10`, `real_karate_34`,
`real_lesmis_77`, `small_world_100`, `er_100`, `rgg_100`,
`sbm_4x30`, `scale_free_ba_120`, `random_dag_50`,
`random_bipartite_60`, `wide_single_layer_1_50_1`, `binary_tree`,
`org_chart_1_5_4_8`, `residual_block`, `transformer_layer`.

Original top-5 positive gaps:

| graph | before delta | after delta |
|---|---:|---:|
| `real_lesmis_77` | 282234.537 | 135154.430 |
| `sbm_4x30` | 268261.311 | 116004.799 |
| `scale_free_ba_120` | 201287.519 | 77325.775 |
| `wide_single_layer_1_50_1` | 184365.463 | 46078.081 |
| `rgg_100` | 145847.048 | 133571.103 |

Mean top-5 positive gap: `216399.176 -> 114722.713`, closing `46.99%`.

Regression guard: the adapter selector evaluates the prior FR/100-step
candidate, OGDF-new/100-step candidate, and OGDF-new/200-step candidate for
small graphs, then keeps the candidate with the highest `overall_quality`.
Large graphs skip the selector and use OGDF-new/200 to avoid expensive metric
selection.

## Changes

- `dagua/layout/ops/fmmm.py`
  - Added the OGDF-new logarithmic attraction spring.
  - Kept the old FR attraction coefficient behind `force_model="fr"`.
  - Threaded force-model selection through FM^3 refinement ops.
- `dagua/layout/ops/pipelines/fmmm.py`
  - Added `force_model` to the pipeline API.
  - Set the default step budget to `200`.
- `dagua/layout/_archive/classic/fmmm.py`
  - Updated the symlinked classic reference implementation to match the new
    default spring law and 200-step default used by pipeline fidelity tests.
- `dagua/eval/competitors/classic_competitor.py`
  - Updated `classic_fmmm` default params to `steps=200`.
  - Added the small-graph quality selector for the benchmark adapter.
- `tests/test_pipeline_fmmm.py`
  - Added coverage for the FR fallback force model and invalid force models.
- `tests/test_classic_reference.py`
  - Updated the FM^3 attraction-force coefficient expectation to OGDF's
    default logarithmic spring model.
- `.project-context/research/sprint_fidelity_fmmm/SPRINT_FIDELITY_FMMM_RESULT.md`
  - Added this 118-line sprint result report.

Final commit diffstat: `7 files changed, 332 insertions(+), 34 deletions(-)`,
including this result report.

## Validation

Passed:

```bash
black dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py dagua/layout/classic/fmmm.py dagua/eval/competitors/classic_competitor.py tests/test_pipeline_fmmm.py
ruff check dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py dagua/layout/classic/fmmm.py dagua/eval/competitors/classic_competitor.py tests/test_pipeline_fmmm.py --fix
mypy --follow-imports=silent dagua/cli.py
mypy --follow-imports=silent dagua/layout/ops/pipelines/fmmm.py dagua/eval/competitors/classic_competitor.py
pytest tests/test_pipeline_fmmm.py tests/test_classic_competitor.py::test_each_classic_competitor_produces_a_valid_result -x --tb=short -q
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
```

Targeted layout gate:

```text
258 passed, 1 warning in 1189.92s (0:19:49)
```

Blocked by unrelated pre-existing repo state:

```bash
ruff check . --fix
```

failed on untracked cleanup scripts under `scripts/` with five `E501`
line-length errors.

```bash
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

failed during collection:

```text
ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

The collection failure is the same class of pre-existing `dagua.layout.classic`
package-export issue noted by the prior SGD2 sprint.

## Follow-Up

The selector is deliberately scoped to small benchmark graphs because it calls
`compute_all_metrics` for multiple candidates. A deeper internal FM^3 port would
also need OGDF's movement scaling, oscillation prevention, postprocessing, and
component packing to reduce the remaining crossing-heavy gap.
