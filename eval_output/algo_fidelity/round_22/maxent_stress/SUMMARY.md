# Round 22 maxent_stress Summary

## Changes

- Applied the top recommended small bundle from `ROUND_21_DIFF_maxent_stress.md`:
  runner precision (`scripts/ogdf_runner.cpp:232-240`), non-entropy step variants
  on majorization (`dagua/layout/ops/pipelines/maxent_stress.py:131-142`), and
  OGDF stress iteration wiring (`scripts/ogdf_runner.cpp:159-162`,
  `dagua/eval/variants.py:1068-1088`).
- Also applied the recommended float64 majorization-distance cleanup from
  `ROUND_21_DIFF_maxent_stress.md:621-629`.

## Measurement

- Baseline subset: median `0.000000`, p95 `0.000106`, worst
  `mixed_width_labels 0.000133`.
- After subset: median `0.000000`, p95 `0.000106`, worst
  `mixed_width_labels 0.000133`.

## Commit Criterion

Median was already effectively zero, so numerical improvement was not expected on
the default live subset. This round qualifies under the opt-in fidelity
infrastructure criterion because step-variant OGDF iterations are now plumbed
through variant params and covered by regression tests.

## Tests

- `ruff check <touched python files> --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "maxent_stress"`: passed,
  `5 passed, 286 deselected`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: failed on
  unrelated `FRFinalizePositionsConfig.__init__()` keyword
  `scale_by_sqrt_num_nodes`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection on unrelated missing `layout_drl` import from
  `dagua.layout.classic`.
- Full `ruff check .` failed on unrelated line length in
  `tests/test_layout/test_fr_fidelity.py:32`.
- Rebuilding `scripts/ogdf_runner` failed because OGDF headers were not available
  through local `pkg-config`.
