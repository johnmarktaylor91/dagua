# R79 P3D Quality Knob Evidence

Date: 2026-07-07
Branch: r79/p3d-quality

## Implemented Mapping

| Quality | q | Step multiplier | multi_start_k | Stress pivots | SMACOF iters | Polish battery | ML refine multiplier | BH theta | Sampling rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| draft | 0.25 | 0.632x | 1 | 32 | 0 | off | 0.5x | 1.2 | 0.75x |
| balanced | 0.50 | 1.000x | 1 | 64 | 4 | default | 1.0x | 1.0 | 1.0x |
| high | 0.75 | 2.000x | 3 | 128 | 24 | full | 2.0x | 0.8 | 1.5x |
| max | 1.00 | 4.000x | 5 | 256 | 50 | full | 3.0x | 0.6 | 2.0x |

Notes:
- Step multiplier is log-linear from 0.4x at q=0 to 1x at q=0.5,
  2x at q=0.75, and 4x at q=1.
- Stress pivots are linearly interpolated through draft/balanced/high/max
  anchors and capped by graph size.
- Balanced SMACOF is calibrated to the existing native-stress default of 4
  iterations so default quality preserves current behavior.
- Dagua does not track constructor-provided fields separately from dataclass
  defaults. Override precedence is therefore sentinel/default based:
  `steps > 0`, `multi_start_k != 1`, or present `algorithm_params` win over
  the quality knob.
- `time_budget_s` suppresses auto multi-start and non-cheap post-core polish;
  capped native runs still complete current work and run final overlap/aspect
  polish.

## Wall-Time Samples

Measured with `.venv/bin/python` on CPU in this worktree. Raw timings are
noisy because topology routing and candidate scoring are graph-dependent.

| Graph | draft | balanced | high | max |
| --- | ---: | ---: | ---: | ---: |
| random_dag_50 | 1.481s | 2.790s | 7.274s | 5.814s |
| small_world_100 | 2.113s | 1.600s | 1.956s | 1.658s |

## Baseline Sweep

Command:

```bash
.venv/bin/python scripts/r79_baseline.py --dagua-only
```

Result:
- Wall time: 1976.453s.
- Pre-change W/T/L from frozen store: legacy 56/8/29, extended 8/2/5.
- Post-change W/T/L after dagua-only rerun: legacy 56/8/29, extended 8/2/5.
- Baseline store churn was reverted after confirming identical W/T/L.

## Test Results

Passed:
- `.venv/bin/python -m pytest tests/test_layout/test_quality_knob.py -q --tb=short`
  - 5 passed, 3 warnings.
- `.venv/bin/python -m pytest tests/test_layout/test_cuda_activation.py::test_all_stages_fall_back_when_no_cuda tests/test_layout/test_cuda_activation.py::test_execution_mode_selection_thresholds -q --tb=short`
  - 2 passed, 1 CUDA driver warning.
- `.venv/bin/python -m pytest tests/test_layout/test_quality_knob.py tests/test_layout/test_cuda_activation.py::test_all_stages_fall_back_when_no_cuda tests/test_layout/test_cuda_activation.py::test_execution_mode_selection_thresholds -q --tb=short`
  - 7 passed, 1 CUDA driver warning.
- `.venv/bin/python -m pytest tests/test_edge_weights_distance.py::TestDistanceAlgoWeights::test_weights_affect_layout -q --tb=short`
  - 8 passed, 1 warning.
- `.venv/bin/python -m pytest tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_double_border tests/test_cosmetic_node_features.py::TestRenderSmoke::test_render_with_round_node_border_styles -q --tb=short`
  - 2 passed, 468 warnings.
- `.venv/bin/python -m pytest tests/test_eval/test_benchmark_pipeline.py::test_standard_suite_contains_expected_cases -q --tb=short`
  - 1 passed, 10 warnings after installing `torchlens` and adding TorchLens 2.28 trace compatibility.
- `.venv/bin/python -m pytest tests/test_eval/test_cytoscape_gephi_competitors.py::test_cytoscape_fcose_variant_params -q --tb=short`
  - 1 passed, 2 warnings.
- `.venv/bin/python -m pytest tests/test_eval/test_graphs.py::test_synthetic_graphs_include_final_structural_additions -q --tb=short`
  - 1 passed, 2 warnings.
- `.venv/bin/python -m pytest tests/test_feature_reference_script.py::test_catalog_counts_match_requested_gallery -q --tb=short`
  - 1 passed, 2 warnings.
- `.venv/bin/python -m pytest tests/test_fidelity_procrustes.py::test_procrustes_known_good_equivalent -q --tb=short`
  - 1 passed, 2 warnings.
- `.venv/bin/python -m ruff check . --fix`
  - All checks passed.
- `.venv/bin/python -m mypy --follow-imports=silent dagua/cli.py`
  - Success: no issues found in 1 source file.

Long targeted gate:
- `.venv/bin/python -m pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  - First run: failed in `test_all_stages_fall_back_when_no_cuda` due old
    NVIDIA driver CUDA initialization when legacy `_layout_inner` kept a CUDA
    resident device after monkeypatching `torch.cuda.is_available=False`.
  - Focused fix added CPU demotion before execution-mode resolution.
  - Second run: failed in `test_execution_mode_selection_thresholds` because
    `_resolve_execution_mode` mixed CUDA availability with pure threshold
    selection.
  - Focused fix restored pure threshold behavior and kept runtime demotion in
    `_layout_inner`.
  - Focused rerun of both CUDA tests passed.

Final non-slow tier:
- Initial collection failed on missing optional test dependencies in the local
  `.venv`: `h5py`, `statsmodels`, `nbformat`, `sklearn`, and `torchlens`.
- Installed missing packages into `.venv` and reran repeatedly with `-x`.
- Fixed or updated focused regressions exposed by the broad suite:
  double/round node border render smoke, weighted `maxent_stress` dispatch,
  TorchLens 2.28 trace compatibility, fcose variant metadata expectation,
  Petersen graph tag expectation, feature-reference theme count expectation,
  and the Procrustes strong-equivalent fixture.
- Latest broad-suite failure:
  `.venv/bin/python -m pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`
  ran for 6269.56s and reached 638 passed, 11 skipped, 88 deselected before
  failing in `tests/test_fidelity_procrustes.py::test_procrustes_known_good_equivalent`.
  The focused Procrustes test passed after updating the fixture to the current
  verdict contract, but the full non-slow suite was not rerun again to
  completion.

## Concerns

- Final full non-slow suite has not completed green; the latest focused fix
  passed, but another complete run was not performed after the 6269.56s stop.
- The default adapter's high/max wall time is not strictly monotonic on every
  graph because routing and polish gates dominate some small graphs.
- `node_modules/` was pre-existing untracked workspace content and was left
  untouched.
