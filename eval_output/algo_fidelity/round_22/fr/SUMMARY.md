# Round 22 FR Summary

## Changes

- Added `networkx_compat` finalization for FR so fidelity runs can use constant NetworkX-adapter scale instead of dagua's legacy `50 * sqrt(N)` scale.
- Pointed `classic_fr` fidelity adapter at direct `layout_fr_pipeline(steps=50, networkx_compat=True)` to bypass the default 200/50 selector.
- Forced `nx_spring` adapter defaults to `method="force"` so large-graph comparisons remain FR-force comparisons instead of NetworkX auto energy mode.
- Added FR regression coverage for dense NetworkX parity, the `ClassicFR` adapter path, and weighted duplicate edge last-write semantics.

## Spec References

- Finalization mode: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fr.md:336-340`, `:394-396`.
- Strict direct 50-step path: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fr.md:342-346`, `:398-400`.
- Large-graph force semantics: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fr.md:348-352`, `:402-405`.
- Regression tests: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fr.md:407-410`.

## Measurements

- Baseline command: `python scripts/algo_fidelity_live_compare.py classic_fr nx_spring --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_22/fr/baseline`
- Baseline median: `0.148850`.
- After command: same command with `--output-dir eval_output/algo_fidelity/round_22/fr/after`.
- After median: `0.148850`.
- TOST tier: unchanged; all five subset graphs remained `equivalent_at_1x`.

## Gate Decision

Median did not improve by `>= 0.03`, and aggregate TOST did not move up. The bundle meets the relaxed commit criterion because it adds a clean opt-in `networkx_compat` fidelity path with regression tests.

## Verification

- `pytest tests/test_layout/test_fr_fidelity.py -x --tb=short -q`: passed, `3 passed`.
- `pytest tests/test_layout/ -x --tb=short -q -k "fr"`: passed earlier, then later blocked by unrelated untracked `tests/test_layout/test_fmmm_fidelity.py` importing missing `_GALAXY_CHOICE_LOWER`.
- `ruff check dagua/layout/ops/pipelines/fr.py dagua/layout/ops/postprocess.py dagua/eval/competitors/classic_competitor.py dagua/eval/competitors/networkx_competitor.py tests/test_layout/test_fr_fidelity.py`: passed.
- `ruff check . --fix`: passed before the final adapter re-application; targeted lint passed after.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- Final tier command `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked by pre-existing import error in `tests/test_classic_drl.py` (`layout_drl` missing from `dagua.layout.classic`).
