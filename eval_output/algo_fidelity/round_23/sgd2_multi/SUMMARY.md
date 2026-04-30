# Round 23 sgd2_multi Summary

Pairing: `classic_sgd2_multi` vs `sgd2_multi_ref`.

## Measurement

- Baseline command: `python scripts/algo_fidelity_live_compare.py classic_sgd2_multi sgd2_multi_ref --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_23/sgd2_multi/baseline`
- Baseline median: `0.10648379474878311`
- Post-fix command: `python scripts/algo_fidelity_live_compare.py classic_sgd2_multi sgd2_multi_ref --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_23/sgd2_multi/post_fix`
- Post-fix median: `0.10648379474878311`
- Delta: `0.0`
- Worst graph after fix: `tl_mlp_3layer` at `0.11052870005369186`

The live comparison uses cached `sgd2_multi_ref` target coordinates, so adapter-only reproducibility fixes do not move this metric until reference outputs are regenerated.

## Ranked Items

1. Restore/vendor actual upstream `GD2` files.
   - Status: partially addressed in commit `1760d312083d615fd6dc0944de3d6585a6636671`.
   - Fix size: about 34 lines in `dagua/eval/competitors/sgd2_multi_competitor.py`.
   - Change: availability now requires both `gd2.py` and `criteria.py`; layout returns a clear missing-source diagnostic before import.
   - Deferred part: vendoring or pinning the actual upstream files was not technically feasible inside the <200 line round scope because the referenced upstream repository currently lacks those modules.

2. Align RNG sources.
   - Status: addressed in commit `1760d312083d615fd6dc0944de3d6585a6636671`.
   - Fix size: 2 net source lines plus regression coverage.
   - Change: `SGD2MultiRef.layout_with_variant` seeds Python `random` alongside torch and NumPy.

3. Investigate/align sampler semantics.
   - Status: skipped.
   - Reason: Round 21 and Round 22 both found this depends on the missing upstream `GD2.optimize` implementation. Implementing a guessed sampler would be speculative.

4. Fix crossing no-pair objective mismatch.
   - Status: addressed in commit `1760d312083d615fd6dc0944de3d6585a6636671`.
   - Fix size: about 56 source lines.
   - Change: the native `sgd2_multi` ops pipeline strips crossing-based schedules when no non-incident edge pairs exist and falls back to stress-only when crossing was the only active objective.

5. Align weighted/multiedge adjacency semantics.
   - Status: addressed in commit `1760d312083d615fd6dc0944de3d6585a6636671`.
   - Fix size: about 40 source lines.
   - Change: weighted duplicate undirected edges now use the minimum parallel weight locally in `sgd2_multi`, matching the archived classic distance helper instead of shared additive adjacency.

6. Align aspect-ratio SVD/clamp and sampler semantics.
   - Status: skipped.
   - Reason: changing dagua's clamp to match the patched reference would remove the boundary guard that prevents BCE singularities. Sampler semantics still require the missing upstream optimizer source.

7. Remove/gate public `s_gd2` fallback during fidelity comparison.
   - Status: addressed in commit `1760d312083d615fd6dc0944de3d6585a6636671`.
   - Fix size: 7 source lines.
   - Change: `layout_sgd2_multi_pipeline` now requires explicit `use_reference_fallback=True` before substituting the optional canonical `s_gd2` backend.

8. Align or document direct defaults.
   - Status: skipped.
   - Reason: changing public defaults from `steps=10000`, `lr=1.0`, `batch_size=16`, `grad_clamp=4.0` to reference-adapter defaults would be a broad behavior change outside the fidelity variant surface. The canonical reference-aligned entrypoint remains the benchmark variant configuration.

## Verification

- `ruff check dagua/eval/competitors/sgd2_multi_competitor.py dagua/layout/ops/sgd2_multi.py dagua/layout/ops/pipelines/sgd2_multi.py tests/test_layout/test_sgd2_multi_fidelity.py --fix`: passed.
- `ruff check . --fix --diff`: passed with no diff output.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_sgd2_multi_fidelity.py -q`: `4 passed in 0.93s`.
- `pytest tests/test_layout/ -x --tb=short -q -k "sgd2_multi"` before commit: `4 passed, 324 deselected in 1.11s`.
- `pytest tests/test_layout/ -x --tb=short -q -k "sgd2_multi"` after commit: `4 passed, 332 deselected in 1.15s`.

## Commit Note

The `sgd2_multi` changes landed in commit `1760d312083d615fd6dc0944de3d6585a6636671`, which also includes pre-existing staged `umap` changes from the shared dirty workspace. I did not intentionally edit the `umap` files. I did not rewrite that mixed commit to avoid disturbing parallel work.

## Dead Code

No newly unreachable code was identified. The optional `s_gd2` fallback remains reachable only through the new explicit `use_reference_fallback=True` flag.
