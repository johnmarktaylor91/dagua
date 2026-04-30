# Round 23 Pivot-MDS Fidelity Sweep

## Commands

Baseline:

```bash
python scripts/algo_fidelity_live_compare.py classic_pivot_mds ogdf_pivot_mds --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_23/pivot_mds/baseline
```

Post-fix:

```bash
python scripts/algo_fidelity_live_compare.py classic_pivot_mds ogdf_pivot_mds --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_23/pivot_mds/post_fix
```

Post-commit test:

```bash
pytest tests/test_layout/ -x --tb=short -q -k "pivot_mds"
```

## Result

Commit: `01fe62f feat(fidelity): round 23 pivot_mds -- ogdf fidelity controls`

Baseline and post-fix metrics were identical on the bounded 5-graph subset:

| Metric | Baseline | Post-fix |
| --- | ---: | ---: |
| Rows | 30 | 30 |
| Graphs | 5 | 5 |
| Median RMSD | 0.000000 | 0.000000 |
| p95 RMSD | 0.072776 | 0.072776 |
| Worst | mixed_width_labels 0.090971 | mixed_width_labels 0.090971 |

The unchanged post-fix aggregate is expected because the local OGDF source tree is not built, so
`scripts/ogdf_runner` could not be rebuilt to consume the new Pivot-MDS pivot-count option.

## Ranked Items

1. Expose and apply OGDF pivot count in runner and adapter: partially addressed in `01fe62f`.
   `ogdf_competitor.py` forwards `n_pivots` as `numberOfPivots`, and Pivot-MDS variants now set
   original params. Runner-source changes were not committed because rebuild failed:
   `config_autogen.h` and `/home/jtaylor/projects/_references/ogdf/build/lib` were absent.
2. OGDF-compatible pivot-selection mode: addressed in `01fe62f`. Pipeline variants use
   `first_pivot="first_node"`; lower-level `distance.py` support already existed on current HEAD.
3. OGDF path special case: addressed in `01fe62f` with opt-in `ogdf_path_special_case=True`.
4. Float64 internal math: addressed in `01fe62f`; Pivot-MDS can keep pivot distances and SVD
   centering in `float64` until final output cast.
5. OGDF uniform edge-cost scale: addressed in `01fe62f` with opt-in `distance_scale=100.0`.
6. OGDF-style eigensolver: skipped. Estimated larger than 200 net lines and risky without
   dedicated numerical tests; current sweep already remains strong-equivalent under Procrustes.
7. Final normalization modes: partially addressed through raw OGDF path output. General raw-output
   mode skipped because the current comparator is Procrustes-normalized and changing default
   finalization would affect classic adapter behavior.
8. Disconnected graph policy: skipped as intentional incompatibility. OGDF rejects disconnected
   Pivot-MDS graphs; current OK-set aggregation already excludes those failures.
9. ClassicPivotMDS layout vs variant-param execution: audited and covered by regression tests.
   Variant execution goes through `layout_with_variant`; direct `layout()` remains the default
   50-pivot convenience path.
10. Runner seed/initial-position noise: documented as dead for Pivot-MDS; no code change.

## Verification

- `ruff check dagua/layout/ops/pipelines/pivot_mds.py dagua/layout/ops/distance.py dagua/layout/ops/embed.py dagua/eval/competitors/ogdf_competitor.py dagua/eval/variants.py tests/test_layout/test_pivot_mds_fidelity.py --fix` passed.
- `pytest tests/test_layout/ -x --tb=short -q -k "pivot_mds"` passed after commit:
  `5 passed, 329 deselected in 0.26s`.

## Concerns

- Rebuild remains blocked until the OGDF reference tree is configured/built. Without a rebuilt
  runner, `numberOfPivots` is adapter plumbing only for local live comparisons.
- The shared worktree had concurrent Round 23 edits and commits for other families. Commit
  `01fe62f` was created with an isolated index and contains only pivot MDS files.
