# Round 22 Residual: `classic_pivot_mds` vs `ogdf_pivot_mds`

The top recommended bundle was implemented as opt-in fidelity infrastructure, but the round was not
committed because verification was blocked by concurrent unrelated edits in the shared worktree.

## Implemented

1. OGDF runner/adapter pivot-count plumbing:
   - `scripts/ogdf_runner.cpp` accepts optional `numberOfPivots`.
   - `dagua/eval/competitors/ogdf_competitor.py` forwards `n_pivots` to runner options.
   - `dagua/eval/variants.py` maps Pivot-MDS variant pivot counts to OGDF original params.

2. OGDF-compatible first pivot:
   - `dagua/layout/ops/distance.py` adds `PivotSelectionConfig.first_pivot`.
   - `first_pivot="first_node"` starts the max-min sweep at node zero.
   - Pivot-MDS variants opt into `first_node`.

3. Float64 internal math:
   - `dagua/layout/ops/distance.py` lets pivot distance queries store `torch.float64`.
   - `dagua/layout/ops/embed.py` lets Pivot-MDS centering/SVD run in a requested dtype.
   - `dagua/layout/ops/pipelines/pivot_mds.py` exposes `compute_dtype`, including string values
     for variant registry use.

4. Regression tests:
   - `tests/test_layout/test_pivot_mds_fidelity.py` covers pivot order, float64 internal state,
     variant mappings, and OGDF runner option forwarding.

## Blockers

- `pytest tests/test_layout/ -x --tb=short -q -k "pivot_mds"` failed during collection before pivot
  tests ran due to an unrelated FA2 test import error:
  `ImportError: cannot import name '_FA2_REFERENCE_PACKAGE_ORDER'`.
- Direct pivot test execution was also blocked by package import failures from unrelated active
  edits in non-pivot ops, including:
  - `dagua/layout/ops/sugiyama.py`: undefined `_SUGIYAMA_ACYCLIC_EDGE_WEIGHTS_KEY`.
  - `dagua/layout/ops/fmmm.py`: undefined `_GALAXY_CHOICE_HIGHER`.
- After re-measure could not run for the same import-time reason.
- `scripts/ogdf_runner.cpp` could not be rebuilt with the simple local `g++` command because
  `ogdf/basic/Graph.h` was not available in the default include path.

## Baseline

Baseline command completed successfully before the unrelated import breakage:

- Rows: 30
- Graphs: 5
- Median: `0.000000`
- p95: `0.072776`
- Worst: `mixed_width_labels 0.090971`

## Next Step

After the concurrent non-pivot edits settle, rerun:

```bash
pytest tests/test_layout/ -x --tb=short -q -k "pivot_mds"
python scripts/algo_fidelity_live_compare.py classic_pivot_mds ogdf_pivot_mds \
    --seeds 3 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_22/pivot_mds/after
```

If the local OGDF build environment is available, rebuild `scripts/ogdf_runner` before the after
comparison so `numberOfPivots` takes effect in subprocess runs.
