# Round 23 FMMM Exhaustive Sweep

## Measurement

Baseline command:

```text
python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_23/fmmm/baseline
```

Baseline result:

```text
graphs: 5
median: 0.056231
p25: 0.041064
p75: 0.099737
p95: 0.218034
worst: parallel_multiedge_bundle 0.247608
```

Final command after reverting the regressed attempt:

```text
python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm --seeds 3 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_23/fmmm/post_fix
```

Final result:

```text
graphs: 5
median: 0.056231
p25: 0.041064
p75: 0.099737
p95: 0.218034
worst: parallel_multiedge_bundle 0.247608
```

## Ranked Items

1. Galaxy choice to OGDF lower-star-mass sampling: already addressed in Round 22 reference mode.
2. OGDF-style coarsest random placement: already addressed in Round 22 reference mode.
3. OGDF force scaling, threshold, and oscillation damping: already addressed in Round 22 reference mode.
4. OGDF postprocessing resize/fine-tune loop: attempted in `a308fd7`; reverted in `06e530f` because the bounded subset regressed from median `0.056231` to `0.287466`.
5. Disconnected-component solve and MAAR-like packing: skipped. Estimated patch is larger than 200 net lines because it requires component extraction, per-component hierarchy solves, rectangle construction/rotation, and deterministic packing.
6. Individual ideal edge lengths with node radii and OGDF unit edge length: skipped. Estimated patch is larger than 200 net lines because FMMM currently receives only tensors, while the OGDF behavior depends on GraphAttributes node radii and unit-edge-length semantics.
7. OGDF NMM behavior or exact cutoff reference mode: cutoff-only reference mode attempted in `a308fd7`; reverted in `06e530f` with the regressed postprocess bundle. True NMM remains skipped as XL.
8. Seed-controllable `ogdf_fmmm` runner: skipped. I attempted the required rebuild, but the OGDF reference tree is missing generated build artifacts (`config_autogen.h`), so runner-side changes were not kept.
9. Integer-position finalization reference mode: attempted in `a308fd7`; reverted in `06e530f` with the regressed bundle.
10. Reference adapter output precision: verified already present in `scripts/ogdf_runner.cpp` via `std::setprecision(17)`; no new code change.

Lower-priority items from the diff:

- Single-level fallback edge-length handling: attempted in `a308fd7`; reverted in `06e530f` with the regressed bundle.
- Coincident-node random perturbation: skipped because it would change default stochastic force behavior and was not separately proven to improve the bounded subset.
- Temperature key cleanup: skipped as code clarity only; no measurable fidelity value identified for this round.

## Verification

Passed before the regressed attempt was committed:

```text
ruff check dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py tests/test_layout/test_fmmm_fidelity.py --fix
All checks passed!

pytest tests/test_layout/test_fmmm_fidelity.py -x --tb=short -q
5 passed in 0.03s

pytest tests/test_layout/ -x --tb=short -q -k "fmmm"
5 passed, 322 deselected in 0.35s
```

Attempted runner rebuild:

```text
g++ -std=c++17 -O2 scripts/ogdf_runner.cpp -I/home/jtaylor/projects/_references/ogdf/include -L/home/jtaylor/projects/_references/ogdf/build/lib -logdf -o scripts/ogdf_runner
fatal error: ogdf/basic/internal/config_autogen.h: No such file or directory
```

Passed after revert:

```text
pytest tests/test_layout/ -x --tb=short -q -k "fmmm"
4 passed, 331 deselected in 0.31s
```

Commit stat checks:

```text
a308fd7^..a308fd7
dagua/layout/ops/fmmm.py                | 124 +++++++++++++++++++++++++++++++-
dagua/layout/ops/pipelines/fmmm.py      |   6 ++
tests/test_layout/test_fmmm_fidelity.py |  20 ++++++

06e530f^..06e530f
dagua/layout/ops/fmmm.py                | 124 +-------------------------------
dagua/layout/ops/pipelines/fmmm.py      |   6 --
tests/test_layout/test_fmmm_fidelity.py |  20 ------
```

## Conclusion

No Round 23 FMMM code change remains active. Every small remaining fix that could be bundled under the line budget was either already covered in Round 22, attempted and reverted due a measured regression, verified already present, or skipped because it exceeded the requested patch-size threshold.
