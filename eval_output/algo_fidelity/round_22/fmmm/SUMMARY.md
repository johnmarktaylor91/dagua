# Round 22 FMMM Summary

## Changes

- Added an opt-in `reference_mode` to `layout_fmmm_pipeline`.
- Implemented the top three Round 21 levers behind that mode:
  - lower-star-mass galaxy selection, matching OGDF `NonUniformProbLowerMass`
    (`ROUND_21_DIFF_fmmm.md:431-434`, `ROUND_21_DIFF_fmmm.md:493-495`);
  - OGDF-style random coarsest placement instead of FR initialization
    (`ROUND_21_DIFF_fmmm.md:436-439`, `ROUND_21_DIFF_fmmm.md:497-500`);
  - OGDF-compatible force scaling with average edge length squared,
    `forceScalingFactor=0.05`, oscillation damping, and threshold stop
    (`ROUND_21_DIFF_fmmm.md:441-445`, `ROUND_21_DIFF_fmmm.md:502-505`).
- Added focused regression tests in `tests/test_layout/test_fmmm_fidelity.py`.

## Measurement

Baseline:

```text
graphs: 5
median: 0.056231
p25: 0.041064
p75: 0.099737
p95: 0.218034
worst: parallel_multiedge_bundle 0.247608
```

After, with `reference_mode` left opt-in:

```text
graphs: 5
median: 0.056231
p25: 0.041064
p75: 0.099737
p95: 0.218034
worst: parallel_multiedge_bundle 0.247608
```

Forcing `classic_fmmm` through `reference_mode` worsened this subset
(`median: 0.132290`), so the mode remains opt-in. The relaxed commit criterion
is met by the clean opt-in fidelity flag plus regression tests.

## Verification

Passed:

```text
ruff check dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py tests/test_layout/test_fmmm_fidelity.py --fix
All checks passed!

pytest tests/test_layout/test_fmmm_fidelity.py -x --tb=short -q
4 passed in 0.02s

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Blocked by unrelated dirty-worktree failures:

```text
pytest tests/test_layout/ -x --tb=short -q -k "fmmm"
ERROR tests/test_layout/test_gem_fidelity.py
ImportError: cannot import name '_glibc_rand_values' from 'dagua.layout.ops.gem'
```

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
FAILED tests/test_layout/test_fr_fidelity.py::test_classic_fr_competitor_uses_strict_networkx_path
AssertionError: Tensor-likes are not close!
1 failed, 186 passed, 5 warnings in 1230.76s
```
