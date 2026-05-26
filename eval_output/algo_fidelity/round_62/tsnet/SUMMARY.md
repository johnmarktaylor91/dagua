# Round 62 tsNET Delegation Audit

## Verdict

tsNET is a real local implementation for the exact fidelity path. I found no
`sklearn.manifold.TSNE(...)` construction and no `fit_transform` delegation in
`dagua/layout/ops/pipelines/tsnet.py` or `dagua/layout/ops/tsnet.py`.

The only sklearn runtime import in the audited source is:

- `sklearn.manifold._t_sne._joint_probabilities` in
  `dagua/layout/ops/pipelines/tsnet.py`.

Classification: acceptable math primitive. It deterministically converts a
fixed precomputed distance matrix and perplexity into the condensed joint
probability vector `P`; it carries no t-SNE embedding state and performs no
optimization loop.

## Audit

- `sklearn.manifold.TSNE(...)`: not present in either audited file.
- `sklearn.manifold._t_sne._joint_probabilities(...)`: present once and
  documented as a sklearn math primitive in `_fit_tsnet_exact_condensed`.
- `sklearn.manifold._t_sne._kl_divergence(...)`: not imported or called.
  `_kl_divergence_exact` is dagua's local NumPy/SciPy port.
- `sklearn.utils.check_random_state(...)`: not imported or called.
  `_check_random_state` is dagua's local compatibility helper.
- Other sklearn calls: none found.

## Implementation Notes

- Updated docstrings that still implied the public exact path was routing to
  sklearn's `TSNE` estimator. The public exact path now describes the local
  sklearn-compatible port.
- Documented the accepted sklearn math primitive using the required wording:
  `sklearn math primitive used: sklearn.manifold._t_sne._joint_probabilities`.
- Kept exact fidelity's default output dtype aligned with sklearn exact output
  (`float32`) while preserving explicit `fidelity_dtype=torch.float64`.

## Verification

Source grep:

```text
rg -n "sklearn\.manifold\.TSNE\(" dagua/layout/ops/pipelines/tsnet.py dagua/layout/ops/tsnet.py || true
<no output>
```

Smoke vs `sklearn.manifold.TSNE(method="exact")`, 4 topologies x 3 seeds,
perplexity 30, max_iter 300:

```text
path: mean_rmsd=0.000000000000000000e+00 max_rmsd=0.000000000000000000e+00
star: mean_rmsd=0.000000000000000000e+00 max_rmsd=0.000000000000000000e+00
clustered: mean_rmsd=0.000000000000000000e+00 max_rmsd=0.000000000000000000e+00
grid: mean_rmsd=0.000000000000000000e+00 max_rmsd=0.000000000000000000e+00
overall_mean_rmsd=0.000000000000000000e+00
overall_max_rmsd=0.000000000000000000e+00
max_abs_coordinate_diff=0.000000000000000000e+00
```

Targeted checks:

```text
ruff check dagua/layout/ops/pipelines/tsnet.py dagua/layout/ops/tsnet.py --fix
All checks passed!

pytest tests/test_pipeline_tsnet.py -x --tb=short -q
14 passed, 2 warnings in 1.62s

mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file

pytest tests/test_graph.py -x --tb=short -q
37 passed, 2 warnings in 0.60s
```

Broader gate status:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
exited with code -1 and no pytest traceback/output

pytest tests/test_layout/ -x --tb=short -q
stopped after running CPU-bound for more than 20 minutes with no failure output
```

I did not run the final full non-slow suite after the layout gate became
runaway; the TSNET-specific and graph checks above passed.
