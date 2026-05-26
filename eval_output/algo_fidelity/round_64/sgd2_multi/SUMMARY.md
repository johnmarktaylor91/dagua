# Round 64: sgd2_multi Real Port

## Result

Replaced the runtime delegation in `dagua/layout/ops/pipelines/sgd2_multi.py`
with the existing native GD2 ops pipeline. The layout path no longer imports
the stress-only package or the measurement-only competitor adapter.

## Reference Notes

Read `/tmp/graph-drawing/gd2.py`, `/tmp/graph-drawing/criteria.py`, and the
installed stress-SGD package for context. The multicriteria reference:

- seeds `torch`, `numpy`, and Python `random`
- initializes positions as `sqrt(N) * torch.randn([N, 2])`
- constructs the crossing detector before shuffled mini-batch iteration
- uses shuffled `DataLoader` epochs with a smaller final batch when needed
- computes stress with all unordered node pairs and weights `1 / (D^2 + 1e-6)`
- optimizes with Nesterov SGD, gradient clamp, and ReduceLROnPlateau cooling
- evaluates aspect ratio on the sampled node batch via SVD and BCE

## Verification

Spot checks against `SGD2MultiRef` on path, cycle, diamond, and K4 graphs:

- stress-only RMSD: max `6.29e-08`
- stress + ideal edge length RMSD: max `3.48e-07`
- stress + aspect ratio RMSD: max `7.18e-08`
- stress + crossings RMSD: `2.08e-07`
- stress + crossing-angle RMSD: `0.0`

Forbidden-pattern check:

```text
rg -n "import s_gd2|from dagua\\.eval\\.competitors|subprocess" \
  dagua/layout/ops/pipelines/sgd2_multi.py dagua/layout/ops/sgd2_multi.py
```

returned no matches.

## Residual

No fidelity residual above `1e-3` was observed in the smoke comparisons run for
this patch.
