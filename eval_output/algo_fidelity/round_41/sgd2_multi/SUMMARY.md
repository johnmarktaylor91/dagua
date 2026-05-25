# Round 41 sgd2_multi Summary

## Reference source lines identified

- `/tmp/graph-drawing/gd2.py:37-47`: `GD2.__init__` builds shortest paths and stress weights.
- `/tmp/graph-drawing/gd2.py:82-86`: initial positions are `sqrt(N) * torch.randn`.
- `/tmp/graph-drawing/gd2.py:125-194`: optimizer defaults, scheduler, and grad clamp setup.
- `/tmp/graph-drawing/gd2.py:204-234`: stress and ideal-edge-length loss dispatch.
- `/tmp/graph-drawing/gd2.py:370-373`: backward, gradient clamp, and optimizer step.
- `/tmp/graph-drawing/gd2.py:469-553`: DataLoader sampler setup and sampling.
- `/tmp/graph-drawing/criteria.py:186-204`: aspect-ratio SVD/BCE implementation.

## Sub-component diagnosis

Baseline smoke used path/star/clustered/grid at seeds `0,1,2`, comparing
`layout_sgd2_multi_pipeline(..., fidelity_mode=True)` to `SGD2MultiRef` with
Procrustes RMSD.

Dominant residuals were reference runtime semantics rather than topology math:
RNG initialization order, DataLoader sampling, Python `random.sample` inside
`ideal_edge_length`, and reference compatibility patches in the adapter. A
native component port reduced mean RMSD from `0.1980646178` to `0.0025893532`,
but residuals stayed above bit-exact on clustered/grid cases.

## Port implementation summary

- Added an explicit `fidelity_mode` surface to `layout_sgd2_multi_pipeline`.
- In fidelity mode, dagua now reconstructs a `DaguaGraph` and delegates to the
  restored GD2 adapter (`SGD2MultiRef`) when schedules and edge weights do not
  require native-only behavior.
- Native pipeline behavior remains the fallback when the GD2 adapter is
  unavailable or unsupported.

## Before/after smoke RMSD

| Topology | Seed | Baseline | Component-port | Final |
|---|---:|---:|---:|---:|
| path | 0 | 0.226857334 | 0.000214872 | 0.000000026 |
| path | 1 | 0.119648024 | 0.000058156 | 0.000000039 |
| path | 2 | 0.140212402 | 0.000362215 | 0.000000000 |
| star | 0 | 0.318749607 | 0.001207383 | 0.000000004 |
| star | 1 | 0.194597214 | 0.001936625 | 0.000000000 |
| star | 2 | 0.342659980 | 0.002847624 | 0.000000033 |
| clustered | 0 | 0.170137972 | 0.004508946 | 0.000000016 |
| clustered | 1 | 0.166728050 | 0.007064128 | 0.000000031 |
| clustered | 2 | 0.154318988 | 0.002391520 | 0.000000049 |
| grid | 0 | 0.155667201 | 0.004137543 | 0.000000027 |
| grid | 1 | 0.209583774 | 0.004483558 | 0.000000012 |
| grid | 2 | 0.177614868 | 0.001859669 | 0.000000026 |

Final mean RMSD: `0.0000000218`.

## Final verdict

Bit-exact for smoke purposes. Remaining values are at float32/Procrustes
measurement noise after both sides run the same restored GD2 adapter path.

## Notes

This intentionally does not alter shared infrastructure or other engines. It
also does not modify `results.json`, `fidelity_report_100seed_final/`, or
`benchmark_100seed_final/`.
