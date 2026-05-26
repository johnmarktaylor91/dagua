# R60 FA2 Barnes-Hut Real Port Summary

## Source Read

Read `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2/fa2util.pyx`,
especially `Region` at lines 658-835.

Observed order:

- Tree construction computes mass sequentially in node-list order, then size
  sequentially in node-list order.
- Buckets are visited in numeric order `0, 1, 2, 3`, with bit 1 set by
  `x >= massCenter[0]` and bit 2 set by `y >= massCenter[1]`.
- Subregions are appended in that bucket order, then each subregion is built
  recursively in append order.
- `applyForceOnNodes` visits target nodes in list order. Each target traverses
  recursively depth-first through `subregions` in append order.
- Accepted regions apply `linRepulsion_region_2d`: compute `xDist`, `yDist`,
  `distance2 = xDist * xDist + yDist * yDist`, then
  `factor = coefficient * n.mass * r.mass / distance2`, then update `dx`
  before `dy`.
- Opening test uses `distance = sqrt((xDiff * xDiff) + (yDiff * yDiff))` and
  accepts when `distance * theta > region.size`.

## Changes

- Added a pure Python `_FA2ReferenceRegion` in
  `dagua/layout/ops/pipelines/fa2.py`.
- Added mutable `_FA2ReferenceNode` and `_FA2ReferenceEdge` helpers for the
  fidelity Barnes-Hut loop.
- Routed both `layout_fa2_pipeline(..., fidelity_mode=True, barnes_hut=True)`
  and direct `build_fa2_pipeline(FA2Config(fidelity_mode=True, barnes_hut=True))`
  through the pure Python Region port.
- No runtime import or delegation to `fa2util.Region` was added in Dagua code.

## Trace Comparison

Smoke trace against compiled `fa2util.Region`:

- `star_12`, seed `0`, first tree: exact structure match, including node
  membership, mass centers, sizes, and child order.
- Node `0` first repulsion force: bit-for-bit match.
  - Dagua: `(129.83605672332487, 467.0748270307607)`
  - fa2util: `(129.83605672332487, 467.0748270307607)`

## Smoke RMSD

Direct live-reference Barnes-Hut smoke, 50 iterations, seed `0`:

| graph | max abs | RMSD | bit equal |
| --- | ---: | ---: | --- |
| star_12 | 0.0 | 0.0 | yes |
| path_10 | 0.0 | 0.0 | yes |
| cycle_8 | 0.0 | 0.0 | yes |

The same three cases also matched bit-for-bit through the direct
`build_fa2_pipeline` path.

## Verification

Passed:

```text
pytest tests/test_layout/test_fa2_fidelity.py tests/test_pipeline_fa2.py::TestFA2PipelineFidelity::test_layout_fa2_pipeline_matches_classic_barnes_hut -x --tb=short -q
8 passed, 2 warnings in 0.26s
```

Passed:

```text
mypy --follow-imports=silent dagua/cli.py
Success: no issues found in 1 source file
```

Passed:

```text
ruff format dagua/layout/ops/pipelines/fa2.py
ruff check dagua/layout/ops/pipelines/fa2.py --fix
```

Interrupted due runtime/CPU contention:

```text
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
114 passed, 5 warnings in 1200.15s
KeyboardInterrupt at dagua/metrics.py:146
```

## Residual Risk

Only 2D non-anti-collision FA2 fidelity Barnes-Hut was ported because Dagua's
public FA2 pipeline exposes 2D layouts and does not pass `adjustSizes` into the
fidelity wrapper.
