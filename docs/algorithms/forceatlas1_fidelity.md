# ForceAtlas1 Fidelity

## Reference Status

Target reference: Gephi `ForceAtlasLayout.java` from the layout plugin, plus
`ForceVectorUtils.java` and `AbstractLayout.ensureSafeLayoutNodePositions`.

Reference runtime status: runnable. `scripts/verify_forceatlas1_fidelity.py`
downloads `org.gephi:gephi-toolkit:0.10.1:all` from Maven Central into
`/tmp/dagua-fa1-gephi`, compiles
`scripts/gephi_forceatlas1_runner/ForceAtlas1ReferenceRunner.java`, and runs it
with `java -Djava.awt.headless=true`.

Implemented path: source-faithful port of the Java loop in
`dagua/layout/ops/pipelines/forceatlas1.py`.

Reference path: direct Gephi Toolkit `GraphModel` construction, fixed initial
node coordinates, `org.gephi.layout.plugin.forceAtlas.ForceAtlasLayout`, and
manual fixed-iteration `goAlgo()` calls. The runner uses fixed initial
coordinates because Gephi's all-zero safeguard initializes with `Math.random()`
and exposes no public seed setter.

First divergent stage: arithmetic surface. The port matches Gephi's dynamics at
the positional tier; remaining raw coordinate drift is float-operation/order
rounding on the order of `1e-5`, while similarity-aligned residuals are around
`1e-8`.

## Model Match

The port follows these ForceAtlas1-specific dynamics, which differ from the
existing ForceAtlas2 pipeline:

- Java `Random` initialization from the all-zero position path:
  `(0.01 + nextDouble()) * 1000 - 500`, stored as float node coordinates.
- Inertia model: each step copies `old_dx/old_dy`, then keeps
  `dx *= inertia`, `dy *= inertia`.
- Repulsion is pairwise over ordered node pairs and uses
  `repulsionStrength * (1 + degree(source)) * (1 + degree(target))`.
- Attraction is linear, weighted by edge weight and optionally divided by
  `1 + degree(source)` for outbound-attraction distribution.
- `adjustSizes` switches to Gephi's anti-collision distance for repulsion and
  attraction.
- Gravity follows Gephi's `0.0001 * gravity` origin pull.
- Freeze-balance applies the Gephi freeze inertia/strength damping before
  cooling and max-displacement limiting.

## Verification

Command:

```bash
python scripts/verify_forceatlas1_fidelity.py
```

Current report, generated against the headless Gephi Toolkit runtime:

| graph | residual | max abs | tier | quality |
| --- | ---: | ---: | --- | ---: |
| path_default | 3.184e-08 | 1.534e-05 | positional | 38.90 |
| weighted_outbound | 3.331e-08 | 2.234e-05 | positional | 24.15 |
| adjust_sizes | 8.950e-09 | 7.178e-06 | positional | 32.33 |
| no_freeze_gravity | 2.585e-08 | 7.636e-06 | positional | 31.60 |

Overall tier: positional against Gephi Toolkit 0.10.1. This supersedes the
earlier source-port self-check status.

## Guardrail

`tests/test_pipeline_forceatlas1.py` includes an AST guard that rejects imports
from competitor adapters, subprocesses, and Java bridge modules in the
production pipeline. The ForceAtlas1 pipeline does not delegate to Gephi at
runtime.
