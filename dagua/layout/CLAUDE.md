# Layout Subpackage

See `AGENTS.md` in this directory for implementation guide, dependency rules,
gotchas, and test commands.

Key design constraint: the layout engine is **headless** -- operates on tensors,
not Graph objects. `Graph.layout()` extracts tensors, calls into this package,
stores results back.

Two layers: core engine (constraints + optimization) and composable ops (268
primitives in `ops/`, composed into 23 algorithm pipelines in `ops/pipelines/`).
`LayoutConfig(algorithm="fr")` dispatches to ops pipelines.
