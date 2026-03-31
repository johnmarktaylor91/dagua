# Layout Subpackage

See `AGENTS.md` in this directory for implementation guide, dependency rules,
gotchas, and test commands.

Key design constraint: the layout engine is **headless** -- operates on tensors,
not Graph objects. `Graph.layout()` extracts tensors, calls into this package,
stores results back.
