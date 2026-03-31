# Render Subpackage

See `AGENTS.md` in this directory for implementation guide, dependency rules,
gotchas, and test commands.

Key design constraint: renderers accept structured data, not Graph objects.
Three independent backends (mpl, svg, graphviz), no shared state.
