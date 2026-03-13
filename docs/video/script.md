# Dagua Explainer Script

## Short Version

Dagua is a graph layout engine that treats layout as optimization.

Instead of hardcoding every layout decision into a tangle of heuristics, it turns graph drawing aesthetics into loss terms:
- preserve directional flow
- reduce crossings
- avoid overlap
- keep edges readable
- maintain cluster structure

For small graphs, that means flexible, inspectable layout behavior.
For large graphs, it means a multilevel pipeline that solves the broad composition first, then refines detail.

And because the whole system is built around optimization, you can actually watch the layout settle into place.

## Longer Beat Structure

1. Show a graph appearing with `dagua.draw()`
2. Explain that the graph is optimized, not just heuristically arranged
3. Show pinning/alignment animation
4. Explain multilevel coarsening and refinement
5. Show routing modes
6. Show a poster/tour of a large graph
7. Close on the Python-native API and differentiable mindset
