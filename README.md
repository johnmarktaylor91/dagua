# dagua

GPU-accelerated differentiable graph layout engine built on PyTorch.

**DAG + agua.** Directed acyclic graphs + water. Named after the Dagua River in Colombia — a river flows downhill (like a DAG), never cycles back (acyclic), and finds its own path through the landscape (like gradient descent finding optimal node positions).

## Why?

Graphviz has dominated graph visualization for 30 years but has hard scaling limits. No existing Python package provides pip-installable, hierarchical (Sugiyama-style) graph layout. Dagua fills this gap: `pip install dagua`, pure Python + PyTorch, GPU-accelerated, hierarchical layout with composable constraints.

## Status

Pre-alpha. Under active development.

## License

MIT
