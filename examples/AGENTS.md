# Examples

## Purpose

Runnable scripts demonstrating dagua's API. Each should be self-contained and produce
visible output (saved image or displayed plot).

## Files

- **quickstart.py** — Minimal 5-line example. The "hello world" of dagua. Should appear
  verbatim in README.md.
- **neural_network.py** — DNN-style graph with module clusters. Demonstrates cluster
  constraints and domain-specific styling. Relevant to TorchLens use case.
- **custom_constraints.py** — Writing a custom constraint callable. Shows the
  `(pos, graph_data) -> loss` protocol with a real example.
- **large_graph.py** — 10K+ node demo. Showcases GPU acceleration and negative sampling.

## Conventions

- Each example is runnable with `python examples/<name>.py`
- No external dependencies beyond dagua's optional deps (matplotlib)
- Comments explain what's happening, aimed at first-time users
