# dagua

GPU-accelerated differentiable graph layout engine built on PyTorch.

**DAG + agua.** Directed acyclic graphs + water. Named after the Dagua River in Colombia — a river flows downhill (like a DAG), never cycles back (acyclic), and finds its own path through the landscape (like gradient descent finding optimal node positions).

## Why?

Graphviz has dominated graph visualization for 30 years but has hard scaling limits. No existing Python package provides pip-installable, hierarchical (Sugiyama-style) graph layout. Dagua fills this gap: `pip install dagua`, pure Python + PyTorch, GPU-accelerated, hierarchical layout with composable constraints.

## Status

Pre-alpha. Under active development.

## CLI

The package now ships with a small CLI:

```bash
dagua benchmark-status --output-dir eval_output --suite standard
dagua benchmark-watch --output-dir eval_output --suite standard --follow --interval 15
dagua benchmark-list --output-dir eval_output --suite standard
dagua benchmark-show residual_block --output-dir eval_output --suite standard --competitor dagua
dagua benchmark-report --output-dir eval_output --no-pdf
dagua benchmark-deltas --output-dir eval_output
dagua benchmark-freeze baseline-a --output-dir eval_output --suite standard
dagua benchmark-compare-runs 2026-03-12T00:00:00+00:00 2026-03-12T01:00:00+00:00 --output-dir eval_output
```

For cinematic exports:

```bash
dagua poster graph.yaml poster.png --scene powers_of_ten --device cuda
dagua tour graph.yaml trailer.mp4 --scene zoom_pan --device cuda
```

For the exhaustive reference manual:

```bash
python scripts/build_glossary.py --output-dir docs/glossary
```

That command regenerates the LaTeX source, explanatory visuals, manifest, and PDF
when `pdflatex` is available.

You can also render directly from saved benchmark positions, which is useful for
large graphs where you do not want to relayout just to produce a poster or tour:

```bash
dagua poster unused.json /tmp/residual.png \
  --benchmark-graph residual_block \
  --benchmark-suite standard \
  --competitor dagua \
  --output-dir eval_output
```

## License

MIT
