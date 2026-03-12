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

## Agent Guide

This repo includes a public, user-facing guide for LLMs and coding agents using Dagua:

- [docs/LLM_TUTORIAL.md](/home/jtaylor/projects/dagua/docs/LLM_TUTORIAL.md)

It is separate from `CLAUDE.md` / `AGENTS.md`, which are for developing the repo itself.
The LLM tutorial is the short, task-oriented entrypoint for agents that want to build,
layout, render, and export graphs efficiently.

## Image To Graph / Theme

Dagua can also turn a graph screenshot or reference visualization into:
- a `DaguaGraph`
- a raw graph/theme dict
- best-practice Dagua code
- a `Theme`

Canonical setup:

```python
import dagua

dagua.configure_image_ai(provider="openai", api_key_env="OPENAI_API_KEY")
```

Then use whichever return mode you want:

```python
graph = dagua.from_image("diagram.png")
graph_dict = dagua.graph_dict_from_image("diagram.png")
graph_code = dagua.graph_code_from_image("diagram.png")

theme = dagua.theme_from_image("diagram.png")
theme_dict = dagua.theme_dict_from_image("diagram.png")
theme_code = dagua.theme_code_from_image("diagram.png")
```

Supported common formats include:
- PNG
- JPEG
- GIF
- WebP
- BMP
- TIFF
- SVG

The code-return helpers intentionally emit the cleaner Dagua style: explicit graph
construction, clusters, and styles in a reusable builder function instead of ad hoc blobs.

## License

MIT
