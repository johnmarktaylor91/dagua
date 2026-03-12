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
graph_script = dagua.graph_code_from_image("diagram.png", include_demo_script=True)

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

The code-return helpers intentionally emit the cleaner Dagua style:
- default: reusable builder code with explicit `add_node` / `add_edge` / `add_cluster`
- optional: a polished ready-to-run demo script with layout and export already wired in

## FAQ

### What is Dagua different from?

Dagua is most directly comparable to:
- Graphviz `dot`
- ELK layered
- dagre
- other hierarchical / Sugiyama-style layout tools

It is less like:
- force-directed tools such as NetworkX spring layout or Graphviz `sfdp`

### What is different about Dagua vs Graphviz, ELK, or dagre?

The main difference is architectural.

Those tools are largely heuristic rule-based layout engines. Dagua treats layout as continuous optimization:
- node positions are optimized with PyTorch
- aesthetics are loss terms
- GPU acceleration is available naturally
- constraints like pins, alignment, spacing preferences, and cluster behavior fit into the same framework

So the point of Dagua is not just “another graph drawer.” The point is:
- hierarchical layout
- Python-native workflow
- inspectable optimization behavior
- composable constraints
- scaling paths that can exploit modern accelerators

### When should I use Dagua instead of Graphviz?

Use Dagua when you want:
- a Python-native hierarchical layout tool
- GPU acceleration
- cinematic exports, optimization animations, or large-graph tours
- constraint-style control over layout behavior
- one library that covers graph creation, optimization, and visualization together

Use Graphviz when you want:
- a mature external tool with decades of ecosystem history
- quick static output and you are already happy with its workflow

### Is Dagua a force-directed layout library?

No. It uses optimization internally, but the intended visual language is hierarchical / layered DAG layout, not generic force-directed blob layout.

### Do I have to call `layout()` manually?

No for the basic case.

```python
fig, ax = dagua.draw(g)
```

`draw()` is the convenience path and will lay out the graph for you.

Use `layout()` explicitly when you want to:
- inspect positions
- render multiple times from the same layout
- intervene between layout and rendering
- route edges or place labels manually

### What happens if I change the graph after layout?

Dagua now tracks layout lifecycle state.

If you mutate the graph structure or relevant styles:
- cached layout-derived artifacts are invalidated
- `draw()` will naturally relayout
- explicit stale usage is easier to detect and reason about

This is intentional: the basic path should feel automatic, but not magical in a way you cannot inspect.

### Can I inspect what the optimizer did?

Yes.

Useful surfaces include:
- `dagua.layout(...)` for direct positions
- optimization animations via `dagua.animate(...)`
- cinematic stills and tours via `dagua.poster(...)` and `dagua.tour(...)`
- graph lifecycle state such as cached layout freshness

### Does Dagua support clusters and nested clusters?

Yes.

Recommended hand-authored pattern:
- add nodes first
- then add clusters by node id

Declarative hierarchy is also supported, but for handwritten Python code the bottom-up pattern is usually clearer.

### Can Dagua handle large graphs?

Yes. Large-scale support is a core part of the project.

That said, there are different regimes:
- small and medium graphs: normal interactive/programmatic usage
- large graphs: multilevel layout and benchmark/scaling tooling
- huge graphs: dedicated poster/tour workflows and large-graph rendering strategies

If your goal is “show off a giant graph,” use:
- `dagua.poster(...)`
- `dagua.tour(...)`

instead of trying to treat a giant graph like a tiny static diagram.

### Is Dagua only for machine learning graphs?

No.

ML systems are an important use case, but the same API is meant for:
- workflow graphs
- business processes
- dependency graphs
- architecture diagrams
- logistics / operations flows
- any structured directed graph where layered layout is useful

### Is there documentation for humans and agents separately?

Yes.

For humans:
- docs index: [docs/README.md](/home/jtaylor/projects/dagua/docs/README.md)
- developer overview: [docs/DEVELOPER_OVERVIEW.md](/home/jtaylor/projects/dagua/docs/DEVELOPER_OVERVIEW.md)
- tutorial notebook: [docs/tutorial_walkthrough.ipynb](/home/jtaylor/projects/dagua/docs/tutorial_walkthrough.ipynb)
- glossary/reference: [docs/glossary/dagua_glossary.pdf](/home/jtaylor/projects/dagua/docs/glossary/dagua_glossary.pdf)
- how Dagua works: [docs/how_dagua_works.md](/home/jtaylor/projects/dagua/docs/how_dagua_works.md)
- showcase gallery: [docs/gallery/README.md](/home/jtaylor/projects/dagua/docs/gallery/README.md)
- video resources: [docs/video/README.md](/home/jtaylor/projects/dagua/docs/video/README.md)

For agents using Dagua:
- [docs/LLM_TUTORIAL.md](/home/jtaylor/projects/dagua/docs/LLM_TUTORIAL.md)

For agents developing the repo itself:
- `CLAUDE.md` / `AGENTS.md`

## License

MIT
