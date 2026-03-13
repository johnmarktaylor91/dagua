# Dagua Developer Overview

This is the end-to-end map of Dagua as a codebase.

The goal is simple: after reading this file, you should understand how Dagua is put together, how data moves through it, where the major abstractions live, and which files matter when changing a given behavior.

This is developer-facing. It complements:

- `docs/tutorial_walkthrough.ipynb` for users
- `docs/glossary/` for exhaustive reference
- `docs/how_dagua_works.md` for public algorithm explanation
- `docs/LLM_TUTORIAL.md` for agent-efficient usage guidance
- `CLAUDE.md` / `AGENTS.md` for maintainer workflow notes

For current iteration strategy, also read:

- `docs/CRITERIA_LEDGER.md`
- `docs/ITERATION_WORKFLOW.md`
- `docs/STATUS.md`

## 1. What Dagua Is

Dagua is a Python library for:

- programmatic graph construction
- differentiable hierarchical layout optimization
- edge routing and label placement
- static rendering
- animation / tour / poster export
- evaluation against competitor layout engines
- documentation and visual-audit tooling

Its center of gravity is the layout engine: a PyTorch-based optimizer for DAG-aware layered graphs, with a multilevel path for large graphs.

At a high level, the default user flow is:

1. build a `DaguaGraph`
2. compute node sizes from labels/styles
3. optimize node positions with `layout()`
4. route edges and place edge labels
5. render or export

The convenience entrypoint `dagua.draw(...)` performs that full pipeline in one call.

At a high level, the current optimization strategy is:

1. stage 0: keep the criteria inventory explicit
2. stage 1: optimize node placement numerically
3. stage 2: optimize downstream geometry numerically
4. stage 3: narrow the remaining aesthetic defaults visually

That staged split matters because Dagua's current weak layer is largely in stage 2
and stage 3, not in the existence of a layout engine at all.

## 2. Top-Level Codebase Map

Core package:

- `dagua/__init__.py`
  - public API and `draw()`
- `dagua/graph.py`
  - `DaguaGraph`, lifecycle state, graph construction
- `dagua/config.py`
  - `LayoutConfig`
- `dagua/styles.py`
  - style dataclasses, theme registry, built-in themes
- `dagua/defaults.py`
  - global defaults / configuration
- `dagua/flex.py`
  - user-facing pinning/alignment/flex constraints
- `dagua/io.py`
  - JSON/YAML/image IO, image-to-graph/theme/code helpers

Layout and routing:

- `dagua/layout/engine.py`
  - direct layout engine and top-level `layout()`
- `dagua/layout/init_placement.py`
  - initialization / barycenter / spectral ordering
- `dagua/layout/multilevel.py`
  - hierarchy build, coarsest solve, prolongation/refinement
- `dagua/layout/constraints.py`
  - differentiable loss functions and hard projections
- `dagua/layout/projection.py`
  - overlap projection helpers
- `dagua/layout/edge_optimization.py`
  - edge-curve refinement
- `dagua/layout/layers.py`
  - layer-index helpers

Rendering and export:

- `dagua/render/mpl.py`
  - main render path
- `dagua/render/svg.py`
  - SVG-specific helpers
- `dagua/edges.py`
  - route edges, place edge labels
- `dagua/animation.py`
  - optimization animation, tours, posters, large-graph LOD

Evaluation and reporting:

- `dagua/eval/graphs.py`
  - benchmark/test graph corpus
- `dagua/eval/competitors/`
  - Graphviz / ELK / dagre / NetworkX wrappers
- `dagua/eval/benchmark.py`
  - persistent benchmark pipeline
- `dagua/eval/report.py`
  - report, comparison visuals, scaling curve, similarity summaries
- `dagua/eval/visual_audit.py`
  - visual-iteration trench workflow
- `dagua/eval/aesthetic.py`
  - offline aesthetic iteration harness

Scripts and docs:

- `scripts/`
  - builders for gallery, glossary, algorithm explainer, visual audit, large benchmark
- `docs/`
  - public documentation and generated assets
- `tests/`
  - smoke, regression, eval, render, layout, IO, notebook coverage

## 3. The Core Data Model

### `DaguaGraph`

The central type is `DaguaGraph` in `dagua/graph.py`.

It stores:

- topology
  - `num_nodes`
  - `edge_index`
  - `_pending_edges` for efficient incremental graph construction
- node-facing data
  - `node_labels`
  - `node_types`
  - `node_styles`
  - `node_sizes`
  - `node_font_sizes`
- edge-facing data
  - `edge_labels`
  - `edge_types`
  - `edge_styles`
- cluster data
  - `clusters`
  - `cluster_styles`
  - `cluster_labels`
  - `cluster_parents`
- graph-level layout/render defaults
  - `direction`
  - `default_node_style`
  - `default_edge_style`
  - `flex`
  - storage dtypes

It also tracks layout lifecycle state:

- `revision`
- `layout_status`
- `has_fresh_layout`
- `last_positions`
- `last_curves`
- `last_label_positions`

This matters because Dagua tries to make the common case automatic while keeping state inspectable:

- graph mutation invalidates cached layout-derived artifacts
- `draw()` auto-relayouts when necessary
- `render()` can use cached fresh positions, but should not silently fake freshness

### Style Resolution

Styles live in `dagua/styles.py`.

The important dataclasses are:

- `NodeStyle`
- `EdgeStyle`
- `ClusterStyle`
- `GraphStyle`
- `Theme`

Theme resolution is a cascade:

1. per-element override
2. type-specific theme style
3. graph default style
4. built-in theme/defaults

That style cascade is used in two places:

- pre-layout sizing decisions
- final render/routing decisions

### Config

Optimization behavior is controlled by `LayoutConfig` in `dagua/config.py`.

This contains:

- step counts
- loss weights
- spacing defaults
- device choice
- multilevel thresholds
- performance knobs
- edge-optimization controls
- verbosity

Global defaults live in `dagua/defaults.py`, and user code can override them via `dagua.configure(...)`.

## 4. End-to-End Runtime Flow

This is the core runtime path.

### Step A: Graph Construction

Users can build graphs through:

- `DaguaGraph()`
- `DaguaGraph.from_edge_list(...)`
- `graph_from_json(...)`
- `graph_from_yaml(...)`
- interoperability helpers
- image-to-graph helpers in `dagua/io.py`

During construction:

- node IDs are mapped to integer indices
- edges accumulate in `_pending_edges`
- clusters are registered by name and membership
- styles and labels are stored, but layout has not happened yet

### Step B: Node Sizing

Before layout, Dagua computes node sizes from:

- label text
- font settings
- shape
- padding
- overflow policy

This lives in:

- `dagua/utils.py`
- `dagua/graph.py` (`compute_node_sizes()`)

This is important because layout operates on node boxes, not abstract points.

### Step C: Layout Entry

For iteration purposes, it helps to think of the runtime in two separate geometry phases:

- node placement
- downstream geometry

Those map only partly onto code today. The code already separates node layout from
edge routing/refinement, but the evaluation workflow is still catching up to that
architecture.

Public layout entrypoints:

- `dagua.layout(graph, config)`
- `dagua.draw(graph, ...)`

In `dagua/__init__.py`, `draw()` does:

1. choose/effective config
2. decide whether to reuse cached positions or relayout
3. call `layout()`
4. route edges
5. optionally optimize edges
6. place edge labels
7. render

So `draw()` is the “just make it work” API, while `layout()` is the explicit seam for power users.

### Step D: Direct Layout vs Multilevel

In `dagua/layout/engine.py`, the top-level `layout()` chooses:

- direct layout for smaller graphs
- multilevel layout for graphs above `config.multilevel_threshold`

That split is one of the most important architectural boundaries in the codebase.

#### Direct layout

For smaller/medium graphs:

- move graph tensors to the target device
- initialize positions
- build a layer index
- run the gradient loop in `_layout_inner(...)`

#### Multilevel layout

For very large graphs:

- keep hierarchy-building on CPU
- build coarsened levels in `dagua/layout/multilevel.py`
- solve the coarsest graph
- prolong positions to finer levels
- refine each level with shorter optimization passes

This is the reason the billion-node path is even plausible.

## 5. Initialization

Initialization lives in `dagua/layout/init_placement.py`.

Its job is not final aesthetics. Its job is to provide a strong starting point so the optimizer does not waste time climbing out of nonsense.

Main components:

- longest-path layering
- vectorized barycenter ordering
- optional spectral ordering on suitable graphs
- fanout spreading for hub children

For giant coarse graphs, initialization now includes a safe device-selection guard:

- if GPU init ordering would exceed conservative VRAM headroom
- compute the ordering on CPU
- move only final positions to GPU

That guard exists because the coarsest graph in a multilevel run can still be too large for naive CUDA-side initialization.

## 6. The Optimization Loop

The main loop lives in `_layout_inner(...)` in `dagua/layout/engine.py`.

At a high level it:

1. gets initial positions
2. computes layer assignments / layer index
3. chooses adaptive behavior based on graph size
4. repeatedly computes losses
5. backpropagates and updates positions
6. occasionally projects overlaps / pins
7. stops when the configured step budget or convergence logic says so

The losses themselves live in `dagua/layout/constraints.py`.

Important families:

- DAG ordering
- edge attraction
- repulsion
- overlap avoidance
- crossing loss
- alignment / spacing consistency
- cluster compactness / containment / separation
- edge straightness
- edge length variance
- pinning / flex constraints
- special losses like fanout distribution and back-edge compactness

The engine also includes scaling strategies:

- exact or approximate repulsion based on size
- edge batching
- adaptive overlap projection cadence
- optional hybrid CPU/GPU behavior
- per-loss backward / checkpoint-friendly structure

Conceptually, this is Dagua’s heart: layout as continuous optimization over a DAG-aware layered initialization.

## 7. Multilevel Layout

Multilevel logic lives in `dagua/layout/multilevel.py`.

This subsystem:

1. builds a hierarchy of coarsened graphs
2. solves the smallest coarse graph
3. prolongs positions back upward
4. refines each level

Important details:

- hierarchy build stays on CPU because very large hash/unique/dedup passes are memory-bound and often not GPU-friendly at giant scale
- coarsening is layer-aware
- non-streaming coarsening uses more structural heuristics for better same-layer grouping
- streaming coarsening is the conservative path for the largest graphs
- prolongation has a guarded GPU path when headroom is available

This is the subsystem you touch for:

- billion-scale survival work
- hierarchy quality improvements
- coarsening heuristics
- CPU/GPU split decisions

## 8. Edge Routing and Label Placement

Once node positions are known, Dagua computes edge geometry.

This lives in:

- `dagua/edges.py`
- `dagua/layout/edge_optimization.py`

The routing path:

1. resolve styles / shapes / ports
2. generate initial curves
3. optionally refine control points with edge optimization
4. place edge labels

Important point:

- node placement quality and routing quality are related, but they are different problems
- for the current reset, node placement is the core asset; routing and visual language can be rebuilt later on top of it

Edge label placement currently supports:

- position along the edge
- preferred side
- preferred offset
- avoidance adjustments when collisions are enabled

## 9. Rendering

Rendering is primarily in `dagua/render/mpl.py`.

The render stack draws:

- clusters
- edges
- nodes
- node labels
- edge labels

The rendering system also handles:

- raster export
- vector export
- SVG hover text
- consistent output-format inference from path

Large-graph rendering is not just “draw everything and hope.” The cinematic stack includes LOD logic for large scenes, especially for tours and posters.

## 10. Animation, Tours, and Posters

This lives in `dagua/animation.py`.

There are really three feature families here:

### Optimization animation

- captures actual optimization snapshots
- renders after the fact
- useful for teaching and debugging constraints

### Graph tour

- camera choreography over a final layout
- automatic scenes plus keyframe mode
- intended for both medium graphs and giant showcase graphs

### Poster

- still-image cinematic export
- shares large-graph LOD logic with tours

Large-graph support includes:

- density-style overview rendering
- sampled edge-flow textures
- zoom-triggered detail reveal
- label suppression when density is too high

These features matter because Dagua is not only a layout engine; it is also a presentation/export tool.

## 11. IO and Interop

The IO surface in `dagua/io.py` is broader than simple file load/save.

It includes:

- graph JSON/YAML serialization
- style serialization
- interop with NetworkX / igraph / SciPy / PyG / DOT
- image-to-graph
- image-to-theme
- image-to-code / script generation

The image-AI path now has a clear provider/key config surface:

- `ImageAIConfig`
- `configure_image_ai(...)`
- `get_image_ai_config()`

It also has multiple return modes:

- graph/theme data structure
- graph/theme code text
- runnable demo-style script text

This is one of the higher-level “magic” surfaces, but it still plugs back into the same core graph/theme abstractions.

## 12. Evaluation and Benchmarking

Evaluation is a substantial subsystem.

### Graph corpus

`dagua/eval/graphs.py` defines the benchmark/test graph collection.

It includes:

- structural basics
- ML-ish motifs
- kitchen-sink graphs
- label stress graphs
- style/routing stress graphs
- placement challenge graphs
- scale ladder graphs

These are not just tests. They are also the corpus used for iteration and diagnosis.

### Competitors

`dagua/eval/competitors/` wraps:

- Graphviz `dot`
- Graphviz `sfdp`
- ELK layered
- dagre
- NetworkX spring

The benchmark code respects known scale ceilings and skips gracefully when a competitor is unavailable or out of range.

### Benchmark pipeline

`dagua/eval/benchmark.py` provides:

- standard suite
- rare suite
- persistent timestamped run storage
- saved positions
- partial checkpoints
- cached competitor reuse
- merge of latest standard + rare runs
- run-progress metadata

Important artifacts:

- `results.json`
- `results.partial.json`
- `progress.json`
- `metadata.json`
- saved positions under `positions/`
- merged `combined_latest.json`

This benchmark layer is deliberately persistent because layout computation is expensive but metrics/report regeneration is cheap once positions are stored.

### Report pipeline

`dagua/eval/report.py` builds:

- comparison visuals
- scaling curve
- benchmark deltas
- layout similarity summaries
- placement-only summaries
- LaTeX report and optional PDF

The layout-similarity summary matters because it helps answer:

- are two tools genuinely finding different layouts?
- or are they finding similar geometry with different styling?

The placement-only summary matters because it lets the team evaluate the node-placement engine independently from the currently weak visual layer.

## 13. Visual-Audit Infrastructure

`dagua/eval/visual_audit.py` is the trench-workflow subsystem for design iteration.

It builds:

- complexity ladders
- decomposition views
- kill-switch matrices
- diff dashboards
- competitor stepwise comparisons
- typography and edge-language sheets
- metric cards
- frozen baselines

This is there so iteration does not happen by vague memory or random screenshots. It gives a repeatable visual-debug workflow.

If you are redesigning the visual layer, this subsystem is one of the most important places in the repo.

## 14. Documentation and Generated Assets

The docs surface is intentionally broad.

Important public docs:

- `docs/README.md`
- `docs/tutorial_walkthrough.ipynb`
- `docs/glossary/`
- `docs/how_dagua_works.md`
- `docs/gallery/`
- `docs/video/`
- `docs/LLM_TUTORIAL.md`

Important maintainer-facing docs:

- `CLAUDE.md`
- `docs/MAINTENANCE_CHECKLIST.md`
- this file

Many docs are generated by scripts in `scripts/`, and several have smoke tests to prevent silent drift.

## 15. CLI Surface

`dagua/cli.py` exposes user-facing tooling around the major workflows:

- benchmark status / watch / list / show / freeze / compare / report / deltas
- visual audit build / freeze
- poster
- tour

This CLI layer is not the core engine, but it is the operational surface for iteration and artifact generation.

## 16. TorchLens Integration

TorchLens integration lives in the TorchLens repo, but Dagua contains part of the contract:

- built-in `torchlens` theme in `dagua/styles.py`
- direction-aware draw/render support
- style/type hooks that TorchLens can target

In TorchLens itself, the Dagua bridge maps `ModelLog` data into a `DaguaGraph`. Graphviz remains the default there, while Dagua is opt-in.

This distinction matters:

- TorchLens semantic mapping is one problem
- Dagua layout quality is another
- Dagua visual language is yet another

Keeping those separate prevents confusion when debugging.

## 17. The Most Important Design Boundaries

These boundaries explain much of the codebase:

### Graph state vs layout-derived state

User-authored graph structure and style live on `DaguaGraph`.
Computed positions, curves, and label positions are cached separately and invalidated on mutation.

### Node placement vs rendering

The node-placement engine may be strong even when the visual layer is weak.
Do not confuse layout quality with styling quality.

### Direct layout vs multilevel

Small/medium graphs go through direct optimization.
Very large graphs go through CPU-built hierarchy + coarsest solve + refinement.

### Public user docs vs maintainer docs vs agent docs

The repo intentionally has separate surfaces for:

- humans using Dagua
- agents using Dagua
- agents/humans developing the repo

That separation is deliberate, not duplication.

## 18. Where To Start For Common Changes

If you want to change:

### Graph construction or lifecycle

Start in:

- `dagua/graph.py`
- `dagua/__init__.py`
- `dagua/io.py`

### Losses / optimization behavior

Start in:

- `dagua/layout/engine.py`
- `dagua/layout/constraints.py`
- `dagua/layout/init_placement.py`

### Multilevel / giant-scale behavior

Start in:

- `dagua/layout/multilevel.py`
- `dagua/layout/init_placement.py`
- `scripts/bench_large.py`

### Edge routing / labels

Start in:

- `dagua/edges.py`
- `dagua/layout/edge_optimization.py`
- `dagua/render/mpl.py`

### Themes / visual language

Start in:

- `dagua/styles.py`
- `dagua/render/mpl.py`
- `dagua/eval/visual_audit.py`
- `docs/gallery/`

### Benchmarking / reports

Start in:

- `dagua/eval/benchmark.py`
- `dagua/eval/report.py`
- `dagua/eval/graphs.py`
- `dagua/eval/competitors/`

### Docs and generated public assets

Start in:

- `docs/README.md`
- `docs/MAINTENANCE_CHECKLIST.md`
- corresponding `scripts/build_*.py`

## 19. Current Strategic Read

The codebase now broadly has its major architecture in place.

The strongest current story is:

- scalable hierarchical layout
- multilevel giant-graph path
- persistent evaluation/reporting infrastructure
- strong iteration tooling

The weakest current story is:

- the visual language itself
- text / edge / box / cluster design quality

That means the correct development stance right now is:

- protect and improve node placement quality
- keep pushing scale robustness
- use benchmark metrics to lock down placement
- rebuild the visual layer from first principles afterward

That is a much healthier position than the reverse.

## 20. Short Mental Model

If you only keep one model in your head, make it this:

Dagua is a layered graph compiler.

It takes a user-authored graph plus style/config intent, compiles that into sized nodes and constraints, solves for positions through a differentiable optimizer, then compiles the result again into routed edges, labels, renders, animations, reports, and galleries.

The optimizer is the engine.
The graph object is the contract.
The render layer is the presentation.
The benchmark and visual-audit layers are the reality check.
