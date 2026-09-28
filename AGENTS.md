# Dagua project instructions

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

## Project Overview

Dagua is a GPU-accelerated, differentiable graph layout engine built on PyTorch. It replaces
Graphviz's layout algorithms with continuous optimization: node positions are learnable
parameters, layout aesthetics are loss functions, and the solver is `loss.backward();
optimizer.step()`. Named after the Dagua River in Colombia — DAG + agua.

## Architecture

See `internal-notes/architecture.md` for the full map.

Key entry points:
- main: `dagua/__init__.py` (public API), `dagua/graph.py` (DaguaGraph), `dagua/cli.py` (CLI)
- tests: `pytest tests/ -x --tb=short`
- build: `uv pip install -e ".[dev]"` / `python -m build`

## How to Read This Codebase

- Start with `dagua/__init__.py` for the public API surface, then `dagua/graph.py` for DaguaGraph
- Layout algorithms live in `dagua/layout/ops/` (268 registered ops) composed into pipelines
  in `dagua/layout/ops/pipelines/` (23 algorithms). The engine in `dagua/layout/engine.py`
  dispatches to these pipelines via `LayoutConfig(algorithm="fr")`.
- Complexity lives in `dagua/layout/engine.py` (optimization loop + pipeline dispatch),
  `dagua/layout/ops/` (composable primitives), and `dagua/eval/` (benchmark/comparison system)
- Stable: graph data model, styles, config, render backends, composable ops, algorithm pipelines
- In-flux: eval system, edge optimization, default algorithm tuning

## Design Principles

1. **PyTorch is the only required dependency** (matplotlib optional for rendering)
2. **Constraints are composable loss functions** — users can write custom ones in 3 lines
3. **Layout and rendering are separate** — `layout()` returns coordinates
4. **Layout engine is headless** — takes tensors, not Graph objects
5. **Domain knowledge enters as loss terms**, not algorithmic modifications
6. **GPU acceleration is automatic** — same code on CPU or CUDA via `device=`
7. **Steal Graphviz's aesthetic wisdom** (spacings, weight ratios) but not its architecture
8. **Layout algorithms are composable op pipelines** -- 268 registered primitives, 23 algorithms
   built purely by composing them. Monolithic reimplementations archived at `dagua/layout/_archive/`.

## Key API Surface

```python
import dagua

# Tier 0: Zero config
dagua.draw(g)
dagua.set_theme("dark")

# Tier 1: Flat configure + flex
dagua.configure(font_size=10, node_sep=40, background_color="#1A1E24")
with dagua.defaults(theme="minimal"):
    dagua.draw(g)

# Flex: soft layout targets
g.pin("input", x=0, y=0)
g.align(["a", "b", "c"], axis="x")
config = dagua.LayoutConfig(
    flex=dagua.LayoutFlex(
        node_sep=dagua.Flex.firm(40),
    )
)

# Tier 1.5: Algorithm selection
config = dagua.LayoutConfig(algorithm="fr")  # Fruchterman-Reingold
config = dagua.LayoutConfig(algorithm="kk", algorithm_params={"spring_k": 1.5})
# 24 algorithms: fr, kk, fa2, stress_sgd, sfdp, umap, tsnet, sugiyama, spectral, ...

# Tier 2: Full control
pos = dagua.layout(g, config)
dagua.render(g, pos, config, output="graph.png")
```

## Relationship to TorchLens

Dagua is standalone. TorchLens is a downstream consumer that provides domain-specific
neural network constraints. The dependency is one-way: TorchLens depends on Dagua.

## Dispatch Configuration

### Branch Strategy
Implementers work on feature branches: `task/<task-id>`
Default: one branch at a time besides main. Don't spawn extras unless asked.

### Task ID Convention
Descriptive kebab-case: `fix-trace-module`, `add-sweep-dataclass`

## What NOT to Dispatch

- API design decisions (discuss with user first)
- Changes to public interfaces without approval
- CI/CD, deployment, release configs
- AGENTS.md, internal-notes/*.md modifications

## Documentation Maintenance

Full checklist: `docs/MAINTENANCE_CHECKLIST.md`

Generated docs (rebuild when their sources change):
- Glossary: `scripts/build_glossary.py` (API/config/styles/losses/metrics/CLI)
- Gallery: `scripts/build_gallery.py` (rendering defaults/example graphs)
- Explainer: `scripts/build_how_dagua_works.py` (layout pipeline/routing)
- Visual audit: `scripts/build_visual_audit.py`
- Notebooks + `docs/LLM_TUTORIAL.md`: update when API shape or workflows change

## Benchmark Iteration

- **Data integrity:** `results.json` and `positions/` (per-run .pt files) MUST stay in sync.
  `fidelity_analysis.py` aborts if >10 engines have results but missing positions.
  Some scripts also support consolidated `positions.h5` (HDF5) format.
  - Validate: `python scripts/validate_benchmark_integrity.py`
  - Purge variants: `python scripts/safe_purge_variants.py <engines> --confirm`
    (purges BOTH stores atomically -- never purge one manually)
  - Post-analysis: `python scripts/validate_fidelity_output.py --data <dir> --previous <old_dir>`
- Benchmark/report workflow is persistent and reuse-aware (non-Dagua competitors cached)
- Long runs write `progress.json`; check with `dagua benchmark-status` or `dagua benchmark-watch`
- Key report artifacts: `eval_output/report/benchmark_deltas.md`, `layout_similarity.md`,
  `placement_summary.md`, `placement_tuning.md`
- Placement iteration loop:
  1. `dagua placement-tune --output-dir eval_output/report`
  2. Inspect `placement_tuning.md`
  3. Validate visually in `visual_review_session`
  4. Rerun standard benchmark with cached competitors

## Build & Packaging

- Build system: setuptools via `pyproject.toml`
- Install (dev): `uv pip install -e ".[dev]"`
- Install (test): `uv pip install -e ".[test]"`
- CLI entry point: `dagua` → `dagua.cli:main`
- Version: `pyproject.toml:project.version` + `dagua/__init__.py:__version__`
- Release: semantic-release v9 on push to main publishes to PyPI via OIDC, so push to main only when JMT asks.

## Commit Convention

Conventional commits: `<type>(<scope>): <description>`

Types: `fix:` (patch), `feat:` (minor), `feat!:` (major), `chore:`, `docs:`, `ci:`,
`refactor:`, `test:`, `perf:`.

## Testing Tiers

```
# Tier 1 — Fast (run on every change)
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"

# Tier 2 — Medium (run when module boundaries change)
pytest tests/ -x --tb=short

# Tier 3 — Full (run during downtime / before release)
ruff check . --fix && mypy --follow-imports=silent dagua/cli.py && pytest tests/ -v
```

## Quality Gates (every implementation task must pass)

```
# Tier 1: Run FIRST during iteration (fast, targeted)
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q  # targeted

# Tier 2: Run ONCE at the end as final verification
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

During development iteration, ONLY run Tier 1 (targeted tests for the modules
you changed). Run Tier 2 ONCE as the final check before reporting done. Do NOT
run the full suite on every iteration — it takes 5+ minutes and wastes time.
Match test files to changed modules:
- Changed `engine.py` -> run `tests/test_layout/`
- Changed `constraints.py` -> run `tests/test_layout/test_constraints.py`
- Changed `multilevel.py` -> run `tests/test_layout/`
- Changed `graph.py` -> run `tests/test_graph.py`
- Changed `utils.py` -> run `tests/test_smoke.py tests/test_layout/`
- Changed `config.py` -> run `tests/test_layout/test_engine.py`
- Changed `layout/ops/<name>.py` -> run `tests/test_ops_<name>.py`
- Changed `layout/ops/pipelines/<name>.py` -> run `tests/test_pipeline_<name>.py`

## Code Quality Standards (mandatory for ALL tasks)

Every function you create or modify MUST have:
- **Type hints** on all parameters and return values. Use `Optional`, `Union`,
  `Tuple`, `List`, `Dict` from typing. For torch tensors, annotate as
  `torch.Tensor` with shape/dtype documented in the docstring.
- **Docstring** in NumPy format: short summary, Parameters, Returns, and
  optionally Notes/Examples. Include tensor shapes like `[N, 2]` in parameter
  descriptions.
- **Meaningful comments** for non-obvious logic. Don't comment `x += 1` but DO
  comment algorithmic choices, magic numbers, and performance tradeoffs. If you
  know WHY something is done a certain way, say so.

Leave every file you touch BETTER than you found it:
- If a function you're calling lacks type hints, add them.
- If a function you're modifying lacks a docstring, add one.
- If you see a magic number in code you're editing, name it as a constant.
- If you see a stale comment that contradicts the code, fix it.

Scope: improve what you TOUCH. Don't rewrite unrelated functions or go on
cleanup crusades through files you weren't asked to modify.

**Tests are ALWAYS in scope.** When a spec says "do not modify other files,"
test files are exempt — you MUST create or update tests for your changes.
`tests/` is never out of scope. If the spec includes a Testing section,
implement those tests. If it doesn't, write your own regression tests for
the functionality you added or changed.

Context for smart comments: the spec should include enough project knowledge
for you to write comments that explain WHY, not just WHAT. If you don't know
why something is done, say so in the comment rather than guessing.

Key project facts for comment context:
- dagua is a GPU-accelerated graph layout engine using PyTorch
- Positions are learnable parameters, layout aesthetics are loss functions
- The engine is headless: takes tensors, not Graph objects
- Multilevel coarsening handles N > 20K; direct layout for smaller
- Constraints are composable: `(pos, graph_data) -> scalar loss`
- GPU acceleration is automatic via `device=` parameter
- Memory management matters: OOM at 1B nodes is unacceptable

## Linting & Type Checking

- `ruff format` + `ruff check --fix` (line-length 100, target py39)
- mypy strict module: `dagua/cli.py` (check_untyped_defs + disallow_untyped_defs)
- mypy broader check: `dagua/eval/visual_audit.py`, `dagua/layout/multilevel.py`
- Keep the strict CLI module passing; treat broader check as debt-reduction pressure

## PR Workflow

```bash
# Create (only when JMT asks: a PR is outward in his name)
gh pr create --title "<title>" --body "<description>"

# After merge (user says "merged" or "clean up")
git checkout main && git pull origin main
git branch -d <branch> && git remote prune origin
```

## Project Structure

```
dagua/
├── __init__.py          # public API re-exports + draw() convenience function
├── graph.py             # DaguaGraph — central orchestrator
│                        #   holds nodes/edges/clusters, ID→index mapping
│                        #   5-level style cascade, pin/align helpers
│                        #   from_* classmethods (thin wrappers over io.py)
├── elements.py          # Node, Edge, Cluster dataclasses (pure data)
├── edges.py             # Edge label placement + edge routing
├── flex.py              # Flex, LayoutFlex, AlignGroup — soft layout targets
├── defaults.py          # thread-safe global defaults: configure(), defaults() ctx mgr
├── styles.py            # NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme, cascade
├── config.py            # LayoutConfig with all tunable parameters + flex field
├── metrics.py           # Layout quality metrics (crossings, stress, etc.)
├── routing.py           # bezier edge routing (heuristic)
├── io.py                # JSON/YAML IO, interop, LLM-based construction
├── cli.py               # CLI entry point (dagua command)
├── utils.py             # text measurement, graph topology helpers
├── graphviz_utils.py    # graphviz utility helpers
├── animation.py         # animate(), tour(), poster() — cinematic exports
├── playground.py        # interactive playground launcher
├── reference_glossary.py # glossary builder
├── showcase_gallery.py  # gallery builder
├── layout/              # [see dagua/layout/AGENTS.md]
│   ├── engine.py        # optimization loop + algorithm dispatch to ops pipelines
│   ├── constraints.py   # DAG, Repel, Attract, Overlap, Cluster, Pin, Align, FlexSpacing
│   ├── projection.py    # hard overlap + hard pin projection
│   ├── schedule.py      # annealing schedules for constraint weights
│   ├── init_placement.py # topological sort (y) + barycenter (x) initialization
│   ├── layers.py        # layer assignment algorithms
│   ├── multilevel.py    # multilevel/coarsening layout
│   ├── cycle.py         # cycle detection and handling
│   ├── edge_optimization.py # edge-aware position refinement
│   ├── ops/             # 268 registered composable layout primitives
│   │   ├── __init__.py  # op registry, @register_op, get_pipeline_function
│   │   ├── base.py      # Op protocol, registration machinery
│   │   ├── state.py     # SolveState (9 typed fields for cross-algo composability)
│   │   ├── graph_utils.py # BFS, Dijkstra, APSP, adjacency builders (self-contained)
│   │   ├── force.py     # 22 ops: repulsion, attraction, displacement, cooling
│   │   ├── stress.py    # stress majorization ops
│   │   ├── embed.py     # 14 ops: spectral, MDS, pivot MDS embedding
│   │   ├── init.py      # 15 ops: position initialization strategies
│   │   ├── anneal.py    # 12 ops: temperature/weight annealing schedules
│   │   ├── loss_engine.py  # 16 ops: engine-level loss computation
│   │   ├── loss_classic.py # 14 ops: classical loss functions
│   │   ├── optimize.py  # optimizer creation/stepping
│   │   ├── converge.py  # convergence detection
│   │   ├── postprocess.py  # 13 ops: post-layout refinement
│   │   ├── preprocess.py   # graph preprocessing
│   │   ├── coarsen.py   # multilevel coarsening ops
│   │   ├── prolong.py   # multilevel prolongation ops
│   │   ├── project.py   # constraint projection ops
│   │   ├── coordinate.py   # coordinate transforms
│   │   ├── context.py   # context managers for ops
│   │   ├── utility.py   # shared op utilities
│   │   ├── taxonomy.py  # op categorization metadata
│   │   ├── [algo].py    # per-algorithm ops (drl, fmmm, gem, lgl, neulay, ...)
│   │   └── pipelines/   # 23 algorithm pipelines (pure op composition)
│   │       ├── __init__.py      # PIPELINE_REGISTRY, get_pipeline_function()
│   │       ├── fr.py            # Fruchterman-Reingold
│   │       ├── kk.py            # Kamada-Kawai
│   │       ├── fa2.py           # ForceAtlas2
│   │       ├── stress_sgd.py    # Stress via SGD
│   │       ├── sfdp.py          # Scalable Force-Directed
│   │       ├── umap_layout.py   # UMAP layout
│   │       ├── tsnet.py         # t-SNE network embedding
│   │       ├── sugiyama.py      # Sugiyama hierarchical
│   │       ├── spectral.py      # Spectral layout
│   │       └── ... (14 more)    # classical_mds, drl, gem, graphopt, etc.
│   └── _archive/        # frozen monolithic reimplementations (reference only)
├── render/              # [see dagua/render/AGENTS.md]
│   ├── mpl.py           # matplotlib: PatchCollection, LineCollection, batched
│   ├── svg.py           # direct SVG string output (zero deps)
│   └── graphviz.py      # optional neato -n2 passthrough
├── eval/                # evaluation and benchmarking system
│   ├── benchmark.py     # benchmark runner (standard + rare suites)
│   ├── report.py        # report generation (deltas, placement, dashboards)
│   ├── compare.py       # multi-engine comparison infrastructure
│   ├── sweep.py         # placement tuning / hyperparameter sweep
│   ├── aesthetic.py     # aesthetic evaluation
│   ├── visual_audit.py  # visual audit suite builder
│   ├── graphs.py        # test graph collection for evaluation
│   ├── quick.py         # quick evaluation helpers
│   ├── runtime_env.py   # runtime environment detection
│   └── competitors/     # 6 competitor engine adapters
│       ├── base.py
│       ├── dagua_competitor.py
│       ├── graphviz_competitor.py
│       ├── elk_competitor.py
│       ├── dagre_competitor.py
│       ├── networkx_competitor.py
│       └── igraph_competitor.py
└── graphs/              # 30+ YAML reference graphs for benchmarks + eval
```

## Render Tuning Notes

Constants tuned during the cosmetic polish sprint. Preserve their visual intent
unless the task explicitly calls for retuning:

- `mpl.py:_GRAPHVIZ_DOT_PATTERN = (1.2, 3.0)` -- visible gap after pt-to-data conversion
- `mpl.py:_CROSSING_*` -- crossing-jump shape; `_SHARP_HEIGHT_WIDTH_FACTOR = 3.5` sets arch height
- `mpl.py:_SELF_LOOP_ARROWHEAD_MAX_NODE_FRACTION = 0.18` -- caps arrowheads on compact loops
- `mpl.py:_DEFAULT_EXTERNAL_LABEL_FONT_POINTS = 7.0`, `_EDGE_LABEL_HEIGHT_FRACTION = 0.18` -- quieter secondary labels
- `mpl.py` auto backgrounds: pie/striped `0.92`, hatched `0.75`, gradient `0.90` alpha
- `mpl.py` box3d: top face `0.12`, right face `0.18` alpha (faux light direction)
- `edges/collection.py:MIN_TAPER_WIDTH = 0.3` -- prevents zero-width taper endpoints
- `edges/collection.py` terminal redistribution is 8-way (cardinal+intercardinal buckets)
- `edges/dashes.py:DOTTED_ON_RATIO = 0.15`, `DOTTED_OFF_RATIO = 1.8` -- survives antialiasing
- `borders/dashes.py` curvature-adaptive: `_SENSITIVITY = 8.0`, `_MIN_SCALE = 0.4`
- `borders/shapes.py` cosmetic ratios: note fold `0.45`, star `0.25`, tab `0.38/0.28`
- `text/paths.py:_SYNTHETIC_ITALIC_SHEAR_DEGREES = 15.0` -- oblique fallback for missing italic
- `styles.py:text_outline_width = 1.4`, `utils.py:avg_char_width = 0.52*fs`, `edges.py:arc = 1.1*max(sw,sh)`

## Makefile Targets

```
make benchmark-status    # check running benchmark status
make placement-tune      # run placement tuning sweep
make visual-audit        # build visual audit suite
make visual-session      # build visual review session
make glossary            # rebuild reference glossary
make gallery             # rebuild showcase gallery
make explainer           # rebuild algorithm explainer
make artifact-index      # rebuild report artifact index
```

## Scale Work (100M+ nodes)

Read `internal-notes/knowledge/scaling_principles.md` before any task at
this scale. Key rules: budget peak memory (3-4x base), gate on topology
sketch (N+E+depth+degree), measure before choosing GPU vs CPU, every fix
creates a guardrail, test at 100K/1M/10M (not single smoke test).
