# Dagua — Implementation Guide

## Build & Packaging

- Build system: setuptools via `pyproject.toml`
- Install (dev): `uv pip install -e ".[dev]"`
- Install (test): `uv pip install -e ".[test]"`
- CLI entry point: `dagua` → `dagua.cli:main`
- Version: `pyproject.toml:project.version` + `dagua/__init__.py:__version__`
- Release: semantic-release v9 on push to main, publishes to PyPI via OIDC

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

## Quality Gates (every Codex task must pass)
```
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/ -x --tb=short -m "not slow and not benchmark and not rare"
```

## Linting & Type Checking

- `ruff format` + `ruff check --fix` (line-length 100, target py39)
- mypy strict module: `dagua/cli.py` (check_untyped_defs + disallow_untyped_defs)
- mypy broader check: `dagua/eval/visual_audit.py`, `dagua/layout/multilevel.py`
- Keep the strict CLI module passing; treat broader check as debt-reduction pressure

## PR Workflow
```bash
# Create
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
│   ├── engine.py        # optimization loop — wires flex/pin/align constraints
│   ├── constraints.py   # DAG, Repel, Attract, Overlap, Cluster, Pin, Align, FlexSpacing
│   ├── projection.py    # hard overlap + hard pin projection
│   ├── schedule.py      # annealing schedules for constraint weights
│   ├── init_placement.py # topological sort (y) + barycenter (x) initialization
│   ├── layers.py        # layer assignment algorithms
│   ├── multilevel.py    # multilevel/coarsening layout
│   ├── cycle.py         # cycle detection and handling
│   └── edge_optimization.py # edge-aware position refinement
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
