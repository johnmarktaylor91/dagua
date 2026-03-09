# CHANGELOG


## v0.0.0 (2026-03-09)

### Chores

- Add project structure, CI/CD plumbing, and module scaffolding
  ([`436752c`](https://github.com/johnmarktaylor91/dagua/commit/436752c6155b825dea443645b1e421d8f999d12d))

- Full source layout: elements, graph, style, defaults, io, routing, utils - Layout subpackage:
  engine, constraints, projection, schedule - Render subpackage: mpl, svg, graphviz - CI/CD: lint
  (ruff auto-fix), quality (mypy + pip-audit), release (semantic-release v9 + PyPI OIDC) -
  Pre-commit hooks: trailing-whitespace, EOF fixer, check-yaml, large files, ruff - pyproject.toml:
  coverage, mypy, semantic-release config - CLAUDE.md documentation for all subpackages, tests,
  benchmarks, examples - Test scaffolding mirroring source structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Initial project scaffold
  ([`4a53fea`](https://github.com/johnmarktaylor91/dagua/commit/4a53feab837e8b3a7d7980ce9ac2a7ba92ce75df))

Dagua — GPU-accelerated differentiable graph layout engine built on PyTorch. Project structure,
  pyproject.toml, LICENSE (MIT), README, CLAUDE.md.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
