---
run: cairo
created: 2026-04-30T19:18
state: ACTIVE
current_round: 1
note: Sprint B of the post-dial-tuning workstream. Add cairo as opt-in matplotlib backend for dagua. Auto-detect default per `feedback_cairo_default_policy` (cairo if mplcairo installed, else Agg). Sprint A (data-coord) closed at round 16. Round 13/14/15 made the render path backend-agnostic by using `Figure(...)` + explicit canvas attach -- this sprint just wires the resolver.
---

# cairo -- Autonomous Loop State

## Goal

Cairo backend opt-in for dagua. After Sprint B converges:

- `pip install dagua` -> Agg by default. Zero system deps. Python-only pitch survives.
- `pip install 'dagua[cairo]'` -> mplcairo installed -> cairo by default. Best rendering quality.
- Either way, user can override per-render: `dagua.render(g, pos, backend="agg" | "cairo")`.
- `dagua.set_default_backend(name)` for global override.

The structural argument: dagua aspires to "graphviz-grade rendering quality" but currently uses matplotlib Agg, which produces sub-pixel AA + font hinting that differs from cairo (graphviz's rasterizer). Cairo backend closes this rendering-stack residual to near-zero on the gallery_audit Tier A metric.

The user's policy directive (`feedback_cairo_default_policy.md`):
- Auto-detect via try/except import; no explicit user config
- Either install profile gets the right default

## Stop criteria

PRIMARY: an Opus 4.7 visual auditor compares dagua-cairo gallery output to graphviz reference and returns ZERO findings classified as `fixable_with_cairo_tweak`. The cairo render should be visually indistinguishable from graphviz at the rasterizer level (sub-pixel AA, font hinting, stroke geometry).

SECONDARY: comparison gallery shows quantitative L1 drop on Tier A cards under cairo backend. Round-12 baseline (Agg) was mean Tier A L1 = 1.515. Cairo is predicted to drop this to near-noise-floor (auditor target: <= 0.8).

ANTI-FLAIL: 3 consecutive rounds with the same un-closeable issue -> mark `principled_residual_outside_cairo_scope`.

HARD CAP: max_rounds=10.

## Architectural plan (Round 1)

1. **`pyproject.toml`**: add `[project.optional-dependencies] cairo = ["mplcairo>=0.6"]`
2. **`dagua/render/_backend.py` (new)**: `_resolve_backend(name: str | None) -> tuple[type[FigureCanvas], str]` returns the canvas class + the resolved name. None -> auto-detect via try/except import. "cairo" with mplcairo missing -> ImportError with install instructions. "agg" -> always Agg.
3. **`dagua/render/mpl.py:1192-1196`**: replace hardcoded `FigureCanvasAgg(fig)` with `canvas_cls, _ = _resolve_backend(backend); canvas_cls(fig)`. Same for any other `FigureCanvasAgg` attach sites.
4. **Public API**: `dagua.render(g, pos, *, backend: str | None = None, ...)` and `dagua.draw(...)`. None triggers auto-detect.
5. **`dagua.set_default_backend(name: str | None)`**: global override; threads through to `_resolve_backend`.
6. **Tests**: parametrize key visual sanity tests over both backends. `pytest.mark.skipif(not _cairo_available(), reason="mplcairo not installed")` for cairo-only tests.
7. **README**: add a section documenting both install paths + the libcairo system-dep note for Linux/Mac.

## Round 2+: comparison gallery + iteration

After Round 1 makes cairo work, Round 2 generates the comparison gallery -- render the existing 174 Tier A cards under both backends side-by-side. Quantify the visual delta. This is the empirical sanity check on "cairo is meaningfully better."

If cairo's L1 drops as predicted (~1.515 -> <0.8), iterate on any cards that DIDN'T benefit -- those are the cairo-specific residuals (font fallback, gradient edge cases, etc.) and Round 3+ closes them.

## Wake-up case routing

Same pattern as Sprint A: codex committed -> regen + audit; codex still running -> ack; codex died -> investigate; quota -> pause + reset.

## Hard guardrails

- DO NOT touch `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`, `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`
- DO NOT touch `_DENSITY_LABEL_FONT_FLOOR`, `_MIN_VISIBLE_STROKE_POINTS`, `density_aware_size_factor()`
- DO NOT touch GRAPHVIZ_STRICT_THEME numerics
- DO NOT touch algo_fidelity territory (`dagua/layout/ops/*`, `dagua/eval/*`, `scripts/ogdf_*`, `tests/test_classic_*`, `tests/test_variant_*`, `tests/test_layout/test_neato.py`)
- DO NOT regress data-coord-everything (Sprint A's invariant). Cairo backend MUST work with the data-coord ribbon construction; if there's a cairo-specific path that needs display-point conversion at the rasterizer boundary, that's fine -- but the artist construction layer stays data-coord.
- Single working branch: develop. NO new branches.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Cairo backend status | Notes |
|---|---|---|---|---|---|---|
| 1 (codex) | TBD | — | — | — | — | wire optional dep + resolver + per-figure attach + public API + tests |
