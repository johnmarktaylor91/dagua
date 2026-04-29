# Report B2 — graphviz_strict cosmetic parity

Date: 2026-04-29
Base: `develop` before B2 edits

## Summary

Round B2 implemented the five A2 high-severity cosmetic findings that were safe
inside theme/render/import scope. The run improved `tiny_graph` SSIM in the
targeted sparse-panel check, restored normal arrowheads to a triangular outline
for ordinary Graphviz-scale edges, restored strict edge labels to 14pt, and
threaded native `arrowsize` through DOT import and strict Graphviz attr shims.

The figure-aspect fix is intentionally limited to strict renders that do not
receive an explicit `figsize`. The parity pixel harness passes the native dot
canvas size explicitly, so overriding that path produced centered padding and
worse sparse-panel scores. This is a deviation from the original F1 numeric
target; the safer implementation keeps API-provided canvases authoritative.

## Before / After

Baseline numbers are B1/A2 current HEAD before B2 edits.

### `scripts/parity_metrics.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Panels in tolerance | 37 / 45 | 37 / 45 |
| Features compared | 7371 | 7371 |
| In tolerance | 7317 | 7317 |
| Out of tolerance | 54 | 54 |
| In-tolerance % | 99.27% | 99.27% |
| `ellipse_rx_pt` in tolerance | 464 / 487 | 464 / 487 |
| `ellipse_aspect_pct` in tolerance | 463 / 487 | 463 / 487 |
| `arrow_width_pt` in tolerance | 639 / 643 | 639 / 643 |

### `scripts/parity_pixel_diff.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Full 45-panel mean L1 RGB/px | 16.9495 | 17.1183 |
| Full 45-panel mean SSIM | 0.761481 | 0.759206 |
| Full 45-panel worst SSIM | 0.529021 | 0.522577 |
| Targeted mean SSIM (`pipeline,tiny_graph,single_edge`) | 0.683031 | 0.689060 |
| `tiny_graph` SSIM | 0.679015 | 0.707681 |
| `pipeline` SSIM | 0.717735 | 0.716180 |
| `single_edge` SSIM | 0.652343 | 0.643320 |

## Fix Outcomes

### F1 — Figure Aspect Mismatch

Implemented strict content-bbox figure sizing for graphviz_strict renders when
the caller does not pass `figsize`: `figsize=(bbox_w_pt/72, bbox_h_pt/72)`.
Explicit caller canvases remain authoritative because the pixel harness already
passes dot's native pixel dimensions. Overriding explicit `figsize` made raw
Dagua renders smaller than the reference canvas and increased center padding.

Outcome: partial. The F1 requested `+0.05` SSIM jump did not materialize across
all three targeted panels. `tiny_graph` improved by `+0.028666`, while
`pipeline` and `single_edge` moved slightly negative.

### F2 — Arrowhead Polygon Triangle

Restored the ordinary `normal` primitive to a three-vertex filled isosceles
triangle. Thick ribbon edges still use the joined-neck geometry required by the
existing custom-edge tests.

Outcome: implemented. Unit coverage stays green and Graphviz-scale arrows no
longer use the squat joined-kite/rhombus geometry.

### F3 — Per-edge `arrowsize`

Added `arrowsize` parsing in `from_dot()` and a graphviz_strict shim for
`_graphviz_edge_attrs` so native Graphviz per-edge `arrowsize` reaches
`EdgeStyle.arrowsize` before render and metric extraction.

Outcome: implemented for real per-edge attributes. The current `arrow_types`
parity fixture still reports the same four `arrow_width_pt` misses because
those misses are shape-extractor differences in the fixture, not authored
`arrowsize` attrs in the source graph.

### F4 — Short Single-line Ellipses

Added a render-only compact-ellipse oval floor for graphviz_strict:
small ellipses widen to at least `1.85:1` visual aspect. The gate is limited to
compact ellipses to avoid disturbing long-label metric sizing.

Outcome: implemented visually. Declarative ellipse metrics remain unchanged.

### F5 — Long-label Ellipses and Edge-label Font

Restored graphviz_strict `edge_label_font_size` to `14.0pt` and updated the
theme assertion. A first attempt at length-dependent ellipse width scaling
overshot many already-good labels, so it was reverted to avoid a broad metric
regression.

Outcome: edge-label font implemented; long-label rx remains a residual
(`ellipse_rx_pt` stays `464/487` in tolerance).

## Verification

Commands run:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py tests/test_parity_metrics.py -x --tb=short -q
python scripts/parity_metrics.py
python scripts/parity_pixel_diff.py
pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
```

Results:
- `ruff check . --fix`: passed
- `mypy --follow-imports=silent dagua/cli.py`: passed
- Targeted pytest: `260 passed, 1 warning in 52.86s`
- `python scripts/parity_metrics.py`: completed, `99.27%` in tolerance
- `python scripts/parity_pixel_diff.py`: completed, mean SSIM `0.759206`
- Final non-slow pytest: blocked during collection by existing import error:
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`
  in `tests/test_classic_drl.py`

## Deviations / Concerns

- F1 did not meet the requested `+0.05` per-panel SSIM target. The harness's
  explicit dot-sized canvas conflicts with unconditional strict figure-size
  override.
- Full-suite pixel SSIM regressed slightly despite targeted `tiny_graph`
  improvement. The restored 14pt edge labels and compact oval widening are
  visually aligned with A2 but not globally SSIM-positive.
- `arrow_types` metric residuals remain because the fixture misses are
  shape-specific extraction differences, not actual `arrowsize` propagation.

## Dead Code

No newly unreachable code was identified.
