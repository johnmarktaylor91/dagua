# Report B3 — graphviz_strict cosmetic parity

Date: 2026-04-29
Base: `develop` after B2

## Changes

- `dagua/render/mpl.py`: 6 changed lines.
  - `_GRAPHVIZ_STRICT_MIN_OVAL_ASPECT`: `1.85 -> 1.50`.
  - Added `_GRAPHVIZ_STRICT_EDGE_WIDTH_RENDER_MULTIPLIER = 1.2`.
  - `_edge_style_for_render()` now applies the 1.2x width multiplier only during
    `graphviz_strict` rendering. The stored theme width remains Graphviz's
    declarative `1.0pt`, so parity metrics for `edge_stroke_width_pt` remain
    locked.

## Fix Outcomes

### F1 — Compact Oval Floor

Implemented. The strict compact-ellipse visual aspect floor is back at `1.50`,
matching the dot short-label aspect called out by A3.

### F2 — Darker Edge Stroke

Implemented as a render-layer multiplier, not a theme declaration change. This
keeps metrics at `1.0pt` while drawing strict edges at `1.2x` coverage to reduce
matplotlib antialiasing washout.

### F3 — Long-label Ellipse Kerning

Attempted and reverted. The gated label-length compensation with
`1.0 + (len(label) - 10) * 0.005` preserved declarative metrics but regressed
pixel parity versus F1/F2-only:

| Variant | Mean L1 RGB/px | Mean SSIM | Worst SSIM |
| --- | ---: | ---: | ---: |
| F1/F2/F3 attempt | 17.2364 | 0.759780 | 0.526320 |
| F1/F2 only | 17.2260 | 0.759966 | 0.526320 |

Per the B3 gating instruction, F3 remains a render-stack residual.

## Before / After

### `python scripts/parity_metrics.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Panels in tolerance | 37 / 45 | 37 / 45 |
| Features compared | 7371 | 7371 |
| In tolerance | 7317 | 7317 |
| Out of tolerance | 54 | 54 |
| In-tolerance % | 99.27% | 99.27% |
| `ellipse_rx_pt` in tolerance | 464 / 487 | 464 / 487 |
| `ellipse_aspect_pct` in tolerance | 463 / 487 | 463 / 487 |
| `edge_stroke_width_pt` in tolerance | 646 / 646 | 646 / 646 |
| `arrow_width_pt` in tolerance | 639 / 643 | 639 / 643 |

### `python scripts/parity_pixel_diff.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Full 45-panel mean L1 RGB/px | 17.1183 | 17.2260 |
| Full 45-panel mean SSIM | 0.759206 | 0.759966 |
| Full 45-panel worst SSIM | 0.522577 | 0.526320 |

## Assumptions

- The B2 report numbers are the baseline for B3; I also re-ran both parity
  scripts before editing and reproduced them.
- F2 was interpreted as a visual stroke-darkening fix. Changing the actual
  `graphviz_strict` theme width would intentionally break the locked
  declarative `edge_stroke_width_pt` metric, so the 1.2x width is render-only.

## Test Results

```bash
ruff check . --fix
# All checks passed!

mypy --follow-imports=silent dagua/cli.py
# Success: no issues found in 1 source file

pytest tests/test_parity_metrics.py -x --tb=short -q
# 1 passed in 4.21s

pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py tests/test_parity_metrics.py -x --tb=short -q
# 260 passed, 1 warning in 52.38s

python scripts/parity_metrics.py
# in-tolerance: 7317 / 7371, 99.27%

python scripts/parity_pixel_diff.py
# mean L1 RGB/px: 17.2260
# mean SSIM: 0.759966
# worst SSIM: 0.526320
```

The warning is the existing matplotlib "More than 20 figures have been opened"
warning from `tests/test_render/test_mpl.py`.

## Controversial Choices

- F2 uses `_edge_style_for_render()` rather than `GRAPHVIZ_STRICT_THEME`.
  This preserves the metric contract while addressing the rasterized visual
  stroke issue.
- F3 was not kept despite being targeted, because the first controlled attempt
  regressed pixel parity against the accepted F1/F2-only result.

## Concerns

- Mean L1 worsened slightly while SSIM improved. This is consistent with darker
  strokes increasing absolute pixel difference in some regions while improving
  structural alignment.
- Long-label ellipse `rx` remains the same declarative residual as B2.

## Knowledge

- `parity_metrics.py` reads edge stroke width from the strict-themed graph
  style, not from the matplotlib artist. Render-only width changes can improve
  pixel output without breaking `edge_stroke_width_pt`.
- The 1.85 compact oval floor was purely render-side and invisible to the
  declarative ellipse metrics, which is why A3 caught it via pixel inspection.

## Dead Code

No newly unreachable code was introduced.
