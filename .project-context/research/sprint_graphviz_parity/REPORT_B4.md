# Report B4 -- graphviz_strict edge stroke crispness final pass

Date: 2026-04-29
Base: `develop` after B3

## Changes

- `dagua/render/mpl.py`: 4 changed lines.
  - `_GRAPHVIZ_STRICT_EDGE_WIDTH_RENDER_MULTIPLIER`: `1.2 -> 1.5`.
  - `_edge_style_for_render()` now forces `line_cap="butt"`,
    `line_join="miter"`, and `opacity=1.0` for `graphviz_strict` render-local
    edge styles.

## Fix Outcome

### F1 -- Stronger Edge Stroke Crispness

Implemented. The declarative `EdgeStyle.width` remains `1.0pt`; only the
matplotlib render-local style is inflated to compensate for antialiasing.

The 1.5x pass was visually checked on:

- `eval_output/parity_pixel_diff/hires/bipartite_5x5/dagua.png`
- `eval_output/parity_pixel_diff/hires/bipartite_5x5/dot.png`

The dagua edge strokes no longer read as gray against dot's solid black on the
hi-res panel, so the 1.7x fallback was not attempted.

## Before / After

### `python scripts/parity_metrics.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Panels in tolerance | 37 / 45 | 37 / 45 |
| Features compared | 7371 | 7371 |
| In tolerance | 7317 | 7317 |
| Out of tolerance | 54 | 54 |
| In-tolerance % | 99.27% | 99.27% |
| `edge_stroke_width_pt` in tolerance | 646 / 646 | 646 / 646 |

### `python scripts/parity_pixel_diff.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Full 45-panel mean L1 RGB/px | 17.2260 | 17.6229 |
| Full 45-panel mean SSIM | 0.759966 | 0.758500 |
| Full 45-panel worst SSIM | 0.526320 | 0.523260 |

The pixel score regression is expected for this narrow cosmetic change: the
rendered strokes are darker and closer by eye, while the dominant residual
geometry mismatch remains the dot-rasterizer-vs-matplotlib render stack.

## Assumptions

- "Stop when visually match" was interpreted as the hi-res `bipartite_5x5`
  edge strokes no longer appearing gray relative to dot. The 1.5x pass met that
  bar, so 1.7x was not tried.
- Existing untracked A*/prompt files in the parity research directory were left
  untouched.

## Test Results

```bash
ruff check . --fix
# All checks passed!

mypy --follow-imports=silent dagua/cli.py
# Success: no issues found in 1 source file

pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q
# 258 passed, 1 warning in 1220.66s (0:20:20)

python scripts/parity_metrics.py
# in-tolerance: 7317 / 7371, 99.27%

python scripts/parity_pixel_diff.py
# mean L1 RGB/px: 17.6229
# mean SSIM: 0.758500
# worst SSIM: 0.523260

pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"
# ERROR collecting tests/test_classic_drl.py
# ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'
```

The broad final suite failed at collection before executing tests. The failure is
outside this task's theme/render scope: `dagua/layout/classic/` is an implicit
namespace of symlinks with no `layout_drl` package export.

## Controversial Choices

- Kept the 1.5x result despite worse mean L1/SSIM because this round's success
  criterion was visual stroke darkness, not global pixel score. A4 already
  identified the remaining pixel gap as dominated by render-stack geometry
  residuals.

## Concerns

- The final non-slow suite currently cannot collect `tests/test_classic_drl.py`.
  That should be handled in a layout/classic export task, not this cosmetic
  round.
- The 1.5x stroke improves visual darkness but cannot address the remaining
  Cairo-vs-matplotlib geometry residuals.

## Knowledge

- `graphviz_strict` edge darkness can be adjusted render-locally without moving
  `edge_stroke_width_pt`, because parity metrics read the declared style rather
  than the matplotlib artist.

## Dead Code

No newly unreachable code was introduced.
