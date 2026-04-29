# Report B1 — graphviz_strict cosmetic parity

Date: 2026-04-28
Base: `64a0936` (`develop`)

## Summary

Round B1 implemented the A1 high-severity theme/render fixes that were in
scope for the cosmetic sprint. Layout-scope cluster geometry was left
unchanged.

## Before / After

Baseline was measured from a clean temporary worktree at `HEAD` before the B1
edits.

### `scripts/parity_metrics.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Panels in tolerance | 14 / 45 | 37 / 45 |
| Features compared | 7371 | 7371 |
| In tolerance | 7057 | 7317 |
| Out of tolerance | 314 | 54 |
| In-tolerance % | 95.74% | 99.27% |
| `margin_pt` median / max delta | 14.00 / 14.00 | 0.00 / 0.00 |
| `ellipse_rx_pt` in tolerance | 349 / 487 | 464 / 487 |
| `ellipse_aspect_pct` in tolerance | 320 / 487 | 463 / 487 |
| `ellipse_ry_pt` max delta | 40.47pt | 2.76pt |

### `scripts/parity_pixel_diff.py`

| Metric | Before | After |
| --- | ---: | ---: |
| Full 45-panel mean L1 RGB/px | 17.3736 | 16.9495 |
| Full 45-panel mean SSIM | 0.771616 | 0.761481 |
| Full 45-panel worst SSIM | 0.584576 | 0.529021 |
| Representative mean L1 (`pipeline,tiny_graph,colors_showcase`) | 27.4639 | 20.9864 |
| Representative mean SSIM (`pipeline,tiny_graph,colors_showcase`) | 0.706446 | 0.703732 |

## Fix Outcomes

### F1 — Canvas Fill

Implemented for `graphviz_strict`:
- strict graph margin changed from 18pt to 4pt, matching dot's tight content
  bbox convention in the sampled panels
- strict saves bypass Matplotlib `bbox_inches="tight"` and use a full-figure
  axes box
- strict axes use `aspect="auto"` so an already Graphviz-sized canvas is not
  letterboxed by Matplotlib's equal-aspect box adjustment

Outcome: `margin_pt` delta went from 14pt on all panels to 0pt on all panels.
Representative panel L1 dropped from 27.4639 to 20.9864. Full-suite L1 improved
slightly, while SSIM regressed; this is documented as a residual to inspect
because the content is better scaled but remaining x-width/text differences
still affect structural similarity.

### F2 — Auto-Wrap / Long Labels

The observed failure was not a plain-text auto-wrap path. `text_wrap` was
already `"none"` under `graphviz_strict`; the tall ellipses came from
graphviz_strict's ellipse aspect caps. Removed the strict curved-node aspect
cap for expand-node sizing and bypassed the final 10:1 global ratio guard only
for strict curved expand-node sizing.

Outcome: `long_labels.n3` no longer inflates to a ~58pt semi-height; global
`ellipse_ry_pt` max delta dropped from 40.47pt to 2.76pt.

### F3 — Ellipse RX Narrowing / Kerning Investigation

Investigated `matplotlib.ft2font.FT2Font.get_kerning()` with the TeX Gyre
Termes `qtmr.pfb` face used by strict Times rendering. For tested labels such
as `Postprocess`, `attention_block`, and
`conv2d_batch_norm_relu_dropout_3`, cumulative kerning was 0.0pt and
`linearHoriAdvance` matched `TextToPath` width. The kerning hypothesis was not
confirmed.

Implemented a conservative strict single-line ellipse width factor of 1.28
while keeping multiline labels at the existing 1.22 factor.

Outcome: `ellipse_rx_pt` in-tolerance improved from 349/487 to 464/487 and
`ellipse_aspect_pct` from 320/487 to 463/487.

### F4 — Arrow Defects

Implemented:
- added `EdgeStyle.arrowsize` and applied it to fixed arrow length/width in
  render marker sizing
- restored Graphviz `open` to a stroked chevron instead of aliasing it to a
  filled normal polygon
- set strict graph-level edge label font size to 11pt for visual subordination
  versus node labels

Deferred / residual:
- The existing parity extractor still reports four `arrow_width_pt` misses and
  one `arrow_filled` miss on `arrow_types`. Those are tied to shape-specific
  SVG extraction for `vee`, `tee`, `crow`, and `open`, not the newly added
  `arrowsize` multiplier path.
- Edge stroke darkness was not changed. Declarative `edge_stroke_width_pt` and
  `edge_stroke_color` remained 100% in tolerance; the remaining visible
  difference appears rasterizer/AA-sensitive.

### F5 — Nested Cluster Geometry

Unchanged by design. The A1 nested-cluster findings involve cluster bbox
sizing, sibling overlap, and label-content overlap, which are layout-scope per
the architecture rule for this sprint.

## Verification

Commands run:

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_style.py tests/test_render tests/test_custom_edges.py tests/test_arrowheads.py tests/test_parity_metrics.py -x --tb=short -q
python scripts/parity_metrics.py
python scripts/parity_pixel_diff.py
```

Results:
- `ruff check . --fix`: passed
- `mypy --follow-imports=silent dagua/cli.py`: passed
- Targeted pytest: `260 passed, 1 warning in 50.34s`
- Full parity metrics and pixel diff: completed; numbers above

## Deviations / Concerns

- Full-suite pixel SSIM regressed even though full-suite mean L1 improved and
  representative sparse-panel L1 improved substantially. This needs a visual
  audit before further canvas scaling changes.
- `arrow_types` still has extractor-visible arrow geometry misses; shape
  primitives should be audited against per-shape dot SVG paths in a later arrow
  round.
- F3 kerning API was available, but did not explain the width gap for tested
  labels.

## Dead Code

No newly unreachable code was identified.
