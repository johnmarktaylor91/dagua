# Graphviz Cosmetic Parity Sprint — Final Summary

**Period:** 2026-04-27 to 2026-04-29
**Outcome:** ceiling reached — no further fixable cosmetic gaps without breaking declarative parity or porting dot's label-fitter (multi-day, layout-scope).

## Final state

### Declarative parity (per `scripts/parity_metrics.py`)
- **99.27% in tolerance** across 7,371 measurements (45 panels × 19 features)
- 14 features at 100% lock; 3 more at 99%+
- Out-of-tolerance residual: long-label `ellipse_rx_pt` (matplotlib TextToPath kerning gap vs Cairo on labels >= 10 chars)

### Pixel parity (per `scripts/parity_pixel_diff.py`)
- Mean L1 RGB / pixel: **17.62**
- Mean SSIM: **0.759**
- Worst-panel SSIM: 0.526 (`bipartite_5x5`)

### Locked + regression-tested
- `tests/test_parity_metrics.py` asserts global in-tol stays >= 94% AND each of 14 features stays at 100%

## Iteration log

| Round | Type | Outcome | Notes |
|---|---|---|---|
| (pre-loop) | infra | parity_pixel_diff.py + audit template + procedure docs | commit 64a0936 |
| A1 | audit | FAIL, 5 HIGH findings | canvas-fill, label-wrap, ellipse_rx, arrows×4, cluster (deferred) |
| B1 | fixes | committed 9c14892 | 95.74% → 99.27%, ellipse_ry max delta 40pt → 2.8pt |
| A2 | audit | FAIL, 5 HIGH | figure aspect phase 2, arrowhead rhombus regression, arrowsize, ellipse aspect, edge label |
| B2 | fixes | committed 27646de | F2/F4/F5 PASS, F1 partial, F5a long-rx reverted (regression) |
| A3 | audit | FAIL, 5 HIGH | oval floor over-correction, edge stroke gray, long-label rx |
| B3 | fixes | committed 6a931aa | F1/F2 landed, F3 reverted (long-label rx broad regression) |
| A4 | audit | **STOP** verdict | dot rasterizer ~26% rx inflation that SVG attrs don't reflect = render-stack floor |
| B4 | fixes | committed b00f434 | edge-stroke 1.5x + capstyle + opacity — slight metric regression confirms ceiling |

## What we hit

The dominant remaining gap is **structural, not cosmetic**: dot's PNG rasterizer renders ellipses ~26% wider than the rx attribute it declares in SVG. dagua faithfully follows the declared rx. Closing this would either require (a) breaking declarative parity by inflating rx by 26%, or (b) porting dot's label-fit pipeline to dagua (multi-day project, crosses into layout-scope per the orthogonality rule).

We also exercised every theme-level + render-level cosmetic lever the audit could find. After B4, each new fix produces a metric wash or slight regression — clear ceiling signal.

## Accepted residuals (principled, all classified)

1. **matplotlib TextToPath kerning gap** vs Cairo on labels ≥ 10 chars → declarative `ellipse_rx_pt` long-label tail. Render-stack residual.
2. **dot PNG rasterizer rx inflation ~26%** → dominant pixel SSIM gap. Render-stack residual that would break declarative parity to fix.
3. **B-spline edge routing** (e.g. bipartite_5x5 outer column edges bow as splines) → layout-scope per architecture orthogonality rule.
4. **Cluster bounding box geometry** (nested_clusters cluster overlaps, node-A protrusion, sibling overlap) → layout-scope, deferred since A1.
5. **Star arrowhead small-angle stroke softness** at acute apex → matplotlib AA + FreeType anti-alias floor at thin perpendicular stroke widths.

## What's NOT residual — locked at 100%

- font_size_pt, font_family (TeX Gyre Termes via Times,serif alias)
- node_fill, node_stroke, node_stroke_width_pt
- edge_stroke_color, edge_stroke_width_pt (declared)
- bg_color, margin_pt
- cluster_fill, cluster_stroke, cluster_stroke_width_pt
- cluster_label_font_size_pt
- ellipse_ry_pt (99.38%)
- arrow_filled, arrow_length_pt, arrow_width_pt (99%+)

## Commits this sprint

```
b00f434 feat(theme): graphviz_strict round B4 — edge stroke crispness final pass
6a931aa feat(theme): graphviz_strict round B3 — oval floor 1.50, edge stroke darker, long-label kerning
27646de feat(theme): graphviz_strict round B2 — figure aspect, arrowhead triangle, arrowsize, ellipse aspect, edge label font
9c14892 feat(theme): graphviz_strict round B1 — canvas fill, label wrap, kerning, arrow defects
64a0936 feat(parity): rock-solid visual iteration infrastructure (pixel diff, hi-res inspection, audit template)
```

Plus prior round 17 metric-driven baseline:
```
738d016 feat(parity): conditional margin + principal-axis arrow metric + regression test
e113fdf feat(scripts): add parity_metrics.py
009652c feat(theme): graphviz_strict metric-driven values match dot SVG declarations
```

## Major lessons (recorded globally)

- Visual parity is optimization, not review (`~/.claude/CLAUDE.md` Visual + AI iteration loops)
- Watcher self-match deadlock (`~/.claude/knowledge/watcher_self_match_postmortem.md`)
- Multi-round autonomous loops require state file + PID-based watchers (`~/.claude/CLAUDE.md`)

## Status

**DONE.** Sent to JMT via iMessage with 5 representative side-by-side images.
