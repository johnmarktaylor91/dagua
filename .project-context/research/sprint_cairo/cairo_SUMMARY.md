# Cairo Backend Sprint B -- Final Summary

**Period:** 2026-04-30 19:18 to 2026-05-01 00:34 (~5.5 hours, 3 implementation rounds + 1 audit round)
**Outcome:** Cairo opt-in shipped. Auto-detect default per the cairo policy directive. Mean Tier A L1: Agg 1.515 vs cairo 1.495 (cairo wins by 0.020 net; classical strengths win much larger but L1 metric structurally undersells them).

## Goal

Add cairo as opt-in matplotlib backend for dagua, enabling graphviz-grade rasterization quality (sub-pixel AA, font hinting, dashed-stroke completeness) for users who install `'dagua[cairo]'`. Auto-detect default per `feedback_cairo_default_policy.md`: cairo if mplcairo installed, else Agg.

## What shipped

### Round 1 (commit 5b48e16): backend wiring

- `pyproject.toml`: `[project.optional-dependencies] cairo = ["mplcairo>=0.6"]`
- `dagua/render/_backend.py` (new, 190 lines): `_resolve_backend(name | None)`, `_cairo_available()`, `get_default_backend()`, `set_default_backend(name)`
- `dagua/render/mpl.py`: every `FigureCanvasAgg(fig)` attach replaced with `canvas_cls, _ = _resolve_backend(backend); canvas_cls(fig)`
- Public API: `dagua.render(g, pos, *, backend: str | None = None, ...)`, `dagua.set_default_backend(...)`, `dagua.get_default_backend()`
- 7 backend tests pass (`tests/test_render_backend.py`); 1 test skipped (cairo-missing path; mplcairo IS installed on dev machine)
- Smoke renders under both backends produce valid output. Auto-detect default verified: `_resolve_backend(None)` returns cairo on this machine.

### Round 2 (commit cddbba1): comparison gallery + metric

- `scripts/build_gallery_audit.py`: `--backend` and `--output-dir-suffix` flags
- `scripts/per_card_pixel_diff.py`: `--gallery-dir` flag
- `scripts/build_backend_comparison_gallery.py` (new, 466 lines): generates dagua-agg | dagua-cairo | graphviz triptychs
- Generated parallel cairo gallery at `eval_output/gallery_audit_cairo/`
- `eval_output/backend_comparison/SUMMARY.md` populated with quantitative Agg-vs-cairo metrics

### Round 2 audit (`AUDIT_round_2_OPUS.md`): the surprise finding

Initial expectation: cairo would close the rendering-stack residual to near-zero (~0.7 mean Tier A L1). Reality: cairo essentially tied Agg (1.515 vs 1.513).

Three hypotheses tested:
- **Hypothesis A** (Sprint A already closed it): partially confirmed. Geometry identical between backends; only stroke/glyph rasterization differs.
- **Hypothesis B** (L1 washes out cairo's wins): **confirmed.** Smoking gun: `clusters_stroke_dash_dashed` -- under Agg the outer dashed cluster has missing left/right strokes; under cairo it forms a complete dashed rectangle matching graphviz. **Massive visual improvement; only 0.07 L1 drop** because the missing strokes are 1px wide and contribute trivially to absolute pixel intensity.
- **Hypothesis C** (mplcairo not actually using cairo): rejected. The differences are systematic and match cairo's known rasterizer characteristics -- if cairo weren't being used, we'd see noise, not a clear pattern.

Auditor verdict: `STOP_CONVERGED_HYPOTHESIS_B`. Cairo IS visibly better on classical strengths (dashed strokes, curve AA, font hinting); the L1 metric is structurally blind to thin-feature wins.

Optional 30-min follow-up flagged: cairo stroke-weight calibration to close a small `+0.08 L1` regression on `nodes_shapes_rect` and `nodes_shapes_tab`.

### Round 3 (commit d5af420): stroke-weight calibration

User opted to take the 30-min polish.

- `_CAIRO_STROKE_WIDTH_SCALE = 0.86` constant in `_backend.py` + `stroke_width_scale_for(backend_name)` accessor
- Empirically `0.86`, NOT the audit's predicted `1.15` -- codex discovered the actual issue was opposite to the auditor's hypothesis. Under the data-coord ribbon path, cairo strokes are slightly THICKER than Agg's, not thinner. Multiplying by 0.86 brings them down to ink-density parity.
- Applied at node, cluster, marker-terminal, and text stroke ribbon construction sites; edge bodies left on the existing width path to preserve thin-edge visibility.

Post-calibration numbers:

| Card | Agg | Cairo R2 | Cairo R3 (post-calibration) | Status |
|---|---|---|---|---|
| nodes_shapes_rect | 2.435 | 2.518 (+0.08) | 2.479 (+0.04) | half-closed; within ±0.05 gate |
| nodes_shapes_tab | 2.659 | 2.738 (+0.08) | 2.692 (+0.03) | half-closed; within ±0.05 gate |
| clusters_stroke_dash_dashed | 0.929 | 0.855 | **0.836** | smoking-gun win improved |
| combo_pie_bold | 1.957 | 1.930 | **1.913** | round-9 win improved |
| combo_donut_shadow | 2.128 | 2.084 | **2.068** | round-9 win improved |
| evil_donut_diamond | 2.024 | 2.024 | 2.020 | round-9 win unchanged |
| clusters_opacity_1_0 | 1.519 | 1.536 | 1.529 | within noise |
| **Mean Tier A L1** | **1.515** | **1.513** | **1.495** | cairo wins by 0.020 net |

## Final state

| Metric | Pre-Sprint-B (Agg only) | Post-Sprint-B (cairo opt-in) |
|---|---|---|
| Backend choice available | no | yes; cairo + Agg |
| Default backend | Agg only | auto-detect (cairo if mplcairo installed, else Agg) |
| Per-render override API | n/a | `dagua.render(g, pos, backend="agg" | "cairo")` |
| Global override API | n/a | `dagua.set_default_backend(name)` |
| Optional install | n/a | `pip install 'dagua[cairo]'` (system: `apt install libcairo2-dev` on Linux, `brew install cairo` on Mac, none on Windows) |
| Mean Tier A L1 | 1.515 | 1.495 (under cairo on dev machine) |
| Backend tests | 0 | 8 (7 pass + 1 skip for cairo-missing path) |
| Comparison gallery infrastructure | n/a | yes; `scripts/build_backend_comparison_gallery.py` |
| Round-9 visual wins preserved | baseline | improved or preserved across the board |

## What we learned

1. **Sprint A had already closed most of the rendering-stack residual.** The data-coord-everything refactor made the geometric primitives identical between backends; cairo's contribution is purely rasterization polish on top. The hypothesized "cairo will drop L1 by 50%" was wrong; the empirical answer is "cairo provides marginal L1 improvement but real visual quality wins on classical-cairo features."

2. **L1 metric is structurally blind to thin-feature improvements.** The `clusters_stroke_dash_dashed` smoking gun -- complete vs broken dashed cluster outline -- is a dramatic visual quality improvement that the L1 metric registers as 0.07. Future visual quality measurement should pair L1 with a perceptual metric (SSIM, MS-SSIM) or a feature-presence test for thin-stroke completeness.

3. **Empirical calibration > auditor prediction when stakes are mechanical.** The auditor predicted a `1.15` stroke-weight scale based on cairo internals reasoning; codex found `0.86` empirically. Both can be right depending on the exact rasterization path, and this codebase's data-coord ribbon construction goes a different direction than the auditor's mental model. Trust the empirical sweep.

4. **Cairo's classical strengths matter more than the metric.** Even at "near-tied" L1, cairo delivers:
   - Complete dashed cluster outlines (Agg has broken outer strokes)
   - Smoother curve AA on pies / donuts / arrows
   - Better font hinting on small labels
   - Cleaner shadow gradients with no quantization banding
   These are real, observable quality wins that justify cairo as the recommended backend when mplcairo installs cleanly.

## Commits this sprint (3)

```
5b48e16  feat(render): cairo backend as opt-in matplotlib alternative
cddbba1  feat(scripts): add cairo comparison gallery metrics
d5af420  feat(render): cairo stroke-weight calibration to match Agg ink density
```

## What's next

Both post-dial-tuning sprints are now closed:
- **Sprint A (data-coord-everything):** done (round 16, commit 3b701a4)
- **Sprint B (cairo opt-in):** done (round 3, commit d5af420)

Possible future work tracked in `.project-context/todos.md`:
- Pixel-unit overrides as opt-in user-facing API (`NodeStyle.stroke_width_override` etc.)
- Perceptual metric infrastructure (SSIM, MS-SSIM) to better measure cairo's wins
- The other render-backend candidates (svg-cairo, pure-cairo without matplotlib intermediate) if rendering-stack performance becomes a concern

But the immediate post-dial-tuning workstream is structurally complete:
1. dagua's render path is data-coord-everything (differentiable layout invariant restored)
2. cairo backend opt-in for graphviz-grade rasterization quality
3. The dpi-invariance regression test makes calibrate-once enforceable
4. The cairo comparison gallery infrastructure makes future backend additions easy

dagua's rendering layer is now structurally honest about what it is -- a differentiable layout engine that produces high-quality graph diagrams via either Agg (zero-deps default) or cairo (graphviz-grade quality on opt-in install).
