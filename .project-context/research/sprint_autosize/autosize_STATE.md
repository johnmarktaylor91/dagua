---
run: autosize
created: 2026-05-01T01:18
completed: 2026-05-01T05:23
state: DONE
final_round: 3
final_commit: 16a7a91
final_mean_tier_a_l1: 1.217
note: Sprint C converged at round 3. NodeStyle.auto_size_to_label + GRAPHVIZ_STRICT_THEME + dagua.render(fit_to_canvas=...) + aspect-aware padding + tightened pair-shape gap (110 from 260). Mean Tier A L1: 1.495 -> 1.217 (-0.278). Shape parity cards dropped from L1 ~3 to L1 < 0.8. Round-9 wins preserved. autosize_SUMMARY.md written. Sprint D (perceptual metric) next.
---

# autosize -- Autonomous Loop State

## Goal

Make dagua match graphviz's auto-sizing semantics natively. NodeStyle should support an `auto_size_to_label: bool` field; when True, node W/H is computed from label text width + padding (with min_width/min_height as floors, not absolute values). Set this True in GRAPHVIZ_STRICT_THEME so the user gets graphviz-equivalent auto-sizing by default when using the strict theme.

This is FEATURE PARITY work, not just metric tuning. dagua becomes a graphviz drop-in replacement at the rendering level (in addition to the algorithm-fidelity work the parallel sprint is closing).

## Stop criteria

PRIMARY: Opus 4.7 visual auditor inspects the post-autosize gallery (under cairo backend) and confirms dagua's shape/text/cluster cards are visually-matched to graphviz at the auto-sized level (labels fit naturally, no obvious scale mismatch). Verdict: zero `fixable_scale_mismatch` findings.

SECONDARY: mean Tier A L1 under cairo drops from 1.495 toward <= 0.8. Top-20 worst residuals shift from "scale mismatch on shape cards" to "rasterizer-floor + competitor glitches."

ANTI-FLAIL: 3 consecutive rounds with same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=10.

## Architectural plan

The data-coord-everything sprint already wired the right mechanism: nodes have `min_width` and `min_height` fields in NodeStyle (data-coord units). What's missing is a switch that says "treat min_width/min_height as the FLOOR; compute actual width from label content."

1. Add `NodeStyle.auto_size_to_label: bool = False` (default False = current behavior; explicit override only).
2. In the node-size resolution path (likely `compute_node_sizes` or equivalent), when `auto_size_to_label` is True:
   - Measure rendered label width + padding (using existing label-measurement infrastructure)
   - Use `max(measured_width + 2*padding, min_width)` as actual width
   - Same for height
3. Set `auto_size_to_label=True` in `GRAPHVIZ_STRICT_THEME` so users opt-in via theme choice.
4. In `scripts/build_gallery_audit.py`: REMOVE the hardcoded `min_width=200, min_height=110` override on pair-fixture cards. Let the theme default take over. Cards using non-strict themes should keep their existing min sizes via explicit setting.

This makes dagua's native API (when configured to graphviz_strict) produce graphviz-equivalent auto-sized nodes without fixture overrides.

## Wake-up case routing

Same pattern as Sprints A/B: codex commit -> regen + audit; codex still running -> ack; codex died -> investigate; quota -> pause + reset.

## Hard guardrails

- DO NOT touch the algo_fidelity territory (`dagua/layout/ops/*`, `dagua/eval/*`, `scripts/ogdf_*`, `tests/test_classic_*`, `tests/test_variant_*`, `tests/test_layout/test_neato.py`)
- DO NOT touch `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`, `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`, `_DENSITY_LABEL_FONT_FLOOR`, `_MIN_VISIBLE_STROKE_POINTS`, `_CAIRO_STROKE_WIDTH_SCALE`, `density_aware_size_factor()` -- these are calibrated and locked
- DO NOT regress data-coord-everything (Sprint A invariant). Auto-sized widths still flow through data-coord ribbons.
- DO NOT regress cairo opt-in (Sprint B invariant). Auto-sizing must work under both backends.
- DO NOT regress round-9/11/12/13/14/15 visual wins (combo_pie_bold labels readable, box3d edge stem visible, dpi-invariance test passes, etc.)
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Mean L1 (cairo) | Notes |
|---|---|---|---|---|---|---|
| 1 (codex) | 01:20 | — | — | — | — | dispatched: auto-size feature + theme integration + fixture cleanup |

## Sprint chain (after this converges)

1. **Sprint C (this): autosize** -- match graphviz auto-sizing semantics
2. **Sprint D: perceptual metric** -- add SSIM / MS-SSIM to per_card_pixel_diff so cairo's wins are measurable
3. **Sprint E: stroke-pattern-aware cairo calibration** -- per-pattern-type cairo scale (heavy fills, thin dashes, etc.)
4. **Sprint F: pixel-unit override API** -- NodeStyle.*_override fields for users wanting precise paper-figure typography
5. **Sprint G: final visual audit** -- gauntlet under cairo+autosize+all calibrations; declare graphviz-drop-in achieved

I'll chain these autonomously. User said "go."
