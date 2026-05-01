---
run: override
created: 2026-05-01T07:15
state: ACTIVE
current_round: 1
note: Sprint F of the graphviz-drop-in push. Add pixel-unit override API per the standing 2026-03-23 directive (data-coord by default; pixel-points OPT-IN OVERRIDE only). Users who want literal point-perfect typography for paper figures can specify NodeStyle.stroke_width_override / EdgeStyle.width_override / NodeStyle.font_size_override fields that bypass data-coord and route directly to display-points.
---

# override -- Autonomous Loop State

## Goal

Implement the opt-in override API spec'd in `feedback_data_coord_everything_strict.md`:

> Override option (future opt-in API): expose pixel/point sizing as an explicit override. Something like `NodeStyle.stroke_width_override_points: float | None = None`. When set, that value bypasses data-coord and goes straight to display-points. Document loud and clear that override values are NOT differentiable.

This is the user-facing escape hatch from data-coord-everything for users who want precise paper-figure typography.

Fields to add (all `Optional[float] = None`):
- `NodeStyle.stroke_width_override_points` -- override the data-coord stroke-width ribbon for borders
- `NodeStyle.font_size_override_points` -- override the data-coord font_size for labels
- `EdgeStyle.width_override_points` -- override the data-coord ribbon width for edge bodies
- `EdgeStyle.font_size_override_points` -- override the data-coord font_size for edge labels
- `ClusterStyle.stroke_width_override_points` -- override the data-coord cluster border width
- `ClusterStyle.font_size_override_points` -- override the data-coord cluster label font_size

When None: data-coord behavior (current default; differentiable).
When set: bypass data-coord conversion; pass the value directly as matplotlib `linewidth=` or `fontsize=` (display-points). Documented as NOT differentiable.

## Stop criteria

PRIMARY: Opus 4.7 audit confirms:
- All 6 override fields are wired through the render path
- Setting an override produces literal point-perfect rendering (independent of dpi-invariance / canvas-fit; user's request is taken verbatim)
- Default behavior (override=None) is unchanged from Sprint E end state
- Documentation clearly distinguishes "differentiable data-coord default" from "non-differentiable display-point override"

SECONDARY: a regression test verifies that:
- Default rendering is dpi-invariant (relative ratios preserved)
- Override rendering is dpi-aware (a 14pt override font really is 14 typographic points at any dpi)

ANTI-FLAIL: 3 consecutive rounds with same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=5 (this is API-shape work; should be quick).

## Hard guardrails

- DO NOT touch any locked constants from prior sprints (`_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`, `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`, `_DENSITY_LABEL_FONT_FLOOR`, `_MIN_VISIBLE_STROKE_POINTS`, `_CAIRO_STROKE_WIDTH_SCALE`, `density_aware_size_factor`)
- DO NOT touch GRAPHVIZ_STRICT_THEME numerics
- DO NOT touch algo_fidelity territory
- DO NOT regress data-coord-everything default (override=None must produce identical rendering to pre-Sprint-F)
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Notes |
|---|---|---|---|---|---|
| 1 (codex) | 07:18 | — | — | — | dispatched: add 6 override fields + render-path wiring + tests |
