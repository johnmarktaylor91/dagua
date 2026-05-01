---
run: fixture_refactor
created: 2026-05-01T (post-final-gauntlet)
completed: 2026-05-01T17:16
state: DONE
final_round: 1
final_commit: 3d3dcb5
note: Sprint K converged at round 1. (1) Theme-activation boundary closed: "Default | Variant" panels now derive per-node graphviz DOT attrs from prepared styles, applying theme defaults to both panels. Activation-boundary card L1 dropped to sub-0.6. (2) border_position_inside/outside reclassified Tier C (Graphviz++ extension; dagua-specific feature). Tier counts cairo: A=174, B=33 (was 35), C=70 (was 68). Tests: 51 pass + 1 skip + 1 xfail (bit-equivalence, gated on algo_fidelity).
---

# fixture_refactor -- Autonomous Loop State

## Goal

Two surgical fixture changes to close the last residual classes from the final gauntlet:

### Action 1: Apply theme to control panels

In `scripts/build_gallery_audit.py`, every "Default | Variant" panel structure currently renders the "Default" control with bare `NodeStyle()` defaults instead of inheriting `GRAPHVIZ_STRICT_THEME` (or whichever theme the comparison is running). Fix: explicitly thread the theme into control panel construction so both panels inherit the same defaults.

Affected card types: `border_opacity_*`, `stroke_width_*`, `font_style_italic`, `font_weight_bold`, `text_valign_*`, `external_label_*`, fill `solid` / `opacity_1_0`, plus the broader "diff one parameter" pattern across nodes/borders, nodes/text, nodes/fills, edges/styles.

### Action 2: Tier C reclassification for graphviz-unmappable border_position cards

Per `feedback_themes_set_defaults_users_override.md`: dagua's `NodeStyle.border_position` field with values "inside" / "outside" is a Graphviz++ feature (graphviz doesn't have it). The two cards `nodes_borders_border_position_inside` and `nodes_borders_border_position_outside` should be reclassified Tier C with reason "dagua-specific feature; graphviz lacks inside/outside border modes (Graphviz++ extension)".

Currently they're Tier B against cytoscape (which has CSS-spec border-position but renders with quirks documented in Sprint I). Reclassifying them Tier C drops them from the Tier A+B mean and properly contextualizes them as dagua-specific features.

## Stop criteria

PRIMARY: Mean Tier A L1 (cairo) drops by at least 0.05 from the fixture refactor (every minor_residual card on "Default | Variant" cards should drop a small amount that aggregates).

SECONDARY: All existing tests pass. Visual spot-check confirms control panels match the variant panel's theme.

ANTI-FLAIL: 3 consecutive rounds same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=3.

## Hard guardrails

- DO NOT touch any locked constants
- DO NOT touch algo_fidelity territory
- DO NOT regress any prior sprint's wins
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Notes |
|---|---|---|---|---|---|
| 1 (codex) | (TBD) | — | — | — | dispatched: fixture refactor + Tier C reclass |
