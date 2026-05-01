---
run: canvas
created: 2026-05-01T (post-resume)
state: ACTIVE
current_round: 1
note: Sprint H of the graphviz-drop-in push. Make graphviz's canvas rules (margin=0.11in, DPI=96, content-sized output, size/ratio/pad attribute support) the DEFAULT in dagua.render(). Pre-release means no users to migrate. The current "fit_to_canvas" stays as an explicit override for the fixed-panel use case (dashboards, jupyter cells), no longer the default. After this sprint + algo_fidelity convergence, dagua should produce visually-identical output to `dot -Tpng` for any DOT input.
---

# canvas -- Autonomous Loop State

## Goal

Replace dagua's current default canvas math with graphviz's. Specifically:

- Default margin = 0.11 inches (8 points) per side
- Default DPI = 96 for raster (overridable via graph-level `dpi` attribute)
- Output dimensions = content-sized: `(graph_extent + 2*margin) * DPI`
- Support graph-level attributes matching graphviz: `margin`, `pad`, `size`, `ratio`, `dpi`
- `size="W,H"` = max bounds; `size="W,H!"` = force-fit; `ratio` = `fill | compress | expand | auto`

`fit_to_canvas` parameter on `dagua.render()` becomes an explicit OPT-IN for the "fill this fixed panel" use case (jupyter cells, dashboards). Not the default behavior.

GRAPHVIZ_STRICT_THEME doesn't need a special canvas knob — the default IS graphviz-equivalent now.

The gallery_audit harness keeps `fit_to_canvas=True` explicitly for parity panel comparison (it's a different use case: forced fixed dimensions for side-by-side layout).

## Stop criteria

PRIMARY: An Opus 4.7 visual auditor renders a known DOT through `dot -Tpng -Gdpi=96` AND through `dagua.from_dot(...) -> dagua.layout(algorithm="sugiyama") -> dagua.render(theme=GRAPHVIZ_STRICT_THEME)`, then pixel-compares the two PNGs. Verdict requires:
- SSIM >= 0.95 between the two PNGs (allows for rasterizer-stack noise)
- Visual inspection: no obvious size/position/margin difference

SECONDARY: All 162 existing render tests pass. Mean Tier A L1 in the cosmetic gallery may shift (gallery still uses fit_to_canvas explicitly so combo cards' relative numbers stay similar).

ANTI-FLAIL: 3 consecutive rounds same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=8.

## Hard guardrails

- DO NOT touch any locked constants from prior sprints
- DO NOT touch algo_fidelity territory
- DO NOT regress data-coord-everything (Sprint A invariant). Canvas math is uniform scale at the rendering boundary; optimizer manifold unchanged.
- DO NOT regress dpi-invariance test
- DO NOT regress round-9/11/12/13/14/15 visual wins (combo_pie_bold labels readable, box3d edge stem visible, etc.)
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Notes |
|---|---|---|---|---|---|
| 1 (codex) | (TBD) | — | — | — | dispatched: graphviz canvas rules as default |
