---
run: cairo_svg
created: 2026-05-01T (post-resume)
state: PENDING
current_round: 1
note: Sprint J of the graphviz-drop-in push. Adds an OPTIONAL bit-equivalent rasterization path for users who want pixel-perfect parity with `dot -Tpng`. Implementation: render dagua to SVG via dagua's existing SVG backend, then rasterize the SVG via the same cairo invocation graphviz uses (cairosvg or rsvg-convert). NOT the default. Opt-in via `dagua.render(..., bit_equivalent=True)` or similar. Dispatches AFTER Sprint H + Sprint I land.
---

# cairo_svg -- Autonomous Loop State

## Goal

Provide an optional rendering path that produces pixel-equivalent output to `dot -Tpng` for users who want bit-perfect parity (academic figure reproduction, bit-equivalence claims, etc.). NOT the default behavior.

Architecture:
- New rendering option, e.g., `dagua.render(..., bit_equivalent=True)` or `output_format="cairo_svg_png"`
- Pipeline: dagua.layout -> dagua.render_to_svg (existing) -> cairosvg/rsvg-convert -> PNG
- Both dagua and graphviz can produce SVG; both rasterize through identical cairo. Result: pixel-equivalent PNGs (modulo SVG content differences, which are eliminated by Sprint H + algo_fidelity).

## Stop criteria

PRIMARY: A regression test renders the same DOT through `dot -Tpng` and through `dagua.render(..., bit_equivalent=True)` and pixel-compares with SSIM >= 0.99 (much tighter than Sprint H's 0.95).

SECONDARY: All existing render tests pass. The default (non-bit-equivalent) path is unchanged.

ANTI-FLAIL: 3 consecutive rounds same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=4.

## Hard guardrails

- DO NOT touch any locked constants
- DO NOT touch algo_fidelity territory
- DO NOT change default rendering behavior (must stay opt-in)
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Notes |
|---|---|---|---|---|---|
| 1 (codex) | (TBD) | — | — | — | dispatched: cairosvg integration as opt-in render path |
