---
run: border_position
created: 2026-05-01T (post-resume)
completed: 2026-05-01T
state: DEFERRED
final_round: 1
final_verdict: PRINCIPLED_RESIDUAL_DEFER
note: Sprint I deferred at audit-first round 1. Auditor confirmed dagua's `border_position` math is ALREADY correct per CSS spec. The L1=10 residual is dominated by cytoscape-side rendering choices (forced corner-radius on rect, bbox-expansion on outside-stroke, WebGL AA, stroke paint-order on high-stroke-ratio cards). Closing the gap requires comparator-side cytoscape stylesheet calibration (out of scope; partial fix only) OR Sprint J's bit-equivalent path (which targets graphviz, not cytoscape -- doesn't apply). User decision pending.
---

# border_position -- Autonomous Loop State

## Goal

Close the cytoscape parity gap on `nodes_borders_border_position_inside_vs_cytoscape.png` (L1=10.008) and `nodes_borders_border_position_outside_vs_cytoscape.png` (L1=10.415). Currently dagua and cytoscape both support inside/outside variants of NodeStyle.border_position, but the math differs.

Investigation steps:
1. Inspect the comparison images visually to identify the geometric difference
2. Read cytoscape's docs / source on border-position math: https://js.cytoscape.org/#style/node-body
3. Read dagua's NodeStyle.border_position implementation
4. Identify the calibration delta (offset, alpha, anti-aliasing handling, etc.)
5. Patch dagua's render path to match cytoscape's math

## Stop criteria

PRIMARY: L1 on both border_position cards drops below 4.0 (matching the Tier B mean for cytoscape comparisons). Visual parity verified by Opus 4.7 audit.

SECONDARY: All existing render tests pass. No regression on cards that don't use border_position inside/outside.

ANTI-FLAIL: 3 consecutive rounds same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=4.

## Hard guardrails

- DO NOT touch any locked constants
- DO NOT touch algo_fidelity territory
- DO NOT regress any other Tier A or Tier B card
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Notes |
|---|---|---|---|---|---|
| 1 (audit-first) | (TBD) | — | — | — | will dispatch audit to investigate cytoscape math first, then codex round 2 to fix |
