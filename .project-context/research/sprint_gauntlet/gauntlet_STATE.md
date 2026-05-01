---
run: gauntlet
created: 2026-05-01T08:52
completed: 2026-05-01T09:50
state: DONE
final_round: 1
final_verdict: ACHIEVED_WITH_DOCUMENTED_RESIDUALS
note: Sprint G converged at round 1. Comprehensive Opus 4.7 visual audit of ~22 cards under cairo+autosize+all-calibrations stack. ZERO fixable findings in the rendering layer. dagua is a graphviz-drop-in replacement at rendering. Three CAIRO WINS over graphviz: graphviz fails on dashed edges, dotted edges, dashed clusters; dagua more correct. Combo card layout-scale residuals are algo_fidelity territory, out of scope. gauntlet_SUMMARY.md captures the full 7-sprint chain.
---

# gauntlet -- Autonomous Loop State

## Goal

Final visual gauntlet under the full sprint-A-through-F stack:
- Sprint A: data-coord-everything
- Sprint B: cairo opt-in
- Sprint C: autosize + canvas-fit + aspect-padding
- Sprint D: SSIM perceptual metric
- Sprint E: dash/dot edge visibility
- Sprint F: pixel-unit override API

Inspect the gallery under cairo backend (the recommended graphviz-drop-in config) and declare:
- **graphviz drop-in achieved**: zero `fixable_for_graphviz_drop_in` findings; remaining residuals are out-of-scope (algo_fidelity / cosmetic curiosities)
- **final residuals identified**: one or two more fix rounds before declaring achieved

The auditor is the final arbiter. User said "iterate till ceiling" — the auditor's verdict is the ceiling.

## Stop criteria

PRIMARY: Opus 4.7 visual auditor inspects ~20 representative cards from across Tier A under the cairo+autosize+all-calibrations config and declares either:
- `GRAPHVIZ_DROP_IN_ACHIEVED` — sprint chain done; document and ship
- `CONTINUE_ROUND_N` — specific final residuals to close

SECONDARY: Mean Tier A L1 (cairo) is at the noise floor (likely between 1.0-1.3 given current 1.232); SSIM_loss is < 0.04 mean; round-9 wins all preserved.

ANTI-FLAIL: 3 consecutive rounds with same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=8.

## Hard guardrails

- DO NOT touch any locked constants from prior sprints
- DO NOT touch algo_fidelity territory
- DO NOT regress dpi-invariance, data-coord-everything, round-9 wins, Sprint E edge visibility, Sprint F default-path
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Notes |
|---|---|---|---|---|---|
| 1 (audit-first) | 08:53 | — | n/a | — | dispatched: comprehensive cairo+autosize visual gauntlet under all-sprint config |
