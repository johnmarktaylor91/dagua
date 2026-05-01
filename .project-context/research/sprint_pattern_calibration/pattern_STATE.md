---
run: pattern_calibration
created: 2026-05-01T06:08
completed: 2026-05-01T07:14
state: DONE
final_round: 2
final_commit: b2bac8d
note: Sprint E converged at round 2. Round 1 audit found the SSIM-flagged "dashed/dotted/italic L1-blind class" was mostly layout-scale mismatch (algo_fidelity scope, not cosmetic), EXCEPT for a real render-path defect: dashed/dotted edge body + arrowhead vanishing at thin widths. Round 2 closed that: enforced _MIN_VISIBLE_STROKE_POINTS on dashed/dotted ribbon construction + decoupled arrowhead placement from dash phase. Visibility defect closed (visual + 2 new pixel-probe regression tests). Italic "defect" was a graphviz limitation, not dagua bug -- left alone. Combo-card layout-scale residuals are out of scope for cosmetic sprint.
---

# pattern_calibration -- Autonomous Loop State

## Goal

Identify and close the L1-blind perceptual defects in dashed/dotted/italic rendering. SSIM_loss for these cards is 0.06+ while SSIM_loss for shape parity (already optimized) is ~0.016. The 4x perceptual gap suggests dagua's dash/dot/italic rendering has a real defect that L1 was missing.

Specific cards (from SSIM_loss rank 1-20 in Sprint D's divergence report):
- combo_parallelogram_dotted (SSIM_loss 0.065)
- combo_star_dotted (SSIM_loss 0.065)
- combo_dashed_diamond_opacity (SSIM_loss 0.061)
- combo_diamond_dashed_opacity_italic (SSIM_loss 0.060)
- combo_cluster_dashed_shadow (probably similar)
- nodes_text_font_style_italic (SSIM_loss 0.023)
- edges_styles_style_dotted (SSIM_loss 0.024)
- edges_styles_style_dashed (SSIM_loss 0.023)

## Stop criteria

PRIMARY: Opus 4.7 visual auditor inspects the affected cards and confirms:
- The defect is identified visually (font misrendering, dash-pattern misalignment, dotted-stroke spacing issue, etc.)
- Either the fix lands and SSIM_loss drops, OR the residual is principled (e.g., matplotlib-Agg-rendering-stack residual that cairo can't help with)

SECONDARY: Mean Agg SSIM_loss for the L1-blind class drops from ~0.06 toward ~0.03. Mean Tier A SSIM stays >= 0.96.

ANTI-FLAIL: 3 consecutive rounds with same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=8.

## Hard guardrails

- DO NOT touch any locked constants from prior sprints
- DO NOT touch algo_fidelity territory
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | SSIM_loss change | Notes |
|---|---|---|---|---|---|---|
| 1 (audit-first) | 06:10 | — | n/a | — | — | dispatched: investigate the dashed/dotted/italic defect visually; recommend fix path |
