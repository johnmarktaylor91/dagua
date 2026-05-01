---
run: perceptual
created: 2026-05-01T05:25
state: ACTIVE
current_round: 1
note: Sprint D of the graphviz-drop-in push. Adds perceptual metrics (SSIM, MS-SSIM, possibly LPIPS) to per_card_pixel_diff. Cairo round-2 audit established that L1 is structurally blind to thin-feature wins (clusters_stroke_dash_dashed: dramatic visual fix registered as 0.07 L1). Perceptual metrics should surface those wins AND find new residuals on combo cards that L1 underweights.
---

# perceptual -- Autonomous Loop State

## Goal

Add SSIM (Structural Similarity Index) and MS-SSIM (Multi-Scale SSIM) metrics to `scripts/per_card_pixel_diff.py`. Re-evaluate cards under perceptual metrics. The output should:

1. Confirm L1's structural blindness on thin-feature cards (cluster_dash, font hinting)
2. Identify cards where perceptual quality is meaningfully worse than L1 says
3. Provide a more honest baseline for future graphviz-drop-in optimization

This is INFRASTRUCTURE work, not optimization work. Once perceptual metrics are in the pipeline, future sprints can iterate using them.

## Stop criteria

PRIMARY: Opus 4.7 visual auditor inspects the SSIM-augmented per-card report and confirms:
- SSIM picks up cairo round-2's smoking gun (cluster_dash_dashed) as a significant cairo win
- New top-residual list under perceptual metric reveals at least one card class that L1 missed
- Or: perceptual metric corroborates L1 on top residuals (in which case combo card residuals are real, not artifact)

SECONDARY: scripts/per_card_pixel_diff.py outputs SSIM and MS-SSIM alongside L1 in the summary table. The metric infrastructure is reusable for downstream sprints.

ANTI-FLAIL: 3 consecutive rounds with same un-closeable issue -> `principled_residual`.

HARD CAP: max_rounds=5 (this is infrastructure work; should be quick).

## Architectural plan

1. Add a perceptual metric computation function alongside the L1 in `per_card_pixel_diff.py`. Use `scikit-image.metrics.structural_similarity` for SSIM. For MS-SSIM, use `pytorch-msssim` or implement multi-scale SSIM via Gaussian pyramid.
2. Extend the per-card output JSON and Markdown summary table to include SSIM and MS-SSIM columns.
3. Generate a parallel-axis report: for each Tier A card, show L1 + SSIM + MS-SSIM. Highlight cards where L1 says "good" but SSIM says "bad" (the metric-blind class).
4. Run on both Agg and cairo galleries to surface the cairo-vs-Agg perceptual delta.

## Sprint chain (resumed)

1. ~~Sprint A (data-coord)~~ DONE
2. ~~Sprint B (cairo opt-in)~~ DONE
3. ~~Sprint C (autosize + canvas-fit)~~ DONE
4. **Sprint D (perceptual metric)** -- this sprint
5. Sprint E: stroke-pattern-aware cairo calibration (with SSIM as the better signal)
6. Sprint F: pixel-unit override API (NodeStyle.*_override)
7. Sprint G: final visual audit gauntlet -- declare graphviz-drop-in achieved or iterate

## Hard guardrails

- DO NOT touch any locked constants from prior sprints
- DO NOT touch `dagua/render/*.py` -- this is METRIC infrastructure work
- DO NOT touch algo_fidelity territory
- Add new dependency for SSIM: prefer `scikit-image` (already in many scientific Python deployments) over a heavier choice
- DO NOT regress L1 metric output -- keep current numbers, just add new columns
- Single working branch: develop. NO new branches.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | New top SSIM residuals | Notes |
|---|---|---|---|---|---|---|
| 1 (codex) | 05:30 | — | — | — | — | dispatched: add SSIM + MS-SSIM to per_card_pixel_diff |
