---
run: data_coord
created: 2026-04-30T16:50
completed: 2026-04-30T19:25
state: DONE
final_round: 16
final_commit: 3b701a4
final_mean_tier_a_l1: 1.515
note: Sprint A converged at round 16. Calibrate-once invariant restored across `dagua/render/`. 4 implementation rounds (13/14/15/16) + 3 audit rounds (after-13/after-14/after-15). Round-16 audit verdict: STOP_CONVERGED with one non-blocking suggestion (3 defense-in-depth fixtures), addressed in round 16. Sprint B (cairo opt-in) dispatched 19:37 (PID 3287332).
---

# data_coord -- Autonomous Loop State

This file is the canonical "where are we" record for this run.
Every wake-up event (watcher fire, user ping, schedule trigger) MUST
read this file FIRST and act on the case routing below given the
current observable state. Do not act on intuition; act on the case.

## Goal

ALL render primitives in `dagua/render/mpl.py` flow through data coordinates
by default. Pixel/display-points are reserved for explicit user overrides
(`NodeStyle.*_override`) and for an EXPLICIT minimum-width clamp on the
data-coord ribbon path -- never as silent fallbacks like round 11's
`_edge_uses_display_stroke_body`.

The structural argument (per `feedback_data_coord_everything_strict`):
display-points are outside the optimizer's manifold. Anything in display-
points cannot appear in a loss term, cannot be optimized differentiably,
and breaks the calibrate-once-correct-everywhere invariant.

## Stop criteria (observable, quantitative)

PRIMARY: an Opus 4.7 audit subagent returns ZERO findings classified as
`fixable_data_coord_violation` across the full render path. The auditor
inspects: (a) the codebase for any `linewidth=` / `fontsize=` calls that
aren't routed through `_compute_display_scale(ax)` or `display_scale`,
(b) gallery comparison images (round-11/12 visual wins must be preserved),
(c) the dpi-invariance regression test passes.

ANTI-FLAIL: 3 consecutive rounds with the same un-closeable issue ->
mark as `principled_residual` (e.g., a matplotlib quirk that genuinely
cannot be closed without swapping rasterizer; defer to the cairo sprint).

HARD CAP: max_rounds=12. If we hit 12 rounds without convergence, run
shutdown with `state: BLOCKED_AT_CAP`.

QUOTA: codex AND Opus subagents both quota-blocked -> write `state: BLOCKED`,
send iMessage, ScheduleWakeup for reset time + 5 min.

## Visual-wins preservation gates

Round-11/12 visual fixes must survive round-13's data-coord refactor:

1. Edge stem visible on simple-shape pair-fixture cards. Pixel probe
   on `eval_output/gallery_audit/cards/comparisons/nodes/shapes/box3d_vs_graphviz.png`
   must show >= 50 dark pixels in the inter-node corridor (y=270-330,
   x=380-420 on the 1600x600 panel).
2. Combo card label legibility. Render `combo_pie_bold` and pixel-probe
   the "Ingest" node's label-width-to-node-width ratio; must remain
   <= 0.95 (i.e., text fits inside the node).
3. Mean Tier A L1 within +/- 0.15 of round-12 baseline (1.701).
4. Round-9 "wins" L1 within +/- 0.15 of round-12 values (combo_pie_bold
   2.034, combo_donut_shadow 2.195, evil_donut_diamond 2.118,
   clusters_opacity_1_0 1.797).

If ANY of these regress beyond tolerance, the round 13 fix is wrong;
back it out and try a different approach. The test gates are how we
know data-coord-everything didn't accidentally break the visual outcomes.

## DPI-invariance regression test (the calibrate-once enforcer)

New test in `tests/test_render_dpi_invariance.py`:

- Render the canonical pair fixture (2 nodes, default style, default theme)
  at dpi values 100, 150, 200, 300.
- For each rendering: extract relative geometry ratios (border-stroke-
  width-as-fraction-of-node-width, font-size-as-fraction-of-node-height,
  edge-stroke-width-as-fraction-of-node-spacing). Use pixel measurement
  on the resulting PNG.
- Assert: relative ratios are identical (within tolerance) across all
  dpi values. If they differ, somewhere in the render path is using
  display-points instead of data-coord, breaking calibrate-once.

This test is the PERMANENT regression guard. Once it passes, future
display-point leakage will trip it automatically.

## Wake-up case routing

When a wake-up event fires, run this triage BEFORE doing anything else:

```
1. git log --oneline -5     # did codex commit since last turn?
2. kill -0 <CODEX_PID>       # is codex still running?
3. cat eval_output/gallery_audit/per_card_pixel_diff_summary.md  # current state
4. read this file's "current_round" line
```

Then route per these cases:

### Case A: codex committed AND not running (HAPPY PATH)

1. Re-render gallery: `python scripts/build_gallery_audit.py --all`
2. Re-run diffs: `python scripts/per_card_pixel_diff.py`
3. Run regression test: `pytest tests/test_render_dpi_invariance.py tests/test_render_pair_edges.py tests/test_render_density_label.py -q` -- all must pass
4. Compare new metrics to round-12 baseline (mean Tier A L1 1.701)
5. Visual spot-check round-11/12 wins (Read box3d_vs_graphviz.png, combo_pie_bold_vs_graphviz.png)
6. If gates pass: dispatch Opus auditor for the round
7. If gates fail: write back-out spec, dispatch codex again

### Case B: codex committed AND still running

Acknowledge to user in 1-2 sentences if asked. Do NOT redispatch.
Wait for next event.

### Case C: codex died WITHOUT committing

Read /tmp/dial_round_<N>.log for failure reason. If quota-block, fall
back to Opus subagent path. If real error, surface to user.

### Case D: codex hit quota mid-round

Pause. Set state to BLOCKED. Send iMessage. ScheduleWakeup for reset.

### Case E: Opus auditor returns STOP_CONVERGED

Write SUMMARY.md, update STATE to DONE. Surface to user. Sprint B
(cairo opt-in) starts only on user signal.

### Case F: Opus auditor returns CONTINUE_ROUND_<N+1>

Write next round's spec from auditor's findings. Dispatch codex.

## Hard guardrails (DO NOT VIOLATE)

- DO NOT touch algo_fidelity dirty files (`dagua/layout/ops/*`,
  `dagua/eval/*`, `scripts/ogdf_*`, `tests/test_classic_*`,
  `tests/test_variant_*`, `tests/test_layout/test_neato.py`,
  `.project-context/research/sprint_algo_fidelity/*`,
  `eval_output/algo_fidelity/*`). The parallel sprint owns these.
- DO NOT touch `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`,
  `_MIN_HATCH_LINEWIDTH_POINTS`, `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`
  in `dagua/render/mpl.py`. Locked.
- DO NOT touch `density_aware_size_factor()` formula or the
  `_DENSITY_LABEL_FONT_FLOOR` value (round 12 closed those).
- DO NOT touch `GRAPHVIZ_STRICT_THEME` numeric values in `dagua/styles.py`.
- DO NOT bump versions. NO new branches. Single working branch is develop.
- Use explicit `git add <path>` per file. NEVER `-A` or `.`.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Mean L1 | Notes |
|---|---|---|---|---|---|---|
| 13 (codex) | 16:54 | 17:30 | a0f9678 | n/a | 1.756 | Reverted round-11 display-stroke fallback. Refactored to Figure(...)+FigureCanvasAgg. Added dpi-invariance regression test (1 fixture). Mean L1 rose slightly from 1.701 -> 1.756 due to data-coord ribbon AA on thin edges. |
| 14 (audit) | 17:32 | 17:42 | n/a | CONTINUE_R14 | n/a | Found 4 specific linewidth leakages (`_draw_node_border_path`, double_circle inner ring, cylinder rim, cluster solid border). dpi-invariance test "passes by accident" -- 2-node fixture doesn't exercise 3 of 4 leakages. AUDIT_round_14_OPUS.md. |
| 14 (codex) | 18:04 | 18:35 | bbd4c97 | n/a | 1.516 | All 4 leakages closed via data-coord ribbon pattern. 3 new dpi-invariance fixtures (cluster, double_circle, cylinder); TDD-verified (failed pre-fix, pass post-fix). Mean L1 1.756 -> 1.516 (dropped favorably; data-coord ribbons match graphviz cairo paths better than display-point strokes). |
| 15 (audit) | 18:38 | 18:50 | n/a | PARTIAL_DEFER | n/a | mpl.py clean (0 new leakages across 46 linewidth + 14 fontsize matches). Found 4 fixable items OUTSIDE mpl.py: text outline (collection.py:581), bold emphasis (:602), port indicator markers (intentional display-points; documentation-question), stale SCALING.md. AUDIT_round_15_OPUS.md. User: "iterate till ceiling = no fixable findings", so push through. |
| 15 (codex) | 18:43 | 19:07 | 042a73d | n/a | 1.515 | text outline + bold emphasis as data-coord ribbons. Port markers Path A (data-coord, not Path B documentation). SCALING.md rewritten (~85 lines). Mean L1 1.516 -> 1.515. |
| 16 (audit) | 19:09 | 19:13 | n/a | STOP_CONVERGED | n/a | Final sweep: zero leakages remain across `dagua/render/`. All 4 round-15 fixes verified. Visual gates preserved. Non-blocking suggestion: 3 defense-in-depth fixtures (text outline / port indicator / bold emphasis). AUDIT_round_16_OPUS.md. |
| 16 (codex) | 19:16 | 19:25 | 3b701a4 | n/a | 1.515 | 3 defense-in-depth fixtures added. All 7 dpi-invariance tests pass. Audit-by-grep gap closed. |
| shutdown | 19:25 | — | — | DONE | 1.515 | Sprint A converged at honest ceiling. SUMMARY.md written. Sprint B (cairo opt-in) dispatched 19:37 (PID 3287332). |
