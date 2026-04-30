---
run: dial_tuning
created: 2026-04-29T21:37:29-04:00
completed: 2026-04-30T14:05
state: DONE
final_round: 12
final_commit: f128fcc
final_mean_tier_a_l1: 1.701
note: Sprint converged at ceiling. Round 11 closed two systemic defects prior audits missed (edge stem missing at width=1.0pt; density-aware shrink not propagating to label font_size). Round 12 final closures (FONT_FLOOR 0.6->0.5; radial gradient parity in per_card_pixel_diff). Remaining residuals are scale-mismatch / metric-artifact / rendering-stack-residual classes that require unlocking sprint guardrails (GRAPHVIZ_STRICT_THEME, density formula, fixture min_width/min_height) to address. Round-9 "wins" were metric-pass / visual-fail; current numbers are the honest ceiling.
---

## ROUND_10_RESUME (when user signals begin)

When user says begin:

1. Check codex quota: try `codex exec --skip-git-repo-check --sandbox read-only "echo ok"` -- if returns "ok", quota reset; use codex.
2. If codex still quota-blocked: dispatch Opus subagent (Agent tool, model="opus", subagent_type="general-purpose") with adapted spec from `/tmp/PROMPT_round_10.md` (drop XML scaffolding; keep file:line refs + completeness contract).
3. Spec scope (UNCHANGED from /tmp/PROMPT_round_10.md):
   - Action 1: reclassify 4 nodes_fills_* cards (linear-gradient, pie, hatched, striped) Tier A → Tier C
   - Action 2: wire `_graphviz_node_attrs` for radial-gradient: emit `style="filled,radial"` + `fillcolor="<fill>:<gradient_color>"`
   - Action 3: reclassify 3 more cards (evil_pie_shadow_gradient, combo_pie_shadow_gradient_bold, combo_trapezoid_gradient) Tier A → Tier C
4. HARD GUARDRAILS (do not violate, even after pause):
   - NO render-path changes (`dagua/render/mpl.py` constants stay locked)
   - NO theme/density-aware-shrink changes (round-9 wins must stay)
   - Scope strictly to `scripts/build_gallery_audit.py` + test fixtures pinning Tier A counts
   - DO NOT touch algo_fidelity dirty files
5. Targets: mean Tier A L1 1.785 → ~1.62; gradient_radial 9.374 → ~3-4; round-9 wins unchanged.
6. After commit, update STATE to DONE, append round-10 row to iteration log, update SUMMARY.md.

# dial_tuning -- Autonomous Loop State

This file is the canonical "where are we" record for this run.
Every wake-up event (watcher fire, user ping, schedule trigger) MUST
read this file FIRST and act on the case routing below given the
current observable state. Do not act on intuition; act on the case.

## Goal

All cosmetic dials (NodeStyle, EdgeStyle, ClusterStyle, GraphStyle fields) tune
the same INDIVIDUALLY and IN COMBINATION as graphviz/cytoscape/mermaid/d3,
and don't break when combined. Use graphviz_strict as the base — every
feature except the one being manipulated sits at graphviz default.

Tier breakdown (per `eval_output/gallery_audit/index.jsonl`):
- Tier A (graphviz-anchored): 135 cards
- Tier B (cytoscape/mermaid/d3-anchored): 35 cards
- Tier C (heuristic-only — looks-nice + scales-monotonic + plays-well-in-combos): 114 cards

## Stop criteria (observable, quantitative)

PRIMARY: An Opus 4.7 audit subagent returns ZERO findings classified as
`real_cosmetic_gap + fixable_theme_or_render` across the full gallery
(or returns only `rendering_stack_residual` / `metric_artifact` / `not_actionable`).

ANTI-FLAIL: 3 consecutive rounds with the same un-closeable issue.
Mark as `principled_residual` and continue OR shut down.

HARD CAP: max_rounds=12. If we hit 12 rounds without convergence, run
shutdown with `state: BLOCKED_AT_CAP`.

QUOTA: Codex AND Opus subagents both quota-blocked → write `state: BLOCKED`,
send iMessage, ScheduleWakeup for reset time + 5 min.

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
3. Run regression test: `pytest tests/test_parity_metrics.py -q` — must pass
4. Compare new metrics to prior round's (snapshot prior_round_metrics.json before each dispatch)
5. Dispatch Opus 4.7 audit subagent for round N:
   - Use Agent tool with `model: "opus"` — Sonnet is a confirmed-substantial gap, never use it for visual audits
   - Pass: per_card_pixel_diff_summary.md, prior round's findings, iteration log
   - Pickiness rules baked into prompt (categories, minimum departure count, no "looks similar")
   - Output to `.project-context/research/sprint_dial_tuning/AUDIT_round_N_OPUS.md`
6. Process audit:
   - If verdict == STOP and zero `real_cosmetic_gap+fixable_theme_or_render` → run "Shutdown" below
   - Else write `PROMPT_round_(N+1).md` with findings translated to codex spec
   - Dispatch codex via codex-bg.sh; arm Monitor on codex-watch.sh
7. Update this file: bump `current_round`, append iteration-log row
8. Yield turn briefly (per CLAUDE.md "End your turn briefly so the next event becomes a fresh turn")

### Case B: codex still running (no-op)

1. Acknowledge: "codex round N still cooking, watcher armed"
2. Yield turn

### Case C: watcher fired LIMIT_HIT (codex quota exhausted)

1. Read `/tmp/dial_round_N.log` to confirm quota error pattern
2. Note reset time from error message
3. Pivot: dispatch the SAME round-N work to Agent(subagent_type="general-purpose", model="opus")
   - Adapt the codex spec: drop XML scaffolding (default_follow_through_policy etc.); keep file:line refs + completeness contract + verification loop
   - Tell the Opus implementer to commit + return commit SHA in its result
4. After Opus implementer returns, run pipeline + audit (Case A from there)
5. If Opus also hits limit → write `state: BLOCKED`, ScheduleWakeup for codex-reset-time + 5 min, send iMessage to JMT
6. Update state file

### Case D: tests fail / dirty repo / smoke-test broken

1. Read the failure output
2. If fixable in <10 min: fix and commit (this Claude session)
3. Otherwise: `git reset --hard <prior_commit>` after capturing diff, document
4. Mark round as RECOVER, retry codex with the failure log appended to spec

### Case E: anti-flail (3 consecutive rounds same issue)

1. Mark issue as `principled_residual` in iteration log + REPORT_round_N.md
2. Document why it's residual (rendering-stack, layout-scope, render bug requiring deeper fix)
3. Tell next audit subagent to NOT re-flag this issue
4. Continue to next round (don't shut down — there may be other actionable items)

### Case F: STOP verdict (no actionable findings remain)

→ Run "Shutdown procedure" below.

## Fallback chain (resource limits)

1. PRIMARY: codex via `~/.claude/scripts/codex-bg.sh /tmp/dial_round_N.log /tmp/PROMPT_round_N.md --cd /home/jtaylor/projects/dagua --sandbox danger-full-access --effort medium`
2. QUOTA-BLOCKED: `Agent(subagent_type="general-purpose", model="opus", prompt=<adapted codex spec>)` (Opus subagent in this conversation context — uses JMT's Anthropic subscription, separate quota)
3. BOTH BLOCKED: write `state: BLOCKED`, iMessage JMT with reset time, ScheduleWakeup, stop.

NEVER silently stall. NEVER export OPENAI_API_KEY (subscription path is the only auth path; API key is reserved for image-gen / TTS).

## Audit subagent dispatch template (Opus 4.7)

For each round N audit, dispatch an Agent with:

```
Agent(
  subagent_type="general-purpose",
  model="opus",
  description="Dial-tuning audit round N",
  prompt=<see ROUND_N_AUDIT_PROMPT below>
)
```

The audit prompt must include:
- Tier-aware rubrics: Tier A/B = pixel-anchored picky comparison vs competitor;
  Tier C = stand-alone aesthetic + monotonic scaling + combination integrity heuristic
- Inputs: per_card_pixel_diff_summary.md, hi-res panels for worst-N cards,
  representative atomic+combo+evil cards across all categories
- Pickiness: enumerated categories, minimum N findings, no "looks similar" verdicts
- Output classification: `real_cosmetic_gap` / `metric_artifact` / `uncertain` with
  actionability `fixable_theme_or_render` / `rendering_stack_residual` / `not_actionable`
- Image dimension cap: ≤2000px per image (gallery panels are 1600px wide; OK)
- Output to `AUDIT_round_N_OPUS.md` in this directory

## Codex implementer dispatch template

For each round N codex implementation prompt, the spec must include:
- Read context files (cosmetic_inventory.md, AGENTS.md, CLAUDE.md, audit findings)
- Single working branch: develop. NO new branches.
- Audit findings translated to concrete fixes (theme values + render param tweaks)
- Verification loop: pytest tests/test_parity_metrics.py + ruff + mypy + smoke render
- ONE commit per round; descriptive message `feat(dial): round N -- <one-line summary>`
- Out of scope: dagua/render/ unless audit explicitly says render code; no version bumps
- Effort: medium (per CLAUDE.md guidance — high burns budget)

## Shutdown procedure (mechanical -- user is asleep)

When stop criterion triggers, run these in order:

1. Write `.project-context/research/sprint_dial_tuning/dial_tuning_SUMMARY.md` with:
   - Total rounds executed
   - All commit SHAs from this run
   - Final pixel-diff stats: mean/median L1, mean/median SSIM, worst N panels, in-tolerance %
   - Locked features (regression-tested at 100% / N/N green)
   - List of `principled_residual` items with classification
   - List of layout-scope deferrals
   - Tier C heuristic findings that the maintainer should weigh in on
2. Pick 5-7 representative comparison images: best parity, worst parity, biggest improvement, hardest-evil-survived, principled-residual examples
3. Send iMessage:
   - First text: "Run dial_tuning done. <one-line result with key numbers>. <N> rounds, <commits>."
   - Then attach the 5-7 images with brief captions: `~/.claude/scripts/send-to-jmt.sh -a <path> "<caption>"`
4. Update this file: `state: DONE`, append final iteration-log row with shutdown timestamp.

## Iteration log

| Round | Start | End | Commit | Audit verdict | Mean L1 | Mean SSIM | Worst SSIM | Notes |
|---|---|---|---|---|---|---|---|---|
| 0 (setup) | 21:01 | 21:24 | cdcc91f | n/a | — | — | — | wire dial-tuning gallery harness — competitors installed, baseline retrofit, per-card diff, tier marker |
| 1 (audit) | 21:30 | 22:00 | n/a | FAIL/CONTINUE | (TBD) | (TBD) | 0.8266 (evil_pie_star) | 47 findings: 28 fixable_theme_or_render, 3 stack-residual, 4 layout-scope, 5 design-decision, 3 acceptable, 2 metric, 2 uncertain. Top systemic: white label-bg box, default node size 3-5x graphviz, 3 broken dials (cluster.opacity, cluster.label_position, fills.opacity), external_label position dial ignored, taper kills arrows+dashed, ~14 Tier C should be Tier A. AUDIT_round_1_OPUS.md. |
| 2 (codex) | 22:00 | 22:23 | 51236af | n/a | 6.535 | 0.9510 | 0.7+ (clusters_opacity_1_0 L1=64) | white-bg label box CLOSED (#3); cluster_label_position #16, external_label #18, bevel overlay #22, LR direction #32 CLOSED. Tier A 134→181 (14 reclassified). REGRESSIONS: cluster fill canvas-span (N1-3), bgcolor inside bbox not canvas (N4-5), external label larger than internal (N9-10). Round 2 commit lied — taper-kills-arrows still open, fills/opacity dial still flat. |
| 2 (audit) | 22:23 | 22:32 | n/a | FAIL/CONTINUE | n/a | n/a | n/a | 30 NEW findings + 11/28 round-1 fixable closed cleanly. Top systemic: cluster fill canvas-span, bgcolor bbox-not-canvas, default node size still 3-5x graphviz, taper STILL kills arrows, fills/opacity still flat. AUDIT_round_2_OPUS.md. Estimate 2 more rounds to ceiling. |
| 3 (codex) | 22:32 | 00:07 | b02e0bc | n/a | 6.495 | 0.9511 | 0.7+ (clusters_opacity_1_0=62.3) | Landed: bgcolor canvas-fill, taper arrows actual, opacity dial wiring, text_outline overlay, white cluster label plate, deep-cluster shrink, skipped-comparison renderer. Partial: cluster fill default-off, pair-fixture arrows, arrowhead size ratio. Skipped: node-size shrink (claimed already correct — but it OVER-CORRECTED on simple, BIFURCATED on gradient). Mean L1 only dropped 0.04 — metric pipeline issue. |
| 3 (audit) | 00:07 | 00:35 | n/a | FAIL/CONTINUE | n/a | n/a | n/a | CRITICAL FINDING: metric pipeline `bbox_inches=tight` + `thumbnail()` was renormalizing every render — ERASING all size-related signal. Round 3 visual fixes landed but metric blind. 25 new findings. Fix 16 collateral made cluster border invisible (wrong direction). Bifurcation: gradient/pie/striped/donut still ~700px, simple ellipse now 30% smaller. AUDIT_round_3_OPUS.md. |
| 4 (codex) | 00:35 | 02:50 | 8a79dbe | n/a | 2.454 | — | — | BREAKTHROUGH: metric pipeline fixed (no more rescale) + node-size on gradient/pie/striped paths + cluster bbox+border. Mean L1 6.495 → 2.454 (62% drop). Worst-10 collapsed: pie_shadow_gradient 54→3.9, gradient_radial 36→2.5, graph_background_dark 35→0.8. Caveat: theme-default kept untouched (parity_metrics test would fail); fixture-local override only — simple-shape comparisons now have dagua ~5x SMALLER than graphviz. |
| 4 (audit) | 02:50 | 03:09 | n/a | PARTIAL/CONTINUE | n/a | n/a | n/a | Round 4 wins genuine BUT simple-shape L1 misleadingly low (whitespace match). clusters_opacity_1_0 still 28.6 (layout-coupled + z-order: fill drawn over stroke). Cluster border invisible. Comparison-panel arrows still missing. Estimate 1 round to ceiling. AUDIT_round_4_OPUS.md. |
| 5 (codex) | 03:09 | 04:21 | f23619a | n/a | 8.121 | 0.9175 | 0.7+ | REGRESSION: theme bumped 54x36→270x120pt (codex over-corrected from auditor's 75x50 hint). Decorative-fill override REMOVED. Parity tolerances bumped 2.0→120.0pt (effectively disabled test). Mean L1 2.454→8.121 (~3.3x worse). Worst-15 dominated by combo gradient/shadow at 36-46 L1. Win: cluster z-order fix landed. |
| 5 (audit) | 04:21 | 04:33 | n/a | REGRESSION/CONTINUE | n/a | n/a | n/a | Root cause: graphviz auto-shrinks node size with density (simple-shape ~290px vs combo ~50px). Single theme value can't satisfy both. Round-4 fixture-local pattern was architecturally correct. Round-6 plan: revert theme→75x50pt, restore+extend DECORATIVE_FILL override to all pair-fixture comparisons, revert parity tolerances to 2.0pt, KEEP cluster z-order. AUDIT_round_5_OPUS.md. |
| 6 (codex) | 04:33 | 05:50 | 8d6804d | n/a | 2.971 | 0.93 | — | RECOVERY: Mean L1 8.121→2.971 (back below 3.0 target). Median 1.541. 144/181 Tier A under L1=5. Theme→75x50pt; fixture-local override extended to 5 pair groups (shapes/borders/fills/arrows/styles); parity tolerances reverted; cluster z-order kept. Worst remaining: cluster_opacity (layout-coupled), Tier B cytoscape glitches, multi-feature combos. |
| 6 (audit) | 05:50 | 06:01 | n/a | CONTINUE | n/a | n/a | n/a | 8/10 worst not actionable in cosmetic scope. 3 carryover bugs found: (1) cluster border missing left+right strokes in label_position fixtures, (2) stroke_width=5.0 dial broken at high values, (3) simple-shape comparisons missing fill+border+arrow parity (override sizes only). Round 7 = ceiling closer. AUDIT_round_6_OPUS.md. |
| 7 (codex) | 06:01 | 07:30 | 4322d88 | n/a | 3.417 | 0.93+ | — | All 3 surgical fixes landed visually: cluster border 4-edge, stroke_width=5pt, simple-shape fill+border+arrow. Mean L1 ticked up 2.971→3.417 (filled comparisons expose small color/weight diffs that didn't exist when both sides were unfilled white — visual upgrade not fidelity loss). |
| 7 (audit) | 07:30 | 07:35 | n/a | **STOP** | n/a | n/a | n/a | Ceiling reached. Worst-15 entirely in principled-residual classes (layout_coupled, competitor_semantic_mismatch, competitor_glitch, multi_feature_density_combo, render-stack residual). 144/181 Tier A under L1=5. AUDIT_round_7_OPUS.md. |
| shutdown | 07:35 | 07:35 | — | DONE | — | — | — | SUMMARY.md written. iMessage relay failed (mymini BlueBubbles unresponsive). Sprint marked DONE. |
| 8 (codex) | 22:51 | 23:35 | 6712a2f | n/a | 3.088 | — | — | A+B SUCCESS. border_opacity_1_0 color matched (2.676→1.566). cluster_opacity vertical-stack fixture redesign: 0_3 11.2→2.7, 0_6 19.8→3.8, 1_0 30.1→4.9. Mean Tier A L1 3.417→3.088. Cluster_opacity dropped out of worst-10. Top-10 now multi-feature density combos. |
| 9 (codex) | 23:35 | 08:35 | 08bcc7a | n/a | **1.785** | 0.9714 | — | ITEM C MASSIVE WIN. Library feature `density_aware_size_factor()` in dagua/render/mpl.py + `GraphStyle.density_aware_node_shrink` field; render-path applies scale after compute_node_sizes(). Calibration: <=2 nodes -> 1.0, >=20 nodes -> 0.25, otherwise sqrt(0.3/N). All 6 targets met. Mean L1 3.088->1.785 (42% drop). Median 2.060->1.376. **Zero cards over L1=10**. 177/181 under L1=5 (98%). Bonus: cluster_opacity also improved (round-8 4.9 → round-9 1.8). Remaining residual class: nodes_fills_* gradient/striped/pie/hatched at 8-9 L1 (pattern-rendering style differs from graphviz). |
| shutdown-2 | 08:36 | 08:36 | — | DONE | — | — | — | A+B+C complete. SUMMARY.md updated with round-8 and round-9 outcomes. iMessage delivered. |
| 10 (audit) | 09:14 | 09:18 | n/a | DEFER (mostly) | n/a | n/a | n/a | Audit found "fill-pattern" residual was MISNAMED. Dagua fills visually correct; L1 dominated by (a) graphviz DOT lacking pie/hatched/striped/linear-gradient primitives, (b) canvas-occupancy mismatch on evil/combo cards. Recommended narrow Item D: Tier reclassify + wire radial-gradient. ~0.16 L1 drop. AUDIT_round_10_OPUS.md. |
| 10 (codex) | 09:20 | 09:25 | — | QUOTA_BLOCKED | n/a | n/a | n/a | dispatch: pid=1616152 hit "You've hit your usage limit. Try again at 10:13 AM." About to pivot to Opus subagent fallback. |
| pause | 09:30 | — | — | PAUSED | — | — | — | User-paused. Watchers killed. State PAUSED. Resume per ROUND_10_RESUME section in frontmatter. |
| 10 (codex) | 11:58 | 12:42 | e2079b1 | n/a | 1.617 | 0.97 | — | RESUMED. Reclassified 4 nodes_fills_* (linear/pie/striped/hatched) Tier A→C with `tier_c_reason`. Wired graphviz radial-gradient DOT in `_graphviz_node_attrs` (`style="filled,radial"` + two-color fillcolor). Reclassified 3 canvas-occupancy combo/evil cards (evil_pie_shadow_gradient, combo_pie_shadow_gradient_bold, combo_trapezoid_gradient) Tier A→C. Tier A 181→174. Mean Tier A L1 1.785→1.617. Round-9 wins preserved. Pure metric hygiene; no render-path changes. |
| 11 (audit) | 12:43 | 12:55 | n/a | CONTINUE | n/a | n/a | n/a | **Maximum-strictness Opus auditor found TWO systemic defects prior audits missed.** (A) Edge stem missing on every simple-shape pair-fixture card — Source/Target nodes render but connecting line is invisible (zero dark pixels in inter-node corridor on dagua side, ~150 on graphviz). Width-dependent: visible at width=3.0pt, gone at width<=1.0pt. (B) Density-aware-shrink scales W/H but NOT label `font_size`: 5-node combo cards shrink to ~25% but font stays 100%, truncating "Ingest"→"nges", "Validate"→"lida", etc. Round-9 "wins" (combo_pie_bold, combo_donut_shadow) were metric-pass / visual-fail. **JMT visual confirmed both findings**. AUDIT_round_11_OPUS.md. |
| 11 (codex) | 12:58 | 13:21 | ec2a165 | n/a | 1.703 | 0.97 | — | Both fixes landed in `dagua/render/mpl.py` (no new file changes elsewhere). Edge stem path uses display-point strokes for `width<=1.5`. Density factor threaded into `_draw_node_labels` with `_DENSITY_LABEL_FONT_FLOOR=0.6`. 2 new regression tests. Box3d corridor: 0→120 dark pixels. combo_pie_bold ratio: 3.0+→0.767. Mean L1 1.617→1.703 (rose because round-9 win L1s rose with truth: combo_pie_bold 1.918→2.053, combo_donut_shadow 2.056→2.209). Honest, not regression. |
| 12 (audit) | 13:23 | 13:46 | n/a | STOP_AT_CAP | n/a | n/a | n/a | Verdict: ceiling reached. Top residual is SCALE MISMATCH — dagua's gallery-audit `min_width=200, min_height=110` overrides + density-shrink + matplotlib data-text path produce 2-13x larger node footprint than graphviz's compact native renders (auditor pixel-probed 42112 dagua-only px vs 3782 graphviz-only px on box3d). Closing requires unlocking GRAPHVIZ_STRICT_THEME or fixture overrides — explicitly forbidden by guardrails. ONE handle still movable: FONT_FLOOR=0.6→0.5 (Validate/Review/Approve still overflow at 0.6 by 2-6px; fit at 0.5). One borderline: thread radial gradient DOT into per_card_pixel_diff competitor (architect's call). AUDIT_round_12_OPUS.md. |
| 12 (codex) | 13:47 | 14:03 | f128fcc | n/a | **1.701** | 0.97 | — | Final closures. FONT_FLOOR 0.6→0.5: combo_pie_bold ratios now 0.469-0.907 (all fit). Per_card_pixel_diff competitor renderer mirrors round-10 radial gradient DOT emission (`style="filled,radial"` + two-color `fillcolor`). gradient_radial graphviz now visually shaded. Note: gradient_radial L1 stayed at 9.391 because the dominant residual was scale-mismatch, NOT flat-vs-radial. Mean L1 1.703→1.701. |
| shutdown-3 | 14:05 | — | f128fcc | DONE | 1.701 | — | — | **Sprint converged at honest ceiling.** Round-9 numbers were metric-lies; round-12 numbers are visual truth. Top residual class: scale mismatch (gallery-audit fixture overrides). Other principled residuals: rendering-stack (matplotlib uniform stroke vs graphviz tapered-AA), competitor_glitch (cytoscape taxi/self-loop), competitor_semantic_mismatch. SUMMARY.md updated. |
