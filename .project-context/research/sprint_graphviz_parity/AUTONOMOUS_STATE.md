# Overnight autonomous state

**Started:** 2026-04-28 night
**Goal:** iterate audit → fix → audit until visual audit returns ZERO findings of class `real_cosmetic_gap + fixable_theme_or_render`. Then send summary text to user.

## Current state
- Round: **B1 in flight** (codex implementing 4 fixes from AUDIT_A1)
- Next expected wake-up: watcher `brldhf1u6` fires `CODEX_B1_DONE` or `CODEX_B1_LIMIT_HIT`
- Most recent commit: 64a0936 (procedure infra)

## Wake-up protocol (FOLLOW EXACTLY ON EACH NEW TURN)

On ANY new turn (watcher event OR user message), FIRST run this triage:
```
1. Check git log: `git log --oneline -3` — did codex commit since last turn?
2. Check codex alive: `pgrep -af "codex/codex exec"` — codex still running?
3. Check this file's "Current state" line — what round are we on?
```

Then route per these cases:

### Case A: codex committed AND not running (good path)
1. Run pipeline:
   - `python scripts/parity_metrics.py` (regenerates JSON + summary.md)
   - `python scripts/parity_pixel_diff.py` (regenerates pixel diff)
   - `python scripts/parity_pixel_diff.py --hires <worst 5-7 panels by SSIM from new summary>`
2. Dispatch next audit (Opus subagent) with all updated inputs and prior AUDIT_<prev>.md as "Prior findings to re-check".
3. Process audit output:
   - If audit verdict says STOP → run "Final shutdown" section below
   - If CONTINUE → write next PROMPT_<next>.md, dispatch codex `--background`, arm watcher
4. Update "Current state" in this file with the new round letter.

### Case B: codex still running (no-op turn)
1. Just acknowledge in chat: "still cooking, watcher is armed"
2. End turn

### Case C: watcher fired LIMIT_HIT (codex quota)
1. Read codex log to confirm quota error
2. Switch fallback path: dispatch Opus subagent with the SAME PROMPT_<round>.md as implementer
3. After Opus finishes, run pipeline + audit (Case A from there)
4. If Opus also hits limit, run "Final shutdown" with status="quota blocked"

### Case D: tests fail / dirty repo
1. Investigate: read the test failure output
2. If fixable in <10 min, fix and commit
3. Otherwise revert codex's changes (`git reset --hard HEAD` carefully) and skip to next round
4. Document in REPORT what failed

### Case E: anti-flail (3 consecutive rounds with same unclosed issue)
1. Mark issue as `principled_residual` even if it's classified differently
2. Document in REPORT_<round>.md
3. Continue to next round

### Final shutdown
When STOP criterion is met:
1. Write final summary at `.project-context/research/sprint_graphviz_parity/SUMMARY.md` with:
   - All commits in this overnight run
   - Final metric stats (in-tolerance %, mean/median SSIM, mean/median L1)
   - Locked features count
   - List of accepted residuals with classification
   - Anything deferred to layout work
2. Send text via `~/.claude/scripts/send-to-jmt.sh` with the highlights
3. Send 4-5 representative side-by-side images via `send-to-jmt.sh -a`
4. Update this file: Round = "DONE"

## Stop criteria

- Audit returns NO findings classed `real_cosmetic_gap + fixable_theme_or_render`, OR
- 3 consecutive rounds with same un-closeable issue (anti-flail), OR
- Codex AND Opus subagents both quota-blocked (resume in morning), OR
- Catastrophic test failure that can't be recovered without user input

## Iteration log

- A1 audit: FAIL, 5 fixable HIGH findings (canvas-fill, label-wrap, ellipse_rx, arrows×4, cluster-geom-deferred). AUDIT_A1.md.
- B1 fixes: COMMITTED `9c14892`. 95.74% → 99.27% in tolerance. Mean L1 17.37→16.95, mean SSIM 0.77→0.76 (slight regression). REPORT_B1.md.
- A2 audit: FAIL/CONTINUE. 5 HIGH findings: figure aspect (canvas-fill phase 2), arrowhead rhombus regression, per-edge arrowsize ignored, single-line ellipses too circular, long-label rx + edge-label font over-correction. AUDIT_A2.md.
- B2 fixes: COMMITTED `27646de`. Declarative still 99.27%, pixel SSIM regressed slightly (0.761→0.759, worst 0.529→0.523) due to over-corrected oval floor.
- A3 audit: FAIL/CONTINUE. 5 HIGH findings: oval floor 1.85→1.50 (one-line fix, root cause of 4 downstream issues), edge stroke gray vs black (matplotlib AA), long-label rx still narrow. AUDIT_A3.md.
- B3 fixes: COMMITTED `6a931aa`. Modest pixel improvement.
- A4 audit: STOP verdict — render-stack ceiling.
- B4 fixes: COMMITTED `b00f434`. Edge stroke 1.5x + capstyle/opacity. Slight metric regression confirms ceiling.
- **Round: DONE 2026-04-29 morning.** Final 99.27% declarative in-tolerance, mean SSIM 0.759. SUMMARY.md + 5 images sent to JMT.
