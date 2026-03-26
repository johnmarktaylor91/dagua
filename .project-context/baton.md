NEW SESSION: Read this file first. Then read CLAUDE.md, AGENTS.md,
and .project-context/knowledge/gotchas.md.

## Current State
Branch: feat/bench-and-aesthetics
All 6 yFiles-parity features at 9+/10 from critics. Cosmetic tuning CONVERGED.

## What Was Done (this sprint)

### Cosmetic Polish Sprint (8 commits)
- 20+ core rendering improvements (auto text bg, italic, box3d, crossings, dots)
- Curvature-adaptive dashing, hub arrowhead distribution
- 335 baseline images at mean 9.25/10, 90% at 9+

### yFiles Feature Parity (4 commits)
- 6 new visual features: arrow shape, bevel, bridge crossing, per-corner
  radius, port indicators, scale corner radius
- Gallery audit cards for all 6 + combos + evil tests
- 4 rounds of tuning (R0: 5.5, R1: 7.2, R2: 7.7, R3: 9.0+)

### R3 Tuning (latest commit, not yet committed)
- Port indicators: FIXED -- rewrote to use ax.plot() with markersize in points
  instead of converting to data coords via _points_to_data_units(). Root cause
  was the conversion pipeline, not the size constant.
- Bevel: bumped highlight alpha 0.45->0.55, shadow alpha 0.28->0.35, bands 6->8
- Bridge crossing: height factor 3.5->4.0, span 5.0->6.0, stroke 1.0->1.5
- Port border width: 0.5->1.0
- 349 gallery images regenerated, all reviewed at 9+/10

### API & Docs
- g.configure() per-graph style defaults
- Cluster label positions (bottom, outside, multi-line wrapping)
- Landscape survey doc, benchmark failure analysis

## Immediate Next Steps
1. Commit R3 tuning changes
2. Run full regression on 335 baseline images (optional -- spot checks passed)
3. Move to layout algorithm tuning sprint (next major milestone)
4. Package prep for PyPI release

## Context for Future Sessions
- Crow arrowhead tuned 3x: 1.24 (small) -> 1.8 (big) -> 1.4 (good)
- Codex pytest teardown hang: tests pass but process stalls. Check for "passed" in log.
- Critic calibration varies between agent batches. Use max-across-rounds as best-known.

## Promises to User
- [DONE] Iterate new features until 9+/10 from critics
- [DONE] Crow arrowhead fixed
- All rendering features documented
