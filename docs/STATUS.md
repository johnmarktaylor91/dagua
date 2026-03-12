# Status

Current high-level project state, meant as a fast handoff note between sessions.

## Core Read

- Node placement and scaling are the strongest parts of the system.
- The current weak layer is visual language:
  - text hierarchy
  - edge styling / routing expression
  - cluster box treatment
  - overall composition
- The active strategy is:
  1. lock down placement quality with metrics and benchmarks
  2. use the visual-audit/session workflow to redesign rendering defaults
  3. keep the billion-node path alive and progressively more robust

## Benchmarks

- Standard benchmark suite:
  - persistent
  - resumable
  - cached competitor reuse
  - report + delta artifacts
- Current placement read from the finished standard run:
  - Dagua is strong on DAG consistency and overlap avoidance.
  - Dagua is not yet winning on edge crossings against `dot`, `ELK`, and `dagre`.
  - Edge-length uniformity is mixed:
    - often better than `dot`
    - mixed vs `ELK` / `dagre`
    - usually worse than force-directed baselines on that specific metric
- Rare suite:
  - explicit/manual
  - scaling ladder through `1b`

Read first:
- `docs/BENCHMARK_ARTIFACT_GUIDE.md`
- `docs/ITERATION_WORKFLOW.md`
- `docs/MONEY_GRAPHS.md`

## Scale

- Billion-node work is real and active.
- The main remaining billion-node risks are:
  - memory spikes in hierarchy/coarsening
  - giant coarse-level initialization / refinement transitions
  - long-run robustness rather than basic architecture

## Visual State

- Do not mistake current rendering quality for placement quality.
- Use:
  - `eval_output/visual_audit/`
  - `eval_output/visual_review_session/`
- Compare against competitors before changing defaults.

## Current Priorities

1. Reduce the placement gap on edge crossings against the best hierarchical competitors.
2. Keep hardening the 1B path.
3. Iterate on default visual settings using the numbered session workflow.
4. Expand strict typing and maintainability incrementally, not performatively.
