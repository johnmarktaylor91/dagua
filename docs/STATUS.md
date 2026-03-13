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
  0. keep the criteria ledger explicit so we do not miss whole geometry classes
  1. lock down placement quality with metrics and benchmarks
  2. build a real stage-2 downstream geometry workflow for edges / labels / clusters
  3. use the visual-audit/session workflow to redesign rendering defaults
  4. keep the billion-node path alive and progressively more robust

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
- `docs/CRITERIA_LEDGER.md`
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
2. Define and implement the stage-2 numerical workflow for edge / text / cluster geometry.
3. Keep hardening the 1B path.
4. Iterate on default visual settings using the numbered session workflow.
5. Expand strict typing and maintainability incrementally, not performatively.

## Recent 1B Hardening

- graph checkpointing is in place
- layering checkpointing is in place
- duplicate-run ownership is now guarded even if metadata is missing
- shell-wrapper false positives in the duplicate-run guard were fixed
- checkpoint tensors are shape-validated before restore
