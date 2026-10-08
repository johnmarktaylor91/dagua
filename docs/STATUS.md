# Status

Current high-level project state, meant as a fast handoff note between sessions.
Last refreshed 2026-10-07.

## Core Read

- Dagua is a working, public engine (package version 0.5.0) with a large composable layout core and a heavy
  evaluation stack.
- **Development has been paused since 2026-09-01.** The last engineering commit is dated that day; everything
  after it is repository housekeeping.
- The next effort has not been chosen. Candidates are resuming placement work, finishing the ruler v4 fit,
  or starting the visual reset. Nothing below is in flight.

## Placement

- Goal: make native layout beat dot, sfdp, ELK, dagre, igraph and OGDF on a shared corpus
  (`docs/native_algo_iteration.md`), judged by a frozen, hard-to-game metric called the "ruler".
- July result on the honest ruler (July 6 to 10): 73 of 108 graphs best-or-tied, up from 63 with no regressions.
  On 271 unseen holdout graphs the win rate was 73.8 percent, but only 25 percent on the suitesparse family.
  Earlier 74 to 90 of 108 figures came from a ruler that favoured Dagua and should not be quoted.
- Scaling: multilevel layout with tiled GPU loss reaches 1B nodes in the architecture; the Graphviz runtime
  crossover is about 3 to 4K nodes (March measurement). Last scaling fixes landed early August.

## Ruler v4 (parked)

- Built August 14 to September 1: `dagua/eval/ruler_v4/` with 45 facet contracts, composition, an ordinal
  headline, a traced differentiable surrogate and a fit harness (ordered probit, JND pairwise, holdout).
  Scorer performance work finished August 23 to 31.
- **Parked.** The code stays as it is. Two freezes are decided only when placement work resumes, and every real
  fit fails closed until then: the lapse prior (family and strength) and the graph-to-half assignment. They
  are entries 56 and 57 in `dagua/eval/ruler_v4/DISCREPANCIES.md`.
- When resumed: record the two freezes, add a lock guard to the pilot runner, run the first real fit, then
  re-run the standard benchmark under ruler v4.

## Sprint 2 branches

- Lanes W1-A, W1-B, W1-C, W2-1 and W2-3 are on main.
- W2-2 (stress-family arms) and W2-4a (shadow champion) are combined on branch `sprint2/w2-2-w2-4a`. The
  combination has **not been tested against the July baseline**; that comparison waits for placement work to
  resume.
- Two preserved work-in-progress branches on the ruler v4 cluster facets, `wip/clusters-ad8164eb` and
  `wip/clusters-aec43828`, are kept untouched until that work is reviewed.

## Visuals

- Visual Parity v2 (July 14 to 16) is done: node scale and fonts pixel-matched against Graphviz 7.0.5,
  arrowheads 96.37 percent declarative parity, shapes and composition 99.86 percent
  (`scripts/visual_parity`).
- The visual reset in `docs/VISUAL_RESET_BRIEF.md` has **not started**. The weak layer is still visual language:
  text hierarchy, edge language, cluster box treatment and overall composition. Next in that line are the
  parity-v2 gap plan (cluster and dense panels, waivers, metric artifacts) and the minimal-baseline reset.
- Port-fan edge routing (corridor-budgeted polylines for layered directed drawings, an unmerged July branch)
  is archived as a git bundle with notes, to be revisited with the visual work. It needs a rebase onto the
  current `dagua/edges.py`.
- Do not mistake current rendering quality for placement quality. Use `eval_output/visual_audit/` and
  `eval_output/visual_review_session/` and compare against competitors before changing defaults.

## Benchmarks

- Standard benchmark suite: persistent, resumable, cached competitor reuse, report and delta artifacts.
  The rare suite is explicit and runs a scaling ladder through 1B nodes.
- Read first: `docs/BENCHMARK_ARTIFACT_GUIDE.md`, `docs/CRITERIA_LEDGER.md`, `docs/ITERATION_WORKFLOW.md`,
  `docs/MONEY_GRAPHS.md`.

## Scale

- The 1B path is real. Checkpointing for graph and layering, duplicate-run guards and checkpoint shape
  validation are in place.
- Remaining risks: memory spikes in hierarchy and coarsening, giant coarse-level initialisation and refinement
  transitions, and long-run robustness.
