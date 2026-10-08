# Open Issues

This is the short active ledger for major unresolved work. Keep it current. Keep it terse.
Last refreshed 2026-10-07. Development is paused (see `docs/STATUS.md`); the next effort is not chosen.

## Placement

- Suitesparse-family graphs are the weak spot: 25 percent holdout win rate, against 73.8 percent overall on 271
  unseen graphs (July honest ruler; 73 of 108 best-or-tied).
- Edge crossings were the main gap against `graphviz_dot`, ELK and dagre in the last full standard run;
  they have not been re-measured under ruler v4.
- Sprint 2 W2-2 (stress-family arms) and W2-4a (shadow champion) sit combined on `sprint2/w2-2-w2-4a`, untested
  against the July baseline. The stress-family candidates are not registered in the shadow champion's
  new-arm prefix list (`NEW_ARM_FAMILY_PREFIXES`), so a stress-arm win does not trigger the shadow re-run;
  decide that when testing.
- Cluster-aware placement criteria (sibling overlap, containment margin, cluster separation) and hierarchical
  cluster-mediated interactions are open. Two preserved WIP branches, `wip/clusters-ad8164eb` and
  `wip/clusters-aec43828`, hold ruler v4 cluster-facet work awaiting review.
- Overlapping cluster boxes start in node placement, not in styling.

## Ruler v4

- Parked. Two freezes block every real fit: the lapse prior (family and strength) and the graph-to-half
  assignment (`dagua/eval/ruler_v4/DISCREPANCIES.md`, entries 56 and 57). Decide them when placement resumes.
- The pilot runner still needs a lock guard against duplicate concurrent runs.

## Scale

- Keep pushing the 1B run until it completes end to end; keep adding telemetry where long phases are opaque.
- Longer term: checkpoint and resume for giant runs beyond logs and run directories.

## Visual Language

- The visual reset (`docs/VISUAL_RESET_BRIEF.md`) has not started. Text hierarchy, edge language and cluster
  treatment are the weak layer.
- Visual Parity v2 is done; its gap plan (cluster and dense panels, waivers, metric artifacts) is not.
- Port-fan polyline routing is archived, unmerged, and needs a rebase onto the current `dagua/edges.py`.
- Add a distinct stage-2 numerical workflow for edge, text and cluster geometry before the final aesthetic pass.
- Use the visual-audit suite and competitor stepwise comparisons rather than ad hoc screenshots.

## Benchmarking

- Future rounds should mostly reuse cached non-Dagua competitor results.
- Re-run the standard benchmark under ruler v4 when placement work resumes.
- Keep report surfaces aligned: `benchmark_deltas.md`, `layout_similarity.md`, `placement_summary.md`,
  `placement_dashboard.md`.

## TorchLens

- Graphviz remains the default; keep Dagua opt-in until the visual reset is genuinely good.
- TorchLens semantic mapping and Dagua visual language should be debugged separately.

## Docs / Workflow

- Keep the docs index, developer overview, command cheat sheet and maintenance checklist current as the
  surface area evolves.
- Freeze baselines once the benchmark baseline stabilises and the visual reset begins.
