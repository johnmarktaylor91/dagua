# Open Issues

This is the short active ledger for major unresolved work.

Keep it current. Keep it terse.

## Placement

- Finish the placement-first benchmark loop against competitor baselines.
- Confirm where Dagua genuinely loses on placement metrics vs where the visual layer is only making it look worse.
- Keep expanding the challenge-graph corpus when new structural blind spots appear.

## Scale

- The 1B run now gets through hierarchy build and into coarsest optimization; keep pushing until it completes end-to-end.
- Longer-term: checkpoint/resume for rare giant runs should improve beyond logs and run directories.
- Keep adding targeted telemetry where long phases are still too opaque.

## Visual Language

- Full visual reset still pending.
- Text hierarchy, edge language, and cluster treatment are currently the weak layer.
- Use the visual-audit suite and competitor stepwise comparisons rather than ad hoc screenshots.

## Benchmarking

- The first full standard baseline is still expensive; future rounds should mostly reuse non-Dagua competitors.
- Keep the report surfaces aligned:
  - `benchmark_deltas.md`
  - `layout_similarity.md`
  - `placement_summary.md`
  - `placement_dashboard.md`

## TorchLens

- Graphviz remains the default; keep Dagua opt-in until the visual reset is genuinely good.
- TorchLens semantic mapping and Dagua visual language should be debugged separately.

## Docs / Workflow

- Keep the docs index, developer overview, command cheat sheet, and maintenance checklist current as the surface area evolves.
- Maintain frozen baselines once the benchmark baseline stabilizes and the visual reset begins in earnest.
