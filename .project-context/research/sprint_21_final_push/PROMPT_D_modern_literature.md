# Area D — Modern algorithmic literature scan (post-2024)

## Question

What graph-drawing techniques from 2024-2026 papers/preprints have
NOT been tried in dagua and could plausibly close gaps on the
remaining 11 sub-dominate graphs?

Sprint-19 wave-2 already surveyed pre-2024 literature. This sprint
focuses on:

- GD 2024 / GD 2025 papers
- TVCG 2024-2026 graph layout papers
- arXiv 2024-2026 preprints (cs.HC, cs.DS, math.CO categories)
- ICML / NeurIPS 2024 graph-drawing-adjacent papers (graph
  representation learning that produces 2D embeddings)
- Conformal mapping / Tutte / Schnyder / harmonic embedding
  developments
- Differentiable layout solvers (gradient-based, where dagua sits)
- Constraint-projection methods

## Specific targets (dagua's remaining loss patterns)

1. **Lattice-style graphs (-2 to -2.5)**: graphviz_dot wins by
   deterministic grid placement. Are there modern conformal /
   harmonic-embedding methods that produce uniform edge lengths
   without sacrificing depth_spearman?

2. **Non-planar regular (-2.7)**: petersen_10. What 2024-2026
   work targets non-planar regular graphs specifically?

3. **Large DAGs (-2.9)**: dependency_500. Modern hierarchical
   layout work for N >= 500?

4. **Multi-component cyclic (-1.99)**: disconnected_label_cycle_collage.
   Modern per-component tiling that respects depth_spearman?

5. **Stochastic-noise reduction in metric-driven optimization**:
   the polish op picks via composite(full(...)) with seeded RNG;
   the metric itself uses sampling. Is there a deterministic
   variant of stress / crossings metrics that scales?

## Research method

Use exa MCP web_search_exa for primary search. Hit:
- "graph drawing 2024 lattice"
- "differentiable graph layout 2024"
- "stress majorization neural 2024"
- "Sugiyama variant non-planar 2025"
- "edge length uniform graph drawing"
- specific paper titles cited in older dagua research (Gortler et al
  2006 conformal one-form, Schnyder embedding, etc.) and check
  citations to find recent extensions.

Use web_fetch_exa to read promising abstracts in full.

## Output format

`.project-context/research/sprint_21_final_push/D_modern_literature__<your_agent_name>.md`

Include:
- TL;DR with the 3 most promising techniques to integrate
- For each technique:
  - Citation (paper title, authors, venue, year)
  - Core idea (1 paragraph)
  - Predicted impact on dagua's specific gaps
  - Implementation difficulty (LOC estimate)
  - Risk of regression
- Anti-recommendations: techniques that LOOK promising but won't
  help dagua's metric set (e.g. edge bundling improves aesthetics
  but the dagua metric doesn't reward it).

## Constraints

- READ-ONLY. Findings file only.
- Read `.project-context/research/sprint_21_final_push/CONTEXT.md` first.
- Use web search aggressively. Cite every paper.
- Budget: 1500-3000 words.
