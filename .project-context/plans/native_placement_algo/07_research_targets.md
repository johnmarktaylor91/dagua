# Research Targets

Literature to mine and competitor implementations to study. Use
`superpowers:search-first` -- prefer adapting a proven approach over
writing net-new code when quality is equivalent.

## Priorities by sprint

| Sprint | Primary targets |
|--------|----------------|
| 1 | Initialization papers (topological, spectral, warm-start); OGDF init code |
| 2 | Multilevel / coarsening: SFDP, FM^3, Walshaw multilevel framework |
| 3 | Hybrid papers: layer-sweep, Brandes-Kopf coord assignment, ECS/maxent |
| 4 | Hierarchical clustered layouts: Sander (1996), ELK compound, cola.js |
| 5 | Constrained graph layout with pinning: Dwyer stress majorization with constraints |
| 6 | Edge routing: Holten FDEB, stub bundling, confluent drawings |
| 7 | Label placement: Kakoulis-Tollis, iLabel |
| 8 | Scale: Barnes-Hut, grid approximation, GPU ForceAtlas2 |
| 9 | Tuning: multi-objective tuning, Pareto methods |

## Literature

Required reading. Prioritize authors and papers with installed reference code
if available, per the authority hierarchy in
`.project-context/knowledge/gotchas.md`: installed code > upstream repo > paper.

- Purchase (1997): aesthetic criteria ranking
- Sugiyama, Tagawa, Toda (1981): hierarchical framework
- Gansner et al. (1993): Graphviz algorithm and network simplex
- Gansner et al. (2005): SFDP multilevel
- Hachul, Junger (2004): FM^3 multilevel force-directed
- Walshaw (2001): multilevel framework for graph drawing
- Brandes, Kopf (2001) + Brandes, Walter, Zink 2020 erratum: coord assignment
- Dwyer, Koren (2005): constrained stress majorization for user pins
- Holten (2006), Holten & van Wijk (2009): force-directed edge bundling
- Dickerson et al. (2003), Bach et al. (2017): confluent drawings
- Pupyrev et al. (2013): stub bundling
- Dogrusoz et al. (2009): force-directed compound layout
- Hobby (1986): aesthetic spline fitting
- Ahmed, Kobourov, et al. (2022): sgd^2 variant
- Kruiger et al. (2017): graph layout quality evaluation
- Huang et al. (2014): graph readability meta-review
- Kwon, Ma (2019): A Deep Generative Model for Graph Layout
- Giovannangeli et al. (2021-2024): neural graph drawing
- Wang et al. (2020): GraphTSNE, graph embedding-based layouts
- Born et al. (2023): ELK paper (Domros alternative reference)

## Implementations to read

Hierarchy: installed source code beats paper. For each, identify the exact
file and line range where the relevant algorithm lives.

- **OGDF** (C++). Reference implementations of FM^3, SFDP variants, compound
  layout, layer sweeps. Clone to /home/jtaylor/projects/refs/OGDF.
- **Gephi** (Java). ForceAtlas2 canonical implementation. Barnes-Hut.
- **Graphviz sfdp** (C). Coarsening + refinement pipeline.
- **NetworkX** (Python). spring_layout, kamada_kawai_layout, spectral_layout.
  Small and readable; good for initialization patterns.
- **igraph** (C + bindings). layout_with_sugiyama, layout_with_sugiyama_c,
  layout_with_fruchterman_reingold.
- **cytoscape.js / cose-bilkent** (JavaScript). Compound layout.
- **cola.js** (JavaScript). Constrained layout -- best reference for Dwyer
  constrained stress.
- **WebCola** (derived cola). Similar reference.
- **ELK / elkjs** (Java/JavaScript). Hierarchical with compound; known to
  stack-overflow at 5-7K but the algo is instructive.
- **dagre** (JavaScript). Sander-based compound layout; readable code.
- **grandalf** (Python). Pure-Python Sugiyama. Good simple reference.
- **sgd2 / s_gd2** (Python + C). Already reimplemented in Dagua; re-read for
  warm-start patterns.
- **umap-learn / cuGraph UMAP** (Python). Embedding-based; structural init
  ideas for undirected.

## Extraction targets

For each implementation, extract:

- Initialization policy (how they start).
- Coarsening policy (maximal matching, heavy-edge, star-contraction).
- Prolongation policy (how coarse positions seed fine positions).
- Convergence criterion (when to stop).
- Annealing schedule (temperatures, weights over iterations).
- Edge length heuristics (uniform, weighted, span-aware).
- Pinning / constraint handling.
- Cluster / compound layout bookkeeping.
- Specific parameter defaults that look well-tuned.

Do not extract code structure or class hierarchy -- extract ideas. Dagua's
architecture is ops-first; the borrowed idea lands as a registered op, not
as a class.

## Subagent dispatch patterns

For literature review, dispatch Claude subagents in parallel:
- Subagent A: "Read Dwyer constrained stress papers and summarize the
  implementation trick for hard pins."
- Subagent B: "Read SFDP source in Graphviz and extract the coarsening schedule."
- Both return <=400 words. Claude synthesizes into the relevant sprint file.

For competitor code reads, similarly parallel. Target: 5-subagent batch for
Sprint 2 (SFDP, FM^3, dagre, cose-bilkent, OGDF multilevel).

## Research side-notes

Push long-form notes into `.project-context/knowledge/research/<topic>.md` with
the format:
```
# <topic>

Source: <paper/repo URL or file path>
Authority: installed / upstream / paper-only
Relevance: sprint N

Key ideas:
- ...

Extraction candidate for Dagua:
- ...
```

The sprint file links to the side-note(s) but does not embed the raw
research.

## Don'ts

- Do not treat papers as code. Always cross-check against the installed
  reference. Papers are ambiguous.
- Do not re-read the same material for multiple sprints. Synthesize once,
  cite often.
- Do not dispatch more than 5 parallel subagents without the pre-dispatch
  checklist.
- Do not port C++ or Java verbatim. Extract the idea, re-implement as a
  Dagua registered op.

## See also

11_competitor_weaving.md is the per-sprint execution map for extraction.
This file (07) is the pool of candidates; 11 names the ones each sprint
MUST attempt, with op names and exit checks.
