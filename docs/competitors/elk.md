# ELK Layered

## Official docs

- ELK Layered algorithm reference:
  <https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html>
- ELK reference options index:
  <https://eclipse.dev/elk/reference/options.html>

Useful option families:

- edge label placement
- edge label side selection
- edge routing
- node spacing / edge spacing / label spacing
- individual spacing
- hierarchy handling
- cluster crossing penalty
- BK edge straightening

## Read these topics first

- supported graph features
- compound / cluster / port support
- edge label placement strategies
- spacing model
- hierarchy handling and crossing-minimization options

## Source entrypoints to inspect

- layered algorithm implementation
- BK placement / edge straightening
- label placement
- compound / hierarchy handling

## Why ELK matters for Dagua

- strongest modern layered competitor in the same broad problem family
- more explicit than Graphviz about geometry options
- useful reference for:
  - ports
  - labels
  - compound graphs
  - routing choices

## Local outputs to compare

- `../../eval_output/visuals/comparisons/`
- `../../eval_output/visual_audit/competitor_stepwise/`
- `../../eval_output/visual_review_session/`

## Questions to ask while reading

- Which ELK geometry decisions are explicit options vs hardcoded policy?
- Where does ELK gain readability from ports and label handling?
- What can Dagua emulate numerically and what should remain heuristic?
