# NetworkX spring layout

## Official docs

- Stable `spring_layout` reference:
  <https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html>

## Read these topics first

- Fruchterman-Reingold model
- `k`, `iterations`, `scale`, `center`
- `fixed` node behavior
- `method='force'` vs `method='energy'`
- gravitational behavior in the energy-based path

## Source entrypoints to inspect

- `networkx.drawing.layout.spring_layout`
- Fruchterman-Reingold implementation details
- energy-based optimization branch

## Why NetworkX matters for Dagua

- not a DAG-aware competitor, but a useful placement baseline
- strong on certain spacing / length-uniformity properties
- useful counterexample when layered structure should clearly win

## Local outputs to compare

- `../../eval_output/visuals/comparisons/`
- `../../eval_output/visual_audit/competitor_stepwise/`
- `../../eval_output/visual_review_session/`

## Questions to ask while reading

- Where does force-directed spacing look good despite poor DAG semantics?
- Which geometric qualities should Dagua match without giving up hierarchy?
- Which metrics flatter force layouts but are not actually aligned with our goals?
