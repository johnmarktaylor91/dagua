# Graphviz

## Official docs

- Documentation hub: <https://graphviz.org/documentation/>
- `dot` layout engine: <https://graphviz.org/docs/layouts/dot/>
- Attributes reference: <https://graphviz.org/doc/info/attrs.html>
- `compound` cluster-edge support: <https://graphviz.org/docs/attrs/compound/>

## Read these topics first

- ranking / layering controls
- cluster behavior
- compound edges via `lhead` and `ltail`
- label placement attributes
- spline and arrow behavior
- node and graph spacing defaults

## Source entrypoints to inspect

- Graphviz source browser from the `dot` docs page
- `dot` ranking / mincross / spline routing code
- cluster handling and compound-edge logic

## Why Graphviz matters for Dagua

- strongest classical baseline for disciplined layered layout
- important reference for crossings, rank handling, and visual restraint
- especially relevant for:
  - cluster semantics
  - edge routing discipline
  - conservative default styling

## Local outputs to compare

- `../../eval_output/visuals/comparisons/`
- `../../eval_output/visual_audit/competitor_stepwise/`
- `../../eval_output/visual_review_session/`

## Questions to ask while reading

- Why does `dot` usually beat Dagua on crossings?
- How does it separate cluster semantics from node semantics?
- What geometry is native to Graphviz that our wrappers flatten away?
- Which defaults reflect strong design judgment vs old rendering baggage?
