# dagre

## Official docs

- Repository and README: <https://github.com/dagrejs/dagre>
- Wiki and configuration background: linked from the repository README

## Read these topics first

- graph configuration:
  - `ranksep`
  - `nodesep`
  - `edgesep`
  - `rankdir`
  - `acyclicer`
  - `ranker`
- node dimension requirements
- edge label handling
- client-side / JS-first tradeoffs

## Source entrypoints to inspect

- ranking
- ordering
- coordinate assignment
- label / edge midpoint handling

## Why dagre matters for Dagua

- clean, compact layered implementation
- good reference for practical web-facing defaults
- useful for understanding:
  - minimal layered pipelines
  - small/medium graph tradeoffs
  - how much quality can be extracted from a fairly lean stack

## Local outputs to compare

- `../../eval_output/visuals/comparisons/`
- `../../eval_output/visual_audit/competitor_stepwise/`
- `../../eval_output/visual_review_session/`

## Questions to ask while reading

- Why does dagre still beat Dagua on crossings in some regimes?
- What parts of dagre are essential layered structure vs implementation simplicity?
- Which defaults feel web-native and which are just sparse?
