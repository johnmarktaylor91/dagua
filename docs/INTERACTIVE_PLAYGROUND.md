# Interactive Playground

Dagua now includes a notebook-first tuning playground for building intuition
about layout parameters.

What it does:

- choose a graph from a curated complexity ladder
- or switch to a 4-graph panel view
- turn the core layout dials
- watch nodes move into new positions
- see placement metrics update live against the default baseline

This is meant to support default-settings iteration, not just demos.

## Launch in a notebook

```python
import dagua

ui = dagua.launch_playground()
ui
```

The playground requires `ipywidgets` in a Jupyter environment.

## What it is good for

- building intuition about `node_sep`, `rank_sep`, and crossing pressure
- seeing how a candidate tuning behaves across several graph families
- spotting tradeoffs between crossings, overlap, DAG consistency, and edge-length regularity
- making default-layout iteration feel tactile rather than abstract

## Modes

- `Single graph`
  - best for learning one structure deeply
- `Panel`
  - best for seeing whether a change helps one graph while hurting others

## Current scope

The first version focuses on stage-1 node placement metrics.
It intentionally keeps edge optimization off by default so the signals stay
about placement rather than downstream geometry.

Later, this same surface should grow into a stage-2 geometry playground for:

- edge routing
- edge-label placement
- cluster geometry
- text placement
