# How Dagua Works

Dagua is a hierarchical graph layout engine, but the interesting part is *how* it gets there.

Instead of hardcoding the whole layout as a fixed sequence of graph heuristics, Dagua treats layout as optimization:
- graph structure becomes tensors
- node positions become variables
- layout aesthetics become loss terms
- optimization pushes the graph into a readable arrangement

That is the core idea.

## The Pipeline

![Pipeline overview](/home/jtaylor/projects/dagua/docs/how_dagua_works/figures/pipeline_overview.png)

The broad pipeline is:

1. Build a `DaguaGraph`
2. Measure node sizes from labels and styles
3. Assign layers consistent with graph direction
4. Coarsen very large graphs into smaller hierarchy levels
5. Optimize node positions
6. Route edges
7. Place edge labels
8. Render or export

For the smallest graphs, Dagua can optimize directly. For large graphs, it uses a multilevel path.

## Why Optimization?

The payoff of the optimization framing is that a lot of “layout rules” become one system:
- DAG ordering
- repulsion
- edge attraction
- crossing reduction
- overlap avoidance
- cluster compactness and separation
- edge straightness
- length consistency
- pins and alignments

This makes the system unusually composable. A “visual preference” is often just another term in the objective or another post-step correction.

## Multilevel Layout

![Multilevel hierarchy](/home/jtaylor/projects/dagua/docs/how_dagua_works/figures/multilevel_hierarchy.png)

Large graphs are not optimized in full detail immediately.

Instead, Dagua:
- assigns layers
- compresses nodes within layers into coarse hierarchy levels
- optimizes the smaller coarse graph first
- prolongs that solution back to finer levels
- refines each finer level in turn

This is what makes the large-scale path viable. The optimizer first solves the broad composition, then adds detail.

## Constraints Are Visual Ideas

One of the nicest things about Dagua is that visual ideas are visible in the optimization story.

Pins are a good example:
- some nodes are fixed
- the rest of the graph must settle around them

![Pinning constraint animation](/home/jtaylor/projects/dagua/docs/how_dagua_works/figures/pinning_constraint.gif)

That same logic applies to alignment groups, spacing preferences, cluster containment, and routing aesthetics. The optimization animation is not just a gimmick; it exposes the actual mechanics.

## Routing Matters

Node positions are only part of the story. After layout, Dagua routes edges into a visual language:

![Routing comparison](/home/jtaylor/projects/dagua/docs/how_dagua_works/figures/routing_comparison.png)

Different routing modes emphasize different qualities:
- straight: direct and spare
- bezier: smoother flow
- orthogonal: more diagrammatic and rectilinear

These are not separate tools. They are part of the same pipeline.

## What Makes Dagua Distinct

The distinctive combination is:
- hierarchical DAG-aware layout
- pure Python + PyTorch workflow
- optimization-first architecture
- GPU-friendly scaling path
- cinematic exports and large-graph storytelling

Graphviz, ELK, and dagre are useful comparisons, but Dagua’s design center is different. It is trying to make graph layout feel like a modern optimization and visualization system, not just an old diagram engine with a Python wrapper.

## Where To Go Next

If you want the full reference:
- [docs/glossary/dagua_glossary.pdf](/home/jtaylor/projects/dagua/docs/glossary/dagua_glossary.pdf)

If you want hands-on usage:
- [docs/tutorial_walkthrough.ipynb](/home/jtaylor/projects/dagua/docs/tutorial_walkthrough.ipynb)

If you want polished examples:
- [docs/gallery/README.md](/home/jtaylor/projects/dagua/docs/gallery/README.md)

If you want the future video version:
- [docs/video/README.md](/home/jtaylor/projects/dagua/docs/video/README.md)
