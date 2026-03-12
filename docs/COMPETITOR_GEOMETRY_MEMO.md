# Competitor Geometry And Aesthetic Memo

This memo is for Dagua iteration work on everything downstream of node placement:

- text sizing and placement
- edge routing and edge-label placement
- cluster / compound boundary placement
- default visual language: fonts, widths, padding, arrows, color conservatism

The goal is not to imitate competitors blindly. The goal is to understand:

1. what each competitor actually computes
2. what its documentation claims
3. what the user actually sees
4. which parts are worth stealing or improving with Dagua's differentiable pipeline

## Stage 0: Criteria Inventory

This memo now serves a second purpose:

- an explicit inventory of the criteria competitors optimize or encode
- a starting point for Dagua's own measurable geometry criteria

Important:

- this list is still not “all satisfiable at once”
- many criteria conflict
- stage 0 is not to pretend they can all be maxed simultaneously
- stage 0 is to name them explicitly so we stop missing whole classes of geometry

The practical grouping is:

1. node-placement criteria
2. downstream geometry criteria
3. paint / aesthetic-default criteria

### Competitor-Derived Criteria Checklist

The following criteria are clearly present in one or more competitors, either as
explicit objectives, encoded geometry rules, or public configuration semantics.

#### Node placement

- DAG direction consistency
- edge crossing reduction
- edge length regularity
- same-rank ordering quality
- rank / layer spacing
- node-node spacing
- compound / cluster-aware separation
- port-aware attachment orientation
- long-edge / dummy-node handling
- aspect / compactness pressure

#### Text geometry

- node label participation in sizing
- edge label participation in routing / ranking
- edge label side selection
- edge label source / center / target anchoring
- edge label distance from edge
- edge label rotation policy
- cluster label location policy
- cluster label alignment policy
- external label handling
- text bbox / backing treatment for legibility

#### Edge geometry

- routing mode selection
  - spline / polyline / orthogonal / curved
- bend-point count and placement
- arrowhead size / width
- edge-edge separation
- edge-node clearance
- edge-label clearance
- head / tail semantic labeling zones
- port-side attachment consistency

#### Cluster / compound geometry

- sibling cluster separation
- parent / child containment
- cluster padding
- cluster label clearance
- compound border reconstruction
- border-anchor stability across ranks
- foreign-edge intrusion through cluster interiors

#### Aesthetic defaults / restraint

- font family choice
- font size hierarchy
- stroke width hierarchy
- arrow size restraint
- label density / omission defaults
- color conservatism vs semantic coloring
- fill opacity restraint
- whitespace discipline

### What Competitors Explicitly Cover

| Criterion family | Graphviz | ELK | dagre | NetworkX | igraph |
| --- | --- | --- | --- | --- | --- |
| Node placement / ranking | Strong | Strong | Strong | Moderate | Moderate |
| Label-aware geometry | Strong | Strong | Moderate | Weak | Moderate |
| Cluster / compound geometry | Strong | Strong | Moderate | None | None |
| Port-aware routing | Moderate | Strong | Weak-moderate | None | None |
| Edge-label policy richness | Strong | Strong | Moderate | Weak | Weak-moderate |
| Conservative visual defaults | Strong | External renderer dependent | External renderer dependent | Weak | Backend dependent |

### Dagua-Specific Criteria We Should Add Beyond Competitors

Competitors tell us a lot, but they do not exhaust what we should care about.

These are the extra criteria Dagua should consider because a differentiable
pipeline can reason about them more continuously than classic engines do.

#### Node placement additions

- cluster-overlap penalty at the box level, not just node level
- cluster interleaving penalty
  - discourage nodes from sibling clusters weaving through one another
- backbone readability penalty
  - preserve a strong main flow when one exists
- skip-edge span readability
  - long skips should stay legible without crushing local spacing

#### Downstream geometry additions

- text-node, text-edge, and text-cluster collision penalties in one system
- “stay near preferred anchor” penalties instead of fixed discrete offsets
- cluster box pathology penalties
  - too thin
  - too flat
  - too bloated
- label rhythm / offset smoothness
  - avoid erratic label jumps on nearby edges
- foreign-edge intrusion through unrelated cluster interiors
- cluster sibling overlap and parent-child margin as measurable quantities

#### Style / information-density additions

- default-text burden
  - penalize turning on too much always-visible text by default
- hierarchy signal-to-noise
  - color, stroke, and fill should communicate structure without muddying it
- semantic emphasis discipline
  - default emphasis should track graph semantics, not arbitrary decoration

### Recommended Stage-0 Output Surfaces

Before pushing stage 1 and stage 2 optimization harder, Dagua should maintain an
explicit criteria ledger with:

- which criteria are already measured
- which criteria are partly measured
- which criteria are currently only judged visually
- which criteria conflict and therefore need Pareto treatment rather than one scalar

At minimum, the next pass should track:

- crossings
- DAG consistency
- node overlaps
- cluster sibling overlap
- parent / child containment margin
- edge-label collision count
- edge-node clearance
- cluster-edge intrusion count
- edge length CV
- runtime

This is the real starting point for the next optimization stages.

## Audit Scope

Audited competitors:

- Graphviz `dot`
- Graphviz `sfdp` / `neato` / `fdp` where relevant
- ELK layered via `elkjs`
- dagre
- NetworkX drawing + layout
- igraph layout + plotting

Audit inputs:

- literal local code
- official docs
- actual outputs from local probes
- benchmark comparison renders in `eval_output/visuals/comparisons/`

Local versions inspected:

- Graphviz `8.0.3`
- `elkjs` `0.11.1`
- `dagre` `0.8.5`
- `networkx` `3.6.1`
- `igraph` `1.0.0`

Visual examples inspected:

- `eval_output/visuals/comparisons/residual_block_comparison.png`
- `eval_output/visuals/comparisons/nested_shallow_enc_dec_comparison.png`
- the rest of the comparison set under `eval_output/visuals/comparisons/`
- local Graphviz native SVG probe for a clustered residual-style graph

## The Crucial Distinction

Do not compare these packages as if they all own the same amount of geometry.

- Graphviz and ELK are integrated geometry systems. They compute a lot more than node centers.
- dagre computes meaningful edge and compound geometry, but not a complete styled visual language.
- NetworkX and igraph are mostly position engines plus a host plotting layer.

That means there are really two comparison axes:

1. layout / geometry intelligence
2. rendering defaults / aesthetic restraint

If we do not separate these, we will copy the wrong lessons.

## What Our Benchmark Currently Preserves Vs Discards

This matters a lot.

### Graphviz wrapper

`dagua/eval/competitors/graphviz_competitor.py` calls `dagua.graphviz_utils.layout_with_graphviz(...)`.

What Graphviz itself computes:

- node positions
- edge control points
- edge label anchor positions
- cluster labels
- cluster bounds

What our benchmark wrapper currently keeps:

- node positions only

What it discards:

- edge routes
- edge-label geometry
- cluster geometry
- Graphviz's own renderer defaults

### ELK wrapper

`dagua/eval/competitors/elk_competitor.py` builds a minimal ELK graph with fixed node width/height and returns node `x,y`.

What ELK itself can compute:

- node positions
- edge sections / bend points
- edge label positions
- port-aware attachment geometry
- compound layout geometry

What our wrapper keeps:

- node positions only

What it discards:

- returned edge sections
- labels
- ports
- compound structure

### dagre wrapper

`dagua/eval/competitors/dagre_competitor.py` returns node centers only.

What dagre itself computes:

- node positions
- edge points
- edge-label coordinates
- compound node geometry

What our wrapper keeps:

- node positions only

### NetworkX / igraph wrappers

These are honest position-only adapters already. Their rendering defaults live elsewhere.

## Quick Comparison Table

| Engine | Text geometry | Edge geometry | Cluster / compound geometry | Built-in visual defaults |
| --- | --- | --- | --- | --- |
| Graphviz `dot` | Strong | Strong | Strong | Strong and conservative |
| ELK layered | Strong if label sizes are supplied | Strong | Strong | Minimal; renderer is external |
| dagre | Moderate | Moderate | Moderate | None in core library |
| NetworkX | Weak in layout engine; handled by matplotlib drawing | Weak in layout engine; handled by matplotlib drawing | None | Basic matplotlib defaults |
| igraph | Moderate plotting controls, not solver-driven | Moderate plotting controls | None native | Cairo / matplotlib backend defaults |

## Graphviz

### Literal code / output facts

From local engine output:

- native SVG defaults are `font-family="Times,serif"` and `font-size="14.00"`
- node outlines and cluster outlines default to black
- node stroke weight is effectively the default pen width
- edge arrows are black filled polygons
- default node shape in a plain DOT probe is ellipse
- cluster boundaries are plain polygons with no fill unless asked

From local JSON output:

- clusters have labels and label positions via `lp`
- edges have spline geometry in `pos`
- edge labels have anchor positions via `lp`

From Dagua's wrapper and probes:

- we currently throw away all of the above except node centers

### Official docs / documented strategy

Relevant attrs:

- `fontname`
- `fontsize`
- `splines`
- `headlabel`
- `taillabel`
- `xlabel`
- `labelloc`
- `labeljust`
- `penwidth`
- `arrowsize`
- `labeldistance`
- `labelangle`

Graphviz's design philosophy is very explicit:

- labels are layout-relevant
- clusters are graph elements with their own label alignment and location policy
- edge labels can be attached to the edge body or to head / tail areas
- “external” labels exist via `xlabel`

This is not post-hoc decoration. Graphviz reserves space and routes around it.

### What it is doing well

- extremely conservative defaults
  - black strokes
  - serif text
  - modest arrowheads
  - very little ornamental styling
- label geometry is solved as part of the graph geometry
- cluster labels feel stable because they are anchored by explicit cluster policy
- edge labels can live in distinct semantic zones:
  - center label
  - head label
  - tail label
  - external label

### Weaknesses

- defaults can feel stiff and old
- edge labels often look mechanically inserted rather than beautifully composed
- clusters can feel boxy and literal
- once graphs get dense, readability falls off because the solver is not doing richer label/edge aesthetics, just robust conventional diagram geometry

### Lessons for Dagua

Steal:

- treat edge labels as first-class geometry, not pure post-hoc text
- separate edge-label modes:
  - center
  - source-side
  - target-side
  - external
- give clusters explicit label alignment and location policy
- keep a conservative fallback visual mode

Improve with differentiable Dagua:

- do not just reserve hard rectangular label space
- optimize label placement against overlaps, edge readability, and visual rhythm
- use continuous penalties to prefer “calm” label offsets instead of Graphviz's more mechanical discrete offsets

## ELK Layered

### Literal code / output facts

From `elkjs` output on a local probe:

- edges return explicit `sections` with start/end points and bend points
- edge labels return explicit `x,y`
- the engine supports labels and ports as part of the returned layout graph

Important local probe result:

- if label sizes are not supplied, returned label width/height can remain `0`
- this is a real integration lesson: ELK wants size-aware diagram elements

### Official docs / documented strategy

ELK is explicit that it is a layout engine, not a renderer.

Relevant options include:

- font options
- edge thickness
- node padding
- node-label padding
- edge-label spacing
- edge-label placement
- edge-label side selection
- center-label placement strategy
- edge routing
- node-node spacing
- node-node spacing between layers

ELK's core idea is: all diagram objects are configurable layout objects.

That includes:

- labels
- ports
- compound structure
- padding
- routing style

### What it is doing well

- ports are first-class
- labels are first-class
- orthogonal routing and layered routing options are mature
- compound graphs are not an afterthought
- configuration surface is broad and semantically rich

### Weaknesses

- it needs more accurate size information than our current benchmark adapter provides
- because it has no native visual language, out-of-the-box appearance depends heavily on the host renderer
- label and padding results are only as good as the caller's geometry model

### Lessons for Dagua

Steal:

- treat labels, ports, padding, and routing as the same geometric system
- separate placement policy from paint style
- expose more edge-label policy knobs:
  - side selection
  - center-label strategy
  - spacing from edge

Improve with differentiable Dagua:

- instead of making labels entirely caller-sized, estimate sizes internally and refine them with continuous collision losses
- use ELK's “everything is geometry” philosophy, but solve tradeoffs with smooth objectives instead of only combinatorial policy switches

## dagre

### Literal code / output facts

From local source:

- default graph spacing:
  - `ranksep: 50`
  - `nodesep: 50`
  - `edgesep: 20`
  - `rankdir: "tb"`
- edge defaults:
  - `labeloffset: 10`
  - `labelpos: "r"`
- edge labels are made layout-relevant by:
  - halving `ranksep`
  - doubling edge `minlen`
  - padding edge width/height for side labels
- edge label proxies are injected as dummy nodes when labels have nonzero size
- compound nodes are handled via border dummy chains per rank
- final compound node width/height/x/y are reconstructed from those border nodes

This is much more sophisticated than dagre's tiny surface API suggests.

### Official docs / documented strategy

dagre presents itself as a client-side directed graph layout library.

Important practical reality:

- dagre core computes geometry
- dagre core does not define a strong house visual language
- in real products, people usually pair it with a separate renderer, often SVG / D3 / dagre-d3

### What it is doing well

- compact, pragmatic Sugiyama implementation
- labels participate in layout, not only after layout
- compound / cluster support is real, not fake
- edge points are returned directly

### Weaknesses

- the label model is still comparatively blunt:
  - side label or center label
  - fixed offset logic
- node/edge sizing quality depends entirely on caller-supplied geometry
- visual polish is outside core dagre

### Lessons for Dagua

Steal:

- edge-label proxies / label-aware ranking idea
- cluster border chains as a clean way to derive compound bounds

Improve with differentiable Dagua:

- use continuous label repulsion and edge-label anchor optimization instead of dagre's fixed offset and dummy-only treatment
- preserve the compound-border insight, but allow cluster bounds to relax under aesthetic losses instead of only being the tight box induced by rank borders

## NetworkX

### Literal code / output facts

The layout engines in our benchmark are `spring_layout` and `kamada_kawai_layout`.

The rendering defaults come from `networkx.drawing.nx_pylab`, not the layout algorithms.

Local signature / implementation facts:

- `draw_networkx(..., with_labels=True)`
- node labels default to:
  - `font_size=12`
  - `font_color='k'`
  - `font_family='sans-serif'`
- edge labels default to:
  - `label_pos=0.5`
  - `font_size=10`
  - `font_color='k'`
  - `font_family='sans-serif'`
  - `rotate=True`
  - `connectionstyle='arc3'`
- edge label default bbox is a white rounded box with white border

This is renderer behavior, not solver behavior.

### Official docs / documented strategy

NetworkX is explicit that its drawing module is basic and may be deprecated or moved in the future. The project recommends dedicated tools for serious visualization.

### What it is doing well

- simple defaults that make toy graphs readable quickly
- rotated edge labels follow the edge direction
- white edge-label bbox keeps text legible over lines

### Weaknesses

- no native cluster semantics
- no integrated label-aware layout
- defaults are generic matplotlib defaults, not graph-design defaults
- quickly becomes visually amateurish on anything nontrivial

### Lessons for Dagua

Steal:

- the white label-backing trick is useful in dense situations
- “good enough immediately” matters for the simplest use case

Do not copy:

- letting renderer convenience define the visual system

Improve with differentiable Dagua:

- keep the simplicity of a quick default, but make label placement and collision handling layout-aware instead of purely post-hoc text drawing

## igraph

### Literal code / output facts

Relevant local plotting controls include:

- `vertex_label`
- `vertex_label_dist`
- `vertex_label_angle`
- `vertex_label_size`
- `vertex_shape`
- `edge_curved`
- `edge_width`
- `edge_arrow_size`
- `edge_arrow_width`
- `autocurve`

Important Sugiyama doc facts from local docstring:

- vertices are assigned to layers
- same-layer ordering uses the barycenter heuristic
- long edges get dummy vertices
- cycles are broken approximately if needed
- layout returns dummy rows / control points for long edges

### Official docs / documented strategy

igraph is stronger as a graph-analysis and layout library than as an opinionated visual-design system.

It does expose more direct plotting controls than NetworkX:

- label distance and angle are explicit
- curved edges are explicit
- edge arrow size / width are explicit

### What it is doing well

- practical plotting controls for labels and curved edges
- explicit control over label distance / angle is useful
- Sugiyama path is conceptually honest and documented

### Weaknesses

- no native compound / cluster semantics in the same sense as Graphviz or ELK
- aesthetic defaults are backend-dependent
- geometry around text and edges is not as integrated or semantically rich as Graphviz / ELK

### Lessons for Dagua

Steal:

- explicit label distance and angle are good public APIs
- curved-edge controls are useful when they carry meaning

Improve with differentiable Dagua:

- instead of angle/distance being purely manual plotting parameters, treat them as initialization hints inside a collision-aware optimization

## Cross-Cutting Competitor Strategies

### 1. Good systems solve label geometry early

Graphviz, ELK, and dagre all make labels layout-relevant when they care about them.

Implication for Dagua:

- edge labels and cluster labels should not be purely afterthoughts
- they need either:
  - explicit layout participation
  - or a refinement phase strong enough to act like it

### 2. Conservative defaults matter more than clever styling

Graphviz's visual language is not modern, but it is disciplined:

- black strokes
- restrained widths
- stable typography
- little decorative color

Implication for Dagua:

- before inventing a bold theme, we need a neutral “serious diagram” baseline

### 3. Compound / cluster geometry is an algorithmic problem

Both Graphviz and dagre encode compounds into the layout procedure itself. ELK treats compounds as first-class layout objects.

Implication for Dagua:

- cluster bounds should not be pure rendering artifacts
- cluster label placement, padding, and sibling separation should be optimized, not just painted

### 4. Ports are a major separator

ELK's port model explains much of its clarity on structured diagrams.

Implication for Dagua:

- we should continue improving port-aware routing and label-side semantics
- port choice should eventually inform edge-label anchoring too

### 5. Several competitors look better than they are because they show less

This is especially true of Graphviz in simple examples:

- less color
- fewer text layers
- restrained edge language

Implication for Dagua:

- information density is currently one of our biggest weaknesses
- “default view vs opt-in detail” is a design problem, not just a style problem

## What Dagua Can Beat Differentiably

The competitors mostly rely on:

- dummy nodes
- hard offsets
- fixed spacing rules
- manual label modes

Dagua has a chance to do better if we use the differentiable machinery for the right downstream tasks:

### Text

- optimize edge-label offsets continuously instead of fixed side offsets
- penalize text-edge, text-node, and text-cluster collisions jointly
- allow “stay near preferred anchor” rather than “must be exactly here”

### Edges

- optimize curve control points against:
  - crossings
  - label clearance
  - angular resolution
  - bundle coherence
- use semantic edge classes to set different priors, not just different paint

### Clusters

- optimize cluster padding and sibling separation continuously
- place cluster labels with explicit composition losses
- keep nested clusters calm instead of tight-by-construction

## Recommended Near-Term Dagua Work

### Priority 1: edge labels become first-class geometry

Add a real edge-label model:

- preferred anchor:
  - source / center / target
- preferred side:
  - auto / left / right
- preferred distance
- continuous optimization around those preferences

### Priority 2: cluster labels and cluster bounds

Add explicit cluster-label policy:

- top / bottom
- left / center / right
- inside vs outside

Then optimize cluster bounds with:

- containment
- sibling clearance
- label clearance
- visual compactness

### Priority 3: conservative baseline theme

Before any ambitious style work:

- black or near-black edges
- restrained stroke widths
- stable neutral font stack
- very quiet cluster treatment
- ruthless reduction of default text density

### Priority 4: benchmark richer competitor geometry where feasible

If we want fairer comparisons on downstream geometry, our benchmark adapters should eventually preserve more than node centers for:

- Graphviz
- ELK
- dagre

Otherwise we are benchmarking placement only, which is fine, but we should say so.

## Bottom Line

The main lesson from competitors is not “copy their look.”

The main lesson is:

- Graphviz wins by solving more label and cluster geometry than it admits, then painting it conservatively.
- ELK wins by treating every diagram element as a layout object.
- dagre wins by using a compact set of pragmatic geometry hacks that are better than they look.
- NetworkX and igraph are reminders that plotting controls alone are not enough.

Dagua's opportunity is stronger than any one of these:

- keep the scalable node-placement core
- pull text, edge, and cluster geometry into the same optimization worldview
- adopt a much more restrained visual baseline

That is the path to something better than a prettier wrapper around a weaker engine.

## Reference Links

Graphviz:

- `https://graphviz.org/docs/attrs/fontname/`
- `https://graphviz.org/docs/attrs/fontsize/`
- `https://graphviz.org/docs/attrs/splines/`
- `https://graphviz.org/docs/attrs/headlabel/`
- `https://graphviz.org/docs/attrs/labeldistance/`
- `https://graphviz.org/docs/attrs/labelangle/`
- `https://graphviz.org/docs/attrs/labelloc/`
- `https://graphviz.org/docs/attrs/labeljust/`
- `https://graphviz.org/docs/attrs/penwidth/`
- `https://graphviz.org/docs/attrs/arrowsize/`
- `https://graphviz.org/docs/attrs/xlabel/`

ELK:

- `https://eclipse.dev/elk/reference.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-edgeLabels-placement.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-layered-edgeLabels-sideSelection.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-layered-edgeLabels-centerLabelPlacementStrategy.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-spacing-edgeLabel.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-nodeLabels-padding.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-spacing-nodeNode.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-layered-spacing-nodeNodeBetweenLayers.html`
- `https://eclipse.dev/elk/reference/options/org-eclipse-elk-layered-spacing-edgeNodeBetweenLayers.html`

dagre:

- `https://github.com/dagrejs/dagre/wiki#configuring-the-layout`
- local source:
  - `node_modules/dagre/lib/layout.js`
  - `node_modules/dagre/lib/add-border-segments.js`

NetworkX:

- `https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html`
- `https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_labels.html`
- `https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edge_labels.html`

igraph:

- `https://python.igraph.org/en/stable/api/igraph.Graph.html`
- local inspection:
  - `igraph.Graph.__plot__`
  - `igraph.Graph.layout_sugiyama`
