# Graph Visualization Landscape

A neutral survey of graph layout and rendering tools, their strengths,
architecture, and where they fit in the ecosystem. This is a reference
document, not a comparison chart -- each tool serves a different audience
and makes different tradeoffs.

## Layout Engines

### Graphviz
- **Language:** C
- **License:** Open source (Eclipse Public License)
- **Interface:** CLI (`dot`, `neato`, etc.) + C library + language bindings
- **Layout algorithms:** dot (hierarchical/Sugiyama), neato (stress), fdp
  (force-directed), sfdp (scalable force-directed), circo (circular),
  twopi (radial)
- **Rendering:** Built-in. SVG, PDF, PNG, PostScript. Functional quality --
  clean but not highly customizable.
- **Clusters:** Post-layout bounding box packing. No structural nesting.
- **Scale:** Thousands of nodes (dot), tens of thousands (sfdp)
- **Strengths:** Ubiquitous. The DOT language is a de facto standard.
  Extremely well-tested across millions of real-world graphs. Zero
  dependencies. Runs everywhere.
- **Limitations:** Fixed algorithm pipeline -- no user-defined objectives.
  C codebase is difficult to extend. No GPU support. The rendering system
  hasn't evolved significantly since the early 2000s.

### ELK (Eclipse Layout Kernel)
- **Language:** Java (also available as elkjs via JS transpilation)
- **License:** Open source (Eclipse Public License)
- **Interface:** Java API, or elkjs for Node.js/browser
- **Layout algorithms:** ELK Layered (Sugiyama-style with advanced options),
  force-directed, radial, box packing, stress
- **Rendering:** None. Layout only -- returns coordinates.
- **Clusters:** First-class compound graph support. INCLUDE_CHILDREN mode
  considers cluster containment during layout, not as post-processing.
- **Scale:** Thousands of nodes
- **Strengths:** The most configurable open-source hierarchical layout.
  Modular architecture -- each layout phase is independently replaceable.
  Strong academic foundation (Kiel University). Compound node support is
  best-in-class among open-source tools.
- **Limitations:** Java ecosystem adds friction for Python/data-science
  users. No rendering -- consuming applications must provide their own.
  The elkjs transpilation has performance overhead.

### OGDF (Open Graph Drawing Framework)
- **Language:** C++
- **License:** Open source (GNU GPL)
- **Layout algorithms:** Sugiyama, stress majorization, FM^3 (fast
  multipole), Maxent-Stress, orthogonal, planarization, and many more.
  The broadest algorithm collection of any open-source library.
- **Rendering:** Minimal. Can output SVG but the focus is algorithmic.
- **Clusters:** Supported in some algorithms (compound graph layouts).
- **Scale:** Designed for research-scale experiments. Performance varies
  by algorithm.
- **Strengths:** Exhaustive algorithm coverage. Many papers in the graph
  drawing literature include OGDF implementations. The reference
  implementation for academic layout research.
- **Limitations:** C++ API is complex. Not designed for end-user
  applications. Documentation is sparse. Integration with other ecosystems
  requires significant effort.

### NetworkX
- **Language:** Python
- **License:** Open source (BSD)
- **Interface:** Python API
- **Layout algorithms:** spring (Fruchterman-Reingold), Kamada-Kawai,
  spectral, shell, planar, and others.
- **Rendering:** Delegates to matplotlib. Basic circle-and-line rendering.
- **Clusters:** No built-in cluster layout support.
- **Scale:** Hundreds to low thousands of nodes (pure Python, no GPU).
- **Strengths:** The default graph library for Python. Excellent graph
  analysis (centrality, community detection, shortest paths). Huge
  ecosystem of algorithms. Easy to get started.
- **Limitations:** Layout algorithms are basic implementations without
  advanced tuning. Rendering is functional but not publication-quality.
  Not designed as a visualization-first tool.

## Rendering-Focused Tools

### Cytoscape.js
- **Language:** JavaScript
- **License:** Open source (MIT)
- **Interface:** JS API, browser-based
- **Layout:** Built-in force-directed (fcose, cose-bilkent), also
  supports external layout plugins (ELK, dagre).
- **Rendering:** Canvas-based. Good shape support, compound nodes,
  custom styling via a CSS-like selector system.
- **Clusters:** Compound node model -- nodes can contain other nodes.
  fcose handles compound layout natively.
- **Strengths:** The most complete open-source graph visualization for
  the web. Strong in bioinformatics (originated from Cytoscape desktop).
  Interactive by default.
- **Limitations:** JavaScript ecosystem. Canvas rendering can be blurry
  at high DPI. Not designed for static publication-quality export.

### D3.js
- **Language:** JavaScript
- **License:** Open source (ISC)
- **Interface:** JS API, SVG/Canvas manipulation
- **Layout:** d3-force (force-directed simulation), d3-dag (Sugiyama).
  Low-level -- the user assembles the layout pipeline.
- **Rendering:** User-built from SVG/Canvas primitives. Maximum
  flexibility, no built-in graph rendering.
- **Clusters:** No built-in cluster support in layout or rendering.
- **Strengths:** The foundation of modern web data visualization.
  Observable (Mike Bostock's platform) showcases extraordinary graph
  visualizations built with D3. Unmatched flexibility for custom
  interactive visualizations.
- **Limitations:** Assembly required. There is no "draw my graph"
  function. Every visual element is the user's responsibility.

### Sigma.js
- **Language:** JavaScript
- **License:** Open source (MIT)
- **Interface:** JS API, WebGL-based
- **Layout:** No built-in layout. Typically paired with Graphology for
  graph data and external layout algorithms.
- **Rendering:** WebGL. Optimized for rendering large graphs (100K+
  nodes) at interactive frame rates.
- **Strengths:** Performance. When you need to render and interact with
  very large networks in the browser, Sigma is the standard choice.
- **Limitations:** Limited shape vocabulary (circles and labels). Not
  designed for diagramming or publication output. WebGL only.

## Commercial Products

### yFiles
- **Language:** Java, JavaScript (separate products)
- **License:** Commercial (proprietary, closed source)
- **Interface:** Java API / JS API. yEd is a free desktop viewer.
- **Layout:** Full suite -- hierarchical, organic, orthogonal, circular,
  radial, tree, bus routing. Arguably the most polished layout
  implementations available.
- **Rendering:** Full suite -- custom shapes, port constraints, compound
  nodes, automatic label placement with collision avoidance, interactive
  editing, print-quality export.
- **Clusters:** Native compound graph support with expand/collapse.
- **Strengths:** The most complete commercial graph visualization
  toolkit. 25+ years of development. Both layout and rendering are
  best-in-class.
- **Limitations:** Expensive (enterprise pricing). Closed source -- cannot
  inspect, modify, or learn from the implementations. Java/JS only.

### Tom Sawyer Software
- **Language:** Java, .NET, JavaScript
- **License:** Commercial
- **Interface:** API + visual designer
- **Layout:** Hierarchical, orthogonal, symmetric, circular.
- **Rendering:** Full interactive visualization with drill-down.
- **Strengths:** Enterprise integration. Used in network management,
  fraud detection, supply chain visualization.
- **Limitations:** Enterprise pricing and sales process. Not accessible
  to individual developers or researchers.

### GoJS
- **Language:** JavaScript
- **License:** Commercial (free for evaluation)
- **Interface:** JS API, browser-based
- **Layout:** TreeLayout, ForceDirectedLayout, LayeredDigraphLayout,
  CircularLayout, and others.
- **Rendering:** Canvas/SVG. Interactive diagrams with drag-and-drop,
  link routing, collapsible groups.
- **Strengths:** Excellent documentation and examples. Quick to get
  started for web-based diagram editors. Reasonable pricing for
  commercial use.
- **Limitations:** JavaScript only. Layout algorithms are competent but
  not research-grade. Closed source.

## Diagram Authoring Tools

These tools focus on the authoring experience rather than algorithmic
layout. They typically use simple layout engines (or none) and rely on
manual positioning or basic auto-layout.

- **Mermaid** -- Markdown-based diagram syntax. Uses dagre or ELK for
  layout. Widely adopted in documentation (GitHub, GitLab, Notion).
  The appeal is the text-based authoring, not the visual quality.
- **Draw.io (diagrams.net)** -- Free web-based diagram editor. Manual
  layout with optional auto-layout via built-in algorithms or ELK.
  The most widely used free diagramming tool.
- **Excalidraw** -- Hand-drawn sketch aesthetic. Manual positioning only.
  The appeal is the informal, whiteboard-like feel.
- **PlantUML** -- Text-based UML diagrams. Uses Graphviz for layout.
  Focused on software engineering diagram types.

## Research Implementations

The graph drawing research community produces layout algorithms as
paper artifacts, typically evaluated on standard benchmarks (Rome graphs,
AT&T graphs, random DAGs) using quality metrics (edge crossings, stress,
angular resolution, area).

These implementations advance the state of the art in layout quality and
scalability but are generally not packaged for end-user consumption.
They often lack rendering, documentation, and API design. Their value is
in the algorithms themselves.

Notable recent directions:
- **Stress majorization with SGD** -- gradient-based stress minimization
  using stochastic gradient descent, competitive with classical
  approaches at larger scale.
- **GNN-guided layout** -- using graph neural networks to predict good
  initial positions or to learn aesthetic objectives from human
  preference data.
- **GPU-accelerated force-directed layout** -- massively parallel
  repulsion computation for million-node graphs.
- **Differentiable graph drawing** -- expressing layout objectives as
  differentiable loss functions and optimizing with automatic
  differentiation. Enables custom, composable objectives and
  GPU acceleration through deep learning frameworks.

A rendering engine that can produce publication-quality output from any
algorithm's coordinate output serves the entire research community --
researchers get beautiful figures, and their algorithms reach a wider
audience through accessible visualization.

## Summary Table

| Tool | Layout | Rendering | Open Source | Language | GPU | Clusters |
|------|--------|-----------|-------------|----------|-----|----------|
| Graphviz | Yes | Yes | Yes | C | No | Post-hoc |
| ELK | Yes | No | Yes | Java/JS | No | Native |
| OGDF | Yes | Minimal | Yes | C++ | No | Partial |
| NetworkX | Basic | Basic | Yes | Python | No | No |
| Cytoscape.js | Yes | Yes | Yes | JS | No | Compound |
| D3.js | Partial | DIY | Yes | JS | No | No |
| Sigma.js | No | Yes (WebGL) | Yes | JS | WebGL | No |
| yFiles | Yes | Yes | No | Java/JS | No | Native |
| GoJS | Yes | Yes | No | JS | No | Groups |
| Mermaid | Via dagre/ELK | Via browser | Yes | JS | No | Via backend |
