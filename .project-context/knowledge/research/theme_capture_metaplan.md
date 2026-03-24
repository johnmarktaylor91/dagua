# Theme Capture Metaplan

Date: 2026-03-22

## Goal

Capture every visual feature from every major graph viz tool, and package
every recognizable aesthetic as a named dagua theme. Two workstreams:

1. **Feature union**: add every shape, edge style, arrowhead, etc. that
   exists in any tool but is missing from dagua
2. **Theme packaging**: create named themes for each tool's default look

## What Dagua Already Has

### Node Shapes (13)
rect, roundrect, ellipse, circle, diamond, triangle, hexagon, pentagon,
octagon, star, cylinder, parallelogram, trapezoid

### Arrowheads (17 primitives + compound syntax)
normal, diamond, dot, box, simple, fancy, wedge, vee, tee/bar, crow,
curve, icurve, bracket, open, circle, inv, none
Plus Graphviz-style compound specs (o/l/r modifiers, up to 4 per spec)

### Edge Routing (3)
bezier, straight, ortho

### Edge Line Styles (3)
solid, dashed, dotted

### Themes (6)
default, dark, minimal, torchlens, graphviz, graphviz_strict

---

## Gap Analysis: Features Missing From Dagua

### Node Shapes to Add

| Shape | Present in | Priority |
|-------|-----------|----------|
| cloud | Mermaid, Draw.io | MED |
| house / invhouse | Graphviz | LOW |
| egg | Graphviz | LOW |
| tab / note / folder | Graphviz | MED |
| box3d | Graphviz, yEd | MED |
| component | Graphviz | LOW |
| arrow shape (fat arrow) | Graphviz, yEd, Draw.io | LOW |
| tag | Cytoscape | LOW |
| barrel | Cytoscape | LOW |
| vee (node shape) | Cytoscape | LOW |
| cut-rectangle | Cytoscape | LOW |
| concave-hexagon | Cytoscape | LOW |
| round-triangle | Cytoscape | LOW |
| round-diamond | Cytoscape | LOW |
| round-pentagon | Cytoscape | LOW |
| round-hexagon | Cytoscape | LOW |
| round-heptagon | Cytoscape | LOW |
| round-octagon | Cytoscape | LOW |
| heptagon | Cytoscape | LOW |
| double-circle | Graphviz, Mermaid | MED |
| stadium | Mermaid | MED |
| subroutine | Mermaid | LOW |
| document | Mermaid, Draw.io | MED |
| card (notched rect) | Mermaid | LOW |
| bolt (lightning) | Mermaid | LOW |
| delay (half-rounded) | Mermaid | LOW |
| image/icon node | All tools | HIGH |

**Summary**: 27 missing shapes. HIGH: image nodes. MED: 7 shapes
(cloud, tab, box3d, double-circle, stadium, document, note).
LOW: 19 niche shapes.

### Arrowheads to Add

| Arrow | Present in | Priority |
|-------|-----------|----------|
| chevron | Cytoscape | LOW |
| triangle-tee | Cytoscape | MED |
| triangle-cross | Cytoscape | LOW |
| triangle-backcurve | Cytoscape | LOW |
| circle-triangle | Cytoscape | LOW |
| concave / convex | yEd | LOW |
| crows_foot_one_mandatory | yEd | MED (ER diagrams) |
| crows_foot_many_mandatory | yEd | MED (ER diagrams) |
| crows_foot_many_optional | yEd | MED (ER diagrams) |
| crows_foot_one | yEd | MED (ER diagrams) |
| crows_foot_optional | yEd | MED (ER diagrams) |
| halfCircle | Draw.io | LOW |
| async | Draw.io | LOW |

**Summary**: 13 missing arrows. MED: 6 (crow's foot ER set + triangle-tee).

### Edge Routing to Add

| Style | Present in | Priority |
|-------|-----------|----------|
| taxi (right-angle with corners) | Cytoscape, Draw.io | HIGH |
| round-taxi (ortho + rounded) | Cytoscape | MED |
| polyline (user-defined bends) | Graphviz, D3, Cytoscape, yEd | MED |
| tapered edges | Graphviz, Cytoscape | LOW |
| edge bundling | D3, Gephi | LOW |

**Summary**: taxi routing is the biggest gap (popular in architecture diagrams).

### Edge Line Styles to Add

| Style | Present in | Priority |
|-------|-----------|----------|
| bold / thick | Graphviz, Mermaid | MED |
| dashed-dotted | yEd | LOW |
| custom dash pattern | already supported (stroke_dash_pattern) | -- |

### Visual Effects to Add

| Effect | Present in | Priority |
|--------|-----------|----------|
| Gradient fills (linear) | Graphviz, Cytoscape, yEd, Draw.io | HIGH (stub exists) |
| Gradient fills (radial) | Graphviz, Cytoscape | MED (stub exists) |
| Drop shadows | Cytoscape, yEd, Draw.io | already supported |
| Sketch/hand-drawn mode | Draw.io | LOW (fun but niche) |
| Glass effect | Draw.io | LOW |
| Edge jump styles (arc/gap) | Draw.io | LOW |
| Striped/wedged fills | Graphviz, Cytoscape | LOW |
| Pie chart node fills | Cytoscape | LOW |
| Node outline (double border) | Cytoscape | LOW |
| Border position (inside/outside) | Cytoscape | LOW |
| Text outline | already supported | -- |
| Text background | render layer exists, not in styles | MED |
| Edge gradient | Cytoscape | LOW |
| Line cap (butt/round/square) | Cytoscape, D3 | LOW |
| Line join (miter/bevel/round) | Cytoscape, D3 | LOW |
| Image/icon backgrounds | Cytoscape, yEd, Draw.io | MED |

**Summary**: Gradient fills are the biggest gap (stub exists, just needs
rendering). Text background exposure is easy. Sketch mode is fun but low
priority.

### Label Features to Add

| Feature | Present in | Priority |
|---------|-----------|----------|
| External labels (xlabel) | Graphviz | MED |
| Head/tail edge labels | Graphviz | MED |
| Text wrapping | Cytoscape | MED |
| Text ellipsis overflow | Cytoscape | LOW |
| Text transform (upper/lower) | Cytoscape | LOW |
| Text rotation | Cytoscape | LOW |
| HTML-like table labels | Graphviz | LOW (complex) |
| Label decoration line | Graphviz (decorate) | LOW |

---

## Themes to Create

### Tier 1 (distinctive, widely recognized)

**1. mermaid** -- The documentation diagram look
- Rounded rectangles with pastel fills (#f9f, #bbf, #f96, etc.)
- Thin borders, subtle shadows optional
- Sans-serif (Trebuchet MS typical)
- Dagre-style top-down layout
- Straight or polyline edges, standard triangle arrows
- Subgraph boxes with grey background + labels
- Reference: `mmdc` CLI renders

**2. d3** -- The modern web/Observable look
- Circular nodes, no borders
- Categorical color palette (d3.schemeCategory10)
- Small nodes, lots of whitespace
- Straight edges, no arrows by default
- Sans-serif (system font)
- Labels outside nodes or on hover
- Organic force-directed placement
- Reference: Node.js + jsdom SVG generation

**3. cytoscape** -- The academic/bioinformatics look
- Elliptical nodes with thin borders
- Pale fills, dark borders
- Bezier curve edges
- Small filled triangle arrows
- Labels inside nodes, smaller font
- Tight spacing, dense layouts
- Reference: cytosnap renders

### Tier 2 (recognizable, worth capturing)

**4. gephi** -- The network science publication look
- Circular nodes sized by degree/centrality
- Community-colored (modularity partition)
- Dark or light backgrounds
- Thin curved edges, low opacity
- Node labels with outline text
- Dense, organic placement
- Reference: Gephi Toolkit JAR renders (requires Java)

**5. obsidian** -- The knowledge graph / constellation look
- Dark background (#1e1e1e)
- Small circular dot nodes, no labels (or faded)
- Lavender/purple accent (#7f6df2)
- Thin pale edges, no arrows
- Very organic, no structure
- Reference: replicate style (no tool needed)

**6. yed** -- The engineering diagram look
- Clean rectangles with gradient fills
- Orthogonal edge routing
- Precise grid-aligned spacing
- Professional, structured aesthetic
- Drop shadows, rounded corners
- Reference: replicate style (tool blocked for automation)

### Bonus (no external tool needed)

**7. drawio** -- The whiteboard diagram look
- Blue rounded rectangles with white fill
- Orthogonal edge routing with rounded bends
- Standard triangle arrows
- Optional drop shadows
- Grid-aligned, clean spacing
- Reference: replicate style

---

## Tool Installation Requirements

| Tool | Install Command | Deps | Size |
|------|----------------|------|------|
| Mermaid CLI | `npm install -g @mermaid-js/mermaid-cli` | Node.js, auto-Puppeteer | ~400MB (Chromium) |
| D3 + jsdom | `npm install d3 jsdom` | Node.js | ~5MB |
| Cytoscape + cytosnap | `npm install cytoscape cytosnap` | Node.js, Puppeteer | ~400MB |
| Gephi Toolkit | Download JAR from gephi.org | Java 11+ | ~30MB |
| yEd | -- | -- | BLOCKED (license) |
| Draw.io | -- | -- | Replicate style only |
| Obsidian | -- | -- | Replicate style only |

## Sprint Execution Order

### Pre-sprint: Feature gaps (one batch)
Add missing HIGH/MED features before any theme work:
- Implement gradient rendering (stubs exist)
- Add taxi edge routing
- Add image/icon nodes
- Add double-circle, cloud, stadium, tab, document, box3d shapes
- Expose text background in styles
- Add crow's foot ER arrowhead set
- Add head/tail edge labels, external labels

### Sprint 1: Mermaid
- **1A**: Install mmdc, build comparison script, capture all Mermaid
  shapes/colors/edge styles, verify dagua can reproduce each one
- **1B**: Build `mermaid` theme (strict + improved). Side-by-side gallery.
  Document departures.

### Sprint 2: D3-force
- **2A**: Install d3+jsdom, build comparison script, define canonical
  "D3 Observable" defaults, capture the minimal aesthetic
- **2B**: Build `d3` theme. Side-by-side gallery.

### Sprint 3: Cytoscape.js
- **3A**: Install cytoscape+cytosnap, build comparison script, capture
  all Cytoscape shapes/edges/arrows
- **3B**: Build `cytoscape` theme. Side-by-side gallery.

### Sprint 4: Gephi
- **4A**: Install Java + Gephi Toolkit JAR, build comparison script,
  capture the publication render style
- **4B**: Build `gephi` theme. Side-by-side gallery.

### Sprint 5: Replicate-only themes (obsidian, yed, drawio)
- No external tool needed. Build themes from documentation/screenshots.
- `obsidian`, `yed`, `drawio` themes.

### Post-sprints
- Add fcose (Cytoscape) and YifanHu (Gephi) as layout competitors (separate from themes)
- Add remaining LOW priority shapes/arrows as needed

## Key Principle

Every sprint follows the Graphviz recipe:
1. Three-way comparison script (reference vs dagua-strict vs dagua-improved)
2. 10 showcase graphs (simple -> complex)
3. Dual-critic calibration rounds
4. Departure documentation
5. Frozen baselines between rounds
