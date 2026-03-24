# Visual Styling Options Survey Across Graph Visualization Tools

Date: 2026-03-22

This is an exhaustive survey of every configurable visual option across 8 major
graph visualization tools. Purpose: define the full union of features dagua
needs to support (or consciously exclude).

---

## 1. Graphviz

### Node Shapes (56 shapes)

**Polygon-based (54):**
box, polygon, ellipse, oval, circle, point, egg, triangle, plaintext, plain,
diamond, trapezium, parallelogram, house, pentagon, hexagon, septagon, octagon,
doublecircle, doubleoctagon, tripleoctagon, invtriangle, invtrapezium, invhouse,
Mdiamond, Msquare, Mcircle, rect, rectangle, square, star, none, underline,
cylinder, note, tab, folder, box3d, component, promoter, cds, terminator, utr,
primersite, restrictionsite, fivepoverhang, threepoverhang, noverhang, assembly,
signature, insulator, ribosite, rnastab, proteasesite, proteinstab, rpromoter,
rarrow, larrow, lpromoter

**Record-based (2):**
record, Mrecord (largely superseded by HTML-like labels)

**Synonyms:** rect = rectangle = box; none = plaintext

### Edge Spline Modes
- spline (default for dot -- B-spline curves)
- line (straight lines)
- curved (curved arcs)
- polyline (axis-aligned polylines)
- ortho (orthogonal routing, 90-degree segments)
- none (edges hidden but still influence layout)
- compound (fdp only -- routes around clusters)

### Arrowhead Types (11 primitives, 42+ combinations)

**Primitives:** box, crow, curve, diamond, dot, icurve, inv, none, normal, tee, vee

**Modifiers (combinatorial):**
- `l` -- clip to left half
- `r` -- clip to right half
- `o` -- open (unfilled) version

**Modifier compatibility:**
- box, diamond, inv, normal: support l/r and o
- crow, tee, vee: support l/r only
- dot: supports o only
- curve, icurve: partial
- none: no modifiers

Multiple shapes can be concatenated (e.g., `invdot`, `oldiamond`) yielding
42 single-arrow shapes and far more multi-arrow combinations.

### Label Options
- `label` -- main text
- `xlabel` -- external label (positioned outside node/edge)
- `headlabel` / `taillabel` -- text near edge head/tail
- `labelloc` -- vertical placement (t/c/b)
- `labeljust` -- horizontal justification (l/c/r)
- `labelangle` -- angle for head/tail labels
- `labeldistance` -- scaling factor for label distance
- `labelfloat` -- less constrained positioning
- `forcelabels` -- force placement even if overlapping
- `decorate` -- connect edge label to edge with line

**HTML-like labels:** Full table layout with cells, BGCOLOR, BORDER,
CELLPADDING, CELLSPACING, COLSPAN, ROWSPAN, PORT, ALIGN, VALIGN, WIDTH,
HEIGHT, FIXEDSIZE, HREF, TITLE, TOOLTIP, IMG SRC. Tables support
ROUNDED and RADIAL styles. Cells support RADIAL.

### Font/Text
- fontname (default: Times-Roman)
- fontsize (default: 14pt)
- fontcolor (default: black)
- labelfontname, labelfontsize, labelfontcolor (for edge head/tail labels)
- fontpath (bitmap font directory)

### Color/Fill
- color (drawing color, supports colorList for gradient)
- fillcolor (background, supports gradient via colorList `color1:color2`)
- bgcolor (canvas/cluster background, supports gradient)
- pencolor (cluster bounding box)
- fontcolor
- colorscheme: X11 (default, ~750 names), SVG, Brewer (categorical palettes)
- Gradient: linear (default) or radial (via style=radial)
- gradientangle (direction in degrees)

### Style Options
**Nodes:** filled, striped, wedged, diagonals, rounded, dashed, dotted,
solid, bold, invis, radial

**Edges:** dashed, dotted, solid, bold, invis, tapered

**Clusters:** filled, striped, rounded, dashed, dotted, solid, bold, invis, radial

### Other Visual Attributes
- penwidth (line thickness in points)
- peripheries (number of boundary lines -- e.g., doublecircle = 2)
- distortion, skew (polygon warping)
- regular (force regular polygon)
- margin, pad
- arrowsize (multiplicative scale factor)
- dir (edge direction: forward, back, both, none)
- _background (arbitrary xdot drawing commands)

### Special Features
- Compound subgraph clusters with independent styling
- Record nodes with port-based connections
- HTML-like label tables with full cell styling
- Image nodes (shapefile attribute)
- Transparent colors via alpha channel (#RRGGBBAA)

---

## 2. Mermaid

### Node Shapes (30+ shapes)

**Classic syntax (14):**
rectangle `[text]`, rounded rectangle `(text)`, stadium `([text])`,
subroutine `[[text]]`, cylinder `[(text)]`, circle `((text))`,
asymmetric `>text]`, diamond `{text}`, hexagon `{{text}}`,
parallelogram `[/text/]`, parallelogram-alt `[\text\]`,
trapezoid `[/text\]`, trapezoid-alt `[\text/]`, double circle `(((text)))`

**Extended shapes (v11.3.0+, via `@{ shape: name }`):**
bang, card (notched rectangle), cloud, collate (hourglass), bolt (lightning),
braces (curly), lean-r, lean-l, cyl (database), delay (half-rounded rect),
h-cyl (horizontal cylinder), lin-cyl (lined cylinder), display (curved
trapezoid), div-rect (divided rectangle), doc (document), flag (paper tape),
extract (triangle), fork (filled rectangle), win-pane (window/internal storage),
junction (filled circle), lin-doc (lined document), lin-rect (lined rectangle),
notch-pent (loop limit), manual-file (flipped triangle), manual-input (sloped
rectangle), odd (trapezoid variant), stacked-doc (multi-document),
stacked-rect (multi-rectangle)

**Special shapes:** icon, image

### Edge/Link Types
- Solid arrow: `-->`
- Open link (no arrow): `---`
- Dotted arrow: `-.->`
- Dotted open: `-.-`
- Thick arrow: `==>`
- Thick open: `===`
- Invisible: `~~~`
- Circle edge: `--o`
- Cross edge: `--x`
- Bidirectional: `<-->`
- Text on links: `-->|text|` or `-- text -->`
- Edge IDs and animation (v11.10.0+)
- Link length via extra dashes: `----` spans more ranks

### Arrowhead Types
- Standard arrow (triangle)
- Circle (`o`)
- Cross (`x`)
- No arrow (open link)

### Label Options
- Text on links (inline)
- Node text (inside shape)
- Subgraph titles

### Color/Theme
**5 built-in themes:** default, neutral, dark, forest, base

**Theme variables (base theme only):**
Core: darkMode, background, fontFamily, fontSize, primaryColor,
primaryTextColor, primaryBorderColor, secondaryColor, secondaryBorderColor,
secondaryTextColor, tertiaryColor, tertiaryBorderColor, tertiaryTextColor

Notes: noteBkgColor, noteTextColor, noteBorderColor

Diagram: lineColor, textColor, mainBkg, errorBkgColor, errorTextColor

Flowchart: nodeBorder, clusterBkg, clusterBorder, defaultLinkColor,
titleColor, edgeLabelBackground, nodeTextColor

Sequence: actorBkg, actorBorder, actorTextColor, actorLineColor,
signalColor, signalTextColor, labelBoxBkgColor, labelBoxBorderColor,
labelTextColor, loopTextColor, activationBorderColor, activationBkgColor,
sequenceNumberColor

Pie: pie1-pie12, pieTitleTextSize, pieTitleTextColor, pieSectionTextSize,
pieSectionTextColor, pieLegendTextSize, pieLegendTextColor, pieStrokeColor,
pieStrokeWidth, pieOuterStrokeWidth, pieOuterStrokeColor, pieOpacity

### Special Features
- CSS class targeting via `:::className`
- Subgraph nesting
- Direction control (TB, BT, LR, RL)
- Click events and links
- Edge animation

---

## 3. D3.js (d3-force typical usage)

D3 is a low-level SVG/Canvas library -- it has no built-in "graph" component.
Everything is programmable. These are the commonly-used primitives.

### Node Representations
**Built-in d3.symbol types (14):**

Fill symbols (7): symbolCircle, symbolCross, symbolDiamond, symbolSquare,
symbolStar, symbolTriangle, symbolWye

Stroke symbols (7): symbolAsterisk, symbolCircle, symbolDiamond2,
symbolPlus, symbolSquare2, symbolTimes, symbolTriangle2

**SVG primitives commonly used as nodes:**
- `<circle>` -- most common
- `<rect>` -- rectangles/squares
- `<ellipse>` -- ellipses
- `<polygon>` -- arbitrary polygons
- `<path>` -- any shape via SVG path commands
- `<image>` -- raster/SVG images
- `<foreignObject>` -- embed HTML (labels, rich text)
- `<g>` -- composite groups (icon + label + badge)

### Edge Representations
**SVG elements:**
- `<line>` -- straight segments
- `<path>` -- curves, arcs, any shape

**Curve interpolation (20 factories):**
curveBasis, curveBasisClosed, curveBasisOpen, curveBumpX, curveBumpY,
curveBundle (with beta param, used for edge bundling), curveCardinal,
curveCardinalClosed, curveCardinalOpen, curveCatmullRom, curveCatmullRomClosed,
curveCatmullRomOpen, curveLinear, curveLinearClosed, curveMonotoneX,
curveMonotoneY, curveNatural, curveStep, curveStepAfter, curveStepBefore

**Configurable parameters:**
- curveBundle.beta(0-1) -- bundling tightness
- curveCardinal.tension(0-1)
- curveCatmullRom.alpha(0-1) -- 0=uniform, 0.5=centripetal, 1=chordal

### Arrowhead Types
Custom via SVG `<marker>` elements in `<defs>`:
- marker-start, marker-mid, marker-end attributes on paths
- viewBox, refX, refY, markerWidth, markerHeight, orient
- Arbitrary shapes inside marker (triangle, circle, diamond, etc.)
- No built-in arrowhead types -- all user-defined

### Label Options
- SVG `<text>` elements with x, y, dx, dy, text-anchor, dominant-baseline
- `<foreignObject>` for HTML labels with full CSS
- Force-based label placement (custom)
- Collision avoidance (custom)

### Color/Fill
Full SVG/CSS styling:
- fill, stroke, stroke-width, stroke-dasharray, stroke-dashoffset
- stroke-linecap (butt, round, square)
- stroke-linejoin (miter, round, bevel)
- opacity, fill-opacity, stroke-opacity
- CSS gradients (linearGradient, radialGradient in SVG defs)
- CSS filters (drop-shadow, blur, etc.)
- d3 color scales (sequential, diverging, categorical)

### Special Features
- Canvas rendering alternative (better perf, less styling)
- WebGL rendering (via plugins like d3-force-3d)
- Transitions/animations on any property
- Zoom and pan (d3-zoom)
- Drag interaction (d3-drag)
- Voronoi overlays for interaction
- Hierarchical edge bundling (curveBundle)

---

## 4. Cytoscape.js

### Node Shapes (27 built-in)
ellipse, triangle, round-triangle, rectangle, round-rectangle,
bottom-round-rectangle, cut-rectangle, barrel, rhomboid, right-rhomboid,
diamond, round-diamond, pentagon, round-pentagon, hexagon, round-hexagon,
concave-hexagon, heptagon, round-heptagon, octagon, round-octagon, star,
tag, round-tag, vee, polygon (custom via shape-polygon-points)

### Edge Curve Styles (9)
- haystack -- fast bundled straight edges (default)
- straight -- single straight line
- straight-triangle -- straight with triangle fill
- bezier -- bundled curves
- unbundled-bezier -- manual control points
- segments -- polyline
- round-segments -- polyline with rounded corners
- taxi -- right-angle (orthogonal) routing
- round-taxi -- orthogonal with rounded corners

### Arrowhead Shapes (12)
triangle, triangle-tee, circle-triangle, triangle-cross, triangle-backcurve,
vee, tee, square, circle, diamond, chevron, none

**Arrow positions:** source, mid-source, target, mid-target
**Arrow fill:** filled, hollow
**Arrow width:** match-line or numeric
**Arrow scale:** global multiplier

### Label Options
- label, source-label, target-label
- text-halign: left, center, right
- text-valign: top, center, bottom
- text-rotation: autorotate, none, or angle
- source-text-offset, target-text-offset
- text-margin-x/y, source/target-text-margin-x/y
- text-wrap: none, wrap, ellipsis
- text-max-width
- text-overflow-wrap: whitespace, anywhere
- text-justification: left, center, right, auto
- line-height
- text-transform: none, uppercase, lowercase
- text-outline-color/opacity/width
- text-background-color/opacity/shape (rectangle, round-rectangle, circle)
- text-background-padding
- text-border-opacity/width/style/color
- min-zoomed-font-size

### Color/Fill
**Node background:**
- background-color, background-opacity, background-blacken
- background-fill: solid, linear-gradient, radial-gradient
- background-gradient-stop-colors/positions
- background-gradient-direction: to-bottom/top/left/right and diagonals

**Node border:**
- border-width, border-color, border-opacity
- border-style: solid, dotted, dashed, double
- border-cap: butt, round, square
- border-join: miter, bevel, round
- border-dash-pattern, border-dash-offset
- border-position: center, inside, outside

**Node outline:**
- outline-width, outline-style (solid/dotted/dashed/double), outline-color,
  outline-opacity, outline-offset

**Edge line:**
- line-color, line-style: solid, dotted, dashed
- line-cap: butt, round, square
- line-opacity
- line-fill: solid, linear-gradient, radial-gradient
- line-gradient-stop-colors/positions
- line-outline-width, line-outline-color
- line-dash-pattern, line-dash-offset

### Background Images
- background-image (URL or data URI)
- background-image-opacity, background-image-smoothing
- background-image-containment: inside, over
- background-fit: none, contain, cover
- background-repeat: no-repeat, repeat-x, repeat-y, repeat
- background-position-x/y, background-offset-x/y
- background-clip: node, none
- background-width-relative-to / background-height-relative-to: inner,
  include-padding

### Special Features

**Pie chart backgrounds (16 slices):**
pie-size, pie-start-angle, pie-hole, pie-i-background-color/size/opacity

**Stripe chart backgrounds (16 stripes):**
stripe-size, stripe-direction (vertical/horizontal),
stripe-i-background-color/size/opacity

**Ghost/shadow:**
ghost (yes/no), ghost-offset-x/y, ghost-opacity

**Overlay/underlay:**
overlay-color/padding/opacity/shape (round-rectangle, ellipse)
underlay-color/padding/opacity/shape

**Compound nodes:**
compound-sizing-wrt-labels (include/exclude)
min-width/height with bias controls
padding, padding-relative-to

**Visibility/layering:**
display (element/none), visibility (visible/hidden), opacity
z-index, z-compound-depth (bottom/orphan/auto/top)

**Animation:**
transition-property/duration/delay/timing-function
(29 easing functions including spring, cubic-bezier, and all standard easings)

---

## 5. Gephi

### Node Rendering
**Shapes:**
- Circle/disk (default, "Disk 2D")
- Sphere ("Sphere 3D" mode)
- Square, triangle, diamond (with plugins)
- Arbitrary polygon (via PolygonNodes plugin -- N-sided)
- Custom images

**Node properties:**
- Size (data-driven or manual)
- Color (data-driven via Ranking or Partition)
- Border width (NODE_BORDER_WIDTH)
- Border color (node color or custom)
- Opacity (0-100)
- Per-node opacity

### Edge Rendering
- Straight lines (default)
- Curved edges (toggle)
- Edge bundling (Force-Directed Edge Bundling plugin)
- Thickness (data-driven or fixed)
- Color modes: source, target, mixed, original, custom
- Opacity (0-100)
- Edge radius (extra distance from node)
- Weight rescaling (min/max)
- Antialiasing (4x to 16x)

### Arrowhead Types
- Single arrow type (triangular)
- Arrow size (ARROW_SIZE)
- Direction indication only (no shape variety)

### Label Options
**Node labels:**
- Show/hide toggle
- Font family and size
- Color (node color or custom)
- Proportional size (scale with node)
- Shorten labels (max characters)
- Outline: size, opacity, color
- Background box: show/hide, color, opacity

**Edge labels:**
- Show/hide toggle
- Font family and size
- Color (edge color or custom)
- Shorten labels (max characters)
- Outline: size, opacity, color

### Color Options
- Ranking: continuous color gradient mapped to attribute
- Partition: discrete colors per category
- Custom colors per node/edge
- Background color

### Special Features
- 3D mode (Sphere rendering)
- Preview vs. Overview rendering (different quality levels)
- Plugin architecture for custom renderers
- Multiple export formats (PDF, SVG, PNG)
- Margin percentage setting
- Visibility ratio control

---

## 6. yEd

### Node Shapes (17+ built-in geometric)
rectangle, rectangle3d, roundrectangle, diamond, ellipse, fatarrow,
fatarrow2, hexagon, octagon, parallelogram, parallelogram2, star5, star6,
star8, trapezoid, trapezoid2, triangle

**Node categories:**
- Shape Nodes (geometric primitives above)
- Modern Nodes (rounded rects with gradients/shadows)
- Group Nodes (container nodes, open/closed states, tabbed folder style)
- Swimlane Nodes (1D lane containers)
- Table Nodes (2D grid containers)
- People (SVG vector person icons)
- Computer Network (hardware/device icons)
- UML (class, sequence, state, activity, component, use case, package,
  profile, communication, timing, interaction overview)
- Flowchart (standard flowchart symbols)
- BPMN (business process symbols)
- Entity Relationship (ER diagram symbols)
- Custom (import SVG, PNG, JPG as node shapes)

### Edge Types (5 routing styles)
- Polyline (straight segments with bends)
- Arc (smooth arc between nodes)
- Bezier (cubic Bezier curves)
- Quadratic Curve
- Spline (smooth spline through control points)
- Crow's Foot notation (ER diagrams)

### Edge Line Styles
- Solid (LINE_STYLE)
- Dashed (DASHED_STYLE, widths 1.0-5.0)
- Dotted (DOTTED_STYLE)
- Dashed-dotted (DASHED_DOTTED_STYLE)
- Configurable line width

### Arrow Types (20)
none, standard, white_delta, diamond, white_diamond, short, plain, concave,
convex, circle, transparent_circle, dash, skewed_dash, t_shape,
crows_foot_one_mandatory, crows_foot_many_mandatory, crows_foot_many_optional,
crows_foot_one, crows_foot_many, crows_foot_optional

### Label Options
**Placement models:**
- Internal: centered, top, bottom, left, right (inside node)
- External: Eight Pos (8 positions around node)
- Free: unrestricted placement
- Smart Free: auto-optimized free placement

**Edge label placement:**
- On edge: source, center, target
- Free positioning

**Properties:**
- Font family and size
- Font color
- Alignment and positioning
- Auto-placement during layout

### Color/Fill
- Node fill color (solid)
- Gradient fills (various directions)
- Border/line color
- Edge color
- Background color

### Special Features
- Drop shadows (drawShadow on GroupNodeStyle)
- Group node styling (tabbed folders, open/closed states)
- Swimlane columns/rows
- Properties mapper (map data attributes to visual properties)
- Custom SVG/image import as node shapes
- 18+ layout algorithms: Hierarchical, Organic, Orthogonal (Compact),
  Circular, Tree, Balloon, Radial, Compact Disk, Series-Parallel,
  Flowchart, BPMN, SBGN, Family Tree, Tree Map, Swimlane (Hierarchic),
  Swimlane (Organic), Tabular, Random, Component Arrangement

---

## 7. Draw.io (diagrams.net)

### Shape Types
**Built-in geometric shapes:**
rectangle, ellipse, rhombus, triangle, hexagon, cloud, cylinder, line,
arrow, actor, swimlane, label, connector, image

**Shape libraries (20+ categories):**
General, Advanced (auto-layout), Flowchart, UML (11 diagram types),
BPMN, Entity Relationship, Network 2025, AWS, Azure, IBM Cloud, Alibaba
Cloud, Citrix, Dynamics365, OpenStack, SAP, Signs, People, Arrows

### Edge Styles (8 routing algorithms)
- orthogonalEdgeStyle (with rounded=1 or curved=1 variants)
- segmentEdgeStyle
- elbowEdgeStyle (horizontal/vertical)
- entityRelationEdgeStyle
- isometricEdgeStyle
- loopEdgeStyle
- sideToSideEdgeStyle
- topToBottomEdgeStyle
- No style (straight line)

### Arrow Types (20)
none, classic, classicThin, block, blockThin, open, openThin, oval,
diamond, diamondThin, box, halfCircle, circle, circlePlus, cross,
baseDash, doubleBlock, dash, async, openAsync, manyOptional

**Arrow properties:**
- startArrow, endArrow (type)
- startSize, endSize (numeric)
- startFill, endFill (0=hollow, 1=filled)

### Label Options
- html (enable HTML rendering)
- whiteSpace: wrap, nowrap
- fontSize, fontFamily, fontColor
- fontStyle: bitmask (1=bold, 2=italic, 4=underline, combinable)
- align: left, center, right
- verticalAlign: top, middle, bottom
- labelPosition: left, center, right
- verticalLabelPosition: top, middle, bottom
- overflow: visible, hidden, fill, width
- spacing, spacingTop/Bottom/Left/Right
- textOpacity (0-100)
- labelBackgroundColor, labelBorderColor
- labelWidth
- textDirection: default, ltr, rtl
- horizontal (0=vertical text, 1=horizontal)

### Color/Fill
- fillColor (#RRGGBB, none, default)
- gradientColor (#RRGGBB, none)
- gradientDirection: north, south, east, west
- strokeColor (#RRGGBB, none, default)
- strokeWidth (px)
- opacity (0-100)
- fillOpacity (0-100)
- strokeOpacity (0-100)
- glass (0/1 -- glass effect)
- shadow (0/1)
- dashed (0/1)
- dashPattern (e.g., "1 3")

### Sketch/Hand-drawn Mode
- sketch (0/1)
- comic (0/1)
- fillStyle: solid, hachure, cross-hatch, dots
- fillWeight (numeric)
- hachureGap, hachureAngle
- jiggle (line wobble amount)
- curveFitting (0-1)
- simplification (0-1)

### Geometry/Layout
- rounded (0/1), arcSize (0-50%)
- aspect: variable, fixed
- direction: north, south, east, west
- flipH, flipV (0/1)
- rotation (degrees)
- perimeter: rectanglePerimeter, ellipsePerimeter, rhombusPerimeter,
  trianglePerimeter, etc.

### Connection Control
- exitX/Y, exitDx/Dy, exitPerimeter
- entryX/Y, entryDx/Dy, entryPerimeter
- portConstraint: eastwest, northsouth, perimeter, fixed
- jettySize: auto or numeric
- sourceJettySize, targetJettySize
- orthogonalLoop (0/1)
- jumpStyle: arc, gap, sharp
- jumpSize

### Container/Swimlane
- container (0/1), collapsible (0/1)
- recursiveResize (0/1)
- swimlaneFillColor
- startSize (tab height)
- childLayout: stackLayout, treeLayout, flowLayout

### Image Properties
- image (URL), imageWidth, imageHeight
- imageAlign, imageVerticalAlign
- imageAspect (0/1)

### Special Features
- Shadow with customizable color, offset, opacity, blur
- Glass effect
- Sketch/hand-drawn mode with multiple fill patterns
- Jump styles for crossing edges (arc, gap, sharp)
- Rich container/swimlane support
- Custom shape libraries (import/export)
- Multi-color shapes (Network 2025 -- separate outline, fill, background)
- Behavior flags (movable, resizable, rotatable, bendable, editable,
  deletable, cloneable, foldable, connectable)

---

## 8. Obsidian (Graph View)

Obsidian's graph view is extremely limited compared to dedicated tools.
It's a WebGL-rendered force-directed view with minimal customization.

### Node Appearance
- Size: proportional to number of incoming links (not configurable per-node)
- Node size slider (global)
- Color: via CSS classes and color groups

### Edge Appearance
- Link thickness slider (global)
- Direction arrows: toggle on/off

### CSS Color Classes
- .graph-view.color-fill (default node color)
- .graph-view.color-fill-focused (focused/active node)
- .graph-view.color-fill-tag (tag nodes)
- .graph-view.color-fill-attachment (attachment nodes)
- .graph-view.color-fill-highlight (highlighted nodes)
- .graph-view.color-fill-unresolved (unresolved link nodes)
- .graph-view.color-arrow (arrow color)
- .graph-view.color-circle (node border)
- .graph-view.color-line (edge color)
- .graph-view.color-text (label color)
- .graph-view.color-line-highlight (highlighted edge color)
- Theme-dependent: .theme-dark / .theme-light prefixes

### Force Settings
- Center force (compactness -- how circular the graph is)
- Repel force (node push-away strength)
- Link force (connection elasticity/tension)
- Link distance (line length between notes)

### Filter Options
- Search files (full search syntax)
- Tags: show/hide
- Attachments: show/hide
- Existing files only: toggle
- Orphans: show/hide unlinked notes

### Display Options
- Arrows: show/hide direction
- Text fade threshold (zoom-based label transparency)
- Node size slider
- Link thickness slider
- Animate (chronological time-lapse)

### Grouping
- Custom color groups based on search queries
- Each group gets a distinct color

### Special Features
- Local graph view (depth slider for connection layers)
- Time-lapse animation
- WebGL rendering (not SVG -- CSS can't target individual elements directly,
  only bridge colors)

---

## Cross-Tool Feature Matrix

### Node Shapes -- Union Set

| Shape | Gviz | Merm | D3 | Cyto | Gephi | yEd | Draw | Obs |
|---|---|---|---|---|---|---|---|---|
| Rectangle/box | Y | Y | Y | Y | - | Y | Y | - |
| Rounded rect | Y* | Y | Y | Y | - | Y | Y | - |
| Ellipse/oval | Y | - | Y | Y | Y | Y | Y | - |
| Circle | Y | Y | Y | Y | Y | - | - | Y** |
| Diamond/rhombus | Y | Y | Y | Y | Y* | Y | Y | - |
| Triangle | Y | Y | Y | Y | Y* | Y | Y | - |
| Inv triangle | Y | - | - | - | - | - | - | - |
| Pentagon | Y | - | - | Y | - | - | - | - |
| Hexagon | Y | Y | - | Y | - | Y | Y | - |
| Heptagon | - | - | - | Y | - | - | - | - |
| Octagon | Y | - | - | Y | - | Y | - | - |
| Star | Y | - | Y | Y | - | Y*** | - | - |
| Parallelogram | Y | Y | - | Y**** | - | Y | - | - |
| Trapezoid | Y | Y | - | - | - | Y | - | - |
| Cylinder | Y | Y | - | Y***** | - | - | Y | - |
| Cloud | - | Y | - | - | - | - | Y | - |
| House/invhouse | Y | - | - | - | - | - | - | - |
| Egg | Y | - | - | - | - | - | - | - |
| Tab/note/folder | Y | - | - | - | - | - | - | - |
| Box3D | Y | - | - | - | - | Y | - | - |
| Component | Y | - | - | - | - | - | - | - |
| Arrow shape | Y | - | - | Y | - | Y | Y | - |
| Tag | - | - | - | Y | - | - | - | - |
| Barrel | - | - | - | Y | - | - | - | - |
| Vee | - | - | - | Y | - | - | - | - |
| Cut-rectangle | - | - | - | Y | - | - | - | - |
| Concave hexagon | - | - | - | Y | - | - | - | - |
| Custom polygon | Y | - | Y | Y | Y* | Y | Y | - |
| Image/icon | Y | Y | Y | Y | Y | Y | Y | - |
| Bio shapes | Y | - | - | - | - | - | - | - |

*via plugin **circles only ***star5/6/8 ****rhomboid *****barrel shape

### Edge Curve Types -- Union Set

| Curve Style | Gviz | D3 | Cyto | yEd | Draw | Gephi |
|---|---|---|---|---|---|---|
| Straight | Y | Y | Y | Y | Y | Y |
| Spline/Bezier | Y | Y | Y | Y | - | - |
| Orthogonal | Y | - | Y | - | Y | - |
| Polyline | Y | Y | Y | Y | Y | - |
| Curved arc | Y | Y | Y | Y | Y | Y |
| Taxi/elbow | - | - | Y | - | Y | - |
| Rounded corners | - | - | Y | - | Y | - |
| Step function | - | Y | - | - | - | - |
| Bundle | - | Y | - | - | - | Y* |
| Tapered | Y | - | Y**| - | - | - |
| Quadratic | - | - | - | Y | - | - |
| Catmull-Rom | - | Y | - | - | - | - |
| Cardinal | - | Y | - | - | - | - |
| Natural cubic | - | Y | - | - | - | - |
| Monotone | - | Y | - | - | - | - |

*plugin **straight-triangle

### Arrow Shapes -- Union Set

| Arrow | Gviz | Cyto | yEd | Draw |
|---|---|---|---|---|
| Triangle/normal | Y | Y | Y | Y |
| Inverted triangle | Y | - | - | - |
| Vee/open | Y | Y | - | Y |
| Diamond | Y | Y | Y | Y |
| Dot/circle | Y | Y | Y | Y |
| Box/square | Y | Y | - | Y |
| Tee | Y | Y | Y | - |
| Crow (crow's foot) | Y | - | Y | - |
| Curve/icurve | Y | - | - | - |
| Chevron | - | Y | - | - |
| Triangle-tee | - | Y | - | - |
| Triangle-cross | - | Y | - | - |
| Triangle-backcurve | - | Y | - | - |
| Circle-triangle | - | Y | - | - |
| Concave/convex | - | - | Y | - |
| White delta | - | - | Y | - |
| Half-circle | - | - | - | Y |
| Cross | - | - | - | Y |
| Dash | - | - | Y | Y |
| Async | - | - | - | Y |
| Hollow variants | Y | Y | Y | Y |
| Half (l/r clip) | Y | - | - | - |
| None | Y | Y | Y | Y |

### Label Features -- Union Set

| Feature | Gviz | Cyto | yEd | Draw | Gephi |
|---|---|---|---|---|---|
| Inside node | Y | Y | Y | Y | Y |
| Outside node | Y | Y | Y | Y | - |
| Edge center | Y | Y | Y | Y | Y |
| Edge source/target | Y | Y | Y | - | - |
| External/xlabel | Y | - | Y | - | - |
| Rotation | - | Y | - | - | - |
| Background fill | Y* | Y | - | Y | Y |
| Border/outline | - | Y | - | Y | Y |
| Text transform | - | Y | - | - | - |
| HTML rich text | Y | - | - | Y | - |
| Wrapping | Y* | Y | - | Y | - |
| Ellipsis | - | Y | - | - | Y |
| Font per label | Y | Y | Y | Y | Y |
| Min zoom font | - | Y | - | - | - |
| Proportional size | - | - | - | - | Y |
| Label decoration | Y | - | - | - | - |

*via HTML-like labels

### Special Visual Features -- Union Set

| Feature | Gviz | D3 | Cyto | yEd | Draw | Gephi | Merm | Obs |
|---|---|---|---|---|---|---|---|---|
| Gradients | Y | Y | Y | Y | Y | - | - | - |
| Shadows | - | Y | Y | Y | Y | - | - | - |
| Glass effect | - | - | - | - | Y | - | - | - |
| Sketch/hand-drawn | - | - | - | - | Y | - | - | - |
| Pie chart nodes | - | - | Y | - | - | - | - | - |
| Stripe chart nodes | - | - | Y | - | - | - | - | - |
| Compound/group | Y | - | Y | Y | - | - | - | - |
| Swimlanes | - | - | - | Y | - | - | - | - |
| 3D rendering | Y* | Y**| - | - | - | Y | - | - |
| Image fill | Y | Y | Y | Y | Y | Y | Y | - |
| Transparent | Y | Y | Y | - | Y | Y | - | - |
| Animation | - | Y | Y | - | - | - | Y | Y |
| Edge bundling | - | Y | - | - | - | Y | - | - |
| Edge jump/cross | - | - | - | - | Y | - | - | - |
| Tapered edges | Y | - | - | - | - | - | - | - |
| Striped fill | Y | - | Y | - | - | - | - | - |
| Wedged fill | Y | - | - | - | - | - | - | - |
| Multiple borders | Y | - | Y | - | - | - | - | - |

*box3d shape **via Three.js plugins

---

## Key Takeaways for Dagua

### Must-Have (present in 4+ tools)
1. Basic shapes: rectangle, rounded-rect, ellipse, circle, diamond, triangle
2. Extended shapes: hexagon, pentagon, octagon, star, parallelogram, trapezoid, cylinder
3. Edge styles: straight, spline/bezier, orthogonal, polyline, curved arc
4. Arrow types: triangle, vee/open, diamond, circle, tee, box, none, hollow variants
5. Line styles: solid, dashed, dotted, bold/thick
6. Colors: fill, stroke, gradient (linear + radial), opacity/transparency
7. Labels: inside node, outside node, on edge (source/center/target), font control
8. Compound/group nodes
9. Image/icon nodes

### Should-Have (present in 2-3 tools, high value)
1. Additional shapes: tag, barrel, arrow-shape, cloud, inverted variants
2. Edge features: taxi/orthogonal routing, rounded corners on polylines, tapered
3. Arrow features: crow's foot, half-arrows (l/r clip), chevron, composite arrows
4. Label features: background fill, border/outline, rotation, rich text/HTML
5. Shadows and drop shadows
6. Sketch/hand-drawn mode
7. Edge bundling
8. Pie chart / data-viz node fills
9. Multiple border rings (peripheries)
10. Custom polygon shapes

### Nice-to-Have (unique to 1 tool, distinctive)
1. Bio shapes (Graphviz -- promoter, cds, terminator, etc.)
2. Glass effect (Draw.io)
3. Edge jump styles at crossings (Draw.io)
4. Striped/wedged fills (Graphviz)
5. Stripe chart backgrounds (Cytoscape.js)
6. Ghost/shadow offset effect (Cytoscape.js)
7. Hachure/cross-hatch fills (Draw.io sketch mode)
8. Per-label min-zoom-font (Cytoscape.js)
