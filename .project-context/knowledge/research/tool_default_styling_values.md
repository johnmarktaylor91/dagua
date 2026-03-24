# Exact Default Styling Values for Graph Visualization Tools

Date: 2026-03-22
Sources: Direct source code inspection via GitHub raw files

---

## 1. Mermaid (Default Theme)

Source: `packages/mermaid/src/themes/theme-default.js`, `theme-base.js`,
`diagrams/flowchart/styles.ts`, `schemas/config.schema.yaml`

### Colors
| Variable              | Value                                      |
|-----------------------|--------------------------------------------|
| primaryColor          | `#ECECFF` (lavender)                       |
| primaryTextColor      | invert of primaryColor (computed -> dark)   |
| primaryBorderColor    | mkBorder(primaryColor) -> `#9370DB`        |
| secondaryColor        | `#ffffde` (pale yellow)                    |
| tertiaryColor         | hue-shift of primaryColor by -160 degrees  |
| background            | `white`                                    |
| mainBkg               | `#ECECFF`                                  |
| secondBkg             | `#ffffde`                                  |
| textColor             | `#333`                                     |
| lineColor             | `#333333`                                  |
| border1               | `#9370DB` (medium purple)                  |
| border2               | `#aaaa33`                                  |
| nodeBorder            | `#9370DB`                                  |
| clusterBkg            | `#ffffde`                                  |
| clusterBorder         | `#aaaa33`                                  |
| edgeLabelBackground   | `rgba(232, 232, 232, 0.8)`                |
| arrowheadColor        | `#333333`                                  |
| nodeTextColor         | same as primaryTextColor                   |
| noteBkgColor          | `#fff5ad`                                  |
| noteTextColor         | `#333`                                     |

### Typography
| Property   | Value                                         |
|------------|-----------------------------------------------|
| fontFamily | `"trebuchet ms", verdana, arial, sans-serif`  |
| fontSize   | `16px`                                        |

### Flowchart Layout
| Property       | Value   |
|----------------|---------|
| nodeSpacing    | 50      |
| rankSpacing    | 50      |
| diagramPadding | 8       |
| padding        | 15      |
| curve          | `basis`  |

### Flowchart Rendering
| Property           | Value  |
|--------------------|--------|
| edge stroke-width  | `2.0px`|
| node stroke-width  | `1px`  |
| cluster stroke-width | `1px`|
| edgeLabel bg opacity | `0.5` |
| title font-size    | `18px` |
| tooltip font-size  | `12px` |

---

## 2. D3.js Observable Defaults

Source: `d3-scale-chromatic/src/categorical/category10.js`,
`d3-force/src/link.js`, `d3-force/src/manyBody.js`,
Observable `@d3/force-directed-graph` notebook

### schemeCategory10 (the canonical D3 palette)
| Index | Hex       | Description    |
|-------|-----------|----------------|
| 0     | `#1f77b4` | steel blue     |
| 1     | `#ff7f0e` | orange         |
| 2     | `#2ca02c` | green          |
| 3     | `#d62728` | red            |
| 4     | `#9467bd` | purple         |
| 5     | `#8c564b` | brown          |
| 6     | `#e377c2` | pink           |
| 7     | `#7f7f7f` | gray           |
| 8     | `#bcbd22` | olive/yellow   |
| 9     | `#17becf` | cyan           |

### Force Layout Defaults
| Property         | Value  |
|------------------|--------|
| link distance    | `30`   |
| charge strength  | `-30`  |
| link strength    | `1 / min(degree_source, degree_target)` |

### Observable ForceGraph Visual Defaults (de facto standard)
| Property            | Value          |
|---------------------|----------------|
| node radius         | `5` px         |
| node fill           | `currentColor` |
| node stroke         | `#fff`         |
| node stroke-width   | `1.5` px       |
| node stroke-opacity | `1`            |
| link stroke         | `#999`         |
| link stroke-width   | `1.5` px       |
| link stroke-opacity | `0.6`          |
| link stroke-linecap | `round`        |
| canvas width        | `640` px       |
| canvas height       | `400` px       |
| background          | transparent    |

---

## 3. Cytoscape.js Defaults

Source: `src/style/properties.mjs`

### Node Defaults
| Property           | Value                                    |
|--------------------|------------------------------------------|
| background-color   | `#999`                                   |
| background-opacity | `1`                                      |
| border-color       | `#000`                                   |
| border-width       | `0`                                      |
| border-opacity     | `1`                                      |
| width              | `30`                                     |
| height             | `30`                                     |
| shape              | `ellipse`                                |
| opacity            | `1`                                      |
| outline-color      | `#999`                                   |
| corner-radius      | `auto`                                   |
| color (text)       | `#000`                                   |
| font-size          | `16`                                     |
| font-family        | `Helvetica Neue, Helvetica, sans-serif`  |

### Edge Defaults
| Property            | Value      |
|---------------------|------------|
| line-color          | `#999`     |
| curve-style         | `haystack` |
| line-style          | `solid`    |
| line-opacity        | `1`        |
| source-arrow-shape  | `none`     |
| target-arrow-shape  | `none`     |
| arrow-color         | `#999`     |
| arrow-fill          | `filled`   |
| arrow-width         | `1`        |
| line-cap            | `butt`     |

### Common Defaults
| Property    | Value     |
|-------------|-----------|
| text-color  | `#000`    |
| text-opacity| `1`       |
| visibility  | `visible` |
| display     | `element` |
| z-index     | `0`       |

---

## 4. Gephi Defaults

Source: `modules/VisualizationImpl/.../VizConfig.java`,
`modules/PreviewPlugin/.../NodeRenderer.java`,
`modules/PreviewPlugin/.../EdgeRenderer.java`,
`modules/PreviewPlugin/.../NodeLabelRenderer.java`,
`modules/PreviewPlugin/.../ArrowRenderer.java`,
`graphstore/.../NodeImpl.java`

### Background
| Property              | Value (light)      | Value (dark)           |
|-----------------------|--------------------|------------------------|
| background-color      | `#ffffff` (white)  | `rgb(52, 55, 57)`      |

### Node Defaults
| Property       | Value                                    |
|----------------|------------------------------------------|
| color (rgba)   | `r=0, g=0, b=0, a=255` (black, via Java float default + alpha) |
| size           | `0.0f` (Java default; Gephi UI sets ~10)  |
| border-width   | `1.0f`                                    |
| border-color   | `Color.BLACK` (`#000000`)                 |
| opacity        | `100` (percent)                           |

NOTE: In practice, Gephi's generators and importers do NOT set node colors.
The GraphStore initializes nodes to black (r=0, g=0, b=0) with full alpha.
The Gephi Overview panel then renders them; the Preview panel adds the border.
Users typically see nodes colored by the "Appearance" panel's partition/ranking.

### Edge Defaults
| Property        | Value                |
|-----------------|----------------------|
| color-mode      | `MIXED` (blend source/target colors) |
| thickness       | `1.0f`               |
| opacity         | `100` (percent)      |
| curved          | `true`               |
| arc curviness   | `1.2f`               |
| use-weight      | `true`               |
| rescale-weight  | `true`               |
| weight-min      | `0.4f`               |
| weight-max      | `8.0f`               |
| scale           | `2.0f` (range 0.4-10)|

### Arrow Defaults
| Property   | Value |
|------------|-------|
| arrow-size | `3.0f`|

### Label Defaults
| Property             | Value                        |
|----------------------|------------------------------|
| font-family          | `Arial`                      |
| font-style           | `Bold`                       |
| font-size            | `32` (then scaled by 0.5x)  |
| font-color           | original node color          |
| outline-color        | `Color.WHITE` (`#ffffff`)    |
| outline-size         | `4` (scaled by fontSize/32)  |
| outline-opacity      | `40%`                        |
| show-labels          | `false` (hidden by default!) |
| proportional-sizing  | `true`                       |

### Selection Colors
| Property           | RGB                     |
|--------------------|-------------------------|
| edge-in-selected   | `rgb(32, 95, 154)`      |
| edge-out-selected  | `rgb(196, 66, 79)`      |
| edge-both-selected | `rgb(248, 215, 83)`     |

### Scale Defaults
| Property         | Value |
|------------------|-------|
| node-scale       | `1.0f`|
| edge-scale       | `2.0f`|
| node-label-scale | `0.5f`|
| edge-label-scale | `0.5f`|

---

## 5. Obsidian Graph View

Source: `obsidian-developer-docs/.../Colors.md`,
`shimmering-focus/types/obsidian-variables.css` (extracted from app.css)

### Accent Color (the signature purple)
| Variable    | Value |
|-------------|-------|
| --accent-h  | `254` |
| --accent-s  | `80%` |
| --accent-l  | `68%` |

Computed accent: `hsl(254, 80%, 68%)` -> approximately `#8b7bde` (periwinkle purple)

### Graph CSS Variables (variable indirection)
| Variable                 | Resolves to             | Light mode value | Dark mode value |
|--------------------------|-------------------------|------------------|-----------------|
| --graph-line             | --color-base-35         | `#d4d4d4`        | `#3f3f3f`       |
| --graph-node             | --text-muted            | `#5c5c5c`        | (base-60 area)  |
| --graph-node-unresolved  | --text-faint            | `#ababab`        | (base-50 area)  |
| --graph-node-focused     | --text-accent           | `hsl(254,80%,68%)` | same          |
| --graph-node-tag         | --color-green           | `#08b94e`        | varies          |
| --graph-node-attachment  | --color-yellow          | `#e0ac00`        | varies          |
| --graph-text             | --text-normal           | `#222222`        | (base-100 area) |

### Base Color Scale (light / dark)
| Variable        | Light     | Dark      |
|-----------------|-----------|-----------|
| --color-base-00 | `#ffffff` | `#1e1e1e` |
| --color-base-05 | `#fcfcfc` | `#212121` |
| --color-base-10 | `#fafafa` | `#242424` |
| --color-base-20 | `#f6f6f6` | `#262626` |
| --color-base-25 | `#e3e3e3` | `#2a2a2a` |
| --color-base-30 | `#e0e0e0` | `#363636` |
| --color-base-35 | `#d4d4d4` | `#3f3f3f` |
| --color-base-40 | `#bdbdbd` | `#555555` |
| --color-base-50 | `#ababab` | `#666666` |
| --color-base-60 | `#707070` | `#999999` |
| --color-base-70 | `#5a5a5a` | `#bababa` |
| --color-base-100| `#222222` | `#dadada` |

### Typography
| Property        | Value  |
|-----------------|--------|
| --font-text-size| `16px` |
| --line-height-normal | `1.5` |
| Font family     | System default (user-configurable, no hardcoded default) |

NOTE: Obsidian's graph view is a WebGL canvas. The CSS variables control
the colors but the actual rendering sizes (node radius, line width) are
determined by the graph view's internal logic, not CSS. The graph view
does not document its node/line size defaults.

---

## 6. yEd / yFiles Defaults

Source: `yfiles-for-html-demos/.../ThemeVariantsDemo.ts`,
`yfiles-for-html-demos/.../demo-colors.ts`

### yEd Desktop Defaults (not directly sourced -- yEd is closed-source)
yEd is a desktop application by yWorks. Its defaults come from the yFiles
library. The closest we can get is the yFiles demo defaults:

### yFiles Library Defaults
| Property        | Value                |
|-----------------|----------------------|
| node fill       | `#CCCCCC` (light gray)|
| node stroke     | `#000000` (black)    |
| node stroke-width | `1px`              |
| group node fill | `#EEEEEE`           |
| edge color      | `#AAAAAA` (medium gray)|
| font family     | `Arial`              |
| font size       | `24px` (demo) / `12px` (typical) |

### yFiles Demo Color Palette (6 primary sets)
| Palette        | Fill      | Stroke    |
|----------------|-----------|-----------|
| demo-orange    | `#ff6c00` | `#662b00` |
| demo-blue      | `#242265` | `#0e0e28` |
| demo-red       | `#ca0c3b` | `#510518` |
| demo-green     | `#61a044` | `#27401b` |
| demo-purple    | `#a37ab3` | `#413148` |
| demo-lightblue | `#46a8d5` | `#1c4355` |

NOTE: yEd desktop's actual out-of-the-box defaults (what you see when you
create a new node) are not available from source. yEd uses a light orange
fill (`#FFCC00` range) for new shape nodes, with black border and Arial font.
The 96-entry demo palette covers theme variants for the SDK.

---

## 7. Draw.io / diagrams.net Defaults

Source: `mxgraph/.../mxConstants.js`, `mxgraph/.../mxStylesheet.js`,
`drawio/.../Sidebar.js`, `drawio/.../default.xml`, `drawio/.../Graph.js`,
`drawio/.../Dialogs.js`

### mxGraph Core Defaults (underlying engine)
| Property           | Value              |
|--------------------|--------------------|
| vertex fillColor   | `#C3D9FF` (light blue) |
| vertex strokeColor | `#6482B9` (medium blue) |
| vertex fontColor   | `#774400` (brown)   |
| edge strokeColor   | `#6482B9`          |
| edge fontColor     | `#446299` (blue-gray)|
| edge endArrow      | `classic`          |
| shape (vertex)     | rectangle          |
| shape (edge)       | connector          |

NOTE: Draw.io overrides these mxGraph defaults. The modern draw.io
default.xml defines its own styles:

### Draw.io default.xml Styles
| Property (vertex) | Value                 |
|--------------------|-----------------------|
| shape              | `label`               |
| perimeter          | `rectanglePerimeter`  |
| fontFamily         | `Helvetica`           |
| fontSize           | `12`                  |
| align              | `center`              |
| verticalAlign      | `middle`              |

| Property (edge)    | Value                 |
|--------------------|-----------------------|
| shape              | `connector`           |
| endArrow           | `classic`             |
| fontFamily         | `Helvetica`           |
| fontSize           | `11`                  |
| rounded            | `1`                   |

### Draw.io Sidebar Shape Defaults
Basic shapes in the sidebar palette use NO hardcoded fill/stroke colors.
When dropped, they inherit from the graph's default style. Stencil shapes
get fallback: `fillColor=#ffffff;strokeColor=#000000;strokeWidth=2`.

### Draw.io Signature Color Palette (from default.xml and Dialogs.js)
| Style     | Fill      | Gradient  | Stroke    |
|-----------|-----------|-----------|-----------|
| Blue      | `#DAE8FC` | `#7EA6E0` | `#6C8EBF` |
| Green     | `#D5E8D4` | `#97D077` | `#82B366` |
| Yellow    | `#FFF2CC` | `#FFD966` | `#D6B656` |
| Orange    | `#FFCD28` | `#FFA500` | `#D79B00` |
| Red       | `#F8CECC` | `#EA6B66` | `#B85450` |
| Purple    | `#E1D5E7` | `#8C6C9C` | `#9673A6` |
| Pink      | `#E6D0DE` | `#B5739D` | `#996185` |
| Turquoise | `#D5E8D4` | `#67AB9F` | `#6A9153` |
| Gray      | `#F5F5F5` | `#B3B3B3` | `#666666` |

The "blue" variant (`#DAE8FC` fill, `#6C8EBF` stroke) is draw.io's
signature look -- what most people associate with "a draw.io diagram."

### mxGraph Constants
| Constant                    | Value    |
|-----------------------------|----------|
| DEFAULT_FONTFAMILY          | `Arial,Helvetica` |
| DEFAULT_FONTSIZE            | `11`     |
| DEFAULT_STARTSIZE           | `40`     |
| DEFAULT_MARKERSIZE          | `6`      |
| DEFAULT_IMAGESIZE           | `24`     |
| ARROW_WIDTH                 | `30`     |
| ARROW_SIZE                  | `30`     |
| ARROW_SPACING               | `0`      |
| LINE_ARCSIZE                | `20`     |
| RECTANGLE_ROUNDING_FACTOR   | `0.15`   |
| HANDLE_SIZE                 | `6`      |
| SHADOW_OPACITY              | `1` (Draw.io overrides to `0.25`) |
| SHADOW_COLOR                | `gray`   |

### Draw.io Grid & Page
| Property          | Light       | Dark        |
|-------------------|-------------|-------------|
| grid-color        | `#d0d0d0`   | `#424242`   |
| page-break-color  | `#c0c0c0`   | --          |
| shadow-opacity    | `0.25`      | --          |
| shadow-color      | `#000000`   | --          |

---

## 8. Neo4j Browser

Source: `neo4j-browser` GitHub repo (cloned at HEAD), specifically:
- `src/neo4j-arc/graph-visualization/models/GraphStyle.ts` (DEFAULT_STYLE, DEFAULT_COLORS)
- `src/neo4j-arc/graph-visualization/GraphVisualizer/Graph/styled.tsx` (SVG styles)
- `src/neo4j-arc/graph-visualization/GraphVisualizer/Graph/visualization/renderers/init.ts`
- `src/neo4j-arc/graph-visualization/GraphVisualizer/Graph/visualization/utils/PairwiseArcsRelationshipRouting.ts`
- `src/neo4j-arc/graph-visualization/constants.ts` (force layout params)
- `src/neo4j-arc/graph-visualization/utils/StraightArrow.ts`
- `src/neo4j-arc/common/styles/themes.tsx` (UI theme)
- `src/browser/styles/themes.ts` (light/dark themes)
- `src/neo4j-arc/common/components/LabelAndReltypes.ts` (label/reltype chips)
- `@neo4j-devtools/word-color@0.0.8` (npm package, extracted)

### Node Defaults (from DEFAULT_STYLE)
| Property             | Value       | Notes                           |
|----------------------|-------------|---------------------------------|
| shape                | circle      | Always circle, never other shapes |
| diameter             | `50px`      | Radius = 25px                   |
| color (fill)         | `#A5ABB6`   | Medium gray (fallback only)     |
| border-color         | `#9AA1AC`   | Slightly darker gray            |
| border-width         | `2px`       |                                 |
| text-color-internal  | `#FFFFFF`   | White text inside node          |
| font-size            | `10px`      |                                 |
| font-family          | `sans-serif`| Hardcoded in GraphGeometryModel |

### Available Node Sizes
| Preset | Diameter |
|--------|----------|
| XS     | `10px`   |
| S      | `20px`   |
| M      | `50px` (default) |
| L      | `65px`   |
| XL     | `80px`   |

### Node Color Palette (12 colors, assigned round-robin by label)
| Index | Fill      | Border    | Text Internal | Description    |
|-------|-----------|-----------|---------------|----------------|
| 0     | `#604A0E` | `#423204` | `#FFFFFF`     | Dark brown     |
| 1     | `#C990C0` | `#b261a5` | `#FFFFFF`     | Orchid pink    |
| 2     | `#F79767` | `#f36924` | `#FFFFFF`     | Salmon/orange  |
| 3     | `#57C7E3` | `#23b3d7` | `#2A2C34`     | Teal/cyan (signature) |
| 4     | `#F16667` | `#eb2728` | `#FFFFFF`     | Coral red      |
| 5     | `#D9C8AE` | `#c0a378` | `#2A2C34`     | Tan/beige      |
| 6     | `#8DCC93` | `#5db665` | `#2A2C34`     | Soft green     |
| 7     | `#ECB5C9` | `#da7298` | `#2A2C34`     | Light pink     |
| 8     | `#4C8EDA` | `#2870c2` | `#FFFFFF`     | Medium blue    |
| 9     | `#FFC454` | `#d7a013` | `#2A2C34`     | Gold/amber     |
| 10    | `#DA7194` | `#cc3c6c` | `#FFFFFF`     | Rose           |
| 11    | `#569480` | `#447666` | `#FFFFFF`     | Teal green     |

NOTE: The "signature Neo4j teal" is palette index 3 (`#57C7E3`). In the
Movies demo, `Person` nodes get orchid pink (`#C990C0`) and `Movie` nodes
get the first available color. The gray default (`#A5ABB6`) is only used
when no label is present.

### Generated Colors (word-color algorithm, newer Neo4j versions)
When `useGeneratedDefaultColors=true`, colors are computed from the label
name using `@neo4j-devtools/word-color@0.0.8`:
1. Hash the label string using `djb2` variant: `hash = ((hash << 5) - hash + charCode) << 0`
2. Generate 3 cascaded hashes (each hashes the previous result's string)
3. Map to OKLCH color space:
   - L (lightness): hash1 mapped to range `[70, 95]` (percent)
   - C (chroma): hash2 mapped to range `[5, 20]` (percent)
   - H (hue): hash3 mapped to range `[0, 360]`
4. Border: darken fill by -20% (per-channel shade)
5. Text: pick most readable between `#2A2C34` and `#FFFFFF` (WCAG AA, ratio >= 4.5)

### Relationship Defaults (from DEFAULT_STYLE)
| Property             | Value       | Notes                             |
|----------------------|-------------|-----------------------------------|
| color (fill)         | `#A5ABB6`   | Same gray as default node         |
| shaft-width          | `1px`       | Very thin                         |
| font-size            | `8px`       | Smaller than node text            |
| padding              | `3px`       | Padding around caption text       |
| text-color-external  | `#000000`   | Black text outside shaft          |
| text-color-internal  | `#FFFFFF`   | White text inside thick shafts    |
| caption              | `<type>`    | Shows relationship type by default|

### Available Shaft Widths
| Preset | Width  |
|--------|--------|
| 1      | `1px` (default) |
| 2      | `2px`  |
| 3      | `3px`  |
| 4      | `5px`  |
| 5      | `8px`  |
| 6      | `13px` |
| 7      | `25px` |
| 8      | `38px` |

### Arrow/Edge Geometry
| Property              | Value | Notes                               |
|-----------------------|-------|-------------------------------------|
| arrow head width      | `shaft-width + 6` | So default = 7px           |
| arrow head height     | same as head width | Equilateral triangle       |
| arrow style           | filled path, no stroke | `fill=color, stroke=none` |
| caption placement     | `internal` if shaft-width > font-size, else `external` |
| multi-edge deflection | 30 deg per step, max 150 deg total |
| self-loop straight length | `40px` |

Arrow rendering: The arrow is drawn as a single SVG `<path>` with `fill`
set to the relationship color and `stroke: none`. It is NOT a line with a
separate arrowhead marker. The entire shaft+head is one filled shape:
- StraightArrow: rectangular shaft + triangular head
- ArcArrow: curved shaft (SVG arc) + triangular head
- LoopArrow: circular arc + triangular head

The shaft has width (configurable), and the head widens to `shaft-width + 6`.
When caption is "external", the shaft has a gap cut in the middle where the
text sits above/beside the edge.

### Background Colors
| Theme  | SVG Background (frameBackground) | Notes              |
|--------|----------------------------------|--------------------|
| Light  | `#F9FCFF`                        | Very pale blue-white |
| Dark   | `#292C33`                        | Dark blue-gray     |

### Selection & Hover
| Element      | State    | Color     | Opacity |
|--------------|----------|-----------|---------|
| Node ring    | normal   | (hidden)  | `0`     |
| Node ring    | hover    | `#6ac6ff` | `0.3`   |
| Node ring    | selected | `#fdcc59` | `0.3`   |
| Rel overlay  | normal   | (hidden)  | `0`     |
| Rel overlay  | hover    | `#6ac6ff` | `0.3`   |
| Rel overlay  | selected | `#fdcc59` | `0.3`   |
| Ring stroke-width | --  | `8px`     | --      |
| Ring radius  | --       | `node.radius + 4` | --   |

### UI Label/RelType Chips (legend panel)
| Element         | Property        | Value       |
|-----------------|-----------------|-------------|
| Label chip      | border-radius   | `20px` (pill) |
| Label chip      | font-size       | `12px`      |
| Label chip      | font-weight     | `bold`      |
| Label chip      | padding         | `4px 7px 4px 9px` |
| Label chip      | bg-color        | node's label color |
| Label chip      | text-color      | node's text-color-internal |
| RelType chip    | border-radius   | `3px` (rounded rect) |
| RelType chip    | font-size       | `12px`      |
| RelType chip    | font-weight     | `bold`      |
| RelType chip    | padding         | `4px 7px 4px 5px` |
| RelType chip    | bg-color        | relationship color |
| Default chip bg (no style) | bg-color | `#9195a0` |
| Default chip text          | color    | `#30333a` |
| Font family     | --              | `"Helvetica Neue", Helvetica, Arial, sans-serif` |

### Typography
| Context              | Font Family                                    | Size   |
|----------------------|------------------------------------------------|--------|
| Node captions        | `sans-serif` (hardcoded in canvas measurement) | `10px` |
| Relationship captions| `sans-serif`                                   | `8px`  |
| UI text              | `"Helvetica Neue", Helvetica, Arial, sans-serif` | varies |
| Drawer headers       | `'Open Sans', 'HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', Helvetica, Arial, sans-serif` | varies |

### Force Layout (d3-force parameters)
| Parameter            | Value       | Notes                           |
|----------------------|-------------|---------------------------------|
| velocity decay       | `0.4`       | Friction                        |
| charge strength      | `-400`      | Much stronger than d3 default (-30) |
| center X strength    | `0.03`      | Weak centering                  |
| center Y strength    | `0.03`      |                                 |
| link distance        | `node_a.radius + node_b.radius + 90` | = 2 * LINK_DISTANCE (45) + radii |
| collide radius       | `node.radius + 25` |                            |
| alpha                | `1`         |                                 |
| alpha min            | `0.05`      |                                 |
| dragging alpha       | `0.8`       |                                 |
| max precomputed ticks| `300`       | Or 250ms, whichever first       |
| extra ticks/render   | `10`        |                                 |
| zoom min scale       | `0.1`       |                                 |
| zoom max scale       | `2`         |                                 |
| zoom fit padding     | `5%`        |                                 |

### Key Visual Identity Notes
1. **Nodes are always circles** -- no rectangles, diamonds, or other shapes
2. **Border is always present** at 2px, slightly darker than fill
3. **Text is always white** on dark fills, `#2A2C34` on light fills
4. **The signature look** comes from the 12-color palette with saturated,
   pastel-adjacent colors on a near-white background
5. **Relationships are very thin** (1px default) with filled arrow paths
6. **Relationship text** is tiny (8px) and sits at the midpoint of the shaft
7. **No edge labels with background boxes** -- text is directly on/near the edge
8. **The teal (#57C7E3)** is the most recognizable Neo4j color, often the
   first label color users see in tutorials (e.g., Movie database)

---

## Cross-Tool Comparison Summary

### Signature Colors (what makes each tool recognizable)
| Tool         | Primary Fill   | Primary Stroke | Text     | Background |
|--------------|----------------|----------------|----------|------------|
| Mermaid      | `#ECECFF`      | `#9370DB`      | `#333`   | `white`    |
| D3 Observable| `currentColor` | `#fff`         | system   | transparent|
| Cytoscape.js | `#999`         | `#000` (w=0)   | `#000`   | white      |
| Gephi        | `#000000`      | `#000000`      | node color| `#ffffff` |
| Obsidian     | `#5c5c5c`      | --             | `#222222`| `#ffffff`  |
| yEd/yFiles   | `#CCCCCC`      | `#000000`      | inherited| white      |
| Draw.io      | `#DAE8FC`      | `#6C8EBF`      | `#774400`| white      |
| mxGraph raw  | `#C3D9FF`      | `#6482B9`      | `#774400`| white      |
| Neo4j Browser| `#57C7E3` (teal)| `#23b3d7`     | `#2A2C34`| `#F9FCFF`  |

### Default Font Stacks
| Tool         | Font Family                                     | Size   |
|--------------|-------------------------------------------------|--------|
| Mermaid      | `trebuchet ms, verdana, arial, sans-serif`       | `16px` |
| D3           | system default                                   | varies |
| Cytoscape.js | `Helvetica Neue, Helvetica, sans-serif`          | `16`   |
| Gephi        | `Arial, Bold`                                    | `32` (scaled 0.5x = effective 16) |
| Obsidian     | system default (user-configurable)               | `16px` |
| yFiles       | `Arial`                                          | `12px` |
| Draw.io      | `Helvetica` (vertex `12`, edge `11`)             | `12`   |
| mxGraph      | `Arial,Helvetica`                                | `11`   |
| Neo4j Browser| `sans-serif` (node `10px`, rel `8px`)            | `10px` |

### Default Node Shapes
| Tool         | Shape      | Default Size     |
|--------------|------------|------------------|
| Mermaid      | roundrect  | content-sized + padding 15 |
| D3           | circle     | r=5              |
| Cytoscape.js | ellipse    | 30x30            |
| Gephi        | circle     | (not set; UI varies) |
| Obsidian     | circle     | (WebGL internal)  |
| yEd/yFiles   | rectangle  | varies            |
| Draw.io      | rectangle  | content-sized     |
| Neo4j Browser| circle     | diameter 50px (r=25) |

---

## Confidence Notes

- **HIGH confidence**: Mermaid, D3, Cytoscape.js, Draw.io -- values come
  directly from source code on GitHub.
- **MEDIUM confidence**: Obsidian -- CSS variables confirmed from developer
  docs and theme source; graph-specific rendering sizes undocumented.
- **MEDIUM confidence**: Gephi -- many defaults found in source but node
  color defaults are surprisingly sparse (black by init, colored by UI).
- **HIGH confidence**: Neo4j Browser -- values come directly from the
  neo4j-browser GitHub repo source code and the @neo4j-devtools/word-color
  npm package (extracted and read). Every value verified against source.
- **LOW confidence**: yEd desktop -- closed source; values extrapolated from
  yFiles SDK demos, which may not match yEd's actual defaults exactly.
