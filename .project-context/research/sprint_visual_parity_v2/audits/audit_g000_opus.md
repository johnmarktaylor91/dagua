# S0 Round-0 VLM Census -- Opus 4.8 (audit_g000_opus)

- Round: g000 | Reference: graphviz 7.0.5 (svg-cairo) | geometry_mode: injected
- Model: claude-opus-4-8 (primary secondary auditor) | Panels: 9 (tiny_graph, shape_atlas,
  arrowhead_atlas, edge_styles_showcase, colors_showcase, cluster_showcase, label_variety,
  spline_stress, mixed_styles)
- Verdict: **FAIL** | Findings: 29 | Inspection log: complete (9 panels x 9 categories)

## Summary (verbatim)
> The dominant, pervasive gap is node sizing: dagua renders nodes far smaller than graphviz
> (roughly 0.4-0.5x area) and does NOT expand nodes to fit their labels, so single-line,
> multi-line, and long labels overflow/clip across nearly every panel. Second, shape and
> arrowhead fidelity collapse: shape_atlas/arrowhead_atlas show distinct graphviz primitives
> reduced to near-uniform small ellipses / indistinct tips. Third, edge/routing diverges:
> strokes and arrowheads scale too hard with penwidth, splines/back-edges render straight, and
> the stage-1 self-loop is missing; fills and background colors otherwise match well --
> confirming the gaps are geometry/sizing/routing rather than color.

## Themed inventory (29 findings)

### 1. NODE SIZING / AUTOSIZE -- DOMINANT, HIGH -> S1
- f1 tiny_graph: ellipses ~0.4x width / 0.5x height, tiny fixed size regardless of label (conf .95)
- f10 edge_styles: node area ~60-70% smaller, labels clipped ("solid 1x" overflows) (conf .95)
- f20 label_variety: long labels ~50-70% too small -> overflow/clip on ~5 nodes (conf .95)
- f21 label_variety: multi-line node height ~0.5x needed, text overlap (conf .90)
- f5 shape_atlas: shapes collapse toward small uniform ellipses ~50-70% smaller (conf .85)
- f18 cluster_showcase: intra-cluster nodes ~0.4x (conf .80)
- f9/f14/f25/f26 (arrowhead_atlas/colors/spline/mixed): nodes ~0.45-0.5x; fill HUE matches, geometry not color
- f2/f6/f15 typography: node font ~0.45x apparent size (coupled to autosize)

### 2. SHAPE PRIMITIVE FIDELITY -- HIGH/LOW -> S1 sizing + S6
- f5 shape_atlas: star/diamond/parallelogram/cylinder/folder collapse to ellipses
- f7 peripheries / double-border (doublecircle, Mdiamond) under-rendered ~0.5-0.7pt thin (conf .5)

### 3. ARROWHEADS -- HIGH/MED -> S3
- f8 arrowhead_atlas: primitives tiny/faint/indistinct ~40-60% smaller (conf .8)
- f12 edge_styles: arrowhead size tracks penwidth too strongly (thick ~1.5x large, thin ~0.6x small)
- f22 label_variety: arrowheads ~1.3x too large

### 4. EDGE STROKE / DASH -- MED -> S5 / theme
- f11 edge_styles: thick strokes ~1.3-1.5x heavier; dotted pitch ~1.2x coarser
- f24 spline_stress: red dashed ~1.4x heavier; dotted pitch ~1.2x coarser
- f27 mixed_styles: colored dashed strokes ~1.3-1.5x heavier

### 5. SPLINES / ROUTING -- HIGH -> S5
- f23 spline_stress: splines render straight, **self-loop MISSING**, back-edges straight not curved (conf .85)

### 6. CLUSTERS -- HIGH/LOW -> S6
- f17 cluster_showcase: aspect distorted, too tall/narrow, height ~1.5-2x, nodes ~0.4x (conf .85)
- f19 nested cluster insets/margins off ~1.3x (conf .55)

### 7. CANVAS / RANKSEP -- MED/LOW
- f4 tiny_graph / f16 colors: inter-node vertical gap ~1.5-1.6x (ranksep/node_sep too large)

### 8. LABELS -- LOW (secondary to layout)
- f13 edge_styles / f28 mixed_styles: edge-label offsets differ ~10-25px (follow shifted layout)

### CLEAN (match well)
- fills/colors: hues match across colors_showcase / mixed_styles
- f29 canvas/background: white bg + header strip match ~exactly

## S1 implication
Node autosize is the #1 P0 gap and is FOUNDATIONAL: nodes are clamped to the theme floor
(min_width=54 / min_height=36 on graphviz_strict) and do NOT grow to the label bbox, so labels
clip and everything reads "too small". Unlocking autosize (grow node to label + padding) should
cascade-fix label clipping and most of the apparent under-sizing across all panels. First S1
sweep: autosize + min-size + padding coupling.
