# S0 Round-0 VLM Census -- GPT-5.6 Sol (audit_g000_sol)

- Round: g000 | Reference: graphviz 7.0.5 (svg-cairo) | geometry_mode: injected
- Model: gpt-5.6-sol (primary auditor) | Panels: 9
- Verdict: **FAIL** | Findings: 49 | Severity: {'HIGH': 34, 'MED': 14, 'LOW': 1}

## Summary (verbatim)
> The baseline fails decisively: Dagua systematically underscales node geometry and node typography while rendering many edge strokes too heavily, producing a severe internal scale inconsistency. Routing parity is also far from baseline in the cluster, spline, and mixed-style panels, including missing or malformed self-loops, detached endpoints, straightened splines, cluster-title crossings, and clipped long labels.

## HIGH findings
- **tiny_graph** / typography: Dagua node labels are substantially smaller than Graphviz despite matching text content. _(m: glyph height ~10-12 px versus ~21-23 px, approximately 48% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **tiny_graph** / node geometry: Dagua ellipses are much smaller and vertically flatter than the Graphviz nodes. _(m: typical ellipse ~40x23 px versus ~90x58 px; width ~56% smaller and height ~60% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **tiny_graph** / canvas: Dagua uses far less of its half-canvas, especially horizontally. _(m: content width ~41 px versus ~91 px and total height ~256 px versus ~300 px; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **shape_atlas** / typography: Dagua labels are reduced to near-microtext relative to the Graphviz labels. _(m: typical glyph height approximately 45-60% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **shape_atlas** / node geometry: Dagua compresses the shape envelopes so strongly that several distinct primitives become tiny icon-like silhouettes rather than Graphviz-sized nodes. _(m: common shape widths ~50-70% smaller; several narrow primitives are only ~4-8 px wide; real_cosmetic_gap/fixable_theme_or_render)_
- **arrowhead_atlas** / typography: Dagua sample labels are materially smaller and less legible than the Graphviz labels. _(m: glyph height approximately 50-65% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **arrowhead_atlas** / node geometry: Dagua shrinks the endpoint nodes from visible ellipses into very small marks. _(m: endpoint height ~3-6 px versus ~11-15 px, approximately 55-75% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **arrowhead_atlas** / arrowheads: Dagua arrowheads are too small for open, filled, and compound variants to retain Graphviz's visible primitive distinctions. _(m: most primitives occupy ~1-4 px versus ~3-8 px; roughly 40-65% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **edge_styles_showcase** / typography: Dagua node text is about half the Graphviz size. _(m: glyph height ~10-12 px versus ~21-23 px, approximately 48% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **edge_styles_showcase** / node geometry: Dagua ellipses are dramatically smaller and flatter than Graphviz's nodes. _(m: typical width ~45-65 px versus ~115-135 px and height ~16-20 px versus ~55-62 px; real_cosmetic_gap/fixable_theme_or_render)_
- **edge_styles_showcase** / strokes/borders: Dagua's ordinary and thin stems are too heavy in absolute pixels. _(m: nominal stems ~3 px versus ~1.5 px; approximately 1.8-2.0x too thick; real_cosmetic_gap/fixable_theme_or_render)_
- **colors_showcase** / typography: Dagua text is substantially smaller than the Graphviz labels. _(m: glyph height ~13-15 px versus ~21-23 px, approximately 35-40% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **colors_showcase** / node geometry: Dagua nodes are far smaller and much flatter while preserving roughly the same center positions. _(m: typical node ~44-56x15-18 px versus ~90-120x58-61 px; height ~70% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **colors_showcase** / canvas: Dagua's colored chain occupies less than half the reference width. _(m: node footprint ~44-56 px wide versus ~90-120 px, approximately 50-55% narrower; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **cluster_showcase** / typography: Dagua cluster-member labels are much smaller than Graphviz's node labels. _(m: glyph height ~8-11 px versus ~20-23 px, approximately 50-60% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **cluster_showcase** / node geometry: Dagua member ellipses are reduced to thin capsules instead of Graphviz-sized ellipses. _(m: typical height ~15-20 px versus ~54-62 px, approximately 65-72% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **cluster_showcase** / clusters: Dagua cluster widths do not track Graphviz's content-driven widths and are disproportionately narrow. _(m: Tiny width ~60 px versus ~151 px; Medium ~70 px versus ~184 px; Large ~168 px versus ~335 px; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **cluster_showcase** / edges: Dagua replaces multiple curved or carefully offset Graphviz routes with long straight diagonals that cross cluster borders and title regions. _(m: route displacement commonly ~80-250 px; at least three long edges cross different regions than the reference; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **cluster_showcase** / labels: A long Dagua cross-cluster edge passes through the large-cluster title area, reducing label clearance absent in Graphviz. _(m: stem intersects the title band over roughly 80-120 px; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **label_variety** / typography: Dagua labels are materially smaller than Graphviz's labels across short, long, and multiline cases. _(m: glyph height ~12-15 px versus ~21-23 px, approximately 35-45% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **label_variety** / node geometry: Dagua nodes are vertically flattened and substantially smaller than the Graphviz envelopes. _(m: short-node height ~17-20 px versus ~58-61 px; approximately 65-70% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **label_variety** / labels: Dagua clips or truncates both ends of the longest label while Graphviz renders the complete string. _(m: approximately 3-5 leading characters and 2-4 trailing characters are not legible; real_cosmetic_gap/fixable_theme_or_render)_
- **spline_stress** / typography: Dagua node labels are substantially smaller than Graphviz's labels. _(m: glyph height ~10-12 px versus ~20-22 px, approximately 45-50% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **spline_stress** / node geometry: Dagua nodes are much smaller and flatter than the Graphviz nodes. _(m: typical height ~17-20 px versus ~56-62 px, approximately 65-70% smaller; real_cosmetic_gap/fixable_theme_or_render)_
- **spline_stress** / edges: The Graphviz black self-loop at stage 1 is absent from the Dagua panel. _(m: one complete loop of approximately 55x45 px is missing; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **spline_stress** / edges: Dagua reduces the green loop to a short bent fragment rather than a closed return loop around stage 3. _(m: visible route length ~35-45 px versus ~85-100 px; roughly 55% shorter; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **spline_stress** / edges: Dagua's long black route terminates far from the flat-b node and lacks Graphviz's curved approach. _(m: terminal gap roughly 75-100 px from the target boundary; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **spline_stress** / edges: Dagua renders the red returns as near-straight dashed tracks with offset endpoints instead of Graphviz's broad curved splines. _(m: lateral path deviation roughly 40-130 px; arrow tips land ~20-40 px from the reference-relative contact points; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **spline_stress** / arrowheads: The Dagua blue arrowhead is detached well above the target rather than contacting the node boundary. _(m: tip-to-target gap ~35-45 px versus approximately 0-2 px in Graphviz; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **mixed_styles** / typography: Dagua node text is significantly smaller than Graphviz's text; Review also appears relatively heavier. _(m: glyph height approximately 40-55% smaller; Review weight visually about one step heavier; real_cosmetic_gap/fixable_theme_or_render)_
- **mixed_styles** / node geometry: Dagua nodes are severely undersized and vertically flattened. _(m: Start ~36x20 px versus ~91x59 px; Review ~45x18 px versus ~126x61 px; real_cosmetic_gap/fixable_theme_or_render)_
- **mixed_styles** / edges: Graphviz's curved branch routes become mostly straight Dagua segments, changing label association and visual flow. _(m: mid-route displacement typically ~30-85 px; curvature is effectively reduced to zero on several branches; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **mixed_styles** / labels: Dagua edge labels use a sans-serif face and sit farther from their associated stems than Graphviz's serif labels. _(m: label-to-stem displacement differs by ~25-60 px across needs sign-off, approved, rework, and retry; real_cosmetic_gap/needs_layout_or_routing_scope)_
- **mixed_styles** / strokes/borders: Dagua stems are too thick and its dash/dot cadence is more open than Graphviz's. _(m: stems ~3 px versus ~1.5 px; dash or dot pitch approximately 1.5-2x larger; real_cosmetic_gap/fixable_theme_or_render)_

## MED / LOW findings
- (MED) tiny_graph/arrowheads: Dagua arrowheads are shorter and narrower than the Graphviz triangles.
- (MED) shape_atlas/canvas: Dagua leaves substantially more unused vertical and horizontal space inside the atlas group boxes because rendered shapes are underscaled.
- (MED) arrowhead_atlas/canvas: The Dagua samples occupy a much smaller fraction of each boxed atlas row than the Graphviz samples.
- (MED) edge_styles_showcase/typography: Dagua edge labels use a sans-serif face while the Graphviz reference uses a Times-like serif face.
- (MED) edge_styles_showcase/strokes/borders: Dagua's dot spacing is much more open than Graphviz's dotted pattern.
- (MED) edge_styles_showcase/labels: Dagua edge labels dominate the underscaled graph and are displaced from the corresponding stem midpoints.
- (MED) edge_styles_showcase/canvas: Dagua compresses the graph into a much narrower footprint while retaining nearly the full vertical run.
- (MED) colors_showcase/strokes/borders: Dagua stems are heavier than Graphviz's lines.
- (LOW) colors_showcase/arrowheads: Dagua arrowheads are modestly smaller than Graphviz's triangles.
- (MED) cluster_showcase/typography: Dagua cluster titles use a sans-serif face rather than Graphviz's serif face.
- (MED) cluster_showcase/strokes/borders: Dagua's long edges are about twice as heavy as the reference edges.
- (MED) label_variety/labels: Dagua provides almost no horizontal padding around the long single-line label, with glyphs crowding the ellipse boundary.
- (MED) label_variety/strokes/borders: Dagua's edge lines are heavier than Graphviz's despite the much smaller nodes.
- (MED) spline_stress/strokes/borders: Dagua edges are substantially heavier than the Graphviz strokes.
- (MED) mixed_styles/canvas: Dagua's graph is horizontally compressed and centered into a narrower region.

Raw structured findings: audit_g000_sol.json (same directory).