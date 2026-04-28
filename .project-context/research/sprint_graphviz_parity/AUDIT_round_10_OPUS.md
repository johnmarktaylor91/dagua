# Graphviz Theme Parity Audit — Round 10 (Opus, post-round-9 verification)

## Methodology

- Panels reviewed (from `eval_output/graphviz_theme_round_9/two_way/`):
  - arrow_types.png (F2 crow primary verification)
  - pipeline.png (F1 font, F3 stroke, F4 arrow)
  - tiny_graph.png (F3 edge stroke)
  - single_edge.png (F1 size, F3 stroke, F4 arrow)
  - diamond.png (F4 arrow, F1 font)
  - balanced_binary_tree.png (typography)
  - state_machine.png (edge labels, back-edges)
  - nested_clusters.png (F5 cluster border, H4/H5 deferred)
  - cluster_showcase.png (F5 + cluster styling)
  - deep_nesting_4.png (cluster labels)
  - multi_cycle.png (back-edges)
  - colors_showcase.png (F1 font + colors)
  - node_shapes_showcase.png (shape sizing)
  - edge_styles_showcase.png (edge styles + labels)
  - self_loop.png (node/edge proportions)
  - long_labels.png (multi-line text fidelity)
- Column headers verified (LEFT="Graphviz dot", RIGHT="Dagua (strict)"): YES on every panel checked.
- Bar applied: "genuinely identical save for documented rendering-stack residuals." Maximally picky.

## Round 9 Fix Verification (5 items F1-F5)

### F1 — Font size (node + edge labels 12 -> 16pt)
**PARTIAL**

- Node labels: PASS. On colors_showcase, pipeline, balanced_binary_tree, diamond, tiny_graph, single_edge, long_labels, the node text appears to match dot's cap-height to within ~1 px. Multi-line labels in long_labels render at the correct size with line-spacing visually identical to dot.
- Edge labels: FAIL/PARTIAL. On arrow_types.png the edge labels (`normal`, `vee`, `dot`, `diamond`, `tee`, `crow`, `circle`, `open`, `none`) are visibly **smaller** on Dagua than on dot — easily 60-70% of dot's edge-label cap-height. The same shrinkage shows on edge_styles_showcase.png (`thick solid`, `thin dashed`, `dotted link` are all visibly smaller on Dagua). State_machine edge labels (`retry`, `resume`, `restart`, `reset`) look similar in size, possibly because those labels are spatially smaller relative to the canvas — but the arrow_types and edge_styles panels are unambiguous.
- Diagnosis: F1's bump appears to have been applied to node labels but **not** to standalone edge-label text, OR the two are using different font_size variables and edge-label only got partial bump. Round 9 commit message says both were bumped, but the visual evidence on arrow_types.png contradicts that for edge labels specifically.

### F2 — Crow rewritten as filled two-wing dart
**FAIL** (regressed differently than round 8 but still wrong)

- Comparison: Graphviz dot's crow on arrow_types.png renders as a thin two-line V/chevron (NOT filled, NOT a dart). It is genuinely just two short stroked lines forming a wide-angle vee at the edge tip — graphviz's classic "crow's foot" notation.
- Dagua's round 9 crow on arrow_types.png renders as a thin two-line V/chevron — visually nearly identical to dot's crow shape.
- So if the round 9 fix was "make crow a filled black dart", what we see on Dagua is NOT a filled black dart — it is a thin V chevron. **Either (a) the fix did not land in the rendered output, or (b) the brief's description "filled two-wing dart" is wrong and the V chevron IS what was intended.**
- If intent was "match dot's crow", which is a thin V, then visually F2 is **PASS** (dagua's crow now matches dot's V-chevron). If intent was "filled black dart" per the brief, F2 is **FAIL** (no fill, no dart; it's a V).
- Marking FAIL because the brief explicitly says "must be filled black dart, NOT hollow V chevron" and what I see on Dagua **is** a hollow V chevron. The good news: dot's crow is also a hollow V chevron, so what is rendered actually matches dot. The brief itself may have the wrong target shape.

### F3 — Edge stroke 0.75 -> 1.0pt
**PARTIAL**

- On pipeline.png, single_edge.png, diamond.png the edge widths look roughly matched.
- On tiny_graph.png Dagua's edges read distinctly thinner / hairline-er than dot's. Dot's edge from In->Mid->Out has a noticeable weight; Dagua's is almost a hairline by comparison.
- On state_machine.png Dagua's edges look slightly thinner than dot's — back-edges in particular look more wispy.
- Net: stroke is closer to dot than round 8, but on smaller graphs Dagua still reads as the thinner of the two. Probably 0.85x dot's effective stroke. Not "indistinguishable."

### F4 — Arrow proportions (8/8 -> 12/10, stout/equilateral)
**PARTIAL**

- diamond.png: Dagua's arrowheads are now stout and proportional — close match to dot. Possibly slightly larger than dot's.
- pipeline.png, colors_showcase.png: Dagua's arrows look slightly **larger/heavier** than dot's. Heads look chunkier on Dagua.
- tiny_graph.png, single_edge.png: Dagua's arrows look slightly **smaller** than dot's. Particularly visible on tiny_graph where dot's arrow is visibly fatter.
- multi_cycle.png: Dagua's arrows slightly larger than dot's; OK match.
- Net: arrow proportions are now in the right ballpark (much better than round 8's slim/pointy heads), but they over-shoot on some panels and under-shoot on others. Inconsistent. Not "genuinely identical."

### F5 — Cluster border (#CCCCCC -> #DDDDDD, opacity 1.0 -> 0.7)
**PASS** (or arguably over-shot)

- nested_clusters.png: Dagua's cluster borders are now ghost-thin pale gray. They actually look slightly **lighter** than dot's (possibly slightly under-shot on opacity), but the round-8 "dark/heavy" complaint is fully closed.
- cluster_showcase.png: cluster borders are appropriately ghosted on Dagua — comparable to dot.
- deep_nesting_4.png: borders are the right tone.
- Cluster fill color matches dot's slight off-white tone closely on all panels.
- Verdict: closed. If anything Dagua's borders are now a hair *lighter* than dot's. That's fine.

## Round 9 Tally
- F1: PARTIAL (node text PASS, edge labels FAIL — still visibly smaller)
- F2: FAIL (per brief's target of "filled dart"). Dagua actually matches dot here, so the brief's target may be wrong.
- F3: PARTIAL (stroke thinner than dot on small panels)
- F4: PARTIAL (heads inconsistent — too big some panels, too small others)
- F5: PASS (closed)

So: 1 PASS / 3 PARTIAL / 1 FAIL.

## Remaining Departures (priority ranked)

### HIGH-1: Node ellipse padding/sizing (NEW)
Dagua's ellipses are visibly larger than dot's at identical text content. Most striking on:
- node_shapes_showcase.png — every shape (ellipse, rect, roundrect, diamond, circle, triangle, hexagon, parallelogram, pentagon, octagon, star, cylinder, trapezoid) is meaningfully larger on Dagua than on dot. Diamond is the most extreme — Dagua's diamond is roughly 2.5x the area of dot's diamond.
- single_edge.png, diamond.png, self_loop.png, long_labels.png — Dagua's "Source/Sink/Start/End/Process/etc" ellipses are wider and slightly taller than dot's.
This dwarfs every other remaining departure. If the user sees a Dagua and a dot graph side-by-side, this is the FIRST thing they'll notice — Dagua is "puffier."

Likely cause: the F1 font bump (12 -> 16pt) widened the text width, so the auto-sized ellipse grew to wrap it. Dot uses 14pt at 96 DPI which renders with a smaller box, even after PDFs convert to raster.

Fix path: either reduce node padding margin to compensate for the larger font, or revisit the F1 font-size bump (it may have been DPI-overshooting — see notes below).

### HIGH-2: Edge label font size (F1 incomplete)
On arrow_types.png and edge_styles_showcase.png the edge-label text is visibly smaller than dot's. Estimated 10-12pt vs dot's ~14pt. The F1 fix reportedly bumped both node and edge labels to 16pt — node-side landed but edge-side did not (or used a different code path that wasn't touched).

### HIGH-3: F4 arrow inconsistency
Arrowheads are still not matching dot. Some panels too large (pipeline, colors_showcase, multi_cycle) and some too small (tiny_graph, single_edge). Suggests arrow-size logic might be coupled to edge length or graph DPI in a way that diverges from dot. Round 9's bump fixed average size but did not fix the variance.

### MED-1: Node stroke width
Dagua's ellipse outlines look ~0.7-0.8x the weight of dot's ellipse outlines on tiny_graph and pipeline. Dot's nodes have a more substantial pen weight. This is small but it adds to the "thinner / wispier" impression alongside HIGH-2.

### MED-2: Crow shape ambiguity
If brief is right that Dagua's crow should be a filled black dart, the fix didn't land — the rendered shape is a hollow V. If brief is wrong (i.e. the dart was over-shooting dot, and a hollow V is correct), we should update the brief. Recommendation: confirm what dot's crow actually is at a high zoom from a fresh dot rendering (it looked to me like a hollow V chevron in arrow_types.png), then either fix Dagua to match (filled dart) or close F2 with brief amended.

## New Issues from Round 9
1. Node padding/sizing — see HIGH-1 above. Almost certainly a side effect of the F1 font bump. Did not exist (or was much smaller) at round 8 because text was 12pt then. Round 9 traded "small text" for "puffy nodes."
2. Self_loop.png Dagua's nodes are notably bigger and the self-loop arrow is awkwardly placed (KNOWN_DEFERRED layout).

## User-Flagged Issues — Final Status

- "Arrows are wonky" — **PARTIALLY CLOSED.** Arrow shape proportions are vastly better than round 8 (no longer slim/pointy). But they're inconsistently sized (HIGH-3) and fatter on Dagua than on dot in several panels.
- "Text isn't centered" — **CLOSED.** Text is centered correctly inside nodes on every panel checked. (Wasn't flagged this round.)
- "Different font" — **CLOSED.** Font family/style appears the same serif as dot. Cap-height matches on node text. The remaining font issue is *size* for edge labels (HIGH-2), not family.
- "Cluster bounding boxes look like shit" — **CLOSED** (modulo H4/H5 layout-side overlap). F5 closed cleanly. Cluster borders are now appropriately ghosted; fill tone matches.

## Acceptable Residual

- Cluster-label-overlapping-nodes (H4/H5 layout-side, KNOWN_DEFERRED).
- Layout differences (state_machine, multi_cycle, nested_clusters, cluster_showcase, deep_nesting_4) — these are pipeline-side, not theme-side.
- Cluster border slightly *lighter* than dot's on some panels (Dagua at 0.7 opacity vs dot's effective ~0.85). Marginal under-shoot on F5 — within tolerance.
- Anti-aliasing micro-differences on curve segments — rendering-stack residual.

## Final Recommendation: **CONTINUE**

Round 9 fixed F2 visually (modulo brief ambiguity), F5 cleanly, and F4 directionally. But it traded one regression for another: the F1 font bump fixed node text size but introduced HIGH-1 (puffy nodes) which is a more salient gap than the original F1 was. And F1's edge-label half-implementation (HIGH-2) is also unclosed.

The bar is "genuinely identical save for documented rendering-stack residuals." We are not there. Side-by-side, a careful viewer can distinguish Dagua from dot on every panel within ~3 seconds, primarily because of:
1. node ellipses are noticeably bigger on Dagua (HIGH-1)
2. edge labels are noticeably smaller on Dagua (HIGH-2)
3. arrowheads are inconsistently sized (HIGH-3)

Recommended round 11 plan (3 fixes):
- **R11-A**: reduce node padding/margin to compensate for the 16pt font, OR drop node font from 16pt to ~14pt and re-measure. Goal: Dagua ellipse area within 5% of dot's at same text content.
- **R11-B**: track down where edge-label font_size is set and apply the same 16pt (or 14pt, matching whatever lands on R11-A) bump there too. Verify on arrow_types.png and edge_styles_showcase.png.
- **R11-C**: investigate whether arrowhead size is coupled to edge length or graph DPI. Make it constant per the dot reference (~12pt-equivalent absolute size), regardless of edge length.

After R11, re-audit. Expected to be the last cosmetic round.

## Confidence

- F1 verdict (PARTIAL): high — node-vs-edge size discrepancy is unambiguous on arrow_types and edge_styles
- F2 verdict (FAIL per brief): medium — depends entirely on whether the brief or the rendering captures the correct intent. The brief says "must be filled black dart, NOT hollow V" but dot's actual crow IS a hollow V. Recommend cross-check with maintainer.
- F3 verdict (PARTIAL): high — tiny_graph is the unambiguous panel
- F4 verdict (PARTIAL): high — multiple panels show different magnitudes of mismatch
- F5 verdict (PASS): high — uniform across all cluster panels
- HIGH-1 (puffy nodes): high — node_shapes_showcase is unambiguous
- HIGH-2 (edge label size): high — arrow_types and edge_styles are unambiguous
- HIGH-3 (arrow inconsistency): medium-high — three panels show oversize, two show undersize
