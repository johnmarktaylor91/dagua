# Cluster Sprint Final Visual Audit (post-Phase 6)

- Auditor: Opus 4.7 (1M context), maximally picky.
- Inputs:
  - `eval_output/parity_metrics.json` + `parity_metrics_summary.md`
  - `eval_output/parity_pixel_diff/summary.md`
  - Hi-res pairs (dot-positions harness): `eval_output/parity_pixel_diff/hires/<slug>/{dot,dagua}.png`
  - Three-way gallery (Dagua own placement): `eval_output/cluster_phase_6_check_dagua_placement/three_way/<slug>.png`
  - Three-way gallery (dot-positions harness): `eval_output/cluster_phase_6_check_dot_positions/three_way/<slug>.png`
- Cluster panels inspected (8): `nested_clusters`, `cluster_showcase`, `transformer_block`,
  `cross_cluster_edges`, `deep_nesting_4`, `microservices`, `data_pipeline`, `flat_many_clusters`.
- Image read budget: 16 of 16 used (8 hi-res pairs + 6 dagua-placement three-ways + 2
  reference for cross-checks).
- Critical context: I now have BOTH galleries available (Phase 6's `--use-dagua-placement`
  finally exercised). The two galleries tell very different stories and the divergence is
  itself the most important audit finding.

---

## Verdict

- Prior items (G1-G4 from `AUDIT_post_phase_5`):
  - G1 (deep_nesting_4 concentric nesting): PARTIAL. The render fix works on Dagua-placement;
    dot-positions harness still shows broken nesting because of bbox-cap interaction with
    overlapping sibling positions.
  - G2 (edge bodies clipped to stubs): PASS on `microservices` dot-positions render --
    Search Service -> Order DB / User DB / Redis Cache / Search Index now show full edge
    bodies. PASS on `deep_nesting_4` dot-positions -- chain edges have visible bodies.
  - G3 (cluster label fragmented by node fill): FAIL. `nested_clusters` dot-positions hi-res
    still shows "Outer Group" rendered as fragments "O...ap" with node A's white fill
    masking the middle. Phase 6 reported this fixed by raising label z-order above node
    fills; the rendered panel disagrees. Either the z-order shipped only on the
    Dagua-placement path, or A is still drawn with z>=label even with the new constant.
  - G4 (bypass edges through foreign clusters): PARTIAL. `transformer_block` hi-res still
    shows the FFN bypass curving along/through the FFN right side (visible as a tight curve
    that hugs / crosses the cluster border). Phase 6 implementer flagged this as
    near-border-sensitive and asked it stay on the next audit list.

- New audit: `PARTIAL` -- meaningful Phase 6 progress, but **two NEW HIGH-severity regressions
  surface in the dot-positions harness gallery** (paired-vertical-stroke "trough" pattern on
  five panels, plus G3 unresolved) and **a placement-quality concern is exposed by the
  Dagua-placement gallery** that prevents declaring cluster functionality bulletproof.

- Stop criteria status: `CONTINUE` -- there are findings classified as
  `real_cosmetic_gap` + `fixable_theme_or_render` (H1, H2 below). However, the user should
  weigh whether one more focused round is worth it, because a chunk of the remaining cluster
  ugliness on the harness gallery is harness-artifact, and a chunk on the Dagua-placement
  gallery is `needs_layout_scope`. Scope discussion in the "Honest answer" section.

The Phase 6 wins are real and substantial: when Dagua owns placement, `deep_nesting_4`
now renders four properly concentric rectangles; `nested_clusters` shows Left Branch and
Right Branch concentrically inside Outer Group; the cluster-aware driver no longer falls
back through `dagua_native`. The `microservices` edge-body fix (G2) is visible even on the
dot-positions harness, which is the strongest evidence Phase 6 delivered. But:

- The dot-positions harness gallery (the "default" parity_pixel_diff gallery) shows a
  HIGH-severity render regression I will call **H1: paired-vertical-stroke "[ ]" pattern**
  on `nested_clusters`/Right Branch+Left Branch, `transformer_block`/FFN,
  `cross_cluster_edges`/Cluster Y+Cluster Z, `data_pipeline`/Transform, and
  `flat_many_clusters`/Alpha+Beta+Gamma+Delta+Beta. Five of eight panels render at least
  one cluster as two un-joined vertical strokes (the cluster's TOP horizontal edge is
  missing). This is a real render bug, not just a harness artifact: the bottom edge of one
  cluster coincides with the top edge of another and the path-build collapses. The fix
  is in the render bbox cap interaction with sibling-overlap on dot-position input, but
  the render code IS Dagua's so this counts.
- The dot-positions harness gallery still shows G3 `nested_clusters` "Outer Group" label
  fragmenting through node A. Phase 6 reported G3 fixed; the panel disagrees.
- The Dagua-placement gallery shows that Dagua's cluster-aware FR placement is qualitatively
  worse than dot's hierarchical layout on the directed-flow cluster panels
  (`microservices`, `transformer_block`, `data_pipeline`, `cross_cluster_edges`,
  `flat_many_clusters`). Cluster rectangles draw correctly, but the inner node placement
  is congested and overlapping. Cluster *render* is fine; Cluster *placement* on hierarchical
  graphs is not at dot's level.

Honest summary up front: Phase 6 closed several Phase 5 regressions but revealed three
new structural gaps. We are closer than we were two phases ago, but not yet at the user's
bar of "bulletproof, at least as nice as graphviz."

---

## Prior Item Recheck

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| G1 deep_nesting_4 concentric nesting | PARTIAL | hires dagua.png shows Level 2/3/4 stacked into bottom of Level 1 with paired vertical strokes; `cluster_phase_6_check_dagua_placement/three_way/deep_nesting_4.png` shows 4 concentric rectangles correctly | Render fix is real; dot-position harness exercises a bbox-cap edge case. |
| G2 edge bodies as stubs | PASS | `parity_pixel_diff/hires/microservices/dagua.png` -- Search Service -> 4 DBs all have full visible edge bodies; `deep_nesting_4/dagua.png` -- Source -> Outer 1, Outer 2 -> Mid 1, etc all show visible edge bodies (no stub-arrowhead-only artefact). | Phase 6 segment composition fix landed and held. Regression test in `test_edge_cluster_clip.py` makes this a stable gain. |
| G3 cluster label fragmented by node fill | FAIL | `parity_pixel_diff/hires/nested_clusters/dagua.png` -- Outer Group label visible as fragments "O...ap" with the middle masked by node A's white fill | Phase 6 claimed `1.5 + depth*0.01` z-order makes labels above node fills, but on the dot-positions render the label still falls behind A's fill. Either the constant is still below node fill z, or A is being drawn after the label. Re-check `mpl.py` z-order ordering. |
| G4 bypass edges through foreign clusters | PARTIAL | `parity_pixel_diff/hires/transformer_block/dagua.png` -- bypass edge from Input Embedding to Add curves around MHA right cleanly; second bypass to Add hugs FFN right side and visibly crosses the FFN top + bottom strokes (right edge curves into FFN's interior) | Tested in unit test for source-out/target-out bypass; visual still shows residual on near-border cases. Implementer flagged this as remaining. |

---

## New Findings

| # | Severity | Panel | Element/Region | Finding | Finding Class | Actionability | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | HIGH | 5 of 8 cluster panels in dot-positions gallery | cluster TOP edge missing, leaving paired vertical strokes | `nested_clusters` Right Branch + Left Branch top edges missing (paired `[`-troughs); `transformer_block` FFN top edge missing; `cross_cluster_edges` Cluster Y + Cluster Z top edges missing; `data_pipeline` Transform top edge missing; `flat_many_clusters` Alpha + Beta + Gamma + Delta all show paired vertical strokes with no top horizontal edge. The bottom edge of one cluster coincides with where the top of the next cluster should be, and the path-build collapses both into one segment. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/<slug>/dagua.png on each panel above. Compare hires/<slug>/dot.png which draws all top edges cleanly. The Phase 5/6 render-bbox cap is clamping siblings tight to their footprint; when dot's positions place sibling clusters with no vertical airspace between them (common for harness-injected positions because dot leaves padding around the OUTER cluster, not between siblings), the clamp produces zero-height shared border that the path-builder treats as not-present. |
| H2 | HIGH | `nested_clusters` dot-positions hires | "Outer Group" label rendered as fragments "O...ap" | Phase 6 claimed `cluster_label_z = 1.5 + depth*0.01` raised labels above node fills. The rendered panel still shows the middle of "Outer Group" masked by node A's white fill. Either the constant was set but the wrong order is in effect at draw time, or the cluster label's bbox mask is drawn at z<node fill while only the text glyphs are at z>node fill. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/nested_clusters/dagua.png. Re-check the z-order between (a) cluster label background mask, (b) cluster label glyphs, (c) node fill, (d) node glyphs. If the mask is at z<fill, A's fill overwrites it, then the label glyphs land on top of A's fill and read as fragments. |
| H3 | HIGH | `microservices`, `transformer_block`, `data_pipeline`, `cross_cluster_edges`, `flat_many_clusters` (Dagua-placement gallery only) | Dagua's cluster-aware FR placement quality on hierarchical/flow graphs | All five hierarchical-flow cluster panels show Dagua's own placement pile nodes into a small area with heavy edge overlap. Cluster rectangles draw correctly (so renderer is fine), but the inner node positions are congested -- e.g. `transformer_block` puts all transformer block components into a dense ball, `data_pipeline` collapses the entire pipeline into a vertical near-line, `microservices` spreads layers but inner nodes still overlap. | `real_cosmetic_gap` | `needs_layout_scope` | `cluster_phase_6_check_dagua_placement/three_way/{microservices,transformer_block,data_pipeline,cross_cluster_edges,flat_many_clusters}.png`. The cluster-aware driver uses FR for inner placement; FR is not strong on directed-flow graphs. Sugiyama / hierarchical placement would be a closer fit. The Dagua-native pipeline does not yet support cluster-aware placement (Phase 6 noted this), so the harness falls through to FR. This is THE follow-up sprint, not a render fix. |
| H4 | MED | `nested_clusters` dot-positions | Node A pierces Outer Group top stroke; A's bottom half is inside the cluster bbox | Same as G3 prior finding. A is positioned outside the outer cluster by dot, but Dagua's render expands the bbox upward into A's space. This is harness artifact + render-padding interaction. | `real_cosmetic_gap` (layered on harness artifact) | `fixable_theme_or_render` for the bbox upward-expansion limit; `needs_layout_scope` for the placement | hires/nested_clusters/dagua.png. The bbox upper expansion is exceeding what dot intended. Cap upward expansion to (dot-position cluster top - half-label-band) to keep the node A clearance dot has. |
| H5 | MED | `data_pipeline` dot-positions | "Data Warehouse" extends below Load cluster bottom | F9/G11 from prior audits. Still visible: the Load cluster bottom edge cuts through the Data Warehouse ellipse near its lower third. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/data_pipeline/dagua.png bottom of Load cluster. Member bbox max-y missing half-height of bottom-row ellipses. Phase 6 changelog does not list this as fixed. |
| H6 | MED | `microservices` dot-positions | Order Service / User Service partially clipped by Service Layer left edge; Notification Service bottom outside Service Layer | Order Service ellipse left half is OUTSIDE Service Layer bbox; User Service partially same; Notification Service ellipse extends below Service Layer bottom edge. Containment failure on three of five Service Layer members. | `real_cosmetic_gap` (layered on harness artifact) | `needs_layout_scope` for placement; `fixable_theme_or_render` for bbox padding to fully contain members | hires/microservices/dagua.png. |
| H7 | MED | `cluster_showcase` dot-positions | "outer a" partially outside Outer Cluster top | outer a sits with its top half above the Outer Cluster top stroke. Same as G5 prior. Harness artifact + render padding. | `real_cosmetic_gap` (layered on harness artifact) | `fixable_theme_or_render` (cap top expansion) or `needs_layout_scope` | hires/cluster_showcase/dagua.png upper-left. |
| H8 | MED | `transformer_block` dot-positions | FFN bypass edge crosses FFN body twice | Bypass edge from Add to second Add hugs FFN right side and crosses the cluster body twice (entering on right edge top area, exiting on right edge bottom area). Phase 6 segmented bypass implementation handles source-out/target-out cases but this near-border curve still reads as crossing. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/transformer_block/dagua.png right side. Increase tolerance for "edge segment lies within cluster interior" so near-border curves register as crossings, OR push routing outside cluster perimeter by half-stroke-width before clipping. |
| H9 | MED | `cross_cluster_edges` dot-positions | Cluster Y bbox overlaps Cluster Z bbox; Y3, Y4 on the boundary line | Cluster Y right edge cuts through Y2/Y4; Y3 between Y bottom and Z top. Same as G6 prior, dominated by harness artifact. | `real_cosmetic_gap` (harness artifact dominant) | `needs_layout_scope` for the placement; `not_actionable` in this harness | hires/cross_cluster_edges/dagua.png. The Dagua-placement gallery also shows this poorly (H3) so it's not "harness only" -- both galleries suffer. |
| H10 | LOW | `cluster_showcase` dot-positions | Tiny Cluster too tight on small 0 | Tiny Cluster top stroke sits very close to small 0 ellipse top -- no breathing space for the label band. Same as G9 prior. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/cluster_showcase/dagua.png top-right. |
| H11 | LOW | `cluster_showcase` | Tiny Cluster small 1 sits below Tiny Cluster bottom edge | small 1 is positioned below Tiny Cluster bbox bottom (containment failure on a single-column 2-node cluster). | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/cluster_showcase/dagua.png Tiny Cluster lower-left. |
| H12 | LOW | universal cluster panels | node ellipses thicker / more circular than dot's | Persistent across cluster panels; out of cluster scope. | `metric_or_measurement_artifact` for cluster scope | `not_actionable` | parity_metrics summary `ellipse_rx_pt` 95.28%, max delta 13.4pt. |
| H13 | LOW | `flat_many_clusters` dot-positions | sibling cluster labels top-center vs dot's top-left | Dot puts each sibling cluster label flush-left ("Alpha", "Beta", "Gamma", "Delta" anchored at top-left of each cluster); Dagua puts them top-center. Same as F12 prior. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/flat_many_clusters/dagua.png. |

Severity tallies: 3 HIGH, 6 MED, 4 LOW.

---

## Metric Artifact Review

- All declarative cluster features (`cluster_fill`, `cluster_stroke`,
  `cluster_stroke_width_pt`, `cluster_label_font_size_pt`, `cluster_rect_missing`) report
  41/41 in tolerance (100.00%). The new `cluster_rect_missing` metric correctly catches the
  prior-audit F1/F2/F3 cases now that they are fixed -- this metric is doing real work and
  is now a green guardrail.
- The `cluster_rect_missing` metric does NOT catch H1 (paired-vertical-stroke "trough"
  pattern). It only registers when the entire bbox is missing, not when only the top edge
  is missing. The five panels with broken top edges all read as 0 missing rectangles,
  which is wrong. **Recommend extending the metric to count cluster rectangles where
  fewer than 4 of 4 edges are within tolerance distance of expected, not just bbox
  presence.**
- Pixel-diff says cluster panels remain in the lower-mid of SSIM. Cluster panel SSIMs:
  `nested_clusters` 0.697 (rank 7 worst overall), `cross_cluster_edges` 0.725,
  `flat_many_clusters` 0.742, `deep_nesting_4` 0.745, `data_pipeline` 0.754,
  `cluster_showcase` 0.819, `microservices` 0.850, `transformer_block` 0.852. Same
  ordering as post-Phase-5; SSIM moved by tenths of a percent. Pixel-diff is barely
  registering Phase 6's progress because most of the wins are on the Dagua-placement
  path that the parity gallery doesn't render.
- Background L1 still dominates total L1; Text/Node/Edge regions still report 0.0 across
  the board because the masks make those regions invisible to the metric. Cluster
  label-stroke collisions therefore land entirely in Background L1. This is a measurement
  blind spot and is documented as such in prior audits.

---

## Rendering-Stack Residuals

- Dot's font hinting / rasterizer anti-aliasing on stroke endpoints: not driving any of the
  findings above.
- One-pixel sub-pixel rounding on cluster corners: not driving the structural defects.
- Bezier routing geometry inherited from Graphviz positions: applies to the dot-positions
  gallery; H4, H6, H7 carry harness-artifact disclaimers. The Dagua-placement gallery
  removes this residual, but introduces H3 (placement quality on hierarchical-flow graphs).
- Dagua's render-time bbox upward-expansion exceeding dot's clearance for outside-cluster
  members (H4/H7) is not a residual, it is a fixable cap.

---

## Recommended Next Fixes

Ranked by user-visible impact, only `real_cosmetic_gap` + `fixable_theme_or_render`:

1. **H1 -- cluster top edge missing on 5 of 8 panels (dot-positions gallery).**
   The render-bbox cap added in Phase 5 + parent-containment in Phase 6 collapse a
   sibling cluster's top edge to zero height when adjacent siblings or parents share
   that y-coordinate. Repro: `flat_many_clusters` Alpha + Beta on dot's positions --
   their tops are aligned to the same y as Source's bottom, and the cap clamps them
   without leaving room for distinct top strokes. Two possible fixes:
   - Reserve a minimum 1-stroke-width vertical gap between any two cluster bbox edges
     before the path-build, even at the cost of moving rectangles slightly outside the
     placement.
   - Always emit all four edges of a closed roundrect in the path-build, even when the
     edge length is below stroke-width. The render is willing to draw a 1-pixel-tall
     rectangle; the path-builder is incorrectly treating it as degenerate.
   Code area: cluster bbox containment + path-build in `dagua/render/mpl.py` (likely
   the same function that the Phase 5+6 cap was added to).

2. **H2 -- "Outer Group" label fragmented behind node A fill.**
   Phase 6 set `cluster_label_z = 1.5 + depth*0.01` to put labels above node fills. The
   rendered panel disagrees. Trace the actual matplotlib z-orders for: (a) cluster
   roundrect path, (b) cluster label background mask, (c) cluster label text, (d) node
   fill patch, (e) node label text. The fragmentation pattern is: label glyphs at z high,
   label mask at z low, node fill at z medium. Setting only the glyph z high, but
   leaving the mask at low, leaves the mask buried under the node fill so the fill paints
   directly under the glyphs.
   Code area: `mpl.py` cluster label drawing block; verify z is set on BOTH the mask
   patch and the Text artist.

3. **H3 -- Dagua cluster-aware placement quality on hierarchical-flow graphs.**
   The cluster-aware driver currently routes through FR (Fruchterman-Reingold) for inner
   node placement. FR is force-directed; on a directed-flow graph (`transformer_block`,
   `data_pipeline`) it produces a ball, not a flow. The dot-quality experience is from
   Sugiyama-style hierarchical placement.
   Concrete options:
   - Add `algorithm="sugiyama"` support to the cluster-aware driver. (Phase 7 work.)
   - For graphs with a clear directional hint (`flowchart` rendered_label or DAG with
     few cycles), default the cluster-aware inner algorithm to `sugiyama`.
   - Document that `--use-dagua-placement` requires algorithm="fr" and is intended for
     undirected/symmetric graphs until cluster-aware Sugiyama lands.
   Code area: `dagua/layout/cluster_aware/driver.py` (or wherever Phase 2's driver
   landed); check the algorithm dispatch.

4. **H4 / H7 -- bbox upward expansion exceeds dot's outside-cluster clearance.**
   Cap the render-time upward expansion of a top-level cluster's bbox at
   (dot-position-top - half-label-band) when a node is placed above the cluster.
   Currently the bbox expands upward to wrap the label band even into the area dot
   reserved for the outside node.
   Code area: `mpl.py` cluster bbox upward expansion limit.

5. **H8 -- bypass edge near-border crossing on `transformer_block` FFN.**
   Phase 6's segmented bypass works when the bypass enters and exits at clearly
   non-tangent points. When the curve runs along the cluster border (one extreme
   tangent + one barely-inside crossing), the segmenter does not detect a body-inside
   segment. Either widen the "inside cluster body" tolerance to half-stroke-width, or
   re-route the body to clear the border by stroke-width before clipping.
   Code area: same as Phase 6 segmented-body code.

6. **H5 -- "Data Warehouse" below Load cluster bottom on `data_pipeline`.**
   Member bbox max-y missing half-height of bottom-row ellipses. Same fix proposed in
   prior audits: `+ node_h/2 + bottom_padding` on the member-extent computation.
   Code area: `mpl.py` cluster bottom padding.

7. **H6 -- Service Layer member containment failures on `microservices` (harness mode).**
   Order Service / User Service / Notification Service partially outside Service Layer
   bbox. Two things to verify: (a) is the bbox using member-extent or member-center for
   left/bottom edges? (b) is the harness's dot-position injection respecting cluster
   member bounds?
   Code area: `mpl.py` cluster bbox member-extent computation.

8. **(Audit infra) Extend `cluster_rect_missing` metric to count edges-of-bbox, not bbox
   presence.**
   Today the metric reports 41/41 in tolerance and the rendered gallery has 5 of 8
   panels with at least one cluster rectangle missing its top edge. Metric is a guardrail
   and should catch H1. Walk the rendered cluster path and count distinct horizontal +
   vertical segments; require >= 4 within tolerance.
   Code area: `parity_metrics.py` cluster feature extraction.

9. **H13 -- sibling cluster label position should be top-left in graphviz themes.**
   Per multiple prior audits: outer cluster gets top-center, sibling clusters get
   top-left. Encode as a depth-conditional default. Low priority but trivial fix.
   Code area: `dagua/styles.py` Theme default; or `mpl.py` label-anchor selection.

10. **H10 / H11 -- Tiny Cluster top tightness + small 1 outside Tiny Cluster bottom.**
    Single-column small clusters not honoring label-band reservation + bottom padding.
    Code area: `mpl.py` cluster top/bottom padding for `len(members) <= 2`.

---

## Inspection Log

For each panel: nodes, labels, edges, arrowheads, cluster borders + labels + masks, and
worst-metric regions inspected on both galleries where available.

- **`nested_clusters`** (SSIM 0.697, L1 23.98, rank 7-worst overall):
  - dot-positions hires: outer cluster border drawn complete; inner Right Branch / Left
    Branch with TOP edges missing (H1 paired vertical strokes); A pierces outer top stroke
    (H4); "Outer Group" label fragments through A (H2); A->C and A->B arrowheads on outer
    top stroke (G10 from prior audit, dependent on H4); C->E and B->D edge bodies very
    short (compressed inner cluster height); F outside outer cluster bottom; ellipses
    notably more circular than dot (H12).
  - Dagua-placement three-way: Outer Group drawn complete; Left Branch + Right Branch
    drawn concentrically with all four edges; A above Outer Group cleanly; F below cleanly.
    Phase 6 G1 fix is real on this path.

- **`cluster_showcase`** (SSIM 0.819, L1 13.56):
  - dot-positions hires: all five clusters drawn including Large Cluster (F2/F3 PASS);
    labels masked (F4/G3 PASS on cluster_showcase specifically); outer a partially outside
    Outer Cluster top (H7); Tiny Cluster too tight on small 0 (H10); small 1 outside Tiny
    Cluster bottom (H11); Outer Cluster contains Nested Inner concentrically.
  - Dagua-placement three-way: cluster rectangles render correctly but layout is messier
    than dot (H3 -- placement quality, not render quality).

- **`transformer_block`** (SSIM 0.852, L1 11.31):
  - dot-positions hires: MHA cluster border complete with clean label; FFN cluster TOP
    edge MISSING (H1 paired vertical strokes); endpoint arrowheads inside MHA clip cleanly
    to perimeter (G2 fix held); Input Embedding bypass curves around MHA right cleanly;
    second bypass to Add curves along/through FFN right side (H8 / G4 partial); inner MHA
    layout OK.
  - Dagua-placement three-way: clusters drawn but inner layout is dense/overlapping (H3).

- **`cross_cluster_edges`** (SSIM 0.725, L1 19.70):
  - dot-positions hires: Cluster X drawn complete; Cluster Y top edge MISSING (H1);
    Cluster Z top edge MISSING (H1); Y3, Y4 sit on Y/Z boundary (containment ambiguous);
    cluster labels masked.
  - Dagua-placement three-way: clusters drawn but Y/Z severely overlap; X1 floats outside
    Cluster X (H3 placement issue). This panel suffers BOTH harness and placement issues.

- **`deep_nesting_4`** (SSIM 0.745, L1 17.76):
  - dot-positions hires: Levels 1..4 drawn but as semi-stacked into the bottom of Level 1
    rather than concentric; Level 3 has paired vertical strokes (H1); Level 4 is inside
    Level 3 cleanly; Level 2 has visible top edge (good); edge bodies between Source ->
    Outer 1, etc are visible (G2 fix held); cluster labels mask cleanly.
  - Dagua-placement three-way: 4 properly concentric rectangles, label band cleanly above
    each level, but Source/Outer 2 outside Level 1 (placement places them outside, dot
    has them inside Level 1). Phase 6 G1 fix is real on this path.

- **`microservices`** (SSIM 0.850, L1 11.88):
  - dot-positions hires: 4 cluster boxes drawn with clean labels (no missing top edges
    on this panel because siblings have airspace); Search Service -> Order DB / User DB /
    Redis Cache / Search Index all show full edge bodies (G2 PASS, biggest Phase 6 win
    visible here); Order Service / User Service / Notification Service partial containment
    failures (H6); endpoint clipping engages on entries/exits.
  - Dagua-placement three-way: cluster rectangles draw correctly but heavy overlap
    between Service Layer and Data Layer; Order DB at top with arrow into it from
    "outside" the Service Layer; layout is congested (H3).

- **`data_pipeline`** (SSIM 0.754, L1 18.68):
  - dot-positions hires: Extract drawn complete; Transform top edge MISSING (H1); Load
    drawn complete; Data Warehouse extends below Load bottom (H5); cluster labels mask
    cleanly; edges from Sources to extractors land at Extract top correctly; many edges
    from Query DB / Fetch API into Transform members.
  - Dagua-placement three-way: clusters drawn but inner layout collapses to a near-line
    (H3 placement).

- **`flat_many_clusters`** (SSIM 0.742, L1 19.18):
  - dot-positions hires: ALL 4 sibling clusters (Alpha, Beta, Gamma, Delta) show paired
    vertical strokes with no top horizontal edge (H1 affects all 4); B4 outside Beta
    bottom; D4/D5 outside Delta bottom; sibling cluster labels top-center vs dot's
    top-left (H13); Source -> A1/B1/C1/D1 edges visible.
  - Dagua-placement three-way: clusters drawn but heavily collapsed (H3).

Pixel-diff reality check: 5 of 7 evaluated cluster panels remain below mean SSIM (0.761).
The two panels above mean are `microservices` (0.850) and `transformer_block` (0.852).
SSIM has barely moved between Phase 5 and Phase 6 because the Dagua-placement gallery is
not part of the parity_pixel_diff harness; Phase 6's largest wins (proper concentric
nesting, no `dagua_native` fallback) live on a path the parity metric doesn't see.

---

## Honest answer to user's bar

> "make our cluster functionality bulletproof with this sprint!! it should look at least
> as nice as graphviz and its a CORE FEATURE."

**Closer, but NOT bulletproof.** The Phase 6 wins are real and substantial: when Dagua
owns placement, `deep_nesting_4` shows four properly concentric rectangles, `nested_clusters`
shows Left Branch and Right Branch concentric inside Outer Group, the cluster-aware driver
runs without the `dagua_native` fallback warning, and the visible `microservices` edge-body
collapse from Phase 5 is fixed and tested. Those are bankable improvements that will not
regress.

But three meaningful gaps remain:

1. **The dot-positions parity gallery (the user's default visual review surface) shows a
   HIGH-severity render regression on 5 of 8 cluster panels: cluster top edges collapse to
   paired vertical strokes** when sibling clusters share y-coordinates. This is a real
   render bug, not just harness artifact -- the path-build is treating zero-height edges
   as degenerate, but the actual rendered output should still draw a 1-pixel-tall
   rectangle. Fixable in one focused change to the cluster path-builder.
2. **`nested_clusters` "Outer Group" label is still fragmented by node A's white fill**,
   contradicting Phase 6's claim that label z-order was raised above node fills.
   Either the change shipped only on the Dagua-placement code path, or only the glyph z
   was raised while the mask remained below the node fill. Fixable in one focused change.
3. **Dagua's cluster-aware placement on hierarchical-flow graphs is qualitatively worse
   than dot's**, because the inner placement algorithm is FR (force-directed) and dot's
   placement is hierarchical / Sugiyama. This is a placement-driver feature gap, not a
   render bug. The cluster RECTANGLES draw correctly under Dagua placement, but the
   inner node positions read as a ball or near-line on `transformer_block`,
   `data_pipeline`, `cross_cluster_edges`, `flat_many_clusters`. A user asking "does
   this look as nice as graphviz on hierarchical clusters" would say no. Fix is a Phase
   7 or later feature: add Sugiyama-style support to the cluster-aware driver.

If the user accepts that:
- (1) and (2) get fixed in a tight Phase 7 corrective round (1-2 days, similar in scope to
  Phase 5 and Phase 6 corrective rounds);
- (3) is scoped as a separate cluster-aware-Sugiyama sprint (this is real new layout
  algorithm work, not a corrective fix);

then the cluster sprint can land at "rendering bulletproof, placement on directed-flow
needs follow-up." That is an honest place to call it.

If the user wants "as nice as graphviz on every cluster panel" before declaring done, then
both (1)+(2) AND (3) are required. (3) is the bigger lift by a wide margin.

CONTINUE recommended for one more focused render round to close H1, H2, H4, H7, H8 and
extend the `cluster_rect_missing` metric to detect missing top edges. Sugiyama under
clusters is a separate decision.

The user was explicit that clusters are a CORE FEATURE. The render-side work has come a
very long way in six phases. The placement-side work for clustered hierarchical graphs
is essentially un-done at the cluster-aware-algorithm level. That is the right answer to
deliver back instead of soft-pedaling.
