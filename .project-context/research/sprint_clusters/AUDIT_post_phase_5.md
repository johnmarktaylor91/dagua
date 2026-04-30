# Cluster Sprint Post-Phase-5 Visual Audit

- Auditor: Opus 4.7 (1M context), maximally picky.
- Inputs:
  - `eval_output/parity_metrics.json` + `parity_metrics_summary.md`
  - `eval_output/parity_pixel_diff/summary.md`
  - Hi-res pairs: `eval_output/parity_pixel_diff/hires/<slug>/{dot,dagua}.png`
- Cluster panels inspected (7): `nested_clusters`, `cluster_showcase`, `transformer_block`,
  `cross_cluster_edges`, `deep_nesting_4`, `microservices`, `data_pipeline`.
- Image read budget: 14 of 14 used.
- Critical context: The harness `scripts/graphviz_theme_comparison.py` STILL renders Dagua with
  Graphviz's positions injected (the `--use-dagua-placement` flag that Phase 5 added was NOT
  used to regenerate the audit images). Therefore sibling-cluster overlap and member-spillover
  observations below are downstream of dot's positions filtered through Dagua's bbox/padding
  rules. Phase 2 placement gains do NOT show in this gallery and have to be probed separately.

---

## Verdict

- Prior items (F1-F6 from AUDIT_post_phase_4): F1 PASS, F2 PASS, F3 PASS, F4 PARTIAL (endpoint
  clip works; bypass case still open and is a Phase 6 candidate per the report), F5 PASS
  (mechanism + flag landed; gallery did not exercise it), F6 PASS (metric exists, 41/41).
- New audit: `PARTIAL`.
- Stop criteria status: `CONTINUE` -- there are NEW `real_cosmetic_gap` +
  `fixable_theme_or_render` regressions (G1, G2, G3, G6, G7) and at least one prior
  `real_cosmetic_gap` (F4 bypass case, now G4) that is not yet bulletproof. The cluster
  sprint is closer to dot but is not yet "at least as nice as graphviz" on `deep_nesting_4`,
  `nested_clusters`, or `microservices`.

The visible Phase 5 wins are real and substantial: missing inner rectangles are gone on
`nested_clusters`; the "Large Cluster With Longer Label" rectangle now draws on
`cluster_showcase`; cluster labels uniformly mask the top stroke; endpoint arrowheads on
`transformer_block`/`microservices`/`data_pipeline` clip to cluster perimeters where they
should. But Phase 5 introduced two regressions worth flagging (G1 nested concentric collapse
on `deep_nesting_4`, G2 over-eager edge body trimming on `microservices`/`deep_nesting_4`)
and the bypass clipping is still missing (G4 was F4 partial). Honest answer: closer, not yet
bulletproof.

---

## Prior Item Recheck

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| F1 nested_clusters inner rects missing | PASS | hires/nested_clusters/dagua.png | Right Branch / Left Branch boxes are now closed rectangles. |
| F2 cluster_showcase Large rectangle missing | PASS | hires/cluster_showcase/dagua.png | "Large Cluster With Longer Label" rectangle draws cleanly. |
| F3 label mask undersized -> top stroke through label | PASS | hires/cluster_showcase, microservices, data_pipeline, transformer_block, deep_nesting_4 dagua.png | All cluster labels show clean cutouts of the top stroke. The (6,4) padding plus `Text.get_window_extent()` is sufficient on the panels inspected. |
| F4 Phase 4 endpoint clipping not engaging | PARTIAL | hires/transformer_block, microservices, data_pipeline dagua.png | Endpoint clipping engages on entries/exits where source and target sit in DIFFERENT clusters and the edge is not a bypass. Bypass edges (Input Embedding -> Add around MHA on transformer_block) still cross foreign cluster bodies un-broken. Continues as G4 below. |
| F5 render bbox bloat / harness can use Dagua placement | PASS for the mechanism | mpl.py bbox cap + harness flag landed | The audit gallery itself was generated with the OLD harness mode (dot positions injected into Dagua), so siblings still overlap visually. Mechanism for the fix is in place. |
| F6 cluster_rect_missing metric | PASS | parity_metrics_summary.md | 41/41 in tolerance; the metric reports zero missing rectangles, which the rendered gallery now agrees with. |

---

## New Findings

| # | Severity | Panel | Element/Region | Finding | Finding Class | Actionability | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | HIGH | `deep_nesting_4` | nested cluster rectangles (Level 1..4) | Levels are drawn as STACKED ADJACENT rectangles instead of CONCENTRIC. Dot draws Level 2 INSIDE Level 1, Level 3 inside Level 2, etc. Dagua draws four rectangles end-to-end vertically -- Level 1 above Level 2 above Level 3 above Level 4. Inner rectangles also show as paired vertical strokes ("[" trough shapes) with top/bottom edges nearly invisible because they coincide with the parent's bottom/top stroke. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/deep_nesting_4/dot.png vs dagua.png. The render-time bbox cap (Phase 5 F5) likely clamps each child to its own placement footprint without re-expanding to fully wrap inside its parent; for deep nesting this collapses the concentric structure. Code area: mpl.py cluster bbox computation + ancestor-aware width/height expansion. |
| G2 | HIGH | `deep_nesting_4`, `microservices` | edge bodies between nodes that span clusters | Edge bodies are clipped TO STUBS while the arrowhead remains. On `deep_nesting_4` Source->Outer 1, Outer 2->Mid 1, Mid 2->Inner 1, Inner 2->Core, Core->Exit, Exit->Sink all show only a tiny arrowhead with essentially no visible body line. On `microservices` Search Service -> {Order DB, User DB, Redis Cache, Search Index} render as four stub arrowheads stacked at the top of Data Layer. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/deep_nesting_4/dagua.png; hires/microservices/dagua.png. Phase 4 endpoint clip likely trims back to perimeter then re-clips again at target node bbox; with concentric clusters the two clips overlap and chew the body. Code area: edges clip pass; need to clip ONCE per (cluster, node) intersection in order, not iteratively. |
| G3 | HIGH | `nested_clusters` | node A vs outer cluster top stroke; "Outer Group" label fragments | Node A still pierces the Outer Group top stroke -- A's bottom half is inside the cluster bbox. The "Outer Group" label is rendered as fragments "O...ap" because A's white fill is masking the middle of the label text. (Harness uses dot positions; this is partially harness artifact, but the visible label fragmentation is a render-time z-order / mask problem inside Dagua.) | `real_cosmetic_gap` | `fixable_theme_or_render` for the label-vs-node z-order; `needs_layout_scope` for the placement of A relative to Outer Group | hires/nested_clusters/dagua.png. Cluster label should be drawn ABOVE node fills along its mask path or its mask should also extend through overlapping node fills. Re-run with `--use-dagua-placement` to confirm placement piece. |
| G4 | HIGH | `transformer_block`, `data_pipeline` (Query DB lines) | bypass edges crossing FOREIGN cluster bodies | Bypass edge from Input Embedding to first Add is routed around MHA's right side cleanly, but then the second bypass (Add to second Add or to FFN-bypass) crosses the Feed-Forward Network cluster's right edge twice (entry top, exit bottom) without a perimeter break. On `data_pipeline` the Query DB -> Validate/Normalize/Deduplicate/Enrich lines arguably cross Transform's top edge cleanly (ok), but the multi-fanout bypass edges from Source nodes traverse Extract before reaching Transform without a break at Extract's bottom. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/transformer_block/dagua.png right side; hires/data_pipeline/dagua.png. This is the Phase 6 work the implementer flagged: edge collection has one continuous body; need segmented/gapped body or perimeter-aware reroute. |
| G5 | MED | `cluster_showcase` | "outer a" containment failure | Node "outer a" is partially OUTSIDE the Outer Cluster bbox (its right half sits beyond the cluster's right edge). The Outer Cluster border bisects "outer a" near the midline. Dot draws "outer a" comfortably inside Outer Cluster. | `real_cosmetic_gap` (layered on harness artifact) | `needs_layout_scope` (Phase 2 verify); `fixable_theme_or_render` for any bbox-padding contribution | hires/cluster_showcase/dagua.png upper-left of the Outer Cluster region. Same harness caveat as F2/F11 in prior audit: re-run with `--use-dagua-placement`. |
| G6 | MED | `cross_cluster_edges` | cluster X right side overlaps cluster Y top-right | Cluster X bbox right edge passes RIGHT THROUGH the X3, X4 nodes (which are members of Cluster X) and visually crosses the top-right corner of Cluster Y. Cluster Y's right side passes through the same X3/X4 nodes. The result: X3 and X4 sit on the boundary line of both clusters, and Cluster X's right border is INSIDE its own bbox content. | `real_cosmetic_gap` (harness artifact dominant) | `needs_layout_scope` for the placement; the render bbox itself is tight, so `not_actionable` in this harness | hires/cross_cluster_edges/dagua.png. Re-run with `--use-dagua-placement` before deciding render-side fix. |
| G7 | MED | `microservices` | "Order Service" sits on Service Layer left edge | "Order Service" is positioned at the very left of the Service Layer bbox -- the cluster's left stroke runs through the node's left edge. Dot keeps a clear gap. (Harness artifact, but worth confirming Dagua-placement does not reproduce it.) | `real_cosmetic_gap` (harness artifact dominant) | `needs_layout_scope`; minor `fixable_theme_or_render` if Dagua's render padding could be widened | hires/microservices/dagua.png left edge of Service Layer. |
| G8 | MED | `cluster_showcase`, `nested_clusters` | sibling cluster gap is too tight | Right Branch and Left Branch (in nested_clusters) sit so close that their bounding boxes nearly touch (visible thin gap, dot has a clear airy gap). On `cluster_showcase` the Outer Cluster and Large Cluster regions overlap entirely. | `real_cosmetic_gap` (harness artifact dominant on cluster_showcase, mostly render-side on nested_clusters) | `fixable_theme_or_render` for sibling padding; `needs_layout_scope` to fully verify | hires/nested_clusters/dagua.png inner siblings. |
| G9 | MED | `cluster_showcase` | "Tiny Cluster" border crops `small 0` | Tiny Cluster bbox is so tight that the `small 0` ellipse top sits ON the cluster's top stroke (no breathing space between node top and cluster top stroke). Dot leaves a comfortable label-band gap. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/cluster_showcase/dagua.png top-right of "Tiny Cluster". The label-band reservation is not honoring node-extent on this small cluster. |
| G10 | LOW | `nested_clusters` | A->C and A->B edges overlap the outer cluster top stroke | The arrowheads of A->C and A->B land ON the outer-cluster top stroke (near where A intersects). Because A is mis-positioned (G3), the edges look truncated. Once G3 is resolved, recheck. | `uncertain_needs_targeted_probe` | dependent on G3 | hires/nested_clusters/dagua.png. |
| G11 | LOW | `data_pipeline` | "Data Warehouse" extends below Load cluster bottom | The Load cluster bottom edge cuts through Data Warehouse near the lower third of the ellipse. F9 from prior audit -- still visible. | `real_cosmetic_gap` | `fixable_theme_or_render` | hires/data_pipeline/dagua.png. Member bbox max-y is missing the bottom half of the ellipse. |
| G12 | LOW | universal cluster panels | node ellipses thicker / more circular than dot's | Persistent across cluster panels (and elsewhere). Out of cluster scope per metric: `ellipse_rx_pt` max delta 13.4pt. Not driving cluster regressions. | `metric_or_measurement_artifact` for cluster scope | `not_actionable` here | parity_metrics_summary.md row 1. |

Severity tallies: 4 HIGH, 5 MED, 3 LOW.

---

## Metric Artifact Review

- All declarative cluster features (`cluster_fill`, `cluster_stroke`,
  `cluster_stroke_width_pt`, `cluster_label_font_size_pt`, `cluster_rect_missing`) report
  41/41 in tolerance. The new `cluster_rect_missing` metric correctly catches the
  prior-audit F1/F2 cases now that they are fixed -- this is a real instrument improvement.
  But the metric STILL does not see G1 (concentric -> stacked), G2 (edge body chewed
  to stubs), G4 (bypass crossing), or G3 (label fragmented by node fill). Those are
  visible in the rendered gallery and not in any declarative feature.
- Pixel-diff says cluster panels remain in the lower-mid of SSIM:
  `nested_clusters` 0.697 (rank 7 worst), `cross_cluster_edges` 0.725, `flat_many_clusters`
  0.742, `deep_nesting_4` 0.745, `data_pipeline` 0.754, `cluster_showcase` 0.819,
  `microservices` 0.850, `transformer_block` 0.852. `nested_clusters` and `deep_nesting_4`
  remain in the worst quartile. SSIM moved up on `transformer_block` (0.852 vs prior 0.851)
  and `nested_clusters` (0.697 vs prior 0.693) but did not move much on the others.
- All region L1 readings (Text/Node/Edge/Background) still report 0 for Text/Node/Edge --
  the masks make those regions invisible to the metric, so cluster-label-vs-stroke
  collisions are still measured only in Background L1. Worth fixing in the audit
  instrumentation, but not driving these findings.

---

## Rendering-Stack Residuals

- Dot's font hinting / rasterizer anti-aliasing on stroke endpoints: not driving any of the
  findings above.
- One-pixel sub-pixel rounding on cluster corners: not driving the structural defects.
- Bezier routing geometry inherited from Graphviz positions: this gallery uses dot positions
  for Dagua, so all sibling-overlap and member-spillover cases (G5, G6, G7, G8) carry a
  harness-artifact disclaimer. The implementer added `--use-dagua-placement` for exactly
  this purpose; running the next audit gallery with it would isolate placement vs render.

---

## Recommended Next Fixes

Ranked by user-visible impact, only `real_cosmetic_gap` + `fixable_theme_or_render`:

1. **G1 -- concentric-not-stacked nested cluster rendering on `deep_nesting_4`.**
   The render bbox cap added in Phase 5 F5 must respect ancestor-cluster footprints. A
   child cluster's render bbox should be EXPANDED (not clamped down) to fit fully
   inside its parent's interior bbox after subtracting the parent's label-band and
   bottom padding. Likely root cause: the cap clamps ALL clusters to their own member
   footprint + 2pt; nested children whose siblings include the parent itself end up
   placed sequentially.
   Code area: `dagua/render/mpl.py` cluster bbox computation; likely the function that
   added the Phase 5 footprint cap.

2. **G2 -- edge body clipped to stubs.**
   The endpoint clip and the cluster perimeter clip are running independently and the
   composition leaves only a tiny segment between them. Trace one
   `microservices` Search Service -> Order DB edge end-to-end. Expected: clip body so
   it RUNS from source-node perimeter, exits Service Layer perimeter, enters Data
   Layer perimeter, lands at Order DB perimeter. Observed: body collapses to ~1pt
   stub plus arrowhead at Order DB top.
   Code area: edge polyline clipping in `dagua/render/mpl.py` or wherever Phase 4
   clipping was implemented.

3. **G4 -- bypass edges through foreign clusters need segmented bodies (Phase 6).**
   Implementer already flagged this. A bypass edge that ENTERS and EXITS a foreign
   cluster needs either (a) a multi-segment body with a gap inside the foreign
   cluster, or (b) a re-routed body that bends around the foreign cluster perimeter.
   Default to (a) since (b) requires a routing pass.
   Code area: edge collection structure (currently one continuous body curve); add
   support for multi-segment bodies; modify clip pass to insert gap segments where
   the body intersects a foreign cluster's interior.

4. **G3 -- cluster label z-order vs overlapping node fill.**
   When a node's bbox overlaps a cluster's label band, the node's white fill currently
   masks the middle of the label text. Either (a) draw the cluster label AFTER nodes
   in z-order (with the existing top-stroke mask still in place), or (b) extend the
   label background mask to also cover the overlapping node region.
   Code area: render z-order for cluster labels vs node fills.

5. **G9 -- "Tiny Cluster" cropping `small 0`.**
   Small clusters with one node + label-band are not honoring the label-band reservation;
   the render bbox starts at node-top instead of node-top + label-band.
   Code area: `dagua/render/mpl.py` cluster top-padding when `len(members) == 1`.

6. **G11 -- "Data Warehouse" extends below Load cluster bottom.**
   Member bbox max-y missing half-height of bottom-row ellipses. Add `+ node_h/2 +
   bottom_padding` to the member-extent computation for the bottom edge.
   Code area: `dagua/render/mpl.py` cluster bottom padding.

7. **(Audit infra) Re-run gallery with `--use-dagua-placement` to isolate placement vs render.**
   Phase 5 added the flag; next audit cycle should produce TWO galleries -- one
   harness-position, one Dagua-placement -- and diff them to separate harness artifacts
   from actual placement / render bugs. Pre-empts G5/G6/G7/G8 noise in future audits.

8. **G8 -- sibling cluster sibling-padding constant on `nested_clusters`.**
   Once G1 is resolved, increase sibling-cluster minimum gap so Right Branch / Left
   Branch are visibly separated by clear airspace, matching dot.
   Code area: cluster sibling separation default in styles or layout; verify after G1
   to avoid double-correcting.

9. **(Audit infra) Extend region masks to include cluster perimeter strokes.**
   The pixel-diff treats cluster strokes as background. A dedicated `cluster_stroke L1`
   region would surface label-stroke collisions and edge-perimeter crossings as
   measurable signals without needing visual review.

---

## Inspection Log

For each panel: nodes, labels, edges, arrowheads, cluster borders + labels + masks, and
worst-metric regions inspected.

- **`nested_clusters`** (SSIM 0.697 vs prior 0.693): outer cluster border drawn complete;
  inner Right Branch / Left Branch clusters drawn complete (F1 PASS); inner labels masked
  (F3 PASS); but A pierces outer top stroke (G3); "Outer Group" label fragments through A's
  fill (G3); A->C and A->B arrowheads on outer top stroke (G10); F sits below outer cluster
  bottom stroke; E->F and D->F edge bodies barely visible (G2); ellipses notably more
  circular than dot's (G12).
- **`cluster_showcase`** (SSIM 0.819 vs prior 0.818): all five clusters drawn including
  Large Cluster (F2 PASS); labels masked (F3 PASS); "outer a" partially outside Outer
  Cluster (G5); Tiny Cluster too tight on small 0 (G9); Outer Cluster + Large Cluster
  positioning overlap (G8 / harness); cross-cluster edges run with arrowheads at target
  nodes (correct for dot parity).
- **`transformer_block`** (SSIM 0.852 vs prior 0.851): MHA + FFN clusters drawn with clean
  labels; endpoint arrowheads (LayerNorm and other internal edges) clip cleanly to perimeters
  (F4 PASS for endpoint case); bypass edge from Input Embedding goes around MHA right side
  cleanly; bypass edge through FFN still crosses FFN right twice (G4); inner layout OK.
- **`cross_cluster_edges`** (SSIM 0.725): three cluster borders drawn; cluster labels
  masked; sibling clusters overlap dramatically -- Cluster X right side cuts through X3/X4,
  Cluster Y right side cuts through Y2 area, Z3->X1 bypass loops cleanly outside (G6 /
  harness); Y3 sits in the Y/Z overlap (harness); Y4 is inside both Cluster Y and Cluster Z;
  no stray fragments on Z (F10 from prior audit appears to be GONE).
- **`deep_nesting_4`** (SSIM 0.745 vs prior 0.733): Levels 1..4 are drawn but as STACKED
  ADJACENT rectangles instead of CONCENTRIC (G1 -- this is the most severe regression);
  inner clusters render as paired vertical strokes ("[" troughs) because top/bottom edges
  coincide with neighboring rectangles; Source->Outer 1, Outer 2->Mid 1, Mid 2->Inner 1,
  Inner 2->Core, Core->Exit, Exit->Sink all show as STUB ARROWHEADS with no visible body
  (G2 -- another severe regression). Cluster labels (Level 1..4) all draw and mask
  cleanly but on the wrong (stacked) bboxes.
- **`microservices`** (SSIM 0.850): four cluster boxes drawn (API Layer, Service Layer,
  Data Layer, Worker Layer); all labels masked cleanly (F3 PASS); endpoint clipping engages
  on API Layer exit (Auth Service / Rate Limiter arrowheads land at API Layer bottom,
  correctly); but Service Layer -> Data Layer edges (Search Service -> 4 DBs) collapse to
  arrowhead stubs (G2); Order Service sits on Service Layer's left edge (G7 / harness).
- **`data_pipeline`** (SSIM 0.754): Extract / Transform / Load all drawn with clean labels
  (F3 PASS); endpoint arrowheads from CSV/API/DB Source land cleanly at top of Extract
  (F4 PASS for entry); arrowheads from Parse CSV / Fetch API / Query DB to Validate /
  Normalize / Deduplicate / Enrich land at TARGET NODES (matches dot behavior, correct);
  Data Warehouse extends below Load bottom stroke (G11, prior F9 still partial).

Pixel-diff reality check: 5 of 7 cluster panels are below mean SSIM (0.761). The two
panels above mean are `microservices` (0.850) and `transformer_block` (0.852). The two
panels at the bottom of the cluster set are `nested_clusters` (0.697, rank 7 worst) and
`cross_cluster_edges` (0.725). `deep_nesting_4` (0.745) regressed on visual structure
even though SSIM nudged up because the simpler stacked rectangles produce LESS pixel
disagreement than the prior-audit's broken concentric attempt.

---

## Honest answer to user's bar

> "make our cluster functionality bulletproof with this sprint!! it should look at least as
> nice as graphviz and its a CORE FEATURE."

Closer, but **not yet bulletproof**. The Phase 5 wins are real: missing inner rectangles,
the missing Large Cluster rectangle, label-stroke collisions, and endpoint arrowhead
clipping all materially improved. But two NEW regressions surfaced (G1 concentric->stacked
nesting on `deep_nesting_4`, G2 edge bodies clipped to stubs on `microservices` and
`deep_nesting_4`) and one prior gap remains structurally open (G4 bypass edges through
foreign clusters). On `deep_nesting_4` the result is materially WORSE than dot's (the
nesting is broken, not just cosmetically off). On `microservices` cross-cluster edges
carry visual semantics ("Search Service writes to Order DB and 3 others") and that
semantic is currently lost to stub arrowheads.

CONTINUE for at least one more focused round (Phase 6) covering G1, G2, G4, G3 as the
top 4. Once those land:
- `nested_clusters` should show A above outer with clear airspace, "Outer Group" label
  visible end-to-end, arrowheads cleanly clipped at outer top.
- `deep_nesting_4` should show 4 concentric rectangles with continuous edge bodies.
- `microservices` should show full edge bodies between Service and Data layers.
- `transformer_block` bypass should not cross FFN right side.

Re-run the audit gallery with `--use-dagua-placement` so harness artifacts (G5, G6, G7,
G8) stop dominating subsequent reviews.
