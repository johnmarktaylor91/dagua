# r80 Visual Audit -- portfolio flips + routing quality (Opus, picky)

Auditor render path: `dagua.layout(g)` FRESH (shipping path) for dagua RIGHT panel;
frozen-store `.pt` for best external LEFT panel; BOTH rendered through the standard
`dagua.render()` draw path so `route_edges()` (node avoidance / port spread / label)
is applied identically. Panels 2-col, each panel <=900px, composite <=2000px longest
side (verified). Renders in /tmp/r80_visual/panels/. External = best per graph from
eval_output/r79_baseline/results.json.

Fresh dagua layout == frozen stored dagua position BIT-FOR-BIT (max_abs_diff 0.0 on
the two degeneracy graphs), so every composite score below was computed on exactly
the drawing shown.

## SUMMARY VERDICTS

| Graph | dagua | best ext | P13 verdict | Visual verdict |
|---|---|---|---|---|
| random_bipartite_60 | 79.09 | 78.77 sfdp | T/W (flip) | **VISUAL REGRESSION (CRITICAL)** |
| heavy_tail_weights_50 | 67.47 | 80.53 neato | L (showcase) | **VISUAL LOSS -- routing-showcase failure** |
| regular_4_40 | 71.71 | 73.34 sfdp | L | WIN-WITH-ARTIFACTS / degenerate compaction |
| real_karate_34 | 68.79 | 74.67 neato | L | honest loss, readability collapse |
| weighted_clusters_3x10 | 68.05 | 72.87 neato | L | honest loss, tiny-node readability |
| clustered_medium_5x20 | 65.31 | 66.88 dot | L (showcase) | honest loss, cluster-band overlap + lasso tangle |
| r79_undirected_sbm_high_mix_3x30 | 46.88 | 47.54 sfdp | L | honest loss (both hairball) |
| petersen_10 | 79.02 | 79.41 neato | T (flip) | WIN-WITH-ARTIFACTS (central knot) |
| planar_60 | 77.72 | 77.80 sfdp | T (flip) | WIN-WITH-ARTIFACTS (arc crowding + void stubs) |
| weighted_karate_34 | 69.55 | 74.67 neato | L | honest loss, visually competitive |
| regular_3_30 | 84.34 | 84.74 neato | T (flip) | CLEAN WIN (tie) |
| triangular_lattice_36 | 94.48 | 94.48 neato | T (flip) | CLEAN WIN (tie) |
| citation_dag_300 | 59.30 | 57.36 elk | W (showcase) | WIN (marginal; unassessable at scale) |
| hexagonal_lattice_42 | 93.92 | 91.75 dot | W (flip) | **CLEAN WIN (best result)** |

## PATHOLOGY OF THE ROUND (shipping blocker)

**random_bipartite_60 -- isolated-node fling gaming the composite.**
Nodes 29, 30, 45 are the graph's only three degree-0 (isolated) nodes. dagua's force
layout has no anchoring for them, so it flings them 3300 / 4044 / 4680 coordinate units
from the connected core (core median radius = 167). bbox span 4225 vs median radius 167
= 28x. Consequence in the shipped drawing: the connected 57-node graph collapses into an
illegible ~15% corner blob while ~85% of the canvas is empty and two stray dots sit in
the far corners. The composite AWARDS THIS A WIN (79.09 > sfdp 78.77) because edge-based
terms (edge length, crossings) are indifferent to edgeless nodes flung to infinity, and
the aspect/spread guard does not catch a few extreme outliers. This is the textbook
"metrics say win, eyes say catastrophe" pathology the brief was hunting. sfdp places the
same three isolates in nearby open space and stays fully legible.

Recommend BEFORE any flip is claimed on this graph: (a) a composite degeneracy guard on
max-radius / median-radius (or on empty-canvas fraction), and (b) layout anchoring for
degree-0 components so isolates park near the core. This exact class flipped random_bipartite
L->T/W; the flip is not real to a human reader.

## PER-GRAPH OBSERVATIONS (min 2 each)

### random_bipartite_60 -- VISUAL REGRESSION (CRITICAL)
1. Three degree-0 nodes (29/30/45) flung 3.3k-4.7k units; main graph collapses to a
   corner dot, ~85% canvas empty. Confirmed numerically (28x radius ratio).
2. The connected core is an unreadable micro-hairball; labels invisible.
3. sfdp (LEFT) keeps all 60 nodes legible incl. the three isolates in open space.
4. Composite WIN (79.09) is scored on this exact drawing -- metric-gamed.

### heavy_tail_weights_50 -- VISUAL LOSS (routing showcase fails)
1. Nodes rendered as near-invisible tiny horizontal dashes; every label illegible vs
   neato's clearly-labelled 50 nodes.
2. Drawing dominated by long sweeping curved edges forming a chaotic V/heart tangle --
   many long-range crossings, lasso-like curls. Heavy-tail weights dispersed the layout
   (span 9350 vs median radius 1549).
3. No visible structure/clustering. Worst-reading drawing in the set alongside
   random_bipartite. The "upgraded routing" does NOT shine here -- it produces the tangle.
4. Metrics agree it is a loss (67.47 vs 80.53, the largest gap in the set).

### regular_4_40 -- WIN-WITH-ARTIFACTS / degenerate compaction
1. Severe node compaction: nodes fill the canvas edge-to-edge, many borders overlapping/
   touching (32-1, 35-32, 28-6, 39-7, 26-24, 8-6). Almost no inter-node whitespace.
2. Node 36 clipped at the top panel edge (bbox-touching).
3. Central port clutter -- arrowheads converge densely (around 8/6/32), edges hard to trace.
4. sfdp (LEFT) keeps clear inter-node whitespace; dagua reads worse despite similar structure.
   Metrics correctly rank it below sfdp (71.71 < 73.34).

### real_karate_34 -- honest loss, readability collapse
1. Dense central hairball at hub nodes 33/20/32 whose borders touch/stack; edges converge
   in an unreadable knot (port clutter).
2. Peripheral outliers (15 bottom, 11 right, 23/25 left) stretch the bbox so core nodes
   render tiny with barely-legible labels.
3. neato (LEFT) uses large well-spaced nodes -- clearly more readable. Loss is honest
   (68.79 < 74.67), not gamed.

### weighted_clusters_3x10 -- honest loss, tiny-node readability
1. Nodes rendered tiny; sbm/cluster labels barely legible vs neato's large clear nodes.
2. Central stringy web of long near-parallel edges through the 8/12/22 hub.
3. Three clusters ARE preserved with good whitespace (no overlap) -- opposite failure
   mode from regular_4_40. Loss honest (68.05 < 72.87).

### clustered_medium_5x20 -- honest loss (routing showcase underwhelms)
1. Cluster bands are tall vertical strips that overlap/interleave in x (Cluster 0/1/2 not
   cleanly separated) vs dot's clean diagonal staircase of separated boxes.
2. Lower half is a dense tangle of long sweeping curved inter-cluster edges (lasso arcs) --
   port fan-out from cluster boundaries, hard to trace.
3. Some cluster labels (Cluster 0/1) sit crowded among bands/edges, not cleanly above box.
4. Loss honest (65.31 < 66.88). Routing upgrade did not beat dot here.

### r79_undirected_sbm_high_mix_3x30 -- honest loss (both hairball)
1. dagua nodes tinier / labels illegible vs sfdp's readable sbm_XX labels.
2. Cluster boxes (Block 1/2/3) heavily overlap -- no community separation; intrinsic to
   high-mix SBM, both engines show it (not dagua-specific).
3. dagua has several very long perimeter edges sweeping the bottom (near-lasso arcs).
4. Minor: a cluster label prints "Block 1)" with a stray ")" (also messy on LEFT) -- cosmetic.

### petersen_10 -- WIN-WITH-ARTIFACTS
1. dagua more compact than neato but forms a central knot: nodes 3-0 nearly touching,
   8 close to 6 and 5; a dense cluster of ~4-5 arrowheads converges near 4/8 (port clutter).
2. Two near-parallel curved edges top-right form a bundled near-lasso.
3. neato (LEFT) is airier/cleaner. dagua slightly LOWER composite (79.02 < 79.41) -- the
   flip to T is legitimate but dagua is not visually better.

### planar_60 -- WIN-WITH-ARTIFACTS
1. Ring/annulus with a clean central void -- comparable to (arguably cleaner than) sfdp.
2. Lower-right arc crowded: nodes 6/43/18/42/54/5/17/29/41 packed with a couple border
   overlaps (6-43, 42-54) and converging arrowheads.
3. Nodes 8 and 9 have short edge stubs curling into the empty central void (possible
   short-edge curl artifacts). No degenerate compaction. Tie is fair (77.72 ~ 77.80).

### weighted_karate_34 -- honest loss, visually competitive
1. Two-lobe structure; hub 33 carries a dense fan of ~15 edges + arrowheads (port clutter)
   but nodes stay legible (far better render than unweighted real_karate).
2. {5,6,10,4,16} top-right cluster tight with overlapping arrowheads near 6; 33-32 borders
   touch (mild overlap).
3. Spread comparable to neato -- no degenerate compaction. Loss honest (69.55 < 74.67).

### regular_3_30 -- CLEAN WIN (tie)
1. dagua's outer-ring shape closely mirrors neato's; good even spacing, no degenerate
   compaction.
2. Bottom cluster (8/29/19/28/2) is a bit crowded; node 29 receives a small arrowhead fan
   (mild port clutter), 8-29 borders nearly touch.
3. Otherwise equal to neato; tie fair (84.34 ~ 84.74).

### triangular_lattice_36 -- CLEAN WIN (tie)
1. Both render the triangular lattice as a clean diamond; node positions near-identical.
2. dagua has marginally BETTER node separation (less border overlap than neato, e.g. rows
   34-28, 20-13 cleaner). No overlaps in dagua.
3. Edges follow lattice bonds cleanly; no curls/clutter. Exact composite tie 94.48.

### citation_dag_300 -- WIN (marginal, unassessable at scale)
1. Both are dense 300-node hairballs of tiny points; dagua shows clearer horizontal layer
   banding at top, elk a more uniform trapezoid fill.
2. dagua bottom funnel: heavy edge convergence onto 1-2 sink hubs (structural for a
   citation DAG; elk shows the same).
3. Edge-routing quality (node avoidance/curvature) is NOT visually assessable at this
   density -- straight-ish bundles both sides, no obvious lasso curls. No overlap
   catastrophe. Marginal win (59.30 > 57.36) plausible but not visually confirmable.

### hexagonal_lattice_42 -- CLEAN WIN (best result of the round)
1. dagua reveals the TRUE honeycomb / hexagonal cells (tilted lattice); graphviz_dot (LEFT)
   collapses it into a diamond layered stack that hides the hex structure. dagua far more
   faithful.
2. Regular even spacing, zero node overlaps, edges follow lattice bonds; boundary nodes
   (35, 6) dangle naturally.
3. Only nit: lattice tilted ~20deg rather than axis-aligned -- purely cosmetic. Win is real
   and readable (93.92 > 91.75).

## BOTTOM LINE
- Genuine clean flips a human would agree with: hexagonal_lattice_42 (excellent),
  triangular_lattice_36, regular_3_30. petersen_10 and planar_60 are fair ties with
  minor artifacts.
- ONE flip does NOT survive visual audit: random_bipartite_60 is a metric-gamed WIN
  driven by isolated-node fling -- shipping blocker; the composite must be guarded.
- The two "routing showcase" small graphs (heavy_tail_weights_50, clustered_medium_5x20)
  do NOT showcase routing: heavy_tail is the worst-reading drawing in the set (lasso
  tangle + invisible nodes) and clustered_medium loses on cluster-band overlap + long-
  curve tangle. citation_dag_300's win is real on metrics but unverifiable by eye.
- Recurring dagua failure modes vs external: (a) isolated/low-degree node fling
  (random_bipartite), (b) large-spread tiny-node readability collapse (heavy_tail,
  real_karate, weighted_clusters), (c) edge-to-edge overlap compaction (regular_4_40),
  (d) long sweeping-curve lasso tangles from the router on dispersed layouts
  (heavy_tail, clustered_medium, sbm perimeter).

## Evidence -- r80 singleton challenger fix

* Root cause confirmed:
  `dagua/layout/ops/pipelines/native_undirected.py:703-739` and `:751-786`
  previously built the sfdp/neato challengers by solving the full problem in
  one call. Degree-0 nodes are one-node weak components, so they had no local
  edge constraints inside those solvers and could be flung far from the core
  before the composite contest scored the candidate.
* Fix approach:
  chose full per-component challenger solving, not singleton-only extraction.
  `_run_component_packed_challenger` now uses the existing
  `dagua.layout.ops.coordinate._weak_components`, extracts relabeled child
  problems with `_extract_component_problem`, solves each non-singleton child
  with the same challenger algorithm, uses origin positions for singleton
  children, and reassembles with the existing `_tile_component_positions`
  tiler. This is lower-risk than inventing a new isolate packer and keeps
  disconnected non-singleton components consistent with the incumbent tiling
  lifecycle. The incumbent path was not changed.
* Degeneracy guard:
  added `DEGENERACY_MAX_CENTROID_SPREAD_RATIO = 6.0` at
  `native_undirected.py:75`. `_candidate_is_degenerate` now rejects
  challengers when max centroid distance is more than 6x the median centroid
  distance, skipping the check when the median is zero so existing collapse
  checks handle coincident layouts. This guards future far-flung-node
  candidates without special-casing `random_bipartite_60`.
* Unit tests:
  `.venv/bin/pytest tests/test_native_undirected_portfolio.py -x --tb=short`
  -> 23 passed, 6 warnings.
  `.venv/bin/pytest tests/ -k "native_undirected or portfolio or dagua_native" -x --tb=short`
  -> 44 passed, 3383 deselected, 38 warnings.
  Project targeted gate
  `.venv/bin/pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`
  -> failed on pre-existing environment CUDA initialization in
  `test_streaming_coarsen_end_to_end_matches_cpu_layout` after 141 passed and
  18 skipped; NVIDIA driver 12040 is too old for the installed torch CUDA.
* Visual sanity:
  rendered `/tmp/r80_bipartite_fixed.png` through `dagua.draw`.
  Before audit pathology -> isolates at 3300/4044/4680 units from core,
  core median radius 167, ratio about 28x.
  After fix -> bbox=(-1767.026, 1766.800, -1955.352, 2198.621),
  core_node_count=57, isolate_count=3, median_radius=2274.112,
  max_radius=2697.651, ratio=1.186. Isolates are nodes [29, 30, 45]
  with radii [2623.031, 2406.055, 2340.330].
* Sweep:
  `--fresh` was confirmed by `scripts/r79_baseline.py --help`. The default
  output dir refused `--fresh` because 107 dagua rows remained resumable in the
  staging store, so the sweep was launched against clean output dir
  `eval_output/r80_singleton_sweep` with:
  `setsid .venv/bin/python -u scripts/r79_baseline.py --dagua-only --fresh --output-dir eval_output/r80_singleton_sweep > /tmp/r80_singleton_sweep.log 2>&1 < /dev/null &`
  PID 3398381, log `/tmp/r80_singleton_sweep.log`. Results pending --
  architect to harvest.

## Evidence -- r80 singleton guard NARROWED (follow-up, gate verdict)

* Gate sweep verdict on the first guard form: PARTIAL FAIL. The packing fix and
  the random_bipartite_60 honest reversion (79.09 -> 65.24) were accepted, but
  the GLOBAL max/median centroid-spread test was over-broad: it also rejected
  legitimately-dispersed candidates and regressed multi_component_80
  (92.52 -> 81.55, -11.0), er_500 (51.72 -> 46.82, -4.9, a real win flipped to
  loss), and scale_free_ba_120 (-1.9). Multi-component tilings and ER-periphery
  structure legitimately exceed a global radius ratio.
* Narrowing: the guard now judges ISOLATED (degree-0) nodes only -- the actual
  metric-blind pathology. `DEGENERACY_MAX_CENTROID_SPREAD_RATIO = 6.0` replaced
  by `DEGENERACY_MAX_ISOLATED_SPREAD_RATIO = 3.0`
  (native_undirected.py, constants block). A candidate is rejected iff any
  degree-0 node sits further than 3.0x the median centroid distance from the
  layout centroid. Skipped when there are no isolates, when ALL nodes are
  isolated (no connected core exists), or when the median distance is zero
  (true collapse is covered by the existing checks 1-2). Connected-node spread
  is never judged.
* Tests updated: the flung-CONNECTED-node case now asserts PASS (was the
  global-guard rejection test); new flung-ISOLATED-node rejection test; new
  multi_component_80-style test (14-node path + far 2-node component, global
  ratio ~7x, zero isolates) asserts PASS. Full file: 25 passed. ruff clean.
* Gate sweep relaunched: default output dir, --dagua-only --fresh, log
  /tmp/r80_singleton_sweep4.log. Acceptance: only random_bipartite_60 moves vs
  the 74/108 store (its honest reversion); zero other movers. Results pending
  -- architect to harvest.
