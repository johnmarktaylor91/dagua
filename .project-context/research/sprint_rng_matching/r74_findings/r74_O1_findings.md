# r74 O1 — Sugiyama cluster (231 divergent, mode B). READ-ONLY survey.

## TL;DR (premise correction)

The brief assumed dagua's sugiyama is *quality-worse* than the reference (ba_500
"22344 vs 2805 crossings"). **The per_combo data says the opposite for the igraph
target.** In `per_combo.json`, `cross_D_mean`/`stress_D_mean` = **dagua (D-layout,
loaded with `engine`)**, `cross_R_mean`/`stress_R_mean` = **reference (R-layout,
loaded with `reference`)** — see `definitive_fidelity_analysis.py:2125-2126`
(`d_layout=...engine`, `r_layout=...reference`) and `:2222-2223`
(`cross_d=crossing_count(d_layout...)`).

Consequence:
- **166 igraph-target divergent combos: dagua is BETTER or EQUAL quality.** On 141/166,
  dagua's stress is strictly LOWER than igraph's; on the big graphs dagua is ~0.33 vs
  igraph ~0.52 stress, and dagua has FEWER crossings (rgg_2000: dagua 7.2M vs igraph 9.0M;
  ba_500: dagua 117932 vs igraph 140276). These are rung-4 ONLY because mode-B 3Q requires
  the dagua layout to be *equivalent to the reference's* (`stress_direct_equivalent` is
  False for all 231). dagua is "too good" relative to igraph. To reach rung 1-3 it must
  REPRODUCE igraph's (worse) layout, not beat it.
- **65 graphviz-target divergent combos: dagua is genuinely WORSE on stress** (median
  dagua 0.316 vs dot 0.123, ~2.5x; dagua higher on 63/65). On 43/65 crossings are already
  equivalent (`cross_direct_equivalent`), so ONLY stress (x-coordinate geometry) blocks 3Q.
  This is the real quality gap and matches the brief's intuition — but it's the dot variant,
  not igraph.

No combo is `degenerate`; all are mode B; none are FP-chaos floors (these are
deterministic algorithmic divergences with one-line/one-module root causes, exactly the
class r72/r73 warned looks like a floor but isn't).

## Pipeline map (`pipelines/sugiyama.py` -> `ops/sugiyama.py`)

`build_sugiyama_pipeline` (pipelines/sugiyama.py:36) chains Ops:
1. `_ValidateInputs` / `_StoreSpacingParams` / `_ResolveNodeSizes` / `_PrepareAcyclicEdges`.
2. **`_AssignLayers`** (ops/sugiyama.py:2200). Routes by `fidelity_mode`:
   - `igraph` -> `_igraph_glpk_layer_assignments` (:326): Eades feedback set, then SciPy
     **HiGHS LP** with objective `[0.0]*num_nodes` (:379) — pure feasibility.
   - `graphviz` -> `_graphviz_layer_assignments` (:260) -> `graphviz_rank_assignment`
     (`pipelines/dot_rank.py:104`), a full **network-simplex ranker** (feasible tree, cut
     values, top-bottom balance; 1531 lines — faithful `rank.c` port).
   - default -> `_longest_path_layering` (:209).
3. `_ExpandDummyNodes` (:2296) — long-edge chains.
4. **`_BarycenterOrdering`** (:2416). `use_graphviz_mincross` -> `graphviz_mincross`
   (`ops/_dot_mincross.py:14`, 501 lines: median+transpose+best-order). Else incidence /
   barycenter sweeps with igraph qsort tie-break (`_igraph_qsort_indices`:1194).
5. **`_CoordinateAssignment`** (:2555) -> `_coordinate_assignment` (:1399) ->
   **`_brandes_koepf_x_positions`** (:1465): 4-pass BK (ul/ur/dl/dr at :1505-1510),
   type-1-conflict detection, vertical alignment, horizontal compaction, median balance.
   **BK is used for x in BOTH igraph and graphviz modes.** Y = `layer_idx*rank_sep` (:1446).

## Reference ground-truth (corrects dagua's own docstrings)

### igraph 1.0.0 `src/layout/sugiyama.c`
- **Layering** `place_nodes_vertically` (:552). For **directed AND <=1000 nodes** (`:564`):
  GLPK IP. Objective is **NOT zero** — it is `glp_set_obj_dir(MIN)` with per-node coef
  `outdeg_i - indeg_i` (`:611 igraph_vector_sub(&outdegs,&indegs)`, `:615
  glp_set_obj_coef(..., outdegs[i])`). This is the Gansner network-simplex layering
  objective (minimize total edge length). **For undirected OR >1000 nodes it does NOT use
  the LP at all** — it falls to `feedback_arc_set_undirected` / `..._eades` layering
  (`:661-665`). dagua's docstring at `ops/sugiyama.py:346-349` ("objective effectively
  zero") is **wrong**, and dagua runs HiGHS unconditionally.
- **Ordering** `place_nodes_horizontally` first sets per-layer ordinal x then iterates
  barycenter SORT (`order_nodes_horizontally`), then runs a **Brandes-Koepf horizontal
  compaction** (`vertical_alignment`+`horizontal_compaction`, 4 `xs[4]` passes, :842-855).
  So igraph's x-geometry IS BK-family — dagua's BK is architecturally aligned. Divergence
  therefore propagates from LAYERING + ORDERING, not from the x-method.

### graphviz dot `lib/dotgen/position.c`
- **x-coordinates use NETWORK SIMPLEX, not BK.** `dot_position` (:127) builds an auxiliary
  LR graph (`make_LR_constraints`:218 — separation edges weighted by node widths+nodesep;
  `make_edge_pairs`:327 — slack nodes that pull each real edge's endpoints together with
  weight `ED_weight(e)`), then solves `rank(g, balance, nsiter2)` (network simplex, :142-148)
  and `set_xcoords`. Edge omega weights (real-real=1, real-virtual=2, virtual-virtual=8 in
  class2.c) make long edges straight, which is why dot's stress is ~2.5x lower than dagua's
  BK. **This is the architectural cause of the 65-combo graphviz stress gap.**

## The four fix-avenues, ROI-ordered

### A. igraph LP-gating + objective fix (HIGHEST ROI for igraph set)
**What:** In `_igraph_glpk_layer_assignments` (:326):
  (1) Add the missing objective: minimize `sum_i (outdeg_i - indeg_i) * x_i`
      (igraph sugiyama.c:611-615). Replace `[0.0]*num_nodes` (:379) with the
      outdeg-minus-indeg coefficient vector computed exactly as igraph (subtract feedback
      edge contributions first, :594-600).
  (2) Gate exactly like igraph: only run the LP when the (oriented) graph is **directed and
      num_nodes <= 1000**; otherwise route to `_igraph_eades_layer_assignments` (already
      exists at :392). igraph's `is_directed` test maps to "input had a consistent DAG
      orientation"; undirected benchmark graphs (er/ba/rgg/powerlaw/sbm/small_world) take
      the Eades branch in igraph (sugiyama.c:661-665, :667-670).
**Where it plugs in:** single function, `ops/sugiyama.py:326-389`. ~30-50 lines.
**Combos flipped:** of the 166 igraph divergent, **58 are big/undirected** (ba/er/rgg/
  powerlaw/sbm/small_world at 500-2000 nodes) where dagua wrongly runs HiGHS while igraph
  runs Eades — the gating fix targets these and they are the grossest divergences
  (rgg_2000, er_2000). The objective fix targets the **108 directed/small** combos where
  HiGHS picks the wrong feasible vertex. Realistically this won't reach bit-exact (GLPK
  simplex pivot != HiGHS pivot even with matched objective — the brief's "don't reproduce
  GLPK's pivot" caveat holds), but matched objective + matched gating should move a
  meaningful fraction (est. **25-60 combos**) from rung 4 to **rung 2'/3** by producing the
  same (or stress-equivalent) layering+ordering that igraph produces. **Confidence: HIGH on
  the bug being real and source-faithful; MEDIUM on how many combos cross the 3Q/equivalence
  threshold (pivot residual is genuine).** **Effort: ~0.5-1 day** (port + benchmark-path
  verification on the 41 igraph graphs).

### B. LP-canonical secondary objective / lexicographic tie-break (brief's headline idea)
**What:** The brief's "canonical, BETTER layering" idea. Assessment: **partly subsumed by A
  and partly mis-targeted.** Adding the igraph outdeg-indeg objective (A.1) IS the principled
  secondary objective — it is not arbitrary, it's the reference's own objective and it both
  shortens edges (better quality) AND matches igraph. A *further* lexicographic tie-break
  (e.g. among LP optima, minimize sum of ranks, then lexicographically smallest rank vector)
  would make dagua deterministic and canonical but would NOT necessarily match igraph's GLPK
  vertex, so it helps QUALITY/determinism but not FIDELITY-to-igraph. For the **graphviz**
  variant there is no LP-layering ambiguity (network simplex with balance is already
  deterministic via dot_rank.py).
**Verdict:** Do A's objective; skip a separate crossing-proxy LP objective — it risks the
  "laundering"/over-engineering trap and doesn't map to either reference. **Confidence:
  MEDIUM-LOW that a *separate* tie-break flips fidelity combos. Effort if pursued: ~1 day,
  LOW ROI.** Recommend folding into A.

### C. graphviz `position.c` network-simplex x-coordinates (HIGHEST ROI for dot set)
**What:** In graphviz fidelity mode, replace BK x-assignment with a port of dot's aux-graph
  network simplex (`position.c` `dot_position`->`make_LR_constraints`+`make_edge_pairs`->
  `rank`->`set_xcoords`). **Key leverage: dagua ALREADY has a network-simplex engine**
  (`pipelines/dot_rank.py` `_run_network_simplex`, with feasible tree + cut values +
  balance). The x-coordinate problem is the SAME network simplex on a different aux graph;
  the simplex core is reusable. The new work is building the aux graph: separation edges
  (width = `rw(u)+lw(v)+nodesep`, position.c:264) and slack-node edge pairs with omega
  weights (1/2/8). This directly attacks the median-0.316->~0.123 stress gap.
**Where:** new `_CoordinateAssignment` branch when `use_graphviz_mincross`/graphviz mode,
  building aux graph + calling the existing simplex; `ops/sugiyama.py:1399`/`:2555` +
  reuse of `dot_rank.py`. Est. 250-400 new lines (aux-graph construction dominates; simplex
  reused).
**Combos flipped:** the **65 graphviz combos**; on **43 crossings already match**, so an
  x-geometry that matches dot's straightened edges should push many of these 43 to **3Q
  (quality-identical)** and some toward stress-equivalence. The other 22 also need ordering
  fixes (avenue D). Est. **30-45 combos** to 3Q. **Confidence: HIGH that this closes most of
  the stress gap (it is the exact missing mechanism); MEDIUM on exact count crossing 3Q
  margins. Effort: ~2-4 days** (aux-graph fidelity + flat/self-edge minlen details + verify).

### D. graphviz mincross completeness (flat edges, init seeding, clusters)
**What:** `_dot_mincross.py` (501 lines) has median+transpose+best-order but is MISSING
  graphviz `mincross.c` pieces: `flat_breakcycles`/`flat_reorder`/`flat_search` (:121-123,
  flat-edge handling within a rank), `init_order` DFS seeding (initial order strongly
  determines the basin), `mincross_clust` (:129, cluster-constrained ordering), and
  `mincross_options`/`fixLabelOrder`. Different initial order + no flat handling => different
  per-layer permutation => different x => stress/crossing divergence.
**Where:** `ops/_dot_mincross.py` (init seeding ~40 lines; flat handling ~80-120 lines;
  clusters larger, lower priority).
**Combos flipped:** improves fidelity on the dense/clustered graphviz graphs (sbm/clustered/
  compound/grid/interleaved_cluster) and the 22 graphviz combos where crossings still differ.
  Est. **10-20 combos** incremental, and it raises the ceiling for avenue C (correct ordering
  feeds correct x). **Confidence: MEDIUM. Effort: init seeding ~0.5 day (do first); flat
  edges ~1-2 days; clusters ~3-5 days (defer).**

## Recoverability accounting (of 231)

| Bucket | Count | Best fix | Likely tier | Confidence |
|---|---|---|---|---|
| igraph big/undirected (wrong LP path) | 58 | A (gating) | 2'/3 partial | MED-HIGH |
| igraph directed/small (wrong LP objective) | 108 | A (objective)+D-init | 2'/3 partial | MEDIUM |
| graphviz, crossings already match (x-geom gap) | 43 | C | 3Q | MED-HIGH |
| graphviz, crossings also diverge | 22 | C+D | 3Q partial | MEDIUM |

**Genuinely-worse-quality residual:** essentially the **graphviz set's stress gap (65)** is
where dagua is worse and a real geometry port (C) is *required* — but it's recoverable, not a
floor. The **igraph set (166)** is not quality-worse at all; it's "better but different,"
recoverable only by faithfully reproducing igraph's layering/ordering (A). **Realistic r74
landing: ~60-110 combos off rung 4** (A is cheap and broad; C is the big-but-pricey win).
A hard FP-chaos floor for sugiyama is NOT demonstrated and not expected — these are
algorithmic, not last-ULP, divergences.

## Guardrail notes
- NO laundering: none of these reclassify by weakening FDR/BH; A/C/D change the *layout
  itself* to match the reference's algorithm.
- NO runtime delegation: all fixes are source-faithful ports of igraph/graphviz C; the
  reused dot_rank simplex is dagua's own port, not a reference call.
- Verify on the BENCHMARK PATH (LayoutConfig dispatch), not direct pipeline calls.
- Match params+seed: igraph LP gating depends on directed/<=1000 — must mirror exactly.
