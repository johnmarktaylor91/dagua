# Round 27 Diff: dagua dot mode vs Graphviz dot

Scope: diagnostic only. No fixes applied.

Source caveat: this local Graphviz clone does not contain
`/home/jtaylor/projects/_references/graphviz/lib/dotgen/dot.c`; the dot layout
entrypoint is in `dotinit.c`. I used the requested dotgen files that exist:
`dotinit.c`, `rank.c`, `mincross.c`, `position.c`, `flat.c`, `cluster.c`,
`acyclic.c`, `class1.c`, `class2.c`, `aspect.c`, and `dotsplines.c`.

Baseline command:

```bash
python scripts/algo_fidelity_live_compare.py classic_sugiyama graphviz_dot \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/dot/baseline
```

Baseline result: wrote `multi_seed_rmsd.csv` and `multi_seed_summary.json`;
all 5 requested graphs were present in the cached benchmark. Overall median
RMSD: `0.006317`; p25 `0.000000`; p75 `0.007816`; p95 `0.037107`; worst:
`mixed_width_labels 0.044430`.

Per-graph medians:

| graph | median RMSD |
|---|---:|
| `linear_3layer_mlp` | `0.0000000063` |
| `parallel_multiedge_bundle` | `0.0000000000` |
| `nested_shallow_enc_dec` | `0.0078164181` |
| `tl_mlp_3layer` | `0.0063167145` |
| `mixed_width_labels` | `0.0444298722` |

## Execution Path

Graphviz dot full phase order is:
`setEdgeType` + `setAspect` -> init -> `dot_rank` -> `dot_mincross` ->
`dot_position` -> `dot_sameports` -> `dot_splines` -> optional compound edges,
with component packing wrapped around that when `pack` is configured
(`/home/jtaylor/projects/_references/graphviz/lib/dotgen/dotinit.c:296`,
`:299`, `:300`, `:305`, `:312`, `:322`, `:333`, `:334`, `:338`,
`:437`, `:449`, `:465`, `:478`, `:485`, `:510`).

Dagua dispatch sends `algorithm=None` to `dagua_native`
(`/home/jtaylor/projects/dagua/dagua/layout/engine.py:1023`,
`:1055`, `:1067`, `:1069`) and explicit pipeline names through
`get_pipeline_function` (`engine.py:1071`, `:1077`). The baseline requested here
uses `classic_sugiyama`, which maps to `layout_sugiyama_pipeline`
(`/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/__init__.py:85`).
The dot-specific mimic code lives as optional polish candidates inside
`dagua_native.py`, especially `_dot_lattice_lp`, `_back_edge_relayer`, and
`_lattice_uniform_centered_slots`
(`/home/jtaylor/projects/dagua/dagua/layout/ops/pipelines/dagua_native.py:987`,
`:1056`, `:1262`, `:1339`, `:1584`, `:1630`, `:2697`, `:2705`,
`:2775`).

## Rank Assignment

1. Dagua classic uses longest-path layering plus a promotion pass, not dot's
   network simplex ranking. Graphviz collapses ranksets/clusters, classifies
   edges, reverses min/max violations, decomposes, DFS-breaks cycles, then runs
   network simplex (`rank.c:178`, `:347`, `:503`, `:508`, `:509`, `:510`,
   `:511`, `:512`, `:516`). Dagua classic does Kahn longest path and successor
   promotion (`dagua/layout/ops/sugiyama.py:201`, `:230`, `:236`, `:240`,
   `:278`, `:320`, `:329`, `:1556`, `:1560`). Label: algorithm-correctness.
   Fix size: large, +400-900 net lines if porting a true network-simplex ranker.
   Risk: high.

2. `_dot_lattice_lp` only approximates rank assignment with a generic LP over
   `rank[v] - rank[u] >= 1`, objective equal to indegree-outdegree
   (`dagua_native.py:1093`, `:1099`, `:1101`, `:1106`, `:1108`, `:1119`).
   Graphviz's rank objective uses edge `minlen`, `weight`, ranksets, components,
   cluster leaders, nonconstraint edges, and network simplex iteration limits
   (`rank.c:446`, `:452`, `:456`, `:587`, `:591`, `:761`, `:763`, `:804`,
   `:815`, `:1101`). Label: algorithm-correctness. Fix size: large,
   +500-1000 net lines. Risk: high.

3. Dagua ignores `rank=same|min|max|source|sink`; Graphviz explicitly unions and
   propagates those constraints (`rank.c:178`, `:221`, `:228`, `:345`,
   `:386`, `:404`, `:412`, `:466`). No corresponding dagua handling exists in
   `_longest_path_layering`, `_dot_lattice_lp`, or pipeline dispatch
   (`sugiyama.py:201`, `dagua_native.py:1093`). Label: algorithm-correctness.
   Fix size: medium-large, +250-500. Risk: high.

4. Dagua ignores `constraint=false`; Graphviz skips nonconstraint edges in
   rank compilation (`class1.c:22`, `:76`; `rank.c:587`, `:591`,
   `:815`). Dagua rank and LP consume every non-loop edge
   (`sugiyama.py:226`, `dagua_native.py:1084`, `:1096`). Label:
   algorithm-correctness. Fix size: medium, +80-180 plus metadata plumbing.
   Risk: medium.

5. Graphviz doubles edge `minlen` and halves `ranksep` for edge labels
   (`rank.c:160`, `:170`, `:173`, `:174`). Dagua has no label-rank reservation in
   classic or `_dot_lattice_lp`; dummy sizes are zero and labels do not affect
   ranks (`sugiyama.py:382`, `:386`; `dagua_native.py:1123`, `:1137`,
   `:1199`). Label: algorithm-correctness. Fix size: medium, +120-260. Risk:
   medium-high.

6. Graphviz connects disconnected components during newrank ranking with a root
   node and per-component normalization (`rank.c:1022`, `:1032`, `:1037`,
   `:963`, `:981`, `:983`). Dagua classic lays a single global Kahn layering
   unless igraph component packing is selected; dagua_native has separate
   optional component decomposition/tiling (`sugiyama.py:208`, `dagua_native.py:3120`,
   `:3175`). Label: convention. Fix size: medium, +120-300. Risk: medium.

7. Graphviz supports the `newrank` alternative constraint graph
   (`rank.c:522`, `:523`, `:1071`, `:1089`, `:1091`, `:1092`, `:1093`,
   `:1101`). Dagua has no `newrank` equivalent. Label: scaffolding. Fix size:
   large, +400-800. Risk: high.

## Back-edge and Cycle Handling

1. Graphviz's rank phase reverses DFS back-edges in the fast graph and preserves
   virtual-edge/original-edge mappings (`acyclic.c:22`, `:26`, `:27`, `:30`,
   `:33`, `:44`, `:47`, `:58`, `:62`). Dagua classic calls
   `make_acyclic_robust`, then loses dot's virtual-edge semantics and uses
   longest-path ranks (`sugiyama.py:159`, `:188`, `:197`, `:1472`,
   `:1509`). Label: algorithm-correctness. Fix size: medium, +150-350. Risk:
   medium-high.

2. `_back_edge_relayer` is a post-hoc cyclic polish, not dot's acyclic rank
   input. It detects DFS back-edges, removes them, reruns longest-path layering,
   and blends coordinates (`dagua_native.py:1584`, `:1606`, `:1622`, `:1673`,
   `:1682`, `:1686`, `:1704`, `:1716`). Graphviz reverses edges before ranking
   and later treats backward edges in class2/spline routing (`acyclic.c:47`;
   `class2.c:256`, `:282`; `dotsplines.c:351`, `:516`). Label:
   algorithm-correctness. Fix size: medium, +150-300. Risk: medium.

3. Graphviz handles backward edges that shadow forward edges and merges or
   ignores them depending on concentration/ports (`class2.c:256`, `:259`,
   `:264`, `:267`, `:269`, `:273`, `:282`). Dagua's relayer only has a boolean
   back-edge mask and no opposite-edge shadow logic (`dagua_native.py:1673`,
   `:1682`). Label: algorithm-correctness. Fix size: medium, +100-220. Risk:
   medium.

## Dummy Nodes, Edge Classification, and Multiedges

1. Graphviz classifies edges after ranks using `class2`, builds cluster
   skeletons, merges multi-edges, marks flat/self/forward/backward edges, and
   creates virtual chains with labels and weights (`class2.c:68`, `:77`, `:91`,
   `:130`, `:155`, `:163`, `:188`, `:205`, `:225`, `:242`, `:249`, `:256`).
   Dagua classic inserts only zero-size dummy nodes on long forward edges and
   preserves a simple path list (`sugiyama.py:337`, `:382`, `:386`, `:393`,
   `:397`). Label: algorithm-correctness. Fix size: large, +500-900. Risk:
   high.

2. `_dot_lattice_lp` gives long-edge dummy segments fixed weight `8.0` and
   adjacent real edges weight `1.0` (`dagua_native.py:1132`, `:1139`, `:1141`,
   `:1202`, `:1208`). Graphviz derives virtual weights through
   `virtual_weight`, merges `ED_weight`, `ED_count`, `ED_xpenalty`, and port
   equivalence (`class2.c:91`, `:92`, `:130`, `:139`, `:141`, `:142`,
   `:150`). Label: numerical. Fix size: medium, +100-250. Risk: medium.

3. Graphviz merges parallel multiedges and flat multiedges with label/port
   checks (`class2.c:190`, `:206`, `:212`, `:217`; `cluster.c:174`,
   `:188`, `:196`). Dagua's classic neighbor lists aggregate weights by neighbor
   for barycenters but dummy expansion does not merge chains or encode port
   identity (`sugiyama.py:487`, `:519`, `:521`, `:522`). Label:
   algorithm-correctness. Fix size: medium, +150-300. Risk: medium.

4. Graphviz has self-edge sizing before X constraints and self-edge spline
   routing (`position.c:245`, `:254`, `:260`; `dotsplines.c:388`, `:404`).
   Dagua filters self-loops before acyclic layering and has no self-loop width
   feedback in dot mode (`sugiyama.py:188`, `dagua_native.py:1004`, `:1086`,
   `:1674`). Label: algorithm-correctness. Fix size: medium, +100-250. Risk:
   medium.

## Mincross and Ordering

1. Graphviz mincross is multi-pass: build ranks, break/reorder flat edges,
   run cluster-aware mincross, merge clusters, optionally remincross, cache
   crossing counts, save/restore best orders, and finish with transpose
   (`mincross.c:331`, `:352`, `:355`, `:357`, `:364`, `:367`, `:379`,
   `:382`, `:392`, `:690`, `:700`, `:722`, `:732`, `:733`, `:746`).
   Dagua classic does fixed barycenter sweeps only (`sugiyama.py:526`,
   `:598`, `:602`, `:615`, `:643`). Label: algorithm-correctness. Fix size:
   large, +500-900. Risk: high.

2. Graphviz uses weighted medians with port order (`VAL(node,port)`), special
   even-median weighting, `ED_xpenalty`, and fixed nodes from flat edges
   (`mincross.c:1619`, `:1621`, `:1633`, `:1635`, `:1642`, `:1657`,
   `:1665`, `:1671`). Dagua classic computes weighted averages by neighbor
   position, not weighted medians, except the optional `_median_transpose_polish`
   which still ignores ports/xpenalty and projects onto existing slots
   (`sugiyama.py:649`, `:680`, `:685`, `:692`; `dagua_native.py:2230`,
   `:2357`, `:2363`, `:2391`). Label: algorithm-correctness. Fix size:
   medium-large, +250-500. Risk: high.

3. Graphviz transpose keeps swapping until no positive delta and has a reverse
   tie policy (`mincross.c:632`, `:642`, `:654`, `:673`, `:680`, `:687`).
   Dagua classic has no transpose in the base pipeline; dagua_native has 8
   native transpose passes by config and a separate polish with max 4 inner
   passes (`config.py:185`, `:189`, `:190`; `dagua_native.py:2367`,
   `:2370`, `:2380`). Label: algorithm-correctness. Fix size: medium,
   +120-260. Risk: medium.

4. Graphviz's crossing count is exact adjacent-rank crossing count weighted by
   `ED_xpenalty`, plus local port crossings (`mincross.c:1512`, `:1522`,
   `:1527`, `:1534`, `:1539`, `:1545`, `:1551`). Dagua's local polish crossing
   count is cheap and only considers two candidate nodes in adjacent layers
   (`dagua_native.py:2315`, `:2325`, `:2333`, `:2377`). Label: numerical.
   Fix size: medium, +150-300. Risk: medium.

5. Graphviz handles label-order feasibility for flat labels before positioning
   (`mincross.c:288`, `:297`, `:309`, `:320`). Dagua has no equivalent. Label:
   algorithm-correctness. Fix size: medium, +100-220. Risk: medium-high.

## X-coordinate Position

1. Graphviz dot X coordinates are another network-simplex solve over auxiliary
   LR constraints, edge-pair constraints, cluster containment, compression, and
   same-rank spacing (`position.c:127`, `:141`, `:142`, `:148`, `:183`,
   `:218`, `:238`, `:264`, `:327`, `:525`, `:528`, `:529`, `:530`,
   `:531`, `:571`). Dagua classic uses Brandes-Koepf compaction
   (`sugiyama.py:720`, `:772`, `:786`, `:825`, `:851`, `:858`, `:872`,
   `:873`). `_dot_lattice_lp` uses a generic absolute-deviation LP without dot
   auxiliary nodes or cluster constraints (`dagua_native.py:1202`, `:1206`,
   `:1212`, `:1225`, `:1234`, `:1239`). Label: algorithm-correctness. Fix
   size: large, +600-1000. Risk: high.

2. Graphviz stores node left/right extents (`ND_lw`, `ND_rw`) and constrains
   adjacent nodes by `rw(left)+lw(right)+nodesep` (`position.c:243`,
   `:244`, `:264`). Dagua classic uses symmetric width halves in
   `_minimum_separation`; `_dot_lattice_lp` uses one mean width times 1.5 for
   every adjacent pair (`sugiyama.py:1258`, `:1282`; `dagua_native.py:1197`,
   `:1225`, `:1233`). Label: numerical. Fix size: small-medium, +60-150. Risk:
   medium.

3. Graphviz adds x constraints for edge ports (`position.c:326`, `:338`,
   `:345`, `:346`) and flat-edge labels/endpoints (`position.c:269`, `:276`,
   `:289`, `:301`, `:307`, `:316`). Dagua dot paths ignore ports and flat label
   constraints. Label: algorithm-correctness. Fix size: medium-large,
   +250-450. Risk: high.

4. `_lattice_uniform_centered_slots` is explicitly a benchmark polish, not dot:
   it rewrites each layer to uniform slots at `0.75 * median pitch`
   (`dagua_native.py:1339`, `:1347`, `:1363`, `:1420`, `:1424`, `:1430`).
   Graphviz never performs this post-LP uniform-slot rewrite; its final X is
   network-simplex rank values (`position.c:142`, `:571`, `:580`). Label:
   scaffolding. Fix size: deletion/disable small, -40 to -100 for fidelity mode.
   Risk: low-medium.

5. `_dot_lattice_lp` anchors `x[0] = 0`, then subtracts min and recenters by
   mean (`dagua_native.py:1234`, `:1253`, `:1258`). Graphviz stores network
   simplex ranks as x, computes bounding boxes, and later may scale/round for
   ratio (`position.c:571`, `:580`, `:831`, `:904`, `:966`). Label:
   convention. Fix size: small, +30-80. Risk: low.

## Y-coordinate and Spacing Units

1. R3 addressed the biggest visible point-spacing issue in classic Sugiyama:
   direct calls default to Graphviz-like `rank_sep=72.0` and `node_sep=18.0`
   (`sugiyama.py:29`, `:30`, `:50`, `:54`, `:201`, `:203`). This is already
   addressed for the measured `classic_sugiyama` path. Label: already-addressed
   R3.

2. Graphviz rank separation is not a fixed center-to-center distance. It uses
   per-rank half-heights, primitive-vs-cluster separation, cluster labels,
   optional exact ranksep, and then copies y from the leftmost node in each rank
   (`position.c:729`, `:736`, `:754`, `:772`, `:777`, `:779`, `:780`,
   `:781`, `:812`, `:820`). Dagua classic sets `y = layer * rank_sep`
   (`sugiyama.py:763`, `:767`); `_dot_lattice_lp` sets `rank_int * ranksep`
   (`dagua_native.py:1199`, `:1257`). Label: algorithm-correctness. Fix size:
   medium, +150-350. Risk: medium-high.

3. `_dot_lattice_lp` spacing did not receive the R3 constants: `nodesep =
   mean_width * 1.5`, `ranksep = mean_height * 2.0`
   (`dagua_native.py:1197`, `:1199`). Graphviz dot units come from attributes in
   points, with `nodesep`/`ranksep` propagated through init and modified for
   labels (`dotinit.c:353`, `:354`; `rank.c:174`; `position.c:230`, `:236`,
   `:779`). Label: parameter-default. Fix size: small, +20-60. Risk: low-medium.

4. Graphviz supports `rankdir`/flip in labels, aspect, and spline routing
   (`dotinit.c:352`; `class2.c:30`; `flat.c:162`; `position.c:712`,
   `:912`, `:962`). Dagua dot paths assume a single top-down coordinate
   convention. Label: convention. Fix size: medium, +100-240. Risk: medium.

## Cluster Positioning

1. Graphviz clusters participate in ranking, mincross, and x-positioning:
   collapse cluster, build skeleton, expand cluster, merge ranks, build l/r
   virtual boundary nodes, contain nodes/subclusters, keep outsiders out, and
   separate sibling clusters (`rank.c:320`, `:328`, `:339`; `cluster.c:217`,
   `:280`, `:343`, `:350`, `:380`; `position.c:354`, `:392`, `:431`,
   `:454`, `:491`, `:1052`, `:1075`). Dagua's `classic_sugiyama` has no
   cluster input. Default dagua dispatch falls back to a separate
   cluster-aware wrapper or warns for unsupported algorithms (`engine.py:1079`,
   `:1084`, `:1087`). Label: algorithm-correctness. Fix size: very large,
   +800-1500. Risk: high.

2. Graphviz cluster labels affect rank heights and bounding boxes
   (`position.c:659`, `:682`, `:710`, `:722`, `:831`, `:872`). Dagua dot path
   ignores cluster label dimensions. Label: numerical. Fix size: medium,
   +120-260. Risk: medium.

3. Graphviz performs cluster-aware re-mincross when `remincross` is true
   (`mincross.c:367`, `:379`, `:380`, `:382`). Dagua has no equivalent. Label:
   algorithm-correctness. Fix size: medium-large, +250-500. Risk: high.

## Aspect Ratio and Component Packing

1. Graphviz parses `aspect` early but this clone disables that attribute with a
   warning (`aspect.c:21`, `:27`, `:33`; `dotinit.c:300`). Dagua native has an
   `AspectRatioFit` op in its pipeline machinery (`dagua_native.py:34`), which
   is not Graphviz-dot behavior for `aspect`. Label: convention. Fix size:
   small, +20-80 to gate/disable in dot-fidelity mode. Risk: low.

2. Graphviz still applies drawing ratio scaling in `position.c` for
   `ratio_kind`/`size`, including rounding coordinates and scaling cluster boxes
   (`position.c:904`, `:910`, `:916`, `:937`, `:949`, `:961`, `:966`,
   `:969`). Dagua classic baseline does no Graphviz ratio scaling. Label:
   parameter-default. Fix size: medium, +100-220. Risk: medium.

3. Graphviz can lay out connected components independently and pack subgraphs
   with cluster info transfer (`dotinit.c:437`, `:441`, `:465`, `:478`,
   `:485`, `:487`). Dagua native decomposition/tiling is area-based row-major
   logic, not Graphviz pack modes (`dagua_native.py:3120`, `:3175`). Label:
   convention. Fix size: medium-large, +250-500. Risk: medium.

## Edge Routing

1. Graphviz dot routes splines after positions, using rank boxes, node bounds,
   multi-edge separation, flat-edge special cases, self-edge routing, labels,
   sameports, and optional compound edges (`dotinit.c:333`, `:334`, `:338`;
   `dotsplines.c:264`, `:267`, `:281`, `:300`, `:322`, `:343`, `:388`,
   `:410`, `:419`, `:422`). Dagua classic only reconstructs dummy-node
   polylines if `return_edge_routes=True` (`sugiyama.py:426`, `:451`, `:452`,
   `:1911`). Label: algorithm-correctness. Fix size: very large, +800-1600.
   Risk: high.

2. Graphviz flat edges get separate routing and labels (`flat.c:242`, `:258`,
   `:299`, `:305`; `dotsplines.c:410`, `:411`, `:1502`). Dagua has no
   equivalent flat-edge route model. Label: algorithm-correctness. Fix size:
   medium-large, +250-500. Risk: high.

3. Graphviz normalizes backward splines back to original tail/head orientation
   (`dotsplines.c:351`, `:430`; also `class2.c:256`). Dagua route output flips
   only for its reversed mask in simple dummy paths (`sugiyama.py:451`,
   `:453`, `:454`). Label: convention. Fix size: small-medium, +60-150. Risk:
   medium.

## Ranked Fix Order for Round 28

1. Replace dot-mode rank assignment with dot-style network-simplex constraints,
   including `minlen`, `weight`, `constraint=false`, ranksets, label ranks, and
   component normalization. This dominates correctness. Risk high.

2. Replace `_dot_lattice_lp` X assignment with a dot-position-compatible
   auxiliary-constraint solve, or gate `_dot_lattice_lp` behind a non-dot
   benchmark-polish flag. Risk high.

3. Port Graphviz mincross semantics: weighted median with ports/xpenalty,
   exact crossing count, save/restore best order, transpose-until-stable, and
   flat-label order checks. Risk high.

4. Implement flat/self/multiedge classification before mincross and position.
   This likely targets `parallel_multiedge_bundle` and `mixed_width_labels`.
   Risk medium-high.

5. Move from fixed y rows to Graphviz rank-height/ranksep y assignment,
   preserving the R3 point-unit defaults but adding node/cluster/label height
   terms. Risk medium.

6. Add cluster skeleton/containment support only after rank/mincross/X basics,
   because it cuts across every phase. Risk high.

7. Treat spline routing as a later fidelity layer unless the comparison expands
   from node RMSD to edge geometry, because the current baseline measures node
   positions. Risk high.

## R3 vs Missed Items

Already addressed by R3:

- Classic Sugiyama defaults now use dot-like point spacing:
  `rank_sep=72.0`, `node_sep=18.0` (`sugiyama.py:29`, `:30`).
- The current requested baseline reflects that: median RMSD is `0.006317`,
  with two graphs at effectively zero RMSD.

Missed after R3:

- `_dot_lattice_lp` still computes spacing from mean node dimensions rather
  than point-unit `nodesep`/`ranksep` (`dagua_native.py:1197`, `:1199`).
- Dot's true rank assignment, mincross, x-position network simplex, clusters,
  flat/self/multiedge handling, aspect/ratio scaling, and splines are not
  replicated. The docstring at `dagua_native.py:1061` overstates fidelity:
  the function is a benchmark-oriented two-LP approximation, not Graphviz dot.
