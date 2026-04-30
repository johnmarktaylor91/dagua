# Round 27 Diff: dagua fdp-mode vs Graphviz fdp

Diagnostic-only line-by-line diff against `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/`.
No source edits were made.

## Scope finding

Dagua does not currently register an in-process `algorithm="fdp"` name. Explicit algorithms are
resolved through `get_pipeline_function(config.algorithm)` at `dagua/layout/engine.py:1071-1139`,
and `PIPELINE_REGISTRY` has `"fmmm"` at `dagua/layout/ops/pipelines/__init__.py:31`, but no
`"fdp"` alias at `dagua/layout/ops/pipelines/__init__.py:12-88`.

The tested "fdp mode" is therefore the benchmark proxy `classic_fmmm` against external
`graphviz_fdp`: `classic_fmmm` calls `layout_fmmm_pipeline(..., steps=200, fidelity_mode=True)`
at `dagua/eval/competitors/classic_competitor.py:1495-1542`; `graphviz_fdp` shells out to
`dot -Tjson -Kfdp` and passes `-Gseed`/`-Gstart` at
`dagua/eval/competitors/graphviz_competitor.py:350-404` and
`dagua/eval/competitors/graphviz_competitor.py:538-542`.

Important correction: Graphviz fdp is not the same implementation target as OGDF FM^3. The local
Graphviz source implements a recursive derived-graph spring-electrical layout with ports, clusters,
component packing, and a second overlap-expansion pass. Dagua's `classic_fmmm` is an OGDF-style
solar-system FM^3 pipeline.

## Baseline

Command run:

```bash
python scripts/algo_fidelity_live_compare.py classic_fmmm graphviz_fdp \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_27/fdp/baseline
```

Output:

- `eval_output/algo_fidelity/round_27/fdp/baseline/multi_seed_rmsd.csv`
- `eval_output/algo_fidelity/round_27/fdp/baseline/multi_seed_summary.json`
- Overall median RMSD: `0.121966`; p25 `0.117278`; p75 `0.237276`; p95 `0.249752`.
- Worst graph: `nested_shallow_enc_dec`, median RMSD `0.252871`.

Per graph:

| graph | median dagua-vs-graphviz RMSD | within dagua median | within graphviz median | TOST |
|---|---:|---:|---:|---|
| `parallel_multiedge_bundle` | `0.007276` | `0.000142` | `0.000000064` | `not_equivalent` |
| `linear_3layer_mlp` | `0.117278` | `0.012339` | `0.000000041` | `not_equivalent` |
| `mixed_width_labels` | `0.121966` | `0.001123` | `0.000000017` | `not_equivalent` |
| `tl_mlp_3layer` | `0.237276` | `0.015794` | one target seed | `not_tested` |
| `nested_shallow_enc_dec` | `0.252871` | `0.012339` | `0.000000076` | `not_equivalent` |

The seeded Graphviz-fdp cache is effectively deterministic for four of the five graphs, so the
TOST margin collapses to the comparator's `1e-6` floor. This makes all nonzero geometry drift fail
equivalence even where the visual gap is small.

## Ranked divergences

### 1. P0: Dagua has no Graphviz-fdp dispatch contract

- Graphviz: `fdp_layout()` is a named engine entrypoint at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:1063-1078`; it initializes fdp
  graph state, runs fdp layout, sets aspect, routes splines, and postprocesses output.
- Dagua: no `fdp` registry entry exists; `classic_fmmm` is only a competitor proxy and public
  dispatch exposes `"fmmm"`, not `"fdp"` (`dagua/layout/ops/pipelines/__init__.py:31`).
- Impact: any `algorithm="fdp"` parity claim is currently out-of-band. The benchmark can compare
  `classic_fmmm` to Graphviz fdp, but dagua does not implement Graphviz fdp semantics.
- Label: `dispatch_contract_gap`.
- Estimated fix size: M. Add an `fdp` alias/adapter only after deciding whether it should be a
  faithful Graphviz-fdp implementation or simply point to current OGDF-FMMM.

### 2. P0: Core algorithm family is different from Graphviz fdp

- Graphviz: `layout()` recursively derives a graph, computes connected components, runs
  `fdp_tLayout()`, recursively expands clusters, removes overlaps, packs components, and copies
  positions back at `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:772-923`.
- Dagua: `build_fmmm_pipeline()` builds OGDF-style hierarchy ops:
  initialize state, initialize coarsest, refine coarsest, uncoarsen, fallback, finalize at
  `dagua/layout/ops/pipelines/fmmm.py:67-92`.
- Impact: fdp-specific cluster/port/component behavior cannot emerge from the current FM^3 proxy.
  This is the largest structural reason Graphviz fdp remains distinct after OGDF-FMMM convergence.
- Label: `algorithm_family_gap`.
- Estimated fix size: L. Implement Graphviz-fdp as its own pipeline rather than tuning FMMM knobs.

### 3. P0: Graphviz fdp coarsening is derived-cluster recursion, not solar-system FM^3

- Graphviz: `deriveGraph()` collapses each cluster to a derived node, copies selected graph
  attributes, creates derived nodes for non-cluster nodes, aggregates inter-derived edges, and adds
  port nodes at `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:374-524`.
- Graphviz: after derived layout, `expandCluster()` generates boundary ports from derived edge
  angles and recursively lays out the cluster interior at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:662-695` and
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:831-847`.
- Dagua: `_build_hierarchy()` repeatedly applies solar-system coarsening until `_COARSE_TARGET=50`
  or the edge-count guard breaks, at `dagua/layout/ops/fmmm.py:657-734`.
- Dagua: `_coarsen_level()` selects sun nodes, assigns planets/moons, builds coarse edges, and
  stores lambda interpolation metadata at `dagua/layout/ops/fmmm.py:466-654`.
- Impact: Graphviz fdp's "multilevel" behavior is hierarchical cluster/port recursion over
  derived graphs; Dagua's is OGDF galaxy coarsening over the whole graph. These are not
  parameter-equivalent.
- Label: `coarsening_strategy_gap`.
- Estimated fix size: L.

### 4. P0: Force law and edge-length semantics are not Graphviz fdp

- Graphviz: defaults `K=0.3`, `maxiter=600`, `seed=1`, and parses graph `K`, `T0`, `maxiter`,
  and `start/seed` in `fdp_initParams()` at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:96-188`.
- Graphviz: edge `weight` maps to `ED_factor`, edge `len` maps to `ED_dist`, defaulting to `K`,
  at `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/fdpinit.c:70-79`.
- Graphviz: attraction is either `(dist - ED_dist) / dist * weight` in `useNew` mode or
  `dist / ED_dist * weight`; repulsion is `K*K/dist2` or `K*K/(dist*dist2)` in `useNew` mode at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:190-310`.
- Dagua: FMMM attraction uses OGDF logarithmic scale
  `log2(distances / desired_lengths) * distances / desired_lengths**3` at
  `dagua/layout/ops/fmmm.py:51-82`, and repulsion is a pure `1/d^2` field at
  `dagua/layout/ops/fmmm.py:873-896`.
- Dagua: base edge lengths are currently `1.0` averages in `_unique_edges_with_lengths()` at
  `dagua/layout/ops/fmmm.py:329-404`; it does not read Graphviz DOT `len` into `ED_dist`.
- Impact: even with similar starts and iteration counts, force equilibrium differs. This directly
  explains residual RMSD on simple non-cluster graphs.
- Label: `force_law_gap`.
- Estimated fix size: M-L. Add a Graphviz-fdp force mode with `K`, `len`, `weight`, and `useNew`
  semantics.

### 5. P0: Iteration schedule and cooling differ

- Graphviz: default `maxiter=600`; `unscaled` controls pass 1 length via
  `T_pass1 = T_unscaled * T_maxIters / 100` at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:152-180`.
- Graphviz: `init_params()` splits iterations between the main layout and overlap-expansion pass
  using `T_loopcnt` and `xpms->loopcnt` at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:115-149`.
- Graphviz: cooling is linear, `T0 * (maxIters - t) / maxIters`, at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:103`.
- Dagua: `classic_fmmm` hardcodes `steps=200` at
  `dagua/eval/competitors/classic_competitor.py:1535-1542`; `FMMMRefineLevel` uses per-level fixed
  step budgets with multiplicative cooling factor `_COOLING_FACTOR=0.99` at
  `dagua/layout/ops/fmmm.py:30-44` and `dagua/layout/ops/fmmm.py:1038-1099`.
- Impact: Graphviz fdp spends up to 600 linear-cooling ticks split between layout and expansion;
  Dagua spends 200 FM^3 refinement ticks distributed across hierarchy levels.
- Label: `iteration_schedule_gap`.
- Estimated fix size: M.

### 6. P0: Initial placement and pin handling differ

- Graphviz: reads node `pos` and `pin`, scales by `inputscale`, and marks nodes `P_SET` or `P_PIN`
  at `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/fdpinit.c:33-68`.
- Graphviz: `initPositions()` computes a pinned-node bounding box, derives an enclosing ellipse or
  rectangle, seeds `srand48()`, places ports on the ellipse, places nodes near positioned
  neighbors when available, and randomizes the rest at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/tlayout.c:414-568`.
- Dagua: FMMM reference mode initializes the coarsest graph with OGDF random mode and prolongs
  via solar-system metadata (`dagua/layout/ops/pipelines/fmmm.py:67-77` and
  `dagua/layout/ops/fmmm.py:1367-1469`).
- Dagua engine dispatch can resolve flex pins for generic pipelines at
  `dagua/layout/engine.py:1115-1120`, but `layout_fmmm_pipeline()` accepts no `config` or flex
  argument at `dagua/layout/ops/pipelines/fmmm.py:97-187`.
- Impact: Graphviz-fdp positional constraints, `pin`, `pos!`, and port-aware starts are absent from
  the current fdp proxy.
- Label: `initialization_pin_gap`.
- Estimated fix size: M.

### 7. P1: Component semantics and packing differ

- Graphviz: `findCComp()` creates generalized connected components where all port-node or pinned
  components merge into the first component at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/comp.c:51-134`.
- Graphviz: after per-component layout, multiple components are packed with `putGraphs()`, with the
  pinned component fixed when needed, at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:859-883`.
- Dagua FMMM: `_build_hierarchy()` operates on the whole edge tensor and does not do Graphviz
  generalized components or pack-aware pinned handling (`dagua/layout/ops/fmmm.py:657-734`).
- Dagua native dispatch has component decomposition for native pipelines at
  `dagua/layout/ops/pipelines/dagua_native.py:195-242`, but `classic_fmmm` does not use
  `dagua_native`.
- Impact: disconnected or port/pin-heavy inputs will diverge in both relative component placement
  and fixed-component behavior.
- Label: `component_packing_gap`.
- Estimated fix size: M-L.

### 8. P1: Overlap removal is a dedicated Graphviz fdp expansion pass

- Graphviz: `fdp_xLayout()` parses the graph `overlap` attribute, supports `n:mode`, runs
  `x_layout()` for the requested number of tries, and then calls `removeOverlapAs()` at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/xlayout.c:312-361`.
- Graphviz: `x_layout()` uses node dimensions, `sep`, overlap-specific repulsion `X_ov`, non-overlap
  repulsion `X_nonov`, and edge attraction that accounts for node radii at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/xlayout.c:66-309`.
- Dagua FMMM: finalization normalizes point coordinates after refinement; no Graphviz-style
  `overlap="9:prism"` default, `sep`, radius-aware expansion, or Prism fallback is present in the
  FMMM pipeline (`dagua/layout/ops/pipelines/fmmm.py:78-82`).
- Impact: label/shape-size graphs such as `mixed_width_labels` remain algorithmically mismatched
  even if point-force layout improves.
- Label: `overlap_expansion_gap`.
- Estimated fix size: L if Graphviz parity is required.

### 9. P1: Ports and clusters are first-class in Graphviz fdp, not in Dagua FMMM

- Graphviz: ports are represented by `bport_t` (`fdp.h:19-23`) and transformed into derived-graph
  port nodes at `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:484-522`.
- Graphviz: edge angles around clusters are sorted, ties are spread by at most two degrees, and
  `genPorts()` creates ordered ports for multiedges at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:527-654`.
- Dagua: `_build_cluster_inner_pipeline()` explicitly supports cluster recursion only for `fr`,
  `kk`, `fa2`, and `sfdp`, not `fmmm`, at `dagua/layout/engine.py:116-149`.
- Dagua: if `algorithm="fmmm"` and clusters exist, engine warns and falls back to flat placement
  unless a supported cluster-aware inner pipeline exists (`dagua/layout/engine.py:1079-1092`).
- Impact: nested cluster examples are expected to be bad fdp matches. Baseline confirms this:
  `nested_shallow_enc_dec` is the worst graph at `0.252871` median RMSD.
- Label: `port_cluster_gap`.
- Estimated fix size: L.

### 10. P1: Multiedge semantics differ in both layout and ports

- Graphviz: `deriveGraph()` stores all real edges behind one derived edge via `ED_to_virt` and
  `ED_count`, while preserving the first edge's `ED_dist` and `ED_factor` at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:339-352` and
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:457-481`.
- Graphviz: `genPorts()` fans multiple real edges around the same derived-edge angle at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:606-654`.
- Dagua: fidelity FMMM sets `sum_parallel_weights=False` so parallel edges reduce to an averaged
  simple edge at `dagua/layout/ops/pipelines/fmmm.py:67-77` and
  `dagua/layout/ops/fmmm.py:329-404`.
- Impact: the small `parallel_multiedge_bundle` RMSD (`0.007276`) says this is not the current
  dominant point-position gap, but it is still a feature gap for ports/splines and edge routing.
- Label: `parallel_edge_semantics_gap`.
- Estimated fix size: M.

### 11. P2: Graphviz fdp spline/compound-edge routing is out of scope for Dagua FMMM

- Graphviz: after node placement, `fdpSplines()` routes edges according to edge type, including
  compound cluster-aware splines, at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:1035-1061`.
- Graphviz: `compoundEdges()` builds obstacle lists from nodes and clusters, calls pathplan, and
  constructs splines or self arcs at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/clusteredges.c:200-270`.
- Dagua FMMM: `layout_fmmm_pipeline()` returns only position tensors at
  `dagua/layout/ops/pipelines/fmmm.py:97-187`.
- Impact: position-only fidelity ignores this, but full fdp-mode visual parity cannot ignore it.
- Label: `spline_routing_gap`.
- Estimated fix size: L, probably render/routing scope and explicitly outside this no-edit round.

### 12. P2: Graphviz fdp postprocess/aspect/input-scale surface is missing

- Graphviz: `fdp_layout()` saves/restores `PSinputscale`, calls `get_inputscale()`, applies
  `neato_set_aspect()`, and runs `gv_postprocess()` at
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:1063-1078`.
- Graphviz: `finalCC()`, `evalPositions()`, and `setBB()` translate nested positions, boxes, labels,
  and margins at `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:71-175`,
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:228-269`, and
  `/home/jtaylor/projects/_references/graphviz/lib/fdpgen/layout.c:925-939`.
- Dagua FMMM: `_FinalizeFMMMPositions` normalizes positions for the returned tensor; it does not
  emit Graphviz bboxes, label margins, or aspect-adjusted JSON geometry.
- Impact: less important for RMSD after Procrustes normalization, but important for any binary-level
  fdp output comparison.
- Label: `postprocess_surface_gap`.
- Estimated fix size: M-L.

## Fix priority

1. Do not continue trying to tune OGDF-FMMM to Graphviz fdp. It can improve numeric proximity but
   cannot cover fdp's derived-graph recursion, ports, cluster expansion, or overlap expansion.
2. Add an explicit `fdp` adapter/pipeline if Graphviz-fdp fidelity matters. The first useful subset
   is: random/pinned init, Graphviz force laws, `K`/`len`/`weight`, linear cooling, 600 maxiter,
   component packing.
3. Treat clusters/ports/compound splines as a second stage. They are essential for full fdp parity,
   but too large to mix with the point-force implementation.
4. Keep `classic_fmmm` aimed at OGDF FMMM. Round 21 and later OGDF fixes should not be overloaded
   with Graphviz-fdp-specific behavior.

## Dead code / removable notes

None identified in this diagnostic pass. No code was refactored.
