# Task & Bug Tracker

## Active Tasks
- [ ] [HIGH] **First-class `dagua.quality` + `dagua.compare` modules (productize the layout comparators).**
  JMT greenlit 2026-06-03: promote the layout-equivalence/quality metrics from internal eval utilities
  (`dagua/eval/equivalence_metrics.py`, committed f9d18e1 + per-component/per-axis extension) into a
  first-class public "measure & compare" pillar alongside layout + render. Do NOT rewrite -- GRADUATE the
  validated core after the RNG-matching sprint has exercised it.
  - **`dagua.quality(pos, graph) -> QualityMetrics`** -- measure ONE layout: normalized stress, edge
    crossings, neighborhood preservation (trustworthiness), distance correlation. Pure-torch, no new deps.
    (Several already exist as differentiable losses internally -- reuse, don't duplicate.)
  - **`dagua.compare(pos_a, pos_b, graph, invariances=(...)) -> ComparisonResult`** -- how-equivalent, via
    the FIVE principled invariances (the agreed ceiling; anything beyond launders real differences):
    1 rigid Procrustes, 2 graph automorphisms, 3 degenerate-eigenspace (spectrum/distance diagnostic),
    4 per-connected-component rigid placement, 5 per-axis anisotropic scaling (OPT-IN per free-aspect
    engine, default {sugiyama}). Emits all raw signals + a combined PRACTICALLY_EQUIVALENT verdict.
  - **Dependency boundary (respect PyTorch-only principle):** pure-torch invariances (rigid, per-axis,
    per-component decomposition can be torch/SciPy, distance/spectrum) + all quality metrics go in CORE,
    zero new required deps. The AUTOMORPHISM-aware comparison needs igraph (BLISS) -> OPTIONAL extra
    `pip install dagua[compare]`, degrade gracefully (skip automorphism signal) when absent. NEVER pull
    igraph into core.
  - **The differentiating feature -- differentiability:** the differentiable subset (stress, orthogonal
    SVD-Procrustes) doubles as LOSSES/OBJECTIVES. Expose a **`MatchTarget` loss / `LayoutConfig(match_target=...)`**
    -> "lay out this graph to MATCH this reference layout" as a differentiable target. Uniquely dagua (ties
    into the "constraints are loss functions" principle). Automorphism-min / component-align have discrete
    argmin -> evaluation-only (document which metrics can serve as objectives vs eval-only).
  - **Architecture:** clean third pillar `dagua/quality/` + `dagua/compare/` (or `dagua/metrics/`); keep
    cohesive, don't sprawl. Reproducibility use case (CPU-vs-CUDA layout equivalence) is a headline demo.
  - **GATES:** (a) public-API surface needs JMT sign-off before building (API design = discuss-first);
    (b) promote only the SPRINT-VALIDATED core, not untested code; (c) docs + glossary + a notebook example.
  - NEXT STEP when ready: CC drafts the public API design doc (signatures, differentiable-vs-eval split,
    optional-dep boundary, MatchTarget loss) for JMT to react to, THEN implement.
- [ ] [HIGH] **Sprint B -- Cairo opt-in (post-data-coord sprint).** Add cairo as a matplotlib backend option:
  - `pyproject.toml` optional dep: `[project.optional-dependencies] cairo = ["mplcairo>=0.6"]`
  - Auto-detect default per `feedback_cairo_default_policy.md`: try `import mplcairo` -> use cairo; else fall back to Agg. NO explicit user config required for default selection. `pip install dagua` -> Agg; `pip install 'dagua[cairo]'` -> cairo.
  - Public override: `dagua.render(g, pos, backend="agg" | "cairo")` and `dagua.set_default_backend(...)` for global override
  - Per-figure canvas attach (no `pyplot` reliance; uses `Figure(...)` + `FigureCanvasAgg(fig)` or `mplcairo.FigureCanvas(fig)`). Round 13 prepares for this by auditing `pyplot` leakage.
  - Comparison gallery: render existing Tier A cards under both backends side-by-side; quantify the visual delta (Sprint B ceiling-test).
  - Tests parametrized over both backends; cairo tests `pytest.mark.skipif(not has_mplcairo)`.
  - README docs explaining the install paths + libcairo system-dep note for Linux/Mac.
  - Iterate on cairo gallery until Opus 4.7 visual auditor returns no fixable findings ("ceiling" same as data-coord sprint).
  - **WAIT for Sprint A (round 13) to fully converge before kicking off.**
- [ ] [HIGH] **Round 13 -- data-coord-everything sweep.** Audit `dagua/render/mpl.py` for ALL display-point leakage and convert to data-coord with `display_scale` conversion at the rendering boundary. Specifically:
  - Revert round-11 thin-edge fallback (`_edge_uses_display_stroke_body` at mpl.py:5583, fallback path at mpl.py:6620). Replace with data-coord ribbon + explicit `display_scale`-based minimum-width clamp on the existing data-ribbon path. Clamp activates at render time only; optimizer sees true data-coord value.
  - Audit ALL `linewidth=` and `fontsize=` calls in `dagua/render/mpl.py`. Each one routes through `display_scale` or gets replaced with data-coord polygon path.
  - `style.stroke_width` (border) -> data-coord ribbon (with display_scale clamp)
  - `style.width` (edges) -> data-ribbon throughout, no display-point fallback
  - `style.font_size` -> data-coord (per `feedback_data_coord_fonts.md`)
  - Add regression test asserting calibrate-once invariant: render same graph at multiple `dpi` values, verify relative geometry (stroke-width-as-fraction-of-node-width, font-size-as-fraction-of-node-height) is identical across dpi values. Catches future display-point regressions automatically.
  - Re-run gallery_audit + per_card_pixel_diff, verify round-11 wins preserved (edge stem visible on box3d/circle/etc, labels readable on combo_pie_bold) under the new data-coord path.
  - **WAIT for JMT signal to begin.** Then do cairo-vs-Agg discussion next.
- [ ] [HIGH] Benchmark finishing (~95.5%) -- run fidelity pipeline when done
- [ ] [HIGH] **Investigate dagua native CUDA OOM** seen in 100-seed benchmark run (2026-04-30 to ~2026-05-04).
  33 errors of "CUDA driver error: out of memory" all on the `dagua` native engine.
  Should NOT be happening; suggests a graph or batch size that exceeds GPU memory. After the 100-seed benchmark finishes, query `eval_output/benchmark_100seed_final/results.json` for entries with `engine=dagua status=error error_message="CUDA driver error: out of memory"` to find the offending (graph, seed) pairs, reproduce, fix the OOM (likely batching or fallback to CPU on memory pressure).
- [ ] [HIGH] Continue Graphviz theme calibration until critics reach min>=8, mean>=9
- [ ] [MED] Close fidelity gaps -- add reimplementations for unpaired algorithms:
  - [ ] Reimplement fcose (Cytoscape force-directed) in PyTorch
  - [ ] Reimplement YifanHu (Gephi force-directed) in PyTorch
  - [ ] Fix LinLog original pairing -- OGDF says "unsupported"
- [ ] [LOW] Fix UMAP on disconnected graphs: replace inf shortest-path distances with 2*max(finite_distances)

## Roadmap -- completing the dagua vision (strategic, post-sprint)
The coherent identity: a rigorous, measurable, *differentiable* layout engine -- reproduce any classical
algorithm, then bend it with custom loss functions, on GPU. The bit-exact algo reproduction is parity
(table stakes); the MOAT is the differentiable custom-objective layer. These items make the star as
strong as the supporting cast + add the gradient-native capabilities Graphviz structurally can't offer.
(Companion: the first-class `dagua.quality` + `dagua.compare` task under Active Tasks is part of this vision.)

- [ ] [HIGH] **Make the constraint/objective layer the HEADLINE pillar.** It is dagua's differentiator and
  must be at least as first-class/polished/documented as the algos and renderer.
  - Audit current maturity: Flex (pin/align), LayoutFlex (firm/soft), the custom-loss API, the existing
    loss library (see memory/loss_functions.md). What exists vs what's missing/rough.
  - Stock a reusable, named constraint/loss LIBRARY: pin, align(axis), distribute/equal-spacing,
    group/cluster containment, flow-direction (DAG top-down), orthogonality, edge-length uniformity,
    label-non-overlap, symmetry, soft-crossing-minimization, MatchTarget (from the compare work).
  - Ergonomics: confirm + document the "custom constraint in 3 lines" promise with real examples;
    composition with weights/Flex; clear docs/guide + notebook + glossary entries.
  - Positioning: foreground this as THE reason to use dagua over Graphviz (README/docs/tutorial).
  - GATE: this is the soul -- prioritize once the RNG-matching/4-tier sprint lands.
- [ ] [MED] **Dynamic / incremental layout** (uniquely enabled by the stateful gradient formulation).
  - Warm-start: `layout(g2, init_from=prev_pos)` -- when the graph changes slightly (add/remove
    nodes/edges), initialize from the previous layout + re-optimize -> STABLE layouts under edits (a
    known-hard problem classical engines handle badly).
  - Animation: expose the optimization trajectory (per-step intermediate positions) so users can animate
    convergence, and animate graph-edit transitions (old layout -> perturb -> resolve).
  - Mental model: positions are persistent learnable state; graph edits = perturb + re-solve. Use cases:
    evolving graphs, interactive pin-and-resolve, training-time NN graph evolution (TorchLens consumer).
- [ ] [MED] **Edges as learnable parameters** (completes "everything is a differentiable parameter").
  - Make edge control points (bezier/polyline) first-class differentiable params with their own losses:
    bundling (group near-parallel edges), node/obstacle avoidance (route around nodes), curvature
    smoothness, length penalty, crossing-angle maximization.
  - Ties into the existing in-flux "edge optimization" + the weighted-edges TODO. Jointly-optimized or
    as a post-pass over fixed node positions.
- [ ] [MED] **Reasonable EXPORT options** (serve the interactivity crowd via data export, NOT by building
  an interactive framework -- keep the headless-engine identity; JMT 2026-06-03).
  - Principle: dagua computes the layout; export lets ANY downstream tool (incl. interactive web) consume
    it. dagua does not render widgets.
  - `dagua.export(g, pos, format=..., output=...)` supporting:
    - **SVG** (vector, editable -- confirm/polish existing).
    - **Structured layout JSON** (tool-agnostic: nodes [id,x,y,style,bbox], edges [endpoints,route,style],
      graph bounds) -- consumable by d3.js / Observable / custom viewers.
    - **Cytoscape.js JSON** (popular interactive web graph lib).
    - **GraphML / GEXF with coordinates** (Gephi / yEd / standard interchange).
    - **DOT with fixed `pos=`** (Graphviz-ecosystem interop).
    - Plain positions (array / DataFrame / .pt) for programmatic use.
  - PyTorch-only core; pull heavy format libs only as optional extras. Data export, not a render framework.

## Bugs
- [ ] [HIGH] Arrowheads placed INSIDE target node boundary instead of OUTSIDE. Root cause: edge router computes endpoint at node boundary, arrowhead extends backward into gap but tip overlaps node surface.
- [ ] [HIGH] Cluster API overhaul -- clusters as first-class members with auto-detected nesting, strict tree validation, auto-propagated membership
- [ ] [MED] Cluster label collision: sibling labels merge into unreadable text. Auto-shrink font on colliding labels.
- [ ] [MED] Long label text overflow: ellipse aspect ratio not adjusting correctly for very wide text with overflow_policy="expand_node"
- [ ] [LOW] Arrowhead sizing slightly smaller than Graphviz's chunky triangles

## Cosmetic Polish (Maybe)
- [ ] Text background corners match node corner_radius
- [ ] Cloud/organic shape text bg follows contour
- [ ] Dash spacing adapts to curvature
- [ ] Hub arrowhead distribution around perimeter
- [ ] Shadow contour matching for elliptical nodes
- [ ] Self-loop arc height tightening
- [ ] Cluster shape variants: pill, cloud, convex hull

## Feature Parity Gaps (static/cosmetic, non-interactive)
- [ ] [HIGH] Record/table nodes -- structured multi-field content inside a node (Graphviz records, yFiles table nodes, database/UML diagrams)
- [ ] [MED] Text on path -- edge labels that curve along the edge path instead of sitting in a box (yFiles, D3)
- [ ] [MED] Intra-node compositional layout -- sub-elements within a node (text + icon + bar chart); GoJS "panels" (horizontal, vertical, table, spot)
- [ ] [MED] Edge bundling rendering -- visually merge parallel edges into a thick band that splits at endpoints
- [ ] [LOW] ~20 niche node shapes from Graphviz: house, invhouse, folder, component, promoter, cds, septagon, egg, point, etc. (bio/UML-specific)

## Roadmap (Post-Benchmark)

### Architecture
- [ ] [HIGH] **Fidelity vs Performance modes**: every fidelity-mode pipeline currently uses sequential Python scalar loops to match the C/C++ reference arithmetic order bit-for-bit. This is correct for the "23-of-24 bit-exact against reference" claim but 50-100x slower than the references for algorithms with many inner iterations (davidson_harel_rounds200, sgd2_multi, lgl_iter300, drl simmer, etc.). Add an explicit `mode='fast'` opt-out that runs the same algorithm via torch reductions / numba JIT / C extension, sacrificing literal arithmetic-order match for speed. Document that `mode='fidelity'` (default for benchmark fidelity_mode) is reproducibility-first while `mode='fast'` is throughput-first. Both should produce GEOMETRICALLY equivalent layouts (RMSD < 1e-3) but only fidelity matches reference bit-for-bit. Discovered post-R66: R66 spent hours on the slow tail of those specific variants.
- [ ] [HIGH] Original algorithm backend: `dagua.layout(g, algorithm="fr", backend="original")` -- user-facing transparency, runs literal originals via competitor adapters
- [ ] [HIGH] Pixel-unit overrides as OPT-IN OVERRIDE: "2pt" / "1.5px" syntax for fixed-size elements (NodeStyle.stroke_width_override, EdgeStyle.width_override, NodeStyle.font_size_override). Default path stays data-coord (per `feedback_data_coord_everything_strict.md`); override values bypass data-coord and route directly to display-points. Document loud and clear that override values are NOT differentiable and break the calibrate-once invariant. Useful when users want literal point-perfect typography for paper figures.
- [ ] [HIGH] Expose text rendering in style fields: text_background, text_underline, text_strikethrough, label_outline

### Layout Quality
- [ ] [HIGH] Theme-suggested layout spacing: suggested_node_sep/rank_sep on Theme
- [ ] [MED] Auto algorithm selection: benchmark all competitors per graph structure, auto-pick winner
- [ ] [MED] Incremental layout: warm-start from previous positions on graph changes
- [ ] [MED] Adaptive fine-level refinement for large-scale graphs

### Rendering
- [ ] [HIGH] Port constraints: named ports on nodes for circuit/flowchart diagrams
- [ ] [HIGH] Edge label collision avoidance
- [ ] [HIGH] Edge-following rotated labels: rotate edge labels to curve tangent (~30 lines)
- [ ] [MED] Orthogonal edge routing with obstacle avoidance
- [ ] [MED] Octilinear edge routing: 45-degree angle mode between ortho and polyline (~100 lines)
- [ ] [MED] Edge bundling: group parallel edges with configurable bundling strength (~200 lines)
- [ ] [MED] Polyline routing with user-defined waypoints/bends (~100 lines)
- [ ] [MED] Icons/images in labels: inline images with text (~150 lines)
- [ ] [MED] Node badges/decorators: overlay indicators at configurable positions (~100 lines)
- [ ] [MED] Pattern fills (tiling image): tile small image across node fill (~50 lines)
- [ ] [MED] Tab header on cluster groups: folder-tab protrusion with label (~80 lines)
- [ ] [MED] Table/swimlane nodes for BPMN
- [ ] [MED] Self-loop style: configurable (semicircular vs compact side-attached)
- [ ] [MED] Edge endpoint offset for arrowhead placement (Graphviz matching)
- [ ] [HARD] Text wrapping to non-rectangular shapes (ellipse, diamond, triangle) (~200 lines)
- [ ] [HARD] Rich text via HTML/markup subset in labels (~300 lines)
- [ ] [HARD] Swimlane/table node with internal grid layout (~500 lines)

### Import Adapters (high star potential -- no good static diagram export exists for most)
- [ ] [HIGH] n8n importer: `dagua.from_n8n(workflow_json)` + CLI `dagua render-n8n` (181K stars, ZERO static export)
- [ ] [HIGH] Airflow importer: `dagua.from_airflow(dag)` -- parse DAG Python or serialized JSON
- [ ] [HIGH] dbt importer: `dagua.from_dbt(manifest_json)` -- parse manifest.json model lineage
- [ ] [HIGH] GitHub Actions importer: `dagua.from_github_actions(workflow_yaml)` -- parse job/step DAG
- [ ] [MED] Terraform importer: `dagua.from_terraform(plan_json)` -- parse `terraform graph` or plan JSON
- [ ] [MED] AWS Step Functions importer: `dagua.from_step_functions(asl_json)` -- parse ASL state machine
- [ ] [MED] Dagster importer: `dagua.from_dagster(job)` -- parse asset/op dependency graph
- [ ] [MED] Argo Workflows importer: `dagua.from_argo(workflow_yaml)` -- parse DAG/steps template
- [ ] [MED] Prefect importer: `dagua.from_prefect(flow)` -- parse flow/task dependencies
- [ ] [LOW] Kubernetes importer: `dagua.from_kubernetes(manifests)` -- parse resource dependencies
- [ ] [LOW] Zapier importer: `dagua.from_zapier(zap_json)` -- parse trigger/action chains
- [ ] [LOW] Luigi importer: `dagua.from_luigi(task)` -- parse task dependency graph
- [ ] [LOW] Make.com (Integromat) importer

### Infrastructure
- [ ] [MED] Image galleries via GitHub Pages
- [ ] [MED] Interactive rendering: pan/zoom, tooltips, collapsible clusters (WebGL/Canvas)
- [ ] [MED] 3D graph rendering
- [ ] [LOW] yFiles benchmark comparison (eval license)
- [ ] [LOW] Print-quality PDF with embedded fonts

### Performance (Large Scale)
- [ ] Fix 2K performance cliff
- [ ] Fused edge loss CUDA kernel
- [ ] V-cycle pipeline parallelism
- [ ] GPU-accelerated coarsening
- [ ] Fix fanout_distribution_loss hub count mismatch at 200M+ scale

## Completed (This Sprint)
- [x] Composable ops migration: 268 ops across 34 modules, 23 algorithm pipelines, all via pure op composition. Monoliths archived at _archive/classic/ as frozen test oracles. Bitwise-identical seed-matched fidelity validation (2,532 tests, 371 pipeline fidelity).
- [x] algorithm_params in LayoutConfig: `LayoutConfig(algorithm="fr", algorithm_params={"cooling": 0.95})`
- [x] Full variant benchmark: 104 variants x 91 graphs x 30 seeds (432K evals)
- [x] Benchmark pitstop: skip-after-3 errors, graph-size timeout, rolling submission, SIGINT handler, watchdog executor
- [x] 4 new competitors: cytoscape fcose, gephi yifanhu, FR->KK chain, KK->FR chain
- [x] Edge weights on all 20/20 classic algorithms
- [x] 306 themes (44 -> 306) with 13 categories + list_themes()/theme_categories() API
- [x] Semicircle/semi-ellipse node shape with 4 orientations
- [x] aspect_ratio field on NodeStyle
- [x] Fidelity analysis pipeline + LaTeX report generator (adversarially critiqued)
- [x] Cosmetic feature recipe: docs/COSMETIC_FEATURE_RECIPE.md
- [x] FA2 reference adapter fix (kwarg filtering)
- [x] SGD2 multi reference adapter fix (criteria patches, vis, evaluate, .pos)
- [x] Parallel heavy engines (was serial, 86GB free)
- [x] Gallery audits Phase 1-3 + cosmetic polish sprint
