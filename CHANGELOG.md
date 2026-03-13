# CHANGELOG


## v0.1.0 (2026-03-13)

### Bug Fixes

- **bench**: Checkpoint hierarchy incrementally
  ([`f824200`](https://github.com/johnmarktaylor91/dagua/commit/f824200ebdf71acc071349a51e1666eae81b9e79))

- **bench**: Guard duplicate large runs without metadata
  ([`ad0e3a8`](https://github.com/johnmarktaylor91/dagua/commit/ad0e3a81d6e117c247f056d63898fa551da6f879))

- **bench**: Harden incremental hierarchy checkpoints
  ([`0f82deb`](https://github.com/johnmarktaylor91/dagua/commit/0f82debd8c58ec950b4044609026f544a720d43d))

- **bench**: Harden resume metadata validation
  ([`2daee67`](https://github.com/johnmarktaylor91/dagua/commit/2daee67fa851e663194094dd97aab993b1c57790))

- **bench**: Ignore shell wrappers in run guard
  ([`f969da3`](https://github.com/johnmarktaylor91/dagua/commit/f969da3736782a9ef90fa2270156710966a148fb))

- **bench**: Reject partial hierarchy resumes
  ([`0613d00`](https://github.com/johnmarktaylor91/dagua/commit/0613d005bbe83ca4d19e747c4b3916ac273e1444))

- **bench**: Require complete hierarchy for coarsest resume
  ([`811745d`](https://github.com/johnmarktaylor91/dagua/commit/811745da81834b41f59258888f3e374c9dbef0b9))

- **bench**: Shard hierarchy checkpoints
  ([`bffa454`](https://github.com/johnmarktaylor91/dagua/commit/bffa454f829ce331423316f19ec5fa4a2a455408))

- **bench**: Validate derived checkpoint signatures
  ([`e64d621`](https://github.com/johnmarktaylor91/dagua/commit/e64d6212796234da254ae97bdaae85006b7fbf4d))

- **bench**: Validate large checkpoint invariants
  ([`da554ec`](https://github.com/johnmarktaylor91/dagua/commit/da554ec6eb02b3f282fac3958fab4ff1bfb16a7d))

- **layout**: Guard giant cuda init placement
  ([`d97c748`](https://github.com/johnmarktaylor91/dagua/commit/d97c7483ab15e33bb6b37953d46e07b71aca4f74))

- **multilevel**: Accept scalar node sizes in coarsening
  ([`0324007`](https://github.com/johnmarktaylor91/dagua/commit/0324007fbb06ab7e1e669e6fe28011169e1d049d))

- **multilevel**: Always retain coarse layer assignments
  ([`ae750ab`](https://github.com/johnmarktaylor91/dagua/commit/ae750ab77ad97d67ef82cdd0ad555c15d80c0180))

- **multilevel**: Avoid resumed layering upcast
  ([`f21ff21`](https://github.com/johnmarktaylor91/dagua/commit/f21ff2123adde3b9370512ba67249844eaa3c57f))

- **multilevel**: Harden streaming coarse size reduction
  ([`6784cd4`](https://github.com/johnmarktaylor91/dagua/commit/6784cd4bffcf445e602f79cf5038796efe94b809))

- **multilevel**: Harden streaming node size fallback
  ([`a223fa3`](https://github.com/johnmarktaylor91/dagua/commit/a223fa386ee28276c5dd0ac557b91de129eda6cc))

- **multilevel**: Preserve node size dtype in coarsening
  ([`74fbaba`](https://github.com/johnmarktaylor91/dagua/commit/74fbabac4c887dda7f98172da9f598ec2826f1a8))

- **multilevel**: Restore hierarchy size normalization
  ([`af7d8ef`](https://github.com/johnmarktaylor91/dagua/commit/af7d8efba8763edfdcdf23af6ae2010a45288397))

- **render**: Silence fallback font and figure warnings
  ([`46d5b1b`](https://github.com/johnmarktaylor91/dagua/commit/46d5b1b5658817a8feba1377aa7f7502e145dea3))

### Chores

- Add 100K node benchmark result (2096s on CPU, Graphviz N/A)
  ([`d5f596e`](https://github.com/johnmarktaylor91/dagua/commit/d5f596ec927eb9e8781198135386ee1a8d2a576f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Add 50M/100M/300M node benchmark scripts
  ([`655767f`](https://github.com/johnmarktaylor91/dagua/commit/655767f656ab0e1070432b2a07c4a07f0a0441d4))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Add eval_output to gitignore
  ([`2a1900c`](https://github.com/johnmarktaylor91/dagua/commit/2a1900cdd6cf70fed7a626668e85539e1850ede9))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Add test_output.log to gitignore
  ([`3a678c9`](https://github.com/johnmarktaylor91/dagua/commit/3a678c91ac896a3f9b0bc396b2d957e6f8a81151))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Add TODO.md, fix param sweep registry, polish final eval
  ([`1502028`](https://github.com/johnmarktaylor91/dagua/commit/150202808ca90d126f17341d40282e35b8c779ef))

- TODO.md with known issues, feature roadmap, architecture decisions - Fix PARAM_REGISTRY import in
  sweep.py (was List, needed Dict) - Evaluation: 17/18 wins vs Graphviz, 81 tests passing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Split CLAUDE.md/AGENTS.md into architect vs implementation roles
  ([`b12ad5e`](https://github.com/johnmarktaylor91/dagua/commit/b12ad5ebff9dc5c4198d90d1774135556f8093a9))

Replace symlink mirroring convention with distinct files: - CLAUDE.md = architect-level context
  (design, rationale, how modules connect) - AGENTS.md = implementation-level context (commands,
  conventions, gotchas)

Populate .project-context/ with architecture map, conventions, decisions, and gotchas. Add
  dispatch/check/clean scripts for task orchestration.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- **dev**: Tighten maintainability guidance
  ([`0dafaa9`](https://github.com/johnmarktaylor91/dagua/commit/0dafaa96bd6b3077ded8f06f64553adadbbdcc67))

- **eval**: Extend rare scaling ladder to 1b
  ([`d2e3eaf`](https://github.com/johnmarktaylor91/dagua/commit/d2e3eaf61237a2b5cd0385f545826772a213f73c))

- **layout**: Add TODOs for streaming coarsening and small-graph speedup
  ([`8d9d5df`](https://github.com/johnmarktaylor91/dagua/commit/8d9d5dff555d2fe0429b2e53a49115645d52f662))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **multilevel**: Add hierarchy progress logging
  ([`45ec6b3`](https://github.com/johnmarktaylor91/dagua/commit/45ec6b37d672379a09e50a7d20b05b968ecebbb2))

- **repo**: Add AGENTS symlinks for Claude docs
  ([`c540839`](https://github.com/johnmarktaylor91/dagua/commit/c54083943e7a0bba050684145f397442d2ca1c2b))

- **repo**: Add criteria and benchmark safeguards
  ([`a3a1e95`](https://github.com/johnmarktaylor91/dagua/commit/a3a1e95d99022de476db32abadb3a7afa062c0ee))

- **report**: Make benchmark review prompts agent-agnostic
  ([`63ad3d2`](https://github.com/johnmarktaylor91/dagua/commit/63ad3d2fd6bdc2196c6fc3c7dad56cd7117c68e5))

### Documentation

- **clusters**: Record hierarchical interaction principle
  ([`81fe2fe`](https://github.com/johnmarktaylor91/dagua/commit/81fe2fec8beee9a7641fd1bfeb216da3e21945f4))

- **competitors**: Add official reading pack
  ([`bb6eed2`](https://github.com/johnmarktaylor91/dagua/commit/bb6eed2a56a6dc629f677e12ce1b58c7fee56fd4))

- **dev**: Add end-to-end codebase overview
  ([`293cfe7`](https://github.com/johnmarktaylor91/dagua/commit/293cfe76e7a8f0baf6bd4a525c7d7c9592f2a25a))

- **dev**: Clarify scaling and comparison helpers
  ([`92c0bb8`](https://github.com/johnmarktaylor91/dagua/commit/92c0bb88bab4c202833a0ff5307c2a613ed8719a))

- **dev**: Clarify staged geometry model
  ([`bcb9185`](https://github.com/johnmarktaylor91/dagua/commit/bcb9185cf01440b40534d099a1924aade85789ce))

- **eval**: Add competitor geometry memo
  ([`ce54b1b`](https://github.com/johnmarktaylor91/dagua/commit/ce54b1b75a8a38bbd8e0a0ec1f637f5ffe2261bf))

- **eval**: Prepare iteration kitchen
  ([`bdfe6a1`](https://github.com/johnmarktaylor91/dagua/commit/bdfe6a1654bd16b951198d680920ed696cf972d0))

- **examples**: Add annotated yaml and json specs
  ([`b203d44`](https://github.com/johnmarktaylor91/dagua/commit/b203d44a46d5594c97c051184c8206d89bece901))

- **explainer**: Add public algorithm walkthrough
  ([`f437b7f`](https://github.com/johnmarktaylor91/dagua/commit/f437b7fa2108dfcf3ebe15f852930861903c551c))

- **gallery**: Add autogenerated showcase gallery
  ([`64404e1`](https://github.com/johnmarktaylor91/dagua/commit/64404e14fd4b84dce703f41323ed526ef5dccab1))

- **geometry**: Add stage-0 criteria inventory
  ([`540850a`](https://github.com/johnmarktaylor91/dagua/commit/540850aa0b0d4944ec088d8418e6c4a12a2358a3))

- **io**: Standardize yaml as human default
  ([`927741c`](https://github.com/johnmarktaylor91/dagua/commit/927741c717c60d4079a3ef642d1906c20a6f6d94))

- **llm**: Add public agent usage guide
  ([`f5965d2`](https://github.com/johnmarktaylor91/dagua/commit/f5965d227c41c6d15f1b058c9a88cb0ec387fae1))

- **maintenance**: Add regular update checklist
  ([`ce8e92d`](https://github.com/johnmarktaylor91/dagua/commit/ce8e92d41b69baaf1cb5d2c8f71d12d03aee65d8))

- **maintenance**: Refresh maintainer notes
  ([`9283457`](https://github.com/johnmarktaylor91/dagua/commit/928345710fc5df323309449d82a8d7401989233d))

- **maintenance**: Sync staged optimization guidance
  ([`f08dc85`](https://github.com/johnmarktaylor91/dagua/commit/f08dc85050010aac8780ec218f40c23e4abcf16a))

- **notebooks**: Add tutorial and QA notebooks
  ([`4bd65e0`](https://github.com/johnmarktaylor91/dagua/commit/4bd65e0ec30e1d9eb190dc0b4eaa35f7bdcf99f3))

- **notebooks**: Normalize tutorial notebook metadata
  ([`16d0b27`](https://github.com/johnmarktaylor91/dagua/commit/16d0b279a7f957606262ea844d511ed11ea0c15d))

- **readme**: Add user faq
  ([`731df05`](https://github.com/johnmarktaylor91/dagua/commit/731df051553cdc0a70989c43a94512103c2a2b01))

- **reference**: Add exhaustive glossary manual
  ([`e0c327d`](https://github.com/johnmarktaylor91/dagua/commit/e0c327d97c9d8b141b8dedc453c6af2c4ffb9402))

- **repo**: Add workflow and status references
  ([`132ca62`](https://github.com/johnmarktaylor91/dagua/commit/132ca62608484f459216f149643dddd995ce0628))

- **status**: Record placement benchmark baseline
  ([`463ff8b`](https://github.com/johnmarktaylor91/dagua/commit/463ff8b06161e5b5b7f60e02c3cb46b90a4bb692))

- **tests**: Add UI feature playground notebook
  ([`adec2bb`](https://github.com/johnmarktaylor91/dagua/commit/adec2bbf89f66e2155e29284a61203d43e33ee6a))

- **todo**: Note small-graph runtime tradeoff
  ([`0763611`](https://github.com/johnmarktaylor91/dagua/commit/07636115f183f04c8d09117bcfc454f86de4f959))

- **tutorial**: Use animation to teach constraints
  ([`d645aa1`](https://github.com/johnmarktaylor91/dagua/commit/d645aa151b7385ac1b223e499d11c3c0a5e74185))

- **workflow**: Add placement sprint prep
  ([`577a6c1`](https://github.com/johnmarktaylor91/dagua/commit/577a6c1dbbc0ec9b01049b81f859fcb6a3c0f186))

- **workflow**: Align artifact and contributor guides
  ([`f164f1d`](https://github.com/johnmarktaylor91/dagua/commit/f164f1daef95418ae2b3f5830cfc925338382b60))

- **workflow**: Extend baseline and money graph guides
  ([`edde503`](https://github.com/johnmarktaylor91/dagua/commit/edde503e309cf374623909d66f2732df55c46233))

- **workflow**: Record staged geometry optimization plan
  ([`83478ca`](https://github.com/johnmarktaylor91/dagua/commit/83478cac1b1367d3c1eab3eba782584f9dc356bf))

- **workflow**: Tighten iteration navigation
  ([`c584cef`](https://github.com/johnmarktaylor91/dagua/commit/c584cef5e2a8013056e43285b2d59dbc89794867))

- **workflow**: Tighten iteration shortcuts
  ([`0b39cc2`](https://github.com/johnmarktaylor91/dagua/commit/0b39cc26b849216090dd520e5f8e4c665e53b9ef))

### Features

- Publication-quality aesthetic system — Wong palette, adaptive spacing, visual refinement
  ([`2d8582c`](https://github.com/johnmarktaylor91/dagua/commit/2d8582c17fb3e954a6ae18726954a4348a08c6eb))

Implement the Dagua Aesthetic Style Guide across the full stack:

Style system (styles.py): - Wong/Okabe-Ito colorblind-safe palette with make_fill/border_from_fill
  utilities - Muted fills (25% blend toward warm white), strong darkened borders - Font stack:
  Helvetica Neue > Helvetica > Arial > DejaVu Sans with auto-resolution - Updated NodeStyle (0.75pt
  stroke, 8.5pt font), EdgeStyle (#8C8C8C, 70% opacity), ClusterStyle (#F5F5F0, 0.5pt, progressive
  nesting colors)

Rendering (render/mpl.py): - Warm white background (#FAFAFA), not pure white - Proportional corner
  radius (18% of shorter dimension) - Smaller arrowheads (5pt × 3.5pt), edge labels offset 4pt with
  subtle bg - Cluster labels top-left, font size decreases per nesting level

Layout (engine.py + constraints.py + config.py): - Adaptive spacing: 1.3x for <20 nodes, 0.7x for
  >1000 nodes - New spacing_consistency_loss: penalizes deviation from target gap within layers -
  w_spacing=0.3 default weight

Node sizing (utils.py + graph.py): - Sans-serif text measurement, min 40×22pt, max 6:1 aspect ratio
  - Per-node style-aware sizing (respects font/padding overrides)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Runtime scaling benchmarks, TorchLens architecture suite, direction-aware metrics
  ([`f1902b9`](https://github.com/johnmarktaylor91/dagua/commit/f1902b9a9b55618f2adb4bc56e8e9f2374e97215))

- Add comprehensive runtime scaling benchmark (benchmarks/bench_layout.py) comparing Dagua vs
  Graphviz from 100 to 50K+ nodes. Dagua is 3.3x faster at 10K nodes; Graphviz times out at 20K+
  while Dagua handles 50K in ~8min.

- Extend TorchLens eval suite from 4 to 12 models covering nested modules, branching, diamond loops,
  long loops, ASPP, FPN, attention, and random architectures.

- Make metrics direction-aware: dag_fraction, edge_straightness, and x_alignment now accept a
  `direction` parameter (TB/BT/LR/RL) to correctly evaluate layouts in any orientation.

- Add 24 new tests: scaling (100-1K nodes + Graphviz comparison), edge cases (self-loops,
  disconnected, wide/dense), direction-aware metrics (BT/LR/RL), from_torchlens integration, BT/RL
  layout directions. 104 total tests pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Tiered scaling architecture — multilevel V-cycle, spectral init, RVS repulsion
  ([`870259b`](https://github.com/johnmarktaylor91/dagua/commit/870259b5f40ef60cc56b9cbe4bf82608cbb25eb5))

Extract _layout_inner() from engine.py as headless core (pure tensors, no Graph dependency). Add
  tiered dispatch: N>50K → multilevel coarsening V-cycle, else direct layout.

- multilevel.py: layer-aware heavy-edge matching, ~50% reduction/level, V-cycle with coarse layout
  (100 steps) → prolong → refine (25 steps/level) - init_placement.py: spectral init via
  torch.lobpcg Fiedler vector for N>10K, falls back to barycenter ordering - constraints.py: RVS
  repulsion (N^3/4 active × N^1/4 random + K_nn neighbors), disabled by default — scatter sampling
  more efficient at direct-layout sizes - config.py: multilevel_threshold, multilevel_min_nodes,
  rvs_threshold, rvs_nn_k

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Vram-aware memory optimization — 20M nodes on GPU
  ([`667c267`](https://github.com/johnmarktaylor91/dagua/commit/667c267dd4fda06188262c9ee1919943523bc20d))

Three composable memory optimizations, auto-selected based on available CUDA VRAM via
  torch.cuda.mem_get_info():

1. Per-loss backward: backward each loss term separately, freeing intermediates between terms. 3-4x
  peak memory reduction, no speed cost. Auto: when estimated memory exceeds available VRAM.

2. Gradient checkpointing: recompute forward activations during backward. ~2x additional memory
  reduction, ~30% more compute. Auto: when per-loss alone isn't enough.

3. Hybrid device: heavy losses (repulsion, overlap) on CPU, edge losses + optimizer on GPU. Only
  [N,2] gradient transfers between devices. Auto: last resort when GPU can't fit even checkpointed
  intermediates.

Auto-escalation: standard → per_loss_bw → +checkpointing → +hybrid.

Power user overrides: per_loss_backward/gradient_checkpointing/hybrid_device = "on"/"off"/"auto" in
  LayoutConfig.

Results on RTX 2080 Ti (11GB): - 20M GPU: 339s (was OOM). Auto picks per_loss_bw + checkpoint. - 20M
  CPU: 1372s. GPU gives 4x speedup. - 5M GPU: 22s (standard mode, fits easily).

Add 20M rare test (CPU only — GPU depends on available VRAM).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **api**: Add draw direction override
  ([`3231687`](https://github.com/johnmarktaylor91/dagua/commit/3231687eadfb9c95302884a20db49a8b988c7748))

- **api**: Add inspectable layout lifecycle state
  ([`7173b05`](https://github.com/johnmarktaylor91/dagua/commit/7173b05656dc89efc6f0528ecda606b77a863bb0))

- **bench**: Add large benchmark graph checkpoints
  ([`d4ef5c9`](https://github.com/johnmarktaylor91/dagua/commit/d4ef5c95a4ee67418713766ccb968872817d6020))

- **bench**: Checkpoint billion-scale layering
  ([`df6c49b`](https://github.com/johnmarktaylor91/dagua/commit/df6c49b29115bc1b235748f2c565c33fe151d13f))

- **cli**: Add benchmark inventory commands
  ([`25877ed`](https://github.com/johnmarktaylor91/dagua/commit/25877edb952ebac3d5835a5559c46945e59223aa))

- **cli**: Add benchmark report and watch commands
  ([`503ab37`](https://github.com/johnmarktaylor91/dagua/commit/503ab37a52f1bcd220695dac0b6170535b20fbef))

- **cli**: Add cinematic export commands
  ([`588882d`](https://github.com/johnmarktaylor91/dagua/commit/588882d3c113329923499d06f5991c52e20d2062))

- **cli**: Add fast visual audit workflow
  ([`6c8203f`](https://github.com/johnmarktaylor91/dagua/commit/6c8203f1d90c062e886eb830b843ee7e2559aeb7))

- **cli**: Add large benchmark status helper
  ([`a97bfa7`](https://github.com/johnmarktaylor91/dagua/commit/a97bfa74ac80d728aaee6f03f894d2cb4915b85e))

- **cli**: Add run freeze and compare commands
  ([`2313e88`](https://github.com/johnmarktaylor91/dagua/commit/2313e88062458c6e64f8ce58406aae1bd0aa6579))

- **clusters**: First-class cluster hierarchy with edge routing
  ([`6dd7d00`](https://github.com/johnmarktaylor91/dagua/commit/6dd7d00af1e9e207dd7c7f72ea706fb79bd6b806))

- Parent-based API: add_cluster("inner", members, parent="outer") with cycle detection and
  dict-of-dicts auto-conversion - Computed properties: cluster_depth, cluster_children,
  leaf_cluster_members, max_cluster_depth, cluster_ids (per-node LongTensor for metrics) -
  cluster_containment_loss: keeps child bboxes inside parent bboxes - cluster_separation_loss: now
  hierarchy-aware (only sibling clusters repel) - cluster_compactness_loss: handles nested dict
  members - Cluster-aware edge routing: deflects bezier control points around foreign cluster bboxes
  in both heuristic routing and differentiable edge optimization - True hierarchy depth in rendering
  (parent chain, not leaf-count sort hack) - JSON IO: parent field serialization with backwards
  compatibility - LLM prompt updated with nested cluster example - 26 new tests covering API,
  constraints, integration, IO, routing, rendering

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Implement full layout engine, renderer, and graph data structures
  ([`d6f4279`](https://github.com/johnmarktaylor91/dagua/commit/d6f4279ca1733eb82b32fa07383be9c1ed952897))

Phase 1-5 of MVP build: - DaguaGraph with from_edge_list, from_networkx, from_edge_index,
  from_torchlens - 10 differentiable loss functions (DAG ordering, attraction, repulsion, overlap,
  cluster compactness/separation, crossing, straightness, length variance) - Hybrid init:
  topological layering + barycenter x-ordering - Projected gradient descent with hard overlap
  resolution - Bezier edge routing with port ordering - Full matplotlib renderer (nodes, edges,
  labels, clusters) - Aesthetic quality metrics (crossings, overlaps, DAG fraction, etc.) -
  LayoutConfig with full parameter registry - Style system with themes and per-node-type styling -
  CPU + CUDA support

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **edges**: Add differentiable edge optimization, label placement, and overflow policy
  ([`77e7c4c`](https://github.com/johnmarktaylor91/dagua/commit/77e7c4cce28c028555f5a4c5e304d1c31c6f3aaa))

Extend the dagua pipeline with gradient-based edge routing optimization, collision-avoiding label
  placement, curvature-aware bezier routing, and configurable node text overflow policies.

New pipeline: layout → route_edges → optimize_edges → place_edge_labels → render

- styles.py: 6 new fields (curvature, label_position, port_style, label_avoidance, overflow_policy,
  min_font_size) - utils.py: compute_node_size returns 3-tuple with effective font size, supports
  shrink_text/expand_node/overflow policies - graph.py: node_font_sizes tensor populated by
  compute_node_sizes() - edges.py: curvature threading, center port style, place_edge_labels() -
  layout/edge_optimization.py: NEW — batched bezier eval, 5 loss functions (crossing, node-crossing,
  angular resolution, curvature consistency/penalty), Adam optimizer with gradient clipping -
  config.py: 7 new LayoutConfig fields for edge optimization - metrics.py: 4 new metrics
  (edge_node_crossing_count, label_overlap_count, edge_curvature_consistency,
  port_angular_resolution) - render/mpl.py: accepts pre-computed curves/labels, per-node font sizes
  - __init__.py: draw() runs full pipeline with edge optimization

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **engine**: Multi-cpu workers for hybrid losses + user-friendly progress reporting
  ([`0231236`](https://github.com/johnmarktaylor91/dagua/commit/0231236d5428285828f794e5f78cad46e15d0269))

Add num_workers config for parallel hybrid-mode loss computation via ThreadPoolExecutor (overlaps
  CPU repulsion/overlap with GPU edge losses). Unify verbose output under [dagua] prefix with phase
  labels, hierarchy timing, indented level headers, and simplified done messages.

Also fix DaguaGraph.from_edge_list() double-counting nodes when num_nodes is passed explicitly.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **eval**: Add benchmark status controls
  ([`52bb944`](https://github.com/johnmarktaylor91/dagua/commit/52bb94420597aa1c6874bffd9126139df7114dca))

- **eval**: Add competitor stepwise visual workflow
  ([`82dd791`](https://github.com/johnmarktaylor91/dagua/commit/82dd7919b17847ce5d6c7a0398f64607d0f9aca1))

- **eval**: Add evaluation suite with Graphviz comparison and parameter sweeps
  ([`70d42a7`](https://github.com/johnmarktaylor91/dagua/commit/70d42a7209215f008a33afc015bd2f27deeaa427))

- graphviz_utils.py: DOT export, Graphviz layout parsing, side-by-side comparison - eval/graphs.py:
  14+ test graphs covering all structural categories + TorchLens traces - eval/compare.py: automated
  Dagua vs Graphviz comparison with metrics - eval/sweep.py: focused and interaction parameter sweep
  engines - eval/report.py: grid generation, comparison grids, HTML dashboard - eval/quick.py: CLI
  entry point for quick evaluation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **eval**: Add numbered visual review workflow
  ([`ea073d8`](https://github.com/johnmarktaylor91/dagua/commit/ea073d8cd033dd56b4ee3166573b420b46f8b7c8))

- **eval**: Add offline aesthetic review workflow
  ([`8256fe1`](https://github.com/johnmarktaylor91/dagua/commit/8256fe19bcd1cc74d71c9cfdb4fcc5ed8a1f5b6b))

- **eval**: Add persistent benchmark and report pipeline
  ([`adced3e`](https://github.com/johnmarktaylor91/dagua/commit/adced3e78e2e90277a48f88772e9425f1ef43c5a))

- **eval**: Add resumable benchmarks and poster renders
  ([`e0bdaf5`](https://github.com/johnmarktaylor91/dagua/commit/e0bdaf5042676e13734929407a6a6176c1063b0a))

- **eval**: Add scale graph generators and consolidate bench scripts
  ([`6ad8070`](https://github.com/johnmarktaylor91/dagua/commit/6ad80701d5406bf6cd42261028972d6f6ab80722))

Add 3 new graph generators (make_grid, make_sparse_layered, make_powerlaw_dag), fix make_bipartite
  O(n²) edge blowup, and add get_scaling_collection() spanning 50 to 2M nodes across 5 topologies.
  Merge 4 separate bench_*.py scripts into scripts/bench_large.py with presets (50m, 100m, 300m, 1b)
  and CLI args. Relax test_500_nodes timing assertion (60s → 120s) to match actual runtime.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **eval**: Add staged placement tuning pipeline
  ([`ef1f135`](https://github.com/johnmarktaylor91/dagua/commit/ef1f135125230e618600c3f24798362de5026d7c))

- **eval**: Add visual audit iteration suite
  ([`1120ace`](https://github.com/johnmarktaylor91/dagua/commit/1120ace1818ddb3a229c53eeb439169d1eb4bb2b))

- **eval**: Checkpoint standard benchmark runs
  ([`51434d6`](https://github.com/johnmarktaylor91/dagua/commit/51434d693c698490725566f289857e5f85766d7d))

- **eval**: Competitive benchmarking pipeline — 9 layout engines, scale tiers, markdown reports
  ([`1bf55ec`](https://github.com/johnmarktaylor91/dagua/commit/1bf55ec11c28a6d0236e8a2fffa778eb692cc393))

Add automated benchmark harness comparing dagua against graphviz (dot/sfdp/neato/fdp), ELK layered,
  dagre, and NetworkX (spring/kamada_kawai) on identical graphs from 100 to 50M+ nodes. Runnable via
  `python -m dagua.eval.benchmark`.

- Competitor adapter pattern: base class + registry in dagua/eval/competitors/ - Scale graph
  generators: chain, wide_dag, random_dag, diamond, tree, bipartite - get_scale_suite(tier) returns
  small/medium/large/huge graph sets - Main harness with per-layout timeout, metrics computation,
  JSON + markdown output - generate_benchmark_markdown() produces GitHub-viewable report with
  summary + per-tier tables

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **eval**: Improve placement iteration workflow
  ([`362fac1`](https://github.com/johnmarktaylor91/dagua/commit/362fac1c3f48bac1171fc7e429b479a2f2ca9a16))

- **graph**: Add configurable storage dtypes
  ([`ce2ebf6`](https://github.com/johnmarktaylor91/dagua/commit/ce2ebf694205fd7cbe0d1cb258c2c488a39fcead))

- **io**: Add comprehensive import/export and multi-engine comparison infrastructure
  ([`3f90ac7`](https://github.com/johnmarktaylor91/dagua/commit/3f90ac7f0ce517a59b57f90450aece1b4fa9ac15))

- Export: to_networkx, to_igraph, to_pyg, to_scipy with try/import guards - Import: from_igraph,
  from_scipy, from_dot (pydot-based DOT parsing) - Graph.py thin wrappers for all new functions
  (methods + classmethods) - igraph competitor adapters: sugiyama, fruchterman_reingold,
  reingold_tilford - N-engine visual comparison: render_multi_comparison(), compare_engines(),
  MultiComparisonResult, generate_multi_comparison_grid(), print_multi_comparison_table() - Optional
  deps in pyproject.toml: [igraph], [scipy], [pydot], [interop] - 36 new tests (100/100 IO+eval
  pass, 144/144 smoke pass)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **io**: Add graph-from-JSON, graph-from-image, and theme-from-image
  ([`e7de4e4`](https://github.com/johnmarktaylor91/dagua/commit/e7de4e44e7d704aa7943c98583c97db329e403ea))

Implement three new features for reconstructing graphs programmatically: - DaguaGraph.from_json() /
  to_json() for JSON import/export - dagua.from_image() to extract graph structure from images via
  LLM - dagua.theme_from_image() to extract visual themes from images via LLM

LLM integration supports Anthropic and OpenAI with auto-detection from env vars. Returns structured
  JSON (never executable code) for safety. Includes 34 tests (28 smoke, 6 mock-based LLM tests).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **io**: Add magical image-to-code script mode
  ([`564e521`](https://github.com/johnmarktaylor91/dagua/commit/564e521416c0ce2166bd27dd6f6021d757c6bb9f))

- **io**: Add YAML/JSON graph IO system with unified load/save API
  ([`15330ef`](https://github.com/johnmarktaylor91/dagua/commit/15330ef3d80db8bce89b5814315c699c732a191f))

- Add YAML import/export (graph_from_yaml, graph_to_yaml) with PyYAML as optional dep - Add unified
  load()/save() with format auto-detection from file extension - Add theme registry (THEME_REGISTRY,
  get_theme) for theme-by-name resolution in YAML - Refactor graph_from_json to use shared
  _graph_from_dict (supports theme: "dark" strings) - Add DaguaGraph.load/save/from_yaml/to_yaml
  classmethods - Add dagua/graphs/ bundled graph library (diamond, pipeline, neural_net,
  nested_clusters) - Export new API at top level: load, save, graph_from_json/yaml,
  graph_to_json/yaml, get_theme - 32 new tests covering YAML, unified API, theme registry, bundled
  graphs, classmethods

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **io**: Finish image to graph and theme workflow
  ([`404dfb6`](https://github.com/johnmarktaylor91/dagua/commit/404dfb68ef651f3fff19b02a44720708cb49a1e8))

- **io**: Normalize common image formats
  ([`7fe069b`](https://github.com/johnmarktaylor91/dagua/commit/7fe069b33210e297846845dbff17ede521ca8bc9))

- **layout**: Add aesthetic-driven loss functions and fix self-loop routing
  ([`ad48a80`](https://github.com/johnmarktaylor91/dagua/commit/ad48a80ecae3116c7f64f0adcd7be436298bbe59))

- Fix self-loop edge routing NaN: detect s==t early, generate teardrop bezier - Reduce rank_sep
  default 50→40 to fix excessive vertical stretching - Enable crossing loss by default
  (w_crossing=1.5) with interval-based amortization to keep overhead <5ms for small graphs - Add
  fanout_distribution_loss: penalizes uneven angular spread of hub children - Add
  back_edge_compactness_loss: penalizes wide back-edge arcs - Add fan-out init heuristic: re-spreads
  children of high-degree hubs - Mark TestExtremeScale (5M+ nodes) as @pytest.mark.slow

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: Add cycle support for recurrent neural networks
  ([`33a57ed`](https://github.com/johnmarktaylor91/dagua/commit/33a57edd354ba763c2876e034b734b8a96e7efb6))

DFS-based back-edge detection + edge reversal lets the layout engine handle cyclic graphs
  transparently. Back edges are reversed before layout (so the engine sees a DAG), then restored
  after. Auto-detection skipped for graphs >1M nodes for performance; users can call
  set_back_edge_mask() explicitly for large cyclic graphs.

- New dagua/layout/cycle.py: detect_back_edges(), make_acyclic() - graph.py: has_cycles,
  back_edge_mask props, prepare/restore lifecycle - engine.py: try/finally wrapper for cycle
  handling - styles.py: "back" edge style in all 3 themes - metrics.py: back_edge_mask param on
  dag_consistency/quick/full - io.py: JSON round-trip for back_edges - 32 new tests in
  tests/test_cycle.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: Increase default rank_sep from 40 to 45
  ([`43d821c`](https://github.com/johnmarktaylor91/dagua/commit/43d821c6fbd2803c86fe820fbad72682981988b3))

Aesthetic round 3 found that rank_sep is the #1 lever for layout quality. The 12.5% increase
  improves vertical hierarchy clarity on complex graphs (data_pipeline, neural_net,
  balanced_binary_tree) and fixes cramped vertical spacing on wide fan-out graphs (star,
  wide_shallow) with zero regressions on any graph type. Scored 7.50 avg vs 6.67 baseline across 6
  structurally diverse test graphs.

Key finding: loss weights (w_dag, w_attract, w_repel, etc.) have no visible effect on
  small-to-medium graphs because init_positions dominates.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **metrics**: Three-tier quality metrics suite with scale-aware sampling
  ([`a7d0167`](https://github.com/johnmarktaylor91/dagua/commit/a7d01679b2724c0714f065c2482f8e645be76fe8))

Rewrite metrics.py with a structured quality evaluation system:

Tier 1 (O(N+E), always compute): edge_length_cv, dag_consistency with violation details,
  depth_position_correlation (Spearman), overlap_count via spatial hashing, aspect_ratio,
  edge_direction_straightness.

Tier 2 (sampled): sampled_stress (BFS + sampling, 200 sources × 1K targets), sampled_crossing_rate
  (vectorized segment intersection, 1M pairs), neighborhood_preservation, angular_resolution.

Tier 3 (DAG-specific): cluster_separation, layer_uniformity, within_layer_compactness.

New API: quick(), full(), compare() (Procrustes), composite() (0-100 score). All old function names
  preserved as backward-compatible wrappers — existing 17 tests pass unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **playground**: Add interactive layout tuning widget
  ([`1601a99`](https://github.com/johnmarktaylor91/dagua/commit/1601a99ba749fb55ae9c987790550efca962d813))

- **render**: Add cinematic graph tour presets
  ([`e3d713b`](https://github.com/johnmarktaylor91/dagua/commit/e3d713b923bb382f2730e63c4199064a3ec9f0e2))

- **render**: Add edge label side and offset controls
  ([`a821b26`](https://github.com/johnmarktaylor91/dagua/commit/a821b26eb8c25f2ae6eae170e652fbb590bde13d))

- **render**: Add large-scale graph tour rendering
  ([`75be7cd`](https://github.com/johnmarktaylor91/dagua/commit/75be7cd17e46f591dca3ce1d5658172aacae1daf))

- **render**: Add optimization animation export
  ([`4adb300`](https://github.com/johnmarktaylor91/dagua/commit/4adb30065d533a0afebffc6306806eb07a0e3911))

- **render**: Add svg hover text
  ([`98e3f56`](https://github.com/johnmarktaylor91/dagua/commit/98e3f56ebf690b3e707310b27fb6a826c6a26b18))

- **report**: Add layout similarity analysis
  ([`0ee0980`](https://github.com/johnmarktaylor91/dagua/commit/0ee09802569f669117798e618f406ed0b0dd622c))

- **style**: Add aesthetic settings system with flex, cascade, and global defaults
  ([`d816e8c`](https://github.com/johnmarktaylor91/dagua/commit/d816e8c2d43d7f5327d7837f5d3b09e29cfc3576))

Three-tier API for controlling layout aesthetics: - Tier 0: dagua.draw(g) / dagua.set_theme('dark')
  / dagua.configure(font_size=10) - Tier 1: Flex.soft(40) spacing, position pins, alignment groups,
  YAML configs - Tier 2: Custom constraints, per-node flex, raw weight tuning

Key additions: - flex.py: Flex (soft/firm/rigid/locked), LayoutFlex, AlignGroup dataclasses -
  defaults.py: Thread-safe global defaults with configure(), defaults() context manager,
  did-you-mean typo suggestions, export_config() - styles.py: 5-level style cascade (per-element >
  cluster member > theme > graph default > global), resolve_node_style/resolve_edge_style functions
  - graph.py: pin(), align(), export_style() helpers, default_node/edge_style fields -
  constraints.py: position_pin_loss, alignment_loss, flex_spacing_loss, project_hard_pins for
  weight=inf enforcement - engine.py: Flex/pin/align wired into optimization loop with ID resolution
  - config.py: flex field on LayoutConfig - io.py: Parse/serialize defaults, flex, member_styles
  YAML/JSON sections

330 tests passing (64 new: test_defaults, test_flex, test_cascade).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **style**: Tune default aesthetic and layout defaults
  ([`1671158`](https://github.com/johnmarktaylor91/dagua/commit/167115843ab7b1cd92a12955e068e14966e13ed8))

- **style**: Tune default aesthetics and fix edge optimization NaN bug
  ([`db622ba`](https://github.com/johnmarktaylor91/dagua/commit/db622ba6502e7689229b29b26e19aaabfd789225))

Iterative aesthetic tuning (rounds E-I) driven by automated critic: - Softer edges (#6B7280, width
  1.2, opacity 0.65) that recede behind nodes - Thinner node strokes (0.6) for a modern, refined
  look - Larger arrowheads (10x7) with 3px inset so tips touch node borders - Input/output nodes get
  extra padding (14,8) for visual hierarchy - Tighter margins (15px) and increased cluster padding
  (25px) - Depth-aware cluster label positioning prevents nested label overlap - Cluster bbox
  expands to fit label text width (fixes clipping) - Font size bump (9.0) for better readability

Fix optimize_edges producing NaN control points: - Proper signed clamping in crossing loss divisor -
  Curvature loss d1_norm clamped to min 1.0 (prevents blowup on short edges) - NaN gradient guard
  with fallback to linear interpolation - Final NaN safety check returns original curves if
  optimization diverged

Also: mark scaling tests (100-1000 nodes) as @slow, add aesthetic_review/ to gitignore, add NaN
  guard to gallery script.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **styles**: Add Theme system, GraphStyle, and comprehensive aesthetic surface
  ([`be3e208`](https://github.com/johnmarktaylor91/dagua/commit/be3e2087d429a1f9f1ee7baab0a20c07c5fc1b3d))

Introduce a unified Theme dataclass bundling NodeStyle, EdgeStyle, ClusterStyle, and GraphStyle. Add
  19 new style fields across all style classes, 3 built-in themes (default, dark, minimal),
  shape-aware node sizing, per-edge routing dispatch (bezier/straight/ortho), and shape-aware port
  positioning. Wire all previously broken style fields in the renderer (corner_radius, arrow="none",
  stroke_dash, label_position, cluster fill/stroke). Replace hardcoded LEVEL_FILLS with HSL depth
  darkening. Remove edge_routing from LayoutConfig (now on EdgeStyle).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **theme**: Add built-in torchlens theme
  ([`309a68e`](https://github.com/johnmarktaylor91/dagua/commit/309a68e1802320eb9852637efdf3bbde01c4db58))

### Performance Improvements

- 5 optimizations for 50M+ node graphs — ~50GB allocation savings
  ([`c400e2e`](https://github.com/johnmarktaylor91/dagua/commit/c400e2e4fefcb0d2f05617ea36defe20ac917422))

- Pass layer_assignments through V-cycle (skip recomputing longest_path_layering at finest level) -
  Replace randperm(N)[:k] with randint(k) at 3 call sites (400-520MB saved per step) - Vectorize RVS
  nearest-neighbor sampling (single tensor op replaces ~20-iteration Python loop) - Lower VRAM
  safety factor 3x→2x (avoids premature hybrid mode for 1M-5M graphs) - Pre-fetch crossing loss
  indices to CPU (eliminates ~200-600 GPU sync stalls per step)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Active-subset overlap for 5M/10M node support — 11x memory reduction
  ([`2db39e6`](https://github.com/johnmarktaylor91/dagua/commit/2db39e601c6a953f5af76a7156843b3ad84b9c84))

Replace full-N overlap scatter ([N, 128] tensors) with RVS-style active subset ([N^(3/4), 64]
  tensors) for graphs over 100K nodes. Reduces peak RAM from 48GB to 5GB at 5M nodes, unlocks GPU
  layout at 5M (previously OOM at 1M).

Results: 5M CPU 274s, 10M CPU 609s, 5M GPU 22s.

Add rare-marked 5M/10M tests (pytest -m rare) with vectorized graph generator for million-node
  scale.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Scalable constraints, improved crossing minimization, O(1) edge construction
  ([`fb47cae`](https://github.com/johnmarktaylor91/dagua/commit/fb47caed05d10e4ae67fc78b5d47ab5559f25a1b))

Scalability (targeting 100K nodes): - Graph construction: O(1) per edge via lazy tensor finalization
  (was O(E²)) - Overlap projection: grid-based spatial hashing for N>500 (was O(N²) memory) -
  Overlap loss: grid-based for N>500 (was O(N²)) - Repulsion: lower threshold to 2000 for exact
  path, fix self-repulsion in sampling - Cluster separation: cap at 50 random pairs for large
  cluster sets - Metrics: vectorized count_overlaps, sampled count_crossings for large graphs

Crossing minimization: - Multi-pass barycenter (up to 30 sweeps, was 2) - Transpose heuristic: swap
  adjacent nodes in layers when it reduces crossings - Layered crossing loss: adjacent-layer sigmoid
  proxy with virtual node decomposition - Sum-based loss scaling (was mean) so gradient competes
  with attraction - Random DAG 50-node crossings: 305→191 (37% reduction)

Bug fixes from adversarial review: - Fix pi constant in edge_straightness metric - Fix
  self-repulsion in negative sampling path - Add input validation to from_edge_index - Fix
  project_overlaps return type annotation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Vectorized layout engine — 23x speedup at 50K nodes
  ([`506df86`](https://github.com/johnmarktaylor91/dagua/commit/506df86c289927cfb7bbd4adad32dfd5244745b1))

Eliminate all per-layer Python loops using scatter/segment tensor operations. Key insight from AMD
  GPU layout memo + ELK algorithm study.

Changes: - constraints.py: _repulsion_scatter samples K neighbors from same/adjacent layers via
  layer_offsets indexing (zero Python loops). Size-aware repulsion scaling per AMD pattern.
  Attraction capped at 1/3 distance. - projection.py: _project_sweep uses composite sort key (layer,
  x) for sweep-line overlap resolution — O(N log N), no per-layer iteration. - init_placement.py:
  _init_positions_vectorized for N>2K uses index_add_ and argsort for tensor-based barycenter
  ordering. - layers.py: LayerIndex data structure for O(1) per-layer node access. - engine.py:
  passes node_sizes to repulsion for size-aware scaling. - bench_layout.py: ELK benchmark support
  via --elk flag.

Sprint 3 benchmark (layout only, 50 steps, CPU): 1K: 0.57s (was 0.80s) 5K: 0.81s (was 4.80s, 6x)

10K: 1.45s (was 17.2s, 12x) 20K: 2.75s (was 68.7s, 25x)

50K: 21.6s (was 482s, 22x) 100K: 67.5s (was 2096s, 31x)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Vectorized multilevel coarsening + 2M node benchmark
  ([`a9279a3`](https://github.com/johnmarktaylor91/dagua/commit/a9279a375f1f81c034545d131ac520cc0c397dd3))

- Vectorize coarsen_once(): replace O(N) Python loop with tensor ops (2M hierarchy build: 15+ min →
  2.6s) - Vectorize longest_path_layering(): wave-based BFS for >10K nodes (2M layering: ~10s →
  ~1.5s) - Vectorize metrics: count_crossings and count_overlaps use tensor sampling instead of
  Python loops (100K: hours → 0.05s) - Tune crossing loss (disabled: w_crossing=0.0, proxy
  counterproductive) - Tune straightness: w_attract_x_bias 4→2, w_straightness 1→2, annealed - Add
  comprehensive benchmark_comparison.py (dagua vs graphviz vs ELK) - 10 real neural network
  architectures - Scaling from 500 to 2M nodes - Runtime + aesthetic quality metrics - LaTeX report
  with figures

Key results: - 2M nodes: 422s CPU, 61s GPU (was impossible before) - GPU 4.5-7.3x speedup at 5K+
  nodes - Dagua CPU beats Graphviz at 10K (4.1s vs 29.1s, 7x) - Dagua GPU beats Graphviz at 5K (1.8s
  vs 5.0s) - ELK fails at 50K (stack overflow) - 64% win rate on aesthetic metrics vs competitors

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **bench**: Add full large-run resume tiers
  ([`409de4a`](https://github.com/johnmarktaylor91/dagua/commit/409de4a181da63c251db6381b3404204aa55719e))

- **bench**: Run large benchmark on cuda
  ([`94e402a`](https://github.com/johnmarktaylor91/dagua/commit/94e402a3df0a4fc618901b1e51ff677459a3113e))

- **eval**: Reuse cached benchmark competitors
  ([`0dcde1d`](https://github.com/johnmarktaylor91/dagua/commit/0dcde1d9f1261cd7a6d5b09f691ecea3c4317061))

- **layout**: 1b node layout within 125 GB — memory optimizations + streaming projection
  ([`61aee47`](https://github.com/johnmarktaylor91/dagua/commit/61aee47f0208fc29c1cdcde60103153453e040aa))

- Free hierarchy levels eagerly during refinement (levels[i].edge_index/node_sizes freed at start of
  iteration, not end — saves ~16 GB at level 0) - malloc_trim(0) to force glibc memory return after
  large frees - del init_pos after clone in engine (saves 8 GB throughout optimization) - del
  optimizer + pos.grad before final projection (saves 24 GB) - Remove dead sorted_layers variable in
  build_layer_index (saves 8 GB temp) - Add _project_sweep_streaming for N > 100M: per-layer sweep
  instead of global argsort — ~5 MB instead of ~54 GB temporaries - Skip spacing_consistency_loss
  for N > 100M: global argsort + autograd created ~49 GB intermediates, infeasible at billion scale
  - Fix pre-existing hybrid GPU bug: tensor truthiness check on line 245 - Reduce bench_1b.py
  cross-connections from 50% to 5% (realistic DAG density) - Remove temporary RSS tracking from
  utils.py and multilevel.py

Verified: 1B nodes (1.05B edges) completes in ~103 min, peak RSS ~61 GB. All 59 non-slow tests pass
  including 20M GPU test.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: 50m-scale optimizations — adaptive projection, hoisted losses, fewer coarse steps
  ([`8ec2077`](https://github.com/johnmarktaylor91/dagua/commit/8ec20777e081e0e63fc04ba24a9e77623d60bbf8))

Six targeted optimizations to reduce 50M-node layout time:

1. projection.py: Skip window-2 overlap check for N > 100K (halves tensor ops) 2. engine.py:
  Adaptive projection iterations (2-5 mid-loop, 5-20 final) scaled by N 3. engine.py: Hoist loss
  function construction out of per-step loop — build once, update weights via mutable refs
  (eliminates 11K lambda allocations per 1000 steps) 4. engine.py: Pre-allocate edge batch buffer,
  reuse via copy_() each step 5. init_placement.py: Skip spectral init (lobpcg) for N > 5M 6.
  multilevel.py: Coarser refinement levels (i > 2) get half steps

Also includes: overlap interval 40 for N > 1M, early stopping on unweighted loss (immune to
  annealing), hybrid wave/BFS layering in utils.py, layer propagation through coarsening hierarchy.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: 6 optimizations for 100M+ node graphs
  ([`ae4397c`](https://github.com/johnmarktaylor91/dagua/commit/ae4397c2341e76da0a4e140e8310f4ff86ca8572))

1. multilevel: drop unused `inverse` from edge_hash.unique() (3-5x memory) 2. multilevel: coarsen by
  triples (//3) instead of pairs — ~67% reduction per level, halving hierarchy depth from 7 to 4
  levels at 100M 3. constraints: vectorize grid overlap — batch small cells into [B,M,M] tensor ops,
  pre-fetch boundaries to CPU once, cap cells at 1000 (5-10x speedup) 4. constraints: simplify RVS
  repulsion — pure random same-layer sampling replaces expensive offset-based "nearest" (2-3x
  faster, same quality) 5. init_placement: lower spectral init threshold from 5M to 2M (skip lobpcg
  for graphs that are too large for it to converge reliably) 6. engine: reduce final overlap
  projection from 5 to 3 iters for N>5M

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: Adaptive parameters for small graph speed (5-9x for N<50)
  ([`59465ff`](https://github.com/johnmarktaylor91/dagua/commit/59465ff5cf33a6235fa0233072692d0a8589155e))

Scale optimization steps, early stopping, projection iterations, and edge optimization steps based
  on graph size instead of using fixed values. Lowers vectorized barycenter threshold from 2000 to
  100. Users who set explicit values get exactly what they asked for (no behavior change).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: Eliminate hot-path allocations for 100M+ node graphs
  ([`0dc3f90`](https://github.com/johnmarktaylor91/dagua/commit/0dc3f90cf50c1f0fa26d2e89bd6d1499034b7d0e))

Eliminate ~460GB transient allocations at 300M nodes: pre-allocate wave_set bool tensor and reuse
  via .zero_() instead of per-wave allocation, return tensors from layering instead of .tolist()
  (avoids ~10GB Python list at 300M), keep layer assignments as tensors throughout hierarchy
  building and engine hot loop, accept tensor in crossing loss to skip per-step torch.tensor()
  re-creation, and cap n_active at 1M in RVS repulsion/overlap to prevent multi-GB intermediates.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: Improve multilevel coarsening via min-neighbor matching
  ([`adef14c`](https://github.com/johnmarktaylor91/dagua/commit/adef14c1f5beed3be8a2c2426c20648ae7cb7354))

Replace degree-based match_score with min_neighbor scatter_reduce for coarsening priority. Nodes
  sharing a low-index neighbor sort consecutively → grouped into the same coarse node → shared edges
  collapse during deduplication, producing better coarse approximations.

Also update bench_1b.py to target 1.5B edges with ceil division.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **layout**: Reduce billion-scale hierarchy memory
  ([`0e8c1d5`](https://github.com/johnmarktaylor91/dagua/commit/0e8c1d5b8112ee04fac485ec3bc88ac6415b4cbf))

- **layout**: Reduce routing and optimizer overhead
  ([`04415bd`](https://github.com/johnmarktaylor91/dagua/commit/04415bd682f3c130f89f8b0e7835f7bfb0be3670))

- **layout**: Streaming coarsening + chunked layering for 1B+ nodes
  ([`0969b24`](https://github.com/johnmarktaylor91/dagua/commit/0969b24ecaf171860db5caadc3714a9181d5daba))

Process edges in 10M chunks and match nodes per-layer to avoid materializing full [E]-sized
  temporaries. Drops peak memory from ~100 GB to ~82 GB at 1B nodes, fitting 128 GB machines with 46
  GB headroom.

- utils.py: chunked in-degree/out-degree scatter_add, _process_wave_edges_chunked helper -
  multilevel.py: _coarsen_once_streaming with per-layer matching + chunked edge dedup -
  test_smoke.py: 6 new smoke tests for structural invariants + layering equivalence

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **multilevel**: Bucket coarse-edge dedup at scale
  ([`8fc99c1`](https://github.com/johnmarktaylor91/dagua/commit/8fc99c11226463f3919724e3793699a957be9713))

- **multilevel**: Guard gpu prolongation
  ([`9c3df34`](https://github.com/johnmarktaylor91/dagua/commit/9c3df34c4908b11d9a4767928be43f39a98abd8c))

- **multilevel**: Improve structural coarsening
  ([`1befab1`](https://github.com/johnmarktaylor91/dagua/commit/1befab197883aa27bf561b192a690efeb9ac5d46))

- **multilevel**: Reuse coarse layer assignments
  ([`7deaecf`](https://github.com/johnmarktaylor91/dagua/commit/7deaecf8a0034edb2f8ed279fe02f78b7c5b405d))

### Refactoring

- **eval**: Clarify torchlens graph fixtures
  ([`5410acc`](https://github.com/johnmarktaylor91/dagua/commit/5410acc60aa3a4ff4ae1aea0a4f7678440e112d6))

- **types**: Finish package mypy cleanup
  ([`e20cc88`](https://github.com/johnmarktaylor91/dagua/commit/e20cc8880702d2f686babeb040867f7bd255d108))

- **types**: Reduce additional typing debt
  ([`75a6535`](https://github.com/johnmarktaylor91/dagua/commit/75a65356cedcca052277ccef610542b13675d7c2))

- **types**: Reduce core typing debt
  ([`06500c6`](https://github.com/johnmarktaylor91/dagua/commit/06500c6e9a64a5c9129a0fb1444c82c9d85fa362))

- **types**: Reduce eval and utility typing debt
  ([`a2c3450`](https://github.com/johnmarktaylor91/dagua/commit/a2c34507e5c345f5ce11823f99c6b5b6d248a6f3))

### Testing

- Add comprehensive test suite (81 tests) and fix projection/engine bugs
  ([`54b7e2b`](https://github.com/johnmarktaylor91/dagua/commit/54b7e2bcb6396c37e9ebbc9b9545fc3890403665))

- 81 tests covering graph construction, layout quality, constraints, projection, rendering, metrics,
  edge routing, and integration - Fix project_overlaps to return tensor instead of None - Fix layout
  engine to use config.direction instead of graph.direction - Fix TorchLens graph extraction
  (vis_mode kwarg)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Mark slower tests with @pytest.mark.slow for faster iteration
  ([`6cfa920`](https://github.com/johnmarktaylor91/dagua/commit/6cfa920bcb91ceb52416b9f1bbcb86d8b1834667))

Tag layout quality, render, scaling comparison, and edge-case tests that take >10s as slow, keeping
  the rapid tier under 30s.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **bench**: Cover large benchmark edge cases
  ([`9c65fc4`](https://github.com/johnmarktaylor91/dagua/commit/9c65fc422ad49a9f7a6fc648fff6486cdf787c86))

- **eval**: Add challenge benchmark graphs
  ([`f6c6dcc`](https://github.com/johnmarktaylor91/dagua/commit/f6c6dcc01744603049479ff318e22aa1e7cb8556))

- **eval**: Add kitchen sink benchmark graphs
  ([`83d3799`](https://github.com/johnmarktaylor91/dagua/commit/83d379912ba3538421fd968ac207cd4b960c84db))

- **eval**: Add label stress benchmark graphs
  ([`112be2d`](https://github.com/johnmarktaylor91/dagua/commit/112be2d3c243adc2297d94d92a7814e4640ba9a9))

- **eval**: Add style stress benchmark graphs
  ([`f0e7269`](https://github.com/johnmarktaylor91/dagua/commit/f0e726955b71ae7a2f6aa8bf3fdf305560d4f014))

- **eval**: Add visual stress benchmark graphs
  ([`4323df5`](https://github.com/johnmarktaylor91/dagua/commit/4323df5220187e2e70a798382ded74bcd4d92238))

- **eval**: Broaden benchmark graph coverage
  ([`a8a1390`](https://github.com/johnmarktaylor91/dagua/commit/a8a139015b6febf698757e6cbe32a023e3557785))

- **eval**: Cover dagua multilevel benchmark path
  ([`2e5af24`](https://github.com/johnmarktaylor91/dagua/commit/2e5af24f30cd85b0a3938f1cf1b296e17527e08a))

- **eval**: Prevent TestGraph pytest collection
  ([`0c741e5`](https://github.com/johnmarktaylor91/dagua/commit/0c741e5f4dd9c211003e6e014e7e3a65c6cff799))

- **graphs**: Add 31 hand-crafted YAML test graphs covering all structural dimensions
  ([`94c7415`](https://github.com/johnmarktaylor91/dagua/commit/94c7415c929b98218faec05210e1ff3012875fab))

Adds comprehensive small-to-medium graph battery (2-20 nodes each) across 10 categories: size
  extremes, width/depth, cycles, cluster nesting (up to 6 levels), topology patterns, disconnected
  components, skip connections, real-world architectures, label/style stress, and all 4 layout
  directions. Includes invariant test that loads all 35 bundled graphs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **render**: Cover vector output formats
  ([`97bae3a`](https://github.com/johnmarktaylor91/dagua/commit/97bae3a31a41034da5817e99a654507596aff351))


## v0.0.2 (2026-03-09)

### Bug Fixes

- **ci**: Test PyPI publish with new version
  ([`ceaac63`](https://github.com/johnmarktaylor91/dagua/commit/ceaac6372e0a856ad9f12dd6a759695c23c9a50c))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.0.1 (2026-03-09)

### Bug Fixes

- **ci**: Verify PyPI trusted publishing pipeline
  ([`36a011f`](https://github.com/johnmarktaylor91/dagua/commit/36a011f574f8d38dd64f555d5f167a1c65b5b051))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.0.0 (2026-03-09)

### Chores

- Add project structure, CI/CD plumbing, and module scaffolding
  ([`436752c`](https://github.com/johnmarktaylor91/dagua/commit/436752c6155b825dea443645b1e421d8f999d12d))

- Full source layout: elements, graph, style, defaults, io, routing, utils - Layout subpackage:
  engine, constraints, projection, schedule - Render subpackage: mpl, svg, graphviz - CI/CD: lint
  (ruff auto-fix), quality (mypy + pip-audit), release (semantic-release v9 + PyPI OIDC) -
  Pre-commit hooks: trailing-whitespace, EOF fixer, check-yaml, large files, ruff - pyproject.toml:
  coverage, mypy, semantic-release config - CLAUDE.md documentation for all subpackages, tests,
  benchmarks, examples - Test scaffolding mirroring source structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- Initial project scaffold
  ([`4a53fea`](https://github.com/johnmarktaylor91/dagua/commit/4a53feab837e8b3a7d7980ce9ac2a7ba92ce75df))

Dagua — GPU-accelerated differentiable graph layout engine built on PyTorch. Project structure,
  pyproject.toml, LICENSE (MIT), README, CLAUDE.md.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
